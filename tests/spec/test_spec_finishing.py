# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Finishing slots, the unsafe-overlap counter and the acceptance metric.

Async scheduling leaves a running request out of a step when its in-flight step
is sure to reach ``max_tokens`` (vLLM 0.30 ``scheduler.py:618-628``). The
scheduler has not seen that output, so the request is still running and used to
look like a hole in the chain (``layout`` break, then a park of 48 layers).
``TTScheduler`` names those requests, the runner sends their slots as
``finishing_slots`` and the chain treats them as finished.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_tt_plugin import async_decode
from vllm_tt_plugin.scheduler import (
    _TT_SEED_REQ_IDS_ATTR,
    TT_VALID_DRAFTS_ATTR,
    TTScheduler,
    get_tt_finishing_req_ids,
    request_will_reach_max_tokens,
    set_tt_finishing_req_ids,
)
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

from .fake_spec_model import FakeSpecModel
from .test_spec_async import PROMPT_LEN, _scheduler_output
from .test_spec_chain_async import (
    Engine,
    K,
    _chain_model,
    _controller,
    _decode_so,
    _make_runner,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("VLLM_TT_SPEC_FUSED", "VLLM_TT_SPEC_FUSED_SAMPLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_DEVICE_TOPK", "1")


# region Detection (the vLLM skip condition)


def _req(req_id, *, placeholders, computed, prompt=10, max_tokens=20):
    return SimpleNamespace(
        request_id=req_id,
        num_output_placeholders=placeholders,
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        max_tokens=max_tokens,
        status=RequestStatus.RUNNING,
    )


def test_the_skip_condition_is_vllms_verbatim():
    # one in-flight verify (1+K placeholders), confirmed count o: the request
    # finishes when o >= max_tokens - 1, i.e. computed = P + o - 1 + (K+1)
    k1 = K + 1
    prompt, mx = 10, 20
    for o in range(1, mx + 1):
        r = _req(
            "r",
            placeholders=k1,
            computed=prompt + o - 1 + k1,
            prompt=prompt,
            max_tokens=mx,
        )
        assert request_will_reach_max_tokens(r) == (o >= mx - 1), o
    # nothing in flight: never skipped, whatever the counts say
    assert not request_will_reach_max_tokens(_req("r", placeholders=0, computed=10_000))


def _tt_scheduler(running):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.running = running
    scheduler.requests = {r.request_id: r for r in running}
    scheduler._finishing_candidates = []
    scheduler._widest_decode_batch_size = 0
    scheduler._pending_forced_reset_discard_counts = {}
    return scheduler


def _run_schedule(scheduler, scheduled, **kw):
    def fake_mode():
        out = SchedulerOutput.make_empty()
        out.num_scheduled_tokens = {r: 1 for r in scheduled}
        for key, value in kw.items():
            setattr(out, key, value)
        return scheduler._finalize_scheduler_output(out, is_decode=True)

    scheduler._schedule_mode = fake_mode
    return scheduler.schedule()


def test_only_requests_the_vllm_rule_skips_are_reported_finishing():
    k1 = K + 1
    sure = _req("sure", placeholders=k1, computed=10 + 18 + k1)  # o = 19 = max-1
    early = _req("early", placeholders=k1, computed=10 + 5 + k1)  # far from max
    idle = _req("idle", placeholders=0, computed=10 + 5)
    scheduler = _tt_scheduler([sure, early, idle])
    out = _run_schedule(scheduler, ["early", "idle"])
    assert get_tt_finishing_req_ids(out) == ("sure",)


def test_a_candidate_the_step_scheduled_or_preempted_is_not_reported():
    k1 = K + 1
    a = _req("a", placeholders=k1, computed=10 + 18 + k1)
    b = _req("b", placeholders=k1, computed=10 + 18 + k1)
    scheduler = _tt_scheduler([a, b])
    # b is scheduled anyway (the loop did not reach the skip for it): never sent
    out = _run_schedule(scheduler, ["b"])
    assert get_tt_finishing_req_ids(out) == ("a",)
    scheduler = _tt_scheduler([a, b])
    out = _run_schedule(scheduler, [], preempted_req_ids={"a"})
    assert get_tt_finishing_req_ids(out) == ("b",)
    # a prefill step hides the decodes: nothing is reported
    scheduler = _tt_scheduler([a, b])
    scheduler._schedule_mode = lambda: scheduler._finalize_scheduler_output(
        SchedulerOutput.make_empty(), is_decode=False
    )
    assert get_tt_finishing_req_ids(scheduler.schedule()) == ()


# endregion

# region The runner and the chain


FIN_CAPS = {
    **FakeSpecModel.model_capabilities,
    "supports_async_decode": True,
    "supports_async_spec_decode": True,
    "supports_finishing_slots": True,
}


class MaxTokensEngine(Engine):
    """``Engine`` whose scheduler applies vLLM's max_tokens skip rule.

    A request with ``stop_after`` outputs has ``max_tokens = stop_after + 1``
    (the prefill token counts). Unlike the plain Engine, the step after the one
    that is sure to finish it does not schedule it; with ``finishing`` on, the
    skipped request is passed to the runner like ``TTScheduler`` does.
    """

    def __init__(self, runner, *, finishing: bool, skip: bool = True, **kw):
        super().__init__(runner, **kw)
        self.finishing = finishing
        self.skip = skip
        self.skipped: dict[int, list[str]] = {}

    def _inflight(self, req_id):
        return sum(1 for _, k in self.inflight if req_id in self.step_rows[k])

    def schedule(self):
        finished, self.finished_pending = self.finished_pending, set()
        will_finish = []
        if self.skip:
            for r in self.live:
                stop = self.stop_after.get(r)
                if stop is None:
                    continue
                request = SimpleNamespace(
                    num_output_placeholders=(K + 1) * self._inflight(r),
                    num_computed_tokens=self.computed[r],
                    num_prompt_tokens=PROMPT_LEN,
                    max_tokens=stop + 1,
                )
                if request_will_reach_max_tokens(request):
                    will_finish.append(r)
        rows = [r for r in self.live if r not in will_finish]
        self.skipped[self.step] = list(will_finish)
        drafts = {r: [PLACEHOLDER_TOKEN_ID] * K for r in rows}
        so = _scheduler_output(
            self.runner, decoding=tuple(rows), finished=finished, drafts=drafts
        )
        import dataclasses

        so.scheduled_cached_reqs = dataclasses.replace(
            so.scheduled_cached_reqs,
            num_computed_tokens=[self.computed[r] for r in rows],
            num_output_tokens=[max(1, len(self.outputs[r])) for r in rows],
        )
        for r in rows:
            self.computed[r] += K + 1
        if self.finishing:
            set_tt_finishing_req_ids(so, will_finish)
        return so


def _fin_run(finishing, *, skip=True, stop=(9, 14, None), steps=14):
    model = _chain_model(accept_cycle=(3, 0, 1, 2), model_capabilities=FIN_CAPS)
    runner = _make_runner(model)
    engine = MaxTokensEngine(runner, finishing=finishing, skip=skip)
    for name, first, s in zip("abc", (1, 40, 80), stop):
        engine.add(name, first, stop_after=s)
    for _ in range(steps):
        if len(engine.done) == sum(1 for s in stop if s is not None):
            break
        engine.iterate()
    engine.finish()
    return model, runner, engine


def test_a_skipped_to_finish_request_is_sent_finishing_and_never_parked():
    model, runner, engine = _fin_run(finishing=True)
    controller = _controller(runner)
    skipped = {k: v for k, v in engine.skipped.items() if v}
    assert skipped, "the max_tokens rule never skipped a request"
    sent = [s for s in model.finishing_log if s]
    assert len(sent) == len(skipped)
    # one slot per skipped request, once, then none again for it
    assert sorted(s for call in sent for s in call) == sorted(
        set(s for call in sent for s in call)
    )
    # no layout break and no host-built verify caused by the hole
    assert not controller._chain_breaks.get("layout")
    assert controller._overlapped_unsafe_submissions == 0
    # the stream is the sync (unchained, nothing skipped) stream
    _, _, plain = _fin_run(finishing=False, skip=False)
    for req in ("a", "b"):
        assert engine.outputs[req] == plain.outputs[req]
    # a skipped request is not scheduled again and keeps no slot claim
    for k, reqs in skipped.items():
        for r in reqs:
            assert all(
                r not in rows for kk, rows in engine.step_rows.items() if kk >= k
            )
            assert r not in runner._req_state_slot


def test_without_the_signal_the_same_run_breaks_on_layout():
    model, runner, engine = _fin_run(finishing=False)
    assert any(engine.skipped.values())
    assert _controller(runner)._chain_breaks["layout"] >= 1
    assert not any(model.finishing_log)


def test_a_model_without_the_capability_never_receives_finishing_slots():
    model = _chain_model(accept_cycle=(3, 0, 1, 2))
    runner = _make_runner(model)
    engine = MaxTokensEngine(runner, finishing=True)
    engine.add("a", 1)
    engine.add("b", 40, stop_after=9)
    engine.add("c", 80, stop_after=14)
    for _ in range(14):
        engine.iterate()
    engine.finish()
    assert not any(model.finishing_log)
    assert any(engine.skipped.values())


def test_a_request_not_guaranteed_to_finish_is_never_sent():
    # no stop_after anywhere: nothing is sure to finish
    model, runner, engine = _fin_run(finishing=True, stop=(None, None, None), steps=8)
    assert not any(engine.skipped.values())
    assert not any(model.finishing_log)


def test_the_chain_treats_finishing_requests_like_finished_ones():
    model = _chain_model(model_capabilities=FIN_CAPS)
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.add("a", 1)
    engine.add("b", 40)
    engine.iterate()
    controller = _controller(runner)
    only_a = _decode_so(runner, ("a",))
    assert controller.chain_blocker(only_a) == "layout"
    set_tt_finishing_req_ids(only_a, ["b"])
    assert controller.chain_blocker(only_a) is None


# endregion

# region The unsafe-overlap counter


def test_a_resolved_pending_step_is_not_an_overlap():
    model = _chain_model(model_capabilities=FIN_CAPS)
    runner = _make_runner(model)
    controller = _controller(runner)

    class Step:
        def __init__(self, resolved):
            self._resolved = resolved

        def is_resolved(self):
            return self._resolved

    runner._pending_async_steps.append(Step(True))
    controller.register_pending_async_step(Step(False), steady=False, overlap_ok=False)
    assert controller._overlapped_submissions == 0
    assert controller._overlapped_unsafe_submissions == 0
    # an unresolved one is
    controller.register_pending_async_step(Step(False), steady=False, overlap_ok=False)
    assert controller._overlapped_submissions == 1
    assert controller._overlapped_unsafe_submissions == 1


def test_no_pending_applies_the_completed_steps_before_the_build(monkeypatch):
    model = _chain_model(model_capabilities=FIN_CAPS)
    runner = _make_runner(model)
    controller = _controller(runner)
    applied = []
    monkeypatch.setattr(
        controller,
        "apply_ready_completed_decode_steps",
        lambda **kw: applied.append(kw),
    )
    so = _decode_so(runner, ())
    controller.apply_resolved_before_build(so)
    assert not applied  # nothing queued: no apply
    runner._completed_decode_steps.append(object())
    controller.apply_resolved_before_build(so)
    assert len(applied) == 1
    # and the runner asks for it when chain_evaluate says no_pending
    calls = []
    monkeypatch.setattr(
        controller, "apply_resolved_before_build", lambda s: calls.append(s)
    )
    monkeypatch.setattr(controller, "chain_evaluate", lambda s: ("no_pending", None))
    assert runner._chain_build_plan(so, []) is None
    assert calls == [so]


# endregion

# region The acceptance metric


def _metric_scheduler(async_scheduling=True):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.num_spec_tokens = K
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
    scheduler._spec_step_seen = set()
    return scheduler


def _offer(req_ids):
    out = SchedulerOutput.make_empty()
    out.num_scheduled_tokens = {r: 1 for r in req_ids}
    out.scheduled_spec_decode_tokens = {r: [-1] * K for r in req_ids}
    return out


def _seed_then_steady_invalid(valid_reports):
    """num_invalid_spec_tokens per step for one request."""
    scheduler = _metric_scheduler()
    result = []
    for report in valid_reports:
        out = _offer(["r"])
        scheduler._track_spec_seed_steps(out)
        runner_out = SimpleNamespace()
        if report is not None:
            setattr(runner_out, TT_VALID_DRAFTS_ATTR, {"r": report})
        scheduler._mark_unoffered_drafts_invalid(out, runner_out)
        result.append(out.num_invalid_spec_tokens.get("r", 0))
    return result


def test_the_seed_step_counts_no_rejected_drafts():
    # seed (no report needed), a full step, a step with 1 real draft
    assert _seed_then_steady_invalid([None, K, 1]) == [K, 0, K - 1]
    # the runner's report decides when present, also for the seed step
    assert _seed_then_steady_invalid([0, K]) == [K, 0]


def test_a_prefill_restarts_the_seed_step():
    scheduler = _metric_scheduler()
    first = _offer(["r"])
    scheduler._track_spec_seed_steps(first)
    assert getattr(first, _TT_SEED_REQ_IDS_ATTR) == {"r"}
    again = _offer(["r"])
    scheduler._track_spec_seed_steps(again)
    assert getattr(again, _TT_SEED_REQ_IDS_ATTR) == frozenset()
    prefill = SchedulerOutput.make_empty()
    prefill.num_scheduled_tokens = {"r": 100}
    scheduler._track_spec_seed_steps(prefill)
    resumed = _offer(["r"])
    scheduler._track_spec_seed_steps(resumed)
    assert getattr(resumed, _TT_SEED_REQ_IDS_ATTR) == {"r"}


def test_vllm_metrics_see_zero_rejected_for_the_seed_step():
    """Through vLLM's own stats code, with the scheduler's invalid count."""
    scheduler = _metric_scheduler()
    out = _offer(["r"])
    scheduler._track_spec_seed_steps(out)
    scheduler._mark_unoffered_drafts_invalid(out, SimpleNamespace())
    stats = Scheduler.make_spec_decoding_stats(
        SimpleNamespace(log_stats=True, num_spec_tokens=K),
        None,
        num_draft_tokens=K,
        num_accepted_tokens=0,
        num_invalid_spec_tokens=out.num_invalid_spec_tokens,
        request_id="r",
    )
    assert stats.num_draft_tokens == 0
    assert stats.num_accepted_tokens == 0


def test_the_runner_reports_real_draft_counts_and_the_log_carries_them(monkeypatch):
    model = _chain_model(accept_cycle=(3, 0, 1, 2), model_capabilities=FIN_CAPS)
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.add("a", 1)
    engine.add("b", 40)
    for _ in range(4):
        engine.iterate()
    controller = _controller(runner)
    # the seed step of every row verified no real draft; later steps did
    assert controller._verify_rows >= 8
    assert 0 < controller._valid_drafts_submitted <= K * controller._verify_rows
    engine.finish()
    assert controller._accepted_drafts > 0
    lines = []
    monkeypatch.setattr(
        async_decode.logger, "info", lambda fmt, *args: lines.append(fmt % args)
    )
    monkeypatch.setattr(async_decode, "_SUBMISSION_LOG_INTERVAL", 1)
    controller.count_decode_submission(
        SimpleNamespace(spec_mode=None, row_req_ids=[], num_valid_drafts=None)
    )
    expected = (
        f"real drafts: {controller._valid_drafts_submitted} valid, "
        f"{controller._accepted_drafts} accepted over "
        f"{controller._verify_rows} verify rows"
    )
    assert any(expected in line for line in lines), lines


def test_tag_valid_drafts_reports_zero_for_a_step_that_verified_nothing():
    model = _chain_model(model_capabilities=FIN_CAPS)
    runner = _make_runner(model)
    controller = _controller(runner)
    out = SimpleNamespace()
    mi = SimpleNamespace(row_req_ids=["a", "b"], num_valid_drafts=None)
    controller.tag_valid_drafts(out, mi)
    assert getattr(out, TT_VALID_DRAFTS_ATTR) == {"a": 0, "b": 0}
    mi = SimpleNamespace(
        row_req_ids=["a", "b"],
        num_valid_drafts=torch.tensor([3, 1, 0, 0], dtype=torch.int32),
    )
    controller.tag_valid_drafts(out, mi)
    assert getattr(out, TT_VALID_DRAFTS_ATTR) == {"a": 3, "b": 1}


# endregion
