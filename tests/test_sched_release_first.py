# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""TT_SCHED_RELEASE_FIRST: a cohort's first tokens released in groups (sched_release_first), on the stub of the base
scheduler's loop of test_sched_defer_final plus a stub of its ``update_from_output``."""

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason

from tests.test_sched_defer_final import CHUNK, _base_schedule, _scheduler
from vllm_tt_plugin.sched_release_first import ReleaseFirstPolicy, value
from vllm_tt_plugin.scheduler import KNOWN_SCHED_FLAGS, check_sched_flags

FINISH_AT = {}  # req_id -> finish the request on its first token (the stub's finish_reason)


def test_flags_and_values():
    assert "TT_SCHED_RELEASE_FIRST" in KNOWN_SCHED_FLAGS
    check_sched_flags(
        {"TT_SCHED_RELEASE_FIRST": "cohort", "TT_SCHED_DEFER_FINAL": "hold"}
    )
    assert value({}) == "" and value({"TT_SCHED_RELEASE_FIRST": "0"}) == ""
    assert (
        value({"TT_SCHED_RELEASE_FIRST": "8"}) == "8"
        and value({"TT_SCHED_RELEASE_FIRST": "cohort"}) == "cohort"
    )
    for bad in ("1", "x", "-2"):
        with pytest.raises(ValueError, match="TT_SCHED_RELEASE_FIRST"):
            value({"TT_SCHED_RELEASE_FIRST": bad})
    assert ReleaseFirstPolicy.from_env({}, defer_enabled=True) is None
    with pytest.raises(ValueError, match="TT_SCHED_DEFER_FINAL"):
        ReleaseFirstPolicy.from_env(
            {"TT_SCHED_RELEASE_FIRST": "cohort"}, defer_enabled=False
        )
    policy = ReleaseFirstPolicy.from_env(
        {"TT_SCHED_RELEASE_FIRST": "8"}, defer_enabled=True
    )
    assert (
        policy is not None
        and policy.group == 8
        and policy.describe() == "release_first(group=8)"
    )
    policy = ReleaseFirstPolicy.from_env(
        {"TT_SCHED_RELEASE_FIRST": "cohort"}, defer_enabled=True
    )
    assert policy is not None and policy.group is None


def _base_update(self, scheduler_output, model_runner_output):
    """The base ``update_from_output`` as the loop sees it: one output per scheduled row that produced a token (a
    prefill chunk that completed the prompt, or a decode row); a request in FINISH_AT finishes on its first."""
    outs = []
    for req_id in scheduler_output.num_scheduled_tokens:
        r = self.requests[req_id]
        if r.num_computed_tokens < r.num_prompt_tokens:
            continue  # an intermediate chunk: no token
        r.num_output_tokens = getattr(r, "num_output_tokens", 0) + 1
        finish = (
            FinishReason.LENGTH
            if FINISH_AT.get(req_id) and r.num_output_tokens == 1
            else None
        )
        outs.append(
            EngineCoreOutput(
                request_id=req_id,
                new_token_ids=[r.num_output_tokens],
                finish_reason=finish,
            )
        )
    return {0: EngineCoreOutputs(outputs=outs)} if outs else {}


@pytest.fixture
def base(monkeypatch):
    monkeypatch.setattr(AsyncScheduler, "schedule", _base_schedule)
    monkeypatch.setattr(AsyncScheduler, "update_from_output", _base_update)
    FINISH_AT.clear()


def _sched(prompts, *, cohort=4, group=None, hold=True):
    scheduler = _scheduler(prompts, cohort=cohort, hold=hold)
    scheduler._release = ReleaseFirstPolicy(group)
    return scheduler


def _step(scheduler, async_order=False, pending=None):
    """One engine step: schedule, execute (the stub), update. With ``async_order`` the schedule of step N+1 runs
    before the update of step N (``pending`` carries the scheduled output across calls)."""
    if async_order:
        out = scheduler.schedule()
        prev = pending.pop() if pending else None
        pending.append(out)
        return (
            [] if prev is None else _delivered(scheduler.update_from_output(prev, None))
        )
    out = scheduler.schedule()
    return _delivered(scheduler.update_from_output(out, None))


def _delivered(outputs):
    return [
        (o.request_id, o.new_token_ids[0])
        for eco in outputs.values()
        for o in eco.outputs
    ]


def test_cohort_release_holds_every_first_token_until_the_last_final(base):
    scheduler = _sched([2 * CHUNK + 100] * 4, cohort=4, group=None)
    delivered = [_step(scheduler) for _ in range(8)]  # 4 bulks of 2 chunks
    assert delivered == [[]] * 8
    delivered = [
        _step(scheduler) for _ in range(4)
    ]  # the 4 finals: 3 held, released at the 4th
    assert delivered[:3] == [[], [], []], delivered
    assert delivered[3] == [("w0", 1), ("w1", 1), ("w2", 1), ("w3", 1)]
    assert scheduler._release.releases == [4] and not scheduler._release.expected
    # the decode step's tokens pass straight through (the stub's row order is the running list's)
    assert sorted(_step(scheduler)) == [(f"w{i}", 2) for i in range(4)]


def test_groups_release_k_at_a_time(base):
    scheduler = _sched([2 * CHUNK + 100] * 4, cohort=4, group=2)
    for _ in range(8):
        _step(scheduler)
    delivered = [_step(scheduler) for _ in range(4)]
    assert delivered == [[], [("w0", 1), ("w1", 1)], [], [("w2", 1), ("w3", 1)]], (
        delivered
    )
    assert scheduler._release.releases == [2, 2]


def test_a_queue_tail_cohort_releases_at_its_end(base):
    scheduler = _sched([2 * CHUNK + 100] * 3, cohort=32, group=None)
    for _ in range(6):
        _step(scheduler)
    delivered = [_step(scheduler) for _ in range(3)]
    assert delivered == [[], [], [("w0", 1), ("w1", 1), ("w2", 1)]], delivered


def test_one_chunk_prompts_are_never_held(base):
    scheduler = _sched([1028] * 3, cohort=3, group=None)
    delivered = [_step(scheduler) for _ in range(3)]
    assert delivered == [[("w0", 1)], [("w1", 1)], [("w2", 1)]], delivered
    assert scheduler._release.releases == []


def test_a_request_finishing_on_its_first_token_is_released_at_once(base):
    FINISH_AT["w1"] = True
    scheduler = _sched([2 * CHUNK + 100] * 3, cohort=3, group=None)
    for _ in range(6):
        _step(scheduler)
    delivered = [_step(scheduler) for _ in range(3)]
    assert delivered == [[], [("w1", 1)], [("w0", 1), ("w2", 1)]], delivered


def test_an_abort_while_expected_does_not_stall_the_release(base):
    scheduler = _sched([2 * CHUNK + 100] * 3, cohort=3, group=None)
    for _ in range(6):
        _step(scheduler)
    assert _step(scheduler) == []  # w0's final: held
    # w1 is aborted between steps: its id reaches the policies through finished_req_ids
    out = SchedulerOutput.make_empty()
    out.finished_req_ids = {"w1"}
    scheduler._finalize_scheduler_output(out)
    scheduler.running = [r for r in scheduler.running if r.request_id != "w1"]
    del scheduler.requests["w1"]
    assert _step(scheduler) == [("w0", 1), ("w2", 1)]


def test_async_order_releases_in_the_update_where_the_last_token_arrives(base):
    scheduler = _sched([2 * CHUNK + 100] * 4, cohort=4, group=None)
    pending = []
    delivered = [_step(scheduler, async_order=True, pending=pending) for _ in range(13)]
    # schedules 0..12: 8 bulks, 4 finals, 1 decode; updates lag by one: the finals' tokens arrive at steps 9..12
    assert delivered[:12] == [[]] * 12, delivered
    assert delivered[12] == [("w0", 1), ("w1", 1), ("w2", 1), ("w3", 1)]
    assert sorted(_step(scheduler, async_order=True, pending=pending)) == [
        (f"w{i}", 2) for i in range(4)
    ]


def test_unset_delivers_every_token_as_it_comes(base):
    scheduler = _scheduler([2 * CHUNK + 100] * 2, cohort=2, hold=True)
    for _ in range(4):
        _step(scheduler)
    assert [_step(scheduler) for _ in range(2)] == [[("w0", 1)], [("w1", 1)]]
