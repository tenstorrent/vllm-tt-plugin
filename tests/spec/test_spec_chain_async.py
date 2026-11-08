# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage S4: a fused verify submitted while the previous one is in flight.

With ``VLLM_TT_SPEC_CHAIN`` on and a model that declares
``supports_async_spec_decode`` (it runs chain and fused), the runner builds step
n+1 without step n's committed tokens (placeholders, optimistic positions), the
device builds the real inputs from step n, and step n's result is read on the
reader thread and applied on the engine thread later. These tests drive that
through the real entry points against a host fake that implements the chain
contract on a deterministic target (``ChainFusedMixin``), with a handle that
only completes when it is waited on.

``Engine`` mimics ``EngineCore.step_with_batch_queue`` with a queue of two:
schedule(k), ``execute_model(k)``, then resolve step k-1 and run the scheduler's
``update_from_output``. A request that finishes in the update of step n has
therefore already been scheduled into n+1, which is the stop lag under test.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections import deque

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.spec_accept import spec_chain_enabled, spec_chain_sampled_enabled
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_FUSED_SAMPLE,
    PLACEHOLDER_TOKEN_ID,
    DraftOutput,
)

from .fake_spec_model import (
    ChainFusedMixin,
    ChainReadMixin,
    FakeFusedHandle,
    FakeSpecModel,
    make_fake_spec_model,
)
from .sampled_target import SAMPLED_VOCAB_SIZE, base_token, rule_logits
from .test_spec_async import (
    BLOCK_SIZE,
    DRAFT_LEN,
    MAX_MODEL_LEN,
    MAX_NUM_REQS,
    PROMPT_LEN,
    _scheduler_output,
)
from .test_spec_async import _runner as _async_runner
from .test_spec_device_topk import exact_support
from .test_spec_fused import _FusedTopkTarget

K = DRAFT_LEN


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (
        "VLLM_TT_SPEC_FUSED",
        "VLLM_TT_SPEC_FUSED_SAMPLED",
        "VLLM_TT_SPEC_CHAIN",
        "VLLM_TT_SPEC_CHAIN_SAMPLED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_TT_SPEC_DEVICE_TOPK", "1")


# region Harness


def _chain_model(**knobs):
    knobs.setdefault("max_supported_num_seqs", MAX_NUM_REQS)
    knobs.setdefault("supported_k", (K,))
    return make_fake_spec_model(serve_chain=True, **knobs)()


def _make_runner(model):
    runner = _async_runner(model)
    if runner.input_batch.vocab_size != int(model.vocab_size):
        # The async harness sizes the batch for the greedy target's
        # vocabulary, and a row without a top-k stores the batch's vocabulary
        # as its top-k, which the logits walk reads against the model's.
        runner.input_batch = InputBatch(
            max_num_reqs=MAX_NUM_REQS,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=MAX_MODEL_LEN,
            vocab_size=int(model.vocab_size),
            block_sizes=[BLOCK_SIZE],
            kernel_block_sizes=[BLOCK_SIZE],
        )
    runner._spec_drafts_from_model = True
    runner._spec_method = "custom_class"
    runner._spec_accept_modes = tuple(type(model).accept_modes)
    runner._spec_plan = type(model).spec_plan(None, MAX_NUM_REQS, K)
    runner.vocab_size = int(model.vocab_size)
    del runner.note_decode_layout_consumed
    runner.note_decode_layout_consumed = (
        TTModelRunner.note_decode_layout_consumed.__get__(runner)
    )
    runner._chain_len = {}
    runner._chain_consumed_tips = set()
    # The real padded sampling params: a fused step reads the temperatures.
    del runner._sampling_params_for_padded_decode
    runner._sampling_params_for_padded_decode = (
        TTModelRunner._sampling_params_for_padded_decode.__get__(runner)
    )
    # The real slot bookkeeping: a request that finishes condenses the batch,
    # and the model finds a chained row's state through the remap.
    del runner._decode_state_slot_remap
    del runner.note_decode_state_slots_settled
    runner._pending_state_slot_settle = None
    runner._pending_state_slot_moves = None
    runner._decode_state_slot_remap = TTModelRunner._decode_state_slot_remap.__get__(
        runner
    )
    runner.note_decode_state_slots_settled = (
        TTModelRunner.note_decode_state_slots_settled.__get__(runner)
    )
    return runner


GREEDY = SamplingParams(temperature=0.0, ignore_eos=True)
SAMPLED = SamplingParams(temperature=1.0, top_k=5, ignore_eos=True)
SEEDED = SamplingParams(temperature=0.9, top_k=6, seed=11, ignore_eos=True)


class Engine:
    """``EngineCore.step_with_batch_queue`` with a queue of two, and a scheduler."""

    def __init__(self, runner, *, threaded: bool = False):
        self.runner = runner
        self.threaded = threaded
        self.live: list[str] = []
        self.outputs: dict[str, list[int]] = {}
        self.stop_after: dict[str, int] = {}
        self.done: set[str] = set()
        self.finished_pending: set[str] = set()
        self.inflight: deque = deque()
        self.step = 0
        self.published: dict[int, dict[str, list[int]]] = {}
        self.step_rows: dict[int, list[str]] = {}
        self.finish_step: dict[str, int] = {}
        self.reader_threads: list[int] = []
        # The scheduler's own computed-token count: it adds 1+K per scheduled
        # step at once and takes back the rejected ones when the output lands.
        self.computed: dict[str, int] = {}
        self.prefill_log: list[int] = []

    def add(self, req_id, first_token, params=GREEDY, stop_after=None):
        """Admit a request the way a join looks to the runner: it is in the
        persistent batch, and the next scheduled step carries it as a cached
        decode row."""
        so = _scheduler_output(self.runner, decoding=tuple(self.live))
        so.scheduled_new_reqs = [
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=list(range(first_token, first_token + PROMPT_LEN)),
                mm_features=[],
                sampling_params=params,
                pooling_params=None,
                block_ids=([0],),
                num_computed_tokens=PROMPT_LEN,
                lora_request=None,
            )
        ]
        so.num_scheduled_tokens[req_id] = 1
        so.total_num_scheduled_tokens += 1
        self.runner._update_states(so)
        self._register(req_id, stop_after)

    def _register(self, req_id, stop_after=None):
        used = set(self.runner._req_state_slot.values())
        self.runner._req_state_slot[req_id] = min(set(range(MAX_NUM_REQS)) - used)
        self.live.append(req_id)
        self.computed[req_id] = PROMPT_LEN
        self.outputs[req_id] = []
        if stop_after is not None:
            self.stop_after[req_id] = stop_after

    def prefill_step(self, news, *, resumed=()):
        """A prefill-only step through ``execute_model``, submitted over
        whatever is in flight. Only the device forward is cut: the drain
        decision, ``_update_states`` and the apply before the build are the
        real ones. ``prefill_log`` records, at the forward, how many verifies
        were still unresolved. The step returns no output of its own, so the
        verify stays the oldest in-flight step, as it does in the engine's FIFO
        queue. A request in ``news`` is live afterwards, as ``add`` leaves it."""
        so = _scheduler_output(
            self.runner,
            new=[(req_id, first) for req_id, first in news],
            decoding=tuple(resumed),
            resumed=resumed,
        )
        for new_req in so.scheduled_new_reqs:
            new_req.sampling_params = GREEDY
        so.scheduled_cached_reqs = dataclasses.replace(
            so.scheduled_cached_reqs, num_output_tokens=[0 for _ in resumed]
        )
        runner = self.runner

        def forward(scheduler_output, grammar_output):
            with runner._steady_decode_lock:
                self.prefill_log.append(
                    sum(not s.is_resolved() for s in runner._pending_async_steps)
                )
            return None

        real_prepare = runner._prepare_model_inputs
        runner._prepare_model_inputs = forward
        try:
            runner.execute_model(so)
        finally:
            runner._prepare_model_inputs = real_prepare
        for req_id, _ in news:
            self._register(req_id)

    def schedule(self):
        finished, self.finished_pending = self.finished_pending, set()
        drafts = {r: [PLACEHOLDER_TOKEN_ID] * K for r in self.live}
        so = _scheduler_output(
            self.runner, decoding=tuple(self.live), finished=finished, drafts=drafts
        )
        # The prefill step committed each request's first token, so a decoding
        # request never reads as context phase (zero outputs).
        so.scheduled_cached_reqs = dataclasses.replace(
            so.scheduled_cached_reqs,
            num_computed_tokens=[self.computed[r] for r in self.live],
            num_output_tokens=[max(1, len(self.outputs[r])) for r in self.live],
        )
        for r in self.live:
            self.computed[r] += K + 1
        return so

    def iterate(self):
        so = self.schedule()
        k = self.step
        self.step += 1
        self.step_rows[k] = list(so.num_scheduled_tokens)
        assert self.runner.execute_model(so) is None
        wrapper = self.runner.sample_tokens(None)
        self.inflight.append((wrapper, k))
        if len(self.inflight) >= 2:
            self.resolve_oldest()

    def resolve_oldest(self):
        wrapper, k = self.inflight.popleft()
        if self.threaded:
            box = []

            def read():
                self.reader_threads.append(threading.get_ident())
                box.append(wrapper.get_output())

            thread = threading.Thread(target=read)
            thread.start()
            handle = wrapper._submission.spec_hidden
            if isinstance(handle, FakeFusedHandle):
                handle.release()
            thread.join(timeout=30)
            assert box, "the reader thread did not finish"
            out = box[0]
        else:
            handle = wrapper._submission.spec_hidden
            if isinstance(handle, FakeFusedHandle):
                handle.release()
            out = wrapper.get_output()
        self.update_from_output(out, k)

    def update_from_output(self, out, k):
        rows = {
            req_id: list(out.sampled_token_ids[out.req_id_to_index[req_id]])
            for req_id in out.req_ids
        }
        self.published[k] = rows
        for req_id, tokens in rows.items():
            if req_id in self.computed and tokens:
                self.computed[req_id] -= K + 1 - len(tokens)
        for req_id, tokens in rows.items():
            if req_id in self.done:
                continue
            stop = self.stop_after.get(req_id)
            if stop is not None:
                tokens = tokens[: stop - len(self.outputs[req_id])]
            self.outputs[req_id].extend(tokens)
            if stop is not None and len(self.outputs[req_id]) >= stop:
                self.done.add(req_id)
                self.finished_pending.add(req_id)
                self.live.remove(req_id)
                self.finish_step[req_id] = k

    def finish(self):
        while self.inflight:
            self.resolve_oldest()


def _controller(runner):
    return runner.async_decode


def _decode_so(runner, live, **kw):
    """A decode-only scheduler output whose requests have all produced output."""
    drafts = kw.pop("drafts", {r: [PLACEHOLDER_TOKEN_ID] * K for r in live})
    so = _scheduler_output(runner, decoding=tuple(live), drafts=drafts, **kw)
    so.scheduled_cached_reqs = dataclasses.replace(
        so.scheduled_cached_reqs, num_output_tokens=[1 for _ in live]
    )
    return so


def _joins_and_finish_run(monkeypatch, chain, *, threaded=False, hold=False, **knobs):
    """Two greedy requests, a join, and a finish, on the chain-capable fake."""
    if chain:
        monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    else:
        monkeypatch.delenv("VLLM_TT_SPEC_CHAIN", raising=False)
    model = _chain_model(accept_cycle=(3, 0, 1, 2), **knobs)
    model.hold_reads = hold
    runner = _make_runner(model)
    engine = Engine(runner, threaded=threaded)
    engine.add("a", 1)
    engine.add("b", 40, stop_after=9)
    for i in range(14):
        if i == 6:
            engine.add("c", 80)
        engine.iterate()
    engine.finish()
    return model, runner, engine


# endregion

# region T3: the lazy mirror


def test_t3_a_chained_step_carries_placeholders_and_optimistic_positions(monkeypatch):
    model, runner, engine = _joins_and_finish_run(monkeypatch, chain=True)
    controller = _controller(runner)
    chained = [c for c in model.chain_log if c["chain_rows"].any()]
    assert chained, "no step was chained"
    assert controller._chained_submissions == len(chained)
    for call in chained:
        live = int(call["chain_rows"].sum())
        assert call["chain_rows"][:live].all() and not call["chain_rows"][live:].any()
        # placeholders in every column, counts -1, on the chained rows only
        assert (call["tokens"][:live] == PLACEHOLDER_TOKEN_ID).all()
        assert (call["accepted_counts"][:live] == -1).all()
        assert (call["tokens"][live:] == 0).all()
        assert (call["accepted_counts"][live:] == 1).all()
        # the provisional position never falls below the real one, and the gap
        # is at most one step's worth of unaccepted drafts per step in flight
        first = call["positions"][:live, 0]
        assert (first >= call["p0"][:live]).all()
        assert (first - call["p0"][:live] <= K).all()
        # the hint is the reservation here (room is never the limit)
        assert (call["num_valid_drafts"][:live] == K).all()
        assert (call["eff_nv"][:live] <= call["num_valid_drafts"][:live]).all()


def test_t3_the_chain_length_is_an_upper_bound_until_the_commit_lands(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = _chain_model()
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.add("a", 1)
    engine.iterate()  # step 0: not chained, nothing pending
    engine.iterate()  # step 1: chained onto step 0
    batch = runner.input_batch
    row = batch.req_id_to_index["a"]
    applied = int(batch.num_tokens[row])
    assert runner._chain_len["a"] == applied + (K + 1)
    assert model.chain_log[1]["positions"][0, 0] == applied + K
    engine.finish()
    engine.iterate()  # applies the finished steps, drops the optimistic length
    assert "a" not in runner._chain_len or runner._chain_len["a"] > int(
        batch.num_tokens[row]
    )


def test_t3_the_applied_state_equals_the_unchained_run(monkeypatch):
    _, chain_runner, chained = _joins_and_finish_run(monkeypatch, chain=True)
    _, plain_runner, plain = _joins_and_finish_run(monkeypatch, chain=False)
    assert chained.outputs == plain.outputs
    assert chained.published == plain.published
    assert _controller(plain_runner)._chained_submissions == 0
    assert _controller(chain_runner)._chained_submissions > 0


# endregion

# region T4: the handle across the reader and engine threads


def test_t4_chained_steps_read_on_the_reader_thread_and_finish_on_the_engine_thread(
    monkeypatch,
):
    model, runner, engine = _joins_and_finish_run(
        monkeypatch, chain=True, threaded=True, hold=True
    )
    controller = _controller(runner)
    main = threading.get_ident()
    assert controller._overlapped_unsafe_submissions == 0
    assert controller._chained_submissions > 0
    # every step that overlapped an outstanding one was a chained step
    assert controller._overlapped_submissions >= controller._chained_submissions
    waited = [h for h in model.handles if h.waited]
    assert waited and all(h.waits == 1 for h in waited)
    # steps resolved by the engine's own loop were read on a reader thread; the
    # ones waited on the engine thread are the drains at a chain break
    on_reader = [h for h in waited if h.wait_thread != main]
    assert len(on_reader) > len(waited) // 2
    assert len(waited) - len(on_reader) <= sum(controller._chain_breaks.values())
    # the lazy finish (propose) ran on the engine thread, after the read, once
    finished = [h for h in model.handles if h.finished]
    assert finished
    assert all(h.waited for h in finished)
    assert len(model.propose_calls) == len(finished)


def test_t4_a_step_is_submitted_before_the_previous_one_is_read(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = _chain_model()
    model.hold_reads = True
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.add("a", 1)
    engine.add("b", 40)
    engine.iterate()
    engine.iterate()  # would read step 0, but the read blocks: release first
    # (iterate released nothing yet; resolve_oldest releases at resolve time)
    # step 1 was enqueued while step 0's handle had not been waited on
    assert model.chain_log[1]["chain_rows"][:2].all()
    assert model.handles[1].tip_id == model.handles[0].tip_id + 1
    assert _controller(runner)._overlapped_unsafe_submissions == 0
    engine.finish()


def test_t4_a_wrong_chain_tip_is_refused_by_the_model(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = _chain_model()
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.add("a", 1)
    engine.iterate()
    runner.async_decode._last_fused_tip += 5
    with pytest.raises(ValueError, match="chain_tip"):
        engine.iterate()


# endregion

# region T5: the stop lag


def test_t5_a_request_that_finishes_at_n_has_its_n_plus_1_output_discarded(monkeypatch):
    model, runner, engine = _joins_and_finish_run(monkeypatch, chain=True)
    n = engine.finish_step["b"]
    # scheduled into n+1 before the finish was known
    assert "b" in engine.step_rows[n + 1]
    # its late output is discarded and nothing extra reaches the stream
    assert engine.published[n + 1]["b"] == []
    assert len(engine.outputs["b"]) == 9
    # n+2 no longer schedules it, and it left the runner
    assert "b" not in engine.step_rows[n + 2]
    assert "b" not in runner.requests
    assert "b" not in runner.input_batch.req_id_to_index
    assert "b" not in runner._req_accepted_counts
    assert "b" not in runner._proposed_draft_token_ids
    assert "b" not in runner._chain_len
    # the finish did not break the chain: n+2 chained with b gone
    call = model.chain_log[n + 2]
    live = len(engine.step_rows[n + 2])
    assert live >= 1
    assert call["chain_rows"][:live].all() and not call["chain_rows"][live:].any()
    assert _controller(runner)._overlapped_unsafe_submissions == 0
    assert not _controller(runner)._chain_breaks.get("finished")


def test_t5_the_discarded_output_matches_the_unchained_run_for_the_others(monkeypatch):
    _, _, chained = _joins_and_finish_run(monkeypatch, chain=True)
    _, _, plain = _joins_and_finish_run(monkeypatch, chain=False)
    for req in ("a", "c"):
        assert chained.outputs[req] == plain.outputs[req]
    assert chained.outputs["b"] == plain.outputs["b"]


# endregion

# region T6: a chain break gives the unchained stream


def test_t6_a_join_breaks_the_chain_and_the_stream_is_unchanged(monkeypatch):
    model, runner, chained = _joins_and_finish_run(monkeypatch, chain=True)
    controller = _controller(runner)
    assert controller._chain_breaks["layout"] >= 1
    # the step after the join is a host-built verify, then chaining resumes
    join_step = 6
    call = model.chain_log[join_step]
    assert not call["chain_rows"].any()
    assert (call["tokens"][:3, 0] >= 0).all()
    assert any(c["chain_rows"].any() for c in model.chain_log[join_step + 1 :])
    _, _, plain = _joins_and_finish_run(monkeypatch, chain=False)
    assert chained.outputs == plain.outputs
    assert chained.published == plain.published
    assert controller._overlapped_unsafe_submissions == 0


def test_t6_the_blockers_by_reason(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = _chain_model()
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.add("a", 1)
    engine.add("b", 40)
    engine.iterate()
    controller = _controller(runner)
    live = ("a", "b")

    def so(**kw):
        return _decode_so(runner, live, **kw)

    assert controller.chain_blocker(so()) is None
    assert controller.chain_blocker(so(new=[("n", 5)])) == "prefill"
    assert controller.chain_blocker(so(resumed=("a",))) == "prefill"
    assert controller.chain_blocker(_decode_so(runner, ("a",))) == "layout"
    from vllm_tt_plugin.scheduler import set_tt_forced_reset_discard_counts

    forced = so()
    set_tt_forced_reset_discard_counts(forced, {"a": 1})
    assert controller.chain_blocker(forced) == "forced_reset"
    real = _decode_so(runner, live, drafts={"a": [1, 2, 3], "b": [1, 2, 3]})
    assert controller.chain_blocker(real) == "real_drafts"
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "0")
    assert controller.chain_blocker(so()) == "disabled"


PREFILL_CAPS = {
    **FakeSpecModel.model_capabilities,
    "supports_async_decode": True,
    "supports_async_spec_decode": True,
    "supports_prefill_over_pending_spec": True,
}


def _prefill_run(
    monkeypatch, *, overlap="1", caps=PREFILL_CAPS, news, resumed=(), cycle=(3, 0, 1, 2)
):
    """Two requests decoding, a prefill step over the pending verify, more steps.

    Returns the engine, the controller and the number of waits the controller
    made during the prefill step.
    """
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_PREFILL_OVERLAP", overlap)
    model = _chain_model(accept_cycle=cycle, model_capabilities=caps)
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.model = model
    controller = _controller(runner)
    waits = []
    real_wait = controller.wait_for_all_pending_async_steps
    controller.wait_for_all_pending_async_steps = lambda: (
        waits.append(1),
        real_wait(),
    )[-1]
    engine.add("a", 1)
    engine.add("b", 40)
    for _ in range(5):
        engine.iterate()
    assert any(not s.is_resolved() for s in runner._pending_async_steps)
    before = len(waits)
    engine.prefill_step(news, resumed=resumed)
    during = len(waits) - before
    for _ in range(8):
        engine.iterate()
    engine.finish()
    return engine, controller, during


@pytest.mark.parametrize("cycle", [(3, 3, 3, 3), None])
@pytest.mark.parametrize("news", [[("c", 80)], [("c", 80), ("d", 120)]])
def test_prefill_overlap_a_prefill_does_not_drain_the_pending_verify(
    monkeypatch, news, cycle
):
    engine, controller, waits = _prefill_run(monkeypatch, news=news, cycle=cycle)
    assert waits == 0
    # the verify was still in flight when the prefill forward ran
    assert engine.prefill_log == [1]
    assert controller._prefill_overlap_steps == 1
    # the step after the prefill cannot chain onto the verify it overtook
    assert controller._chain_breaks["prefill_overlap"] == 1
    assert controller._overlapped_unsafe_submissions == 0
    assert "prefill_overlap=1" in controller._chain_report()
    off, off_controller, off_waits = _prefill_run(
        monkeypatch, overlap="0", news=news, cycle=cycle
    )
    assert off_waits >= 1
    assert off.prefill_log == [0]
    assert off_controller._prefill_overlap_steps == 0
    # the same streams, and the same steps before the prefill
    # The overlapped run speculates one step more, so after the same number of
    # steps its streams are longer: equal over the shorter, each one gap-free.
    for req_id, tokens in engine.outputs.items():
        shorter = min(len(tokens), len(off.outputs[req_id]))
        assert shorter > 8
        assert tokens[:shorter] == off.outputs[req_id][:shorter], req_id
        assert off.outputs[req_id] == list(
            range(tokens[0], tokens[0] + len(off.outputs[req_id]))
        ), req_id
    assert {k: v for k, v in engine.published.items() if k < 5} == {
        k: v for k, v in off.published.items() if k < 5
    }
    for req_id, tokens in engine.outputs.items():
        assert len(tokens) > 8
        assert tokens == list(range(tokens[0], tokens[0] + len(tokens))), req_id
    # The drain loses the speculation step: committed with its rows out of the
    # batch, the verify proposes no drafts, so the first decode after the
    # prefill is draftless; overlapped, it carries the drafts.
    first_decode = 5
    assert int(engine.model.chain_log[first_decode]["num_valid_drafts"].sum()) > 0
    assert int(off.model.chain_log[first_decode]["num_valid_drafts"].sum()) == 0


def test_prefill_overlap_needs_the_capability(monkeypatch):
    caps = {
        k: v
        for k, v in PREFILL_CAPS.items()
        if k != "supports_prefill_over_pending_spec"
    }
    engine, controller, waits = _prefill_run(monkeypatch, caps=caps, news=[("c", 80)])
    assert waits >= 1
    assert engine.prefill_log == [0]
    assert controller._prefill_overlap_steps == 0


def test_prefill_overlap_kill_switch_drains(monkeypatch):
    engine, controller, waits = _prefill_run(monkeypatch, overlap="0", news=[("c", 80)])
    assert waits >= 1
    assert engine.prefill_log == [0]
    assert controller._prefill_overlap_steps == 0


def test_prefill_overlap_a_resumed_request_still_drains(monkeypatch):
    engine, controller, waits = _prefill_run(
        monkeypatch, news=[("c", 80)], resumed=("a",)
    )
    assert waits >= 1
    assert engine.prefill_log == [0]
    assert controller._prefill_overlap_steps == 0


def test_t6_a_second_pending_step_blocks_the_chain(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = _chain_model()
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.add("a", 1)
    engine.iterate()
    engine.iterate()  # step 1 chained; step 0 resolved by the engine loop
    # force a second unresolved step into the pending list
    wrapper = engine.inflight[-1][0]
    runner._pending_async_steps.appendleft(
        type(wrapper)(
            controller=wrapper._controller,
            submission=wrapper._submission,
            model_input=wrapper._model_input,
            completion_event=threading.Event(),
            context=wrapper._context,
        )
    )
    so = _decode_so(runner, ("a",))
    assert _controller(runner).chain_blocker(so) == "multi_pending"


@pytest.mark.parametrize(
    "params, reason",
    [
        (SamplingParams(temperature=0.0, logprobs=1, ignore_eos=True), "logprobs"),
        (
            SamplingParams(temperature=0.0, presence_penalty=0.5, ignore_eos=True),
            "penalties",
        ),
        (
            SamplingParams(
                temperature=0.0, allowed_token_ids=[1, 2, 3], ignore_eos=True
            ),
            "allowed_token_ids",
        ),
    ],
)
def test_t6_a_logprobs_penalty_or_allowlist_row_does_not_chain(
    monkeypatch, params, reason
):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    runner = _make_runner(_chain_model())
    engine = Engine(runner)
    engine.add("a", 1)
    engine.add("p", 40, params)
    so = _decode_so(runner, ("a", "p"))
    assert runner._chain_rows_blocker(["a", "p"], so, 1)[0] == reason


def test_t6_a_row_that_is_not_fused_eligible_does_not_chain(monkeypatch):
    # a sampled row on a model that serves no topk keeps the logits walk
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    runner = _make_runner(_chain_model())
    engine = Engine(runner)
    engine.add("a", 1)
    engine.add("s", 40, SAMPLED)
    so = _decode_so(runner, ("a", "s"))
    assert runner._chain_rows_blocker(["a", "s"], so, 1)[0] == "not_fused"
    assert runner._chain_rows_blocker(["a"], _decode_so(runner, ("a",)), 1)[0] is None


def test_t6_the_length_cap_breaks_the_chain_and_the_stream_is_unchanged(monkeypatch):
    results = {}
    for chain in (False, True):
        if chain:
            monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
        else:
            monkeypatch.delenv("VLLM_TT_SPEC_CHAIN", raising=False)
        model = _chain_model(max_model_len=MAX_MODEL_LEN)
        runner = _make_runner(model)
        engine = Engine(runner)
        engine.add("a", 1, stop_after=MAX_MODEL_LEN - PROMPT_LEN - 1)
        for _ in range(80):
            if "a" in engine.done:
                break
            engine.iterate()
        engine.finish()
        results[chain] = (engine.outputs["a"], runner)
        assert "a" in engine.done
    assert results[True][0] == results[False][0]
    controller = _controller(results[True][1])
    assert controller._chained_submissions > 0
    assert controller._chain_breaks["max_len"] >= 1
    assert controller._overlapped_unsafe_submissions == 0


# endregion

# region T7: stale drafts are never replayed


def test_t7_a_stored_proposal_is_never_verified_for_a_chained_row(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = _chain_model()
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.add("a", 1)
    runner._proposed_draft_token_ids["a"] = [11, 12, 13]  # stale real ids
    so = _scheduler_output(
        runner, decoding=("a",), drafts={"a": [PLACEHOLDER_TOKEN_ID] * K}
    )
    drafts = runner._drafts_to_verify(so, ["a"], {"a": 2})
    assert drafts == {"a": [PLACEHOLDER_TOKEN_ID] * 2}
    assert "a" not in runner._proposed_draft_token_ids


def test_t7_a_length_hint_is_not_verified_by_an_unchained_build(monkeypatch):
    model = _chain_model()
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.add("a", 1)
    runner._proposed_draft_token_ids["a"] = [PLACEHOLDER_TOKEN_ID] * K
    so = _scheduler_output(
        runner, decoding=("a",), drafts={"a": [PLACEHOLDER_TOKEN_ID] * K}
    )
    assert runner._drafts_to_verify(so, ["a"]) == {}
    assert "a" not in runner._proposed_draft_token_ids


def test_t7_a_chained_commit_publishes_only_hints_and_a_final_one_real_ids(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = _chain_model(accept_cycle=(3, 1, 2))
    runner = _make_runner(model)
    published: list[tuple[str, list[int], bool]] = []
    original = runner._publish_draft

    def record(req_id, tokens, device_sampled):
        published.append((req_id, list(tokens), bool(runner._chain_consumed_tips)))
        original(req_id, tokens, device_sampled)

    runner._publish_draft = record
    engine = Engine(runner)
    engine.add("a", 1)
    for _ in range(10):
        engine.iterate()
    hints = [
        p for p in published if p[1] and all(t == PLACEHOLDER_TOKEN_ID for t in p[1])
    ]
    assert hints, "no commit was a chained one"
    assert not any(t >= 0 for p in hints for t in p[1])
    # nothing ever stored is a mix, and no step verified a stale proposal: every
    # chained row carried placeholders only
    assert all(
        (c["tokens"][0] == PLACEHOLDER_TOKEN_ID).all()
        for c in model.chain_log
        if c["chain_rows"].any()
    )
    # after a drain the next unchained step verifies the last step's real drafts
    engine.finish()
    engine.iterate()
    last = model.chain_log[-1]
    assert not last["chain_rows"].any()
    live_tokens = last["tokens"][0]
    assert (live_tokens >= 0).sum() >= 1


# endregion

# region T8: the platform gate


def test_t8_the_chain_fake_declares_what_the_gate_reads():
    caps = type(_chain_model()).model_capabilities
    assert caps["supports_async_spec_decode"] is True
    assert ACCEPT_MODE_FUSED_SAMPLE in type(_chain_model()).accept_modes
    # a model that does not chain does not declare it
    assert not FakeSpecModel.model_capabilities.get("supports_async_spec_decode")


@pytest.mark.parametrize("chain_env, warns", [(None, True), ("0", True), ("1", False)])
def test_t8_async_spec_warns_without_chain_and_keeps_the_gate(
    monkeypatch, vllm_config, chain_env, warns
):
    from tests.spec.test_spec_platform_integration import _run_hook, _speculative

    if chain_env is None:
        monkeypatch.delenv("VLLM_TT_SPEC_CHAIN", raising=False)
    else:
        monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", chain_env)
    vllm_config.scheduler_config.async_scheduling = True
    model = make_fake_spec_model(
        serve_chain=True, max_supported_num_seqs=4, supported_k=(3, 7)
    )
    from vllm_tt_plugin import platform as platform_module

    class _Collect(logging.Handler):
        def __init__(self):
            super().__init__(logging.WARNING)
            self.records = []

        def emit(self, record):
            self.records.append(record)

    collector = _Collect()
    platform_module.logger.addHandler(collector)
    try:
        _run_hook(monkeypatch, _speculative(vllm_config), model)
    finally:
        platform_module.logger.removeHandler(collector)
    assert vllm_config.scheduler_config.async_scheduling is True
    messages = [r.getMessage() for r in collector.records]
    assert any("VLLM_TT_SPEC_CHAIN" in m for m in messages) is warns


def test_t8_no_capability_is_still_refused(monkeypatch, vllm_config):
    from tests.spec.test_spec_platform_integration import _run_hook, _speculative

    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    vllm_config.scheduler_config.async_scheduling = True
    model = make_fake_spec_model(
        max_supported_num_seqs=4,
        model_capabilities={
            **FakeSpecModel.model_capabilities,
            "supports_async_decode": True,
        },
    )
    with pytest.raises(ValueError, match="supports_async_spec_decode"):
        _run_hook(monkeypatch, _speculative(vllm_config), model)


def test_t8_the_gate_defaults_off():
    assert spec_chain_enabled() is False


# endregion

# region Gate off: nothing changes


def test_the_gate_off_never_defers_or_chains(monkeypatch):
    model, runner, engine = _joins_and_finish_run(monkeypatch, chain=False)
    assert all(not c["defer_read"] for c in model.verify_calls)
    assert all(c["chain_rows"] is None for c in model.verify_calls)
    controller = _controller(runner)
    assert controller._chained_submissions == 0
    assert not controller._chain_breaks
    assert controller._overlapped_unsafe_submissions == 0
    assert controller._last_fused_step is None


# endregion

# region The sampled mixed run


class ChainSampledTarget(ChainFusedMixin, ChainReadMixin, _FusedTopkTarget):
    """The sampled target of the S2 tests with its own device drafter and chain mode."""

    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": [],
        "supports_async_decode": True,
        "supports_async_spec_decode": True,
    }
    vocab_size = SAMPLED_VOCAB_SIZE

    def __init__(self):
        super().__init__()
        self._chain_init()
        self.propose_calls: list[object] = []

    def _chain_drafts(self, last: int, position: int) -> list[int]:
        out, token = [], last
        for step in range(self._num_drafts):
            token = base_token(token, position + step)
            out.append(token)
        return out

    def _chain_accept(self, tokens, p0, num_valid, spec_sampling, kwargs):
        from vllm_tt_plugin.spec_accept import (
            SpecSamplingInputs,
            accept_sampled_support,
        )

        rows, width = (int(s) for s in tokens.shape)
        columns = torch.arange(width, dtype=torch.int32).unsqueeze(0)
        positions = p0.unsqueeze(1) + columns
        positions = torch.where(
            (columns > num_valid.unsqueeze(1)) | (p0 < 0).unsqueeze(1),
            torch.full_like(positions, -1),
            positions,
        )
        logits = rule_logits(tokens, positions)
        values, ids = exact_support(logits, self.topk_candidates)
        argmax = logits.argmax(dim=-1).to(torch.int32)
        ss = spec_sampling
        if ss is None:
            temperature = torch.zeros(rows)
            top_k = torch.ones(rows, dtype=torch.int32)
            top_p = torch.ones(rows)
        else:
            temperature, top_k, top_p = ss["temperature"], ss["top_k"], ss["top_p"]
        sampling = SpecSamplingInputs(
            vocab_size=int(logits.shape[-1]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generators={},
        )
        result = accept_sampled_support(
            argmax,
            values,
            ids,
            tokens[:, 1:],
            num_valid,
            sampling,
            None,
            uniforms=None if ss is None else ss["uniforms"],
        )
        return result.committed_token_ids, result.accepted_counts

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        if spec_mode == ACCEPT_MODE_FUSED_SAMPLE:
            # (deferred, carries spec_sampling, chained) per fused step
            self.fused_log = getattr(self, "fused_log", [])
            self.fused_log.append(
                (
                    bool(kwargs.get("defer_read")),
                    kwargs.get("spec_sampling") is not None,
                    kwargs.get("chain_rows") is not None,
                )
            )
        if spec_mode == ACCEPT_MODE_FUSED_SAMPLE and kwargs.get("defer_read"):
            self.verify_modes.append(spec_mode)
            self.spec_sampling_calls.append(kwargs.get("spec_sampling"))
            self._num_drafts = int(tokens.shape[1]) - 1
            return self._chain_forward(
                tokens,
                start_pos,
                kwargs["num_valid_drafts"],
                kwargs["accepted_counts"],
                kwargs.get("spec_sampling"),
                kwargs,
            )
        self._chain_invalidate()
        return super().decode_forward(tokens, start_pos, spec_mode, **kwargs)

    def propose_draft_tokens(
        self,
        num_drafts,
        committed_tokens,
        committed_positions,
        accepted_counts,
        hidden=None,
    ):
        self._num_drafts = num_drafts
        self.propose_calls.append(hidden)
        if isinstance(hidden, FakeFusedHandle):
            return self._chain_propose(
                num_drafts,
                committed_tokens,
                committed_positions,
                accepted_counts,
                hidden,
            )
        rows = int(committed_tokens.shape[0])
        index = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        last = committed_tokens.to(torch.int64).gather(1, index).squeeze(1)
        position = committed_positions.to(torch.int64).gather(1, index).squeeze(1)
        drafts = torch.zeros((rows, num_drafts), dtype=torch.int32)
        for row in range(rows):
            if int(position[row]) >= 0:
                drafts[row] = torch.tensor(
                    self._chain_drafts(int(last[row]), int(position[row])),
                    dtype=torch.int32,
                )
        return DraftOutput(draft_token_ids=drafts)


def _sampled_run(monkeypatch, chain, late_sampled=False):
    if chain:
        monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    else:
        monkeypatch.delenv("VLLM_TT_SPEC_CHAIN", raising=False)
    torch.manual_seed(2024)
    model = ChainSampledTarget()
    runner = _make_runner(model)
    engine = Engine(runner)
    # greedy, unseeded sampled, seeded sampled, and a greedy that finishes early
    engine.add("g1", 1)
    engine.add("g2", 9, stop_after=13)
    if not late_sampled:
        engine.add("s1", 17, SAMPLED)
        engine.add("s2", 25, SEEDED)
    for i in range(30):
        if late_sampled and i == 8:
            engine.add("s1", 17, SAMPLED, stop_after=10)
            engine.add("s2", 25, SEEDED, stop_after=10)
        if i == 9:
            engine.add("j1", 33, stop_after=8)  # joins after g2 finished
        if i == 22:
            engine.add("j2", 41, SEEDED, stop_after=12)  # and after j1
        engine.iterate()
    engine.finish()
    return model, runner, engine


def test_a_mixed_run_has_the_same_streams_with_and_without_chaining(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")
    model_c, runner_c, chained = _sampled_run(monkeypatch, chain=True)
    rng_c = torch.get_rng_state()
    model_p, runner_p, plain = _sampled_run(monkeypatch, chain=False)
    rng_p = torch.get_rng_state()
    assert set(chained.outputs) == {"g1", "g2", "s1", "s2", "j1", "j2"}
    assert all(len(v) >= 8 for v in plain.outputs.values())
    assert chained.outputs == plain.outputs
    assert chained.published == plain.published
    controller = _controller(runner_c)
    assert controller._chained_submissions > 5
    assert controller._overlapped_unsafe_submissions == 0
    # sampled and seeded rows rode the chained steps
    sampled_chained = [c for c in model_c.spec_sampling_calls if c is not None]
    assert sampled_chained
    assert ACCEPT_MODE_FUSED_SAMPLE in set(model_c.verify_modes)
    # a join broke the chain at least once, and g2's finish did not
    assert controller._chain_breaks["layout"] >= 1
    # the global random stream ends where the unchained run leaves it
    assert torch.equal(rng_c, rng_p)
    assert _controller(runner_p)._chained_submissions == 0


def test_t9_chain_sampled_off_keeps_sampled_steps_synchronous(monkeypatch):
    assert spec_chain_sampled_enabled() is False
    model_c, runner_c, chained = _sampled_run(
        monkeypatch, chain=True, late_sampled=True
    )
    rng_c = torch.get_rng_state()
    model_p, runner_p, plain = _sampled_run(monkeypatch, chain=False, late_sampled=True)
    rng_p = torch.get_rng_state()
    assert set(chained.outputs) == {"g1", "g2", "s1", "s2", "j1", "j2"}
    assert chained.outputs == plain.outputs
    assert chained.published == plain.published
    assert torch.equal(rng_c, rng_p)
    # no step that carries spec_sampling was deferred or chained
    sampled_calls = [c for c in model_c.fused_log if c[1]]
    assert sampled_calls
    assert all(not deferred and not chained for deferred, _, chained in sampled_calls)
    # greedy-only steps still defer and chain
    controller = _controller(runner_c)
    assert any(c[0] and not c[1] for c in model_c.fused_log)
    assert controller._chained_submissions >= 1
    assert controller._overlapped_unsafe_submissions == 0


def test_t9_the_blocker_names_a_sampled_row(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = ChainSampledTarget()
    runner = _make_runner(model)
    engine = Engine(runner)
    engine.add("a", 1)
    engine.add("s", 40, SAMPLED)
    so = _decode_so(runner, ("a", "s"))
    assert runner._chain_rows_blocker(["a", "s"], so, 1)[0] == "sampled"
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")
    assert runner._chain_rows_blocker(["a", "s"], so, 1)[0] is None


def test_t9_chain_sampled_on_defers_sampled_steps(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")
    assert spec_chain_sampled_enabled() is True
    model, runner, _ = _sampled_run(monkeypatch, chain=True, late_sampled=True)
    sampled_calls = [c for c in model.fused_log if c[1]]
    assert sampled_calls and any(c[0] for c in sampled_calls)
    assert _controller(runner)._chain_breaks["sampled"] == 0


# endregion
