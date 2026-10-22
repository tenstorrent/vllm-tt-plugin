# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculative decoding in lane mode under asynchronous scheduling.

``test_spec_lane_e2e.py`` drives the synchronous lane tail. These drive the
asynchronous one through the real entry points, ``execute_model`` and
``sample_tokens``, as ``EngineCore.step_with_batch_queue`` does with its queue
of two: schedule step k, submit it, then resolve step k-1 and run the
scheduler's update. A lane step that verifies nothing is the ordinary decode,
submitted over the step still in flight; a verify is deferred through
``AsyncTTSpecDecodeOutput``, walked where it is read, and committed on the
engine thread at the top of the following step.

The model is ``LaneDeviceTarget``: a decode that keeps its own last token and
position per slot, as a traced device decode does, so an overlapped ordinary
step really is built from host state one step behind. Greedy requests must
emit exactly the rule's argmax chain, the same tokens a synchronous run emits.

The layout is two lanes of two slots.
"""

from __future__ import annotations

import inspect
import threading
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import (
    AsyncTTModelRunnerOutput,
    AsyncTTSpecDecodeOutput,
    TTAsyncDecodeController,
)
from vllm_tt_plugin.input_batch import TTLaneInputBatch
from vllm_tt_plugin.lane_scheduler import (
    TTStepPlan,
    _LaneStepState,
    _set_tt_step_state,
)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import (
    _TT_FINISHING_REQ_IDS_ATTR,
    TTScheduler,
    get_tt_verified_drafts,
    set_tt_scheduler_output_field,
)
from vllm_tt_plugin.spec_accept import accept_speculated_tokens
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    PLACEHOLDER_TOKEN_ID,
)

from .lane_device_target import LaneDeviceTarget, greedy_chain, lane_device_target
from .sampled_target import SAMPLED_VOCAB_SIZE, rule_logits

BLOCK_SIZE = 16
MAX_MODEL_LEN = 256
NUM_LANES = 2
PER_LANE = 2
CAPACITY = NUM_LANES * PER_LANE
PROMPT_LEN = 4
K = 3

GREEDY = SamplingParams(temperature=0.0, ignore_eos=True)

# region Harness


def lane_runner(model: LaneDeviceTarget, *, async_scheduling: bool = True):
    """A runner fake over a real lane batch whose methods are the real ones."""
    model.per_lane = PER_LANE
    batch = TTLaneInputBatch(
        num_lanes=NUM_LANES,
        per_lane=PER_LANE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=model.vocab_size,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    plan = type(model).spec_plan(None, CAPACITY, K)
    runner = SimpleNamespace(
        model=model,
        input_batch=batch,
        lane_batch=batch,
        requests={},
        encoder_cache={},
        kv_caches=object(),
        trace_mode="decode_only",
        request_specific_rope=False,
        supports_stage_outputs=False,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        _num_speculative_tokens=K,
        _spec_plan=plan,
        _spec_method="custom_class",
        _spec_supports_narrow_decode=plan.supports_narrow_decode,
        _spec_accept_modes=tuple(plan.accept_modes),
        _spec_drafts_from_model=True,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        _req_state_slot={},
        _ngram_proposer=None,
        tt_per_lane_max_num_seqs=PER_LANE,
        tt_data_parallel_size=NUM_LANES,
        vocab_size=model.vocab_size,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        supports_stage_inputs=False,
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            max_model_len=MAX_MODEL_LEN,
            logits_processors=None,
        ),
        # Set below: an ordinary lane decode samples on the device unless a
        # request needs its logits; a verify that needs logits or accepts on
        # the device turns it off in the builder.
        check_perform_device_sampling=None,
        host_sampler=Sampler(),
        _block_tables_per_layer=lambda _: None,
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
        async_decode_scheduling=async_scheduling,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        _steady_decode_lock=threading.Lock(),
        _pending_async_steps=deque(),
        _pending_async_overlap_ok=deque(),
        _completed_decode_steps=deque(),
        _pending_samples=deque(),
        _spec_row_state=TTModelRunner._spec_row_state,
        _spec_candidate_block=TTModelRunner._spec_candidate_block,
        _committed_positions=TTModelRunner._committed_positions,
    )

    def note_layout_consumed():
        runner._decode_layout_changed_since_last_decode = False

    runner.note_decode_layout_consumed = note_layout_consumed
    runner.check_perform_device_sampling = lambda **_: not any(
        runner._request_has_host_only_controls(req_id) for req_id in runner.requests
    )
    for name, member in vars(TTModelRunner).items():
        if hasattr(runner, name):
            continue
        if isinstance(member, staticmethod):
            setattr(runner, name, member.__func__)
        elif isinstance(member, classmethod):
            setattr(runner, name, member.__func__.__get__(TTModelRunner))
        elif inspect.isfunction(member):
            setattr(runner, name, member.__get__(runner))
    runner.async_decode = TTAsyncDecodeController(runner)
    return runner


def add_request(runner, req_id: str, row: int, first_token: int, params=GREEDY):
    """A fully prefilled request at its stable slot: the prompt, then the first
    token the prefill sampled, which is the rule's argmax (or, sampled, a
    token every run of a test shares)."""
    generator = None
    if params.sampling_type == SamplingType.RANDOM_SEED:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(params.seed)
    prompt = [(first_token + i) % SAMPLED_VOCAB_SIZE for i in range(PROMPT_LEN)]
    first = greedy_chain(prompt[-1], PROMPT_LEN - 1, 1)
    request = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=params,
        generator=generator,
        block_ids=([row + 1],),
        num_computed_tokens=PROMPT_LEN,
        output_token_ids=list(first),
    )
    runner.requests[req_id] = request
    runner.lane_batch.add_request_to_row(request, row)
    runner._decode_layout_changed_since_last_decode = True
    return prompt + first


class LaneEngine:
    """``step_with_batch_queue`` with a queue of two over one lane runner.

    The scheduler half is what ``TTLaneCoordinator`` and ``AsyncScheduler``
    hand the runner: every live request decodes each step with ``K``
    placeholder drafts reserved, the computed-token count runs ahead by
    ``1+K`` per scheduled step and comes back by the rejected count when the
    output lands, and a request that reaches its ``stop_after`` is reported
    finished at the next schedule (the stop lag a batch queue has).

    On a synchronous runner (``depth`` 1) it is ``EngineCore.step`` instead:
    the proposals reach the next schedule through ``take_draft_token_ids``,
    as real draft ids.
    """

    def __init__(self, runner, *, depth: int = 2):
        self.runner = runner
        self.depth = depth
        self.drafts: dict[str, list[int]] = {}
        self.stats = stats_scheduler(runner.async_decode_scheduling)
        # Resolve each output on a reader thread, releasing held reads.
        self.resolve_on_thread = False
        self.live: dict[str, int] = {}
        self.outputs: dict[str, list[int]] = {}
        self.stop_after: dict[str, int] = {}
        self.finished_pending: set[str] = set()
        self.done: set[str] = set()
        self.computed: dict[str, int] = {}
        # AsyncScheduler's num_output_placeholders, and each request's
        # max_tokens (which counts the token the prefill sampled).
        self.placeholders: dict[str, int] = {}
        self.max_tokens: dict[str, int] = {}
        self.finishing_log: list[tuple[str, ...]] = []
        # Requests admitted by ``prefill``, whose first output is the prefill's.
        self.prefilled: set[str] = set()
        self.first: dict[str, int] = {}
        self.inflight: deque = deque()
        self.wrappers: list = []

    def add(
        self, req_id, row, first_token, params=GREEDY, stop_after=None, max_tokens=None
    ):
        tokens = add_request(self.runner, req_id, row, first_token, params)
        self.live[req_id] = row
        self.outputs[req_id] = []
        self.computed[req_id] = len(tokens) - 1
        self.placeholders[req_id] = 0
        if max_tokens is not None:
            self.max_tokens[req_id] = max_tokens
            stop_after = max_tokens - 1
        if stop_after is not None:
            self.stop_after[req_id] = stop_after
        return tokens

    def prefill(self, req_id, row, first_token, params=GREEDY):
        """Admit a request through a lane prefill step, submitted like any
        step and queued behind the steps in flight. Returns its prompt; the
        first token is the prefill's output."""
        prompt = [(first_token + i) % SAMPLED_VOCAB_SIZE for i in range(PROMPT_LEN)]
        so = SchedulerOutput.make_empty()
        so.scheduled_new_reqs = [
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt,
                mm_features=[],
                sampling_params=params,
                pooling_params=None,
                block_ids=([row + 1],),
                num_computed_tokens=0,
                lora_request=None,
            )
        ]
        so.num_scheduled_tokens = {req_id: PROMPT_LEN}
        so.total_num_scheduled_tokens = PROMPT_LEN
        so.finished_req_ids, self.finished_pending = self.finished_pending, set()
        for done in so.finished_req_ids:
            self.live.pop(done, None)
        lanes = [0] * NUM_LANES
        lanes[row // PER_LANE] = 1
        plan = TTStepPlan(
            is_decode=False,
            capacity=CAPACITY,
            scheduled_req_ids=(req_id,),
            scheduled_rows=(row,),
            input_rows=(row,),
            req_id_to_row={**self.live, req_id: row},
            batch_size_per_dp=tuple(lanes),
            prefill_empty_slots=(row,),
            decode_layout_changed=True,
        )
        _set_tt_step_state(so, _LaneStepState(lane_outputs=[], plan=plan))
        assert self.runner.execute_model(so) is None
        out = self.runner.sample_tokens(None)
        self.prefilled.add(req_id)
        self.live[req_id] = row
        self.outputs[req_id] = []
        self.computed[req_id] = PROMPT_LEN
        self.placeholders[req_id] = 1
        self.inflight.append((out, so))
        if len(self.inflight) >= self.depth:
            self.resolve_oldest()
        return prompt

    def schedule_preview(self) -> SchedulerOutput:
        """The next decode's scheduler output, without consuming the step."""
        state = (
            dict(self.live),
            set(self.finished_pending),
            dict(self.computed),
            dict(self.placeholders),
            list(self.finishing_log),
            dict(self.drafts),
        )
        so = self.schedule()
        (
            self.live,
            self.finished_pending,
            self.computed,
            self.placeholders,
            self.finishing_log,
            self.drafts,
        ) = state
        return so

    def _reaches_max_tokens(self, req_id: str) -> bool:
        """``request_will_reach_max_tokens`` on this scheduler's books."""
        limit = self.max_tokens.get(req_id)
        held = self.placeholders[req_id]
        return (
            limit is not None
            and held > 0
            and self.computed[req_id] + 2 - held >= PROMPT_LEN + limit
        )

    def schedule(self) -> SchedulerOutput:
        finished, self.finished_pending = self.finished_pending, set()
        for req_id in finished:
            self.live.pop(req_id, None)
        finishing = tuple(r for r in self.live if self._reaches_max_tokens(r))
        self.finishing_log.append(finishing)
        live = sorted((r for r in self.live if r not in finishing), key=self.live.get)
        so = SchedulerOutput.make_empty()
        if finishing:
            set_tt_scheduler_output_field(so, _TT_FINISHING_REQ_IDS_ATTR, finishing)
        if self.runner.async_decode_scheduling:
            offered = {r: [PLACEHOLDER_TOKEN_ID] * K for r in live}
        else:
            offered = {r: self.drafts[r] for r in live if self.drafts.get(r)}
            self.drafts = {}
        so.num_scheduled_tokens = {r: 1 + len(offered.get(r, ())) for r in live}
        so.total_num_scheduled_tokens = sum(so.num_scheduled_tokens.values())
        so.scheduled_spec_decode_tokens = offered
        so.scheduled_cached_reqs = CachedRequestData(
            req_ids=list(live),
            resumed_req_ids=set(),
            new_token_ids=[[] for _ in live],
            all_token_ids={},
            new_block_ids=[None for _ in live],
            num_computed_tokens=[self.computed[r] for r in live],
            num_output_tokens=[len(self.outputs[r]) + 1 for r in live],
        )
        so.finished_req_ids = set(finished)
        rows = tuple(sorted(self.live[r] for r in live))
        plan = TTStepPlan(
            is_decode=True,
            capacity=CAPACITY,
            scheduled_req_ids=tuple(live),
            scheduled_rows=rows,
            input_rows=tuple(range(CAPACITY)),
            req_id_to_row=dict(self.live),
            batch_size_per_dp=(PER_LANE,) * NUM_LANES,
            prefill_empty_slots=None,
            # A finish frees a slot (``apply_step_plan``); a join is placed
            # by ``add`` directly and marks the runner's layout itself.
            decode_layout_changed=bool(finished),
        )
        _set_tt_step_state(so, _LaneStepState(lane_outputs=[], plan=plan))
        for r in live:
            self.computed[r] += so.num_scheduled_tokens[r]
            self.placeholders[r] += so.num_scheduled_tokens[r]
        return so

    def iterate(self) -> None:
        if not self.live and not self.finished_pending:
            # Nothing left to schedule: the engine only drains its queue.
            self.drain()
            return
        so = self.schedule()
        if not so.total_num_scheduled_tokens:
            # The runner still applies the finishes; no sampler is enqueued.
            self.runner.execute_model(so)
            return
        assert self.runner.execute_model(so) is None
        wrapper = self.runner.sample_tokens(None)
        self.wrappers.append(wrapper)
        self.inflight.append((wrapper, so))
        if len(self.inflight) >= self.depth:
            self.resolve_oldest()

    def resolve_oldest(self) -> None:
        wrapper, so = self.inflight.popleft()
        if self.resolve_on_thread and hasattr(wrapper, "get_output"):
            box: list = []
            reader = threading.Thread(target=lambda: box.append(wrapper.get_output()))
            reader.start()
            for handle in getattr(self.runner.model, "handles", ()):
                handle.release()
            reader.join(timeout=30)
            assert box, "the reader thread did not finish"
            out = box[0]
        else:
            for handle in getattr(self.runner.model, "handles", ()):
                handle.release()
            out = wrapper.get_output() if hasattr(wrapper, "get_output") else wrapper
        if not self.runner.async_decode_scheduling:
            handed = self.runner.take_draft_token_ids()
            if handed is not None:
                self.drafts = dict(
                    zip(handed.req_ids, handed.draft_token_ids, strict=True)
                )
        observe_spec_stats(self.stats, so, out)
        for req_id, index in out.req_id_to_index.items():
            if req_id in self.done or req_id not in self.outputs:
                continue
            tokens = list(out.sampled_token_ids[index])
            if not tokens:
                continue
            if req_id in self.prefilled and so.scheduled_new_reqs:
                # The prefill's own token: the request's first, not an output
                # of its decode steps.
                self.placeholders[req_id] -= 1
                self.first[req_id] = tokens[0]
                continue
            self.computed[req_id] -= so.num_scheduled_tokens[req_id] - len(tokens)
            self.placeholders[req_id] -= so.num_scheduled_tokens[req_id]
            limit = self.stop_after.get(req_id)
            if limit is not None:
                tokens = tokens[: limit - len(self.outputs[req_id])]
            self.outputs[req_id].extend(tokens)
            if limit is not None and len(self.outputs[req_id]) >= limit:
                self.done.add(req_id)
                self.finished_pending.add(req_id)

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.iterate()

    def drain(self) -> None:
        while self.inflight:
            self.resolve_oldest()


def stats_scheduler(async_scheduling: bool) -> TTScheduler:
    """The speculative-metrics half of a lane ``TTScheduler``, nothing else."""
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.log_stats = False
    scheduler.num_spec_tokens = K
    scheduler.collect_spec_decoding_stats = True
    scheduler.pending_spec_decoding_stats = None
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
    return scheduler


def observe_spec_stats(scheduler: TTScheduler, so, out) -> None:
    """What ``TTScheduler.update_from_output`` does for the metrics: mark the
    unverified offer invalid, then vLLM's per-request observation."""
    if scheduler.scheduler_config.async_scheduling:
        scheduler._mark_unverified_drafts_invalid(so, out)
    for req_id, index in out.req_id_to_index.items():
        offered = so.scheduled_spec_decode_tokens.get(req_id)
        generated = out.sampled_token_ids[index]
        if offered and generated:
            scheduler.make_spec_decoding_stats(
                None,
                len(offered),
                len(generated) - 1,
                so.num_invalid_spec_tokens,
                req_id,
            )


def expected_greedy(tokens: list[int], count: int) -> list[int]:
    """The argmax chain after a request's prompt and first token."""
    return greedy_chain(tokens[-1], len(tokens) - 1, count)


def overlap_counters(runner) -> tuple[int, int]:
    controller = runner.async_decode
    return (
        controller._overlapped_submissions,
        controller._overlapped_unsafe_submissions,
    )


# endregion Harness


# region Ordinary steps overlap, verifies drain


def test_ordinary_lane_steps_overlap_in_a_speculating_launch():
    """Two requests per lane: the solo drafter offers nothing, so every step is
    the ordinary decode, each submitted over the one still in flight, and
    each request emits the argmax chain."""
    model = lane_device_target(draft_policy="solo")()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    prompts = {
        "a": engine.add("a", 0, 1),
        "b": engine.add("b", 1, 9),
        "c": engine.add("c", 2, 17),
        "d": engine.add("d", 3, 25),
    }
    engine.run(12)
    engine.drain()

    assert model.verify_modes == []
    overlapped, unsafe = overlap_counters(runner)
    assert overlapped >= 10
    assert unsafe == 0
    # Only the first step after the joins reloaded its inputs from the host.
    reloads = [call["reload_inputs"] for call in model.calls]
    assert reloads[0] is True and not any(reloads[1:])
    for req_id, tokens in prompts.items():
        produced = engine.outputs[req_id]
        assert len(produced) == 12
        assert produced == expected_greedy(tokens, 12)
    assert all(isinstance(w, AsyncTTModelRunnerOutput) for w in engine.wrappers)


def test_a_lane_verify_drains_and_commits_a_step_later():
    """One request per lane: every step after the first verifies. A verify is
    never submitted over an outstanding step, it is deferred through the
    speculative tail, and the run emits the argmax chain."""
    model = lane_device_target(accept_modes=(ACCEPT_MODE_ARGMAX_IDS, "logits"))()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    prompts = {"a": engine.add("a", 0, 1), "b": engine.add("b", 3, 9)}
    engine.run(10)
    engine.drain()

    assert model.plain_steps >= 1
    assert set(model.verify_modes) == {ACCEPT_MODE_ARGMAX_IDS}
    assert any(isinstance(w, AsyncTTSpecDecodeOutput) for w in engine.wrappers)
    _, unsafe = overlap_counters(runner)
    assert unsafe == 0
    for req_id, tokens in prompts.items():
        produced = engine.outputs[req_id]
        assert produced == expected_greedy(tokens, len(produced))
        assert len(produced) > 10, "no draft was ever accepted"


def test_mixed_lanes_verify_with_the_draftless_lane_riding_along():
    """Lane 0 holds two requests (no drafts), lane 1 one (drafted): the verify
    carries both, with nothing to verify for lane 0's rows, and every request
    still emits the argmax chain."""
    model = lane_device_target(draft_policy="solo")()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    prompts = {
        "a": engine.add("a", 0, 1),
        "b": engine.add("b", 1, 9),
        "c": engine.add("c", 3, 17),
    }
    engine.run(10)
    engine.drain()

    verifies = [c for c in model.calls if c["spec_mode"] is not None]
    assert verifies
    for call in verifies:
        assert call["num_valid_drafts"][:2].tolist() == [0, 0]
        assert call["positions"][2].tolist() == [-1] * (K + 1)
    for req_id, tokens in prompts.items():
        produced = engine.outputs[req_id]
        assert produced == expected_greedy(tokens, len(produced))
    assert len(engine.outputs["c"]) > len(engine.outputs["a"])


@pytest.mark.parametrize("policy", ["always", "solo"])
def test_async_lane_run_emits_what_the_synchronous_one_does(policy):
    """Token for token, against the synchronous lane tail on the same model."""

    def run(async_scheduling: bool):
        model = lane_device_target(draft_policy=policy)()
        runner = lane_runner(model, async_scheduling=async_scheduling)
        engine = LaneEngine(runner, depth=2 if async_scheduling else 1)
        engine.add("a", 0, 1)
        engine.add("b", 1, 9)
        engine.add("c", 3, 17)
        engine.run(14)
        engine.drain()
        return engine.outputs

    sync, async_ = run(False), run(True)
    for req_id, tokens in sync.items():
        n = min(len(tokens), len(async_[req_id]))
        assert n > 10
        assert async_[req_id][:n] == tokens[:n]


# endregion Ordinary steps overlap, verifies drain


# region Speculative metrics


@pytest.mark.parametrize("policy", ["always", "solo"])
def test_spec_metrics_count_only_what_a_verify_carried(policy):
    """Every asynchronous step is offered K placeholders, but only a verify's
    real drafts count as drafted, and an ordinary step is no draft at all: the
    drafted total is what the verifies carried, as it is synchronously, where
    the scheduler delivers the real drafts."""

    def run(async_scheduling: bool):
        model = lane_device_target(draft_policy=policy)()
        runner = lane_runner(model, async_scheduling=async_scheduling)
        engine = LaneEngine(runner, depth=2 if async_scheduling else 1)
        engine.add("a", 0, 1)
        engine.add("b", 1, 9)
        engine.add("c", 3, 17)
        engine.run(14)
        engine.drain()
        verify_calls = [c for c in model.calls if c["spec_mode"] is not None]
        drafted = sum(
            int(c["num_valid_drafts"][row])
            for c in verify_calls
            for row in range(CAPACITY)
            if int(c["positions"][row, 0]) >= 0
        )
        return engine, drafted

    for async_scheduling in (False, True):
        engine, drafted = run(async_scheduling)
        stats = engine.stats.pending_spec_decoding_stats
        assert stats is not None
        assert stats.num_draft_tokens == drafted
        assert 0 < stats.num_accepted_tokens < drafted


def test_a_verify_output_reports_its_real_draft_counts():
    model = lane_device_target(draft_policy="solo")()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    engine.add("a", 0, 1)
    engine.add("b", 1, 9)
    engine.add("c", 3, 17)
    engine.run(6)
    engine.drain()
    reports = [
        get_tt_verified_drafts(w.get_output())
        for w in engine.wrappers
        if isinstance(w, AsyncTTSpecDecodeOutput)
    ]
    assert reports
    # Lane 0 holds two requests and drafts nothing; lane 1's request K.
    assert all(report == {"a": 0, "b": 0, "c": K} for report in reports)
    plain = [w for w in engine.wrappers if isinstance(w, AsyncTTModelRunnerOutput)]
    assert plain and all(get_tt_verified_drafts(w.get_output()) == {} for w in plain)


def test_an_unverified_offer_is_no_draft():
    scheduler = stats_scheduler(True)
    so = SchedulerOutput.make_empty()
    so.scheduled_spec_decode_tokens = {r: [PLACEHOLDER_TOKEN_ID] * K for r in "abc"}
    so.num_invalid_spec_tokens = {"a": 1}
    out = SimpleNamespace()
    from vllm_tt_plugin.scheduler import set_tt_verified_drafts

    set_tt_verified_drafts(out, {"a": K, "b": 1})
    scheduler._mark_unverified_drafts_invalid(so, out)
    # A grammar-invalidated draft stays invalid; b verified one of three; c,
    # not reported, verified none.
    assert so.num_invalid_spec_tokens == {"a": 1, "b": 2, "c": K}
    for req_id, accepted in (("a", 1), ("b", 1), ("c", 0)):
        scheduler.make_spec_decoding_stats(
            None, K, accepted, so.num_invalid_spec_tokens, req_id
        )
    stats = scheduler.pending_spec_decoding_stats
    assert (stats.num_drafts, stats.num_draft_tokens, stats.num_accepted_tokens) == (
        2,
        (K - 1) + 1,
        2,
    )


# endregion Speculative metrics


# region Chained fused verifies


def chain_counters(runner) -> tuple[int, dict[str, int]]:
    controller = runner.async_decode
    return controller._chained_submissions, dict(controller._chain_breaks)


def test_greedy_lanes_chain_and_emit_the_synchronous_tokens():
    """One request per lane: after the first verify every verify is submitted
    over the one in flight, built by the device from that verify's own
    acceptance and drafts, and the run emits exactly the synchronous tokens."""

    def run(async_scheduling: bool):
        model = lane_device_target()()
        runner = lane_runner(model, async_scheduling=async_scheduling)
        engine = LaneEngine(runner, depth=2 if async_scheduling else 1)
        prompts = {"a": engine.add("a", 0, 1), "b": engine.add("b", 2, 9)}
        engine.run(20)
        engine.drain()
        return model, runner, engine, prompts

    _, _, sync_engine, _ = run(False)
    model, runner, engine, prompts = run(True)

    chained, breaks = chain_counters(runner)
    assert chained >= 15
    assert breaks == {}
    calls = model.chained_calls
    assert len(calls) == chained
    for call in calls:
        assert call["defer_read"]
        assert call["chain_rows"].tolist() == [True, False, True, False]
        assert (call["tokens"] == PLACEHOLDER_TOKEN_ID).all()
        assert call["accepted_counts"].tolist()[0] == PLACEHOLDER_TOKEN_ID
        assert call["accepted_counts"].tolist()[2] == PLACEHOLDER_TOKEN_ID
        assert call["num_valid_drafts"].tolist() == [K, 0, K, 0]
        # A chained row's provisional position is an upper bound of the
        # device's, never below it; the gaps carry none.
        for row in (0, 2):
            assert int(call["positions"][row, 0]) >= int(
                call["device_positions"][row, 0]
            )
        assert (call["positions"][[1, 3]] == -1).all()
    # Every deferred verify was read back before it was proposed, once; the
    # last ones commit at a step the run never takes.
    assert all(handle.proposed for handle in model.handles[:-2])
    _, unsafe = overlap_counters(runner)
    assert unsafe == 0
    for req_id, tokens in prompts.items():
        produced = engine.outputs[req_id]
        assert produced == expected_greedy(tokens, len(produced))
        n = min(len(produced), len(sync_engine.outputs[req_id]))
        assert n > 30
        assert produced[:n] == sync_engine.outputs[req_id][:n]
        assert len(produced) > 3 * 20 / 2, "chaining lost the accepted drafts"


def test_a_chained_commit_publishes_no_proposal():
    """The device took a chained-onto verify's drafts already: its commit
    still hands the drafter the handle, from the device's positions, but
    publishes nothing for the host to verify."""
    model = lane_device_target()()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    engine.add("a", 0, 1)
    engine.run(8)
    proposals_after_chained_commit = []
    original = runner.commit_spec_acceptance

    def commit(model_input, *args, **kwargs):
        result = original(model_input, *args, **kwargs)
        proposals_after_chained_commit.append(dict(runner._proposed_draft_token_ids))
        return result

    runner.commit_spec_acceptance = commit
    engine.run(6)
    assert proposals_after_chained_commit
    assert all(not p for p in proposals_after_chained_commit)
    handle_calls = [c for c in model.propose_calls if c["hidden"] is not None]
    assert len(handle_calls) >= 10
    assert len({id(c["hidden"]) for c in handle_calls}) == len(handle_calls)


def test_sampled_chained_verifies_draw_from_each_request_s_own_generator():
    """A chained fused verify of seeded and unseeded sampled requests commits
    exactly what the logits walk would commit over the block the device built,
    from the generator states before the step, and leaves each seeded
    generator where that walk leaves it."""
    seeded = SamplingParams(temperature=1.0, top_p=0.95, seed=7, ignore_eos=True)
    top_k = SamplingParams(temperature=0.8, top_k=3, seed=11, ignore_eos=True)
    model = lane_device_target()()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    engine.add("s", 0, 5, params=seeded)
    engine.add("k", 2, 9, params=top_k)
    snapshots: list[dict[int, torch.Tensor]] = []
    original = runner.async_decode.submit_decode

    def submit(model_input, **kwargs):
        snapshots.append(
            {
                slot: generator.get_state()
                for slot, generator in runner.lane_batch.sampling.generators.items()
            }
        )
        return original(model_input, **kwargs)

    runner.async_decode.submit_decode = submit
    engine.run(16)
    engine.drain()

    chained = model.chained_calls
    assert len(chained) >= 8
    checked = 0
    for index, call in enumerate(model.calls):
        if call.get("chain_rows") is None:
            continue
        sampling = call["spec_sampling"]
        assert sampling is not None
        assert sampling["greedy"].tolist() == [False, True, False, True]
        handle = next(h for h in model.handles if h.tip_id == call["chain_tip"] + 1)
        states = snapshots[index]
        for row, params in ((0, seeded), (2, top_k)):
            generator = torch.Generator(device="cpu")
            generator.set_state(states[row])
            tokens = call["device_tokens"][row : row + 1]
            positions = call["device_positions"][row : row + 1]
            result = accept_speculated_tokens(
                target_logits=rule_logits(tokens, positions),
                draft_token_ids=tokens[:, 1:],
                num_valid_drafts=call["num_valid_drafts"][row : row + 1],
                temperature=torch.tensor([params.temperature]),
                top_k=torch.tensor([params.top_k if params.top_k > 0 else 64]),
                top_p=torch.tensor([params.top_p]),
                generators={0: generator},
            )
            count = int(result.accepted_counts[0])
            readout = handle.readout
            assert int(readout.accepted_counts[row]) == count
            assert (
                readout.accepted_token_ids[row, :count].tolist()
                == result.committed_token_ids[0, :count].tolist()
            )
            if index + 1 < len(snapshots):
                assert generator.get_state().equal(snapshots[index + 1][row])
            checked += 1
    assert checked >= 16
    assert all(len(engine.outputs[r]) > 20 for r in ("s", "k"))


def test_a_model_that_answers_defer_read_with_host_tensors_is_drained():
    """Declaring ``supports_async_spec_decode`` with ``fused_sample`` does not
    oblige a model to chain: one that answers ``defer_read`` with host tensors
    has each fused verify read at once and drained, ordinary steps still
    overlap, and the run emits the argmax chain."""
    model = lane_device_target(honor_defer_read=False)()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    prompts = {"a": engine.add("a", 0, 1), "b": engine.add("b", 2, 9)}
    engine.run(12)
    engine.drain()
    assert model.chained_calls == []
    assert chain_counters(runner)[0] == 0
    verifies = [c for c in model.calls if c["spec_mode"] is not None]
    assert verifies and all(c["defer_read"] for c in verifies)
    _, unsafe = overlap_counters(runner)
    assert unsafe == 0
    for req_id, tokens in prompts.items():
        produced = engine.outputs[req_id]
        assert produced == expected_greedy(tokens, len(produced))
        assert len(produced) > 20


def test_a_lane_with_two_requests_keeps_the_step_from_chaining():
    """All or nothing for now: lane 0's two requests draft nothing (the solo
    drafter), so no step chains, each verify drains, and every request still
    emits the argmax chain."""
    model = lane_device_target(draft_policy="solo")()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    prompts = {
        "a": engine.add("a", 0, 1),
        "b": engine.add("b", 1, 9),
        "c": engine.add("c", 3, 17),
    }
    engine.run(12)
    engine.drain()
    chained, breaks = chain_counters(runner)
    assert chained == 0
    assert set(breaks) == {"draft_count"}
    assert model.chained_calls == []
    for req_id, tokens in prompts.items():
        produced = engine.outputs[req_id]
        assert produced == expected_greedy(tokens, len(produced))


def test_a_request_that_stops_mid_chain_leaves_the_others_chaining():
    """``b`` stops after 9 tokens. Its stop reaches the runner a step late, so
    one more chained verify carries it; that output is discarded and from the
    next step ``a`` chains alone."""
    model = lane_device_target()()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    prompts = {"a": engine.add("a", 0, 1), "b": engine.add("b", 2, 9, stop_after=9)}
    engine.run(18)
    engine.drain()
    assert engine.outputs["b"] == expected_greedy(prompts["b"], 9)
    produced = engine.outputs["a"]
    assert produced == expected_greedy(prompts["a"], len(produced))
    late = [c for c in model.chained_calls if c["chain_rows"].tolist()[2]]
    alone = [
        c
        for c in model.chained_calls
        if c["chain_rows"].tolist() == [True, False, False, False]
    ]
    assert late and alone
    assert "b" not in runner.requests
    assert int(runner.lane_batch.num_tokens[2]) == 0


@pytest.mark.parametrize(
    "params, reason",
    [
        (SamplingParams(temperature=0.0, logprobs=1, ignore_eos=True), "not_fused"),
        (
            SamplingParams(temperature=1.0, top_k=3, presence_penalty=0.5),
            "not_fused",
        ),
    ],
    ids=["logprobs", "penalty"],
)
def test_a_request_the_device_cannot_accept_keeps_the_step_unchained(params, reason):
    model = lane_device_target()()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    prompts = {"a": engine.add("a", 0, 1), "x": engine.add("x", 2, 9, params=params)}
    engine.run(10)
    engine.drain()
    assert model.chained_calls == []
    assert chain_counters(runner)[0] == 0
    produced = engine.outputs["a"]
    assert produced == expected_greedy(prompts["a"], len(produced))
    assert ACCEPT_MODE_LOGITS in model.verify_modes


def test_chaining_stops_short_of_max_model_len():
    """Two blocks of headroom or no chain: near the end the verifies drain, and
    the request still emits the argmax chain up to the length cap."""
    model = lane_device_target()()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    # The scheduler finishes a request at max_model_len (FINISHED_LENGTH_CAPPED).
    room = MAX_MODEL_LEN - (PROMPT_LEN + 1)
    tokens = engine.add("a", 0, 1, stop_after=room)
    engine.run(120)
    engine.drain()
    produced = engine.outputs["a"]
    assert len(tokens) + len(produced) == MAX_MODEL_LEN
    assert produced == expected_greedy(tokens, len(produced))
    chained, breaks = chain_counters(runner)
    assert chained > 10
    assert "max_len" in breaks


def test_the_readback_runs_on_another_thread():
    """The engine resolves outputs on whatever thread it likes: the handle is
    read there, the commit and the proposal stay on the engine thread."""
    model = lane_device_target(hold_reads=True)()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    prompts = {"a": engine.add("a", 0, 1)}
    engine.resolve_on_thread = True
    engine.run(12)
    engine.drain()
    produced = engine.outputs["a"]
    assert produced == expected_greedy(prompts["a"], len(produced))
    readers = {t for h in model.handles for t in h.read_threads}
    assert readers and model.engine_thread not in readers
    assert chain_counters(runner)[0] >= 6


@pytest.mark.parametrize(
    "offset, error, match",
    [
        # Past the provisional upper bound: refused where the handle is read.
        (100, ValueError, "column-0 position"),
        # Under it, but not where the host's history ends: refused at the commit.
        (-1, RuntimeError, "history ends at"),
    ],
)
def test_a_misreported_chained_position_is_refused(offset, error, match):
    model = lane_device_target(misreport_p0=offset)()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    engine.add("a", 0, 1)
    with pytest.raises(error, match=match):
        engine.run(10)


def test_a_deferred_verify_nobody_asked_for_is_refused():
    """A model that answers a fused verify with a handle when the runner did
    not send ``defer_read`` (a synchronous launch) is refused by name."""
    model = lane_device_target()()
    runner = lane_runner(model, async_scheduling=False)
    original = model._fused

    def always_deferred(tokens, start_pos, kwargs, call):
        call["defer_read"] = True
        return original(tokens, start_pos, kwargs, call)

    model._fused = always_deferred
    engine = LaneEngine(runner, depth=1)
    engine.add("s", 0, 5, params=SamplingParams(temperature=1.0, top_k=3, seed=3))
    with pytest.raises(ValueError, match="did not send defer_read"):
        engine.run(6)


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "undeclared"])
def test_a_request_finishing_at_max_tokens_frees_its_slot_and_the_chain_goes_on(
    declared,
):
    """The scheduler leaves ``a`` out once its verify in flight is sure to reach
    ``max_tokens``. That step still chains for ``b``; a model declaring
    ``supports_finishing_slots`` is told ``a``'s slot, once, and frees it."""
    capabilities = dict(LaneDeviceTarget.model_capabilities)
    if declared:
        capabilities["supports_finishing_slots"] = True
    model = lane_device_target(model_capabilities=capabilities)()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    # 15 lands a verify in flight that is sure to reach it (the skip fires
    # only when even an all-rejected step reaches max_tokens).
    prompts = {
        "a": engine.add("a", 0, 1, max_tokens=15),
        "b": engine.add("b", 2, 9),
    }
    engine.run(16)
    engine.drain()

    assert engine.outputs["a"] == expected_greedy(prompts["a"], 14)
    produced = engine.outputs["b"]
    assert produced == expected_greedy(prompts["b"], len(produced))
    left_out = [i for i, ids in enumerate(engine.finishing_log) if "a" in ids]
    assert len(left_out) == 1
    sent = [c["finishing_slots"] for c in model.calls if c["finishing_slots"]]
    assert sent == ([[0]] if declared else [])
    assert model.freed == ([0] if declared else [])
    # The step that left ``a`` out chained for ``b`` alone.
    chained, breaks = chain_counters(runner)
    assert "layout" not in breaks
    assert any(
        c["chain_rows"].tolist() == [False, False, True, False]
        for c in model.chained_calls
    )


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "undeclared"])
def test_a_prefill_is_submitted_over_the_fused_verify_in_flight(declared):
    """``b`` joins lane 1 while ``a`` chains in lane 0. A model declaring
    ``supports_prefill_over_pending_spec`` gets the prefill behind the verify
    in flight, which commits before the next decode is built, from host
    state, and the decodes after it chain again, over both lanes. Without
    the declaration the prefill drains the verify first. Either way both
    requests emit their argmax chains."""
    capabilities = dict(LaneDeviceTarget.model_capabilities)
    if declared:
        capabilities["supports_prefill_over_pending_spec"] = True
    model = lane_device_target(model_capabilities=capabilities)()
    runner = lane_runner(model)
    engine = LaneEngine(runner)
    prompt_a = engine.add("a", 0, 1)
    engine.run(6)
    assert chain_counters(runner)[0] >= 3
    prompt_b = engine.prefill("b", 2, 9)
    chained_before = chain_counters(runner)[0]
    engine.run(10)
    engine.drain()

    assert model.unresolved_at_prefill == ([1] if declared else [0])
    controller = runner.async_decode
    assert controller._prefill_overlap_steps == (1 if declared else 0)
    # Without the declaration the prefill is what refused the chain.
    assert ("prefill" in chain_counters(runner)[1]) != declared
    assert chain_counters(runner)[0] >= chained_before + 6
    assert any(
        c["chain_rows"].tolist() == [True, False, True, False]
        for c in model.chained_calls
    )
    produced = engine.outputs["a"]
    assert produced == expected_greedy(prompt_a, len(produced))
    first = engine.first["b"]
    assert first == greedy_chain(prompt_b[-1], PROMPT_LEN - 1, 1)[0]
    produced = engine.outputs["b"]
    assert produced == expected_greedy(prompt_b + [first], len(produced))
    assert len(produced) > 10


# endregion Chained fused verifies


def test_a_decode_never_chains_onto_a_verify_a_prefill_overtook():
    """Held reads keep the overtaken verify in flight through the decode after
    the prefill: that decode drains it rather than chaining onto it."""
    capabilities = {
        **LaneDeviceTarget.model_capabilities,
        "supports_prefill_over_pending_spec": True,
    }
    model = lane_device_target(model_capabilities=capabilities)()
    runner = lane_runner(model)
    engine = LaneEngine(runner, depth=3)
    prompt_a = engine.add("a", 0, 1)
    engine.run(6)
    engine.prefill("b", 2, 9)
    pending = runner.async_decode._last_fused_step
    assert pending is runner.async_decode._prefill_overlapped_step
    assert not pending.is_resolved()
    assert runner.async_decode.chain_blocker(engine.schedule_preview()) == (
        "prefill_overlap"
    )
    engine.run(8)
    engine.drain()
    produced = engine.outputs["a"]
    assert produced == expected_greedy(prompt_a, len(produced))
