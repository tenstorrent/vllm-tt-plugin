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
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
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
from vllm_tt_plugin.scheduler import TTScheduler, get_tt_verified_drafts
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    PLACEHOLDER_TOKEN_ID,
)

from .lane_device_target import LaneDeviceTarget, greedy_chain, lane_device_target
from .sampled_target import SAMPLED_VOCAB_SIZE

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
        # An ordinary lane decode samples on the device; a verify that needs
        # logits or accepts on the device turns it off in the builder.
        check_perform_device_sampling=lambda **_: True,
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
        self.live: dict[str, int] = {}
        self.outputs: dict[str, list[int]] = {}
        self.stop_after: dict[str, int] = {}
        self.finished_pending: set[str] = set()
        self.done: set[str] = set()
        self.computed: dict[str, int] = {}
        self.inflight: deque = deque()
        self.wrappers: list = []

    def add(self, req_id, row, first_token, params=GREEDY, stop_after=None):
        tokens = add_request(self.runner, req_id, row, first_token, params)
        self.live[req_id] = row
        self.outputs[req_id] = []
        self.computed[req_id] = len(tokens) - 1
        if stop_after is not None:
            self.stop_after[req_id] = stop_after
        return tokens

    def schedule(self) -> SchedulerOutput:
        finished, self.finished_pending = self.finished_pending, set()
        for req_id in finished:
            self.live.pop(req_id, None)
        live = sorted(self.live, key=self.live.get)
        so = SchedulerOutput.make_empty()
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
        return so

    def iterate(self) -> None:
        so = self.schedule()
        assert self.runner.execute_model(so) is None
        wrapper = self.runner.sample_tokens(None)
        self.wrappers.append(wrapper)
        self.inflight.append((wrapper, so))
        if len(self.inflight) >= self.depth:
            self.resolve_oldest()

    def resolve_oldest(self) -> None:
        wrapper, so = self.inflight.popleft()
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
            self.computed[req_id] -= so.num_scheduled_tokens[req_id] - len(tokens)
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
