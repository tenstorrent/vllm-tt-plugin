# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculative decoding through the single-process lane-DP step.

A lane step decodes one merged slot batch: lane ``l`` owns rows
``[l * per_lane, (l + 1) * per_lane)``, a request keeps its slot for its whole
life, and the padding rows sit between the live ones rather than after them.
The candidate block of a lane verify keeps that layout, so these tests place
requests in two lanes with empty slots between them and check that every
speculative consumer -- the block builder, the draft cap, the accept walk, the
commit and the proposer -- finds each request at its own slot.

They drive ``TTModelRunner._execute_lane_step`` and ``sample_tokens`` against
the real ``TTLaneInputBatch`` and the contract stand-ins, with no device.
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
    AsyncTTSpecDecodeOutput,
    TTAsyncDecodeController,
)
from vllm_tt_plugin.input_batch import TTLaneInputBatch
from vllm_tt_plugin.lane_scheduler import TTStepPlan
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    DRAFTER_STATE_INTERNAL,
    PLACEHOLDER_TOKEN_ID,
    SpecPlan,
)

from .fake_spec_model import FAKE_VOCAB_SIZE, FakeSpecModel, make_fake_spec_model
from .sampled_sync_harness import _add_request as _add_front_packed_request
from .sampled_sync_harness import _runner as _front_packed_runner
from .sampled_sync_harness import _step as _front_packed_step
from .sampled_target import SAMPLED_VOCAB_SIZE, SampledTarget

BLOCK_SIZE = 16
MAX_MODEL_LEN = 128
NUM_LANES = 2
PER_LANE = 2
TOTAL = NUM_LANES * PER_LANE
PROMPT_LEN = 4
DRAFT_LEN = 3

# region Test helpers


def _lane_runner(
    model,
    *,
    vocab_size: int = FAKE_VOCAB_SIZE,
    async_scheduling: bool = False,
    drafts_from_model: bool = True,
) -> SimpleNamespace:
    """A lane-mode runner fake whose methods are the real ones.

    ``lane_batch`` is the real ``TTLaneInputBatch`` over two lanes of two
    slots; ``_is_lane_mode`` is not consulted by the lane step, which is
    reached through the plan the scheduler attaches.
    """
    batch = TTLaneInputBatch(
        num_lanes=NUM_LANES,
        per_lane=PER_LANE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=vocab_size,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
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
        supports_stage_inputs=False,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        _num_speculative_tokens=DRAFT_LEN,
        # What the stand-in's spec_plan publishes, read off its class.
        _spec_accept_modes=tuple(type(model).accept_modes),
        _spec_supports_narrow_decode=bool(type(model).supports_narrow_decode),
        _spec_method="custom_class" if drafts_from_model else None,
        _spec_drafts_from_model=drafts_from_model,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        _req_state_slot={},
        _ngram_proposer=None,
        async_decode_scheduling=async_scheduling,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        _steady_decode_lock=threading.Lock(),
        _pending_async_steps=deque(),
        _pending_async_overlap_ok=deque(),
        _completed_decode_steps=deque(),
        _pending_samples=deque(),
        vllm_config=SimpleNamespace(
            speculative_config=None,
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
            scheduler_config=SimpleNamespace(max_num_seqs=TOTAL),
        ),
        tt_per_lane_max_num_seqs=PER_LANE,
        tt_data_parallel_size=NUM_LANES,
        vocab_size=vocab_size,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            max_model_len=MAX_MODEL_LEN,
            logits_processors=None,
        ),
        check_perform_device_sampling=lambda **_: False,
        # A draftless lane step commits through the lane host-sampling tail,
        # so the sampler has to be the real one.
        host_sampler=Sampler(),
        _block_tables_per_layer=lambda _: None,
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
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


def _place(
    runner: SimpleNamespace,
    req_id: str,
    row: int,
    *,
    first_token: int = 1,
    params: SamplingParams | None = None,
    vocab_size: int = FAKE_VOCAB_SIZE,
) -> CachedRequestState:
    """A fully prefilled request at slot ``row``, with its own generator if seeded.

    ``first_token`` shifts the whole prompt, so two requests in one step have
    different histories and a row reading another row's state is caught.
    """
    params = params or SamplingParams(temperature=0.0)
    generator = None
    if params.sampling_type == SamplingType.RANDOM_SEED:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(params.seed)
    request = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[
            (first_token + index) % vocab_size for index in range(PROMPT_LEN)
        ],
        mm_features=None,
        sampling_params=params,
        generator=generator,
        block_ids=([0],),
        num_computed_tokens=PROMPT_LEN,
        output_token_ids=[],
    )
    runner.requests[req_id] = request
    runner.lane_batch.add_request_to_row(request, row)
    return request


def _tail(runner: SimpleNamespace, req_id: str) -> int:
    row = runner.lane_batch.req_id_to_index[req_id]
    return int(
        runner.lane_batch.token_ids_cpu[row, runner.lane_batch.num_tokens[row] - 1]
    )


def _agreed_drafts(runner: SimpleNamespace, req_id: str) -> list[int]:
    """Drafts ``FakeSpecModel`` agrees with: ``last + 1 + j``."""
    last = _tail(runner, req_id)
    return [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)]


def _lane_step(
    runner: SimpleNamespace,
    *req_ids: str,
    drafts: dict[str, list[int]] | None = None,
    finished: tuple[str, ...] = (),
):
    """One lane decode step over the scheduled requests, as the coordinator plans it.

    ``finished`` names requests this step reports finished; the plan records
    the layout change their freed rows make, as the coordinator would.
    """
    batch = runner.lane_batch
    rows = sorted(batch.req_id_to_index[req_id] for req_id in req_ids)
    ordered = [batch.req_ids[row] for row in rows]
    for row in rows:
        # Stands in for the scheduler's computed-token count advancing to
        # cover whatever the previous step committed.
        batch.num_computed_tokens_cpu[row] = batch.num_tokens[row]
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.finished_req_ids = set(finished)
    scheduler_output.scheduled_spec_decode_tokens = dict(drafts or {})
    scheduler_output.num_scheduled_tokens = dict.fromkeys(ordered, 1)
    scheduler_output.total_num_scheduled_tokens = len(ordered)
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=list(ordered),
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in ordered],
        all_token_ids={},
        new_block_ids=[None for _ in ordered],
        num_computed_tokens=[int(batch.num_tokens[row]) for row in rows],
        num_output_tokens=[
            len(runner.requests[req_id].output_token_ids) for req_id in ordered
        ],
    )
    plan = TTStepPlan(
        is_decode=True,
        capacity=TOTAL,
        scheduled_req_ids=tuple(ordered),
        scheduled_rows=tuple(rows),
        input_rows=tuple(range(TOTAL)),
        req_id_to_row=dict(batch.req_id_to_index),
        batch_size_per_dp=(PER_LANE,) * NUM_LANES,
        prefill_empty_slots=None,
        decode_layout_changed=any(r in batch.req_id_to_index for r in finished),
    )
    scheduler_output._tt_step_state = SimpleNamespace(plan=plan)
    assert runner.execute_model(scheduler_output) is None, (
        "a lane step defers its sampling to sample_tokens, so execute_model "
        "returns None"
    )
    return runner.sample_tokens(None)


def _by_request(output) -> dict[str, list[int]]:
    return {
        req_id: list(output.sampled_token_ids[index])
        for req_id, index in output.req_id_to_index.items()
    }


# endregion Test helpers

# region The slot layout


def test_a_lane_verify_keeps_each_request_at_its_slot():
    """The block is the slot batch, widened: live rows at their slots, padding between.

    Request ``a`` sits in lane 0's first slot and ``b`` in lane 1's last, so
    rows 1 and 2 are empty slots. The verify sees four rows, the two live ones
    carrying their own drafts, each commits its drafts plus the bonus into its
    own slot, and the drafter is asked over the same four rows.
    """
    model = FakeSpecModel()
    runner = _lane_runner(model)
    _place(runner, "a", 0, first_token=1)
    _place(runner, "b", 3, first_token=21)
    drafts = {"a": _agreed_drafts(runner, "a"), "b": _agreed_drafts(runner, "b")}

    output = _lane_step(runner, "a", "b", drafts=drafts)

    call = model.verify_calls[0]
    assert call["rows"] == TOTAL
    assert call["block_width"] == 1 + DRAFT_LEN
    assert call["spec_mode"] == ACCEPT_MODE_ARGMAX_IDS
    assert call["num_valid_drafts"].tolist() == [DRAFT_LEN, 0, 0, DRAFT_LEN]
    assert call["accepted_counts"].tolist() == [1, 1, 1, 1]

    committed = _by_request(output)
    assert committed["a"] == drafts["a"] + [(drafts["a"][-1] + 1) % FAKE_VOCAB_SIZE]
    assert committed["b"] == drafts["b"] + [(drafts["b"][-1] + 1) % FAKE_VOCAB_SIZE]
    batch = runner.lane_batch
    assert batch.num_tokens[:TOTAL].tolist() == [
        PROMPT_LEN + DRAFT_LEN + 1,
        0,
        0,
        PROMPT_LEN + DRAFT_LEN + 1,
    ]
    assert (
        batch.token_ids_cpu[3, PROMPT_LEN : PROMPT_LEN + DRAFT_LEN + 1].tolist()
        == (committed["b"])
    )
    assert runner.requests["b"].output_token_ids == committed["b"]
    assert runner._req_accepted_counts == {"a": DRAFT_LEN + 1, "b": DRAFT_LEN + 1}

    # The drafter continues each slot from its own commit; an empty slot is a
    # padding row, with a count of 1 and no position.
    proposal = model.propose_calls[-1]
    assert proposal["rows"] == TOTAL
    assert proposal["accepted_counts"].tolist() == [DRAFT_LEN + 1, 1, 1, DRAFT_LEN + 1]
    assert bool((proposal["committed_positions"][1:3] < 0).all())
    assert proposal["committed_positions"][3].tolist() == [
        PROMPT_LEN + j for j in range(DRAFT_LEN + 1)
    ]
    assert set(runner._proposed_draft_token_ids) == {"a", "b"}


def test_the_candidate_block_pads_the_empty_slots():
    """An empty slot is a padding row: no position, no draft, a count of 1."""
    model = FakeSpecModel()
    runner = _lane_runner(model)
    _place(runner, "a", 1, first_token=1)
    drafts = {"a": _agreed_drafts(runner, "a")}
    captured = {}
    original = model.decode_forward

    def capture(tokens, start_pos, **kwargs):
        captured["tokens"] = tokens.clone()
        captured["start_pos"] = start_pos.clone()
        return original(tokens, start_pos, **kwargs)

    model.decode_forward = capture
    _lane_step(runner, "a", drafts=drafts)

    tokens, positions = captured["tokens"], captured["start_pos"]
    assert tokens.shape == positions.shape == (TOTAL, 1 + DRAFT_LEN)
    assert tokens[1].tolist() == [_tail_before(runner, "a")] + drafts["a"]
    assert positions[1].tolist() == [PROMPT_LEN - 1 + j for j in range(1 + DRAFT_LEN)]
    for row in (0, 2, 3):
        assert positions[row].tolist() == [-1] * (1 + DRAFT_LEN)
        assert tokens[row].tolist() == [0] + [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN


def _tail_before(runner: SimpleNamespace, req_id: str) -> int:
    """The prompt's last token, which the step above continued from."""
    return runner.requests[req_id].prompt_token_ids[PROMPT_LEN - 1]


def test_a_lane_step_with_nothing_to_verify_is_the_narrow_decode_and_still_proposes():
    """No drafts anywhere: the plain lane decode runs, and the drafter is still asked.

    The narrow step commits through the lane host-sampling tail, one token per
    scheduled slot, and the proposal that follows is over the slot batch with
    each live slot's committed token at its row.
    """
    model = make_fake_spec_model(supports_narrow_decode=True)()
    runner = _lane_runner(model)
    _place(runner, "a", 0, first_token=1)
    _place(runner, "b", 3, first_token=21)

    output = _lane_step(runner, "a", "b")

    assert model.verify_calls == []
    assert [(call["rows"], call["width"]) for call in model.plain_calls] == [(TOTAL, 1)]
    committed = _by_request(output)
    # The stand-in's plain decode chooses ``last + 1``.
    assert committed == {
        "a": [(_tail_before(runner, "a") + 1) % FAKE_VOCAB_SIZE],
        "b": [(_tail_before(runner, "b") + 1) % FAKE_VOCAB_SIZE],
    }
    assert runner.lane_batch.num_tokens[:TOTAL].tolist() == [
        PROMPT_LEN + 1,
        0,
        0,
        PROMPT_LEN + 1,
    ]
    proposal = model.propose_calls[-1]
    assert proposal["rows"] == TOTAL
    assert proposal["accepted_counts"].tolist() == [1, 1, 1, 1]
    assert proposal["committed_tokens"][:, 0].tolist() == [
        committed["a"][0],
        0,
        0,
        committed["b"][0],
    ]
    assert bool((proposal["committed_positions"][1:3] < 0).all())
    assert set(runner._proposed_draft_token_ids) == {"a", "b"}
    assert runner._req_accepted_counts == {}


def test_the_draft_cap_follows_the_live_rows_of_the_whole_lane_batch():
    """``k_by_rows`` reads the folded live count, which is what one launch verifies."""
    model = FakeSpecModel()
    runner = _lane_runner(model)
    runner._spec_plan = SpecPlan(
        effective_k=DRAFT_LEN,
        lanes_per_request=1,
        extra_bytes_per_seq=0,
        extra_bytes_per_token=0,
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
        drafter_state=DRAFTER_STATE_INTERNAL,
        k_by_rows=((1, DRAFT_LEN), (TOTAL, 1)),
    )
    _place(runner, "a", 0, first_token=1)
    _place(runner, "b", 2, first_token=21)
    drafts = {"a": _agreed_drafts(runner, "a"), "b": _agreed_drafts(runner, "b")}

    # Two live rows across the two lanes: the cap is the two-row cap, 1.
    output = _lane_step(runner, "a", "b", drafts=drafts)
    assert model.verify_calls[-1]["num_valid_drafts"].tolist() == [1, 0, 1, 0]
    committed = _by_request(output)
    assert all(len(tokens) == 2 for tokens in committed.values())

    # One live row: the single-row cap, the full draft length.
    drafts = {"a": _agreed_drafts(runner, "a")}
    output = _lane_step(runner, "a", drafts=drafts)
    assert model.verify_calls[-1]["num_valid_drafts"].tolist() == [DRAFT_LEN, 0, 0, 0]
    assert len(_by_request(output)["a"]) == DRAFT_LEN + 1


def test_a_finished_lane_request_is_released_to_the_drafter():
    """The model's ``release_request`` hears a finished request's slot.

    A model-owned drafter keeps a session per state slot, and nothing else
    tells it a request is gone (contract section 4g); the lane step hands it
    the slot before the row is reused, as the front-packed step does.
    """
    model = FakeSpecModel()
    released: list[int] = []
    model.release_request = released.append
    runner = _lane_runner(model)
    _place(runner, "a", 0, first_token=1)
    _place(runner, "b", 3, first_token=21)
    drafts = {"a": _agreed_drafts(runner, "a"), "b": _agreed_drafts(runner, "b")}
    _lane_step(runner, "a", "b", drafts=drafts)
    assert released == []

    # "b" finishes and "a" decodes on: slot 3 is released, slot 0 is not.
    _lane_step(runner, "a", drafts={"a": _agreed_drafts(runner, "a")}, finished=("b",))

    assert released == [3]
    assert "b" not in runner.requests
    assert runner.lane_batch.occupied_rows() == [0]
    assert model.verify_calls[-1]["num_valid_drafts"].tolist() == [DRAFT_LEN, 0, 0, 0]
    assert "b" not in runner._req_accepted_counts


# endregion The slot layout

# region Asynchronous scheduling


def test_an_async_lane_verify_defers_through_the_speculative_path():
    """A lane verify is deferred like a front-packed one, never overlapped.

    Its commit lands at the next step's drain, into each request's own slot,
    and that step's verify carries the counts the commit recorded.
    """
    model = FakeSpecModel()
    runner = _lane_runner(model, async_scheduling=True)
    _place(runner, "a", 0, first_token=1)
    _place(runner, "b", 3, first_token=21)
    first = {"a": _agreed_drafts(runner, "a"), "b": _agreed_drafts(runner, "b")}
    # Asynchronously the scheduler reserves placeholders and the runner
    # verifies the proposals it retained.
    runner._proposed_draft_token_ids = dict(first)
    reserved = {"a": [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN, "b": [-1] * DRAFT_LEN}

    wrapper = _lane_step(runner, "a", "b", drafts=reserved)

    assert isinstance(wrapper, AsyncTTSpecDecodeOutput)
    assert list(runner._pending_async_overlap_ok) == [False]
    assert runner.lane_batch.num_tokens[:TOTAL].tolist() == [
        PROMPT_LEN,
        0,
        0,
        PROMPT_LEN,
    ]

    # The next step drains the first: the commit lands, the drafter proposes
    # again, and this verify carries those proposals and the recorded counts.
    second = _lane_step(runner, "a", "b", drafts=reserved)

    assert isinstance(second, AsyncTTSpecDecodeOutput)
    committed = _by_request(wrapper.get_output())
    assert committed["a"] == first["a"] + [(first["a"][-1] + 1) % FAKE_VOCAB_SIZE]
    assert committed["b"] == first["b"] + [(first["b"][-1] + 1) % FAKE_VOCAB_SIZE]
    assert runner.lane_batch.num_tokens[:TOTAL].tolist() == [
        PROMPT_LEN + DRAFT_LEN + 1,
        0,
        0,
        PROMPT_LEN + DRAFT_LEN + 1,
    ]
    call = model.verify_calls[1]
    assert call["accepted_counts"].tolist() == [DRAFT_LEN + 1, 1, 1, DRAFT_LEN + 1]
    assert call["num_valid_drafts"].tolist() == [DRAFT_LEN, 0, 0, DRAFT_LEN]


# endregion Asynchronous scheduling

# region The sampled walk


def test_a_lane_logits_verify_commits_what_the_front_packed_verify_commits():
    """The sampled walk reads a lane row's own logits, controls and generator.

    The same seeded request, with the same drafts, is verified once as the
    only request of a front-packed batch and once in lane 1's first slot
    behind a greedy peer and an empty slot. ``SampledTarget`` answers by a
    rule of token and position alone, so the two walks see the same logits
    rows, and the seed makes their draws the same; a walk reading the wrong
    slot's logits, controls or generator commits something else.
    """
    seeded = SamplingParams(temperature=1.0, seed=7, ignore_eos=True)
    drafts = {"s": [5, 9, 13], "g": [2, 4, 6]}

    reference = _front_packed_runner(SampledTarget())
    _add_front_packed_request(reference, "s", seeded, first_token=9)
    _add_front_packed_request(
        reference, "g", SamplingParams(temperature=0.0), first_token=1
    )
    expected = _front_packed_step(reference, "s", "g", drafts=drafts)

    model = SampledTarget()
    runner = _lane_runner(model, vocab_size=SAMPLED_VOCAB_SIZE, drafts_from_model=False)
    _place(runner, "g", 1, first_token=1, vocab_size=SAMPLED_VOCAB_SIZE)
    _place(runner, "s", 2, first_token=9, params=seeded, vocab_size=SAMPLED_VOCAB_SIZE)

    output = _lane_step(runner, "g", "s", drafts=drafts)

    assert model.verify_modes == [ACCEPT_MODE_LOGITS]
    assert _by_request(output) == expected
    # The seeded request's own generator was the one advanced.
    assert runner.requests["s"].generator.get_state().tolist() == (
        reference.requests["s"].generator.get_state().tolist()
    )


@pytest.mark.parametrize("row", [0, 3])
def test_a_lane_logits_verify_advances_only_the_scheduled_request_s_generator(row):
    """A seeded request left unscheduled keeps its stream where it was."""
    waiting = SamplingParams(temperature=1.0, seed=3, ignore_eos=True)
    model = SampledTarget()
    runner = _lane_runner(model, vocab_size=SAMPLED_VOCAB_SIZE, drafts_from_model=False)
    _place(
        runner, "s", row, first_token=9, params=waiting, vocab_size=SAMPLED_VOCAB_SIZE
    )
    _place(
        runner,
        "t",
        1 if row == 0 else 2,
        first_token=1,
        params=SamplingParams(temperature=1.0, seed=5, ignore_eos=True),
        vocab_size=SAMPLED_VOCAB_SIZE,
    )
    before = runner.requests["s"].generator.get_state().clone()

    _lane_step(runner, "t", drafts={"t": [2, 4, 6]})

    assert model.verify_modes == [ACCEPT_MODE_LOGITS]
    assert torch.equal(runner.requests["s"].generator.get_state(), before)
    assert not torch.equal(
        runner.requests["t"].generator.get_state(),
        torch.Generator(device="cpu").manual_seed(5).get_state(),
    )


# endregion The sampled walk
