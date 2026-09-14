# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for the speculative decode input build.

``TTModelRunner._prepare_model_inputs`` builds the per-step model input. A
plain decode step sends one token per row. Once a launch carries a resolved
``SpecPlan``, the same builder sends the candidate block the contract
specifies: ``[B, 1+K]`` tokens and positions, column 0 being the row's last
committed token and columns 1..K its pending drafts, plus two ``[B]`` side
tensors, ``num_valid_drafts`` saying how many of a row's draft columns are
real and ``accepted_counts`` saying how many tokens that row's previous step
committed.

The per-row speculative state lives on the persistent ``InputBatch``, so these
tests cover both halves: what the builder assembles out of that state, and
whether the state follows a request when ``InputBatch.condense`` moves its row
or ``InputBatch.add_request`` reuses its slot.

The contract is specified in
https://github.com/tenstorrent/vllm-tt-plugin/issues/110.
"""

from types import SimpleNamespace

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, PLACEHOLDER_TOKEN_ID

from .fake_spec_model import FakeSpecModel

VOCAB_SIZE = 512
BLOCK_SIZE = 16
MAX_MODEL_LEN = 32
MAX_NUM_REQS = 4
PROMPT_LEN = 4
OUTPUT_LEN = 2
# The row's last committed token id and its position, which column 0 of the
# candidate block carries. ``_add_decoding_request`` numbers a request's tokens
# 0..num_tokens-1, so the two coincide.
LAST_TOKEN = LAST_POSITION = PROMPT_LEN + OUTPUT_LEN - 1

# region Test helpers


def _batch(num_speculative_tokens: int) -> InputBatch:
    return InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
        num_speculative_tokens=num_speculative_tokens,
    )


def _request(req_id: str, num_computed_tokens: int) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(PROMPT_LEN)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=num_computed_tokens,
        output_token_ids=list(range(PROMPT_LEN, PROMPT_LEN + OUTPUT_LEN)),
    )


def _add_decoding_request(batch: InputBatch, req_id: str) -> CachedRequestState:
    """Add a request whose whole history is computed, so it decodes next."""
    request = _request(req_id, num_computed_tokens=PROMPT_LEN + OUTPUT_LEN - 1)
    batch.add_request(request)
    return request


def _fake_runner(
    batch: InputBatch,
    requests: dict[str, CachedRequestState],
    supports_narrow_decode: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_batch=batch,
        requests=requests,
        _output_tokens_per_step=1,
        _spec_supports_narrow_decode=supports_narrow_decode,
        _spec_candidate_block=TTModelRunner._spec_candidate_block,
        tt_per_lane_max_num_seqs=MAX_NUM_REQS,
        tt_data_parallel_size=1,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        model_config=SimpleNamespace(is_multimodal_model=False),
        check_perform_device_sampling=lambda **_: False,
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        _sampling_params_for_padded_decode=lambda params, req_indices, n: params,
        _decode_layout_changed_since_last_decode=False,
        _build_host_generators=TTModelRunner._build_host_generators,
    )


def _prepare(runner: SimpleNamespace, *rows: tuple[str, int, int]):
    """Run ``_prepare_model_inputs`` for ``(req_id, num_scheduled, num_computed)``."""
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {r: s for r, s, _ in rows}
    scheduler_output.total_num_scheduled_tokens = sum(s for _, s, _ in rows)
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=[r for r, *_ in rows],
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in rows],
        all_token_ids={},
        new_block_ids=[None for _ in rows],
        num_computed_tokens=[c for _, _, c in rows],
        num_output_tokens=[OUTPUT_LEN for _ in rows],
    )
    return TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)


def _decode(runner: SimpleNamespace, *req_ids: str):
    return _prepare(runner, *((r, 1, PROMPT_LEN + OUTPUT_LEN - 1) for r in req_ids))


# endregion Test helpers

# region The candidate block


def test_decode_block_is_uniformly_wide_without_drafts():
    """A speculating launch sends 1+K columns even when no row has drafts.

    The contract makes the verify width uniform so a model needs one decode
    shape rather than two. With no drafts pending, every column past 0 is a
    padding column: ``PLACEHOLDER_TOKEN_ID`` for the token and -1 for the
    position, which is the same no-position marker a padded row carries.
    """
    batch = _batch(num_speculative_tokens=3)
    request = _add_decoding_request(batch, "r")
    model_input = _decode(_fake_runner(batch, {"r": request}), "r")

    assert model_input.input_tokens.shape == (MAX_NUM_REQS, 4)
    assert model_input.input_positions.shape == (MAX_NUM_REQS, 4)
    assert model_input.input_tokens[0].tolist() == [LAST_TOKEN, -1, -1, -1]
    assert model_input.input_positions[0].tolist() == [LAST_POSITION, -1, -1, -1]
    assert model_input.num_valid_drafts.tolist() == [0] * MAX_NUM_REQS
    assert model_input.accepted_counts.tolist() == [1] * MAX_NUM_REQS


def test_decode_block_carries_the_pending_drafts():
    """Columns 1..K are the row's drafts at the K positions that follow."""
    batch = _batch(num_speculative_tokens=3)
    request = _add_decoding_request(batch, "r")
    batch.draft_token_ids[0] = [11, 12, 13]
    batch.num_valid_drafts[0] = 3

    model_input = _decode(_fake_runner(batch, {"r": request}), "r")

    assert model_input.input_tokens[0].tolist() == [LAST_TOKEN, 11, 12, 13]
    assert model_input.input_positions[0].tolist() == [
        LAST_POSITION,
        LAST_POSITION + 1,
        LAST_POSITION + 2,
        LAST_POSITION + 3,
    ]
    assert model_input.num_valid_drafts[0] == 3


def test_each_row_pads_at_its_own_draft_count():
    """A row short of K drafts pads only its own tail columns.

    The cap is per row and is never reduced across the batch: one request that
    carries fewer drafts, because its drafter ran out of history or a grammar
    truncated it, must not shorten any other request's speculation.
    """
    batch = _batch(num_speculative_tokens=3)
    requests = {req_id: _add_decoding_request(batch, req_id) for req_id in ("a", "b")}
    batch.draft_token_ids[0] = [11, 12, 13]
    batch.num_valid_drafts[0] = 3
    batch.draft_token_ids[1] = [21, 22, 23]
    batch.num_valid_drafts[1] = 1

    model_input = _decode(_fake_runner(batch, requests), "a", "b")

    assert model_input.input_tokens[0].tolist() == [LAST_TOKEN, 11, 12, 13]
    assert model_input.input_tokens[1].tolist() == [
        LAST_TOKEN,
        21,
        PLACEHOLDER_TOKEN_ID,
        PLACEHOLDER_TOKEN_ID,
    ]
    assert model_input.input_positions[1].tolist() == [
        LAST_POSITION,
        LAST_POSITION + 1,
        -1,
        -1,
    ]
    assert model_input.num_valid_drafts[:2].tolist() == [3, 1]


def test_padding_rows_carry_a_valid_accepted_count():
    """Every row of the padded batch carries a count in [1, 1+K], never 0.

    A count of 0 is outside the contract's domain, and a model that reads
    ``accepted_counts - 1`` to select a candidate state slot would index -1.
    Rows past the active requests are padding, and they stand on their own
    input token, which is the count 1.
    """
    batch = _batch(num_speculative_tokens=3)
    request = _add_decoding_request(batch, "r")
    batch.accepted_counts[0] = 4

    model_input = _decode(_fake_runner(batch, {"r": request}), "r")

    assert model_input.accepted_counts.tolist() == [4, 1, 1, 1]
    assert int(model_input.accepted_counts.min()) >= 1
    assert int(model_input.accepted_counts.max()) <= 1 + 3
    assert model_input.input_tokens[1].tolist() == [0, 0, 0, 0]
    assert model_input.input_positions[1].tolist() == [-1, -1, -1, -1]


def test_side_tensors_are_int32():
    """The contract's ``[B]`` side tensors are int32, which a model checks."""
    batch = _batch(num_speculative_tokens=3)
    request = _add_decoding_request(batch, "r")
    model_input = _decode(_fake_runner(batch, {"r": request}), "r")

    assert model_input.num_valid_drafts.dtype == torch.int32
    assert model_input.accepted_counts.dtype == torch.int32


# endregion The candidate block

# region Narrow decode


def test_narrow_decode_is_kept_when_no_row_carries_a_draft():
    """A model that also serves ``[B, 1]`` keeps it on a draftless step."""
    batch = _batch(num_speculative_tokens=3)
    request = _add_decoding_request(batch, "r")
    runner = _fake_runner(batch, {"r": request}, supports_narrow_decode=True)

    model_input = _decode(runner, "r")

    assert model_input.input_tokens.shape == (MAX_NUM_REQS, 1)
    # Still sent: the model needs the count to pick the candidate state slot
    # its previous step committed from, whatever this step's width.
    assert model_input.accepted_counts.tolist() == [1] * MAX_NUM_REQS
    assert model_input.num_valid_drafts.tolist() == [0] * MAX_NUM_REQS


def test_narrow_decode_widens_when_any_row_carries_a_draft():
    batch = _batch(num_speculative_tokens=3)
    requests = {req_id: _add_decoding_request(batch, req_id) for req_id in ("a", "b")}
    batch.draft_token_ids[1] = [21, 22, 23]
    batch.num_valid_drafts[1] = 2
    runner = _fake_runner(batch, requests, supports_narrow_decode=True)

    model_input = _decode(runner, "a", "b")

    assert model_input.input_tokens.shape == (MAX_NUM_REQS, 4)
    assert model_input.input_tokens[1].tolist() == [
        LAST_TOKEN,
        21,
        22,
        PLACEHOLDER_TOKEN_ID,
    ]


# endregion Narrow decode

# region The non-speculative path


def test_a_non_speculating_decode_is_unchanged():
    """Without a plan the build is what it was: ``[B, 1]`` and no side tensors.

    ``num_speculative_tokens`` is 0 for every model shipping today, so this
    pins that the widening cannot reach them.
    """
    batch = _batch(num_speculative_tokens=0)
    request = _add_decoding_request(batch, "r")

    model_input = _decode(_fake_runner(batch, {"r": request}), "r")

    assert model_input.input_tokens.shape == (MAX_NUM_REQS, 1)
    assert model_input.input_positions.shape == (MAX_NUM_REQS,)
    assert model_input.input_tokens[0].tolist() == [LAST_TOKEN]
    assert model_input.input_positions[0] == LAST_POSITION
    assert model_input.num_valid_drafts is None
    assert model_input.accepted_counts is None


def test_a_prefill_drops_stale_speculative_state():
    """A prefilling row's pending drafts are dropped and its count reset to 1.

    A request resumed from preemption replays its own history, so drafts
    proposed against the pre-preemption state no longer sit at the positions
    they were drafted for. A prefill build carries no candidate block, so the
    side tensors are absent from it.
    """
    batch = _batch(num_speculative_tokens=3)
    request = _request("r", num_computed_tokens=0)
    batch.add_request(request)
    batch.draft_token_ids[0] = [11, 12, 13]
    batch.num_valid_drafts[0] = 3
    batch.accepted_counts[0] = 3

    model_input = _prepare(_fake_runner(batch, {"r": request}), ("r", PROMPT_LEN, 0))

    assert model_input.prompt_lens.tolist() == [PROMPT_LEN]
    assert model_input.num_valid_drafts is None
    assert model_input.accepted_counts is None
    assert batch.num_valid_drafts[0] == 0
    assert batch.accepted_counts[0] == 1
    assert batch.draft_token_ids[0].tolist() == [PLACEHOLDER_TOKEN_ID] * 3


# endregion The non-speculative path

# region Row bookkeeping


def test_condense_moves_speculative_state_with_the_request():
    """``condense`` moves a surviving request's state down to its new row.

    Leaving it behind would hand the moved request the accepted count and the
    drafts of whichever request previously sat at the row it moves into.
    """
    batch = _batch(num_speculative_tokens=3)
    for req_id in ("a", "b"):
        _add_decoding_request(batch, req_id)
    batch.draft_token_ids[1] = [21, 22, 23]
    batch.num_valid_drafts[1] = 2
    batch.accepted_counts[1] = 3

    removed = batch.remove_request("a")
    batch.condense([removed])

    assert batch.req_id_to_index == {"b": 0}
    assert batch.draft_token_ids[0].tolist() == [21, 22, 23]
    assert batch.num_valid_drafts[0] == 2
    assert batch.accepted_counts[0] == 3


def test_a_reused_slot_does_not_inherit_speculative_state():
    """``add_request`` resets the slot it lands in.

    A row freed by a finished request keeps its arrays until something
    overwrites them, and the next request to take that row would otherwise
    verify tokens drafted for another sequence.
    """
    batch = _batch(num_speculative_tokens=3)
    _add_decoding_request(batch, "a")
    batch.draft_token_ids[0] = [11, 12, 13]
    batch.num_valid_drafts[0] = 3
    batch.accepted_counts[0] = 4

    batch.remove_request("a")
    _add_decoding_request(batch, "b")

    assert batch.req_id_to_index == {"b": 0}
    assert batch.draft_token_ids[0].tolist() == [PLACEHOLDER_TOKEN_ID] * 3
    assert batch.num_valid_drafts[0] == 0
    assert batch.accepted_counts[0] == 1


def test_built_rows_follow_the_request_across_a_condense():
    """The built block's row order is the post-condense order.

    ``_prepare_model_inputs`` reads the persistent batch by row, so the move
    ``condense`` performs is what puts a request's drafts and its accepted
    count on the same row as its own last committed token.
    """
    batch = _batch(num_speculative_tokens=3)
    requests = {req_id: _add_decoding_request(batch, req_id) for req_id in ("a", "b")}
    batch.draft_token_ids[1] = [21, 22, 23]
    batch.num_valid_drafts[1] = 3
    batch.accepted_counts[1] = 2

    removed = batch.remove_request("a")
    batch.condense([removed])
    del requests["a"]

    model_input = _decode(_fake_runner(batch, requests), "b")

    assert model_input.row_req_ids == ["b"]
    assert model_input.input_tokens[0].tolist() == [LAST_TOKEN, 21, 22, 23]
    assert model_input.accepted_counts[0] == 2


# endregion Row bookkeeping

# region Conformance


def test_the_contract_model_accepts_the_built_block():
    """A contract-conformant model verifies what the builder produced.

    ``FakeSpecModel.decode_forward`` raises on every input the contract
    forbids: a non-uniform width, a ``num_valid_drafts`` outside [0, K], an
    ``accepted_counts`` outside [1, 1+K] or of the wrong dtype. Feeding it the
    built tensors, padding rows included, is what makes the build conformant
    rather than merely self-consistent.
    """
    batch = _batch(num_speculative_tokens=3)
    requests = {req_id: _add_decoding_request(batch, req_id) for req_id in ("a", "b")}
    batch.draft_token_ids[0] = [11, 12, 13]
    batch.num_valid_drafts[0] = 3
    batch.accepted_counts[0] = 2
    batch.draft_token_ids[1] = [21, 22, 23]
    batch.num_valid_drafts[1] = 1

    model_input = _decode(_fake_runner(batch, requests), "a", "b")

    model = FakeSpecModel()
    verify = model.decode_forward(
        tokens=model_input.input_tokens,
        positions=model_input.input_positions,
        num_valid_drafts=model_input.num_valid_drafts,
        accepted_counts=model_input.accepted_counts,
        spec_mode=ACCEPT_MODE_ARGMAX_IDS,
    )

    assert verify.argmax_ids.shape == (MAX_NUM_REQS, 4)
    # Column 0 is already committed, so a verify repeats it.
    assert verify.argmax_ids[0, 0] == LAST_TOKEN
    # Row 0 drafted 3 and the stand-in agrees with all of them; row 1 drafted
    # 1, so its second draft column is a padding column and diverges.
    assert verify.argmax_ids[0, 1:].tolist() == [11, 12, 13]
    assert verify.argmax_ids[1, 1] == 21


# endregion Conformance
