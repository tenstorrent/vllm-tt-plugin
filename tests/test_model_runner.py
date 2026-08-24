# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for ``TTModelRunner`` (non-lane path)."""

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner

# region Constants
VOCAB_SIZE = 64
BLOCK_SIZE = 16
MAX_MODEL_LEN = 32
MAX_NUM_SEQS = MAX_NUM_REQS = 4
DP_SIZE = 1
SAMPLED_TOKEN_ID = 42
# endregion Constants

# region Test helpers


def _batch_with_one_request(
    prompt_len: int,
    output_len: int,
    num_computed_tokens: int,
) -> tuple[InputBatch, CachedRequestState]:
    """Creates a batch with a single request, for testing purposes."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    request = CachedRequestState(
        req_id="r",
        prompt_token_ids=list(range(prompt_len)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=num_computed_tokens,
        output_token_ids=list(range(prompt_len, prompt_len + output_len)),
    )
    batch.add_request(request)
    return batch, request


def _fake_runner(batch: InputBatch, request: CachedRequestState) -> SimpleNamespace:
    """Creates a fake runner with the given batch and request, for testing purposes."""
    return SimpleNamespace(
        input_batch=batch,
        requests={"r": request},
        _output_tokens_per_step=1,
        tt_per_lane_max_num_seqs=MAX_NUM_SEQS,
        tt_data_parallel_size=DP_SIZE,
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


# endregion Test helpers

# region Prefill classification


@pytest.mark.parametrize(
    "num_computed_tokens, intermediate_prefill_mask",
    [(0, True), (2, True), (5, True), (6, False)],
)
def test_cached_chunked_prefill_classification(
    num_computed_tokens: int,
    intermediate_prefill_mask: bool,
):
    """Classify a cached chunked-prefill continuation and its boundary."""
    batch, request = _batch_with_one_request(
        prompt_len=8,
        output_len=0,
        num_computed_tokens=num_computed_tokens,
    )
    runner = _fake_runner(batch, request)
    num_scheduled_tokens = 2

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": num_scheduled_tokens}
    scheduler_output.total_num_scheduled_tokens = num_scheduled_tokens
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[num_computed_tokens],
        num_output_tokens=[0],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    assert model_input.prompt_lens.tolist() == [
        num_computed_tokens + num_scheduled_tokens
    ]
    assert model_input.intermediate_prefill_mask.tolist() == [intermediate_prefill_mask]


def test_resumed_replay_past_prompt_length_remains_prefill():
    """Resumed replay past the prompt length remains prefill."""
    # 4-token prompt, 6 previously generated tokens (num_tokens=10). Resumed
    # from preemption and already replayed its first chunk up to position 5,
    # which is past the prompt but short of num_tokens: still mid-replay.
    num_computed_tokens = 5
    batch, request = _batch_with_one_request(
        prompt_len=4,
        output_len=6,
        num_computed_tokens=num_computed_tokens,
    )
    runner = _fake_runner(batch, request)
    num_scheduled_tokens = 3

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": num_scheduled_tokens}
    scheduler_output.total_num_scheduled_tokens = num_scheduled_tokens

    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[num_computed_tokens],
        num_output_tokens=[6],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    # A decode step would set `prompt_lens=None` and sample a token.
    # This step must still be a prefill chunk (position 5..8 of a 10-token replay).
    assert model_input.prompt_lens is not None
    assert model_input.prompt_lens.tolist() == [
        num_computed_tokens + num_scheduled_tokens
    ]
    assert model_input.intermediate_prefill_mask.tolist() == [True]


# endregion Prefill classification

# region Decode input construction


def test_completed_cached_request_builds_decode_input():
    prompt_len = num_computed_tokens = 8
    output_len = 0
    batch, request = _batch_with_one_request(
        prompt_len=prompt_len,
        output_len=output_len,
        num_computed_tokens=num_computed_tokens,
    )
    runner = _fake_runner(batch, request)
    num_scheduled_tokens = 1

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": num_scheduled_tokens}
    scheduler_output.total_num_scheduled_tokens = num_scheduled_tokens
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[num_computed_tokens],
        num_output_tokens=[output_len],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    assert model_input.prompt_lens is None
    assert model_input.input_positions.tolist() == [prompt_len - 1] + [-1] * (
        MAX_NUM_SEQS - 1
    )
    assert model_input.input_tokens.shape == (MAX_NUM_SEQS, 1)


def test_final_one_token_prompt_chunk_stays_prefill():
    """A chunked prefill whose remainder is exactly one token is still prompt
    work: classifying it as decode routes it into the mixed-batch filter,
    which drops the scheduled token (EngineCore assert / permanent hang)."""
    prompt_len = 8
    batch, request = _batch_with_one_request(
        prompt_len=prompt_len,
        output_len=0,
        num_computed_tokens=prompt_len - 1,
    )
    runner = _fake_runner(batch, request)

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": 1}
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[prompt_len - 1],
        num_output_tokens=[0],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    assert model_input.prompt_lens.tolist() == [prompt_len]
    assert model_input.intermediate_prefill_mask.tolist() == [False]


def test_mixed_prefill_batch_keeps_scheduled_decode_row():
    """A cached request with an uncomputed token must never be dropped from a
    prefill step it was scheduled into: the forward's output rows must match
    the input batch, and the scheduler never re-schedules a dropped token."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    chunking = CachedRequestState(
        req_id="r1",
        prompt_token_ids=list(range(8)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=2,
        output_token_ids=[],
    )
    steady = CachedRequestState(
        req_id="r2",
        prompt_token_ids=list(range(8)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([1],),
        num_computed_tokens=8,
        output_token_ids=[100],
    )
    batch.add_request(chunking)
    batch.add_request(steady)
    runner = _fake_runner(batch, chunking)

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r1": 2, "r2": 1}
    scheduler_output.total_num_scheduled_tokens = 3
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r1", "r2"],
        resumed_req_ids=set(),
        new_token_ids=[[], []],
        all_token_ids={},
        new_block_ids=[None, None],
        num_computed_tokens=[2, 8],
        num_output_tokens=[0, 1],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    # r1's chunk end is 2+2=4 (mid-prompt); r2's single uncomputed token ends
    # at 8+1=9 -- both rows stay in the prefill batch.
    assert sorted(model_input.prompt_lens.tolist()) == [4, 9]


def test_steady_block_step_builds_decode_input():
    """One whole canvas outstanding is the steady state for a block model.
    Dispatching it as prompt work re-encodes the entire session per canvas
    (quadratic prefill) and the decode path never runs."""
    prompt_len = 8
    canvas = 16
    batch, request = _batch_with_one_request(
        prompt_len=prompt_len,
        output_len=canvas,
        num_computed_tokens=prompt_len,
    )
    runner = _fake_runner(batch, request)
    runner._output_tokens_per_step = canvas

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": canvas}
    scheduler_output.total_num_scheduled_tokens = canvas
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[prompt_len],
        num_output_tokens=[canvas],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    assert model_input.prompt_lens is None


def test_block_resume_replay_past_one_canvas_remains_prefill():
    """More than one canvas outstanding means uncomputed history (preemption
    replay): that chunk must stay prompt work."""
    prompt_len = 8
    canvas = 8
    batch, request = _batch_with_one_request(
        prompt_len=prompt_len,
        output_len=2 * canvas,
        num_computed_tokens=prompt_len,
    )
    runner = _fake_runner(batch, request)
    runner._output_tokens_per_step = canvas

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": canvas}
    scheduler_output.total_num_scheduled_tokens = canvas
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[prompt_len],
        num_output_tokens=[2 * canvas],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    assert model_input.prompt_lens.tolist() == [prompt_len + canvas]


# endregion Decode input construction

# region Output state


def test_apply_sampled_token_updates_request_state():
    try:
        import torch
    except ImportError:
        pytest.xfail("torch is required for sampled-token tensor inputs")

    prompt_len = 8
    batch, request = _batch_with_one_request(
        prompt_len=prompt_len,
        output_len=0,
        num_computed_tokens=prompt_len,
    )
    runner = _fake_runner(batch, request)
    runner.model_config.max_model_len = MAX_MODEL_LEN
    runner._apply_sampled_tokens_to_state = lambda **kwargs: (
        TTModelRunner._apply_sampled_tokens_to_state(runner, **kwargs)
    )
    runner._build_runner_output = lambda **kwargs: TTModelRunner._build_runner_output(
        runner, **kwargs
    )

    output = TTModelRunner.apply_and_build_runner_output(
        runner,
        sampled_token_ids=torch.tensor([[SAMPLED_TOKEN_ID]], dtype=torch.int32),
    )

    assert batch.num_tokens[0] == prompt_len + 1
    assert batch.token_ids_cpu[0, prompt_len] == SAMPLED_TOKEN_ID
    assert request.output_token_ids == [SAMPLED_TOKEN_ID]
    assert output.req_ids == ["r"]
    assert output.sampled_token_ids == [[SAMPLED_TOKEN_ID]]


# endregion Output state
