# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Exercise page-table padding through both model-input assembly paths."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

from vllm_tt_plugin.input_batch import (
    InputBatch,
    TTLaneInputBatch,
    build_cached_request_state,
)
from vllm_tt_plugin.lane_scheduler import TTStepPlan
from vllm_tt_plugin.model_runner import TTModelRunner

BLOCK_SIZE = 64
MAX_MODEL_LEN = 1024
TABLE_WIDTH = MAX_MODEL_LEN // BLOCK_SIZE
PROMPT_LEN = 190


def _allocated_request(phase: str):
    """Allocate the real three-block prefix for the requested scheduler phase."""
    manager = KVCacheManager(
        kv_cache_config=KVCacheConfig(
            num_blocks=32,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["layer"],
                    kv_cache_spec=FullAttentionSpec(
                        block_size=BLOCK_SIZE,
                        num_kv_heads=1,
                        head_size=16,
                        dtype=torch.float32,
                    ),
                )
            ],
        ),
        max_model_len=MAX_MODEL_LEN,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )
    request = Request(
        request_id="request",
        prompt_token_ids=[1] * PROMPT_LEN,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=100),
        pooling_params=None,
    )
    first_chunk = PROMPT_LEN - 1 if phase == "last_prompt_token" else PROMPT_LEN
    assert manager.allocate_slots(request, num_new_tokens=first_chunk) is not None
    if phase == "last_prompt_token":
        request.num_computed_tokens = first_chunk
        assert manager.allocate_slots(request, num_new_tokens=1) is not None
    elif phase == "decode":
        request.num_computed_tokens = PROMPT_LEN
        request.append_output_token_ids(2)
        assert manager.allocate_slots(request, num_new_tokens=1) is not None

    block_ids = manager.get_block_ids(request.request_id)
    assert len(block_ids[0]) == 3
    new_data = NewRequestData.from_request(request, block_ids)
    state = build_cached_request_state(new_data)
    state.output_token_ids.extend(request.output_token_ids)
    return state, new_data


def _scheduler_output(state, new_data, phase: str) -> SchedulerOutput:
    out = SchedulerOutput.make_empty()
    num_scheduled = PROMPT_LEN if phase == "prefill" else 1
    out.num_scheduled_tokens = {state.req_id: num_scheduled}
    out.total_num_scheduled_tokens = num_scheduled
    if phase == "prefill":
        out.scheduled_new_reqs = [new_data]
    else:
        out.scheduled_cached_reqs = CachedRequestData(
            req_ids=[state.req_id],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[state.num_computed_tokens],
            num_output_tokens=[len(state.output_token_ids)],
        )
    return out


def _runner(batch, state):
    # The input builder is real. Only unrelated model, sampling, and ownership
    # hooks are supplied, as in the existing host model-runner tests.
    runner = SimpleNamespace(
        input_batch=batch,
        requests={state.req_id: state},
        _output_tokens_per_step=1,
        tt_per_lane_max_num_seqs=batch.max_num_reqs,
        tt_data_parallel_size=1,
        max_num_blocks_per_req=TABLE_WIDTH,
        model_config=SimpleNamespace(is_multimodal_model=False),
        check_perform_device_sampling=lambda **_: False,
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        _decode_layout_changed_since_last_decode=False,
        _build_host_generators=TTModelRunner._build_host_generators,
    )
    runner._sampling_params_for_padded_decode = MethodType(
        TTModelRunner._sampling_params_for_padded_decode, runner
    )
    return runner


@pytest.mark.parametrize("path", ["ordinary", "lane"])
@pytest.mark.parametrize("phase", ["prefill", "last_prompt_token", "decode"])
def test_input_builder_keeps_unused_pages_readable_and_outside_live_kv(path, phase):
    state, new_data = _allocated_request(phase)
    kwargs = dict(
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * 4,
        vocab_size=16,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    if path == "ordinary":
        batch = InputBatch(max_num_reqs=2, **kwargs)
        row = 0
        batch.add_request(state)
    else:
        batch = TTLaneInputBatch(num_lanes=2, per_lane=2, **kwargs)
        # A request on lane 1 exercises selection of a non-leading row in
        # prefill and its stable position among empty slots during decode.
        row = 2
        batch.add_request_to_row(state, row)
    assert batch.block_table.block_tables[0].num_blocks_per_row[row] == 3
    runner = _runner(batch, state)
    scheduler_output = _scheduler_output(state, new_data, phase)

    if path == "ordinary":
        model_input = TTModelRunner._prepare_model_inputs(
            runner, scheduler_output, None
        )
    else:
        is_decode = phase == "decode"
        plan = TTStepPlan(
            is_decode=is_decode,
            capacity=batch.max_num_reqs,
            scheduled_req_ids=(state.req_id,),
            scheduled_rows=(row,),
            input_rows=tuple(range(batch.max_num_reqs)) if is_decode else (row,),
            req_id_to_row={state.req_id: row},
            batch_size_per_dp=(0, 1),
            prefill_empty_slots=None if is_decode else (row,),
        )
        model_input = batch.build_model_input(runner, scheduler_output, None, plan)

    table = model_input.block_tables
    assert model_input.block_tables_per_group[0] is table
    active_output_row = row if phase == "decode" else 0
    assert table[active_output_row].tolist() == state.block_ids[0] + [0] * (
        TABLE_WIDTH - 3
    )

    if phase == "decode":
        assert model_input.prompt_lens is None
        assert model_input.input_positions[active_output_row] == PROMPT_LEN
        # At position 190, 128-token SDPA chunks read through position 255.
        # The fourth 64-token block is unallocated but must be readable: the
        # cache-fill skip sentinel (-1) is not supported by the decode reader.
        assert table[active_output_row, 3] == 0
        gap_rows = [i for i in range(batch.max_num_reqs) if i != active_output_row]
        assert torch.count_nonzero(table[gap_rows]) == 0
        assert (model_input.input_positions[gap_rows] == -1).all()
    else:
        assert model_input.prompt_lens.tolist() == [PROMPT_LEN]
        assert table.shape == (1, TABLE_WIDTH)
        if phase == "last_prompt_token":
            assert model_input.input_positions.tolist() == [PROMPT_LEN - 1]
