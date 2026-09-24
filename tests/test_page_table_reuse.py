# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host regression for padding writes through reused TT block-table rows.

All block IDs come from the real vLLM KVCacheManager, including decode-time
growth and recycling. No allocator, free-list, or block-table methods are
mocked. Only sampled token values are supplied by the test: their values do
not affect block allocation with prefix caching disabled.

This exercises the scheduler/worker metadata lifecycle, not model execution
or the TT cache-write kernel. Run in the plugin's pinned vLLM environment.
"""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_tt_plugin.input_batch import (
    InputBatch,
    apply_cached_req_state_update,
    build_cached_request_state,
)

BLOCK_SIZE = 64
MAX_MODEL_LEN = 1024
PADDED_PREFILL_BLOCKS = MAX_MODEL_LEN // BLOCK_SIZE


def _manager_and_batch(max_num_reqs: int) -> tuple[KVCacheManager, InputBatch]:
    manager = KVCacheManager(
        kv_cache_config=KVCacheConfig(
            num_blocks=64,
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
    batch = InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * max_num_reqs,
        vocab_size=16,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    return manager, batch


def _admit(
    manager: KVCacheManager,
    batch: InputBatch,
    req_id: str,
    prompt_len: int,
    row: int | None = None,
) -> tuple[Request, CachedRequestState]:
    request = Request(
        request_id=req_id,
        prompt_token_ids=[1] * prompt_len,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=100),
        pooling_params=None,
    )
    allocated = manager.allocate_slots(request, num_new_tokens=prompt_len)
    assert allocated is not None
    request.status = RequestStatus.RUNNING
    state = build_cached_request_state(
        NewRequestData.from_request(request, manager.get_block_ids(req_id))
    )
    batch.add_request(state, row)
    batch.refresh_logitsprocs()
    return request, state


def _generate_100_tokens(
    manager: KVCacheManager,
    batch: InputBatch,
    request: Request,
    state: CachedRequestState,
) -> None:
    """Complete prefill and 99 decode steps, yielding 100 sampled tokens.

    Just as in serving, the last sampled token is not itself processed: a
    100-token prompt plus 100 outputs has 199 computed tokens and four blocks.
    """
    row = batch.req_id_to_index[request.request_id]
    for output_index in range(100):
        if output_index == 0:
            num_scheduled_tokens = request.num_prompt_tokens
        else:
            num_scheduled_tokens = 1
            allocated = manager.allocate_slots(request, num_new_tokens=1)
            assert allocated is not None
            new_block_ids = allocated.get_block_ids()
            # These are the real update calls used by TTModelRunner for a
            # scheduled cached request; no block IDs are constructed here.
            apply_cached_req_state_update(
                state, request.num_computed_tokens, new_block_ids, False
            )
            batch.block_table.append_row(new_block_ids, row)
            batch.num_computed_tokens_cpu[row] = request.num_computed_tokens

        request.num_computed_tokens += num_scheduled_tokens
        request.append_output_token_ids(2)
        state.output_token_ids.append(2)
        batch.token_ids_cpu[row, batch.num_tokens[row]] = 2
        batch.num_tokens[row] += 1

    assert request.num_tokens == 200
    assert request.num_computed_tokens == 199
    assert len(manager.get_block_ids(request.request_id)[0]) == 4


def _assert_safe_export(
    manager: KVCacheManager,
    batch: InputBatch,
    req_ids: list[str],
    width: int,
) -> None:
    rows = torch.tensor([batch.req_id_to_index[rid] for rid in req_ids])
    table = batch.block_tables_for_rows(rows, width=width)[0]
    all_live_ids = {
        block_id for rid in batch.req_ids for block_id in manager.get_block_ids(rid)[0]
    }
    assert table.shape == (len(req_ids), width)
    for exported_row, req_id in zip(table, req_ids):
        expected = manager.get_block_ids(req_id)[0]
        values = exported_row.tolist()
        padding_aliases = [
            (column, block_id)
            for column, block_id in enumerate(values)
            if column >= len(expected) and block_id in all_live_ids
        ]
        diagnostic = (
            f"request={req_id}, allocated={expected}, exported={values}, "
            f"padding aliases live blocks={padding_aliases}"
        )
        print(diagnostic)
        assert values[: len(expected)] == expected, diagnostic
        assert values[len(expected) :] == [0] * (width - len(expected)), diagnostic


@pytest.mark.parametrize("empty_batch_between_requests", [False, True])
def test_padded_prefill_after_decode_request_reuses_only_live_blocks(
    empty_batch_between_requests: bool,
):
    manager, batch = _manager_and_batch(max_num_reqs=1)
    old, state = _admit(manager, batch, "old", prompt_len=100)
    _generate_100_tokens(manager, batch, old, state)
    old_blocks = manager.get_block_ids(old.request_id)[0]

    old.status = RequestStatus.FINISHED_LENGTH_CAPPED
    manager.free(old)
    removed_row = batch.remove_request(old.request_id)
    assert removed_row == 0
    if empty_batch_between_requests:
        batch.condense([removed_row])
        batch.refresh_logitsprocs()
        assert batch.num_reqs == 0

    new, _ = _admit(manager, batch, "new", prompt_len=190, row=removed_row)
    if not empty_batch_between_requests:
        batch.condense([])
    new_blocks = manager.get_block_ids(new.request_id)[0]
    assert len(new_blocks) == 3
    # Establish that the real allocator recycled the old tail into the new
    # live prefix. The IDs themselves are never supplied by this test.
    assert old_blocks[-1] in new_blocks
    print(f"previous request blocks={old_blocks}; next prompt blocks={new_blocks}")
    _assert_safe_export(manager, batch, [new.request_id], PADDED_PREFILL_BLOCKS)


def test_compacted_source_row_cannot_expose_another_requests_live_blocks():
    manager, batch = _manager_and_batch(max_num_reqs=2)
    removed, _ = _admit(manager, batch, "removed", prompt_len=32)
    survivor, _ = _admit(manager, batch, "survivor", prompt_len=256)
    survivor_blocks = manager.get_block_ids(survivor.request_id)[0]
    assert len(survivor_blocks) == 4
    assert batch.req_id_to_index[survivor.request_id] == 1

    removed.status = RequestStatus.FINISHED_ABORTED
    manager.free(removed)
    removed_row = batch.remove_request(removed.request_id)
    assert removed_row == 0
    batch.condense([removed_row])
    batch.refresh_logitsprocs()
    assert batch.req_id_to_index[survivor.request_id] == 0
    replacement, _ = _admit(manager, batch, "replacement", prompt_len=190)
    assert batch.req_id_to_index[replacement.request_id] == 1
    assert len(manager.get_block_ids(replacement.request_id)[0]) == 3

    # After condense, row 1 was the source of a four-block move. Reusing it
    # for three blocks must not expose the survivor's fourth block. Reverse
    # row selection checks per-row counts; width > storage checks new padding.
    _assert_safe_export(
        manager,
        batch,
        [replacement.request_id, survivor.request_id],
        PADDED_PREFILL_BLOCKS + 4,
    )
