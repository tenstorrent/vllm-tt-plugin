# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real host allocator tests for Gemma4 whole-prompt sliding-tail storage.

Use KVCacheManager and its actual BlockPool with one full group and five
sliding groups, representing the model's 5:1 attention grouping. No cache
tensors, TT devices, engines, weights, or servers are created. These tests
prove allocation/ownership contracts, not native prefill or attention results.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_tt_plugin.whole_prompt_cache import (
    WholePromptSlidingWindowManager,
    WholePromptSlidingWindowSpec,
    validate_whole_prompt_cache,
)

BLOCK = 128
WINDOW = 1024
MAX_CONTEXT = 262144
SLIDING_GROUPS = 5


def _ceil_blocks(tokens):
    return (tokens + BLOCK - 1) // BLOCK


def _sliding_spec(*, standard=False):
    cls = SlidingWindowSpec if standard else WholePromptSlidingWindowSpec
    return cls(
        block_size=BLOCK,
        num_kv_heads=16,
        head_size=256,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
    )


def _manager(num_blocks, *, standard=False):
    sliding = _sliding_spec(standard=standard)
    full = FullAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=4,
        head_size=512,
        dtype=torch.bfloat16,
        page_size_padded=sliding.page_size_bytes,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],  # Allocator metadata needs no tensor allocation.
        kv_cache_groups=[KVCacheGroupSpec(["full"], full)]
        + [KVCacheGroupSpec([f"sliding.{i}"], sliding) for i in range(SLIDING_GROUPS)],
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=MAX_CONTEXT,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        enable_caching=False,
    )


def _request(req_id, prompt_len, token=17):
    return Request(
        request_id=req_id,
        prompt_token_ids=[token] * prompt_len,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1024),
        pooling_params=None,
    )


def _config():
    return SimpleNamespace(
        max_in_flight_tokens=MAX_CONTEXT,
        model_config=SimpleNamespace(max_model_len=MAX_CONTEXT),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False,
            max_num_batched_tokens=MAX_CONTEXT,
            long_prefill_token_threshold=0,
            disable_hybrid_kv_cache_manager=False,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )


def _fresh_live_counts(prompt_len):
    # The next decode at P needs history positions max(0, P-W+1)..P-1.
    first_history_position = max(0, prompt_len - WINDOW + 1)
    full_pages = _ceil_blocks(prompt_len)
    sliding_pages = full_pages - first_history_position // BLOCK
    return [full_pages] + [sliding_pages] * SLIDING_GROUPS


def _live_ids(manager, request):
    return [
        block.block_id
        for group in manager.get_blocks(request.request_id).blocks
        for block in group
        if not block.is_null
    ]


def _assert_live_ownership(manager, requests):
    used = []
    for request in requests:
        for group in manager.get_blocks(request.request_id).blocks:
            for block in group:
                if block.is_null:
                    assert block is manager.block_pool.null_block
                    continue
                assert block.ref_cnt == 1
                used.append(block.block_id)
    assert len(used) == len(set(used)), "Live groups/requests share a physical block"
    assert manager.block_pool.get_num_free_blocks() == (
        manager.block_pool.num_gpu_blocks - 1 - len(used)
    )


def _assert_rows(manager, request, *, endpoint, evicted_before):
    """Check rows using logical token positions and the required live suffix."""
    rows = manager.get_blocks(request.request_id).blocks
    expected_width = _ceil_blocks(endpoint)
    assert all(len(row) == expected_width for row in rows)
    assert all(not block.is_null for block in rows[0])
    first_live_page = max(0, evicted_before) // BLOCK
    for row in rows[1:]:
        assert all(block.is_null for block in row[:first_live_page])
        assert all(not block.is_null for block in row[first_live_page:])
        assert len(row) - first_live_page <= 9
        assert not row[(endpoint - 1) // BLOCK].is_null
    return rows


@pytest.mark.parametrize(
    "prompt_len",
    [
        1,
        127,
        128,
        129,
        1023,
        1024,
        1025,
        1151,
        1152,
        1153,
        2047,
        2048,
        2049,
        262111,
        262144,
    ],
)
def test_fresh_prompt_uses_only_live_tail_with_arbitrary_endpoints(prompt_len):
    counts = _fresh_live_counts(prompt_len)
    # Exactly the required live pages plus the allocator's reserved null page.
    manager = _manager(1 + sum(counts))
    request = _request("fresh", prompt_len)

    allocation = manager.allocate_slots(request, num_new_tokens=prompt_len)

    assert allocation is not None
    assert [len(group) for group in allocation.blocks] == counts
    assert type(manager.coordinator.single_type_managers[0]) is FullAttentionManager
    assert all(
        type(group) is WholePromptSlidingWindowManager
        for group in manager.coordinator.single_type_managers[1:]
    )
    rows = _assert_rows(
        manager,
        request,
        endpoint=prompt_len,
        evicted_before=max(0, prompt_len - WINDOW + 1),
    )
    for group, count in zip(rows, counts):
        assert sum(not block.is_null for block in group) == count
    _assert_live_ownership(manager, [request])
    assert manager.block_pool.get_num_free_blocks() == 0

    # Scheduler new/resumed requests use the complete row (including nulls),
    # not merely the compact allocation delta returned by allocate_slots.
    assert all(
        len(ids) == _ceil_blocks(prompt_len)
        for ids in manager.get_block_ids(request.request_id)
    )
    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == sum(counts)


@pytest.mark.parametrize("prompt_len", [2049, 262111, 262144])
def test_standard_sliding_manager_cannot_use_the_same_bounded_tail_budget(prompt_len):
    capacity = 1 + sum(_fresh_live_counts(prompt_len))
    standard = _manager(capacity, standard=True)
    repaired = _manager(capacity)
    assert type(standard.coordinator.single_type_managers[1]) is SlidingWindowManager

    assert standard.allocate_slots(_request("old", prompt_len), prompt_len) is None
    assert standard.block_pool.get_num_free_blocks() == capacity - 1
    assert repaired.allocate_slots(_request("new", prompt_len), prompt_len) is not None
    assert repaired.block_pool.get_num_free_blocks() == 0


@pytest.mark.parametrize("prompt_len", [1023, 1024, 1025, 2047, 2048, 2049])
def test_decode_growth_evicts_old_window_pages_and_keeps_full_history(prompt_len):
    manager = _manager(256)
    request = _request("decode", prompt_len)
    assert manager.allocate_slots(request, prompt_len) is not None
    request.num_computed_tokens = prompt_len
    growth_positions = []
    evicted_ids = set()
    for offset in range(3 * BLOCK + 3):
        position = prompt_len + offset
        before = manager.get_block_ids(request.request_id)
        before_live = set(_live_ids(manager, request))
        request.append_output_token_ids(71)

        allocation = manager.allocate_slots(request, num_new_tokens=1)

        assert allocation is not None
        rows = _assert_rows(
            manager,
            request,
            endpoint=position + 1,
            evicted_before=max(0, position - WINDOW + 1),
        )
        after = manager.get_block_ids(request.request_id)
        assert after[0][: len(before[0])] == before[0]
        expected_new = int(position % BLOCK == 0)
        assert [len(group) for group in allocation.blocks] == [expected_new] * 6
        if expected_new:
            growth_positions.append(position)
            for group, delta in zip(rows, allocation.blocks):
                assert group[position // BLOCK] is delta[0]
        evicted_ids.update(before_live - set(_live_ids(manager, request)))
        _assert_live_ownership(manager, [request])
        request.num_computed_tokens += 1

    assert len(growth_positions) >= 3
    assert len(evicted_ids) >= SLIDING_GROUPS * 2


@pytest.mark.parametrize("concurrent", [4, 32])
def test_concurrent_requests_keep_distinct_pages_across_growth_and_eviction(concurrent):
    manager = _manager(4096)
    requests = [
        _request(f"req-{index}", 1151 + 17 * index, token=100 + index)
        for index in range(concurrent)
    ]
    for request in requests:
        assert manager.allocate_slots(request, request.num_prompt_tokens) is not None
        request.num_computed_tokens = request.num_prompt_tokens
    _assert_live_ownership(manager, requests)

    growth_by_request = {request.request_id: 0 for request in requests}
    for offset in range(2 * BLOCK + 3):
        # Alternate scheduler enumeration; allocation ownership is request-based.
        order = requests if offset % 2 else list(reversed(requests))
        for request in order:
            position = request.num_computed_tokens
            request.append_output_token_ids(200 + requests.index(request))
            allocation = manager.allocate_slots(request, 1)
            assert allocation is not None
            _assert_rows(
                manager,
                request,
                endpoint=position + 1,
                evicted_before=max(0, position - WINDOW + 1),
            )
            if allocation.blocks[0]:
                growth_by_request[request.request_id] += 1
            request.num_computed_tokens += 1
        _assert_live_ownership(manager, requests)
    assert all(count >= 2 for count in growth_by_request.values())

    survivors = requests[1::2]
    for request in requests[::2]:
        manager.free(request)
    _assert_live_ownership(manager, survivors)
    for request in survivors:
        manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == 4095


def test_free_reuse_and_preemption_recompute_rebuild_tail_ownership():
    prompt_len = 2049
    capacity = 1 + sum(_fresh_live_counts(prompt_len))
    manager = _manager(capacity)
    first = _request("first", prompt_len, 17)
    assert manager.allocate_slots(first, prompt_len) is not None
    original_ids = set(_live_ids(manager, first))
    first.num_computed_tokens = prompt_len

    manager.free(first)
    assert manager.block_pool.get_num_free_blocks() == capacity - 1
    assert all(manager.block_pool.blocks[index].ref_cnt == 0 for index in original_ids)

    second = _request("second", prompt_len, 23)
    assert manager.allocate_slots(second, prompt_len) is not None
    # The pool was full, so every newly owned physical block must be reused.
    assert set(_live_ids(manager, second)) == original_ids
    _assert_live_ownership(manager, [second])
    manager.free(second)

    # Resume recomputes from zero: there are no prefix hits or retained rows.
    first.num_computed_tokens = 0
    first.num_preemptions += 1
    computed, num_computed, shared_prefix_boundary = manager.get_computed_blocks(first)
    assert num_computed == shared_prefix_boundary == 0
    assert all(not group for group in computed.blocks)
    assert manager.allocate_slots(first, prompt_len) is not None
    assert set(_live_ids(manager, first)) == original_ids
    _assert_rows(
        manager,
        first,
        endpoint=prompt_len,
        evicted_before=prompt_len - WINDOW + 1,
    )
    _assert_live_ownership(manager, [first])


def test_memory_bound_is_independent_of_full_prompt_scheduler_budget():
    config = _config()
    spec = _sliding_spec()
    validate_whole_prompt_cache(config)
    assert spec.max_memory_usage_bytes(config) == 9 * spec.page_size_bytes
    # This is the actual prior SlidingWindowSpec worst-case arithmetic.
    standard = _sliding_spec(standard=True)
    assert standard.max_memory_usage_bytes(config) == 2049 * standard.page_size_bytes


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("scheduler_config.enable_chunked_prefill", True, "chunked prefill"),
        (
            "scheduler_config.long_prefill_token_threshold",
            1024,
            "long-prefill threshold",
        ),
        ("cache_config.enable_prefix_caching", True, "prefix caching"),
        ("scheduler_config.disable_hybrid_kv_cache_manager", True, "hybrid cache"),
        ("kv_transfer_config", object(), "KV transfer"),
        ("speculative_config", object(), "speculative decoding"),
        ("parallel_config.decode_context_parallel_size", 2, "decode context"),
        ("parallel_config.prefill_context_parallel_size", 2, "prefill context"),
    ],
)
def test_unsupported_config_rejected_before_memory_estimate(path, value, message):
    config = _config()
    parts = path.split(".")
    owner = config if len(parts) == 1 else getattr(config, parts[0])
    setattr(owner, parts[-1], value)
    with pytest.raises(ValueError, match=message):
        validate_whole_prompt_cache(config)
    with pytest.raises(ValueError, match=message):
        _sliding_spec().max_memory_usage_bytes(config)


@pytest.mark.parametrize(
    "kwargs",
    [{"enable_caching": True}, {"dcp_world_size": 2}, {"pcp_world_size": 2}],
)
def test_manager_constructor_rejects_unsupported_modes(kwargs):
    pool = BlockPool(100, enable_caching=False, hash_block_size=BLOCK)
    arguments = dict(enable_caching=False, dcp_world_size=1, pcp_world_size=1)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match="uncached, whole-prompt"):
        WholePromptSlidingWindowManager(
            _sliding_spec(),
            block_pool=pool,
            kv_cache_group_id=0,
            scheduler_block_size=BLOCK,
            **arguments,
        )
    assert pool.get_num_free_blocks() == 99


@pytest.mark.parametrize("mode", ["computed-offset", "external-cache", "lookahead"])
def test_fresh_prefill_rejects_nonzero_start_external_cache_and_lookahead(mode):
    manager = _manager(100)
    request = _request("invalid", 2049)
    kwargs = {"num_new_tokens": request.num_prompt_tokens}
    if mode == "computed-offset":
        request.num_computed_tokens = 1
        kwargs["num_new_tokens"] -= 1
    elif mode == "external-cache":
        kwargs["num_external_computed_tokens"] = 1
    else:
        kwargs["num_lookahead_tokens"] = 1

    with pytest.raises(ValueError, match="Fresh whole-prompt prefill"):
        manager.allocate_slots(request, **kwargs)
    assert manager.block_pool.get_num_free_blocks() == 99
    assert not _live_ids(manager, request)
