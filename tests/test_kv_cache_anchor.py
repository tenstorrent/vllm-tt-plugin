# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The anchored sliding-window manager keeps a prompt's cache-hit window through a long decode.

Uses vLLM's real block pool and managers; no device.
"""

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_tt_plugin.kv_cache_anchor import AnchoredSlidingWindowManager, AnchoredSlidingWindowSpec

BLOCK, WINDOW, GROUPS, PROMPT, SEQUENCE = 128, 1024, 5, 4096, 6000


def spec_args():
    return dict(block_size=BLOCK, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, sliding_window=WINDOW)


def build(manager_class, spec_class):
    pool = BlockPool(512, True, BLOCK)
    spec = spec_class(**spec_args())
    managers = [
        manager_class(
            kv_cache_spec=spec,
            block_pool=pool,
            enable_caching=True,
            kv_cache_group_id=g,
            scheduler_block_size=BLOCK,
            max_admission_blocks_per_request=265,
        )
        for g in range(GROUPS)
    ]
    return pool, spec, managers


def run(manager_class, spec_class):
    """Prompt of PROMPT tokens, decode to SEQUENCE, churn the whole pool, then look for the prompt-end hit."""
    pool, spec, managers = build(manager_class, spec_class)
    blocks_per_request = SEQUENCE // BLOCK + 1
    hashes = [BlockHash(i.to_bytes(32, "big")) for i in range(blocks_per_request)]
    for g, manager in enumerate(managers):
        blocks = pool.get_new_blocks(blocks_per_request)
        for i, block in enumerate(blocks):
            pool._insert_block_hash(make_block_hash_with_group_id(hashes[i], g), block, BLOCK)
        manager.req_to_blocks["req"] = blocks
        manager.num_cached_block["req"] = blocks_per_request
        manager.remove_skipped_blocks("req", SEQUENCE, PROMPT)
    held = sum(1 for m in managers for b in m.req_to_blocks["req"] if not b.is_null)
    churn = pool.get_new_blocks(pool.get_num_free_blocks())
    boundary = PROMPT - BLOCK
    _, hit = SlidingWindowManager.find_longest_cache_hit(hashes, boundary, list(range(GROUPS)), pool, spec, False, BLOCK)
    detached = [b for m in managers for b in m.pop_blocks_for_free("req")]
    pool.free_blocks(reversed(detached))
    pool.free_blocks(churn)
    assert pool.get_num_free_blocks() == 511, "every block returns to the pool"
    assert all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null)
    return held, hit


def test_stock_manager_loses_the_prompt_window_during_a_long_decode():
    held, hit = run(SlidingWindowManager, SlidingWindowSpec)
    assert hit == 0
    assert held == GROUPS * (WINDOW // BLOCK + 1)


def test_anchored_manager_keeps_the_window_ending_at_the_prompt_hit_boundary():
    held, hit = run(AnchoredSlidingWindowManager, AnchoredSlidingWindowSpec)
    assert hit == PROMPT - BLOCK
    window_blocks = (WINDOW - 1 + BLOCK - 1) // BLOCK
    assert held == GROUPS * (WINDOW // BLOCK + 1 + window_blocks)


def test_anchor_survives_incremental_decode_steps():
    pool, spec, (manager,) = (lambda p, s, m: (p, s, m[:1]))(*build(AnchoredSlidingWindowManager, AnchoredSlidingWindowSpec))
    blocks = pool.get_new_blocks(SEQUENCE // BLOCK + 1)
    manager.req_to_blocks["req"] = blocks
    manager.num_cached_block["req"] = len(blocks)
    first, last = manager._anchor_blocks(PROMPT)
    for computed in range(PROMPT, SEQUENCE + 1, 37):
        manager.remove_skipped_blocks("req", computed, PROMPT)
        assert all(not b.is_null for b in blocks[first:last])
    skipped = manager.get_num_skipped_tokens(SEQUENCE) // BLOCK
    assert all(b.is_null for b in blocks[:first])
    assert all(b.is_null for b in blocks[last:skipped])
    assert all(not b.is_null for b in blocks[skipped:])


def test_short_prompt_has_no_anchor_and_matches_the_stock_manager():
    pool, spec, (manager,) = (lambda p, s, m: (p, s, m[:1]))(*build(AnchoredSlidingWindowManager, AnchoredSlidingWindowSpec))
    assert manager._anchor_blocks(100) == (0, 0)


def test_spec_is_registered_as_a_sliding_window_group_and_reserves_the_anchor():
    stock, anchored = SlidingWindowSpec(**spec_args()), AnchoredSlidingWindowSpec(**spec_args())
    assert KVCacheSpecRegistry.get_manager_class(anchored) is AnchoredSlidingWindowManager
    assert KVCacheSpecRegistry.get_uniform_type_base_spec(anchored) is SlidingWindowSpec
    assert anchored.max_admission_blocks_per_request(8192, 262144) == stock.max_admission_blocks_per_request(8192, 262144) + 8
    assert anchored.is_uniform_with_collection({"a": stock, "b": anchored})
