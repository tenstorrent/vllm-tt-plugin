# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Partial prefill chunks end on the configured alignment grid.

A request admitted on a step's leftover budget would otherwise start every
later chunk on an odd offset, which the model realigns and pads up to the next
power of two. The scheduler rounds the chunk end instead, through vLLM's own
block-aligned-split hook, before any block is allocated.
"""

from types import SimpleNamespace

import pytest
import vllm  # noqa: F401  (resolve the platform plugin before importing ours)

from vllm_tt_plugin.config import get_tt_prefill_chunk_alignment
from vllm_tt_plugin.scheduler import TTScheduler


def _scheduler(alignment=128, *, base_mamba=False):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler._prefill_chunk_alignment = alignment
    scheduler.need_mamba_block_aligned_split = base_mamba
    return scheduler


def _request(prompt_tokens, computed=0, mm=False):
    return SimpleNamespace(
        num_prompt_tokens=prompt_tokens,
        num_tokens=prompt_tokens,
        num_computed_tokens=computed,
        mm_features=[object()] if mm else [],
        has_encoder_inputs=mm,
    )


def test_leftover_budget_first_chunk_is_rounded_down_to_the_grid():
    s = _scheduler()
    assert s._mamba_block_aligned_split(_request(32612), 157) == 128
    assert s._mamba_block_aligned_split(_request(32612), 4096) == 4096


def test_a_new_request_that_cannot_reach_the_grid_waits_a_step():
    s = _scheduler()
    assert s._mamba_block_aligned_split(_request(32612), 100) == 0


def test_a_running_request_never_gets_an_empty_chunk():
    s = _scheduler()
    # Started unaligned (e.g. before the feature, or a cached prefix): the first
    # aligned end is 4224, so a 4096 chunk from 157 shrinks to 4067 ...
    assert (
        s._mamba_block_aligned_split(_request(32612, computed=157), 4096) == 4096 - 29
    )
    # ... but a chunk too short to reach the next boundary is kept whole.
    assert s._mamba_block_aligned_split(_request(32612, computed=157), 50) == 50


def test_the_chunk_that_finishes_the_prompt_is_left_alone():
    s = _scheduler()
    assert s._mamba_block_aligned_split(_request(32612, computed=28672), 3940) == 3940
    assert s._mamba_block_aligned_split(_request(300), 300) == 300


def test_decode_and_multimodal_requests_are_untouched():
    s = _scheduler()
    decode = SimpleNamespace(
        num_prompt_tokens=300, num_tokens=310, num_computed_tokens=309, mm_features=[]
    )
    assert s._mamba_block_aligned_split(decode, 1) == 1
    assert s._mamba_block_aligned_split(_request(32612, mm=True), 157) == 157


def test_zero_disables_and_the_hook_is_only_armed_when_needed():
    off = _scheduler(alignment=0)
    assert off.need_mamba_block_aligned_split is False
    assert off._mamba_block_aligned_split(_request(32612), 157) == 157
    on = _scheduler()
    assert on.need_mamba_block_aligned_split is True
    # The base scheduler's own flag still shows through.
    mamba = _scheduler(alignment=0, base_mamba=True)
    assert mamba.need_mamba_block_aligned_split is True


def test_config_accessor_defaults_and_validates():
    cfg = SimpleNamespace(additional_config={"tt": {}})
    assert get_tt_prefill_chunk_alignment(cfg) == 128
    cfg = SimpleNamespace(additional_config={"tt": {"prefill_chunk_alignment": 0}})
    assert get_tt_prefill_chunk_alignment(cfg) == 0
    cfg = SimpleNamespace(additional_config={"tt": {"prefill_chunk_alignment": -8}})
    with pytest.raises(ValueError):
        get_tt_prefill_chunk_alignment(cfg)
