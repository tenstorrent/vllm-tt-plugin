# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Multimodal prompts are never split."""

from types import SimpleNamespace

import vllm  # noqa: F401  (resolve the platform plugin before importing ours)

from vllm_tt_plugin.scheduler import TTScheduler

STEP = 8192


def _scheduler(*, step=STEP, threshold=2048, base_mamba=False):
    s = TTScheduler.__new__(TTScheduler)
    s._prefill_chunk_alignment = 128
    s.need_mamba_block_aligned_split = base_mamba
    s.max_num_scheduled_tokens = step
    s.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=step,
        max_num_scheduled_tokens=None,
        long_prefill_token_threshold=threshold,
    )
    s._step_prefill_tokens = 0
    return s


def _request(prompt, computed=0, mm=False):
    return SimpleNamespace(
        num_prompt_tokens=prompt,
        num_tokens=prompt,
        num_computed_tokens=computed,
        mm_features=[object()] if mm else [],
        has_encoder_inputs=mm,
    )


def test_mm_request_takes_its_whole_prompt_past_the_threshold():
    s = _scheduler(threshold=2048)
    # vLLM already clipped the grant to the 2048 threshold; the guard widens it.
    assert s._mamba_block_aligned_split(_request(5000, mm=True), 2048) == 5000


def test_mm_request_waits_when_the_leftover_step_budget_is_short():
    s = _scheduler()
    assert s._mamba_block_aligned_split(_request(9000), 6144) == 6144  # text first
    assert s._mamba_block_aligned_split(_request(5000, mm=True), 2048) == 0
    # A fresh step gives it the whole budget.
    s._step_prefill_tokens = 0
    assert s._mamba_block_aligned_split(_request(5000, mm=True), 2048) == 5000


def test_waiting_mm_request_cannot_deadlock_behind_its_own_threshold():
    # Remainder above the threshold but within a step: granted on an empty step.
    s = _scheduler(threshold=1024)
    assert s._mamba_block_aligned_split(_request(STEP, mm=True), 1024) == STEP


def test_grants_accumulate_so_a_second_mm_request_never_overshoots_the_step():
    s = _scheduler()
    assert s._mamba_block_aligned_split(_request(5000, mm=True), 2048) == 5000
    assert s._mamba_block_aligned_split(_request(4000, mm=True), 2048) == 0


def test_mm_prompt_larger_than_a_step_is_split_not_stuck():
    """The chunk is not rounded to the grid: its end stays where vLLM's
    multimodal rule (or ``_keep_multimodal_items_whole``) put it, outside the
    items."""
    s = _scheduler(step=4096, threshold=0)
    assert s._mamba_block_aligned_split(_request(10000, mm=True), 4000) == 4000


def test_mm_decode_row_passes_through():
    s = _scheduler()
    request = _request(300, computed=309, mm=True)
    request.num_tokens = 310
    assert s._mamba_block_aligned_split(request, 1) == 1


def test_mm_resumed_chunk_with_progress_is_granted_the_rest():
    s = _scheduler()
    assert (
        s._mamba_block_aligned_split(_request(5000, computed=1000, mm=True), 100)
        == 4000
    )


def test_decode_steps_do_not_consume_the_prefill_budget():
    s = _scheduler()
    decode = _request(300, computed=309)
    decode.num_tokens = 310
    for _ in range(100):
        assert s._mamba_block_aligned_split(decode, 1) == 1
    assert s._step_prefill_tokens == 0
