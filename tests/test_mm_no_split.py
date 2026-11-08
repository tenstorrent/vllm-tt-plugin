# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A multimodal prompt that fits a step is scheduled whole
(``TTScheduler._mm_no_split_grant``)."""

from contextlib import suppress
from types import SimpleNamespace

import vllm  # noqa: F401  (resolve the platform plugin before importing ours)

from vllm_tt_plugin.scheduler import TTScheduler

STEP = 8192


def _scheduler(*, step=STEP, threshold=2048, draft_slots=0):
    s = TTScheduler.__new__(TTScheduler)
    s._prefill_chunk_alignment = 128
    s.need_mamba_block_aligned_split = False
    s.max_num_scheduled_tokens = step
    s.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=step,
        max_num_scheduled_tokens=None,
        long_prefill_token_threshold=threshold,
    )
    s.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(max_num_new_slots_for_drafting=draft_slots)
        if draft_slots
        else None
    )
    s._step_granted_rows = 0
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
    s._step_granted_rows = 0
    assert s._mamba_block_aligned_split(_request(5000, mm=True), 2048) == 5000


def test_a_waiting_mm_request_cannot_deadlock_behind_its_own_threshold():
    # Remainder above the threshold but within a step: granted on an empty step.
    s = _scheduler(threshold=1024)
    assert s._mamba_block_aligned_split(_request(STEP, mm=True), 1024) == STEP


def test_grants_accumulate_so_a_second_mm_request_never_overdraws_the_step():
    s = _scheduler()
    assert s._mamba_block_aligned_split(_request(5000, mm=True), 2048) == 5000
    assert s._mamba_block_aligned_split(_request(4000, mm=True), 2048) == 0


def test_draft_slots_count_against_the_step():
    # vLLM takes tokens plus draft slots out of max_num_batched_tokens.
    s = _scheduler(draft_slots=3)
    assert s._mamba_block_aligned_split(_request(STEP - 3, mm=True), 2048) == STEP - 3
    s._step_granted_rows = 0
    assert s._mamba_block_aligned_split(_request(STEP - 2, mm=True), 2048) == 2048
    s._step_granted_rows = 0
    assert s._mamba_block_aligned_split(_request(4000, mm=True), 2048) == 4000
    # 4003 rows used: 4189 tokens and 3 slots no longer fit.
    assert s._mamba_block_aligned_split(_request(4189, mm=True), 2048) == 0
    assert s._mamba_block_aligned_split(_request(4186, mm=True), 2048) == 4186


def test_mm_prompt_larger_than_a_step_is_split_on_the_grid_not_stuck():
    s = _scheduler(step=4096, threshold=0)
    assert s._mamba_block_aligned_split(_request(10000, mm=True), 4000) == 4000
    # A text-like chunk end: the alignment leaves a multimodal chunk alone.
    assert s._mamba_block_aligned_split(_request(10000, mm=True), 3999) == 3999


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


def test_schedule_starts_each_step_with_an_empty_count(monkeypatch):
    s = _scheduler()
    s.running = []
    s._step_granted_rows = 5000
    seen = {}

    def stop(self):
        seen["rows"] = self._step_granted_rows
        raise StopIteration

    monkeypatch.setattr(TTScheduler, "_has_pending_prefill", stop)
    with suppress(StopIteration):
        s.schedule()
    assert seen["rows"] == 0


def test_decode_steps_do_not_consume_the_prefill_budget():
    s = _scheduler(draft_slots=3)
    decode = _request(300, computed=309)
    decode.num_tokens = 310
    for _ in range(100):
        assert s._mamba_block_aligned_split(decode, 1) == 1
    assert s._step_granted_rows == 0
