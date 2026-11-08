# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Multimodal prompts are never split; text chunks obey a depth-dependent cap."""

from types import SimpleNamespace

import pytest
import vllm  # noqa: F401  (resolve the platform plugin before importing ours)

from vllm_tt_plugin.config import (
    get_tt_prefill_chunk_cap_table,
    parse_prefill_chunk_cap_table,
    store_tt_prefill_chunk_cap_table_default,
)
from vllm_tt_plugin.scheduler import TTScheduler

STEP = 8192


def _scheduler(*, cap_table=(), step=STEP, threshold=2048, base_mamba=False):
    s = TTScheduler.__new__(TTScheduler)
    s._prefill_chunk_alignment = 128
    s._prefill_chunk_cap_table = tuple(cap_table)
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


def test_mm_prompt_larger_than_a_step_is_split_on_the_grid_not_stuck():
    s = _scheduler(step=4096, threshold=0)
    assert s._mamba_block_aligned_split(_request(10000, mm=True), 4000) == 3968


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


CAPS = [[0, 4096], [65536, 2048], [131072, 1024]]


def test_cap_table_is_a_step_function_of_start():
    s = _scheduler(cap_table=parse_prefill_chunk_cap_table(CAPS, 128), threshold=0)
    r = _request(300000)
    assert s._cap_prefill_chunk(r, 0, 8192) == 4096
    assert s._cap_prefill_chunk(r, 65535, 8192) == 4096
    assert s._cap_prefill_chunk(r, 65536, 8192) == 2048
    assert s._cap_prefill_chunk(r, 200000, 8192) == 1024
    assert s._cap_prefill_chunk(r, 200000, 512) == 512


def test_cap_runs_before_alignment_in_the_hook():
    s = _scheduler(cap_table=parse_prefill_chunk_cap_table(CAPS, 128), threshold=0)
    assert s._mamba_block_aligned_split(_request(300000, computed=69632), 8192) == 2048


def test_no_table_or_start_below_first_entry_means_no_cap():
    r = _request(300000)
    assert _scheduler()._cap_prefill_chunk(r, 0, 8192) == 8192
    s = _scheduler(cap_table=((1000, 128),))
    assert s._cap_prefill_chunk(r, 999, 8192) == 8192
    assert s._cap_prefill_chunk(r, 1000, 8192) == 128


def test_cap_never_applies_to_multimodal_requests():
    s = _scheduler(cap_table=parse_prefill_chunk_cap_table(CAPS, 128), threshold=0)
    r = _request(6000, mm=True)
    assert s._cap_prefill_chunk(r, 0, 6000) == 6000
    assert s._mamba_block_aligned_split(r, 6000) == 6000


def test_cap_applies_to_the_final_chunk_too():
    s = _scheduler(cap_table=((0, 1024),), threshold=0)
    r = _request(5000, computed=3072)
    assert s._mamba_block_aligned_split(r, 1928) == 1024


def test_parse_validates_the_table():
    assert parse_prefill_chunk_cap_table(None, 128) == ()
    assert parse_prefill_chunk_cap_table([], 128) == ()
    assert parse_prefill_chunk_cap_table([[10, 256], [0, 128]], 128) == (
        (0, 128),
        (10, 256),
    )
    for bad in ([[0, 100]], [[0]], [[-1, 128]], [[0, 0]], [[0, 128], [0, 256]], 5):
        with pytest.raises(ValueError):
            parse_prefill_chunk_cap_table(bad, 128)
    # No grid, no multiple-of rule.
    assert parse_prefill_chunk_cap_table([[0, 100]], 0) == ((0, 100),)


def test_config_reads_the_table_and_checks_it_against_the_alignment():
    cfg = SimpleNamespace(additional_config={"tt": {"prefill_chunk_cap_table": CAPS}})
    assert get_tt_prefill_chunk_cap_table(cfg)[1] == (65536, 2048)
    cfg = SimpleNamespace(additional_config={"tt": {}})
    assert get_tt_prefill_chunk_cap_table(cfg) == ()
    cfg = SimpleNamespace(
        additional_config={
            "tt": {"prefill_chunk_cap_table": [[0, 96]], "prefill_chunk_alignment": 64}
        }
    )
    with pytest.raises(ValueError, match="multiple"):
        get_tt_prefill_chunk_cap_table(cfg)


def test_model_declared_table_is_copied_and_the_operator_wins():
    cfg = SimpleNamespace(additional_config=None)
    store_tt_prefill_chunk_cap_table_default(cfg, None)
    assert cfg.additional_config is None
    store_tt_prefill_chunk_cap_table_default(cfg, ((0, 4096),))
    assert cfg.additional_config["tt"]["prefill_chunk_cap_table"] == [[0, 4096]]
    cfg = SimpleNamespace(
        additional_config={"tt": {"prefill_chunk_cap_table": [[0, 1024]]}}
    )
    store_tt_prefill_chunk_cap_table_default(cfg, [[0, 4096]])
    assert cfg.additional_config["tt"]["prefill_chunk_cap_table"] == [[0, 1024]]
    with pytest.raises(ValueError):
        store_tt_prefill_chunk_cap_table_default(
            SimpleNamespace(additional_config={}), [[0, 100]]
        )
