# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The topk support walk counts how often it falls back to the full logits."""

from __future__ import annotations

import torch

from vllm_tt_plugin.spec_accept import (
    accept_sampled_support,
    support_fallback_stats,
)

from .test_spec_device_topk import (
    K,
    M,
    V,
    drafts_for,
    exact_support,
    sampling_inputs,
)


def _delta(before: dict[str, int]) -> dict[str, int]:
    after = support_fallback_stats()
    return {name: after[name] - before[name] for name in after}


def _walk(logits, temperature, top_k, fetch):
    rows, width, _ = logits.shape
    drafts, num_valid = drafts_for(logits, 3)
    values, ids = exact_support(logits, M)
    return accept_sampled_support(
        logits.argmax(-1).to(torch.int32),
        values,
        ids,
        drafts,
        num_valid,
        sampling_inputs(rows, temperature, top_k, None, {0: 1, 1: 2}),
        fetch,
    )


def test_a_tie_reaching_the_last_candidate_counts_an_overflow_and_a_fetch():
    rows, width = 2, K + 1
    g = torch.Generator().manual_seed(5)
    logits = torch.stack(
        [torch.randperm(V, generator=g).float() * 1e-2 for _ in range(rows * width)]
    ).reshape(rows, width, V)
    logits[0, :, :500] = 100.0  # row 0 ties far past the M-th candidate
    temperature = torch.tensor([1.0, 1.0])
    top_k = torch.tensor([20, 20], dtype=torch.int32)
    before = support_fallback_stats()
    _walk(logits, temperature, top_k, lambda: logits)
    delta = _delta(before)
    assert delta["support_steps"] == 1
    assert delta["support_positions"] == rows * width
    assert delta["overflow_positions"] == width  # every column of row 0
    assert delta["fetch_calls"] == 1


def test_a_normal_step_counts_positions_but_no_overflow_or_fetch():
    rows, width = 2, K + 1
    g = torch.Generator().manual_seed(6)
    logits = torch.stack(
        [torch.randperm(V, generator=g).float() * 1e-2 for _ in range(rows * width)]
    ).reshape(rows, width, V)
    temperature = torch.tensor([1.0, 0.7])
    top_k = torch.tensor([20, 20], dtype=torch.int32)
    before = support_fallback_stats()
    _walk(logits, temperature, top_k, lambda: logits)
    delta = _delta(before)
    assert delta["support_steps"] == 1
    assert delta["support_positions"] == rows * width
    assert delta["overflow_positions"] == 0
    assert delta["fetch_calls"] == 0


def test_stats_are_a_copy():
    snapshot = support_fallback_stats()
    snapshot["fetch_calls"] += 99
    assert support_fallback_stats()["fetch_calls"] != snapshot["fetch_calls"]
