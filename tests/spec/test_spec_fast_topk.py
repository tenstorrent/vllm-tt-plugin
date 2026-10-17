"""The fast top-k/top-p path of _constrained_probs equals the reference path."""

import pytest
import torch

from vllm_tt_plugin import spec_accept
from vllm_tt_plugin.spec_accept import (
    SpecSamplingInputs,
    _constrained_probs,
    accept_sampled_drafts,
)

V = 4096


def _both(monkeypatch, logits, temperature, top_k, top_p, min_p=None):
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "0")
    ref = _constrained_probs(logits, temperature, top_k, top_p, min_p)
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "1")
    fast = _constrained_probs(logits, temperature, top_k, top_p, min_p)
    return ref, fast


def _check(ref, fast):
    assert torch.equal(ref > 0, fast > 0)
    assert torch.allclose(ref, fast, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_mixed_rows_match_reference(monkeypatch, bf16, seed):
    g = torch.Generator().manual_seed(seed)
    rows, width = 6, 4
    logits = torch.randn(rows, width, V, generator=g) * 3
    if bf16:
        logits = logits.to(torch.bfloat16).to(torch.float32)
    temperature = torch.tensor([0.6, 1.0, 0.0, 0.8, 0.6, 1.3])
    top_k = torch.tensor([20, 1, 20, V, 50, 1024], dtype=torch.int32)
    top_p = torch.tensor([0.95, 0.95, 1.0, 0.95, 1.0, 0.5])
    ref, fast = _both(monkeypatch, logits, temperature, top_k, top_p)
    _check(ref, fast)


@pytest.mark.parametrize("bf16", [False, True])
def test_top_k_disabled_with_top_p_falls_back(monkeypatch, bf16):
    g = torch.Generator().manual_seed(11)
    logits = torch.randn(3, 4, V, generator=g) * 2
    if bf16:
        logits = logits.to(torch.bfloat16).to(torch.float32)
    temperature = torch.full((3,), 0.6)
    top_k = torch.tensor([V, 20, V], dtype=torch.int32)
    top_p = torch.tensor([0.95, 0.95, 1.0])
    ref, fast = _both(monkeypatch, logits, temperature, top_k, top_p)
    _check(ref, fast)
    ref, fast = _both(monkeypatch, logits, temperature, None, top_p)
    _check(ref, fast)
    ref, fast = _both(monkeypatch, logits, temperature, top_k, None)
    _check(ref, fast)


def test_top_k_above_cap_falls_back(monkeypatch):
    logits = torch.randn(2, 3, V, generator=torch.Generator().manual_seed(2))
    top_k = torch.tensor([2000, 20], dtype=torch.int32)
    top_p = torch.tensor([0.9, 0.9])
    ref, fast = _both(monkeypatch, logits, torch.ones(2), top_k, top_p)
    _check(ref, fast)


def test_ties_beyond_the_margin_fall_back(monkeypatch):
    logits = torch.randn(2, 2, V, generator=torch.Generator().manual_seed(3))
    # 500 logits tie at the k-th value: far more than the margin holds.
    logits[0, :, :500] = 10.0
    top_k = torch.tensor([20, 20], dtype=torch.int32)
    top_p = torch.tensor([1.0, 0.95])
    ref, fast = _both(monkeypatch, logits, torch.ones(2), top_k, top_p)
    assert int((ref[0, 0] > 0).sum()) == 500
    _check(ref, fast)


def test_min_p_then_top_k_matches(monkeypatch):
    logits = torch.randn(3, 4, V, generator=torch.Generator().manual_seed(4)) * 3
    top_k = torch.tensor([20, 20, 5], dtype=torch.int32)
    top_p = torch.tensor([0.95, 1.0, 0.9])
    min_p = torch.tensor([0.1, 0.05, 0.0])
    ref, fast = _both(monkeypatch, logits, torch.full((3,), 0.7), top_k, top_p, min_p)
    _check(ref, fast)


def test_caller_logits_are_not_modified(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "1")
    logits = torch.randn(2, 3, V, generator=torch.Generator().manual_seed(5))
    before = logits.clone()
    _constrained_probs(
        logits, torch.ones(2), torch.tensor([20, 20], dtype=torch.int32),
        torch.tensor([0.9, 0.9]),
    )
    assert torch.equal(before, logits)


def test_accept_walk_is_identical_with_the_flag_on_and_off(monkeypatch):
    rows, k_drafts = 4, 3
    for trial in range(40):
        g = torch.Generator().manual_seed(1000 + trial)
        logits = torch.randn(rows, k_drafts + 1, V, generator=g) * 3
        if trial % 2:
            logits = logits.to(torch.bfloat16).to(torch.float32)
        drafts = logits[:, :k_drafts].argmax(-1).to(torch.int32)
        drafts[:, 1] = torch.randint(0, V, (rows,), generator=g, dtype=torch.int32)
        valid = torch.tensor([3, 2, 0, 3], dtype=torch.int32)
        outs = []
        for flag in ("0", "1"):
            monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", flag)
            sampling = SpecSamplingInputs(
                vocab_size=V,
                temperature=torch.tensor([0.6, 0.6, 0.8, 0.0]),
                top_k=torch.tensor([20, 20, 1, 20], dtype=torch.int32),
                top_p=torch.tensor([0.95, 0.95, 0.95, 1.0]),
                generators={r: torch.Generator().manual_seed(7 + r) for r in range(rows)},
            )
            res = accept_sampled_drafts(logits, drafts, valid, sampling)
            outs.append((res.committed_token_ids, res.accepted_counts))
        assert torch.equal(outs[0][0], outs[1][0])
        assert torch.equal(outs[0][1], outs[1][1])
