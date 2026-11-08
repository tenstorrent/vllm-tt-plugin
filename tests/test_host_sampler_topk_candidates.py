# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p

from vllm_tt_plugin.host_sampler import TTTopKTopPSampler


def _logits(rows, vocab, scale, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, vocab, generator=gen) * scale


def _spy_old_path(monkeypatch):
    calls = []

    def old(self, logits, generators, k, p):
        calls.append(1)
        return torch.zeros(logits.shape[0], dtype=torch.long), None

    monkeypatch.setattr(TopKTopPSampler, "forward_native", old)
    return calls


@pytest.mark.parametrize("vocab", [12000, 248320])
@pytest.mark.parametrize("scale", [1.0, 3.0, 6.0])
@pytest.mark.parametrize("k", [1, 20, 50, 1024])
@pytest.mark.parametrize("p", [None, 0.5, 0.95, 1.0])
def test_candidate_cut_keeps_the_full_vocab_kept_set(vocab, scale, k, p):
    logits = _logits(2, vocab, scale, 17)
    k_t = torch.full((2,), k, dtype=torch.int32)
    p_t = None if p is None else torch.full((2,), p)
    result = TTTopKTopPSampler()._top_k_candidates(logits.clone(), k_t, p_t)
    assert result is not None
    cand, ids = result
    expected = torch.isfinite(apply_top_k_top_p(logits.clone(), k_t, p_t))
    kept = torch.zeros_like(expected)
    kept.scatter_(1, ids, torch.isfinite(cand))
    assert torch.equal(kept, expected)
    full = apply_top_k_top_p(logits.clone(), k_t, p_t).softmax(-1)
    fast = torch.zeros_like(full)
    fast.scatter_(1, ids, cand.softmax(-1))
    torch.testing.assert_close(fast, full, rtol=1e-5, atol=1e-7)


def test_sampled_frequencies_match_the_full_path_probabilities():
    torch.manual_seed(7)
    vocab, draws = 12000, 40000
    row = _logits(1, vocab, 3.0, 5)
    k = torch.full((draws,), 20, dtype=torch.int32)
    p = torch.full((draws,), 0.95)
    expected = apply_top_k_top_p(row.clone(), k[:1], p[:1]).softmax(-1)[0]
    tokens, scores = TTTopKTopPSampler()(
        row.expand(draws, vocab).contiguous(), {}, k, p
    )
    assert scores is None
    freq = torch.bincount(tokens, minlength=vocab) / draws
    assert bool((freq[expected == 0] == 0).all())
    torch.testing.assert_close(freq, expected, rtol=0, atol=0.01)


def test_seeded_rows_use_the_old_path(monkeypatch):
    calls = _spy_old_path(monkeypatch)
    logits = _logits(2, 8000, 3.0, 1)
    k = torch.full((2,), 20, dtype=torch.int32)
    gens = {0: torch.Generator().manual_seed(1)}
    TTTopKTopPSampler()(logits, gens, k, None)
    assert len(calls) == 1


def test_no_top_k_uses_the_old_path(monkeypatch):
    calls = _spy_old_path(monkeypatch)
    TTTopKTopPSampler()(_logits(2, 8000, 3.0, 1), {}, None, torch.full((2,), 0.9))
    assert len(calls) == 1


@pytest.mark.parametrize("kmax", [1025])
def test_large_k_uses_the_old_path(monkeypatch, kmax):
    calls = _spy_old_path(monkeypatch)
    k = torch.tensor([20, kmax], dtype=torch.int32)
    TTTopKTopPSampler()(_logits(2, 8000, 3.0, 1), {}, k, None)
    assert len(calls) == 1


def test_invalid_k_uses_the_old_path(monkeypatch):
    calls = _spy_old_path(monkeypatch)
    k = torch.tensor([20, 0], dtype=torch.int32)
    TTTopKTopPSampler()(_logits(2, 8000, 3.0, 1), {}, k, None)
    assert len(calls) == 1


def test_tie_at_kth_value_past_the_candidates_uses_the_old_path(monkeypatch):
    calls = _spy_old_path(monkeypatch)
    logits = _logits(2, 8000, 1.0, 1)
    logits[0, :200] = 50.0  # k=20: tied k-th value fills every candidate slot
    k = torch.full((2,), 20, dtype=torch.int32)
    sampler = TTTopKTopPSampler()
    assert sampler._top_k_candidates(logits, k, None) is None
    sampler(logits, {}, k, None)
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["processed_logits", "processed_logprobs"])
def test_processed_logprobs_modes_use_the_old_path(monkeypatch, mode):
    calls = _spy_old_path(monkeypatch)
    k = torch.full((2,), 20, dtype=torch.int32)
    TTTopKTopPSampler(mode)(_logits(2, 8000, 3.0, 1), {}, k, None)
    assert len(calls) == 1


def test_fast_path_does_not_use_the_old_path(monkeypatch):
    calls = _spy_old_path(monkeypatch)
    k = torch.full((2,), 20, dtype=torch.int32)
    tokens, scores = TTTopKTopPSampler()(_logits(2, 8000, 3.0, 1), {}, k, None)
    assert not calls and scores is None and tokens.shape == (2,)


def test_batch_with_different_k_and_p():
    torch.manual_seed(3)
    logits = _logits(4, 12000, 3.0, 9)
    k = torch.tensor([1, 20, 50, 1024], dtype=torch.int32)
    p = torch.tensor([0.5, 0.95, 1.0, 0.9])
    kept = torch.isfinite(apply_top_k_top_p(logits.clone(), k, p))
    sampler = TTTopKTopPSampler()
    for _ in range(50):
        tokens, scores = sampler(logits.clone(), {}, k, p)
        assert scores is None
        assert kept[torch.arange(4), tokens].all()
    assert (
        sampler(logits.clone(), {}, k[:1].expand(4).contiguous(), None)[0]
        == logits.argmax(-1)
    ).all()
