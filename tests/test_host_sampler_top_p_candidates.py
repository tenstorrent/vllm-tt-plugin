# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

from vllm_tt_plugin.host_sampler import TTTopKTopPSampler

VOCAB = 12000


def _logits(rows, scale, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, VOCAB, generator=gen) * scale


def _nucleus(logits, p):
    return torch.isfinite(apply_top_k_top_p(logits.clone(), None, p))


@pytest.mark.parametrize("scale", [3.0, 6.0])
def test_unseeded_top_p_draws_only_from_the_full_sort_nucleus(scale):
    torch.manual_seed(5)
    logits = _logits(64, scale, 11)
    p = torch.full((64,), 0.95)
    sampler = TTTopKTopPSampler()
    nucleus = _nucleus(logits, p)
    # The candidates hold the nucleus for these rows, so the fast path draws.
    assert int(nucleus.sum(dim=-1).max()) < TTTopKTopPSampler.TOP_P_CANDIDATES
    for _ in range(20):
        tokens, scores = sampler(logits.clone(), {}, None, p)
        assert scores is None
        assert nucleus[torch.arange(64), tokens].all()


def test_unseeded_top_p_matches_the_nucleus_distribution():
    torch.manual_seed(7)
    row = torch.full((VOCAB,), -20.0)
    row[:6] = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -3.0])
    p = torch.full((4000,), 0.9)
    logits = row.expand(4000, VOCAB).contiguous()
    expected = torch.softmax(apply_top_k_top_p(row[None].clone(), None, p[:1]), -1)[0]
    counts = torch.zeros(VOCAB)
    sampler = TTTopKTopPSampler()
    for _ in range(10):
        tokens, _ = sampler(logits.clone(), {}, None, p)
        counts += torch.bincount(tokens, minlength=VOCAB)
    freq = counts / counts.sum()
    assert torch.equal(freq > 0, expected > 0)
    torch.testing.assert_close(freq, expected, rtol=0, atol=0.01)


def test_rows_whose_nucleus_exceeds_the_candidates_use_the_full_sort():
    torch.manual_seed(3)
    peaked = _logits(3, 6.0, 1)
    flat = _logits(3, 0.01, 2)
    logits = torch.cat([peaked, flat])
    p = torch.full((6,), 0.95)
    nucleus = _nucleus(logits, p)
    assert int(nucleus[3:].sum(dim=-1).min()) > TTTopKTopPSampler.TOP_P_CANDIDATES
    tokens, _ = TTTopKTopPSampler()(logits.clone(), {}, None, p)
    assert nucleus[torch.arange(6), tokens].all()
