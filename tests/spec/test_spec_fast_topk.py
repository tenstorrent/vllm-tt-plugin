# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The fast top-k/top-p path of _constrained_probs equals the reference path."""

import os

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
    # The reference breaks a top-p cut inside a tie group by its unstable sort;
    # these comparisons pin the fast path to it, so they ask for that choice.
    monkeypatch.setenv("VLLM_TT_SPEC_TIE_EXACT", "1")
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
        logits,
        torch.ones(2),
        torch.tensor([20, 20], dtype=torch.int32),
        torch.tensor([0.9, 0.9]),
    )
    assert torch.equal(before, logits)


# region The support-only walk (point-mass draft)

W = 8
N = 200_000
CONTROLS = {
    "dense": dict(top_k=None, top_p=None),
    "top-k": dict(top_k=5, top_p=None),
    "top-k-top-p": dict(top_k=6, top_p=0.9),
    "top-p-only": dict(top_k=None, top_p=0.9),
}


def _target(controls, temperature=0.8):
    """The analytic target for one row: the reference path's probabilities."""
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.2, 0.0, -1.0, -3.0])
    k, p = controls["top_k"], controls["top_p"]
    old = os.environ.get("VLLM_TT_SPEC_FAST_TOPK")
    os.environ["VLLM_TT_SPEC_FAST_TOPK"] = "0"
    try:
        probs = _constrained_probs(
            logits.view(1, 1, W),
            torch.tensor([temperature]),
            None if k is None else torch.tensor([k], dtype=torch.int32),
            None if p is None else torch.tensor([p]),
        )
    finally:
        if old is None:
            os.environ.pop("VLLM_TT_SPEC_FAST_TOPK")
        else:
            os.environ["VLLM_TT_SPEC_FAST_TOPK"] = old
    return logits, probs[0, 0]


def _controls_tensors(controls, rows):
    k, p = controls["top_k"], controls["top_p"]
    return (
        None if k is None else torch.full((rows,), k, dtype=torch.int32),
        None if p is None else torch.full((rows,), p),
    )


def _assert_distribution(samples, expected, sigmas=4.5):
    count = samples.numel()
    freq = torch.bincount(samples.to(torch.int64), minlength=W).double() / count
    sigma = (expected.double() * (1 - expected.double()) / count).sqrt().clamp_min(1e-9)
    z = (freq - expected.double()).abs() / sigma
    assert float(z.max()) < sigmas, (freq.tolist(), expected.tolist())
    # Mass outside the support is exactly zero, not merely small.
    assert float(freq[expected == 0].sum()) == 0.0


@pytest.mark.parametrize("name", list(CONTROLS))
def test_a_draftless_draw_follows_the_target(monkeypatch, name):
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "1")
    torch.manual_seed(1)
    controls = CONTROLS[name]
    logits, expected = _target(controls)
    top_k, top_p = _controls_tensors(controls, N)
    result = accept_sampled_drafts(
        logits.view(1, 1, W).expand(N, 2, W).contiguous(),
        torch.zeros(N, 1, dtype=torch.int32),
        torch.zeros(N, dtype=torch.int32),
        SpecSamplingInputs(
            vocab_size=W, temperature=torch.full((N,), 0.8), top_k=top_k, top_p=top_p
        ),
    )
    _assert_distribution(result.committed_token_ids[:, 0], expected)


@pytest.mark.parametrize("name", list(CONTROLS))
def test_the_residual_draw_is_the_target_without_the_draft(monkeypatch, name):
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "1")
    torch.manual_seed(2)
    controls = CONTROLS[name]
    logits, expected = _target(controls)
    draft = int(expected.argmax())
    top_k, top_p = _controls_tensors(controls, N)
    result = accept_sampled_drafts(
        logits.view(1, 1, W).expand(N, 2, W).contiguous(),
        torch.full((N, 1), draft, dtype=torch.int32),
        torch.ones(N, dtype=torch.int32),
        SpecSamplingInputs(
            vocab_size=W, temperature=torch.full((N,), 0.8), top_k=top_k, top_p=top_p
        ),
    )
    rejected = result.accepted_counts == 1
    # Accepted with probability p(draft).
    accept_rate = 1.0 - float(rejected.double().mean())
    sigma = (float(expected[draft]) * (1 - float(expected[draft])) / N) ** 0.5
    assert abs(accept_rate - float(expected[draft])) < 4.5 * sigma
    residual = expected.clone()
    residual[draft] = 0.0
    residual /= residual.sum()
    _assert_distribution(result.committed_token_ids[rejected, 0], residual)


@pytest.mark.parametrize("name", list(CONTROLS))
def test_the_committed_tokens_follow_the_target_end_to_end(monkeypatch, name):
    """Speculative sampling with a point-mass draft is lossless."""
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "1")
    torch.manual_seed(3)
    controls = CONTROLS[name]
    logits, expected = _target(controls)
    top_k, top_p = _controls_tensors(controls, N)
    # Drafts come from an unrelated distribution, some outside the support.
    drafts = torch.randint(0, W, (N, 1), dtype=torch.int32)
    result = accept_sampled_drafts(
        logits.view(1, 1, W).expand(N, 2, W).contiguous(),
        drafts,
        torch.ones(N, dtype=torch.int32),
        SpecSamplingInputs(
            vocab_size=W, temperature=torch.full((N,), 0.8), top_k=top_k, top_p=top_p
        ),
    )
    _assert_distribution(result.committed_token_ids[:, 0], expected)
    # The bonus after a full accept is a draw from the target at column 1.
    full = result.accepted_counts == 2
    assert int(full.sum()) > 1000
    _assert_distribution(result.committed_token_ids[full, 1], expected)


def _seeded_batch(composition, seed=5):
    """Row 0 is the row under test; the others vary between runs."""
    g = torch.Generator().manual_seed(77)
    logits = torch.randn(1, 4, V, generator=g) * 3
    drafts = torch.tensor([[int(logits[0, 0].argmax()), 5, 9]], dtype=torch.int32)
    valid = torch.tensor([3], dtype=torch.int32)
    extra = len(composition)
    if extra:
        og = torch.Generator().manual_seed(1000 + extra)
        logits = torch.cat([logits, torch.randn(extra, 4, V, generator=og) * 3])
        drafts = torch.cat(
            [drafts, torch.randint(0, V, (extra, 3), generator=og, dtype=torch.int32)]
        )
        valid = torch.cat([valid, torch.tensor(composition, dtype=torch.int32)])
    rows = 1 + extra
    generators = {0: torch.Generator().manual_seed(seed)}
    for row in range(1, rows):
        if row % 2:
            generators[row] = torch.Generator().manual_seed(seed + row)
    return accept_sampled_drafts(
        logits,
        drafts,
        valid,
        SpecSamplingInputs(
            vocab_size=V,
            temperature=torch.full((rows,), 0.6),
            top_k=torch.full((rows,), 20, dtype=torch.int32),
            top_p=torch.full((rows,), 0.95),
            generators=generators,
        ),
    )


def test_a_seeded_row_is_deterministic_and_independent_of_other_rows(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "1")
    seen = set()
    for seed in range(30):
        alone = _seeded_batch([], seed)
        again = _seeded_batch([], seed)
        assert torch.equal(alone.committed_token_ids, again.committed_token_ids)
        for composition in ([0], [3, 1], [2, 3, 0, 3, 1]):
            torch.manual_seed(seed + 99)  # unseeded rows draw differently
            batch = _seeded_batch(composition, seed)
            assert torch.equal(
                batch.committed_token_ids[0], alone.committed_token_ids[0]
            )
            assert int(batch.accepted_counts[0]) == int(alone.accepted_counts[0])
        seen.add(tuple(alone.committed_token_ids[0].tolist()))
    assert len(seen) > 5  # the seed matters


# endregion


def _no_reference(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a tie-cut row took the full-vocabulary sort")

    monkeypatch.setattr(spec_accept, "_reference_top_k_top_p", refuse)


def test_tie_cut_keeps_lowest_ids_without_a_full_sort(monkeypatch):
    # Six equal logits at the top, and a top-p that cuts inside them.
    logits = torch.randn(1, 1, V, generator=torch.Generator().manual_seed(3))
    tied = torch.tensor([900, 7, 3000, 41, 2, 77])
    logits[0, 0, tied] = 8.0
    temperature = torch.ones(1)
    top_k = torch.tensor([20], dtype=torch.int32)
    top_p = torch.tensor([0.5])
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "0")
    ref = _constrained_probs(logits, temperature, top_k, top_p)
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_TIE_EXACT", "0")
    _no_reference(monkeypatch)
    fast = _constrained_probs(logits, temperature, top_k, top_p)
    kept = (fast[0, 0] > 0).nonzero().squeeze(1).tolist()
    # The cut kept as many members as the reference did, the lowest ids.
    n = int((ref > 0).sum())
    assert 1 <= n < 6
    assert kept == sorted(tied.tolist())[:n]
    assert torch.allclose(fast.sum(), torch.tensor(1.0))
    assert torch.allclose(fast[fast > 0].sort().values, ref[ref > 0].sort().values)


@pytest.mark.parametrize("seed", range(10))
def test_bf16_rows_need_no_full_sort_and_keep_the_reference_masses(monkeypatch, seed):
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(4, 4, 50000, generator=g) * 3).to(torch.bfloat16)
    logits[..., :40] += torch.randint(5, 12, (4, 4, 40), generator=g).to(torch.bfloat16)
    temperature = torch.tensor([0.6, 1.0, 0.6, 1.0])
    top_k = torch.full((4,), 20, dtype=torch.int32)
    top_p = torch.full((4,), 0.95)
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "0")
    ref = _constrained_probs(logits, temperature, top_k, top_p)
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_TIE_EXACT", "0")
    _no_reference(monkeypatch)
    fast = _constrained_probs(logits, temperature, top_k, top_p)
    # Same probability values per position; only which equal-logit token
    # holds one can differ.
    assert torch.allclose(
        fast.sort(dim=-1).values[..., -64:],
        ref.sort(dim=-1).values[..., -64:],
        rtol=1e-6,
        atol=1e-9,
    )
    assert torch.equal((fast > 0).sum(-1), (ref > 0).sum(-1))


# region Top-p without top-k


def _count_reference_rows(monkeypatch):
    """Rows handed to the reference top-k/top-p from here on, call by call."""
    rows: list[int] = []
    real = spec_accept._reference_top_k_top_p

    def counted(flat, k, p):
        rows.append(int(flat.shape[0]))
        return real(flat, k, p)

    monkeypatch.setattr(spec_accept, "_reference_top_k_top_p", counted)
    return rows


def _top_p_both(monkeypatch, logits, temperature, top_k, top_p, min_p=None):
    """Reference and fast probabilities, and how many positions fell back."""
    ref, _ = _both(monkeypatch, logits, temperature, top_k, top_p, min_p)
    fallen = _count_reference_rows(monkeypatch)
    fast = _constrained_probs(logits, temperature, top_k, top_p, min_p)
    return ref, fast, sum(fallen)


def _peaked(g, rows, width, vocab, spread=2.0):
    """Gaussian logits with a few tens of boosted tokens, as a language model's."""
    logits = torch.randn(rows, width, vocab, generator=g) * spread
    boosted = torch.randint(0, vocab, (rows, width, 40), generator=g)
    logits.scatter_add_(2, boosted, torch.rand(rows, width, 40, generator=g) * 12)
    return logits


@pytest.mark.parametrize("kind", ["peaked", "peaked-bf16", "gaussian"])
@pytest.mark.parametrize("seed", range(12))
def test_top_p_only_rows_match_reference(monkeypatch, kind, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "gaussian":
        logits = torch.randn(4, 4, V, generator=g) * 3
    else:
        logits = _peaked(g, 4, 4, V)
    if kind.endswith("bf16"):
        logits = logits.to(torch.bfloat16).to(torch.float32)
    temperature = torch.tensor([0.6, 1.0, 1.3, 0.8])
    top_p = torch.tensor([0.95, 0.9, 0.95, 0.5])
    # top-k off: no tensor at all, or the vocabulary size, as vLLM passes it.
    for top_k in (None, torch.full((4,), V, dtype=torch.int32)):
        ref, fast, fallen = _top_p_both(monkeypatch, logits, temperature, top_k, top_p)
        _check(ref, fast)
        if kind != "peaked-bf16":  # bf16 cuts often tie, and those fall back
            # The nucleus fits in the candidates, so nearly every position
            # is served without the sort; the margin may send one back.
            assert fallen <= 2
    # The default tie order: the same probability values, served from the
    # candidates however often the cut ties.
    monkeypatch.setenv("VLLM_TT_SPEC_TIE_EXACT", "0")
    fallen = _count_reference_rows(monkeypatch)
    fast = _constrained_probs(logits, temperature, None, top_p)
    assert sum(fallen) <= 2
    assert torch.equal((fast > 0).sum(-1), (ref > 0).sum(-1))
    assert torch.allclose(
        fast.sort(dim=-1).values, ref.sort(dim=-1).values, rtol=1e-6, atol=1e-9
    )


def test_top_p_only_tie_at_the_cut(monkeypatch):
    # Six equal logits at the top, and a top-p that cuts inside them.
    logits = torch.randn(1, 1, V, generator=torch.Generator().manual_seed(3))
    tied = torch.tensor([900, 7, 3000, 41, 2, 77])
    logits[0, 0, tied] = 8.0
    temperature = torch.ones(1)
    top_p = torch.tensor([0.5])
    # Pinned to the reference's unstable sort: the row falls back.
    ref, fast, fallen = _top_p_both(monkeypatch, logits, temperature, None, top_p)
    _check(ref, fast)
    assert fallen == 1
    # The default keeps the group's lowest ids, as many as the reference keeps.
    monkeypatch.setenv("VLLM_TT_SPEC_TIE_EXACT", "0")
    _no_reference(monkeypatch)
    fast = _constrained_probs(logits, temperature, None, top_p)
    kept = (fast[0, 0] > 0).nonzero().squeeze(1).tolist()
    n = int((ref > 0).sum())
    assert 1 <= n < 6
    assert kept == sorted(tied.tolist())[:n]
    assert torch.allclose(fast[fast > 0].sort().values, ref[ref > 0].sort().values)


def test_top_p_only_tie_group_reaching_past_the_candidates_falls_back(monkeypatch):
    # The cut lands in a group of equal logits wider than the candidates, so
    # the candidates cannot show the group's lowest ids.
    monkeypatch.setenv("VLLM_TT_SPEC_TOPP_CANDIDATES", "64")
    logits = torch.randn(1, 1, V, generator=torch.Generator().manual_seed(4)) - 20
    logits[0, 0, 100:300] = 5.0
    top_p = torch.tensor([0.5])
    ref, fast, fallen = _top_p_both(monkeypatch, logits, torch.ones(1), None, top_p)
    _check(ref, fast)
    assert fallen == 1
    monkeypatch.setenv("VLLM_TT_SPEC_TIE_EXACT", "0")
    fallen = _count_reference_rows(monkeypatch)
    _constrained_probs(logits, torch.ones(1), None, top_p)
    assert sum(fallen) == 1


@pytest.mark.parametrize("bf16", [False, True])
def test_a_nucleus_wider_than_the_candidates_falls_back(monkeypatch, bf16):
    g = torch.Generator().manual_seed(5)
    # Flat rows: a 0.95 nucleus holds most of the 4096 tokens, more than 2048.
    logits = torch.randn(2, 3, V, generator=g) * 0.3
    if bf16:
        logits = logits.to(torch.bfloat16).to(torch.float32)
    top_p = torch.tensor([0.95, 0.95])
    ref, fast, fallen = _top_p_both(monkeypatch, logits, torch.ones(2), None, top_p)
    assert int((ref > 0).sum(-1).min()) > 2048
    _check(ref, fast)
    assert fallen == 6
    # Fewer candidates than a peaked row's nucleus: the same, by the override.
    monkeypatch.setenv("VLLM_TT_SPEC_TOPP_CANDIDATES", "4")
    logits = _peaked(g, 2, 3, V)
    ref, fast, fallen = _top_p_both(monkeypatch, logits, torch.ones(2), None, top_p)
    assert int((ref > 0).sum(-1).min()) > 4
    _check(ref, fast)
    assert fallen == 6


def test_top_p_only_disabled_takes_the_reference_path(monkeypatch):
    logits = _peaked(torch.Generator().manual_seed(6), 2, 2, V)
    top_p = torch.tensor([0.9, 0.9])
    monkeypatch.setenv("VLLM_TT_SPEC_TOPP_CANDIDATES", "0")
    ref, fast, fallen = _top_p_both(monkeypatch, logits, torch.ones(2), None, top_p)
    _check(ref, fast)
    assert fallen == 4


@pytest.mark.parametrize("value", ["-1", "many", "2.5"])
def test_a_malformed_candidate_count_is_refused(monkeypatch, value):
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_TOPP_CANDIDATES", value)
    with pytest.raises(ValueError, match="VLLM_TT_SPEC_TOPP_CANDIDATES"):
        _constrained_probs(
            torch.zeros(1, 1, 16), torch.ones(1), None, torch.tensor([0.9])
        )


@pytest.mark.parametrize("p", [0.999, 0.9999999, 0.99999994])
def test_top_p_close_to_one_matches_reference(monkeypatch, p):
    logits = _peaked(torch.Generator().manual_seed(7), 2, 3, V)
    top_p = torch.full((2,), p)
    ref, fast, _ = _top_p_both(monkeypatch, logits, torch.ones(2), None, top_p)
    _check(ref, fast)


def test_top_p_only_with_minus_inf_entries_matches_reference(monkeypatch):
    g = torch.Generator().manual_seed(8)
    logits = torch.randn(3, 4, V, generator=g) * 3
    # A grammar or allowlist leaves only a few tokens finite: the nucleus can
    # hold every finite token, and never a -inf one.
    allowed = torch.zeros(3, 4, V, dtype=torch.bool)
    allowed[0, :, :5] = True
    allowed[1, :, torch.randperm(V, generator=g)[:300]] = True
    allowed[2, :, :1] = True
    logits = logits.masked_fill(~allowed, float("-inf"))
    for p in (0.5, 0.95, 0.999):
        top_p = torch.full((3,), p)
        ref, fast, _ = _top_p_both(
            monkeypatch, logits, torch.full((3,), 0.7), None, top_p
        )
        _check(ref, fast)


def test_near_uniform_rows_match_reference(monkeypatch):
    g = torch.Generator().manual_seed(9)
    rows = torch.stack(
        [
            torch.zeros(V),  # every logit tied
            torch.randn(V, generator=g) * 1e-3,
            torch.randn(V, generator=g) * 1e-6,
        ]
    ).view(3, 1, V)
    top_p = torch.full((3,), 0.95)
    ref, fast, _ = _top_p_both(monkeypatch, rows, torch.ones(3), None, top_p)
    _check(ref, fast)
    # A vocabulary within the candidates: the whole row is decided by them.
    small = rows[..., :1000].contiguous()
    ref, fast, _ = _top_p_both(monkeypatch, small, torch.ones(3), None, top_p)
    _check(ref, fast)


def test_a_small_vocabulary_is_served_whole(monkeypatch):
    g = torch.Generator().manual_seed(10)
    logits = torch.randn(2, 4, 1000, generator=g) * 0.5
    top_p = torch.tensor([0.95, 0.8])
    ref, fast, fallen = _top_p_both(monkeypatch, logits, torch.ones(2), None, top_p)
    assert int((ref > 0).sum(-1).min()) > 500
    _check(ref, fast)
    assert fallen <= 1


@pytest.mark.parametrize("seed", range(10))
def test_top_k_top_p_only_and_plain_rows_mixed(monkeypatch, seed):
    g = torch.Generator().manual_seed(seed)
    rows, width = 8, 4
    logits = _peaked(g, rows, width, V)
    logits[5] = torch.randn(width, V, generator=g) * 0.3  # falls back
    temperature = torch.tensor([0.6, 1.0, 0.0, 0.8, 1.0, 1.0, 1.3, 0.7])
    top_k = torch.tensor([20, V, 20, V, V, V, 1024, V], dtype=torch.int32)
    top_p = torch.tensor([0.95, 0.95, 1.0, 0.9, 1.0, 0.95, 0.5, 0.8])
    ref, fast, _ = _top_p_both(monkeypatch, logits, temperature, top_k, top_p)
    _check(ref, fast)
    min_p = torch.tensor([0.0, 0.05, 0.0, 0.1, 0.0, 0.0, 0.0, 0.02])
    ref, fast, _ = _top_p_both(monkeypatch, logits, temperature, top_k, top_p, min_p)
    _check(ref, fast)


def _walk(monkeypatch, candidates, seed, logits, drafts, valid, top_k, top_p):
    monkeypatch.setenv("VLLM_TT_SPEC_FAST_TOPK", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_TOPP_CANDIDATES", candidates)
    rows = int(logits.shape[0])
    generators = {
        row: torch.Generator().manual_seed(seed * 100 + row)
        for row in range(rows)
        if row % 2 == 0
    }
    torch.manual_seed(seed)  # the unseeded rows' draws
    return accept_sampled_drafts(
        logits,
        drafts,
        valid,
        SpecSamplingInputs(
            vocab_size=V,
            temperature=torch.tensor([1.0, 0.6, 1.0, 1.0, 0.8, 1.0]),
            top_k=top_k,
            top_p=top_p,
            generators=generators,
        ),
    )


@pytest.mark.parametrize("seed", range(20))
def test_the_walk_draws_what_the_reference_table_draws(monkeypatch, seed):
    """Serving top-p rows from their support changes no token the walk commits.

    The support is in token id order, so an inverse-CDF draw over it visits
    tokens in the order the dense reference table does, and the same uniform
    picks the same token: seeded and unseeded rows, bonus, accept and
    correction draws alike.
    """
    g = torch.Generator().manual_seed(seed)
    logits = _peaked(g, 6, 4, V)
    # Drafts the target likes, and one it does not, so rows accept, reject
    # and correct; rows 3 and 4 carry no drafts (a draftless seeded row draws
    # its bonus by the exponential race).
    drafts = logits[:, :3].argmax(dim=-1).to(torch.int32)
    drafts[1, 1] = torch.randint(0, V, (), generator=g)
    valid = torch.tensor([3, 3, 2, 0, 0, 1], dtype=torch.int32)
    top_k = torch.tensor([V, V, V, 20, V, V], dtype=torch.int32)
    top_p = torch.tensor([0.95, 0.9, 0.8, 0.95, 0.95, 0.95])
    reference = _walk(monkeypatch, "0", seed, logits, drafts, valid, top_k, top_p)
    fallen = _count_reference_rows(monkeypatch)
    fast = _walk(monkeypatch, "2048", seed, logits, drafts, valid, top_k, top_p)
    assert sum(fallen) <= 1  # the top-p rows were served from their support
    assert torch.equal(fast.accepted_counts, reference.accepted_counts)
    assert torch.equal(fast.committed_token_ids, reference.committed_token_ids)


# endregion
