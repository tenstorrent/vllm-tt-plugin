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
    z = ((freq - expected.double()).abs() / sigma)
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
