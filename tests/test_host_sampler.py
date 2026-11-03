# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

from vllm_tt_plugin import host_sampler
from vllm_tt_plugin.host_sampler import TTTopKTopPSampler


@pytest.mark.parametrize("batch", [1, 4, 7, 32])
@pytest.mark.parametrize("fp64", [False, True])
@pytest.mark.parametrize(
    "mode", ["raw_logprobs", "processed_logits", "processed_logprobs"]
)
@pytest.mark.parametrize("filters", [False, True])
def test_parallel_sampling_preserves_tokens_scores_and_rng(batch, fp64, mode, filters):
    torch.manual_seed(197)
    logits = torch.randn(batch, 4101)
    logits[:, 0] = -torch.inf
    native = TopKTopPSampler(mode, fp64)
    parallel = TTTopKTopPSampler(mode, fp64)
    seeds = {i: torch.Generator().manual_seed(123 + i) for i in range(batch)}
    copies = {
        i: torch.Generator().set_state(gen.get_state()) for i, gen in seeds.items()
    }
    k = torch.full((batch,), 80, dtype=torch.int32) if filters else None
    p = torch.full((batch,), 0.9) if filters else None
    global_state = torch.get_rng_state().clone()
    try:
        for _ in range(3):
            expected, expected_scores = native.forward_native(
                logits.clone(), seeds, k, p
            )
            actual, actual_scores = parallel(logits.clone(), copies, k, p)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if expected_scores is None:
                assert actual_scores is None
            else:
                torch.testing.assert_close(
                    actual_scores, expected_scores, rtol=0, atol=0
                )
            assert all(
                torch.equal(seeds[i].get_state(), copies[i].get_state()) for i in seeds
            )
            assert torch.equal(torch.get_rng_state(), global_state)
        assert parallel._sampling_pool is not None
        assert parallel._noise_cache
    finally:
        if parallel._sampling_pool is not None:
            parallel._sampling_pool.shutdown()


@pytest.mark.parametrize("kind", ["small_unseeded", "partial", "shared"])
def test_fallback_keeps_global_and_request_rng_state(kind):
    torch.manual_seed(9)
    logits = torch.randn(8, 1024 if kind == "small_unseeded" else 4101)
    native = TopKTopPSampler()
    parallel = TTTopKTopPSampler()
    generators = {i: torch.Generator().manual_seed(20 + i) for i in range(8)}
    if kind == "small_unseeded":
        generators = {}
    elif kind == "partial":
        generators = {i: gen for i, gen in generators.items() if i % 2}
    else:
        generators = dict.fromkeys(range(8), generators[0])
    initial = {i: gen.get_state().clone() for i, gen in generators.items()}
    global_initial = torch.get_rng_state().clone()
    expected, _ = native.forward_native(logits.clone(), generators, None, None)
    final = {i: gen.get_state().clone() for i, gen in generators.items()}
    global_final = torch.get_rng_state().clone()
    for i, gen in generators.items():
        gen.set_state(initial[i])
    torch.set_rng_state(global_initial)
    actual, _ = parallel(logits.clone(), generators, None, None)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), global_final)
    assert all(torch.equal(gen.get_state(), final[i]) for i, gen in generators.items())
    assert parallel._sampling_pool is None


@pytest.mark.parametrize(
    "mode", ["raw_logprobs", "processed_logits", "processed_logprobs"]
)
def test_unseeded_categorical_preserves_filter_and_scores(monkeypatch, mode):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    # Exercise the full-vocabulary path; candidate sampling is tested below.
    monkeypatch.setattr(host_sampler, "TOP_P_CANDIDATES", 4101)

    logits = torch.randn(4, 4101)
    k = torch.tensor([1, 20, 100, 4101], dtype=torch.int32)
    p = torch.tensor([0.0, 0.5, 0.95, 1.0])
    filtered = apply_top_k_top_p(logits.clone(), k, p)
    expected = filtered.softmax(-1)
    observed = []
    original = torch.multinomial

    def draw(probs, num_samples, replacement):
        torch.testing.assert_close(probs, expected, rtol=0, atol=0)
        assert num_samples == 2 and replacement is True
        result = original(probs, num_samples, replacement)
        observed.append(result)
        return result

    monkeypatch.setattr(torch, "multinomial", draw)
    tokens, scores = TTTopKTopPSampler(mode)(logits.clone(), {}, k, p)
    assert len(observed) == 1
    assert torch.equal(tokens, observed[0][:, 0])
    assert tokens.is_contiguous()
    if mode == "processed_logits":
        torch.testing.assert_close(scores, filtered, rtol=0, atol=0)
    elif mode == "processed_logprobs":
        torch.testing.assert_close(scores, filtered.log_softmax(-1), rtol=0, atol=0)
    else:
        assert scores is None


def test_unseeded_invalid_probabilities_preserve_native_rng():
    logits = torch.full((2, 4101), -torch.inf)
    state = torch.get_rng_state()
    expected, _ = TopKTopPSampler().forward_native(logits.clone(), {}, None, None)
    after = torch.get_rng_state()
    torch.set_rng_state(state)
    actual, _ = TTTopKTopPSampler()(logits.clone(), {}, None, None)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), after)


def test_unseeded_large_vocabulary_uses_native_fallback(monkeypatch):
    logits = SimpleNamespace(shape=(1, 2**24 + 1), device=torch.device("cpu"))
    expected = object()

    def native(self, values, generators, k, p):
        assert values is logits
        assert generators == {} and k is None and p is None
        return expected

    monkeypatch.setattr(TopKTopPSampler, "forward_native", native)
    assert TTTopKTopPSampler().forward_native(logits, {}, None, None) is expected


def test_unseeded_categorical_distribution_and_zero_support():
    torch.manual_seed(711)
    logits = torch.full((1024, 4096), -torch.inf)
    probabilities = torch.tensor([0.125, 0.25, 0.625])
    logits[:, [7, 109, 4095]] = probabilities.log()
    sampler = TTTopKTopPSampler()
    tokens = torch.cat([sampler(logits.clone(), {}, None, None)[0] for _ in range(10)])
    counts = torch.bincount(tokens, minlength=4096)
    assert counts[[7, 109, 4095]].sum() == tokens.numel()
    observed = counts[[7, 109, 4095]].double() / tokens.numel()
    torch.testing.assert_close(observed, probabilities.double(), rtol=0, atol=0.02)


def _candidate_probs(monkeypatch, logits, k, p):
    """Return the dense distribution that candidate sampling draws from."""
    captured = []
    original = torch.multinomial

    def draw(probs, num_samples, **kwargs):
        captured.append(probs / probs.sum(-1, keepdim=True))
        return original(probs, num_samples, **kwargs)

    monkeypatch.setattr(torch, "multinomial", draw)
    tokens, exact = host_sampler._sample_top_p_candidates(logits, k, p)
    monkeypatch.setattr(torch, "multinomial", original)
    dense = torch.zeros_like(logits)
    if captured:
        _, indices = logits.topk(host_sampler.TOP_P_CANDIDATES, dim=-1)
        rows = exact.nonzero().squeeze(1)
        dense[rows] = dense[rows].scatter(1, indices[rows], captured[0])
    return tokens, exact, dense


@pytest.mark.parametrize("top_k", [None, "vocab", 20, "mixed"])
def test_top_p_candidates_match_full_filter(monkeypatch, top_k):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    torch.manual_seed(5)
    batch, vocab = 16, 8192
    monkeypatch.setattr(host_sampler, "TOP_P_CANDIDATES", 256)
    # Peaked rows fit inside the candidates; flat rows do not.
    scale = torch.tensor([12.0] * 12 + [0.1] * 4).unsqueeze(1)
    logits = torch.randn(batch, vocab) * scale
    p = torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0, 0.7, 0.8] * 2)
    k = {
        None: None,
        "vocab": torch.full((batch,), vocab),
        20: torch.full((batch,), 20),
        "mixed": torch.tensor([vocab, 1, 20, 256] * 4),
    }[top_k]
    tokens, exact, dense = _candidate_probs(monkeypatch, logits, k, p)
    expected = apply_top_k_top_p(logits.clone(), k, p).softmax(-1)
    assert exact.any()
    torch.testing.assert_close(dense[exact], expected[exact], rtol=0, atol=1e-5)
    assert (expected[exact].gather(1, tokens[exact].unsqueeze(1)) > 0).all()
    if top_k in (None, "vocab"):
        # Rows needing more than the candidates, or p = 1, must fall back.
        assert not exact[12:].any() and not exact[5]


def test_top_p_candidates_reject_large_top_k(monkeypatch):
    monkeypatch.setattr(host_sampler, "TOP_P_CANDIDATES", 64)
    logits = torch.randn(2, 4096)
    k = torch.tensor([65, 4096])
    assert host_sampler._sample_top_p_candidates(logits, k, torch.ones(2)) is None


def test_unseeded_top_p_falls_back_per_row(monkeypatch):
    torch.manual_seed(9)
    monkeypatch.setattr(host_sampler, "TOP_P_CANDIDATES", 64)
    logits = torch.full((2, 4096), -torch.inf)
    # Row 0 fits in the candidates; row 1 spreads mass over 1024 tokens.
    logits[0, [3, 77]] = torch.tensor([0.75, 0.25]).log()
    logits[1, 1000:2024] = 0.0
    p = torch.tensor([0.95, 0.95])
    full = []
    original = TTTopKTopPSampler._sample_unseeded_full

    def spy(self, values, k, p):
        full.append(values.shape[0])
        return original(self, values, k, p)

    monkeypatch.setattr(TTTopKTopPSampler, "_sample_unseeded_full", spy)
    sampler = TTTopKTopPSampler()
    for _ in range(50):
        tokens, scores = sampler(logits.clone(), {}, None, p)
        assert scores is None
        assert tokens[0].item() in (3, 77)
        assert 1000 <= tokens[1].item() < 2024
    assert full == [1] * 50
