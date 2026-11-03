# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

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
