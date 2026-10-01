# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

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
    finally:
        if parallel._sampling_pool is not None:
            parallel._sampling_pool.shutdown()


@pytest.mark.parametrize("kind", ["unseeded", "partial", "shared"])
def test_fallback_keeps_global_and_request_rng_state(kind):
    torch.manual_seed(9)
    logits = torch.randn(8, 4101)
    native = TopKTopPSampler()
    parallel = TTTopKTopPSampler()
    generators = {i: torch.Generator().manual_seed(20 + i) for i in range(8)}
    if kind == "unseeded":
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
