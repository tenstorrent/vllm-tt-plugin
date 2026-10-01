# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

from vllm_tt_plugin.host_sampler import TTTopKTopPSampler


@pytest.mark.parametrize("change", ["none", "reseed", "reorder", "vocab", "eviction"])
def test_prefetch_preserves_live_rng_through_lifecycle_changes(change):
    native = TopKTopPSampler()
    sampler = TTTopKTopPSampler()
    generators = {i: torch.Generator().manual_seed(31 + i) for i in range(4)}
    reference = {i: torch.Generator().manual_seed(31 + i) for i in range(4)}
    if change == "eviction":
        sampler._max_prefetch_bytes = 4096
    try:
        for step in range(3):
            if step == 1 and change == "reseed":
                generators[2].manual_seed(991)
                reference[2].manual_seed(991)
            if step == 1 and change == "reorder":
                generators = {i: generators[3 - i] for i in range(4)}
                reference = {i: reference[3 - i] for i in range(4)}
            vocab = 4133 if step and change == "vocab" else 4096
            logits = torch.randn(4, vocab)
            expected, _ = native.forward_native(logits.clone(), reference, None, None)
            actual, _ = sampler(logits.clone(), generators, None, None)
            assert torch.equal(actual, expected)
            for entry in sampler._noise_cache.values():
                entry.future.result()
            # Even completed speculative work must not advance live RNG state.
            assert all(
                torch.equal(generators[i].get_state(), reference[i].get_state())
                for i in range(4)
            )
            assert sampler._cached_bytes <= sampler._max_prefetch_bytes
            if change == "eviction":
                assert not sampler._noise_cache
    finally:
        if sampler._sampling_pool is not None:
            sampler._sampling_pool.shutdown()


def test_fallback_invalidates_prefetched_state_without_sampling_twice():
    sampler = TTTopKTopPSampler()
    native = TopKTopPSampler()
    seeded = torch.Generator().manual_seed(79)
    reference = torch.Generator().manual_seed(79)
    try:
        for batch in [1, 2, 1]:
            logits = torch.randn(batch, 4096)
            initial_global = torch.get_rng_state()
            expected, _ = native.forward_native(
                logits.clone(), {0: reference}, None, None
            )
            final_global = torch.get_rng_state()
            torch.set_rng_state(initial_global)
            actual, _ = sampler(logits.clone(), {0: seeded}, None, None)
            assert torch.equal(actual, expected)
            assert torch.equal(torch.get_rng_state(), final_global)
            assert torch.equal(seeded.get_state(), reference.get_state())
    finally:
        if sampler._sampling_pool is not None:
            sampler._sampling_pool.shutdown()


def test_prefetch_evicts_idle_generators_without_advancing_them():
    sampler = TTTopKTopPSampler()
    sampler._max_prefetch_bytes = 2 * 4096 * 4
    native = TopKTopPSampler()
    generators = [torch.Generator().manual_seed(100 + i) for i in range(3)]
    references = [torch.Generator().manual_seed(100 + i) for i in range(3)]
    logits = torch.randn(1, 4096)
    try:
        for request in [0, 1, 2, 0, 2, 1]:
            expected, _ = native.forward_native(
                logits.clone(), {0: references[request]}, None, None
            )
            actual, _ = sampler(logits.clone(), {0: generators[request]}, None, None)
            assert torch.equal(actual, expected)
            for entry in sampler._noise_cache.values():
                entry.future.result()
            assert len(sampler._noise_cache) <= 2
            assert sampler._cached_bytes <= sampler._max_prefetch_bytes
            assert all(
                torch.equal(actual.get_state(), reference.get_state())
                for actual, reference in zip(generators, references)
            )
    finally:
        if sampler._sampling_pool is not None:
            sampler._sampling_pool.shutdown()
