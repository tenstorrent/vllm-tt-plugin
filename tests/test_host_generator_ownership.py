# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Input preparation leaves seeded randomness to the host sampler."""

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import random_sample

from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner


@pytest.mark.parametrize("seed", [0, 1, 42, 2**60 + 7])
@pytest.mark.parametrize("row", [0, 3])
def test_host_sampling_matches_direct_request_generator(seed, row):
    generator = torch.Generator().manual_seed(seed)
    reference = torch.Generator().manual_seed(seed)
    unscheduled = torch.Generator().manual_seed(99)
    unscheduled_before = unscheduled.get_state().clone()
    batch = InputBatch(
        max_num_reqs=8,
        max_model_len=16,
        max_num_batched_tokens=16,
        vocab_size=128,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )
    batch.sampling.generators = {row: generator, row + 1: unscheduled}
    probs = torch.full((1, 128), 1 / 128)

    for _ in range(8):
        before = generator.get_state().clone()
        prepared = TTModelRunner._build_host_generators(batch, [row], None)
        assert prepared[0] is generator
        assert torch.equal(generator.get_state(), before)
        actual = random_sample(probs.clone(), prepared)
        expected = random_sample(probs.clone(), {0: reference})
        assert torch.equal(actual, expected)
        assert torch.equal(generator.get_state(), reference.get_state())

    assert torch.equal(unscheduled.get_state(), unscheduled_before)
