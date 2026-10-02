# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Preserve signed 64-bit API seeds across request addition and compaction."""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_tt_plugin.input_batch import InputBatch


@pytest.mark.parametrize(
    "large_seeds",
    [(2**31, -(2**31) - 1), (2**60 + 7, -(2**60) + 19), (2**63 - 1, -(2**63))],
)
def test_large_signed_seeds_survive_request_compaction(large_seeds):
    batch = InputBatch(
        max_num_reqs=4,
        max_model_len=16,
        max_num_batched_tokens=16,
        vocab_size=128,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )
    seeds = [17, *large_seeds]
    for index, seed in enumerate(seeds):
        request = CachedRequestState(
            req_id=f"r{index}",
            prompt_token_ids=[2, 17],
            mm_features=None,
            sampling_params=SamplingParams(temperature=0.7, seed=seed),
            generator=torch.Generator().manual_seed(seed),
            block_ids=([index],),
            num_computed_tokens=2,
            output_token_ids=[],
        )
        batch.add_request(request)
    batch.remove_request("r0")
    batch.condense([0])
    for request_id, row in batch.req_id_to_index.items():
        seed = seeds[int(request_id[1:])]
        assert batch.sampling.seed[row].item() == seed
        reference = torch.Generator().manual_seed(seed)
        assert torch.equal(
            torch.rand(16, generator=batch.sampling.generators[row]),
            torch.rand(16, generator=reference),
        )
