# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host routing preserves the requested distribution and original API seeds."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_tt_plugin.input_batch import InputBatch, SamplingInputBatch
from vllm_tt_plugin.model_runner import TTModelRunner


@pytest.mark.parametrize("cap", [None, 32])
@pytest.mark.parametrize("top_k", [-1, 0, 1, 20, 32, 33, 64])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_top_k_routes_only_active_sampling_rows(cap, top_k, temperature):
    sampling = SamplingInputBatch(4)
    # Slot gaps and stale rows must not force an otherwise supported batch
    # onto the host. Lane scheduling can leave active rows noncontiguous.
    sampling.top_k[:] = 1000
    sampling.temperature[:] = 1.0
    sampling.top_k[[0, 3]] = top_k
    sampling.temperature[[0, 3]] = temperature
    runner = SimpleNamespace(
        sample_on_device_mode="all",
        num_devices=4,
        tt_data_parallel_size=1,
        model=SimpleNamespace(
            model_capabilities={} if cap is None else {"max_device_top_k": cap}
        ),
        model_config=SimpleNamespace(logits_processors=[]),
        input_batch=SimpleNamespace(
            sampling=sampling,
            req_id_to_index={"first": 0, "last": 3},
            no_penalties=True,
            no_allowed_token_ids=True,
            max_num_logprobs=None,
        ),
    )
    expected = cap is None or temperature == 0 or 1 <= top_k <= cap
    assert TTModelRunner.check_perform_device_sampling(runner, True, False) is expected


def test_large_signed_seeds_survive_request_compaction():
    batch = InputBatch(
        max_num_reqs=4,
        max_model_len=16,
        max_num_batched_tokens=16,
        vocab_size=128,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )
    seeds = [17, 2**60 + 7, -(2**60) + 19]
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
