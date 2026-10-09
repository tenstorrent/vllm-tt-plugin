# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Route bounded device samplers without changing requested distributions."""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.input_batch import SamplingInputBatch
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
