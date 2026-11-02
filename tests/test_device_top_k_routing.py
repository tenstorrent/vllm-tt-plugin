# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Route bounded device samplers without changing requested distributions."""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.config import validate_tt_device_top_k_unbounded_as
from vllm_tt_plugin.input_batch import SamplingInputBatch
from vllm_tt_plugin.model_runner import TTModelRunner


@pytest.mark.parametrize("unbounded_as", [None, 32])
@pytest.mark.parametrize("cap", [None, 32])
@pytest.mark.parametrize("top_k", [-1, 0, 1, 20, 32, 33, 64])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_top_k_routes_only_active_sampling_rows(cap, top_k, temperature, unbounded_as):
    sampling = SamplingInputBatch(4)
    # Slot gaps and stale rows must not force an otherwise supported batch
    # onto the host. Lane scheduling can leave active rows noncontiguous.
    sampling.top_k[:] = 1000
    sampling.temperature[:] = 1.0
    sampling.top_k[[0, 3]] = top_k
    sampling.temperature[[0, 3]] = temperature
    runner = SimpleNamespace(
        sample_on_device_mode="all",
        seeded_sampling_policy="auto",
        num_devices=4,
        tt_data_parallel_size=1,
        model=SimpleNamespace(
            model_capabilities=(
                {}
                if cap is None
                else {
                    "max_device_top_k": cap,
                    **(
                        {}
                        if unbounded_as is None
                        else {"device_top_k_unbounded_as": unbounded_as}
                    ),
                }
            )
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
    # the opt-in serves rows without a top-k bound (and above the bound) from
    # the model's top-``unbounded_as`` candidates instead of the host
    expected = (
        cap is None
        or temperature == 0
        or 1 <= top_k <= cap
        or (unbounded_as is not None and (top_k < 1 or top_k > cap))
    )
    assert TTModelRunner.check_perform_device_sampling(runner, True, False) is expected


def test_platform_validates_the_unbounded_top_k_opt_in():
    validate_tt_device_top_k_unbounded_as({"max_device_top_k": 32}, object)
    validate_tt_device_top_k_unbounded_as(
        {"max_device_top_k": 32, "device_top_k_unbounded_as": 32}, object
    )
    validate_tt_device_top_k_unbounded_as(
        {"max_device_top_k": 32, "device_top_k_unbounded_as": None}, object
    )
    validate_tt_device_top_k_unbounded_as(None, object)
    with pytest.raises(ValueError, match="without max_device_top_k"):
        validate_tt_device_top_k_unbounded_as({"device_top_k_unbounded_as": 32}, object)
    for bad in (0, -1, 33, True, 1.5, "32"):
        with pytest.raises(ValueError, match=r"integer in 1\.\.32"):
            validate_tt_device_top_k_unbounded_as(
                {"max_device_top_k": 32, "device_top_k_unbounded_as": bad}, object
            )


@pytest.mark.parametrize("supports_device_sampler", [True, False])
def test_loaded_model_without_device_sampler_uses_host(supports_device_sampler):
    runner = SimpleNamespace(
        sample_on_device_mode="decode_only",
        seeded_sampling_policy="auto",
        model=SimpleNamespace(
            model_capabilities={"supports_sample_on_device": supports_device_sampler}
        ),
        num_devices=2,
        tt_data_parallel_size=1,
        model_config=SimpleNamespace(logits_processors=[]),
        input_batch=SimpleNamespace(
            sampling=SamplingInputBatch(1),
            req_id_to_index={"request": 0},
            no_penalties=True,
            no_allowed_token_ids=True,
            max_num_logprobs=None,
        ),
    )

    assert (
        TTModelRunner.check_perform_device_sampling(runner, True, False)
        is supports_device_sampler
    )
