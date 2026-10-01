# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Keep an explicitly selected host route stable for seeded requests."""

import pickle
from types import SimpleNamespace

import pytest

from vllm_tt_plugin import config as tt_config
from vllm_tt_plugin.input_batch import SEED_NONE_SENTINEL, SamplingInputBatch
from vllm_tt_plugin.model_runner import TTModelRunner


def _runner(policy, *, cap=32):
    sampling = SamplingInputBatch(4)
    sampling.temperature[:] = 1.0
    sampling.top_k[:] = 5
    return SimpleNamespace(
        sample_on_device_mode="all",
        seeded_sampling_policy=policy,
        num_devices=4,
        tt_data_parallel_size=1,
        model=SimpleNamespace(
            model_capabilities={} if cap is None else {"max_device_top_k": cap}
        ),
        model_config=SimpleNamespace(logits_processors=[]),
        input_batch=SimpleNamespace(
            sampling=sampling,
            req_id_to_index={"target": 0, "companion": 3},
            no_penalties=True,
            no_allowed_token_ids=True,
            max_num_logprobs=None,
        ),
    )


@pytest.mark.parametrize("seed", [0, 42, -2, 2**40])
@pytest.mark.parametrize("is_decode", [False, True])
def test_host_policy_keeps_seeded_route_when_companions_change(seed, is_decode):
    runner = _runner("host")
    sampling = runner.input_batch.sampling
    sampling.seed[0] = seed
    for rows, companion_k in (([0], 5), ([0, 3], 5), ([0, 3], 50), ([0], 50)):
        sampling.top_k[3] = companion_k
        assert not TTModelRunner.check_perform_device_sampling(
            runner, is_decode, False, sampling_rows=rows
        )


@pytest.mark.parametrize("cap", [None, 32])
@pytest.mark.parametrize("is_decode", [False, True])
def test_auto_policy_preserves_existing_seeded_routing(cap, is_decode):
    runner = _runner("auto", cap=cap)
    runner.input_batch.sampling.seed[0] = 42
    assert TTModelRunner.check_perform_device_sampling(runner, is_decode, False)
    runner.input_batch.sampling.top_k[3] = 50
    assert TTModelRunner.check_perform_device_sampling(runner, is_decode, False) is (
        cap is None
    )


@pytest.mark.parametrize("is_decode", [False, True])
def test_host_policy_ignores_stale_and_unscheduled_seeded_rows(is_decode):
    runner = _runner("host")
    runner.input_batch.sampling.seed[[1, 2, 3]] = 42
    assert TTModelRunner.check_perform_device_sampling(
        runner, is_decode, False, sampling_rows=[0]
    )
    # Only submitted slots participate, including empty selection.
    assert TTModelRunner.check_perform_device_sampling(
        runner, is_decode, False, sampling_rows=[]
    )
    del runner.input_batch.req_id_to_index["companion"]
    assert TTModelRunner.check_perform_device_sampling(runner, is_decode, False)
    runner.input_batch.req_id_to_index["target"] = 3
    assert not TTModelRunner.check_perform_device_sampling(runner, is_decode, False)


@pytest.mark.parametrize("is_decode", [False, True])
def test_host_policy_moves_unseeded_companion_with_seeded_row(is_decode):
    runner = _runner("host", cap=None)
    sampling = runner.input_batch.sampling
    sampling.seed[0] = 42
    assert sampling.seed[3] == SEED_NONE_SENTINEL
    assert not TTModelRunner.check_perform_device_sampling(runner, is_decode, False)
    assert TTModelRunner.check_perform_device_sampling(
        runner, is_decode, False, sampling_rows=[3]
    )


@pytest.mark.parametrize("temperature,top_k", [(0.0, 5), (1.0, 1)])
def test_explicit_host_policy_includes_seeded_greedy_requests(temperature, top_k):
    runner = _runner("host")
    sampling = runner.input_batch.sampling
    sampling.seed[0] = 42
    sampling.temperature[0] = temperature
    sampling.top_k[0] = top_k
    assert not TTModelRunner.check_perform_device_sampling(runner, True, False)


@pytest.mark.parametrize("policy", ["auto", "host"])
def test_policy_survives_config_pickle(policy):
    config = SimpleNamespace(
        additional_config={"tt": {"seeded_sampling_policy": policy}}
    )
    assert (
        tt_config.get_tt_seeded_sampling_policy(pickle.loads(pickle.dumps(config)))
        == policy
    )


def test_policy_defaults_to_auto():
    assert (
        tt_config.get_tt_seeded_sampling_policy(SimpleNamespace(additional_config={}))
        == "auto"
    )


@pytest.mark.parametrize("invalid", [None, True, False, 0, "device", "HOST", [], {}])
def test_invalid_policy_is_rejected(invalid):
    config = SimpleNamespace(
        additional_config={"tt": {"seeded_sampling_policy": invalid}}
    )
    with pytest.raises(ValueError, match="seeded_sampling_policy"):
        tt_config.get_tt_seeded_sampling_policy(config)
