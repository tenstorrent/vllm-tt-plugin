# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest
import torch
from vllm.v1.sample.sampler import Sampler

from tests.test_lane_input_batch import _lane_batch, _make_req
from vllm_tt_plugin.host_sampler import TTHostSampler


@pytest.mark.parametrize("temperatures", [[1.0, 1.0], [0.0, 1.0], [0.7, 1.0]])
@pytest.mark.parametrize(
    "mode", ["raw_logits", "raw_logprobs", "processed_logits", "processed_logprobs"]
)
def test_fp32_sampling_preserves_tokens_scores_rng_and_raw_logits(temperatures, mode):
    batch = _lane_batch(num_lanes=1, per_lane=2, with_custom=False)
    for row, temp in enumerate(temperatures):
        batch.add_request_to_row(
            _make_req(
                str(row), [1], [], dict(temperature=temp, logprobs=3), seed=42 + row
            ),
            row,
        )
    batch.refresh_logitsprocs()
    metadata = batch.build_merged_sampling_metadata([0, 1], compact=True)
    logits = torch.linspace(-3, 3, 128).reshape(2, 64).to(torch.bfloat16)
    initial = {row: gen.get_state() for row, gen in metadata.generators.items()}
    expected = Sampler(logprobs_mode=mode)(logits.clone(), metadata)
    final = {row: gen.get_state() for row, gen in metadata.generators.items()}
    for row, gen in metadata.generators.items():
        gen.set_state(initial[row])
    owned = logits.float()
    actual = TTHostSampler(logprobs_mode=mode)(owned, metadata)
    assert torch.equal(actual.sampled_token_ids, expected.sampled_token_ids)
    for field in ("logprobs", "logprob_token_ids", "selected_token_ranks"):
        torch.testing.assert_close(
            getattr(actual.logprobs_tensors, field),
            getattr(expected.logprobs_tensors, field),
            rtol=0,
            atol=0,
        )
    assert all(
        torch.equal(gen.get_state(), final[row])
        for row, gen in metadata.generators.items()
    )
    scores = actual.logprobs_tensors.logprobs.clone()
    owned.zero_()
    assert torch.equal(actual.logprobs_tensors.logprobs, scores)


@pytest.mark.parametrize("all_random", [False, True])
@pytest.mark.parametrize("temps", [[1.0, 1.0], [0.0, 1.0], [1.0, 0.8]])
def test_temperature_matches_native_for_special_values(temps, all_random):
    values = torch.tensor([[0.0, -0.0, float("inf")], [-float("inf"), 4.0, -4.0]])
    temp = torch.tensor(temps)
    expected = Sampler.apply_temperature(values.clone(), temp, all_random)
    actual = TTHostSampler.apply_temperature(values, temp, all_random)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


def test_unit_temperature_does_not_write_logits():
    logits = torch.randn(3, 7)
    version = logits._version
    assert TTHostSampler.apply_temperature(logits, torch.ones(3), True) is logits
    assert logits._version == version
