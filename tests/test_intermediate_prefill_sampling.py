# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Capability-gated intermediate sampling routing, without device execution."""

from types import SimpleNamespace

import pytest
import torch
from vllm.utils import torch_utils
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample import sampler as sampler_module
from vllm.v1.sample.ops import penalties as penalties_module

from tests.test_lane_input_batch import (
    BLOCK,
    MAX_MODEL_LEN,
    _add_to_lane,
    _lane_batch,
    _make_req,
)
from tests.test_model_runner import _batch_with_one_request, _fake_runner, _prepare
from vllm_tt_plugin.lane_scheduler import TTStepPlan

CAPABILITY = "supports_intermediate_prefill_device_sampling"


@pytest.fixture(autouse=True)
def unpinned_cpu(monkeypatch):
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    monkeypatch.setattr(sampler_module, "PIN_MEMORY", False)
    monkeypatch.setattr(penalties_module, "PIN_MEMORY", False)


def model(capability):
    return SimpleNamespace(
        model_capabilities={} if capability is None else {CAPABILITY: capability}
    )


@pytest.mark.parametrize("capability", [None, False, True])
@pytest.mark.parametrize("eligible", [False, True])
@pytest.mark.parametrize("intermediate", [False, True])
def test_ordinary_prefill_sampling_preserves_existing_denials(
    capability, eligible, intermediate
):
    computed = 2 if intermediate else 7
    batch, request = _batch_with_one_request(8, 0, computed)
    runner = _fake_runner(batch, request)
    runner.model = model(capability)
    runner.check_perform_device_sampling = lambda **_: eligible

    result = _prepare(runner, ("r", 1, computed, 0))

    assert result.intermediate_prefill_mask.tolist() == [intermediate]
    assert result.perform_device_sampling is (
        eligible and (not intermediate or capability is True)
    )


def lane_case(capability, eligible, mixed):
    batch = _lane_batch(num_lanes=1, per_lane=2, with_custom=False)
    requests = {}
    rows = []
    for req_id, computed in [("continued", 2), ("final", 7)]:
        request = _make_req(req_id, list(range(8)), [], dict(temperature=0.0))
        row = _add_to_lane(batch, request, lane=0)
        requests[req_id] = request
        rows.append(row)
        batch.num_computed_tokens_cpu[row] = computed if mixed else 2
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = dict.fromkeys(requests, 1)
    output.total_num_scheduled_tokens = len(requests)
    plan = TTStepPlan(
        is_decode=False,
        capacity=2,
        scheduled_req_ids=tuple(requests),
        scheduled_rows=tuple(rows),
        input_rows=tuple(rows),
        req_id_to_row=dict(zip(requests, rows)),
        batch_size_per_dp=(2,),
        prefill_empty_slots=tuple(rows),
    )
    runner = SimpleNamespace(
        model=model(capability),
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK,
        _block_tables_per_layer=lambda _: None,
        check_perform_device_sampling=lambda **_: eligible,
        model_config=SimpleNamespace(is_multimodal_model=False),
        requests=requests,
    )
    result = batch.build_model_input(runner, output, None, plan)
    return batch, runner, result, rows


@pytest.mark.parametrize("capability", [None, False, True])
@pytest.mark.parametrize("eligible", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_lane_prefill_sampling_preserves_existing_denials(capability, eligible, mixed):
    _, _, result, _ = lane_case(capability, eligible, mixed)
    assert result.intermediate_prefill_mask.tolist() == [True, not mixed]
    assert result.perform_device_sampling is (eligible and capability is True)


def test_lane_mixed_device_output_keeps_final_row_and_intermediate_mask():
    batch, runner, result, rows = lane_case(True, True, True)
    tokens, logprobs = batch.extract_output(
        runner, torch.tensor([29, 41]), None, result, rows, is_decode=False
    )
    assert tokens.tolist() == [[29], [41]]
    assert logprobs is None
    # Output extraction preserves row order. The existing chunked output
    # builder uses this mask to discard29 and commit only final-row token41.
    assert result.intermediate_prefill_mask.tolist() == [True, False]


@pytest.mark.parametrize("capability", [None, False])
def test_lane_mixed_device_output_rejects_undeclared_model(capability):
    batch, runner, result, rows = lane_case(True, True, True)
    # Model input and extraction must agree on the capability; reject an
    # inconsistent device-sampled payload instead of silently advancing RNG.
    runner.model = model(capability)
    with pytest.raises(ValueError, match=CAPABILITY):
        batch.extract_output(
            runner, torch.tensor([29, 41]), None, result, rows, is_decode=False
        )


def test_lane_all_intermediate_device_outputs_are_discarded():
    batch, runner, result, rows = lane_case(True, True, False)
    tokens, logprobs = batch.extract_output(
        runner, torch.tensor([29, 41]), None, result, rows, is_decode=False
    )
    assert tokens.tolist() == [[0], [0]]
    assert logprobs is None
