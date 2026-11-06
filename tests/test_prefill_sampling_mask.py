# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PC-02 routing, submission and extraction without device execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput

from tests.test_lane_input_batch import (
    _add_to_lane,
    _lane_batch,
    _make_req,
    _without_speculation,
)
from tests.test_model_runner import _batch_with_one_request, _fake_runner, _prepare
from vllm_tt_plugin.lane_scheduler import TTStepPlan
from vllm_tt_plugin.model_runner import TTModelRunner


@pytest.mark.parametrize("lane", [False, True])
@pytest.mark.parametrize("capability", [None, False, True])
@pytest.mark.parametrize("eligible", [False, True])
@pytest.mark.parametrize(
    "final_rows", [(True, True), (False, False), (False, True), (True, False)]
)
def test_prefill_mask_route_and_submission(lane, capability, eligible, final_rows):
    """Real builders must agree with the adapter on token vs logits routing."""
    caps = (
        {}
        if capability is None
        else {"supports_device_prefill_sampling_mask": capability}
    )
    captured = {}

    def prefill_forward(**kwargs):
        captured.update(kwargs)
        if "sampling_params" in kwargs:
            # Host stand-in for a device sampler. Ignored rows deliberately
            # contain invalid token IDs, so final-row order must be preserved.
            return torch.tensor(
                [[51 if final_rows[0] else -999], [52 if final_rows[1] else -999]]
            )
        return torch.zeros((2, 1, 64))

    model = SimpleNamespace(model_capabilities=caps, prefill_forward=prefill_forward)
    if lane:
        batch = _lane_batch(num_lanes=2, per_lane=3, with_custom=False)
        # Occupy the first slot in each lane, leaving submitted slots 1 and 4.
        for rank in range(2):
            _add_to_lane(
                batch, _make_req(f"live-{rank}", [1], [], dict(temperature=0.0)), rank
            )
        requests = {}
        rows = []
        for index, rank in enumerate((1, 0)):
            req = _make_req(
                f"r{index}",
                list(range(4 if final_rows[index] else 8)),
                [],
                dict(temperature=0.0),
            )
            requests[req.req_id] = req
            row = _add_to_lane(batch, req, rank)
            batch.num_computed_tokens_cpu[row] = 2
            rows.append(row)
        scheduler = SchedulerOutput.make_empty()
        scheduler.num_scheduled_tokens = {name: 2 for name in requests}
        scheduler.total_num_scheduled_tokens = 4
        plan = TTStepPlan(
            is_decode=False,
            capacity=6,
            scheduled_req_ids=tuple(requests),
            scheduled_rows=tuple(rows),
            input_rows=tuple(rows),
            req_id_to_row=dict(zip(requests, rows)),
            batch_size_per_dp=(1, 1),
            prefill_empty_slots=tuple(rows),
        )
        runner = _without_speculation(
            SimpleNamespace(
                max_num_blocks_per_req=16,
                _block_tables_per_layer=lambda _: None,
                check_perform_device_sampling=lambda **_: eligible,
                _output_tokens_per_step=1,
                model_config=SimpleNamespace(is_multimodal_model=False),
                requests=requests,
                model=model,
            )
        )
        model_input = batch.build_model_input(runner, scheduler, None, plan)
        slots = [4, 1]
    else:
        batch, request = _batch_with_one_request(4 if final_rows[0] else 8, 0, 2)
        _, other = _batch_with_one_request(4 if final_rows[1] else 8, 0, 2)
        other.req_id = "other"
        batch.add_request(other)
        runner = _fake_runner(batch, request)
        runner.requests[other.req_id] = other
        runner.model = model
        runner.check_perform_device_sampling = lambda **_: eligible
        slots = [3, 1]
        runner._alloc_prefill_state_slots = lambda _: slots
        model_input = _prepare(runner, ("r", 2, 2, 0), ("other", 2, 2, 0))

    expected_device = eligible and (all(final_rows) or capability is True)
    assert model_input.perform_device_sampling is expected_device
    assert model_input.intermediate_prefill_mask.tolist() == [not x for x in final_rows]
    runner.kv_caches = object()
    runner.trace_mode = "all"
    runner.request_specific_rope = False
    runner.async_decode = SimpleNamespace(note_prefill_submitted=lambda: None)
    output = TTModelRunner.submit_prefill(runner, model_input, [1, 1])
    assert captured["empty_slots"] == slots
    assert ("sampling_params" in captured) is expected_device
    if expected_device and capability is True:
        mask = captured["prefill_sample_mask"]
        assert mask.device.type == "cpu" and mask.dtype == torch.bool
        assert mask.tolist() == list(final_rows)
    else:
        assert "prefill_sample_mask" not in captured

    if expected_device:

        def forbidden(*args, **kwargs):
            raise AssertionError("device prefill must not read logits or host-sample")

        runner.host_sampler = forbidden
        runner._check_host_logits = forbidden
        if lane:
            sampled, logprobs = batch.extract_output(
                runner, output, None, model_input, rows, False
            )
            assert logprobs is None
        else:
            runner._is_block_output_model = False
            runner._is_adaptive_block_output = False
            for segments in ([2], [1, 1], [0, 1, 1]):
                # Both a mixed rank and concatenated DP output, including an
                # empty rank, must retain the submitted row offsets.
                extraction_input = replace(
                    model_input,
                    grammar_bitmask=[None] * len(segments),
                    max_num_logprobs=[None] * len(segments),
                )
                sampled_per_dp, logprobs = TTModelRunner._get_output_tokens(
                    runner,
                    output,
                    None,
                    model_input.tt_sampling_params,
                    extraction_input,
                    segments,
                    True,
                    False,
                )
                sampled = torch.cat(sampled_per_dp)
                assert logprobs == [None] * len(segments)
                for index, final in enumerate(final_rows):
                    if final:
                        assert sampled[index].item() == 51 + index
        for index, final in enumerate(final_rows):
            if final:
                assert sampled[index].item() == 51 + index
        if not any(final_rows):
            assert sampled.tolist() == [[0], [0]]
