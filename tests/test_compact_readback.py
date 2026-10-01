# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.sample.sampler import Sampler

from tests.test_lane_input_batch import VOCAB, _lane_batch, _make_req
from vllm_tt_plugin.async_decode import TTAsyncDecodeController, TTDecodeSubmission
from vllm_tt_plugin.model_input import TTCompactedHostLogits


@pytest.mark.parametrize("capability", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("rows", [None, [6, 1]])
def test_readback_requires_capability_and_independent_scheduled_rows(
    capability, independent, rows
):
    calls = []

    def readback(value, **kwargs):
        calls.append(kwargs)
        return torch.randn(len(kwargs.get("sample_rows", range(8))), 1, VOCAB), None

    runner = SimpleNamespace(
        model=SimpleNamespace(
            model_capabilities={"supports_compact_host_logits": capability},
            process_decode_output_host=readback,
        ),
        lane_batch=SimpleNamespace(can_compact_host_sampling=lambda: independent),
    )
    controller = TTAsyncDecodeController.__new__(TTAsyncDecodeController)
    controller.runner = runner
    submission = TTDecodeSubmission(
        object(),
        None,
        [8],
        SimpleNamespace(enable_log_probs=torch.tensor([False])),
        False,
    )
    output = controller.finalize_decode(submission, sampling_rows=rows)
    compact = capability and independent and rows is not None
    assert ("sample_rows" in calls[0]) == compact
    assert isinstance(output.tt_out, TTCompactedHostLogits) == compact
    if compact:
        assert output.tt_out.rows == tuple(rows)
        assert output.tt_out.logits.shape[0] == len(rows)


def test_compact_readback_preserves_grammar_slot_alignment():
    batch = _lane_batch(num_lanes=2, per_lane=4, with_custom=False)
    for name, row in [("a", 6), ("b", 1)]:
        batch.add_request_to_row(
            _make_req(name, [2], [], dict(temperature=1.0), seed=42), row
        )
    batch.refresh_logitsprocs()
    rows = [6, 1]
    allowed = np.array([3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    seen = []

    def grammar(logits, bitmask):
        seen.append(bitmask.copy())
        keep = torch.as_tensor(bitmask)
        original = logits[torch.arange(2), keep].clone()
        logits.fill_(-torch.inf)
        logits[torch.arange(2), keep] = original

    result, _ = batch.extract_output(
        SimpleNamespace(host_sampler=Sampler(), apply_grammar_bitmask=grammar),
        TTCompactedHostLogits(torch.zeros(2, 1, VOCAB), tuple(rows)),
        None,
        SimpleNamespace(
            perform_device_sampling=False,
            grammar_bitmask=[allowed],
            intermediate_prefill_mask=None,
        ),
        rows,
        True,
    )
    assert seen[0].tolist() == allowed[rows].tolist()
    assert result.flatten().tolist() == allowed[rows].tolist()


def test_compact_readback_rejects_wrong_row_order():
    batch = _lane_batch(num_lanes=2, per_lane=4, with_custom=False)
    with pytest.raises(ValueError):
        batch.extract_output(
            SimpleNamespace(),
            TTCompactedHostLogits(torch.zeros(2, 1, VOCAB), (1, 6)),
            None,
            SimpleNamespace(
                perform_device_sampling=False,
                grammar_bitmask=[None],
                intermediate_prefill_mask=None,
            ),
            [6, 1],
            True,
        )
