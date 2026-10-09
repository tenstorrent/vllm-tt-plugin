# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.sample.sampler import Sampler

from tests.test_decode_reload_contract import _accepted_decode_hooks, _submission_input
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
    full_logits = torch.arange(8 * VOCAB).reshape(8, 1, VOCAB).float()

    def readback(value, **kwargs):
        calls.append(kwargs)
        return full_logits[kwargs.get("sample_rows", list(range(8)))].clone(), None

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
        torch.testing.assert_close(output.tt_out.logits, full_logits[rows])
    else:
        torch.testing.assert_close(output.tt_out, full_logits)


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


@pytest.mark.parametrize("invalid", ["order", "count", "processor"])
def test_compact_readback_rejects_incompatible_output(invalid):
    batch = _lane_batch(num_lanes=2, per_lane=4, with_custom=False)
    if invalid == "processor":
        batch.add_request_to_row(
            _make_req("a", [1], [], dict(min_p=0.1, temperature=1.0)), 6
        )
        batch.refresh_logitsprocs()
    output_rows = (1, 6) if invalid == "order" else (6, 1)
    count = 1 if invalid == "count" else 2
    with pytest.raises(ValueError):
        batch.extract_output(
            SimpleNamespace(),
            TTCompactedHostLogits(torch.zeros(count, 1, VOCAB), output_rows),
            None,
            SimpleNamespace(
                perform_device_sampling=False,
                grammar_bitmask=[None],
                intermediate_prefill_mask=None,
            ),
            [6, 1],
            True,
        )


@pytest.mark.parametrize("capability", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("device_sampling", [False, True])
@pytest.mark.parametrize("async_read", [False, True])
def test_selective_readback_captures_rows_before_transfer(
    capability, independent, device_sampling, async_read, monkeypatch
):
    import vllm_tt_plugin.async_decode as mod

    calls = []
    waits = []
    full_logits = torch.arange(8 * VOCAB).reshape(8, 1, VOCAB).float()

    class Model:
        decode_input_update_contract = 1
        model_capabilities = {
            "supports_async_decode": True,
            "supports_compact_host_logits": True,
            "supports_selective_host_readback": capability,
        }

        def decode_forward(self, **kwargs):
            calls.append(("forward", kwargs))
            return torch.zeros(8, 1, VOCAB) if kwargs["read_from_device"] else object()

        def read_decode_output(self, value, **kwargs):
            calls.append(("read", kwargs))
            return object(), ["completed"]

        def process_decode_output_host(self, value, **kwargs):
            assert waits == ["completed"]
            calls.append(("process", kwargs))
            return full_logits[kwargs.get("sample_rows", list(range(8)))].clone()

    runner = _accepted_decode_hooks(
        SimpleNamespace(
            model=Model(),
            trace_mode="decode_only",
            kv_caches=object(),
            request_specific_rope=False,
            lane_batch=SimpleNamespace(can_compact_host_sampling=lambda: independent),
        )
    )
    monkeypatch.setattr(mod.ttnn, "event_synchronize", waits.append, raising=False)
    controller = TTAsyncDecodeController(runner)
    rows = [6, 1]
    submission = controller.submit_decode(
        _submission_input(device_sampling=device_sampling),
        read_from_device=not async_read,
        async_read=async_read,
        sampling_rows=rows,
    )
    selective = capability and independent and not device_sampling
    assert submission.readback_rows == ((6, 1) if selective else None)
    if selective:
        assert calls[0][1]["read_from_device"] is False
        assert calls[1][1]["sample_rows"] == [6, 1]
        rows.reverse()
        with pytest.raises(ValueError):
            controller.finalize_decode(submission, sampling_rows=rows)
        # Later metadata changes cannot change which rows were transferred.
        runner.lane_batch.can_compact_host_sampling = lambda: False
        result = controller.finalize_decode(submission, sampling_rows=[6, 1])
        assert result.tt_out.rows == (6, 1)
        torch.testing.assert_close(result.tt_out.logits, full_logits[[6, 1]])
        assert calls[-1][1]["sample_rows"] == [6, 1]
    else:
        assert all("sample_rows" not in kwargs for name, kwargs in calls)
        result = controller.finalize_decode(submission, sampling_rows=rows)
        if async_read:
            if independent and not device_sampling:
                # A compact-only adapter reads all slots, then selects on host.
                torch.testing.assert_close(result.tt_out.logits, full_logits[rows])
            else:
                torch.testing.assert_close(result.tt_out, full_logits)
                assert "sample_rows" not in calls[-1][1]
        else:
            assert waits == []
            assert isinstance(result.tt_out, torch.Tensor)
            assert result.tt_out.shape == (8, 1, VOCAB)


@pytest.mark.parametrize("device_sampling", [False, True])
def test_legacy_adapter_keeps_full_slots_through_async_readback(
    device_sampling, monkeypatch
):
    import vllm_tt_plugin.async_decode as mod

    waits = []
    pending = object()
    host_output = object()
    full_output = torch.arange(8).reshape(8, 1) if device_sampling else torch.eye(8)

    class LegacyModel:
        decode_input_update_contract = 1
        model_capabilities = {"supports_async_decode": True}

        def decode_forward(self, **kwargs):
            assert not kwargs["read_from_device"]
            return pending

        def read_decode_output(self, value, *, async_read):
            assert value is pending
            assert async_read
            return host_output, ["completed"]

        def process_decode_output_host(self, value, *, is_tokens):
            assert value is host_output
            assert is_tokens is device_sampling
            assert waits == ["completed"]
            return full_output.clone()

    runner = _accepted_decode_hooks(
        SimpleNamespace(
            model=LegacyModel(),
            trace_mode="decode_only",
            kv_caches=object(),
            request_specific_rope=False,
            lane_batch=SimpleNamespace(can_compact_host_sampling=lambda: True),
        )
    )
    monkeypatch.setattr(mod.ttnn, "event_synchronize", waits.append, raising=False)
    controller = TTAsyncDecodeController(runner)
    submission = controller.submit_decode(
        _submission_input(device_sampling=device_sampling),
        read_from_device=False,
        async_read=True,
        sampling_rows=[6, 1],
    )
    result = controller.finalize_decode(submission, sampling_rows=[6, 1])
    assert submission.readback_rows is None
    assert isinstance(result.tt_out, torch.Tensor)
    torch.testing.assert_close(result.tt_out, full_output)
