# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Phase observers follow existing completion and cannot repeat readback."""

import threading
from types import SimpleNamespace

from vllm_tt_plugin.async_decode import DeferredDecodeOutput
from vllm_tt_plugin.execution_phase import finish_observed_phase


def test_async_phase_finishes_after_readback_exactly_once():
    calls = []

    class Output(DeferredDecodeOutput):
        def _get_output_impl(self):
            calls.append("readback")
            return "output"

    output = Output()
    output._completion_event = threading.Event()
    output._init_deferred()
    output._execution_phase = SimpleNamespace(complete=lambda: calls.append("complete"))
    assert output.ensure_finalized() == "output"
    assert output.get_output() == "output"
    assert calls == ["readback", "complete"]
    assert output.is_resolved()


def test_sync_phase_follows_sampler_and_preserves_output():
    calls = []
    phase = SimpleNamespace(complete=lambda: calls.append("complete"))

    def sample(grammar):
        calls.append(("sample", grammar))
        return "output"

    assert finish_observed_phase(sample, phase, "grammar") == "output"
    assert calls == [("sample", "grammar"), "complete"]


def test_failed_readback_does_not_publish_phase_completion():
    import pytest

    calls = []

    class Output(DeferredDecodeOutput):
        def _get_output_impl(self):
            calls.append("readback")
            raise RuntimeError("readback failed")

    output = Output()
    output._completion_event = threading.Event()
    output._init_deferred()
    output._execution_phase = SimpleNamespace(complete=lambda: calls.append("complete"))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="readback failed"):
            output.get_output()
    assert calls == ["readback"]
