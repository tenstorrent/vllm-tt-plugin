# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Warmup as two steps: compile every path with tracing off, then capture.

``TTWorker.compile_or_warm_up_model`` runs both. A host that runs several
engines in one process calls them separately, so it can compile every model
before any model captures a trace.
"""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.worker import TTWorker


class _Model:
    """Records each warmup call as (phase, traced, can_sample_on_device)."""

    def __init__(self, calls: list):
        self.calls = calls
        self.already_warmed_up_prefill = True

    def warmup_model_prefill(self, *, enable_trace, kv_cache, can_sample_on_device):
        self.calls.append(("prefill", enable_trace, can_sample_on_device))
        self.already_warmed_up_prefill = True

    def warmup_model_decode(
        self,
        *,
        enable_trace,
        kv_cache,
        max_batch_size,
        num_blocks,
        can_sample_on_device,
    ):
        assert (max_batch_size, num_blocks) == (32, 4)
        self.calls.append(("decode", enable_trace, can_sample_on_device))


def _runner(calls: list, trace_mode: str = "all", sample_on_device_mode=None):
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.model = _Model(calls)
    runner.kv_caches = []
    runner.tt_max_batch_size = 32
    runner.max_num_blocks_per_req = 4
    runner.trace_mode = trace_mode
    runner.sample_on_device_mode = sample_on_device_mode
    runner.supports_device_grammar = False
    return runner


@pytest.mark.parametrize(
    ("trace_mode", "captured"),
    [
        ("all", [("prefill", True, False), ("decode", True, False)]),
        ("decode_only", [("decode", True, False)]),
        ("none", []),
    ],
)
def test_compile_traces_nothing_and_capture_traces_what_trace_mode_asks(
    trace_mode, captured
):
    calls: list = []
    runner = _runner(calls, trace_mode)

    runner.compile_model()

    assert calls == [("prefill", False, False), ("decode", False, False)]
    # Reset so the traced prefill warmup runs again instead of returning early.
    assert runner.model.already_warmed_up_prefill is False

    calls.clear()
    runner.capture_model()

    assert calls == captured


def test_capture_uses_the_runners_sampling_mode_not_the_platforms():
    """Another engine in the process may reconfigure TTPlatform in between."""
    calls: list = []
    runner = _runner(calls, sample_on_device_mode="all")
    TTPlatform.sample_on_device_mode = "all"
    runner.compile_model()

    TTPlatform.sample_on_device_mode = None
    runner.capture_model()

    assert calls[2:] == [("prefill", True, True), ("decode", True, True)]


def _worker(enable_model_warmup: bool):
    calls: list[str] = []
    worker = TTWorker.__new__(TTWorker)
    worker.enable_model_warmup = enable_model_warmup
    worker.model_runner = SimpleNamespace(
        compile_model=lambda: calls.append("compile"),
        capture_model=lambda: calls.append("capture"),
    )
    return worker, calls


def test_compile_or_warm_up_model_compiles_then_captures():
    worker, calls = _worker(True)

    timings = worker.compile_or_warm_up_model()

    assert calls == ["compile", "capture"]
    assert timings.language_model >= 0.0


def test_both_steps_honour_disabled_warmup():
    worker, calls = _worker(False)

    worker.compile_model()
    worker.capture_model()
    timings = worker.compile_or_warm_up_model()

    assert calls == []
    assert timings.language_model == 0.0
