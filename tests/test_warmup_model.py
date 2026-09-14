# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host coverage of the runner's eager preparation / trace capture ordering."""

from types import SimpleNamespace

import pytest

import vllm_tt_plugin  # noqa: F401 (activate the TT platform before runner imports)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import TTPlatform


@pytest.mark.parametrize("trace_mode", ["all", "decode_only", "none"])
@pytest.mark.parametrize("sampling_mode", [None, "all", "decode_only"])
def test_warmup_prepares_decode_before_prefill_and_captures_prefill_first(
    monkeypatch, trace_mode, sampling_mode
):
    monkeypatch.setattr(TTPlatform, "sample_on_device_mode", sampling_mode)
    kv_caches = object()
    events = []

    class Model:
        already_warmed_up_prefill = False
        decode_inputs = None
        decode_captured = False

        def warmup_model_prefill(self, *, kv_cache, enable_trace, can_sample_on_device):
            assert kv_cache is kv_caches
            assert can_sample_on_device == (sampling_mode == "all")
            # A custom adapter may defer its prefill allocations until this
            # trace-enabled call. They must not run behind a decode trace.
            assert not self.decode_captured
            if enable_trace:
                assert not self.already_warmed_up_prefill
                assert self.decode_inputs is not None
            self.already_warmed_up_prefill = True
            events.append(("prefill", enable_trace))

        def warmup_model_decode(
            self,
            *,
            kv_cache,
            enable_trace,
            max_batch_size,
            num_blocks,
            can_sample_on_device,
        ):
            assert kv_cache is kv_caches
            assert max_batch_size == 8
            # Preparation must get the runner's profiled page-table width,
            # rather than infer it from the model's maximum context length.
            assert num_blocks == 57
            assert can_sample_on_device == (sampling_mode in ("all", "decode_only"))
            if enable_trace:
                assert self.decode_inputs is not None
                self.decode_captured = True
            else:
                assert not self.decode_captured
                self.decode_inputs = object()
            events.append(("decode", enable_trace))

    runner = SimpleNamespace(
        model=Model(),
        trace_mode=trace_mode,
        kv_caches=kv_caches,
        tt_max_batch_size=8,
        max_num_blocks_per_req=57,
    )
    TTModelRunner.warmup_model(runner)

    expected = [("prefill", False), ("decode", False)]
    if trace_mode == "all":
        expected.append(("prefill", True))
    if trace_mode in ("all", "decode_only"):
        expected.append(("decode", True))
    assert events == expected
