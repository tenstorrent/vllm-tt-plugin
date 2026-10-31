# SPDX-License-Identifier: Apache-2.0
"""Requests with multimodal features must not read vLLM's prefix cache (images: a hit hides the pixels of the cached range from the model)."""
from types import SimpleNamespace

from vllm.sampling_params import SamplingParams

from tests.test_block_request_validation import _processor_harness


def _with_mm(monkeypatch, mm, salt=None):
    import vllm.v1.engine.input_processor as ip

    proc_cls, proc = _processor_harness(monkeypatch)
    orig = ip._tt_original_process_inputs

    def wrapped(self, *a, **k):
        r = orig(self, *a, **k)
        r.mm_features = mm
        r.cache_salt = salt
        return r

    monkeypatch.setattr(ip, "_tt_original_process_inputs", wrapped)
    # the installed patch captured ``orig`` in its closure: re-install over the wrapper
    monkeypatch.delattr(ip, "_tt_original_process_inputs")
    monkeypatch.setattr(ip.InputProcessor, "process_inputs", wrapped)
    from vllm_tt_plugin.platform import _install_tt_input_processor_patch

    _install_tt_input_processor_patch()
    return proc_cls, proc


def test_image_request_skips_prefix_cache_read(monkeypatch):
    proc_cls, proc = _with_mm(monkeypatch, [SimpleNamespace(identifier="img")])
    params = SamplingParams(max_tokens=16)
    req = proc_cls.process_inputs(proc, "r1", {"prompt_token_ids": [1] * 200}, params)
    assert req.sampling_params.skip_reading_prefix_cache is True
    assert not params.skip_reading_prefix_cache  # the caller's params are untouched


def test_text_request_keeps_prefix_cache(monkeypatch):
    proc_cls, proc = _with_mm(monkeypatch, None)
    req = proc_cls.process_inputs(proc, "r2", {"prompt_token_ids": [1] * 200}, SamplingParams(max_tokens=16))
    assert req.sampling_params.skip_reading_prefix_cache in (None, False)


def test_env_reenables_mm_prefix_cache(monkeypatch):
    monkeypatch.setenv("TT_MM_PREFIX_CACHE", "1")
    proc_cls, proc = _with_mm(monkeypatch, [SimpleNamespace(identifier="img")])
    req = proc_cls.process_inputs(proc, "r3", {"prompt_token_ids": [1] * 200}, SamplingParams(max_tokens=16))
    assert req.sampling_params.skip_reading_prefix_cache in (None, False)


def test_cache_salt_is_handed_to_the_worker_through_extra_args(monkeypatch):
    proc_cls, proc = _with_mm(monkeypatch, [SimpleNamespace(identifier="img")], salt="tenant-a")
    req = proc_cls.process_inputs(proc, "r4", {"prompt_token_ids": [1] * 200}, SamplingParams(max_tokens=16))
    assert req.sampling_params.extra_args == {"tt_cache_salt": "tenant-a"}
    assert req.sampling_params.skip_reading_prefix_cache is True


def test_no_salt_no_extra_args(monkeypatch):
    proc_cls, proc = _with_mm(monkeypatch, [SimpleNamespace(identifier="img")])
    req = proc_cls.process_inputs(proc, "r5", {"prompt_token_ids": [1] * 200}, SamplingParams(max_tokens=16))
    assert not req.sampling_params.extra_args
