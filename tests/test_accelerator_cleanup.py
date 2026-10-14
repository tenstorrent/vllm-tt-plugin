# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import sys
from types import FunctionType, SimpleNamespace

import pytest
import torch
import vllm.platforms

from vllm_tt_plugin import platform


@pytest.fixture
def active_tt(monkeypatch):
    monkeypatch.setattr(vllm.platforms, "current_platform", platform.TTPlatform())


@pytest.mark.parametrize("available", [False, True])
def test_availability_guard_delegates_and_is_idempotent(
    monkeypatch, active_tt, available
):
    calls = []
    sentinel = object()

    def original():
        calls.append("cache")
        return sentinel

    monkeypatch.setattr(torch.accelerator, "empty_cache", original)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: available)
    platform._install_torch_accelerator_cleanup_patch()
    wrapped = torch.accelerator.empty_cache
    platform._install_torch_accelerator_cleanup_patch()
    assert torch.accelerator.empty_cache is wrapped
    assert wrapped.__wrapped__ is original
    assert wrapped() is (sentinel if available else None)
    assert calls == (["cache"] if available else [])


@pytest.mark.parametrize("failure", ["availability", "cache"])
def test_accelerator_failures_propagate(monkeypatch, active_tt, failure):
    error = RuntimeError(failure)

    def fail():
        raise error

    monkeypatch.setattr(
        torch.accelerator,
        "is_available",
        fail if failure == "availability" else lambda: True,
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", fail)
    platform._install_torch_accelerator_cleanup_patch()
    with pytest.raises(RuntimeError) as raised:
        torch.accelerator.empty_cache()
    assert raised.value is error


def test_non_tt_platform_does_not_install(monkeypatch):
    original = torch.accelerator.empty_cache
    monkeypatch.setattr(vllm.platforms, "current_platform", SimpleNamespace())
    platform._install_torch_accelerator_cleanup_patch()
    assert torch.accelerator.empty_cache is original


def test_actual_cpu_torch_failure_and_guard(monkeypatch, active_tt):
    if torch.accelerator.is_available():
        pytest.skip("This regression exercises Torch without an accelerator")
    original = torch.accelerator.empty_cache
    while hasattr(original, "__wrapped__"):
        original = original.__wrapped__
    monkeypatch.setattr(torch.accelerator, "empty_cache", original)
    with pytest.raises(RuntimeError):
        original()
    platform._install_torch_accelerator_cleanup_patch()
    assert torch.accelerator.empty_cache() is None


@pytest.mark.parametrize("cpu_platform", [False, True])
@pytest.mark.parametrize("host", ["success", "missing", "error"])
def test_original_distributed_cleanup_preserves_host_behavior(
    monkeypatch, active_tt, cpu_platform, host
):
    from vllm.distributed import parallel_state

    events = []

    def host_cache():
        events.append("host")
        if host == "error":
            raise RuntimeError("host failure")

    def unavailable_cache():
        raise RuntimeError("no accelerator")

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(torch.accelerator, "empty_cache", unavailable_cache)
    platform._install_torch_accelerator_cleanup_patch()
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(is_cpu=lambda: cpu_platform, is_rocm=lambda: False),
    )
    original = parallel_state.cleanup_dist_env_and_memory
    # Execute the installed upstream function, replacing only external effects.
    # The upstream function itself and the process-wide Torch module stay intact.
    namespace = dict(original.__globals__)
    namespace.update(
        torch=SimpleNamespace(
            accelerator=SimpleNamespace(
                empty_cache=torch.accelerator.empty_cache,
                **({} if host == "missing" else {"empty_host_cache": host_cache}),
            ),
        ),
        envs=SimpleNamespace(disable_envs_cache=lambda: events.append("envs")),
        gc=SimpleNamespace(
            unfreeze=lambda: events.append("unfreeze"),
            collect=lambda: events.append("collect"),
        ),
        destroy_model_parallel=lambda: events.append("model"),
        destroy_distributed_environment=lambda: events.append("distributed"),
        logger=SimpleNamespace(
            debug=lambda *args: None,
            debug_once=lambda *args: None,
            warning=lambda *args: events.append("warning"),
        ),
    )
    cleanup = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__
    )
    if host == "error" and not cpu_platform:
        with pytest.raises(RuntimeError, match="host failure"):
            cleanup()
    else:
        cleanup()
    expected = ["envs", "unfreeze", "model", "distributed", "collect"]
    if not cpu_platform:
        expected.append("warning" if host == "missing" else "host")
    assert events == expected


@pytest.mark.parametrize("ttnn_available", [False, True])
@pytest.mark.parametrize("tt_active", [False, True])
def test_general_plugin_installation_boundary(
    monkeypatch, active_tt, ttnn_available, tt_active
):
    from vllm_tt_plugin import model_registry

    calls = []
    if not tt_active:
        monkeypatch.setattr(vllm.platforms, "current_platform", SimpleNamespace())
    original = torch.accelerator.empty_cache
    monkeypatch.setattr(
        model_registry, "register_tt_models", lambda **kwargs: calls.append("register")
    )
    monkeypatch.setattr(
        model_registry, "_install_diffusion_gemma_architecture_patch", lambda: None
    )
    monkeypatch.setattr(
        model_registry, "_install_tt_async_spec_method_patch", lambda: None
    )
    if not ttnn_available:
        monkeypatch.setitem(sys.modules, "ttnn", None)
    model_registry.register_tt_models_from_plugin()
    assert calls == (["register"] if ttnn_available else [])
    if ttnn_available and tt_active:
        assert getattr(torch.accelerator.empty_cache, "_tt_availability_guard", False)
    else:
        assert torch.accelerator.empty_cache is original
