# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host coverage of TT cleanup with and without a Torch-managed accelerator."""

from types import SimpleNamespace

import pytest
import torch


def test_upstream_cleanup_with_tt_and_no_torch_accelerator(monkeypatch):
    import vllm.distributed.parallel_state as distributed
    import vllm.platforms

    from vllm_tt_plugin.platform import TTPlatform, _install_tt_empty_cache_patch

    calls = []
    monkeypatch.setattr(vllm.platforms, "current_platform", TTPlatform())
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: None)
    # Exercise the actual Torch empty_cache function and its failing CPU-build
    # guard without depending on which Torch wheel the host CI installed.
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", torch.accelerator.memory.empty_cache
    )

    def absent_allocator():
        raise RuntimeError("Cannot access accelerator device when none is available.")

    monkeypatch.setattr(
        torch._C, "_accelerator_isAllocatorInitialized", absent_allocator
    )
    monkeypatch.setattr(
        distributed.envs, "disable_envs_cache", lambda: calls.append("env-cache")
    )
    monkeypatch.setattr(
        distributed,
        "gc",
        SimpleNamespace(
            unfreeze=lambda: calls.append("unfreeze"),
            collect=lambda: calls.append("collect"),
        ),
    )
    monkeypatch.setattr(
        distributed, "destroy_model_parallel", lambda: calls.append("model-parallel")
    )
    monkeypatch.setattr(
        distributed,
        "destroy_distributed_environment",
        lambda: calls.append("distributed"),
    )
    monkeypatch.setattr(
        torch._C, "_host_emptyCache", lambda: calls.append("host-cache"), raising=False
    )
    with pytest.raises(RuntimeError, match="Cannot access accelerator device"):
        distributed.cleanup_dist_env_and_memory()
    assert calls == [
        "env-cache",
        "unfreeze",
        "model-parallel",
        "distributed",
        "collect",
    ]

    calls.clear()
    _install_tt_empty_cache_patch()
    installed = torch.accelerator.empty_cache
    _install_tt_empty_cache_patch()
    assert torch.accelerator.empty_cache is installed
    distributed.cleanup_dist_env_and_memory()
    assert calls == [
        "env-cache",
        "unfreeze",
        "model-parallel",
        "distributed",
        "collect",
        "host-cache",
    ]


@pytest.mark.parametrize(
    ("platform", "accelerator"),
    [("cpu", None), ("cuda", None), ("tt", torch.device("cuda"))],
)
@pytest.mark.parametrize("raises", [False, True])
def test_empty_cache_delegates_outside_tt_without_accelerator(
    monkeypatch, platform, accelerator, raises
):
    import vllm.platforms

    from vllm_tt_plugin.platform import _install_tt_empty_cache_patch

    calls = []
    marker = object()

    def original():
        calls.append("original")
        if raises:
            raise RuntimeError("allocator failure must propagate")
        return marker

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(device_type=platform)
    )
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: accelerator)
    monkeypatch.setattr(torch.accelerator, "empty_cache", original)
    _install_tt_empty_cache_patch()
    if raises:
        with pytest.raises(RuntimeError, match="allocator failure must propagate"):
            torch.accelerator.empty_cache()
    else:
        assert torch.accelerator.empty_cache() is marker
    assert calls == ["original"]
