# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A TT worker through the engine's lifecycle, with and without a token KV cache.

The engine side is vLLM's own KV planner, so the config each worker receives is
the one ``EngineCore._initialize_kv_caches`` would build from its reported spec.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_tt_plugin import worker
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.worker import TTWorker

NUM_TT_BLOCKS = 7


class _Runner:
    """Records the calls a worker makes on its runner."""

    def __init__(self, **kwargs):
        self.calls: list[str] = []
        self.kv_cache_config = None

    def initialize_kv_cache(self, kv_cache_config):
        self.calls.append("kv")
        self.kv_cache_config = kv_cache_config

    def warmup_model(self):
        self.calls.append("warmup")

    def execute_model(self, scheduler_output):
        self.calls.append("execute")
        return scheduler_output

    def shutdown(self):
        self.calls.append("shutdown")


class _ARRunner(_Runner):
    def initialize_kv_cache(self, kv_cache_config):
        TTModelRunner._validate_kv_cache_groups(kv_cache_config.kv_cache_groups)
        super().initialize_kv_cache(kv_cache_config)


class _ARWorker(TTWorker):
    model_runner_cls = _ARRunner


class _NoKVWorker(TTWorker):
    model_runner_cls = _Runner
    uses_token_kv_cache = False


class _NoKVWorkerWithARRunner(TTWorker):
    model_runner_cls = _ARRunner
    uses_token_kv_cache = False


def _worker(monkeypatch: pytest.MonkeyPatch, worker_cls: type[TTWorker]):
    """A worker whose device, sizing and registry steps run against doubles.

    Returns the worker and the list the TT-side steps append to.
    """
    steps: list[str] = []
    mesh_device = SimpleNamespace(get_num_devices=lambda: 1)
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_rank_local=0,
        data_parallel_index=0,
        assigned_physical_gpu_ids=None,
    )
    cache_config = SimpleNamespace(
        block_size=32,
        cache_dtype="auto",
        num_gpu_blocks_override=None,
        prefix_cache_retention_interval=None,
    )
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        use_mla=False,
        get_num_kv_heads=lambda _parallel_config: 8,
        get_head_size=lambda: 64,
        get_sliding_window=lambda: None,
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        parallel_config=parallel_config,
        cache_config=cache_config,
        model_config=model_config,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
    )

    worker_instance = worker_cls.__new__(worker_cls)
    worker_instance.vllm_config = vllm_config
    worker_instance.parallel_config = parallel_config
    worker_instance.cache_config = cache_config
    worker_instance.model_config = model_config
    worker_instance.device_config = SimpleNamespace(device=None)
    worker_instance.is_driver_worker = True
    worker_instance.trace_mode = "all"
    worker_instance.enable_model_warmup = True
    worker_instance._num_tt_blocks = None

    monkeypatch.setattr(TTPlatform, "check_and_update_config", lambda _cfg: None)
    monkeypatch.setattr(worker, "get_tt_config", lambda _cfg: {})
    monkeypatch.setattr(
        worker, "get_model_architecture", lambda _cfg: (SimpleNamespace(), "model")
    )
    monkeypatch.setattr(TTWorker, "_try_get_spec_from_model_hook", lambda _self: None)

    def open_mesh(*_args, **_kwargs):
        steps.append("open mesh")
        return mesh_device

    def size_pool(_cfg, _num_devices):
        steps.append("size pool")
        return NUM_TT_BLOCKS

    def close_mesh(device, _tt_config):
        assert device is mesh_device
        steps.append("close mesh")

    monkeypatch.setattr(worker, "open_mesh_device", open_mesh)
    monkeypatch.setattr(worker, "close_mesh_device", close_mesh)
    monkeypatch.setattr(worker, "get_num_available_blocks_tt", size_pool)
    monkeypatch.setattr(
        worker,
        "_fit_block_output_max_model_len",
        lambda _cfg, _blocks: steps.append("fit max_model_len"),
    )
    return worker_instance, steps


def _engine_kv_cache_config(worker_instance: TTWorker):
    """What ``EngineCore._initialize_kv_caches`` builds from the worker's spec."""
    vllm_config = worker_instance.vllm_config
    (layout,) = worker_instance.get_supported_kv_cache_layouts()
    vllm_config.cache_config.get_resolved_kv_cache_layout = lambda: KVCacheLayout[
        layout
    ]
    spec = worker_instance.get_kv_cache_spec()
    # The engine asks for memory only when some worker reports a KV spec.
    memory = worker_instance.determine_available_memory() if spec else 0
    groups = get_kv_cache_groups(vllm_config, dict(spec))
    return get_kv_cache_config_from_groups(vllm_config, groups, memory)


def _run_lifecycle(worker_instance: TTWorker):
    worker_instance.init_device()
    kv_cache_config = _engine_kv_cache_config(worker_instance)
    worker_instance.initialize_from_config(kv_cache_config)
    worker_instance.compile_or_warm_up_model()
    scheduler_output = object()
    assert worker_instance.execute_model(scheduler_output) is scheduler_output
    runner = worker_instance.model_runner
    worker_instance.shutdown()
    return runner, kv_cache_config


def test_a_worker_without_token_kv_runs_the_whole_lifecycle(monkeypatch):
    worker_instance, steps = _worker(monkeypatch, _NoKVWorker)

    runner, kv_cache_config = _run_lifecycle(worker_instance)

    assert steps == ["open mesh", "close mesh"]
    assert worker_instance.get_kv_cache_spec() == {}
    assert worker_instance.determine_available_memory() == 0
    assert worker_instance.cache_config.num_gpu_blocks_override is None
    assert kv_cache_config.kv_cache_groups == []
    assert kv_cache_config.kv_cache_tensors == []
    assert runner.kv_cache_config is kv_cache_config
    assert runner.calls == ["kv", "warmup", "execute", "shutdown"]
    assert worker_instance.mesh_device is None


def test_a_worker_with_token_kv_keeps_its_pool_and_spec(monkeypatch):
    worker_instance, steps = _worker(monkeypatch, _ARWorker)

    runner, kv_cache_config = _run_lifecycle(worker_instance)

    assert steps == ["open mesh", "size pool", "fit max_model_len", "close mesh"]
    spec = worker_instance.get_kv_cache_spec()
    assert list(spec) == ["foo"]
    assert isinstance(spec["foo"], FullAttentionSpec)
    assert worker_instance.cache_config.num_gpu_blocks_override == NUM_TT_BLOCKS
    assert kv_cache_config.num_blocks == NUM_TT_BLOCKS
    assert [group.layer_names for group in kv_cache_config.kv_cache_groups] == [["foo"]]
    assert runner.calls == ["kv", "warmup", "execute", "shutdown"]
    assert worker_instance.mesh_device is None


def test_a_worker_without_token_kv_refuses_an_ar_runner(monkeypatch):
    """TTModelRunner keeps its guard: an AR runner cannot run with no KV groups."""
    worker_instance, _steps = _worker(monkeypatch, _NoKVWorkerWithARRunner)
    worker_instance.init_device()

    with pytest.raises(ValueError, match="kv_cache_config has no groups"):
        worker_instance.initialize_from_config(_engine_kv_cache_config(worker_instance))

    worker_instance.shutdown()
