# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from types import SimpleNamespace

from vllm_tt_plugin.loader import TTModelLoader


def test_loader_forwards_vllm_config_to_model_runtime(monkeypatch):
    """The model must see the scheduler/cache contract used for execution."""
    received = {}
    sentinel = object()

    class GeneratedModel:
        @classmethod
        def initialize_vllm_model(cls, *args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs
            return sentinel

    config = SimpleNamespace(
        device_config=SimpleNamespace(device="mesh"),
        cache_config=SimpleNamespace(block_size=64),
    )
    model_config = SimpleNamespace(
        hf_config=object(),
        max_model_len=131072,
    )
    monkeypatch.setattr(
        "vllm_tt_plugin.loader.get_model_architecture",
        lambda _model_config: (GeneratedModel, None),
    )
    monkeypatch.setattr("vllm_tt_plugin.loader.get_tt_config", lambda _config: {})
    monkeypatch.setattr(
        "vllm_tt_plugin.loader.get_tt_data_parallel_size", lambda _config: 1
    )
    monkeypatch.setattr(
        "vllm_tt_plugin.loader.get_tt_max_batch_size", lambda _config: 32
    )

    loader = object.__new__(TTModelLoader)
    assert loader.load_model(config, model_config) is sentinel
    assert received["kwargs"]["vllm_config"] is config
    assert received["kwargs"]["max_seq_len"] == 131072


def test_loader_keeps_legacy_model_initializer_compatible(monkeypatch):
    received = {}
    sentinel = object()

    class LegacyModel:
        @classmethod
        def initialize_vllm_model(
            cls,
            hf_config,
            device,
            max_batch_size,
            *,
            max_seq_len,
            tt_data_parallel,
            optimizations,
        ):
            received["values"] = (
                hf_config,
                device,
                max_batch_size,
                max_seq_len,
                tt_data_parallel,
                optimizations,
            )
            return sentinel

    config = SimpleNamespace(device_config=SimpleNamespace(device="mesh"))
    model_config = SimpleNamespace(hf_config="hf", max_model_len=8192)
    monkeypatch.setattr(
        "vllm_tt_plugin.loader.get_model_architecture",
        lambda _model_config: (LegacyModel, None),
    )
    monkeypatch.setattr("vllm_tt_plugin.loader.get_tt_config", lambda _config: {})
    monkeypatch.setattr(
        "vllm_tt_plugin.loader.get_tt_data_parallel_size", lambda _config: 1
    )
    monkeypatch.setattr(
        "vllm_tt_plugin.loader.get_tt_max_batch_size", lambda _config: 4
    )

    loader = object.__new__(TTModelLoader)
    assert loader.load_model(config, model_config) is sentinel
    assert received["values"] == ("hf", "mesh", 4, 8192, 1, None)
