# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM-Omni resolves the TT Omni platform, and the platform places stages on TT."""

import os
import sys

import pytest

pytest.importorskip("vllm_omni")

import vllm.config  # noqa: E402,F401  # finish vLLM init before the plugin package

import vllm_tt_plugin.platform as tt_platform  # noqa: E402
from vllm_tt_plugin.omni import entrypoints  # noqa: E402
from vllm_tt_plugin.omni.platform import (  # noqa: E402
    TT_OMNI_AR_WORKER_CLS,
    TT_OMNI_GENERATION_WORKER_CLS,
    TTOmniPlatform,
)

TT_OMNI_PLATFORM = "vllm_tt_plugin.omni.platform.TTOmniPlatform"


def test_the_plugin_names_the_platform_when_ttnn_imports():
    assert entrypoints.omni_platform_plugin() == TT_OMNI_PLATFORM


def test_the_plugin_declines_without_ttnn(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "ttnn", None)  # makes `import ttnn` raise

    assert entrypoints.omni_platform_plugin() is None


def test_omni_activates_the_tt_platform(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.platforms as omni_platforms

    monkeypatch.setattr(
        omni_platforms,
        "load_omni_plugins_by_group",
        lambda group: {"tt": entrypoints.omni_platform_plugin},
    )

    assert (
        omni_platforms.resolve_current_omni_platform_cls_qualname() == TT_OMNI_PLATFORM
    )


def test_a_stage_is_confined_through_tt_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "0,1")

    TTOmniPlatform.set_device_control_env_var("1")

    assert TTOmniPlatform.device_control_env_var == "TT_VISIBLE_DEVICES"
    assert TTOmniPlatform.is_out_of_tree(TTOmniPlatform)
    assert os.environ["TT_VISIBLE_DEVICES"] == "1"


def test_ar_and_generation_stages_run_their_tt_workers():
    assert TTOmniPlatform.get_omni_ar_worker_cls() == TT_OMNI_AR_WORKER_CLS
    assert TTOmniPlatform.get_omni_generation_worker_cls() == (
        TT_OMNI_GENERATION_WORKER_CLS
    )


def test_devices_are_counted_from_the_driver_nodes(monkeypatch: pytest.MonkeyPatch):
    import vllm_tt_plugin.omni.platform as omni_platform

    nodes = {"/dev/tenstorrent/[0-9]*": ["/dev/tenstorrent/0", "/dev/tenstorrent/1"]}
    monkeypatch.setattr(omni_platform.glob, "glob", lambda pattern: nodes[pattern])

    assert TTOmniPlatform.get_device_count() == 2


@pytest.mark.parametrize("count", [0, 2])
def test_the_torch_device_queries_vllm_omni_calls_are_answered(
    monkeypatch: pytest.MonkeyPatch, count
):
    """An Omni platform instance forwards a missing name to `torch.tt` and gets None."""
    import vllm_tt_plugin.omni.platform as omni_platform

    nodes = [f"/dev/tenstorrent/{index}" for index in range(count)]
    monkeypatch.setattr(omni_platform.glob, "glob", lambda pattern: nodes)
    platform = TTOmniPlatform()

    assert platform.device_count() == count
    assert platform.is_available() is (count > 0)
    assert platform.empty_cache() is None


def test_the_tts_stage_architectures_are_registered(monkeypatch: pytest.MonkeyPatch):
    from vllm.model_executor.models.registry import ModelRegistry

    registered: dict[str, str] = {}
    monkeypatch.setattr(
        ModelRegistry, "get_supported_archs", staticmethod(lambda: list(registered))
    )
    monkeypatch.setattr(
        ModelRegistry,
        "register_model",
        staticmethod(lambda arch, target: registered.setdefault(arch, target)),
    )

    tt_platform.register_tt_models()

    assert registered["TTQwen3TTSTalkerForConditionalGeneration"] == (
        "models.demos.audio.qwen3_tts.tt.generator_vllm:"
        "Qwen3TTSTalkerForConditionalGeneration"
    )
    assert registered["TTQwen3TTSCode2Wav"] == (
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav:Qwen3TTSCode2Wav"
    )


def test_the_omni_plugin_registers_the_tts_stages_where_omni_validates_them(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_omni.config import pipeline_registry
    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    registered: dict[str, str] = {}
    monkeypatch.setattr(
        OmniModelRegistry,
        "register_model",
        lambda arch, target: registered.setdefault(arch, target),
    )
    monkeypatch.setattr(pipeline_registry, "OMNI_PIPELINES", {})

    entrypoints.register()

    assert registered == {
        tt_platform.TT_QWEN3_TTS_TALKER_ARCH: tt_platform.TT_QWEN3_TTS_TALKER_TARGET,
        tt_platform.TT_QWEN3_TTS_CODE2WAV_ARCH: (
            tt_platform.TT_QWEN3_TTS_CODE2WAV_TARGET
        ),
    }
    assert "tt_qwen3_tts_talker" in pipeline_registry.OMNI_PIPELINES
