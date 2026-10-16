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


def test_ar_stages_run_the_tt_worker_and_generation_stages_are_refused():
    assert TTOmniPlatform.get_omni_ar_worker_cls() == TT_OMNI_AR_WORKER_CLS
    with pytest.raises(NotImplementedError, match="no vLLM-Omni generation worker"):
        TTOmniPlatform.get_omni_generation_worker_cls()


def test_devices_are_counted_from_the_driver_nodes(monkeypatch: pytest.MonkeyPatch):
    import vllm_tt_plugin.omni.platform as omni_platform

    nodes = {"/dev/tenstorrent/[0-9]*": ["/dev/tenstorrent/0", "/dev/tenstorrent/1"]}
    monkeypatch.setattr(omni_platform.glob, "glob", lambda pattern: nodes[pattern])

    assert TTOmniPlatform.get_device_count() == 2


def test_the_talker_stage_architecture_is_registered(monkeypatch: pytest.MonkeyPatch):
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


def test_the_omni_plugin_registers_the_talker_where_omni_validates_it(
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
        tt_platform.TT_QWEN3_TTS_TALKER_ARCH: tt_platform.TT_QWEN3_TTS_TALKER_TARGET
    }
    assert "tt_qwen3_tts_talker" in pipeline_registry.OMNI_PIPELINES
