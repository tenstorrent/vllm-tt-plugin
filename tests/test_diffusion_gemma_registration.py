# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
import vllm.config  # noqa: F401  # finish vLLM init before the plugin package,

# whose bare import re-enters vllm.platforms mid-initialization
import vllm_tt_plugin.platform as tt_platform


def test_diffusion_gemma_uses_tt_architecture_before_upstream_config_hooks(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.config import VllmConfig
    from vllm.config import model as model_config_module
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    hf_config = SimpleNamespace(
        architectures=[
            "DiffusionGemmaForBlockDiffusion",
            "DiffusionGemmaForCausalLM",
            "UnrelatedForCausalLM",
        ]
    )
    monkeypatch.setattr(
        model_config_module, "get_config", lambda *args, **kwargs: hf_config
    )

    tt_platform._install_diffusion_gemma_architecture_patch()
    patched_get_config = model_config_module.get_config
    tt_platform._install_diffusion_gemma_architecture_patch()

    assert model_config_module.get_config is patched_get_config
    assert patched_get_config().architectures == [
        "TTDiffusionGemmaForBlockDiffusion",
        "TTDiffusionGemmaForCausalLM",
        "UnrelatedForCausalLM",
    ]
    assert "DiffusionGemmaForBlockDiffusion" in MODELS_CONFIG_MAP
    assert "TTDiffusionGemmaForBlockDiffusion" not in MODELS_CONFIG_MAP

    upstream_calls = []
    upstream_hook = MODELS_CONFIG_MAP["DiffusionGemmaForBlockDiffusion"]
    monkeypatch.setattr(
        upstream_hook,
        "verify_and_update_config",
        classmethod(lambda cls, config: upstream_calls.append(config)),
    )
    monkeypatch.setattr(
        ModelRegistry,
        "_normalize_arch",
        staticmethod(lambda architecture, model_config: architecture),
    )
    model_config = SimpleNamespace(
        architecture=patched_get_config().architectures[0],
        config_updated=False,
        is_hybrid=False,
        convert_type="none",
    )
    vllm_config = SimpleNamespace(model_config=model_config, diffusion_config=None)

    VllmConfig.try_verify_and_update_config(vllm_config)

    assert upstream_calls == []
    assert model_config.config_updated is True
    assert vllm_config.diffusion_config is None


def test_pre_register_installs_models_before_diffusion_architecture_patch(
    monkeypatch: pytest.MonkeyPatch,
):
    events = []
    monkeypatch.setattr(tt_platform, "_pin_v1_model_runner", lambda: None)
    monkeypatch.setattr(
        tt_platform, "_install_tt_harmony_truncation_patch", lambda: None
    )
    monkeypatch.setattr(
        tt_platform, "_should_pre_register_tt_test_models_from_cli", lambda: True
    )
    monkeypatch.setattr(
        tt_platform,
        "register_tt_models",
        lambda register_test_models=False: events.append(
            ("register", register_test_models)
        ),
    )
    monkeypatch.setattr(
        tt_platform,
        "_install_diffusion_gemma_architecture_patch",
        lambda: events.append(("patch", None)),
    )

    tt_platform.TTPlatform.pre_register_and_update()

    assert events == [("register", True), ("patch", None)]


def test_gemma4_parsers_are_owned_by_upstream_vllm():
    from vllm.reasoning import ReasoningParserManager
    from vllm.reasoning.gemma4_engine_reasoning_parser import (
        Gemma4ParserReasoningAdapter,
    )
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
    from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser

    assert (
        ReasoningParserManager.get_reasoning_parser("gemma4")
        is Gemma4ParserReasoningAdapter
    )
    assert ToolParserManager.get_tool_parser("gemma4") is Gemma4EngineToolParser
    with pytest.raises(KeyError):
        ReasoningParserManager.get_reasoning_parser("diffusion_gemma")
    with pytest.raises(KeyError):
        ToolParserManager.get_tool_parser("diffusion_gemma")
