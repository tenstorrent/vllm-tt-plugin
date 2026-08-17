# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest

from vllm_tt_plugin import platform


def test_diffusion_gemma_model_architecture_aliases(monkeypatch):
    registrations = []
    monkeypatch.setattr(
        platform,
        "_register_model_if_missing",
        lambda _registry, arch, target: registrations.append((arch, target)),
    )

    platform.register_tt_models()

    expected_target = (
        "models.experimental.diffusion_gemma.tt.generator_vllm:"
        "DiffusionGemmaForCausalLM"
    )
    registered = dict(registrations)
    for arch in (
        "DiffusionGemmaForBlockDiffusion",
        "DiffusionGemmaForCausalLM",
        "TTDiffusionGemmaForBlockDiffusion",
        "TTDiffusionGemmaForCausalLM",
    ):
        assert registered[arch] == expected_target


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
