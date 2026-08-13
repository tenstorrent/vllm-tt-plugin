# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from vllm_tt_plugin import platform
from vllm_tt_plugin.entrypoints import (
    _register_tt_reasoning_parsers,
    _register_tt_tool_parsers,
)


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


def test_gemma4_keeps_plugin_parsers_and_diffusion_gemma_uses_upstream():
    """``gemma4`` must keep the plugin parsers (autoregressive Gemma 4 serving
    is unchanged), while ``diffusion_gemma`` resolves to the engine-based
    parser that ships with vLLM 0.24, matching upstream's DiffusionGemma
    serving recipe."""
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    _register_tt_reasoning_parsers()
    _register_tt_tool_parsers()

    assert (
        ReasoningParserManager.get_reasoning_parser("gemma4").__module__
        == "vllm_tt_plugin.gemma4_reasoning_parser"
    )
    assert (
        ToolParserManager.get_tool_parser("gemma4").__module__
        == "vllm_tt_plugin.gemma4_tool_parser"
    )

    assert ReasoningParserManager.get_reasoning_parser(
        "diffusion_gemma"
    ).__module__.startswith("vllm.")
    assert ToolParserManager.get_tool_parser("diffusion_gemma").__module__.startswith(
        "vllm."
    )
