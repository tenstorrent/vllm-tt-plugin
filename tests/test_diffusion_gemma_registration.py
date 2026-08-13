# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

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


def test_gemma4_parsers_resolve_to_upstream_vllm():
    """The plugin registers no parsers; ``gemma4`` must resolve to vLLM's own
    engine-based parser for both reasoning and tool calls."""
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    reasoning_cls = ReasoningParserManager.get_reasoning_parser("gemma4")
    tool_cls = ToolParserManager.get_tool_parser("gemma4")

    assert reasoning_cls.__module__.startswith("vllm.")
    assert tool_cls.__module__.startswith("vllm.")
