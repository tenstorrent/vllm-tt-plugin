# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager

from tests.test_gemma4_reasoning_parser import FakeTokenizer
from vllm_tt_plugin import platform
from vllm_tt_plugin.entrypoints import _register_tt_reasoning_parsers
from vllm_tt_plugin.gemma4_reasoning_parser import Gemma4ReasoningParser


def test_diffusion_gemma_reasoning_parser_alias(monkeypatch):
    registrations = []
    monkeypatch.setattr(
        ReasoningParserManager,
        "register_lazy_module",
        staticmethod(lambda *args: registrations.append(args)),
    )

    _register_tt_reasoning_parsers()

    assert (
        "diffusion_gemma",
        "vllm_tt_plugin.gemma4_reasoning_parser",
        "Gemma4ReasoningParser",
    ) in registrations


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


def test_non_streaming_parser_adapter_uses_raw_token_ids():
    ReasoningParserManager.register_module(
        name="diffusion_gemma",
        module=Gemma4ReasoningParser,
    )
    _register_tt_reasoning_parsers()
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name="diffusion_gemma",
    )
    assert parser_cls is not None

    parser = parser_cls(FakeTokenizer())
    reasoning, content, tool_calls = parser.parse(
        "thought\nReason.Answer.",
        request=None,
        model_output_token_ids=[1, 10, 2, 11],
    )

    assert reasoning == "Reason."
    assert content == "Answer."
    assert tool_calls == []
