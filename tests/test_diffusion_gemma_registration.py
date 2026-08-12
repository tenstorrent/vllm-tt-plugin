# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

from tests.test_gemma4_reasoning_parser import FakeTokenizer
from vllm_tt_plugin import platform
from vllm_tt_plugin.entrypoints import _register_tt_reasoning_parsers
from vllm_tt_plugin.gemma4_reasoning_parser import Gemma4ReasoningParser
from vllm_tt_plugin.gemma4_tool_parser import Gemma4ToolParser


def _unified_parser(
    *,
    with_tools: bool = True,
    reasoning_parser_name: str = "diffusion_gemma",
):
    ReasoningParserManager.register_module(
        name=reasoning_parser_name,
        module=Gemma4ReasoningParser,
    )
    if with_tools:
        ToolParserManager.register_module(
            name="gemma4",
            module=Gemma4ToolParser,
        )
    _register_tt_reasoning_parsers()
    parser_cls = ParserManager.get_parser(
        tool_parser_name="gemma4" if with_tools else None,
        reasoning_parser_name=reasoning_parser_name,
        enable_auto_tools=with_tools,
    )
    assert parser_cls is not None
    return parser_cls(FakeTokenizer(), tools=[])


def test_gemma4_reasoning_parser_aliases(monkeypatch):
    registrations = []
    monkeypatch.setattr(
        ReasoningParserManager,
        "register_lazy_module",
        staticmethod(lambda *args: registrations.append(args)),
    )

    _register_tt_reasoning_parsers()

    for alias in ("diffusion_gemma", "gemma4"):
        assert (
            alias,
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


def test_streaming_unified_parser_preserves_same_canvas_tool_call():
    tokenizer = FakeTokenizer()
    parser = _unified_parser()
    token_ids = [1, 10, 2, 4, 14, 6]
    delta_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    result = parser.parse_delta(
        delta_text=delta_text,
        delta_token_ids=token_ids,
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        prompt_token_ids=[],
        finished=False,
    )

    assert result is not None
    assert result.reasoning == "Reason."
    assert result.content is None
    assert result.tool_calls
    assert result.tool_calls[0].function.name == "get_weather"


def test_unified_parser_handles_implicit_reasoning_to_tool_transition():
    tokenizer = FakeTokenizer()
    token_ids = [1, 10, 4, 14, 6]
    model_output = tokenizer.decode(token_ids, skip_special_tokens=False)
    request = SimpleNamespace(tool_choice="auto", tools=[])

    reasoning, content, tool_calls = _unified_parser().parse(
        model_output,
        request=request,
        enable_auto_tools=True,
        model_output_token_ids=token_ids,
    )
    streamed = _unified_parser().parse_delta(
        delta_text=model_output,
        delta_token_ids=token_ids,
        request=request,
        prompt_token_ids=[],
        finished=False,
    )

    assert reasoning == "Reason."
    assert content is None
    assert tool_calls and tool_calls[0].name == "get_weather"
    assert streamed is not None
    assert streamed.reasoning == "Reason."
    assert streamed.content is None
    assert streamed.tool_calls
    assert streamed.tool_calls[0].function.name == "get_weather"
