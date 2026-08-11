# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import json
from types import SimpleNamespace

from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

from tests.test_gemma4_reasoning_parser import FakeTokenizer
from vllm_tt_plugin import platform
from vllm_tt_plugin.entrypoints import _register_tt_reasoning_parsers
from vllm_tt_plugin.gemma4_reasoning_parser import Gemma4ReasoningParser
from vllm_tt_plugin.gemma4_tool_parser import Gemma4ToolParser


def _unified_parser(*, with_tools: bool = True):
    ReasoningParserManager.register_module(
        name="diffusion_gemma",
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
        reasoning_parser_name="diffusion_gemma",
        enable_auto_tools=with_tools,
    )
    assert parser_cls is not None
    return parser_cls(FakeTokenizer(), tools=[])


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


def test_non_streaming_parser_prefers_text_when_markers_visible():
    """When the request kept special tokens (the tool parser's adjust_request
    forces skip_special_tokens=False whenever tools are enabled), the text
    path must be used so literal <|tool_call> frames survive for the tool
    parser instead of being stripped by segment re-decoding."""
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
        "<|channel>thought\nReason.<channel|><|tool_call>call:get_weather{}",
        request=None,
        model_output_token_ids=[1, 10, 2, 4, 14],
    )

    assert reasoning == "Reason."
    assert content is not None and "<|tool_call>" in content
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


def test_unified_short_reasoning_prefix_emitted_on_next_delta_end():
    parser = _unified_parser(with_tools=False)
    request = SimpleNamespace(tool_choice="none", tools=[])

    first = parser.parse_delta(
        delta_text="<|channel>tho",
        delta_token_ids=[],
        request=request,
        prompt_token_ids=[],
        finished=False,
    )
    second = parser.parse_delta(
        delta_text="<channel|>Answer.",
        delta_token_ids=[],
        request=request,
        prompt_token_ids=[],
        finished=False,
    )

    assert first is None
    assert second is not None
    assert second.reasoning == "tho"
    assert second.content == "Answer."


def test_unified_short_reasoning_prefix_emitted_before_next_delta_tool():
    parser = _unified_parser()
    request = SimpleNamespace(tool_choice="auto", tools=[])

    first = parser.parse_delta(
        delta_text="<|channel>tho",
        delta_token_ids=[],
        request=request,
        prompt_token_ids=[],
        finished=False,
    )
    second = parser.parse_delta(
        delta_text="<|tool_call>call:get_weather{}<tool_call|>",
        delta_token_ids=[],
        request=request,
        prompt_token_ids=[],
        finished=False,
    )

    assert first is None
    assert second is not None
    assert second.reasoning == "tho"
    assert second.content is None
    assert second.tool_calls
    assert second.tool_calls[0].function.name == "get_weather"


def test_unified_held_reasoning_prefix_precedes_definitive_end_id():
    tokenizer = FakeTokenizer()
    parser = _unified_parser(with_tools=False)
    request = SimpleNamespace(tool_choice="none", tools=[])

    first = parser.parse_delta(
        delta_text=tokenizer.decode([1, 10], skip_special_tokens=False),
        delta_token_ids=[1, 10],
        request=request,
        prompt_token_ids=[],
        finished=False,
    )
    held = parser.parse_delta(
        delta_text=tokenizer.decode([27], skip_special_tokens=False),
        delta_token_ids=[27],
        request=request,
        prompt_token_ids=[],
        finished=False,
    )
    final = parser.parse_delta(
        delta_text=tokenizer.decode([2, 11], skip_special_tokens=False),
        delta_token_ids=[2, 11],
        request=request,
        prompt_token_ids=[],
        finished=False,
    )

    assert first is not None and first.reasoning == "Reason."
    assert held is None
    assert final is not None
    assert final.reasoning == "<chan"
    assert final.content == "Answer."


def test_unified_streaming_emits_multiple_tools_and_visible_content():
    tokenizer = FakeTokenizer()
    token_ids = [1, 10, 2, 16, 4, 17, 6, 18, 4, 19, 6, 20]
    delta_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    result = _unified_parser().parse_delta(
        delta_text=delta_text,
        delta_token_ids=token_ids,
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        prompt_token_ids=[],
        finished=False,
    )

    assert result is not None
    assert result.reasoning == "Reason."
    assert result.content == "Before between after"
    assert result.tool_calls
    assert [tool.index for tool in result.tool_calls] == [0, 1]
    assert [tool.function.name for tool in result.tool_calls] == ["a", "b"]


def test_unified_finished_stream_flushes_short_reasoning_prefix():
    tokenizer = FakeTokenizer()
    token_ids = [1, 15]
    delta_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    result = _unified_parser(with_tools=False).parse_delta(
        delta_text=delta_text,
        delta_token_ids=token_ids,
        request=SimpleNamespace(tool_choice="none", tools=[]),
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.reasoning is None
    assert result.content == "tho"


def test_unified_finished_stream_does_not_duplicate_incomplete_reasoning():
    tokenizer = FakeTokenizer()
    token_ids = [1, 12]
    delta_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    result = _unified_parser(with_tools=False).parse_delta(
        delta_text=delta_text,
        delta_token_ids=token_ids,
        request=SimpleNamespace(tool_choice="none", tools=[]),
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.reasoning == "Part "
    assert result.content is None


def test_unified_streaming_handles_split_tool_markers():
    tokenizer = FakeTokenizer()
    parser = _unified_parser()
    request = SimpleNamespace(tool_choice="auto", tools=[])
    first_ids = [1, 10, 2, 21]
    chunks = [
        (tokenizer.decode(first_ids, skip_special_tokens=False), first_ids),
        ("call>call:a{x:1}<tool_", []),
        ("call|>After", []),
    ]
    content = ""
    tool_calls = []

    for delta_text, delta_ids in chunks:
        result = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=delta_ids,
            request=request,
            prompt_token_ids=[],
            finished=False,
        )
        if result is not None:
            content += result.content or ""
            tool_calls.extend(result.tool_calls or [])

    assert content == "BeforeAfter"
    assert "<|tool_" not in content
    assert "<tool_" not in content
    assert [call.index for call in tool_calls] == [0]
    assert tool_calls[0].function.name == "a"
    assert json.loads(tool_calls[0].function.arguments) == {"x": 1}


def test_unified_streaming_hides_held_marker_after_completed_tool():
    parser = _unified_parser()
    request = SimpleNamespace(tool_choice="auto", tools=[])
    chunks = [
        (
            "<|channel>thought\nR.<channel|>"
            "Hi <|tool_call>call:a{x:1}<tool_call|><|tool_"
        ),
        "call>call:b{}<tool_call|>",
    ]
    content = ""
    tool_calls = []

    for chunk in chunks:
        result = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=[],
            request=request,
            prompt_token_ids=[],
            finished=False,
        )
        if result is not None:
            content += result.content or ""
            tool_calls.extend(result.tool_calls or [])

    assert content == "Hi "
    assert "<|tool_" not in content
    assert [call.index for call in tool_calls] == [0, 1]
    assert [call.function.name for call in tool_calls] == ["a", "b"]


def test_unified_split_malformed_tool_never_commits_before_valid_sibling():
    parser = _unified_parser()
    request = SimpleNamespace(tool_choice="auto", tools=[])
    chunks = [
        "<|channel>thought\nR.<channel|><|tool_call>call:bad{xs:[1,",
        "}}<tool_call|><|tool_call>call:good{x:1}<tool_call|>",
    ]
    tool_calls = []

    for chunk in chunks:
        result = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=[],
            request=request,
            prompt_token_ids=[],
            finished=False,
        )
        if result is not None:
            tool_calls.extend(result.tool_calls or [])

    assert len(tool_calls) == 1
    assert tool_calls[0].index == 0
    assert tool_calls[0].id is not None
    assert tool_calls[0].function.name == "good"
    assert json.loads(tool_calls[0].function.arguments) == {"x": 1}
    assert parser._stream_state.history_tool_call_cnt == 1
    assert parser.tool_parser._raw_to_public_tool_index == {1: 0}
    assert parser.tool_parser.prev_tool_call_arr == [
        {"name": "good", "arguments": {"x": 1}}
    ]


def test_unified_strict_completed_tool_isolates_valid_sibling():
    parser = _unified_parser()
    result = parser.parse_delta(
        delta_text=(
            "<|channel>thought\nR.<channel|>"
            "<|tool_call>call:get_wea<tool_call|>"
            "<|tool_call>call:good{x:1}<tool_call|>"
        ),
        delta_token_ids=[],
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        prompt_token_ids=[],
        finished=False,
    )

    assert result is not None
    assert result.tool_calls
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].index == 0
    assert result.tool_calls[0].id is not None
    assert result.tool_calls[0].function.name == "good"
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 1}
    assert parser._stream_state.history_tool_call_cnt == 1
    assert parser.tool_parser._raw_to_public_tool_index == {1: 0}
    assert parser.tool_parser.prev_tool_call_arr == [
        {"name": "good", "arguments": {"x": 1}}
    ]


def test_unified_finish_does_not_fabricate_empty_arguments():
    tokenizer = FakeTokenizer()
    token_ids = [1, 10, 2, 24]

    result = _unified_parser().parse_delta(
        delta_text=tokenizer.decode(token_ids, skip_special_tokens=False),
        delta_token_ids=token_ids,
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.tool_calls
    assert result.tool_calls[0].function.name == "a"
    assert result.tool_calls[0].function.arguments in (None, "")


def test_unified_finish_preserves_partial_meaningful_arguments():
    tokenizer = FakeTokenizer()
    token_ids = [1, 10, 2, 25]

    result = _unified_parser().parse_delta(
        delta_text=tokenizer.decode(token_ids, skip_special_tokens=False),
        delta_token_ids=token_ids,
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.tool_calls
    assert result.tool_calls[0].function.name == "a"
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 1}


def test_unified_finish_flushes_trailing_partial_marker_as_content():
    tokenizer = FakeTokenizer()
    token_ids = [1, 10, 2, 26]

    result = _unified_parser().parse_delta(
        delta_text=tokenizer.decode(token_ids, skip_special_tokens=False),
        delta_token_ids=token_ids,
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.content == "Tail<|too"
    assert not result.tool_calls


def test_unified_finish_does_not_double_flush_completed_call():
    tokenizer = FakeTokenizer()
    token_ids = [1, 10, 2, 4, 17, 6]

    result = _unified_parser().parse_delta(
        delta_text=tokenizer.decode(token_ids, skip_special_tokens=False),
        delta_token_ids=token_ids,
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.tool_calls
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "a"
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 1}


def test_unified_all_text_split_implicit_tool_transition():
    parser = _unified_parser()
    request = SimpleNamespace(tool_choice="auto", tools=[])
    chunks = [
        "<|channel>thought\nReason.<|tool_",
        "call>call:ping{}<tool_call|>",
    ]
    reasoning = ""
    tool_calls = []

    for chunk in chunks:
        result = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=[],
            request=request,
            prompt_token_ids=[],
            finished=False,
        )
        if result is not None:
            reasoning += result.reasoning or ""
            tool_calls.extend(result.tool_calls or [])

    assert reasoning == "Reason."
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "ping"
    assert tool_calls[0].function.arguments == "{}"


def test_unified_finish_flushes_partial_reasoning_marker():
    result = _unified_parser().parse_delta(
        delta_text="<|channel>thought\nReason.<chan",
        delta_token_ids=[],
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.reasoning == "Reason.<chan"
    assert result.content is None
    assert not result.tool_calls


def test_unified_finish_open_object_does_not_fabricate_arguments():
    result = _unified_parser().parse_delta(
        delta_text=("<|channel>thought\nR.<channel|><|tool_call>call:ping{"),
        delta_token_ids=[],
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.tool_calls
    assert result.tool_calls[0].function.name == "ping"
    assert result.tool_calls[0].function.arguments in (None, "")


def test_unified_finish_closed_empty_object_emits_arguments():
    result = _unified_parser().parse_delta(
        delta_text=("<|channel>thought\nR.<channel|><|tool_call>call:ping{}"),
        delta_token_ids=[],
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        prompt_token_ids=[],
        finished=True,
    )

    assert result is not None
    assert result.tool_calls
    assert result.tool_calls[0].function.name == "ping"
    assert result.tool_calls[0].function.arguments == "{}"
