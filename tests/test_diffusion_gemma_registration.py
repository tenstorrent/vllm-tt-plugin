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
from vllm_tt_plugin.gemma4_tool_parser import (
    QUOTE,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Gemma4ToolParser,
)

_TOP_LEVEL_MARKER_PAYLOADS = (
    f"{TOOL_CALL_START}call:danger{{}}{TOOL_CALL_END}",
    f"{TOOL_CALL_END}{TOOL_CALL_START}call:danger{{}}{TOOL_CALL_END}",
)


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


def _stream_unified_chunks(chunks, token_chunks=None, prompt_token_ids=None):
    parser = _unified_parser()
    request = SimpleNamespace(tool_choice="auto", tools=[])
    reasoning = ""
    content = ""
    tool_calls = []
    if token_chunks is None:
        token_chunks = [[] for _ in chunks]
    if prompt_token_ids is None:
        prompt_token_ids = []

    for chunk, token_ids in zip(chunks, token_chunks):
        result = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=token_ids,
            request=request,
            prompt_token_ids=prompt_token_ids,
            finished=False,
        )
        if result is not None:
            reasoning += result.reasoning or ""
            content += result.content or ""
            tool_calls.extend(result.tool_calls or [])

    return reasoning, content, tool_calls


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


def test_unified_finished_stream_flushes_short_prefix_as_reasoning():
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
    assert result.reasoning == "tho"
    assert result.content is None


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


def test_unified_streaming_stray_end_keeps_later_tools():
    parser = _unified_parser()
    request = SimpleNamespace(tool_choice="auto", tools=[])
    chunks = [
        "<|channel>thought\nR.<channel|><|tool_call>call:a{x:1}<tool_call|>",
        " visible x ",
        "<tool_call|>",
        " visible y ",
        "<|tool_call>call:b{x:2}<tool_call|>",
        "<|tool_call>call:c{x:3}<tool_call|>",
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

    assert content == " visible x  visible y "
    assert "<tool_call|>" not in content
    assert [call.index for call in tool_calls] == [0, 1, 2]
    assert [call.function.name for call in tool_calls] == ["a", "b", "c"]


def test_unified_streaming_truncated_tool_resyncs_to_valid_sibling():
    parser = _unified_parser()
    request = SimpleNamespace(tool_choice="auto", tools=[])
    chunks = [
        ("<|channel>thought\nR.<channel|>Before <|tool_call>call:bad{x:1"),
        "<|tool_call>call:good{x:2}<tool_call|> after",
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

    assert content == "Before  after"
    assert len(tool_calls) == 1
    assert tool_calls[0].index == 0
    assert tool_calls[0].function.name == "good"
    assert json.loads(tool_calls[0].function.arguments) == {"x": 2}
    assert parser.tool_parser._raw_to_public_tool_index == {1: 0}


def test_unified_streaming_preserves_paired_quoted_tool_markers():
    parser = _unified_parser()
    request = SimpleNamespace(tool_choice="auto", tools=[])
    chunks = [
        ('<|channel>thought\nR.<channel|><|tool_call>call:a{q:<|"|>before<tool_call|>'),
        "<|tool_call>call:b{x:2}<tool_call|>after",
        '<|"|>}<tool_call|>',
    ]
    content = ""
    tool_calls = []
    interim_tool_deltas = []

    for chunk in chunks:
        result = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=[],
            request=request,
            prompt_token_ids=[],
            finished=False,
        )
        interim_tool_deltas.append(
            list(result.tool_calls or []) if result is not None else []
        )
        if result is not None:
            content += result.content or ""
            tool_calls.extend(result.tool_calls or [])

    assert interim_tool_deltas[:2] == [[], []]
    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].index == 0
    assert tool_calls[0].function.name == "a"
    assert json.loads(tool_calls[0].function.arguments) == {
        "q": ("before<tool_call|><|tool_call>call:b{x:2}<tool_call|>after")
    }


def test_unified_streaming_top_level_paired_quotes_stay_literal():
    request = SimpleNamespace(tool_choice="auto", tools=[])

    for payload in _TOP_LEVEL_MARKER_PAYLOADS:
        parser = _unified_parser()
        chunks = [
            f"<|channel>thought\nR.<channel|>Before {QUOTE}",
            payload,
            f"{QUOTE} after",
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

        assert content == f"Before {QUOTE}{payload}{QUOTE} after"
        assert tool_calls == []


def test_unified_streaming_top_level_unclosed_quote_stays_literal_at_finish():
    request = SimpleNamespace(tool_choice="auto", tools=[])

    for payload in _TOP_LEVEL_MARKER_PAYLOADS:
        parser = _unified_parser()
        chunks = [
            f"<|channel>thought\nR.<channel|>Before {QUOTE}",
            payload,
            " after",
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

        finished = parser.parse_delta(
            delta_text="",
            delta_token_ids=[],
            request=request,
            prompt_token_ids=[],
            finished=True,
        )
        if finished is not None:
            content += finished.content or ""
            tool_calls.extend(finished.tool_calls or [])

        assert content == f"Before {QUOTE}{payload} after"
        assert tool_calls == []


def test_unified_streaming_real_call_after_closed_top_level_quote():
    payload = _TOP_LEVEL_MARKER_PAYLOADS[0]
    parser = _unified_parser()
    request = SimpleNamespace(tool_choice="auto", tools=[])
    chunks = [
        f"<|channel>thought\nR.<channel|>Before {QUOTE}",
        payload,
        (f"{QUOTE} middle {TOOL_CALL_START}call:good{{x:1}}{TOOL_CALL_END} after"),
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

    assert content == f"Before {QUOTE}{payload}{QUOTE} middle  after"
    assert [call.index for call in tool_calls] == [0]
    assert [call.function.name for call in tool_calls] == ["good"]


def test_unified_no_channel_quote_handoff_is_chunk_invariant():
    for payload in _TOP_LEVEL_MARKER_PAYLOADS:
        expected = f"Before {QUOTE}{payload}{QUOTE} after"
        chunkings = (
            [expected],
            [f"Before {QUOTE}", payload, f"{QUOTE} after"],
        )

        for chunks in chunkings:
            reasoning, content, tool_calls = _stream_unified_chunks(chunks)

            assert reasoning == ""
            assert content == expected
            assert tool_calls == []


def test_unified_no_channel_real_tool_after_closed_quote():
    danger = _TOP_LEVEL_MARKER_PAYLOADS[0]
    real_tool = f"{TOOL_CALL_START}call:good{{x:1}}{TOOL_CALL_END}"
    chunks = [
        f"Before {QUOTE}",
        danger,
        f"{QUOTE} middle {real_tool} after",
    ]

    reasoning, content, tool_calls = _stream_unified_chunks(chunks)

    assert reasoning == ""
    assert content == f"Before {QUOTE}{danger}{QUOTE} middle  after"
    assert [call.function.name for call in tool_calls] == ["good"]
    assert all(call.function.name != "danger" for call in tool_calls)


def test_unified_no_channel_quote_handoff_with_real_token_ids():
    tokenizer = FakeTokenizer()
    payload_ids = (
        [4, 29, 6],
        [6, 4, 29, 6],
    )

    for marker_ids in payload_ids:
        token_chunks = ([28, 7], marker_ids, [7, 32])
        chunks = [
            tokenizer.decode(ids, skip_special_tokens=False) for ids in token_chunks
        ]
        reasoning, content, tool_calls = _stream_unified_chunks(chunks, token_chunks)

        assert reasoning == ""
        assert content == "".join(chunks)
        assert tool_calls == []

    real_tool_ids = [4, 31, 6]
    token_chunks = ([28, 7], payload_ids[0], [7, 30, *real_tool_ids, 32])
    chunks = [tokenizer.decode(ids, skip_special_tokens=False) for ids in token_chunks]
    reasoning, content, tool_calls = _stream_unified_chunks(chunks, token_chunks)

    assert reasoning == ""
    assert content == (
        f"Before {QUOTE}{TOOL_CALL_START}call:danger{{}}{TOOL_CALL_END}"
        f"{QUOTE} middle  after"
    )
    assert [call.function.name for call in tool_calls] == ["good"]
    assert all(call.function.name != "danger" for call in tool_calls)


def test_unified_nonstream_open_reasoning_quotes_are_literal():
    tokenizer = FakeTokenizer()
    payloads = (
        (
            _TOP_LEVEL_MARKER_PAYLOADS[0],
            [4, 29, 6],
        ),
        (
            _TOP_LEVEL_MARKER_PAYLOADS[1],
            [6, 4, 29, 6],
        ),
    )

    for payload, payload_ids in payloads:
        quoted = f"{QUOTE}{payload}{QUOTE}"
        reasoning, content, tool_calls = _unified_parser().parse(
            f"<|channel>thought\nReason.{quoted}",
            request=SimpleNamespace(tool_choice="auto", tools=[]),
            enable_auto_tools=True,
        )

        assert reasoning == f"Reason.{quoted}"
        assert content is None
        assert not tool_calls

        token_ids = [1, 10, 7, *payload_ids, 7]
        reasoning, content, tool_calls = _unified_parser().parse(
            tokenizer.decode(token_ids, skip_special_tokens=True),
            request=SimpleNamespace(tool_choice="auto", tools=[]),
            enable_auto_tools=True,
            model_output_token_ids=token_ids,
        )

        assert reasoning == f"Reason.{quoted}"
        assert content is None
        assert not tool_calls

    real_tool = f"{TOOL_CALL_START}call:good{{}}{TOOL_CALL_END}"
    quoted_danger = f"{QUOTE}{_TOP_LEVEL_MARKER_PAYLOADS[0]}{QUOTE}"
    reasoning, content, tool_calls = _unified_parser().parse(
        f"<|channel>thought\nReason.{quoted_danger} middle {real_tool} after",
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        enable_auto_tools=True,
    )

    assert reasoning == f"Reason.{quoted_danger} middle "
    assert content == " after"
    assert [call.name for call in tool_calls] == ["good"]
    assert all(call.name != "danger" for call in tool_calls)

    token_ids = [1, 10, 7, 4, 29, 6, 7, 30, 4, 31, 6, 32]
    reasoning, content, tool_calls = _unified_parser().parse(
        tokenizer.decode(token_ids, skip_special_tokens=True),
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        enable_auto_tools=True,
        model_output_token_ids=token_ids,
    )

    assert reasoning == f"Reason.{quoted_danger} middle "
    assert content == " after"
    assert [call.name for call in tool_calls] == ["good"]
    assert all(call.name != "danger" for call in tool_calls)


def test_unified_fragmented_reasoning_quotes_are_chunk_invariant():
    danger = _TOP_LEVEL_MARKER_PAYLOADS[0]
    good = f"{TOOL_CALL_START}call:good{{}}{TOOL_CALL_END}"
    prefix = "<|channel>thought\nR."
    expected_reasoning = f"R.{QUOTE}{danger}{QUOTE}"
    chunkings = [[f"{prefix}{QUOTE}{danger}{QUOTE}{good}"]]

    for split in range(1, len(QUOTE)):
        chunkings.append(
            [
                f"{prefix}{QUOTE[:split]}",
                f"{QUOTE[split:]}{danger}{QUOTE}{good}",
            ]
        )
        chunkings.append(
            [
                f"{prefix}{QUOTE}{danger}{QUOTE[:split]}",
                f"{QUOTE[split:]}{good}",
            ]
        )

    for chunks in chunkings:
        reasoning, content, tool_calls = _stream_unified_chunks(chunks)

        assert reasoning == expected_reasoning
        assert content == ""
        assert [call.function.name for call in tool_calls] == ["good"]
        assert all(call.function.name != "danger" for call in tool_calls)


def test_unified_empty_prompt_omitted_start_is_chunk_invariant():
    chunkings = (
        ["thought\nR.<channel|>A"],
        ["tho", "ught\nR.", "<channel|>A"],
        ["thought", "\nR.<chan", "nel|>A"],
        ["tho", "ught\nR.<chan", "nel|>A"],
        ["thought\nR.<chan", "nel|>A"],
    )

    for chunks in chunkings:
        reasoning, content, tool_calls = _stream_unified_chunks(
            chunks,
            prompt_token_ids=[],
        )

        assert reasoning == "R."
        assert content == "A"
        assert tool_calls == []

    tokenizer = FakeTokenizer()
    token_ids = [12, 2, 11]
    reasoning, content, tool_calls = _stream_unified_chunks(
        [tokenizer.decode(token_ids, skip_special_tokens=False)],
        [token_ids],
        prompt_token_ids=[],
    )

    assert reasoning == "Part "
    assert content == "Answer."
    assert tool_calls == []


def test_unified_empty_prompt_quoted_end_and_plain_controls():
    quoted_end = f"Before {QUOTE}<channel|>{QUOTE} after"
    for text in (quoted_end, "Plain answer."):
        reasoning, content, tool_calls = _stream_unified_chunks(
            [text],
            prompt_token_ids=[],
        )

        assert reasoning == ""
        assert content == text
        assert tool_calls == []


def test_unified_prompt_open_omitted_start_text_and_ids():
    reasoning, content, tool_calls = _stream_unified_chunks(
        ["thought\nR.<channel|>A"],
        prompt_token_ids=[1],
    )

    assert reasoning == "R."
    assert content == "A"
    assert tool_calls == []

    tokenizer = FakeTokenizer()
    token_ids = [12, 2, 11]
    reasoning, content, tool_calls = _stream_unified_chunks(
        [tokenizer.decode(token_ids, skip_special_tokens=False)],
        [token_ids],
        prompt_token_ids=[1],
    )

    assert reasoning == "Part "
    assert content == "Answer."
    assert tool_calls == []


def test_unified_prompt_quote_carry_blocks_danger_until_close():
    danger = _TOP_LEVEL_MARKER_PAYLOADS[0]
    good = f"{TOOL_CALL_START}call:good{{}}{TOOL_CALL_END}"

    for split in range(1, len(QUOTE)):
        reasoning, content, tool_calls = _stream_unified_chunks(
            [f"{danger}{QUOTE[:split]}", f"{QUOTE[split:]}{good}"],
            prompt_token_ids=[1, 7],
        )

        assert reasoning == f"{danger}{QUOTE}"
        assert content == ""
        assert [call.function.name for call in tool_calls] == ["good"]
        assert all(call.function.name != "danger" for call in tool_calls)

    tokenizer = FakeTokenizer()
    token_ids = [4, 29, 6, 7, 4, 31, 6]
    reasoning, content, tool_calls = _stream_unified_chunks(
        [tokenizer.decode(token_ids, skip_special_tokens=False)],
        [token_ids],
        prompt_token_ids=[1, 7],
    )

    assert reasoning == f"{danger}{QUOTE}"
    assert content == ""
    assert [call.function.name for call in tool_calls] == ["good"]
    assert all(call.function.name != "danger" for call in tool_calls)


def test_unified_prompt_new_turn_reset_and_empty_control():
    reasoning, content, tool_calls = _stream_unified_chunks(
        ["thought\nR.<channel|>A"],
        prompt_token_ids=[1, 7, 3],
    )

    assert reasoning == "R."
    assert content == "A"
    assert tool_calls == []

    reasoning, content, tool_calls = _stream_unified_chunks(
        ["Plain answer."],
        prompt_token_ids=[],
    )

    assert reasoning == ""
    assert content == "Plain answer."
    assert tool_calls == []


def test_unified_unterminated_quote_defers_sibling_until_finish():
    parser = _unified_parser()
    request = SimpleNamespace(tool_choice="auto", tools=[])
    chunks = [
        (
            "<|channel>thought\nR.<channel|>"
            'Before <|tool_call>call:bad{x:<|"|>unterminated'
        ),
        "<|tool_call>call:good{x:2}<tool_call|> after",
    ]
    content = ""
    streamed_calls = []

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
            streamed_calls.extend(result.tool_calls or [])

    assert streamed_calls == []
    finished = parser.parse_delta(
        delta_text="",
        delta_token_ids=[],
        request=request,
        prompt_token_ids=[],
        finished=True,
    )

    assert finished is not None
    content += finished.content or ""
    assert content == "Before  after"
    assert len(finished.tool_calls or []) == 1
    assert finished.tool_calls[0].index == 0
    assert finished.tool_calls[0].function.name == "good"
    assert json.loads(finished.tool_calls[0].function.arguments) == {"x": 2}


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


def test_unified_nonstream_all_malformed_tool_uses_cleaned_content():
    parser = _unified_parser()
    reasoning, content, tool_calls = parser.parse(
        (
            "<|channel>thought\nR.<channel|>"
            "hello <|tool_call>call:f{x:1]}<tool_call|> bye"
        ),
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        enable_auto_tools=True,
    )

    assert reasoning == "R."
    assert content == "hello  bye"
    assert tool_calls == []


def test_unified_nonstream_truncated_tool_resyncs_to_valid_sibling():
    reasoning, content, tool_calls = _unified_parser().parse(
        (
            "<|channel>thought\nR.<channel|>"
            "Before <|tool_call>call:bad{x:1"
            "<|tool_call>call:good{x:2}<tool_call|> after"
        ),
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        enable_auto_tools=True,
    )

    assert reasoning == "R."
    assert content == "Before  after"
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "good"
    assert json.loads(tool_calls[0].arguments) == {"x": 2}


def test_unified_nonstream_cleans_lone_unmatched_tool_end():
    reasoning, content, tool_calls = _unified_parser().parse(
        "<|channel>thought\nR.<channel|>Before <tool_call|> after",
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        enable_auto_tools=True,
    )

    assert reasoning == "R."
    assert content == "Before  after"
    assert tool_calls == []


def test_unified_nonstream_top_level_paired_quotes_stay_literal():
    for payload in _TOP_LEVEL_MARKER_PAYLOADS:
        text = f"Before {QUOTE}{payload}{QUOTE} after"
        reasoning, content, tool_calls = _unified_parser().parse(
            f"<|channel>thought\nR.<channel|>{text}",
            request=SimpleNamespace(tool_choice="auto", tools=[]),
            enable_auto_tools=True,
        )

        assert reasoning == "R."
        assert content == text
        assert not tool_calls


def test_unified_nonstream_top_level_unclosed_quotes_stay_literal():
    for payload in _TOP_LEVEL_MARKER_PAYLOADS:
        text = f"Before {QUOTE}{payload} after"
        reasoning, content, tool_calls = _unified_parser().parse(
            f"<|channel>thought\nR.<channel|>{text}",
            request=SimpleNamespace(tool_choice="auto", tools=[]),
            enable_auto_tools=True,
        )

        assert reasoning == "R."
        assert content == text
        assert not tool_calls


def test_unified_nonstream_real_call_after_closed_top_level_quote():
    payload = _TOP_LEVEL_MARKER_PAYLOADS[0]
    quoted = f"{QUOTE}{payload}{QUOTE}"
    reasoning, content, tool_calls = _unified_parser().parse(
        (
            f"<|channel>thought\nR.<channel|>Before {quoted} middle "
            f"{TOOL_CALL_START}call:good{{x:1}}{TOOL_CALL_END} after"
        ),
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        enable_auto_tools=True,
    )

    assert reasoning == "R."
    assert content == f"Before {quoted} middle  after"
    assert [call.name for call in tool_calls] == ["good"]


def test_unified_nonstream_preserves_paired_quoted_tool_markers():
    reasoning, content, tool_calls = _unified_parser().parse(
        (
            "<|channel>thought\nR.<channel|>"
            '<|tool_call>call:a{q:<|"|>before<tool_call|>'
            "<|tool_call>call:b{x:2}<tool_call|>"
            'after<|"|>}<tool_call|>'
        ),
        request=SimpleNamespace(tool_choice="auto", tools=[]),
        enable_auto_tools=True,
    )

    assert reasoning == "R."
    assert content is None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "a"
    assert json.loads(tool_calls[0].arguments) == {
        "q": ("before<tool_call|><|tool_call>call:b{x:2}<tool_call|>after")
    }


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
