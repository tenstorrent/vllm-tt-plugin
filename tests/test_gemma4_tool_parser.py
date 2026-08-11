# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Unit tests for the Gemma4 tool-call parser.

The parser is stateless for ``extract_tool_calls`` and does not touch the
tokenizer there, so the tests instantiate it with ``tokenizer=None`` and pass
``request=None`` (unused by extraction).
"""

import json
import signal
from contextlib import contextmanager

import pytest
from openai.types.responses import ToolChoiceFunction
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

from vllm_tt_plugin.gemma4_tool_parser import Gemma4ToolParser


@pytest.fixture
def parser() -> Gemma4ToolParser:
    return Gemma4ToolParser(tokenizer=None)


@contextmanager
def _fail_if_parser_stalls(seconds: float = 1.0):
    def _raise_timeout(_signum, _frame):
        raise TimeoutError("parser did not make progress")

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def test_constructor_accepts_delegating_parser_tools():
    parser = Gemma4ToolParser(tokenizer=None, tools=[])

    assert parser.tools == []


def test_required_and_named_requests_keep_native_tool_syntax(
    parser: Gemma4ToolParser,
):
    named = ChatCompletionNamedToolChoiceParam(
        type="function",
        function={"name": "f"},
    )
    response_named = ToolChoiceFunction(type="function", name="f")
    requests = [
        ChatCompletionRequest.model_construct(
            tools=[object()],
            tool_choice="required",
            skip_special_tokens=True,
            structured_outputs=None,
        ),
        ChatCompletionRequest.model_construct(
            tools=[object()],
            tool_choice=named,
            skip_special_tokens=True,
            structured_outputs=None,
        ),
        ResponsesRequest.model_construct(
            tools=[object()],
            tool_choice=response_named,
            skip_special_tokens=True,
            structured_outputs=None,
        ),
    ]

    assert parser.supports_required_and_named is False
    for request in requests:
        adjusted = parser.adjust_request(request)
        assert adjusted.skip_special_tokens is False
        assert adjusted.structured_outputs is None


def test_auto_request_keeps_special_tokens(parser: Gemma4ToolParser):
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "test"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
    )

    adjusted = parser.adjust_request(request)

    assert adjusted.skip_special_tokens is False


def test_no_tool_call_returns_content(parser: Gemma4ToolParser):
    out = "just a normal answer, no tools here"
    result = parser.extract_tool_calls(out, request=None)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == out


def test_single_call_string_arg(parser: Gemma4ToolParser):
    out = '<|tool_call>call:get_weather{location:<|"|>Paris, FR<|"|>}<tool_call|>'
    result = parser.extract_tool_calls(out, request=None)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"location": "Paris, FR"}


def _args(parser: Gemma4ToolParser, out: str) -> dict:
    result = parser.extract_tool_calls(out, request=None)
    return json.loads(result.tool_calls[0].function.arguments)


def test_mixed_scalar_args(parser: Gemma4ToolParser):
    out = '<|tool_call>call:f{flag:true,n:3,r:2.5,s:<|"|>hi<|"|>,z:false}<tool_call|>'
    assert _args(parser, out) == {
        "flag": True,
        "n": 3,
        "r": 2.5,
        "s": "hi",
        "z": False,
    }


def test_nested_object_and_arrays(parser: Gemma4ToolParser):
    out = (
        '<|tool_call>call:g{a:{x:1,y:<|"|>q<|"|>},'
        'xs:[1,2,3],ss:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>'
    )
    assert _args(parser, out) == {
        "a": {"x": 1, "y": "q"},
        "xs": [1, 2, 3],
        "ss": ["a", "b"],
    }


def test_empty_args(parser: Gemma4ToolParser):
    out = "<|tool_call>call:ping{}<tool_call|>"
    result = parser.extract_tool_calls(out, request=None)
    assert result.tool_calls[0].function.name == "ping"
    assert json.loads(result.tool_calls[0].function.arguments) == {}


def test_string_value_with_delimiters(parser: Gemma4ToolParser):
    # The quote token makes the string atomic, so embedded {}/[]/, are literal.
    out = '<|tool_call>call:h{q:<|"|>a{b},c[d]<|"|>}<tool_call|>'
    assert _args(parser, out) == {"q": "a{b},c[d]"}


def test_multiple_calls_and_leading_content(parser: Gemma4ToolParser):
    out = (
        "Sure!<|tool_call>call:a{x:1}<tool_call|>"
        '<|tool_call>call:b{y:<|"|>v<|"|>}<tool_call|>'
    )
    result = parser.extract_tool_calls(out, request=None)
    assert result.tools_called is True
    assert result.content == "Sure!"
    assert [c.function.name for c in result.tool_calls] == ["a", "b"]
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 1}
    assert json.loads(result.tool_calls[1].function.arguments) == {"y": "v"}


def test_non_streaming_preserves_all_content_and_skips_nameless_calls(
    parser: Gemma4ToolParser,
):
    out = (
        "Before"
        "<|tool_call>call:a{x:1}<tool_call|>"
        " middle "
        "<|tool_call>not-a-call{x:2}<tool_call|>"
        " between "
        "<|tool_call>call:b{y:3}<tool_call|>"
        "after"
    )

    result = parser.extract_tool_calls(out, request=None)

    assert result.tools_called is True
    assert [call.function.name for call in result.tool_calls] == ["a", "b"]
    assert result.content == "Before middle  between after"


def test_non_streaming_isolates_malformed_sibling_calls(
    parser: Gemma4ToolParser,
):
    out = (
        "Before "
        "<|tool_call>call:a{x:1}<tool_call|>"
        "<|tool_call>call:bad{xs:[1,}}<tool_call|>"
        "<|tool_call>call:b{y:2}<tool_call|>"
        " after"
    )

    with _fail_if_parser_stalls():
        result = parser.extract_tool_calls(out, request=None)

    assert result.tools_called is True
    assert result.content == "Before  after"
    assert [call.function.name for call in result.tool_calls] == ["a", "b"]
    assert [json.loads(call.function.arguments) for call in result.tool_calls] == [
        {"x": 1},
        {"y": 2},
    ]
    assert "call:bad" not in result.content


_STRICTLY_MALFORMED_COMPLETED_BODIES = [
    pytest.param("call:get_wea", id="missing-opening-brace"),
    pytest.param("call:bad{x:1", id="missing-outer-close"),
    pytest.param(
        'call:bad{x:<|"|>unterminated}',
        id="unterminated-quote-token",
    ),
    pytest.param("call:bad{xs:[1,2}", id="mismatched-array-close"),
]


@pytest.mark.parametrize(
    "malformed_body",
    _STRICTLY_MALFORMED_COMPLETED_BODIES,
)
def test_non_streaming_strict_completed_frame_isolates_valid_sibling(
    parser: Gemma4ToolParser,
    malformed_body: str,
):
    out = (
        f"<|tool_call>{malformed_body}<tool_call|>"
        "<|tool_call>call:good{x:1}<tool_call|>"
    )

    result = parser.extract_tool_calls(out, request=None)

    assert result.tools_called is True
    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "good"
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 1}


def test_incomplete_block_is_ignored(parser: Gemma4ToolParser):
    # No closing <tool_call|>: not emitted as a completed call.
    out = "<|tool_call>call:a{x:1"
    result = parser.extract_tool_calls(out, request=None)
    assert result.tools_called is False
    assert result.content is None


def test_non_streaming_truncated_frame_does_not_leak_special_tokens(
    parser: Gemma4ToolParser,
):
    out = "Visible <|tool_call>call:a{x:1"

    result = parser.extract_tool_calls(out, request=None)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == "Visible "
    assert "<|tool_call>" not in result.content


def test_streaming_assembles_name_and_args(parser: Gemma4ToolParser):
    full = (
        "<|tool_call>call:get_weather"
        '{location:<|"|>Paris<|"|>,units:<|"|>c<|"|>}<tool_call|>'
    )
    # Chunk on boundaries that do not split a special token.
    chunks = [
        "<|tool_call>",
        "call:get_weather{location:",
        '<|"|>Paris<|"|>',
        ",units:",
        '<|"|>c<|"|>',
        "}",
        "<tool_call|>",
    ]
    assert "".join(chunks) == full

    name = None
    args_acc = ""
    prev = ""
    for chunk in chunks:
        cur = prev + chunk
        delta = parser.extract_tool_calls_streaming(
            prev, cur, chunk, [], [], [], request=None
        )
        if delta is not None and delta.tool_calls:
            fn = delta.tool_calls[0].function
            if fn is not None and fn.name:
                name = fn.name
            if fn is not None and fn.arguments:
                args_acc += fn.arguments
        prev = cur

    assert name == "get_weather"
    assert json.loads(args_acc) == {"location": "Paris", "units": "c"}


def test_streaming_emits_multiple_calls_and_surrounding_content(
    parser: Gemma4ToolParser,
):
    delta_text = (
        "Before"
        "<|tool_call>call:a{x:1}<tool_call|>"
        " between "
        "<|tool_call>call:b{y:2}<tool_call|>"
        "after"
    )

    result = parser.extract_tool_calls_streaming(
        "", delta_text, delta_text, [], [], [], request=None
    )

    assert result is not None
    assert result.content == "Before between after"
    assert result.tool_calls
    assert [tool.index for tool in result.tool_calls] == [0, 1]
    assert [tool.function.name for tool in result.tool_calls] == ["a", "b"]
    assert [json.loads(tool.function.arguments) for tool in result.tool_calls] == [
        {"x": 1},
        {"y": 2},
    ]


def test_streaming_isolates_malformed_sibling_calls(
    parser: Gemma4ToolParser,
):
    text = (
        "Before "
        "<|tool_call>call:a{x:1}<tool_call|>"
        "<|tool_call>call:bad{xs:[1,}}<tool_call|>"
        "<|tool_call>call:b{y:2}<tool_call|>"
        " after"
    )

    with _fail_if_parser_stalls():
        result = parser.extract_tool_calls_streaming(
            "", text, text, [], [], [], request=None
        )

    assert result is not None
    assert result.content == "Before  after"
    assert result.tool_calls
    assert [call.index for call in result.tool_calls] == [0, 1]
    assert [call.function.name for call in result.tool_calls] == ["a", "b"]
    assert [json.loads(call.function.arguments) for call in result.tool_calls] == [
        {"x": 1},
        {"y": 2},
    ]
    assert parser.streamed_args_for_tool == ['{"x": 1}', '{"y": 2}']
    assert parser.prev_tool_call_arr == [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]
    assert parser._raw_to_public_tool_index == {0: 0, 2: 1}


@pytest.mark.parametrize(
    "malformed_body",
    _STRICTLY_MALFORMED_COMPLETED_BODIES,
)
def test_streaming_strict_completed_frame_isolates_valid_sibling(
    parser: Gemma4ToolParser,
    malformed_body: str,
):
    text = (
        f"<|tool_call>{malformed_body}<tool_call|>"
        "<|tool_call>call:good{x:1}<tool_call|>"
    )

    result = parser.extract_tool_calls_streaming(
        "", text, text, [], [], [], request=None
    )

    assert result is not None
    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].index == 0
    assert result.tool_calls[0].id is not None
    assert result.tool_calls[0].function.name == "good"
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 1}
    assert parser._raw_to_public_tool_index == {1: 0}
    assert parser.streamed_args_for_tool == ['{"x": 1}']
    assert parser.prev_tool_call_arr == [{"name": "good", "arguments": {"x": 1}}]


def test_split_malformed_call_never_commits_before_valid_sibling(
    parser: Gemma4ToolParser,
):
    chunks = [
        "<|tool_call>call:bad{xs:[1,",
        "}}<tool_call|><|tool_call>call:good{x:1}<tool_call|>",
    ]
    previous_text = ""
    results = []
    tool_calls = []

    for chunk in chunks:
        current_text = previous_text + chunk
        result = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            chunk,
            [],
            [],
            [],
            request=None,
        )
        results.append(result)
        if result is not None:
            tool_calls.extend(result.tool_calls or [])
        previous_text = current_text

    assert results[0] is None
    assert len(tool_calls) == 1
    assert tool_calls[0].index == 0
    assert tool_calls[0].id is not None
    assert tool_calls[0].function.name == "good"
    assert json.loads(tool_calls[0].function.arguments) == {"x": 1}
    assert parser._raw_to_public_tool_index == {1: 0}
    assert parser.streamed_args_for_tool == ['{"x": 1}']
    assert parser.prev_tool_call_arr == [{"name": "good", "arguments": {"x": 1}}]


def test_streaming_reuses_contiguous_index_after_malformed_raw_block(
    parser: Gemma4ToolParser,
):
    chunks = [
        "<|tool_call>call:a{x:1}<tool_call|>",
        "<|tool_call>call:bad{xs:[1,}}<tool_call|><|tool_call>call:b{",
        "y:2}<tool_call|>",
    ]
    previous_text = ""
    tool_calls = []

    for chunk in chunks:
        current_text = previous_text + chunk
        result = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            chunk,
            [],
            [],
            [],
            request=None,
        )
        if result is not None:
            tool_calls.extend(result.tool_calls or [])
        previous_text = current_text

    assert [call.index for call in tool_calls] == [0, 1]
    assert [call.function.name for call in tool_calls] == ["a", "b"]
    assert json.loads(tool_calls[-1].function.arguments) == {"y": 2}
    assert parser._raw_to_public_tool_index == {0: 0, 2: 1}


def test_finish_streaming_reuses_contiguous_index_after_malformed_raw_block(
    parser: Gemma4ToolParser,
):
    chunks = [
        "<|tool_call>call:a{x:1}<tool_call|>",
        "<|tool_call>call:bad{xs:[1,}}<tool_call|><|tool_call>call:b{y:2",
    ]
    previous_text = ""
    tool_calls = []

    for chunk in chunks:
        current_text = previous_text + chunk
        result = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            chunk,
            [],
            [],
            [],
            request=None,
        )
        if result is not None:
            tool_calls.extend(result.tool_calls or [])
        previous_text = current_text

    final = parser.finish_streaming()
    assert final is not None
    tool_calls.extend(final.tool_calls or [])

    assert [call.index for call in tool_calls] == [0, 1]
    assert tool_calls[1].function.name == "b"
    assert json.loads(tool_calls[1].function.arguments) == {"y": 2}
    assert parser._raw_to_public_tool_index == {0: 0, 2: 1}


def test_streaming_skips_nameless_completed_call(parser: Gemma4ToolParser):
    delta_text = "<|tool_call>not-a-call{x:1}<tool_call|>"

    result = parser.extract_tool_calls_streaming(
        "", delta_text, delta_text, [], [], [], request=None
    )

    assert result is None


def test_streaming_split_markers_do_not_duplicate_or_leak(
    parser: Gemma4ToolParser,
):
    chunks = [
        "Before<|tool_",
        "call>call:a{x:1}<tool_",
        "call|>After",
    ]
    previous_text = ""
    content = ""
    tool_calls = []

    for chunk in chunks:
        current_text = previous_text + chunk
        result = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            chunk,
            [],
            [],
            [],
            request=None,
        )
        if result is not None:
            content += result.content or ""
            tool_calls.extend(result.tool_calls or [])
        previous_text = current_text

    assert content == "BeforeAfter"
    assert "<|tool_" not in content
    assert "<tool_" not in content
    assert [call.index for call in tool_calls] == [0]
    assert tool_calls[0].function.name == "a"
    assert json.loads(tool_calls[0].function.arguments) == {"x": 1}


def test_streaming_hides_newly_buffered_suffix_after_completed_call(
    parser: Gemma4ToolParser,
):
    chunks = [
        "Hi <|tool_call>call:a{x:1}<tool_call|><|tool_",
        "call>call:b{}<tool_call|>",
    ]
    previous_text = ""
    content = ""
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}

    for chunk in chunks:
        current_text = previous_text + chunk
        result = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            chunk,
            [],
            [],
            [],
            request=None,
        )
        if result is not None:
            content += result.content or ""
            for call in result.tool_calls or []:
                if call.function.name:
                    names[call.index] = call.function.name
                if call.function.arguments:
                    arguments[call.index] = (
                        arguments.get(call.index, "") + call.function.arguments
                    )
        previous_text = current_text

    assert content == "Hi "
    assert "<|tool_" not in content
    assert names == {0: "a", 1: "b"}
    assert {index: json.loads(value) for index, value in arguments.items()} == {
        0: {"x": 1},
        1: {},
    }


def test_streaming_keeps_visible_content_while_function_name_is_incomplete(
    parser: Gemma4ToolParser,
):
    chunks = [
        "Let me check. <|tool_call>call:",
        "get_weather{}<tool_call|>",
    ]
    previous_text = ""
    content = ""
    names: list[str] = []
    arguments = ""

    for chunk in chunks:
        current_text = previous_text + chunk
        result = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            chunk,
            [],
            [],
            [],
            request=None,
        )
        if result is not None:
            content += result.content or ""
            for call in result.tool_calls or []:
                if call.function.name:
                    names.append(call.function.name)
                if call.function.arguments:
                    arguments += call.function.arguments
        previous_text = current_text

    assert content == "Let me check. "
    assert names == ["get_weather"]
    assert json.loads(arguments) == {}


def test_malformed_nested_array_returns_without_stalling(
    parser: Gemma4ToolParser,
):
    text = "<|tool_call>call:f{xs:[1,}}<tool_call|>"

    with _fail_if_parser_stalls():
        result = parser.extract_tool_calls(text, request=None)

    assert result.tools_called is False
    assert result.content is None


def test_malformed_nested_array_direct_parser_fails_without_stalling(
    parser: Gemma4ToolParser,
):
    with _fail_if_parser_stalls(), pytest.raises(ValueError):
        parser._parse_call("call:f{xs:[1,}}")


def test_malformed_nested_array_streaming_returns_without_stalling(
    parser: Gemma4ToolParser,
):
    text = "<|tool_call>call:f{xs:[1,}}<tool_call|>"

    with _fail_if_parser_stalls():
        result = parser.extract_tool_calls_streaming(
            "", text, text, [], [], [], request=None
        )

    assert result is None


def test_malformed_unfinished_stream_finalizes_without_stalling(
    parser: Gemma4ToolParser,
):
    text = "<|tool_call>call:f{xs:[1,}}"
    parser.extract_tool_calls_streaming("", text, text, [], [], [], request=None)

    with _fail_if_parser_stalls():
        result = parser.finish_streaming()

    assert result is None


def test_finish_streaming_does_not_invent_half_function_name(
    parser: Gemma4ToolParser,
):
    text = "<|tool_call>call:get_wea"
    streamed = parser.extract_tool_calls_streaming(
        "", text, text, [], [], [], request=None
    )

    assert streamed is None
    assert parser.finish_streaming() is None
    assert parser._raw_to_public_tool_index == {}
    assert parser.prev_tool_call_arr == []


def test_finish_streaming_distinguishes_open_and_complete_empty_arguments():
    open_parser = Gemma4ToolParser(tokenizer=None)
    open_text = "<|tool_call>call:ping{"
    open_name = open_parser.extract_tool_calls_streaming(
        "", open_text, open_text, [], [], [], request=None
    )
    open_finish = open_parser.finish_streaming()

    assert open_name is None
    assert open_finish is not None
    assert open_finish.tool_calls[0].function.name == "ping"
    assert open_finish.tool_calls[0].function.arguments is None

    complete_parser = Gemma4ToolParser(tokenizer=None)
    complete_text = "<|tool_call>call:ping{}"
    complete_name = complete_parser.extract_tool_calls_streaming(
        "", complete_text, complete_text, [], [], [], request=None
    )
    complete_finish = complete_parser.finish_streaming()

    assert complete_name is None
    assert complete_finish is not None
    assert complete_finish.tool_calls[0].function.name == "ping"
    assert json.loads(complete_finish.tool_calls[0].function.arguments) == {}
