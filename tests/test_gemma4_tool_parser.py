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

from vllm_tt_plugin.gemma4_tool_parser import (
    QUOTE,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Gemma4ToolParser,
)

_TOP_LEVEL_MARKER_PAYLOADS = [
    pytest.param(
        f"{TOOL_CALL_START}call:danger{{}}{TOOL_CALL_END}",
        id="start-before-end",
    ),
    pytest.param(
        f"{TOOL_CALL_END}{TOOL_CALL_START}call:danger{{}}{TOOL_CALL_END}",
        id="end-before-start",
    ),
]


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


def test_lone_unmatched_end_uses_cleaned_content_branch(parser: Gemma4ToolParser):
    result = parser.extract_tool_calls(
        f"Before {TOOL_CALL_END} after",
        request=None,
    )

    assert result.tools_called is True
    assert result.tool_calls == []
    assert result.content == "Before  after"


@pytest.mark.parametrize("payload", _TOP_LEVEL_MARKER_PAYLOADS)
def test_nonstream_top_level_paired_quote_preserves_literal_markers(
    parser: Gemma4ToolParser,
    payload: str,
):
    text = f"Before {QUOTE}{payload}{QUOTE} after"

    result = parser.extract_tool_calls(text, request=None)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == text


@pytest.mark.parametrize("payload", _TOP_LEVEL_MARKER_PAYLOADS)
def test_nonstream_top_level_unclosed_quote_preserves_literal_markers(
    parser: Gemma4ToolParser,
    payload: str,
):
    text = f"Before {QUOTE}{payload} after"

    result = parser.extract_tool_calls(text, request=None)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == text


def test_nonstream_real_call_after_closed_top_level_quote():
    payload = f"{TOOL_CALL_START}call:danger{{}}{TOOL_CALL_END}"
    quoted = f"{QUOTE}{payload}{QUOTE}"
    text = (
        f"Before {quoted} middle {TOOL_CALL_START}call:good{{x:1}}{TOOL_CALL_END} after"
    )

    result = Gemma4ToolParser(tokenizer=None).extract_tool_calls(text, request=None)

    assert result.tools_called is True
    assert result.content == f"Before {quoted} middle  after"
    assert [call.function.name for call in result.tool_calls] == ["good"]


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


def test_string_value_with_paired_tool_markers(parser: Gemma4ToolParser):
    marker_text = (
        f"before{TOOL_CALL_END}{TOOL_CALL_START}call:b{{x:2}}{TOOL_CALL_END}after"
    )
    out = f'<|tool_call>call:h{{q:<|"|>{marker_text}<|"|>}}<tool_call|>'
    result = parser.extract_tool_calls(out, request=None)

    assert [call.function.name for call in result.tool_calls] == ["h"]
    assert json.loads(result.tool_calls[0].function.arguments) == {"q": marker_text}


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


@pytest.mark.parametrize(
    "malformed_body",
    _STRICTLY_MALFORMED_COMPLETED_BODIES,
)
def test_non_streaming_all_malformed_frames_use_cleaned_content_branch(
    parser: Gemma4ToolParser,
    malformed_body: str,
):
    out = f"Before <|tool_call>{malformed_body}<tool_call|> after"

    result = parser.extract_tool_calls(out, request=None)

    assert result.tools_called is True
    assert result.tool_calls == []
    assert result.content == "Before  after"


def test_incomplete_block_uses_cleaned_content_branch(parser: Gemma4ToolParser):
    # No closing <tool_call|>: not emitted as a completed call.
    out = "<|tool_call>call:a{x:1"
    result = parser.extract_tool_calls(out, request=None)
    assert result.tools_called is True
    assert result.tool_calls == []
    assert result.content is None


def test_non_streaming_truncated_frame_does_not_leak_special_tokens(
    parser: Gemma4ToolParser,
):
    out = "Visible <|tool_call>call:a{x:1"

    result = parser.extract_tool_calls(out, request=None)

    assert result.tools_called is True
    assert result.tool_calls == []
    assert result.content == "Visible "
    assert "<|tool_call>" not in result.content


def test_non_streaming_truncated_frame_resyncs_to_valid_sibling(
    parser: Gemma4ToolParser,
):
    out = "Before <|tool_call>call:bad{x:1<|tool_call>call:good{x:2}<tool_call|> after"

    result = parser.extract_tool_calls(out, request=None)

    assert result.tools_called is True
    assert result.content == "Before  after"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "good"
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 2}


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


def _direct_stream(chunks: list[str]):
    parser = Gemma4ToolParser(tokenizer=None)
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

    assert all(call.id is not None for call in tool_calls)
    summary = [
        (
            call.index,
            call.function.name,
            json.loads(call.function.arguments),
        )
        for call in tool_calls
    ]
    return parser, content, summary


@pytest.mark.parametrize("payload", _TOP_LEVEL_MARKER_PAYLOADS)
def test_streaming_top_level_paired_quote_never_executes_literal_markers(
    payload: str,
):
    parser = Gemma4ToolParser(tokenizer=None)
    chunks = [f"Before {QUOTE}", payload, f"{QUOTE} after"]
    previous_text = ""
    content = ""

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
            assert result.tool_calls == []
        previous_text = current_text

    assert content == "".join(chunks)
    assert parser.finish_streaming() is None
    assert parser._raw_to_public_tool_index == {}


@pytest.mark.parametrize("payload", _TOP_LEVEL_MARKER_PAYLOADS)
def test_streaming_top_level_unclosed_quote_stays_literal_at_finish(
    payload: str,
):
    parser = Gemma4ToolParser(tokenizer=None)
    chunks = [f"Before {QUOTE}", payload, " after"]
    previous_text = ""
    content = ""

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
            assert result.tool_calls == []
        previous_text = current_text

    assert content == "".join(chunks)
    assert parser.finish_streaming() is None
    assert parser._raw_to_public_tool_index == {}


def test_streaming_real_call_after_closed_top_level_quote_emits():
    payload = f"{TOOL_CALL_START}call:danger{{}}{TOOL_CALL_END}"
    parser = Gemma4ToolParser(tokenizer=None)
    chunks = [
        f"Before {QUOTE}",
        payload,
        (f"{QUOTE} middle {TOOL_CALL_START}call:good{{x:1}}{TOOL_CALL_END} after"),
    ]
    previous_text = ""
    content = ""
    tool_calls = []

    for index, chunk in enumerate(chunks):
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
            if index < 2:
                assert result.tool_calls == []
        previous_text = current_text

    quoted = f"{QUOTE}{payload}{QUOTE}"
    assert content == f"Before {quoted} middle  after"
    assert [call.index for call in tool_calls] == [0]
    assert [call.function.name for call in tool_calls] == ["good"]


def test_streaming_stray_end_is_chunk_invariant_and_keeps_call_indices():
    chunks = [
        "<|tool_call>call:a{x:1}<tool_call|>",
        " visible x ",
        TOOL_CALL_END,
        " visible y ",
        "<|tool_call>call:b{x:2}<tool_call|>",
        "<|tool_call>call:c{x:3}<tool_call|>",
    ]

    chunked_parser, chunked_content, chunked_calls = _direct_stream(chunks)
    _, one_delta_content, one_delta_calls = _direct_stream(["".join(chunks)])

    expected_calls = [
        (0, "a", {"x": 1}),
        (1, "b", {"x": 2}),
        (2, "c", {"x": 3}),
    ]
    assert chunked_content == one_delta_content == " visible x  visible y "
    assert TOOL_CALL_END not in chunked_content
    assert chunked_calls == one_delta_calls == expected_calls
    assert chunked_parser._raw_to_public_tool_index == {0: 0, 1: 1, 2: 2}


def test_streaming_quoted_end_is_chunk_invariant_and_keeps_call_indices():
    chunks = [
        '<|tool_call>call:a{q:<|"|>before',
        TOOL_CALL_END,
        'after<|"|>}<tool_call|>',
        " visible ",
        "<|tool_call>call:b{x:2}<tool_call|>",
        "<|tool_call>call:c{x:3}<tool_call|>",
    ]

    chunked_parser, chunked_content, chunked_calls = _direct_stream(chunks)
    _, one_delta_content, one_delta_calls = _direct_stream(["".join(chunks)])

    expected_calls = [
        (0, "a", {"q": "before<tool_call|>after"}),
        (1, "b", {"x": 2}),
        (2, "c", {"x": 3}),
    ]
    assert chunked_content == one_delta_content == " visible "
    assert chunked_calls == one_delta_calls == expected_calls
    assert chunked_parser._raw_to_public_tool_index == {0: 0, 1: 1, 2: 2}


def test_streaming_paired_tool_markers_are_literal_and_chunk_invariant():
    marker_text = (
        f"before{TOOL_CALL_END}{TOOL_CALL_START}call:b{{x:2}}{TOOL_CALL_END}after"
    )
    chunks = [
        f'<|tool_call>call:a{{q:<|"|>before{TOOL_CALL_END}',
        f"{TOOL_CALL_START}call:b{{x:2}}{TOOL_CALL_END}after",
        '<|"|>}<tool_call|>',
    ]

    parser = Gemma4ToolParser(tokenizer=None)
    previous_text = ""
    content = ""
    chunked_calls = []
    interim_tool_deltas = []
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
        interim_tool_deltas.append(
            list(result.tool_calls or []) if result is not None else []
        )
        if result is not None:
            content += result.content or ""
            chunked_calls.extend(result.tool_calls or [])
        previous_text = current_text

    _, one_delta_content, one_delta_calls = _direct_stream(["".join(chunks)])

    expected_calls = [(0, "a", {"q": marker_text})]
    chunked_summary = [
        (
            call.index,
            call.function.name,
            json.loads(call.function.arguments),
        )
        for call in chunked_calls
    ]
    assert interim_tool_deltas[:2] == [[], []]
    assert content == one_delta_content == ""
    assert chunked_summary == one_delta_calls == expected_calls
    assert parser._raw_to_public_tool_index == {0: 0}


def test_streaming_truncated_frame_resyncs_to_valid_sibling():
    chunks = [
        "Before <|tool_call>call:bad{x:1",
        "<|tool_call>call:good{x:2}<tool_call|> after",
    ]

    parser, content, calls = _direct_stream(chunks)

    assert content == "Before  after"
    assert calls == [(0, "good", {"x": 2})]
    assert parser._raw_to_public_tool_index == {1: 0}
    assert parser.prev_tool_call_arr == [{"name": "good", "arguments": {"x": 2}}]


def test_streaming_unterminated_quote_defers_sibling_until_finish():
    chunks = [
        'Before <|tool_call>call:bad{x:<|"|>unterminated',
        "<|tool_call>call:good{x:2}<tool_call|> after",
    ]
    parser = Gemma4ToolParser(tokenizer=None)
    previous_text = ""
    content = ""
    streamed_calls = []

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
            streamed_calls.extend(result.tool_calls or [])
        previous_text = current_text

    assert streamed_calls == []
    finished = parser.finish_streaming()

    assert finished is not None
    content += finished.content or ""
    finished_calls = finished.tool_calls or []
    assert content == "Before  after"
    assert len(finished_calls) == 1
    assert finished_calls[0].index == 0
    assert finished_calls[0].function.name == "good"
    assert json.loads(finished_calls[0].function.arguments) == {"x": 2}
    assert parser._raw_to_public_tool_index == {1: 0}


def test_finish_streaming_emits_all_new_siblings_without_duplicates():
    text = (
        "<|tool_call>call:a{x:1}<tool_call|>"
        'Before <|tool_call>call:bad{x:<|"|>unterminated'
        "<|tool_call>call:b{x:2}<tool_call|>"
        "<|tool_call>call:c{x:3}<tool_call|> after"
    )
    parser = Gemma4ToolParser(tokenizer=None)
    streamed = parser.extract_tool_calls_streaming(
        "", text, text, [], [], [], request=None
    )

    assert streamed is not None
    assert [call.function.name for call in streamed.tool_calls or []] == ["a"]
    finished = parser.finish_streaming()

    assert finished is not None
    assert finished.content == " after"
    assert [call.index for call in finished.tool_calls or []] == [1, 2]
    assert [call.function.name for call in finished.tool_calls or []] == ["b", "c"]
    assert parser._raw_to_public_tool_index == {0: 0, 2: 1, 3: 2}


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

    if malformed_body == 'call:bad{x:<|"|>unterminated}':
        assert result is None
        result = parser.finish_streaming()

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

    assert result.tools_called is True
    assert result.tool_calls == []
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


def test_malformed_completed_quote_is_not_emitted_at_stream_finish(
    parser: Gemma4ToolParser,
):
    text = '<|tool_call>call:f{x:<|"|>unterminated}<tool_call|>'
    streamed = parser.extract_tool_calls_streaming(
        "", text, text, [], [], [], request=None
    )

    assert streamed is None
    assert parser.finish_streaming() is None
    assert parser._raw_to_public_tool_index == {}
    assert parser.prev_tool_call_arr == []


def test_stream_finish_recovers_content_after_final_malformed_quote(
    parser: Gemma4ToolParser,
):
    text = 'Before <|tool_call>call:f{x:<|"|>unterminated}<tool_call|> after'
    streamed = parser.extract_tool_calls_streaming(
        "", text, text, [], [], [], request=None
    )
    finished = parser.finish_streaming()

    assert streamed is not None
    assert streamed.content == "Before "
    assert streamed.tool_calls == []
    assert finished is not None
    assert finished.content == " after"
    assert finished.tool_calls == []


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


def test_finish_streaming_does_not_leak_buffered_marker_inside_open_call(
    parser: Gemma4ToolParser,
):
    text = "<|tool_call>call:ping{x:1}<|tool_"
    streamed = parser.extract_tool_calls_streaming(
        "", text, text, [], [], [], request=None
    )
    finished = parser.finish_streaming()

    assert streamed is None
    assert finished is not None
    assert finished.content is None
    assert len(finished.tool_calls) == 1
    assert finished.tool_calls[0].function.name == "ping"
    assert json.loads(finished.tool_calls[0].function.arguments) == {"x": 1}


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
