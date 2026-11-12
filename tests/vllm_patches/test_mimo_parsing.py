# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""MiMo tool-call parsing in the patched vLLM: values verbatim (until-0.32,
upstream #58019), and a call cut off by max_tokens (common, truncated tool
arguments): streamed arguments are valid JSON equal to the non-streamed ones,
and a repeated parameter keeps its first value."""

import json
from unittest.mock import MagicMock

import pytest

from tests.vllm_patches import requires_patched_vllm

pytestmark = requires_patched_vllm

TOOL_START, TOOL_END = "<tool_call>", "</tool_call>"
SPECIAL = {TOOL_START: 100, TOOL_END: 101, "<think>": 102, "</think>": 103}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "integer"},
                },
                "required": ["path", "content"],
            },
        },
    }
]

# Wrapping newlines and characters JSON must escape, as MiMo writes them.
CONTENT = '\nline 1 "q" \\ tab\there\n\nünïcødé 🌊\n'


def mimo_call(*params: tuple[str, str], close: bool = True) -> str:
    body = "".join(f"<parameter={k}>{v}</parameter>" for k, v in params)
    end = f"</function>{TOOL_END}" if close else ""
    return f"{TOOL_START}<function=write_file>{body}{end}"


CALL = mimo_call(("path", "notes.txt"), ("content", CONTENT), ("mode", "644"))
CALL_REPEATED = mimo_call(
    ("path", "a.txt"),
    ("content", "\nfirst\n"),
    ("content", "second, longer"),
    ("mode", "600"),
)


def _tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.get_vocab.return_value = dict(SPECIAL)
    tok.encode.return_value = [1, 2, 3]
    tok.all_special_tokens = list(SPECIAL)
    tok.all_special_ids = list(SPECIAL.values())
    inverse = {v: k for k, v in SPECIAL.items()}
    tok.decode.side_effect = lambda ids: "".join(inverse.get(i, "?") for i in ids)
    return tok


def _request(tools=TOOLS):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    return ChatCompletionRequest(
        model="mimo",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice="auto",
    )


def _parser(name: str = "mimo", tools=TOOLS):
    from vllm.parser import ParserManager

    reasoning = "qwen3" if name == "qwen3_coder" else name
    cls = ParserManager.get_parser(name, reasoning, enable_auto_tools=True)
    return cls(_tokenizer(), tools, chat_template_kwargs={"enable_thinking": False})


def _tokens(text: str) -> list[tuple[str, list[int]]]:
    out, i = [], 0
    while i < len(text):
        for special, token_id in SPECIAL.items():
            if text.startswith(special, i):
                out.append((special, [token_id]))
                i += len(special)
                break
        else:
            out.append((text[i], [200 + ord(text[i]) % 50]))
            i += 1
    return out


def stream_args(text: str, chunk: int, name: str = "mimo") -> str | None:
    parser, request = _parser(name), _request()
    tokens = _tokens(text)
    groups = [tokens[i : i + chunk] for i in range(0, len(tokens), chunk)] or [[]]
    args = None
    for index, group in enumerate(groups):
        delta = parser.parse_delta(
            "".join(t for t, _ in group),
            [i for _, ids in group for i in ids],
            request,
            finished=index == len(groups) - 1,
        )
        for call in (delta.tool_calls if delta else None) or []:
            if call.function and call.function.arguments:
                args = (args or "") + call.function.arguments
    return args


def full_args(text: str, name: str = "mimo") -> str | None:
    _, _, calls = _parser(name).parse(text, _request(), enable_auto_tools=True)
    return calls[0].arguments if calls else None


@pytest.fixture
def no_safety_net(monkeypatch):
    """Fail if the flush has to close a stream the final result diverged from."""
    from vllm.parser.engine.parser_engine import ParserEngine

    def diverged(self, slot, prev):
        raise AssertionError(f"flush diverged from the streamed prefix: {prev!r}")

    monkeypatch.setattr(ParserEngine, "_close_streamed_json", diverged)


def test_mimo_maps_to_the_mimo_parser_engine():
    from vllm.parser.mimo import MiMoParser
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    assert ToolParserManager.get_tool_parser("mimo")._parser_engine_cls is MiMoParser
    reasoning = ReasoningParserManager.get_reasoning_parser("mimo")
    assert reasoning._parser_engine_cls is MiMoParser


@pytest.mark.parametrize("chunk", [1, 3, 8, 100_000])
def test_mimo_values_are_verbatim_and_typed(no_safety_net, chunk):
    expected = {"path": "notes.txt", "content": CONTENT, "mode": 644}
    full = full_args(CALL)
    assert json.loads(full) == expected
    assert stream_args(CALL, chunk) == full


def test_qwen3_coder_still_trims_its_template_newlines():
    call = CALL.replace("<function=write_file>", "\n<function=write_file>\n").replace(
        f"<parameter=content>{CONTENT}", "<parameter=content>\nbody\n"
    )
    assert json.loads(full_args(call, "qwen3_coder"))["content"] == "body"


@pytest.mark.parametrize("name", ["mimo", "qwen3_coder"])
@pytest.mark.parametrize("chunk", [1, 3, 100_000])
def test_a_repeated_parameter_keeps_its_first_value(no_safety_net, name, chunk):
    full = full_args(CALL_REPEATED, name)
    content = "\nfirst\n" if name == "mimo" else "first"
    assert json.loads(full) == {"path": "a.txt", "content": content, "mode": 600}
    assert stream_args(CALL_REPEATED, chunk, name) == full


@pytest.mark.parametrize("chunk", [1, 5, 100_000])
def test_every_cut_point_streams_valid_json_equal_to_the_full_result(
    no_safety_net, chunk
):
    start, end = CALL.index("<parameter="), CALL.index("</function>")
    for cut in range(start, end + 1):
        text = CALL[:cut]
        full, streamed = full_args(text), stream_args(text, chunk)
        if streamed is None:
            assert full in (None, "{}"), (cut, full)
            continue
        json.loads(streamed)
        assert streamed == full, (cut, streamed, full)


def test_a_cut_string_value_keeps_its_partial_text():
    text = CALL[: CALL.index("ünïcødé")]
    args = json.loads(full_args(text))
    assert args["path"] == "notes.txt"
    assert args["content"] == CONTENT[: CONTENT.index("ünïcødé")]
