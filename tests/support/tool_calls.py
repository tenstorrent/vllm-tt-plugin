# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Runs vLLM's ``qwen3_coder`` tool parser on model text without model files.

``CharTokenizer`` gives the four Qwen3 delimiter tokens their own IDs and every
other character one ID, so streamed deltas carry token IDs as in serving:
``<tool_call>`` and ``</tool_call>`` reach the parser as token-ID terminals.
``extract`` parses a complete output; ``stream`` feeds it token by token.
"""

import json
from collections.abc import Sequence
from typing import Any, NamedTuple

import vllm.config  # noqa: F401  # finish plugin discovery before parser imports
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser

# IDs of the Qwen3 delimiters that are single tokens; characters start above them.
_SPECIAL = {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4}
_CHAR_BASE = 256


class Call(NamedTuple):
    """One parsed tool call: its function name and decoded JSON arguments."""

    name: str
    arguments: Any


class CharTokenizer:
    """Tokenizes Qwen3 delimiters as single tokens and other text per character."""

    def get_vocab(self) -> dict[str, int]:
        """Returns the delimiter vocabulary that the parser engine looks up."""
        return dict(_SPECIAL)

    def encode(self, text: str) -> list[int]:
        """Returns one ID per delimiter and per other character of ``text``."""
        ids: list[int] = []
        pos = 0
        while pos < len(text):
            special = next((s for s in _SPECIAL if text.startswith(s, pos)), None)
            if special is not None:
                ids.append(_SPECIAL[special])
                pos += len(special)
            else:
                ids.append(_CHAR_BASE + ord(text[pos]))
                pos += 1
        return ids

    def decode(self, token_ids: Sequence[int], **kwargs: object) -> str:
        """Returns the text of ``token_ids``."""
        by_id = {v: k for k, v in _SPECIAL.items()}
        return "".join(by_id.get(t) or chr(t - _CHAR_BASE) for t in token_ids)


def request(tools: dict[str, dict]) -> ChatCompletionRequest:
    """Builds a chat request that offers one function tool per ``tools`` entry.

    ``tools`` maps each function name to its parameters schema.
    """
    return ChatCompletionRequest(
        model="qwen3-test",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {"type": "function", "function": {"name": name, "parameters": schema}}
            for name, schema in tools.items()
        ],
    )


def extract(tools: dict[str, dict], text: str) -> list[Call]:
    """Parses a complete model output with the non-streaming parser path."""
    req = request(tools)
    parser = Qwen3EngineToolParser(CharTokenizer(), req.tools)
    result = parser.extract_tool_calls(text, req)
    return [
        Call(call.function.name, json.loads(call.function.arguments))
        for call in result.tool_calls
    ]


def stream(tools: dict[str, dict], text: str, step: int = 1) -> list[Call]:
    """Parses a model output fed ``step`` tokens per delta, as the server streams it.

    The argument deltas of each call are joined and decoded as JSON, so an
    unterminated stream fails with ``json.JSONDecodeError``.
    """
    req = request(tools)
    tokenizer = CharTokenizer()
    parser = Qwen3EngineToolParser(tokenizer, req.tools)
    ids = tokenizer.encode(text)
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}

    def collect(delta: Any) -> None:
        for call in (delta.tool_calls if delta else None) or ():
            if call.function and call.function.name:
                names[call.index] = call.function.name
            if call.function and call.function.arguments:
                arguments[call.index] = (
                    arguments.get(call.index, "") + call.function.arguments
                )

    previous_text, previous_ids = "", []
    for pos in range(0, len(ids), step):
        current_ids = ids[: pos + step]
        current_text = tokenizer.decode(current_ids)
        collect(
            parser.extract_tool_calls_streaming(
                previous_text,
                current_text,
                current_text[len(previous_text) :],
                previous_ids,
                current_ids,
                ids[pos : pos + step],
                req,
            )
        )
        previous_text, previous_ids = current_text, current_ids
    collect(parser.finish_streaming())
    return [Call(names[i], json.loads(arguments.get(i, ""))) for i in sorted(names)]


def call_text(name: str, params: dict[str, str]) -> str:
    """Writes one tool call the way the Qwen3 chat template does."""
    body = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in params.items()
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"
