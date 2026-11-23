# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Checks Kolibri template switches and request-local reasoning extraction.

A deterministic tokenizer supplies only the special-token vocabulary and decode
operations used by the parser; no model files or device initialization are needed.
"""

from collections.abc import Sequence
from typing import cast

import pytest
import vllm.config  # noqa: F401  # finish plugin discovery before parser imports
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import TokenizerLike

from vllm_tt_plugin.parsers.kolibri1 import (
    Kolibri1Parser,
    Kolibri1ReasoningParser,
    thinking_enabled,
)

# Stable IDs for the single-token reasoning and tool delimiters.
_SPECIAL = {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4}


class _StubTokenizer:
    """Supplies the vocabulary and decoding operations needed by these parsers."""

    def get_vocab(self) -> dict[str, int]:
        """Returns a fresh vocabulary so parser setup cannot mutate the token table."""
        return dict(_SPECIAL)

    def decode(self, token_ids: Sequence[int], **kwargs: object) -> str:
        """Decodes the known delimiter tokens without loading a model tokenizer."""
        by_id = {v: k for k, v in _SPECIAL.items()}
        return "".join(by_id.get(t, "") for t in token_ids)


@pytest.fixture(scope="module")
def tokenizer() -> TokenizerLike:
    """Supplies the tokenizer subset used by Kolibri's Qwen3-based engine."""
    return cast(TokenizerLike, _StubTokenizer())


def _request(**chat_kwargs: object) -> ChatCompletionRequest:
    """Builds a real request carrying only the template inputs under test."""
    return ChatCompletionRequest(
        model="kolibri-test", messages=[], chat_template_kwargs=chat_kwargs or None
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({}, True, id="default-enabled"),
        pytest.param({"enable_thinking": True}, True, id="enabled"),
        pytest.param({"enable_thinking": False}, False, id="disabled"),
        pytest.param({"reasoning_effort": "none"}, False, id="effort-none"),
        pytest.param({"reasoning_effort": "high"}, True, id="effort-high"),
        pytest.param(
            {"reasoning_effort": "high", "enable_thinking": False},
            True,
            id="effort-enables-over-false",
        ),
        pytest.param(
            {"reasoning_effort": "none", "enable_thinking": True},
            False,
            id="effort-disables-over-true",
        ),
    ],
)
def test_thinking_enabled_mirrors_template(
    kwargs: dict[str, object], expected: bool
) -> None:
    """The template's effort setting takes precedence over enable_thinking."""
    assert thinking_enabled(kwargs) is expected


def _parser(tokenizer: TokenizerLike, **chat_kwargs: object) -> Kolibri1ReasoningParser:
    """Builds a fresh reasoning adapter with the template inputs under test."""
    return Kolibri1ReasoningParser(tokenizer, chat_template_kwargs=chat_kwargs)


def test_thinking_on_splits_reasoning_and_answer(tokenizer: TokenizerLike) -> None:
    """Separates a complete reasoning block from the following answer."""
    parser = _parser(tokenizer)
    reasoning, content = parser.extract_reasoning(
        "<think>\nPlan the answer.\n</think>\n\nParis.", _request()
    )
    assert reasoning.strip() == "Plan the answer."
    assert content.strip() == "Paris."


def test_thinking_off_by_effort_none_returns_plain_content(
    tokenizer: TokenizerLike,
) -> None:
    """Returns generated text as content when the template disabled thinking."""
    parser = _parser(tokenizer, reasoning_effort="none")
    reasoning, content = parser.extract_reasoning(
        "Paris.", _request(reasoning_effort="none")
    )
    assert reasoning is None
    assert content == "Paris."


def test_effort_overrides_enable_thinking_false(tokenizer: TokenizerLike) -> None:
    """Uses the effort switch when the lower-priority flag contradicts it."""
    kwargs = {"reasoning_effort": "high", "enable_thinking": False}
    parser = _parser(tokenizer, **kwargs)
    reasoning, content = parser.extract_reasoning(
        "<think>\nStill thinking.\n</think>\n\nParis.", _request(**kwargs)
    )
    assert reasoning.strip() == "Still thinking."
    assert content.strip() == "Paris."


def test_unfinished_reasoning_is_reasoning_only(tokenizer: TokenizerLike) -> None:
    """Retains an incomplete thinking block as reasoning without an answer."""
    parser = _parser(tokenizer)
    reasoning, content = parser.extract_reasoning("<think>\nI am not done", _request())
    assert reasoning.strip() == "I am not done"
    assert not content


def test_continued_final_message_is_answer_content(tokenizer: TokenizerLike) -> None:
    """Starts an assistant continuation after the thinking block in its prompt."""
    kwargs = {"continue_final_message": True, "add_generation_prompt": False}
    parser = _parser(tokenizer, **kwargs)
    reasoning, content = parser.extract_reasoning(" Paris.", _request(**kwargs))
    assert reasoning is None
    assert content == " Paris."


def test_continued_final_message_streams_as_content(tokenizer: TokenizerLike) -> None:
    """Keeps every streamed continuation chunk in content state."""
    kwargs = {"continue_final_message": True, "add_generation_prompt": False}
    parser = _parser(tokenizer, **kwargs)
    deltas = []
    text = ""
    for piece in (" Par", "is."):
        previous, text = text, text + piece
        deltas.append(
            parser.extract_reasoning_streaming(previous, text, piece, [], [], [])
        )
    content = "".join(d.content or "" for d in deltas if d is not None)
    assert content == " Paris."
    assert not any(getattr(d, "reasoning", None) for d in deltas if d is not None)


def test_constructor_accepts_positional_tools_without_mutating_options(
    tokenizer: TokenizerLike,
) -> None:
    """Accepts positional tool definitions while preserving caller-owned options."""
    options = {"reasoning_effort": "high", "enable_thinking": False}
    parser = Kolibri1Parser(tokenizer, [], chat_template_kwargs=options)

    assert options == {"reasoning_effort": "high", "enable_thinking": False}
    reasoning, content = parser.extract_reasoning(
        "<think>Plan.</think>Paris.", _request(**options)
    )
    assert reasoning == "Plan."
    assert content == "Paris."
