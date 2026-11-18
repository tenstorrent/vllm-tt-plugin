# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
import vllm.config  # noqa: F401  # finish vLLM init before the plugin package

from vllm_tt_plugin import entrypoints
from vllm_tt_plugin.kolibri1_reasoning_parser import thinking_enabled

# The Kolibri wrapper tokens are single special tokens; the parser engine looks
# them up in the vocabulary. A deterministic stand-in keeps these tests running
# on every host instead of skipping when the real tokenizer is not cached.
_SPECIAL = {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4}


class _StubTokenizer:
    def get_vocab(self):
        return dict(_SPECIAL)

    def decode(self, token_ids, **kwargs):
        by_id = {v: k for k, v in _SPECIAL.items()}
        return "".join(by_id.get(t, "") for t in token_ids)


@pytest.fixture(scope="module")
def tokenizer():
    return _StubTokenizer()


def _request(**chat_template_kwargs):
    return SimpleNamespace(chat_template_kwargs=chat_template_kwargs or None)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, True),
        ({"enable_thinking": True}, True),
        ({"enable_thinking": False}, False),
        ({"reasoning_effort": "none"}, False),
        ({"reasoning_effort": "high"}, True),
        # The template lets reasoning_effort win over enable_thinking.
        ({"reasoning_effort": "high", "enable_thinking": False}, True),
        ({"reasoning_effort": "none", "enable_thinking": True}, False),
    ],
)
def test_thinking_enabled_mirrors_template(kwargs, expected):
    assert thinking_enabled(kwargs) is expected


def test_register_adds_kolibri1_parsers_once():
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    from vllm_tt_plugin.kolibri1_reasoning_parser import Kolibri1ReasoningParser

    entrypoints._register_tt_reasoning_parsers()
    entrypoints._register_tt_tool_parsers()
    # Re-registering must be a no-op (vLLM calls register() in every process).
    entrypoints._register_tt_reasoning_parsers()
    entrypoints._register_tt_tool_parsers()

    assert "kolibri1" in ReasoningParserManager.list_registered()
    assert "kolibri1" in ToolParserManager.list_registered()
    assert (
        ReasoningParserManager.get_reasoning_parser("kolibri1")
        is Kolibri1ReasoningParser
    )
    assert ToolParserManager.get_tool_parser("kolibri1") is Hermes2ProToolParser


def _parser(tokenizer, **chat_template_kwargs):
    from vllm_tt_plugin.kolibri1_reasoning_parser import Kolibri1ReasoningParser

    return Kolibri1ReasoningParser(tokenizer, chat_template_kwargs=chat_template_kwargs)


def test_thinking_on_splits_reasoning_and_answer(tokenizer):
    parser = _parser(tokenizer)
    reasoning, content = parser.extract_reasoning(
        "<think>\nPlan the answer.\n</think>\n\nParis.", _request()
    )
    assert reasoning.strip() == "Plan the answer."
    assert content.strip() == "Paris."


def test_thinking_off_by_effort_none_returns_plain_content(tokenizer):
    # The template renders the closed empty think block, so the model output is
    # the answer only and must not be filed as reasoning.
    parser = _parser(tokenizer, reasoning_effort="none")
    reasoning, content = parser.extract_reasoning(
        "Paris.", _request(reasoning_effort="none")
    )
    assert reasoning is None
    assert content == "Paris."


def test_effort_overrides_enable_thinking_false(tokenizer):
    # Stock qwen3 would read enable_thinking=False and treat everything as the
    # answer; the Kolibri template keeps thinking on here, and so must we.
    kwargs = {"reasoning_effort": "high", "enable_thinking": False}
    parser = _parser(tokenizer, **kwargs)
    reasoning, content = parser.extract_reasoning(
        "<think>\nStill thinking.\n</think>\n\nParis.", _request(**kwargs)
    )
    assert reasoning.strip() == "Still thinking."
    assert content.strip() == "Paris."


def test_unfinished_reasoning_is_reasoning_only(tokenizer):
    parser = _parser(tokenizer)
    reasoning, content = parser.extract_reasoning("<think>\nI am not done", _request())
    assert reasoning.strip() == "I am not done"
    assert not content


def test_continued_final_message_is_answer_content(tokenizer):
    # continue_final_message extends the final assistant message, whose closed
    # think block is already in the prompt: the continuation is the answer.
    kwargs = {"continue_final_message": True, "add_generation_prompt": False}
    parser = _parser(tokenizer, **kwargs)
    reasoning, content = parser.extract_reasoning(" Paris.", _request(**kwargs))
    assert reasoning is None
    assert content == " Paris."


def test_continued_final_message_streams_as_content(tokenizer):
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
