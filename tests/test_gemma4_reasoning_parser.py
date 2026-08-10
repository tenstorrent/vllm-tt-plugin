# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from vllm_tt_plugin.gemma4_reasoning_parser import Gemma4ReasoningParser


class FakeTokenizer:
    pieces = {
        10: "thought\nReason.",
        11: "Answer.",
        12: "thought\nPart ",
        13: "two.",
        14: "call:get_weather{}",
    }

    def get_vocab(self):
        return {
            "<|channel>": 1,
            "<channel|>": 2,
            "<|turn>": 3,
            "<|tool_call>": 4,
            "<|tool_response>": 5,
        }

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(self.pieces.get(token_id, "") for token_id in token_ids)


def _parser() -> Gemma4ReasoningParser:
    return Gemma4ReasoningParser(FakeTokenizer())


def test_whole_canvas_markers_and_content_in_one_delta():
    result = _parser().extract_reasoning_streaming(
        "",
        "thought\nReason.Answer.",
        "thought\nReason.Answer.",
        [],
        [1, 10, 2, 11],
        [1, 10, 2, 11],
    )

    assert result is not None
    assert result.reasoning == "Reason."
    assert result.content == "Answer."


def test_stripped_marker_ids_split_reasoning_across_deltas():
    parser = _parser()
    first = parser.extract_reasoning_streaming(
        "", "thought\nPart ", "thought\nPart ", [], [1, 12], [1, 12]
    )
    second = parser.extract_reasoning_streaming(
        "thought\nPart ",
        "thought\nPart two.Answer.",
        "two.Answer.",
        [1, 12],
        [1, 12, 13, 2, 11],
        [13, 2, 11],
    )

    assert first is not None and first.reasoning == "Part "
    assert second is not None
    assert second.reasoning == "two."
    assert second.content == "Answer."


def test_literal_marker_can_split_across_text_deltas():
    parser = _parser()

    assert (
        parser.extract_reasoning_streaming("", "<|chan", "<|chan", [], [], []) is None
    )
    middle = parser.extract_reasoning_streaming(
        "<|chan",
        "<|channel>thought\nReason.<chan",
        "nel>thought\nReason.<chan",
        [],
        [],
        [],
    )
    final = parser.extract_reasoning_streaming(
        "<|channel>thought\nReason.<chan",
        "<|channel>thought\nReason.<channel|>Answer.",
        "nel|>Answer.",
        [],
        [],
        [],
    )

    assert middle is not None and middle.reasoning == "Reason."
    assert final is not None and final.content == "Answer."


def test_literal_marker_split_after_visible_prefix():
    parser = _parser()

    first = parser.extract_reasoning_streaming(
        "", "Preface <|chan", "Preface <|chan", [], [], []
    )
    second = parser.extract_reasoning_streaming(
        "Preface <|chan",
        "Preface <|channel>thought\nReason.<channel|>Answer.",
        "nel>thought\nReason.<channel|>Answer.",
        [],
        [],
        [],
    )

    assert first is not None and first.content == "Preface "
    assert second is not None
    assert second.reasoning == "Reason."
    assert second.content == "Answer."


def test_no_marker_output_remains_content():
    parser = _parser()
    streamed = parser.extract_reasoning_streaming(
        "", "Plain answer.", "Plain answer.", [], [99], [99]
    )
    reasoning, content = _parser().extract_reasoning("Plain answer.", request=None)

    assert streamed is not None and streamed.content == "Plain answer."
    assert reasoning is None
    assert content == "Plain answer."


def test_non_streaming_markers_split_reasoning_and_content():
    reasoning, content = _parser().extract_reasoning(
        "<|channel>thought\nReason.<channel|>Answer.", request=None
    )

    assert reasoning == "Reason."
    assert content == "Answer."


def test_non_streaming_stripped_markers_split_from_token_ids():
    reasoning, content = _parser().extract_reasoning_from_token_ids(
        [1, 10, 2, 11],
        "thought\nReason.Answer.",
    )

    assert reasoning == "Reason."
    assert content == "Answer."


def test_token_id_extraction_ends_reasoning_at_tool_call():
    """A tool call terminates an open thinking channel; its payload must not
    be swallowed into the reasoning field."""
    reasoning, content = _parser().extract_reasoning_from_token_ids(
        [1, 10, 4, 14], "thought\nReason.call:get_weather{}"
    )

    assert reasoning == "Reason."
    assert content == "call:get_weather{}"


def test_token_id_extraction_without_channel_keeps_content_intact():
    """No thinking channel: a bare tool call (or plain content) must pass
    through as content, not be reinterpreted as reasoning."""
    reasoning, content = _parser().extract_reasoning_from_token_ids(
        [4, 14, 11], "call:get_weather{}Answer."
    )

    assert reasoning is None
    assert content == "call:get_weather{}Answer."
