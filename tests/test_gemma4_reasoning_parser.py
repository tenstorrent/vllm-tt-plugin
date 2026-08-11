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
        15: "tho",
        16: "Before",
        17: "call:a{x:1}",
        18: " between ",
        19: "call:b{y:2}",
        20: "after",
        21: "Before<|tool_",
        24: "<|tool_call>call:a{",
        25: "<|tool_call>call:a{x:1",
        26: "Tail<|too",
        27: "<chan",
    }
    special_pieces = {
        1: "<|channel>",
        2: "<channel|>",
        3: "<|turn>",
        4: "<|tool_call>",
        5: "<|tool_response>",
        6: "<tool_call|>",
    }

    def get_vocab(self):
        return {
            "<|channel>": 1,
            "<channel|>": 2,
            "<|turn>": 3,
            "<|tool_call>": 4,
            "<|tool_response>": 5,
            "<tool_call|>": 6,
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(
            (
                ""
                if skip_special_tokens and token_id in self.special_pieces
                else self.special_pieces.get(token_id, self.pieces.get(token_id, ""))
            )
            for token_id in token_ids
        )


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


def test_whole_canvas_preserves_tool_call_framing():
    tokenizer = FakeTokenizer()
    token_ids = [1, 10, 2, 4, 14, 6]
    delta_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    result = _parser().extract_reasoning_streaming(
        "", delta_text, delta_text, [], token_ids, token_ids
    )

    assert result is not None
    assert result.reasoning == "Reason."
    assert result.content == "<|tool_call>call:get_weather{}<tool_call|>"


def test_implicit_tool_call_preserves_framing_and_content_ids():
    tokenizer = FakeTokenizer()
    parser = _parser()
    token_ids = [1, 10, 4, 14, 6]
    delta_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    result = parser.extract_reasoning_streaming(
        "", delta_text, delta_text, [], token_ids, token_ids
    )

    assert result is not None
    assert result.reasoning == "Reason."
    assert result.content == "<|tool_call>call:get_weather{}<tool_call|>"
    assert parser.extract_content_ids(token_ids) == [4, 14, 6]


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


def test_marker_ids_override_stale_text_buffer():
    tokenizer = FakeTokenizer()
    parser = _parser()

    first = parser.extract_reasoning_streaming(
        "", "Preface <", "Preface <", [], [99], [99]
    )
    start_text = tokenizer.decode([1, 12], skip_special_tokens=False)
    second = parser.extract_reasoning_streaming(
        "Preface <",
        "Preface <" + start_text,
        start_text,
        [99],
        [99, 1, 12],
        [1, 12],
    )
    third = parser.extract_reasoning_streaming(
        "Preface <" + start_text,
        "Preface <" + start_text + "Answer.",
        "Answer.",
        [99, 1, 12],
        [99, 1, 12, 2, 11],
        [2, 11],
    )

    assert first is not None and first.content == "Preface "
    assert second is not None
    assert second.content == "<"
    assert second.reasoning == "Part "
    assert third is not None and third.content == "Answer."
    assert parser._marker_buffer == ""
    assert parser._stream_phase == "content"


def test_short_reasoning_prefix_emitted_when_phase_ends():
    tokenizer = FakeTokenizer()
    token_ids = [1, 15, 2]
    delta_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    result = _parser().extract_reasoning_streaming(
        "", delta_text, delta_text, [], token_ids, token_ids
    )

    assert result is not None
    assert result.reasoning == "tho"
    assert result.content is None


def test_short_reasoning_prefix_emitted_when_next_delta_ends_reasoning():
    parser = _parser()
    first_text = "<|channel>tho"
    second_text = "<channel|>Answer."

    first = parser.extract_reasoning_streaming(
        "",
        first_text,
        first_text,
        [],
        [],
        [],
    )
    second = parser.extract_reasoning_streaming(
        first_text,
        first_text + second_text,
        second_text,
        [],
        [],
        [],
    )

    assert first is None
    assert second is not None
    assert second.reasoning == "tho"
    assert second.content == "Answer."


def test_short_reasoning_prefix_emitted_before_next_delta_tool_call():
    parser = _parser()
    first_text = "<|channel>tho"
    second_text = "<|tool_call>call:get_weather{}<tool_call|>"

    first = parser.extract_reasoning_streaming(
        "",
        first_text,
        first_text,
        [],
        [],
        [],
    )
    second = parser.extract_reasoning_streaming(
        first_text,
        first_text + second_text,
        second_text,
        [],
        [],
        [],
    )

    assert first is None
    assert second is not None
    assert second.reasoning == "tho"
    assert second.content == "<|tool_call>call:get_weather{}<tool_call|>"


def test_held_reasoning_marker_prefix_precedes_definitive_end_id():
    tokenizer = FakeTokenizer()
    parser = _parser()
    first_ids = [1, 10]
    held_ids = [27]
    final_ids = [2, 11]

    first = parser.extract_reasoning_streaming(
        "",
        tokenizer.decode(first_ids, skip_special_tokens=False),
        tokenizer.decode(first_ids, skip_special_tokens=False),
        [],
        first_ids,
        first_ids,
    )
    held = parser.extract_reasoning_streaming(
        tokenizer.decode(first_ids, skip_special_tokens=False),
        tokenizer.decode(first_ids + held_ids, skip_special_tokens=False),
        tokenizer.decode(held_ids, skip_special_tokens=False),
        first_ids,
        first_ids + held_ids,
        held_ids,
    )
    final = parser.extract_reasoning_streaming(
        tokenizer.decode(first_ids + held_ids, skip_special_tokens=False),
        tokenizer.decode(first_ids + held_ids + final_ids, skip_special_tokens=False),
        tokenizer.decode(final_ids, skip_special_tokens=False),
        first_ids + held_ids,
        first_ids + held_ids + final_ids,
        final_ids,
    )

    assert first is not None and first.reasoning == "Reason."
    assert held is None
    assert final is not None
    assert final.reasoning == "<chan"
    assert final.content == "Answer."
    assert parser._marker_buffer == ""


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


def test_non_streaming_visible_tool_call_ends_reasoning():
    output = "<|channel>thought\nReason.<|tool_call>call:get_weather{}<tool_call|>"

    reasoning, content = _parser().extract_reasoning(output, request=None)

    assert reasoning == "Reason."
    assert content == "<|tool_call>call:get_weather{}<tool_call|>"


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
    assert content == "<|tool_call>call:get_weather{}"


def test_token_id_extraction_without_channel_keeps_content_intact():
    """No thinking channel: a bare tool call (or plain content) must pass
    through as content, not be reinterpreted as reasoning."""
    reasoning, content = _parser().extract_reasoning_from_token_ids(
        [4, 14, 11], "call:get_weather{}Answer."
    )

    assert reasoning is None
    assert content == "call:get_weather{}Answer."
