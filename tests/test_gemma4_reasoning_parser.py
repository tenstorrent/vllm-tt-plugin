# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from vllm_tt_plugin.gemma4_reasoning_parser import Gemma4ReasoningParser


class FakeTokenizer:
    pieces = {
        10: "thought\nReason.",
        11: "Answer.",
        14: "call:get_weather{}",
        29: "call:danger{}",
        31: "call:good{}",
        35: "one",
        36: "mid",
        37: "two",
        38: "end",
    }
    special_pieces = {
        1: "<|channel>",
        2: "<channel|>",
        3: "<|turn>",
        4: "<|tool_call>",
        5: "<|tool_response>",
        6: "<tool_call|>",
        7: '<|"|>',
    }

    def get_vocab(self):
        return {
            "<|channel>": 1,
            "<channel|>": 2,
            "<|turn>": 3,
            "<|tool_call>": 4,
            "<|tool_response>": 5,
            "<tool_call|>": 6,
            '<|"|>': 7,
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


def test_no_channel_content_immediately_ends_reasoning_phase():
    parser = _parser()

    result = parser.extract_reasoning_streaming(
        "", 'Before <|"|>', 'Before <|"|>', [], [], []
    )

    assert result is not None
    assert result.reasoning is None
    assert result.content == 'Before <|"|>'
    assert parser.is_reasoning_end_streaming([], [])


def test_quoted_text_markers_stay_in_open_reasoning():
    payloads = (
        "<|tool_call>call:danger{}<tool_call|>",
        "<tool_call|><|tool_call>call:danger{}<tool_call|>",
    )

    for payload in payloads:
        quoted = f'<|"|>{payload}<|"|>'
        reasoning, content = _parser().extract_reasoning(
            f"<|channel>thought\nReason.{quoted}",
            request=None,
        )

        assert reasoning == f"Reason.{quoted}"
        assert content is None


def test_real_tool_token_id_after_closed_quote_ends_reasoning():
    tokenizer = FakeTokenizer()
    quoted_ids = [7, 6, 4, 29, 6, 7]
    real_tool_ids = [4, 31, 6]
    token_ids = [1, 10, *quoted_ids, *real_tool_ids]
    parser = _parser()

    reasoning, content = parser.extract_reasoning_from_token_ids(
        token_ids,
        tokenizer.decode(token_ids, skip_special_tokens=True),
    )

    assert parser.is_reasoning_end(token_ids)
    assert reasoning == (
        'Reason.<|"|><tool_call|><|tool_call>call:danger{}<tool_call|><|"|>'
    )
    assert content == "<|tool_call>call:good{}<tool_call|>"
    assert parser.extract_content_ids(token_ids) == real_tool_ids


def test_repeated_channel_markers_match_nonstream_across_40_chunkings():
    tokenizer = FakeTokenizer()
    token_ids = [1, 35, 2, 36, 1, 37, 2, 38]
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    expected = _parser().extract_reasoning(text, request=None)

    token_boundaries = [(split,) for split in range(1, len(token_ids))]
    token_boundaries += [
        (first, second)
        for first in range(1, len(token_ids))
        for second in range(first + 1, len(token_ids))
    ][:13]
    text_boundaries = [(index * len(text)) // 21 for index in range(1, 21)]
    chunkings = []
    for boundaries in token_boundaries:
        offsets = (0, *boundaries, len(token_ids))
        id_chunks = [token_ids[start:end] for start, end in zip(offsets, offsets[1:])]
        chunkings.append(
            (
                [tokenizer.decode(ids, skip_special_tokens=False) for ids in id_chunks],
                id_chunks,
            )
        )
    for boundary in text_boundaries:
        chunkings.append(([text[:boundary], text[boundary:]], [[], []]))

    assert expected == ("one", "mid<|channel>two<channel|>end")
    assert len(chunkings) == 40
    for chunks, id_chunks in chunkings:
        parser = _parser()
        previous_text = ""
        previous_ids = []
        reasoning = ""
        content = ""
        for chunk, delta_ids in zip(chunks, id_chunks):
            result = parser.extract_reasoning_streaming(
                previous_text,
                previous_text + chunk,
                chunk,
                previous_ids,
                previous_ids + delta_ids,
                delta_ids,
            )
            if result is not None:
                reasoning += result.reasoning or ""
                content += result.content or ""
            previous_text += chunk
            previous_ids += delta_ids

        assert (reasoning, content) == expected, (chunks, id_chunks)


def test_short_reasoning_prefix_flushed_as_reasoning_at_eos():
    parser = _parser()
    text = "<|channel>tho"

    streamed = parser.extract_reasoning_streaming("", text, text, [], [], [])
    fallback = parser.get_streaming_fallback_content(text, request=None)
    final = parser.finish_streaming()

    assert streamed is None
    assert fallback is None
    assert final is not None
    assert final.reasoning == "tho"
    assert final.content is None


def test_finish_streaming_is_idempotent():
    parser = _parser()
    text = "<|channel>tho<chan"

    assert parser.extract_reasoning_streaming("", text, text, [], [], []) is None

    first = parser.finish_streaming()
    second = parser.finish_streaming()

    assert first is not None
    assert first.reasoning == "tho<chan"
    assert first.content is None
    assert second is None


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


def test_token_id_extraction_ends_reasoning_at_tool_call():
    """A tool call terminates an open thinking channel; its payload must not
    be swallowed into the reasoning field."""
    reasoning, content = _parser().extract_reasoning_from_token_ids(
        [1, 10, 4, 14], "thought\nReason.call:get_weather{}"
    )

    assert reasoning == "Reason."
    assert content == "<|tool_call>call:get_weather{}"
