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
        28: "Before ",
        29: "call:danger{}",
        30: " middle ",
        31: "call:good{}",
        32: " after",
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


def _stream_reasoning_text(chunks, parser=None):
    parser = parser or _parser()
    previous_text = ""
    reasoning = ""
    content = ""
    for chunk in chunks:
        current_text = previous_text + chunk
        result = parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            chunk,
            [],
            [],
            [],
        )
        if result is not None:
            reasoning += result.reasoning or ""
            content += result.content or ""
        previous_text = current_text
    return parser, reasoning, content


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


def test_unclosed_quote_keeps_text_tool_marker_in_reasoning():
    text = '<|channel>thought\nReason.<|"|><|tool_call>call:danger{}<tool_call|>'

    reasoning, content = _parser().extract_reasoning(text, request=None)

    assert reasoning == 'Reason.<|"|><|tool_call>call:danger{}<tool_call|>'
    assert content is None


def test_real_text_tool_marker_after_closed_quote_ends_reasoning():
    quoted = '<|"|><|tool_call>call:danger{}<tool_call|><|"|>'
    real_tool = "<|tool_call>call:good{}<tool_call|>"

    reasoning, content = _parser().extract_reasoning(
        f"<|channel>thought\nReason.{quoted}{real_tool}",
        request=None,
    )

    assert reasoning == f"Reason.{quoted}"
    assert content == real_tool


def test_quoted_token_ids_do_not_end_reasoning():
    tokenizer = FakeTokenizer()
    quoted_ids = [7, 4, 29, 6, 7]
    token_ids = [1, 10, *quoted_ids]
    parser = _parser()

    reasoning, content = parser.extract_reasoning_from_token_ids(
        token_ids,
        tokenizer.decode(token_ids, skip_special_tokens=True),
    )

    assert not parser.is_reasoning_end(token_ids)
    assert reasoning == ('Reason.<|"|><|tool_call>call:danger{}<tool_call|><|"|>')
    assert content is None
    assert parser.extract_content_ids(token_ids) == []


def test_unclosed_quote_blocks_token_id_transition():
    tokenizer = FakeTokenizer()
    token_ids = [1, 10, 7, 2, 4, 29, 6]
    parser = _parser()

    reasoning, content = parser.extract_reasoning_from_token_ids(
        token_ids,
        tokenizer.decode(token_ids, skip_special_tokens=True),
    )

    assert not parser.is_reasoning_end(token_ids)
    assert reasoning == ('Reason.<|"|><channel|><|tool_call>call:danger{}<tool_call|>')
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


def test_streaming_token_ids_keep_quoted_tool_literal_across_deltas():
    tokenizer = FakeTokenizer()
    first_ids = [1, 10, 7]
    second_ids = [4, 29, 6, 7]

    for skip_special_tokens in (False, True):
        parser = _parser()
        first_text = tokenizer.decode(
            first_ids, skip_special_tokens=skip_special_tokens
        )
        second_text = tokenizer.decode(
            second_ids, skip_special_tokens=skip_special_tokens
        )
        first = parser.extract_reasoning_streaming(
            "",
            first_text,
            first_text,
            [],
            first_ids,
            first_ids,
        )
        second = parser.extract_reasoning_streaming(
            first_text,
            first_text + second_text,
            second_text,
            first_ids,
            first_ids + second_ids,
            second_ids,
        )

        assert first is not None
        assert first.reasoning == 'Reason.<|"|>'
        assert second is not None
        assert second.reasoning == '<|tool_call>call:danger{}<tool_call|><|"|>'
        assert second.content is None
        assert not parser.is_reasoning_end_streaming(first_ids + second_ids, second_ids)


def test_fragmented_quote_delimiters_preserve_reasoning_transitions():
    quote = '<|"|>'
    danger = "<|tool_call>call:danger{}<tool_call|>"
    good = "<|tool_call>call:good{}<tool_call|>"
    prefix = "<|channel>thought\nR."
    expected_reasoning = f"R.{quote}{danger}{quote}"

    chunkings = [[f"{prefix}{quote}{danger}{quote}{good}"]]
    for split in range(1, len(quote)):
        chunkings.append(
            [
                f"{prefix}{quote[:split]}",
                f"{quote[split:]}{danger}{quote}{good}",
            ]
        )
        chunkings.append(
            [
                f"{prefix}{quote}{danger}{quote[:split]}",
                f"{quote[split:]}{good}",
            ]
        )

    for chunks in chunkings:
        _, reasoning, content = _stream_reasoning_text(chunks)

        assert reasoning == expected_reasoning
        assert content == good


def test_partial_quote_at_eos_flushes_as_reasoning():
    parser, reasoning, content = _stream_reasoning_text(["<|channel>thought\nR.<|"])
    final = parser.finish_streaming()

    assert reasoning == "R."
    assert content == ""
    assert final is not None
    assert final.reasoning == "<|"
    assert final.content is None
    assert parser.finish_streaming() is None


def test_empty_prompt_omitted_start_text_is_chunk_invariant():
    chunkings = (
        ["thought\nR.<channel|>A"],
        ["tho", "ught\nR.", "<channel|>A"],
        ["thought", "\nR.<chan", "nel|>A"],
        ["tho", "ught\nR.<chan", "nel|>A"],
        ["thought\nR.<chan", "nel|>A"],
    )

    for chunks in chunkings:
        parser = _parser()
        parser.adjust_initial_state_from_prompt([])
        _, reasoning, content = _stream_reasoning_text(chunks, parser=parser)

        assert reasoning == "R."
        assert content == "A"
        assert parser._stream_phase == "content"


def test_empty_prompt_omitted_start_token_ids_split_at_end():
    tokenizer = FakeTokenizer()
    parser = _parser()
    parser.adjust_initial_state_from_prompt([])
    token_ids = [12, 2, 11]
    text = tokenizer.decode(token_ids, skip_special_tokens=False)

    result = parser.extract_reasoning_streaming(
        "",
        text,
        text,
        [],
        token_ids,
        token_ids,
    )

    assert result is not None
    assert result.reasoning == "Part "
    assert result.content == "Answer."
    assert parser.extract_content_ids(token_ids) == [11]


def test_empty_prompt_quoted_end_and_plain_text_remain_content():
    quoted_end = 'Before <|"|><channel|><|"|> after'
    for text in (quoted_end, "Plain answer."):
        parser = _parser()
        parser.adjust_initial_state_from_prompt([])
        _, reasoning, content = _stream_reasoning_text([text], parser=parser)

        assert reasoning == ""
        assert content == text
        assert parser._stream_phase == "content"


def test_prompt_open_omitted_start_text_splits_reasoning_and_content():
    parser = _parser()
    parser.adjust_initial_state_from_prompt([1])

    _, reasoning, content = _stream_reasoning_text(
        ["thought\nR.<channel|>A"],
        parser=parser,
    )

    assert reasoning == "R."
    assert content == "A"
    assert parser.is_reasoning_end_streaming([], [])


def test_prompt_open_omitted_start_token_ids_split_reasoning_and_content():
    tokenizer = FakeTokenizer()
    parser = _parser()
    parser.adjust_initial_state_from_prompt([1])
    token_ids = [12, 2, 11]
    text = tokenizer.decode(token_ids, skip_special_tokens=False)

    result = parser.extract_reasoning_streaming(
        "",
        text,
        text,
        [],
        token_ids,
        token_ids,
    )

    assert result is not None
    assert result.reasoning == "Part "
    assert result.content == "Answer."
    assert parser.is_reasoning_end_streaming(token_ids, token_ids)
    assert parser.extract_content_ids(token_ids) == [11]


def test_prompt_quote_state_carries_into_text_and_token_ids():
    tokenizer = FakeTokenizer()
    danger = "<|tool_call>call:danger{}<tool_call|>"
    good = "<|tool_call>call:good{}<tool_call|>"

    text_parser = _parser()
    text_parser.adjust_initial_state_from_prompt([1, 7])
    _, reasoning, content = _stream_reasoning_text(
        [f"{danger}<|", f'"|>{good}'],
        parser=text_parser,
    )

    assert reasoning == f'{danger}<|"|>'
    assert content == good

    id_parser = _parser()
    id_parser.adjust_initial_state_from_prompt([1, 7])
    token_ids = [4, 29, 6, 7, 4, 31, 6]
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    result = id_parser.extract_reasoning_streaming(
        "",
        text,
        text,
        [],
        token_ids,
        token_ids,
    )

    assert result is not None
    assert result.reasoning == f'{danger}<|"|>'
    assert result.content == good
    assert id_parser.is_reasoning_end_streaming(token_ids, token_ids)
    assert id_parser.extract_content_ids(token_ids) == [4, 31, 6]

    content_parser = _parser()
    content_parser.adjust_initial_state_from_prompt([1, 7])
    assert content_parser.extract_content_ids(token_ids) == [4, 31, 6]


def test_prompt_turn_and_tool_response_reset_start_fresh_reasoning():
    for reset_id in (3, 5):
        parser = _parser()
        parser.adjust_initial_state_from_prompt([1, 7, reset_id])

        _, reasoning, content = _stream_reasoning_text(
            ["thought\nR.<channel|>A"],
            parser=parser,
        )

        assert reasoning == "R."
        assert content == "A"


def test_empty_and_arbitrary_prompts_keep_no_channel_content():
    for prompt_token_ids in ([], [99]):
        parser = _parser()
        parser.adjust_initial_state_from_prompt(prompt_token_ids)
        _, reasoning, content = _stream_reasoning_text(
            ["Plain answer."],
            parser=parser,
        )

        assert reasoning == ""
        assert content == "Plain answer."
        assert parser._stream_phase == "content"


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


def test_short_reasoning_prefix_combines_with_held_marker_at_eos():
    parser = _parser()
    first_text = "<|channel>tho"
    held_text = "<chan"

    first = parser.extract_reasoning_streaming("", first_text, first_text, [], [], [])
    held = parser.extract_reasoning_streaming(
        first_text,
        first_text + held_text,
        held_text,
        [],
        [],
        [],
    )
    final = parser.finish_streaming()

    assert first is None
    assert held is None
    assert final is not None
    assert final.reasoning == "tho<chan"
    assert final.content is None


def test_held_start_marker_flushed_as_content_at_eos():
    parser = _parser()
    text = "Answer.<|chan"

    streamed = parser.extract_reasoning_streaming("", text, text, [], [], [])
    final = parser.finish_streaming()

    assert streamed is not None
    assert streamed.reasoning is None
    assert streamed.content == "Answer."
    assert final is not None
    assert final.reasoning is None
    assert final.content == "<|chan"


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
