# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

_THOUGHT_PREFIX = "thought\n"
_TOOL_CALL_START = "<|tool_call>"
_QUOTE = '<|"|>'


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Google Gemma4 unified thinking models."""

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False
        self._stream_phase: str = "unknown"
        self._marker_buffer: str = ""
        self._text_reasoning_ended: bool = False
        self._id_reasoning_ended: bool = False
        self._stream_content_start: int | None = None
        self._prompt_reasoning_active: bool = False
        self._prompt_reasoning_in_quote: bool = False
        self.new_turn_token_id = self.vocab["<|turn>"]
        self.tool_call_token_id = self.vocab["<|tool_call>"]
        self.tool_response_token_id = self.vocab["<|tool_response>"]
        self.quote_token_id = self.vocab.get(_QUOTE)

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        request.skip_special_tokens = False
        return request

    @property
    def start_token(self) -> str:
        return "<|channel>"

    @property
    def end_token(self) -> str:
        return "<channel|>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        _, _, reasoning_ended = self._reasoning_id_state(input_ids)
        return reasoning_ended

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        del delta_ids
        _, _, token_reasoning_ended = self._reasoning_id_state(
            input_ids,
            initial_reasoning_open=self._prompt_reasoning_active,
            initial_in_quote=self._prompt_reasoning_in_quote,
        )
        # _id_reasoning_ended latches ID-path transitions to content, so a
        # trailing <|turn>/<|tool_response> in the same delta as the close
        # (the walk's fresh-turn reset) cannot un-end the orchestrator handoff
        # for a stream whose own extraction already moved past reasoning.
        return (
            self._text_reasoning_ended
            or self._id_reasoning_ended
            or token_reasoning_ended
        )

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        self._prompt_reasoning_active = False
        self._prompt_reasoning_in_quote = False
        if not prompt_token_ids:
            return

        reasoning_open, in_quote, _ = self._reasoning_id_state(prompt_token_ids)
        reset_ids = (self.new_turn_token_id, self.tool_response_token_id)
        last_boundary = next(
            (
                token_id
                for token_id in reversed(prompt_token_ids)
                if token_id
                in (
                    self.start_token_id,
                    self.end_token_id,
                    self.tool_call_token_id,
                    *reset_ids,
                )
            ),
            None,
        )
        if not reasoning_open and last_boundary not in reset_ids:
            return

        self._stream_phase = "reasoning"
        self._prompt_reasoning_active = True
        self._prompt_reasoning_in_quote = in_quote if reasoning_open else False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._stream_content_start is not None:
            return input_ids[self._stream_content_start :]

        prompt_in_quote = (
            self._prompt_reasoning_in_quote if self._prompt_reasoning_active else False
        )
        start_idx, _, _ = _scan_unquoted_token(
            input_ids,
            (self.start_token_id,),
            quote_token_id=self.quote_token_id,
            initial_in_quote=prompt_in_quote,
        )
        search_from = start_idx + 1 if start_idx is not None else 0
        transition_idx, transition_id, _ = _scan_unquoted_token(
            input_ids,
            (self.end_token_id, self.tool_call_token_id),
            start=search_from,
            quote_token_id=self.quote_token_id,
            initial_in_quote=prompt_in_quote if start_idx is None else False,
        )
        if transition_idx is not None:
            if transition_id == self.end_token_id:
                return input_ids[transition_idx + 1 :]
            return input_ids[transition_idx:]
        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        del request
        start_idx, _, _ = _scan_unquoted_marker(model_output, (self.start_token,))
        search_from = start_idx + len(self.start_token) if start_idx is not None else 0
        transition_idx, transition, _ = _scan_unquoted_marker(
            model_output,
            (self.end_token, _TOOL_CALL_START),
            start=search_from,
        )
        if start_idx is None and (
            transition_idx is None or transition != self.end_token
        ):
            return None, model_output

        reasoning_start = search_from
        if transition_idx is None:
            reasoning_end = len(model_output)
            content_start = None
        elif transition == _TOOL_CALL_START:
            reasoning_end = transition_idx
            content_start = transition_idx
        else:
            reasoning_end = transition_idx
            content_start = transition_idx + len(self.end_token)

        reasoning = _strip_thought_label(model_output[reasoning_start:reasoning_end])
        content = model_output[content_start:] if content_start is not None else None
        return reasoning or None, content or None

    def extract_reasoning_from_token_ids(
        self,
        token_ids: Sequence[int],
        fallback_text: str,
    ) -> tuple[str | None, str | None]:
        """Split a complete response when detokenization hid special markers.

        vLLM 0.24 passes raw output token IDs to its unified non-streaming
        parser, but its delegating parser does not forward them to legacy
        reasoning parsers. The plugin installs a narrow adapter for the
        ``diffusion_gemma`` alias that calls this method.
        """
        start_idx, _, _ = _scan_unquoted_token(
            token_ids,
            (self.start_token_id,),
            quote_token_id=self.quote_token_id,
        )
        search_from = start_idx + 1 if start_idx is not None else 0
        transition_idx, transition_id, _ = _scan_unquoted_token(
            token_ids,
            (self.end_token_id, self.tool_call_token_id),
            start=search_from,
            quote_token_id=self.quote_token_id,
        )
        if start_idx is None and (
            transition_idx is None or transition_id != self.end_token_id
        ):
            return None, fallback_text

        # A tool call inside an open thinking channel terminates it even
        # without the close marker (``is_reasoning_end`` treats it that way);
        # keep the call payload out of the reasoning field.
        if transition_id == self.tool_call_token_id:
            reasoning_end = transition_idx
            content_start = transition_idx
        elif transition_id == self.end_token_id:
            reasoning_end = transition_idx
            content_start = transition_idx + 1
        else:
            reasoning_end = len(token_ids)
            content_start = None

        reasoning_start = start_idx + 1 if start_idx is not None else 0
        reasoning = _strip_thought_label(
            self._decode_reasoning(token_ids[reasoning_start:reasoning_end])
        )
        content = (
            self._decode_raw(token_ids[content_start:])
            if content_start is not None
            else None
        )
        return reasoning or None, content or None

    def get_streaming_fallback_content(
        self,
        previous_text: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> str | None:
        del previous_text, request
        # DelegatingParser can only promote fallback text to content. Any text
        # withheld here is still reasoning (a possible ``thought\n`` prefix),
        # so leave it for finish_streaming(), which can preserve that field.
        return None

    def finish_streaming(self) -> DeltaMessage | None:
        reasoning: str | None = None
        content: str | None = None

        if (
            not self._prefix_stripped
            and self._reasoning_text
            and _THOUGHT_PREFIX.startswith(self._reasoning_text)
        ):
            reasoning = self._reasoning_text
            self._prefix_stripped = True

        if self._marker_buffer:
            buffered = self._marker_buffer
            self._marker_buffer = ""
            if self._stream_phase == "reasoning":
                reasoning = (reasoning or "") + buffered
            else:
                content = buffered

        if reasoning is None and content is None:
            return None
        return DeltaMessage(reasoning=reasoning, content=content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del current_text, current_token_ids
        result = self._extract_stream_delta(
            delta_text,
            previous_text,
            previous_token_ids,
            delta_token_ids,
        )
        if result is None or result.reasoning is None:
            return self._flush_short_reasoning_on_transition(result)

        self._reasoning_text += result.reasoning

        if self._prefix_stripped:
            return result

        if self._reasoning_text.startswith(_THOUGHT_PREFIX):
            prefix_len = len(_THOUGHT_PREFIX)
            prev_reasoning_len = len(self._reasoning_text) - len(result.reasoning)
            if prev_reasoning_len >= prefix_len:
                self._prefix_stripped = True
                return result

            chars_of_prefix_in_delta = prefix_len - prev_reasoning_len
            stripped = result.reasoning[chars_of_prefix_in_delta:]
            if stripped:
                self._prefix_stripped = True
                result.reasoning = stripped
                return result

            if len(self._reasoning_text) >= prefix_len:
                self._prefix_stripped = True
                result.reasoning = ""
                return result
            return None

        if _THOUGHT_PREFIX.startswith(self._reasoning_text):
            # If the reasoning marker also ended in this delta, the short text
            # is the complete reasoning body rather than a partial label.
            if self._stream_phase == "content":
                self._prefix_stripped = True
                result.reasoning = self._reasoning_text
                return result
            return None

        self._prefix_stripped = True
        result.reasoning = self._reasoning_text
        return result

    def _flush_short_reasoning_on_transition(
        self, result: DeltaMessage | None
    ) -> DeltaMessage | None:
        if (
            self._stream_phase != "content"
            or self._prefix_stripped
            or not self._reasoning_text
            or not _THOUGHT_PREFIX.startswith(self._reasoning_text)
        ):
            return result

        self._prefix_stripped = True
        if result is None:
            result = DeltaMessage()
        result.reasoning = self._reasoning_text
        return result

    def _extract_stream_delta(
        self,
        delta_text: str,
        previous_text: str,
        previous_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Split a delta by marker IDs, even when decoded marker text is absent."""
        self._stream_content_start = None
        start_idx, _, _ = _scan_unquoted_token(
            delta_token_ids,
            (self.start_token_id,),
            quote_token_id=self.quote_token_id,
        )
        if (
            start_idx is not None
            and self._stream_phase != "reasoning"
            and not self.is_reasoning_end_streaming(previous_token_ids, ())
        ):
            buffered_prefix = self._marker_buffer
            self._marker_buffer = ""
            transition_idx, transition_id, _ = _scan_unquoted_token(
                delta_token_ids,
                (self.end_token_id, self.tool_call_token_id),
                start=start_idx + 1,
                quote_token_id=self.quote_token_id,
            )
            prefix = buffered_prefix + self._decode_raw(delta_token_ids[:start_idx])
            if transition_id == self.tool_call_token_id:
                reasoning_end = transition_idx
                content_start = transition_idx
            elif transition_id == self.end_token_id:
                reasoning_end = transition_idx
                content_start = transition_idx + 1
            else:
                self._stream_phase = "reasoning"
                reasoning = self._decode_reasoning(delta_token_ids[start_idx + 1 :])
                return _delta(prefix, reasoning, None)

            self._stream_phase = "content"
            self._id_reasoning_ended = True
            self._stream_content_start = content_start
            reasoning = self._decode_reasoning(
                delta_token_ids[start_idx + 1 : reasoning_end]
            )
            content = self._decode_raw(delta_token_ids[content_start:])
            return _delta(prefix, reasoning, content)

        if self._stream_phase == "unknown":
            end_idx, _, _ = _scan_unquoted_token(
                delta_token_ids,
                (self.end_token_id,),
                quote_token_id=self.quote_token_id,
            )
            if end_idx is not None:
                buffered_reasoning = self._marker_buffer
                self._marker_buffer = ""
                self._stream_phase = "content"
                self._text_reasoning_ended = True
                self._stream_content_start = end_idx + 1
                return _delta(
                    None,
                    buffered_reasoning
                    + self._decode_reasoning(delta_token_ids[:end_idx]),
                    self._decode_raw(delta_token_ids[end_idx + 1 :]),
                )

        previous_in_reasoning = (
            self.start_token_id in previous_token_ids
            and not self.is_reasoning_end(previous_token_ids)
        )
        if previous_in_reasoning or self._stream_phase == "reasoning":
            initial_in_quote = self._reasoning_id_quote_state(previous_token_ids)
            transition_idx, transition_id, _ = _scan_unquoted_token(
                delta_token_ids,
                (self.end_token_id, self.tool_call_token_id),
                quote_token_id=self.quote_token_id,
                initial_in_quote=initial_in_quote,
            )
            if transition_idx is not None:
                buffered_reasoning = self._marker_buffer
                self._marker_buffer = ""
                if transition_id == self.tool_call_token_id:
                    reasoning_end = transition_idx
                    content_start = transition_idx
                else:
                    reasoning_end = transition_idx
                    content_start = transition_idx + 1
                self._stream_phase = "content"
                self._id_reasoning_ended = True
                self._stream_content_start = content_start
                return _delta(
                    None,
                    buffered_reasoning
                    + self._decode_reasoning(
                        delta_token_ids[:reasoning_end],
                        initial_in_quote=initial_in_quote,
                    ),
                    self._decode_raw(delta_token_ids[content_start:]),
                )
            if any(
                token_id
                in (
                    self.start_token_id,
                    self.end_token_id,
                    self.tool_call_token_id,
                    self.quote_token_id,
                )
                for token_id in delta_token_ids
            ):
                reasoning = self._decode_reasoning(
                    delta_token_ids,
                    initial_in_quote=initial_in_quote,
                )
                return DeltaMessage(reasoning=reasoning) if reasoning else None
            return self._extract_stream_text(
                delta_text,
                previous_text,
                initial_in_quote=self._reasoning_text_quote_state(previous_text),
            )

        if (
            self._marker_buffer
            or self.end_token in delta_text
            or _TOOL_CALL_START in delta_text
        ):
            return self._extract_stream_text(delta_text, previous_text)
        if self.is_reasoning_end(previous_token_ids) or self._stream_phase == "content":
            return DeltaMessage(content=delta_text) if delta_text else None

        # Text fallback covers tokenizers that expose marker text in pieces
        # across deltas. Complete special-token IDs take the path above.
        return self._extract_stream_text(delta_text, previous_text)

    def _extract_stream_text(
        self,
        delta_text: str,
        previous_text: str,
        *,
        initial_in_quote: bool | None = None,
    ) -> DeltaMessage | None:
        combined = self._marker_buffer + delta_text
        self._marker_buffer = ""
        if self._stream_phase == "unknown":
            start_idx, _, _ = _scan_unquoted_marker(combined, (self.start_token,))
            if start_idx is not None:
                prefix = combined[:start_idx]
                after = combined[start_idx + len(self.start_token) :]
                self._stream_phase = "reasoning"
                return self._extract_reasoning_text(
                    after, prefix or None, initial_in_quote=False
                )
            end_idx, _, _ = _scan_unquoted_marker(combined, (self.end_token,))
            if end_idx is not None:
                self._stream_phase = "reasoning"
                return self._extract_reasoning_text(
                    combined, None, initial_in_quote=False
                )
            if combined.startswith(_THOUGHT_PREFIX):
                self._stream_phase = "reasoning"
                return self._extract_reasoning_text(
                    combined, None, initial_in_quote=False
                )

            held = _longest_unquoted_marker_prefix_suffix(
                combined, (self.start_token, self.end_token)
            )
            if (
                combined
                and _THOUGHT_PREFIX.startswith(combined)
                and len(combined) > len(held)
            ):
                held = combined
            if held:
                self._marker_buffer = held
                visible = combined[: -len(held)]
                return DeltaMessage(content=visible) if visible else None
            self._stream_phase = "content"
            self._text_reasoning_ended = True
            return DeltaMessage(content=combined) if combined else None

        if self._stream_phase == "reasoning":
            if initial_in_quote is None:
                initial_in_quote = self._reasoning_text_quote_state(previous_text)
            return self._extract_reasoning_text(
                combined, None, initial_in_quote=initial_in_quote
            )
        return DeltaMessage(content=combined) if combined else None

    def _extract_reasoning_text(
        self,
        text: str,
        prefix_content: str | None,
        *,
        initial_in_quote: bool,
    ) -> DeltaMessage | None:
        marker_idx, marker, _ = _scan_unquoted_marker(
            text,
            (self.end_token, _TOOL_CALL_START),
            initial_in_quote=initial_in_quote,
        )
        if marker_idx is not None:
            reasoning = text[:marker_idx]
            content_start = (
                marker_idx
                if marker == _TOOL_CALL_START
                else marker_idx + len(self.end_token)
            )
            content = text[content_start:]
            self._stream_phase = "content"
            self._text_reasoning_ended = True
            return _delta(prefix_content, reasoning, content)

        held = _longest_unquoted_marker_prefix_suffix(
            text,
            (self.end_token, _TOOL_CALL_START),
            initial_in_quote=initial_in_quote,
        )
        quote_held = _longest_marker_prefix_suffix(text, _QUOTE)
        if len(quote_held) > len(held):
            held = quote_held
        if held:
            self._marker_buffer = held
            text = text[: -len(held)]
        return _delta(prefix_content, text, None)

    def _reasoning_id_state(
        self,
        token_ids: Sequence[int],
        *,
        initial_reasoning_open: bool = False,
        initial_in_quote: bool = False,
    ) -> tuple[bool, bool, bool]:
        reasoning_open = initial_reasoning_open
        in_quote = initial_in_quote if initial_reasoning_open else False
        reasoning_ended = False

        for token_id in token_ids:
            if reasoning_open:
                if token_id in (self.new_turn_token_id, self.tool_response_token_id):
                    reasoning_open = False
                    in_quote = False
                    reasoning_ended = False
                    continue
                if self.quote_token_id is not None and token_id == self.quote_token_id:
                    in_quote = not in_quote
                    continue
                if in_quote:
                    continue
                if token_id == self.start_token_id:
                    in_quote = False
                    reasoning_ended = False
                elif token_id in (self.end_token_id, self.tool_call_token_id):
                    reasoning_open = False
                    reasoning_ended = True
                continue

            if token_id == self.start_token_id and not reasoning_ended:
                reasoning_open = True
                in_quote = False
                reasoning_ended = False
            elif token_id in (self.end_token_id, self.tool_call_token_id):
                reasoning_ended = True
            elif token_id in (self.new_turn_token_id, self.tool_response_token_id):
                reasoning_ended = False

        return reasoning_open, in_quote, reasoning_ended

    def _reasoning_text_quote_state(
        self,
        previous_text: str,
    ) -> bool:
        start_idx, _, _ = _scan_unquoted_marker(
            previous_text,
            (self.start_token,),
            initial_in_quote=self._prompt_reasoning_in_quote,
        )
        text_start = start_idx + len(self.start_token) if start_idx is not None else 0
        _, _, in_quote = _scan_unquoted_marker(
            previous_text,
            (),
            start=text_start,
            initial_in_quote=(
                False if start_idx is not None else self._prompt_reasoning_in_quote
            ),
        )
        return in_quote

    def _reasoning_id_quote_state(
        self,
        previous_token_ids: Sequence[int],
    ) -> bool:
        reasoning_open, in_quote, _ = self._reasoning_id_state(
            previous_token_ids,
            initial_reasoning_open=self._prompt_reasoning_active,
            initial_in_quote=self._prompt_reasoning_in_quote,
        )
        return reasoning_open and self.quote_token_id is not None and in_quote

    def _decode_visible(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self.model_tokenizer.decode(
                list(token_ids), skip_special_tokens=True
            )
        except TypeError:
            return self.model_tokenizer.decode(list(token_ids))

    def _decode_reasoning(
        self,
        token_ids: Sequence[int],
        *,
        initial_in_quote: bool = False,
    ) -> str:
        if self.quote_token_id is None or self.quote_token_id not in token_ids:
            return (
                self._decode_raw(token_ids)
                if initial_in_quote
                else self._decode_visible(token_ids)
            )

        decoded: list[str] = []
        segment_start = 0
        in_quote = initial_in_quote
        for idx, token_id in enumerate(token_ids):
            if token_id != self.quote_token_id:
                continue
            segment = token_ids[segment_start:idx]
            decoded.append(
                self._decode_raw(segment) if in_quote else self._decode_visible(segment)
            )
            decoded.append(self._decode_raw((token_id,)))
            in_quote = not in_quote
            segment_start = idx + 1

        segment = token_ids[segment_start:]
        decoded.append(
            self._decode_raw(segment) if in_quote else self._decode_visible(segment)
        )
        return "".join(decoded)

    def _decode_raw(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self.model_tokenizer.decode(
                list(token_ids), skip_special_tokens=False
            )
        except TypeError:
            return self.model_tokenizer.decode(list(token_ids))


def _strip_thought_label(text: str) -> str:
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text


def _scan_unquoted_marker(
    text: str,
    markers: Sequence[str],
    *,
    start: int = 0,
    initial_in_quote: bool = False,
) -> tuple[int | None, str | None, bool]:
    in_quote = initial_in_quote
    idx = start
    while idx < len(text):
        if text.startswith(_QUOTE, idx):
            in_quote = not in_quote
            idx += len(_QUOTE)
            continue
        if not in_quote:
            for marker in markers:
                if text.startswith(marker, idx):
                    return idx, marker, in_quote
        idx += 1
    return None, None, in_quote


def _scan_unquoted_token(
    token_ids: Sequence[int],
    marker_ids: Sequence[int],
    *,
    start: int = 0,
    quote_token_id: int | None,
    initial_in_quote: bool = False,
) -> tuple[int | None, int | None, bool]:
    in_quote = initial_in_quote
    for idx in range(start, len(token_ids)):
        token_id = token_ids[idx]
        if quote_token_id is not None and token_id == quote_token_id:
            in_quote = not in_quote
            continue
        if not in_quote and token_id in marker_ids:
            return idx, token_id, in_quote
    return None, None, in_quote


def _longest_unquoted_marker_prefix_suffix(
    text: str,
    markers: Sequence[str],
    *,
    initial_in_quote: bool = False,
) -> str:
    held = ""
    for marker in markers:
        max_len = min(len(text), len(marker) - 1)
        for length in range(max_len, 0, -1):
            if not text.endswith(marker[:length]):
                continue
            prefix = text[:-length]
            _, _, in_quote = _scan_unquoted_marker(
                prefix,
                (),
                initial_in_quote=initial_in_quote,
            )
            if not in_quote and length > len(held):
                held = text[-length:]
            break
    return held


def _longest_marker_prefix_suffix(text: str, marker: str) -> str:
    max_len = min(len(text), len(marker) - 1)
    for length in range(max_len, 0, -1):
        if text.endswith(marker[:length]):
            return text[-length:]
    return ""


def _delta(
    prefix_content: str | None,
    reasoning: str | None,
    content: str | None,
) -> DeltaMessage | None:
    combined_content = (prefix_content or "") + (content or "")
    if not reasoning and not combined_content:
        return None
    return DeltaMessage(
        reasoning=reasoning or None,
        content=combined_content or None,
    )
