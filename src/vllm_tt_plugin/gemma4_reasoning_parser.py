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


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Google Gemma4 unified thinking models."""

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False
        self._stream_phase: str = "unknown"
        self._marker_buffer: str = ""
        self.new_turn_token_id = self.vocab["<|turn>"]
        self.tool_call_token_id = self.vocab["<|tool_call>"]
        self.tool_response_token_id = self.vocab["<|tool_response>"]

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
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        new_turn_token_id = self.new_turn_token_id
        tool_call_token_id = self.tool_call_token_id
        tool_response_token_id = self.tool_response_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == tool_call_token_id:
                return True
            if input_ids[i] in (new_turn_token_id, tool_response_token_id):
                return False
            if input_ids[i] == end_token_id:
                return True
        return False

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output

        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning is not None:
            reasoning = _strip_thought_label(reasoning)
        return reasoning, content

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
        try:
            start_idx = token_ids.index(self.start_token_id)
        except ValueError:
            start_idx = None

        search_from = start_idx + 1 if start_idx is not None else 0
        end_idx = _index_after(token_ids, self.end_token_id, search_from)
        if start_idx is None and end_idx is None:
            return None, fallback_text

        # A tool call inside an open thinking channel terminates it even
        # without the close marker (``is_reasoning_end`` treats it that way);
        # keep the call payload out of the reasoning field.
        tool_idx = (
            _index_after(token_ids, self.tool_call_token_id, search_from)
            if start_idx is not None
            else None
        )
        if tool_idx is not None and (end_idx is None or tool_idx < end_idx):
            reasoning_end = tool_idx
            content_start = tool_idx
        elif end_idx is not None:
            reasoning_end = end_idx
            content_start = end_idx + 1
        else:
            reasoning_end = len(token_ids)
            content_start = None

        reasoning_start = start_idx + 1 if start_idx is not None else 0
        reasoning = _strip_thought_label(
            self._decode_visible(token_ids[reasoning_start:reasoning_end])
        )
        content = (
            self._decode_visible(token_ids[content_start:])
            if content_start is not None
            else None
        )
        return reasoning or None, content or None

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
            delta_text, previous_token_ids, delta_token_ids
        )
        if result is None:
            return None

        if result.reasoning is None:
            return result

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
            if result.content is not None:
                self._prefix_stripped = True
                result.reasoning = self._reasoning_text
                return result
            return None

        self._prefix_stripped = True
        result.reasoning = self._reasoning_text
        return result

    def _extract_stream_delta(
        self,
        delta_text: str,
        previous_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Split a delta by marker IDs, even when decoded marker text is absent."""
        if self.start_token_id in delta_token_ids:
            start_idx = delta_token_ids.index(self.start_token_id)
            end_idx = _index_after(delta_token_ids, self.end_token_id, start_idx + 1)
            prefix = self._decode_visible(delta_token_ids[:start_idx])
            if end_idx is None:
                self._stream_phase = "reasoning"
                reasoning = self._decode_visible(delta_token_ids[start_idx + 1 :])
                return _delta(prefix, reasoning, None)

            self._stream_phase = "content"
            reasoning = self._decode_visible(delta_token_ids[start_idx + 1 : end_idx])
            content = self._decode_visible(delta_token_ids[end_idx + 1 :])
            return _delta(prefix, reasoning, content)

        previous_in_reasoning = (
            self.start_token_id in previous_token_ids
            and not self.is_reasoning_end(previous_token_ids)
        )
        if self._marker_buffer or self.end_token in delta_text:
            return self._extract_stream_text(delta_text)
        if previous_in_reasoning or self._stream_phase == "reasoning":
            if self.end_token_id in delta_token_ids:
                end_idx = delta_token_ids.index(self.end_token_id)
                self._stream_phase = "content"
                return _delta(
                    None,
                    self._decode_visible(delta_token_ids[:end_idx]),
                    self._decode_visible(delta_token_ids[end_idx + 1 :]),
                )
            return DeltaMessage(reasoning=delta_text) if delta_text else None

        if self.is_reasoning_end(previous_token_ids) or self._stream_phase == "content":
            return DeltaMessage(content=delta_text) if delta_text else None

        # Text fallback covers tokenizers that expose marker text in pieces
        # across deltas. Complete special-token IDs take the path above.
        return self._extract_stream_text(delta_text)

    def _extract_stream_text(self, delta_text: str) -> DeltaMessage | None:
        combined = self._marker_buffer + delta_text
        self._marker_buffer = ""
        if self._stream_phase == "unknown":
            if self.start_token in combined:
                prefix, _, after = combined.partition(self.start_token)
                self._stream_phase = "reasoning"
                return self._extract_reasoning_text(after, prefix or None)
            held = _longest_marker_prefix_suffix(combined, self.start_token)
            if held:
                self._marker_buffer = held
                visible = combined[: -len(held)]
                return DeltaMessage(content=visible) if visible else None
            self._stream_phase = "content"
            return DeltaMessage(content=combined) if combined else None

        if self._stream_phase == "reasoning":
            return self._extract_reasoning_text(combined, None)
        return DeltaMessage(content=combined) if combined else None

    def _extract_reasoning_text(
        self, text: str, prefix_content: str | None
    ) -> DeltaMessage | None:
        if self.end_token in text:
            reasoning, _, content = text.partition(self.end_token)
            self._stream_phase = "content"
            return _delta(prefix_content, reasoning, content)

        held = _longest_marker_prefix_suffix(text, self.end_token)
        if held:
            self._marker_buffer = held
            text = text[: -len(held)]
        return _delta(prefix_content, text, None)

    def _decode_visible(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self.model_tokenizer.decode(
                list(token_ids), skip_special_tokens=True
            )
        except TypeError:
            return self.model_tokenizer.decode(list(token_ids))


def _strip_thought_label(text: str) -> str:
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text


def _index_after(token_ids: Sequence[int], token_id: int, start: int) -> int | None:
    for idx in range(start, len(token_ids)):
        if token_ids[idx] == token_id:
            return idx
    return None


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
