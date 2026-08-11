# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import json
from collections.abc import Sequence
from typing import Any

from openai.types.responses import ToolChoiceFunction
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

# Gemma4 tool-call wire format (emitted by the model, see the chat template):
#
#   <|tool_call>call:NAME{key:<|"|>str<|"|>,flag:true,n:3,xs:[1,2]}<tool_call|>
#
# Strings are wrapped in the ``<|"|>`` quote token; booleans render as
# ``true``/``false``; objects/arrays use ``{k:v,...}`` / ``[v,...]`` with
# *unescaped* keys; numbers are bare. This differs from FunctionGemma's
# ``<start_function_call>`` / ``<escape>`` format, so Gemma4 gets its own parser.
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
QUOTE = '<|"|>'
CALL_PREFIX = "call:"


class Gemma4ToolParser(ToolParser):
    """Tool parser for Google Gemma4 unified models.

    Parses the ``<|tool_call>``/``<tool_call|>`` delimited function-call format
    and normalizes arguments into a JSON string for the OpenAI tool-call schema.
    """

    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Any] | None = None):
        super().__init__(tokenizer, tools)

        # Streaming state.
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self._raw_to_public_tool_index: dict[int, int] = {}
        self._next_public_tool_index: int = 0

        self.tool_call_start_token: str = TOOL_CALL_START
        self.tool_call_end_token: str = TOOL_CALL_END

        # Buffer holding a possibly-partial trailing special token across
        # streaming deltas (the tokens span multiple pieces).
        self.buffered_delta_text: str = ""
        self._last_current_text: str = ""

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if request.tools:
            tool_choice = request.tool_choice
            if tool_choice == "required" or isinstance(
                tool_choice,
                (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
            ):
                request.skip_special_tokens = False
                return request
        request = super().adjust_request(request)
        # The format is built entirely out of special tokens, so they must not
        # be stripped from the decoded output.
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    # ------------------------------------------------------------------
    # Gemma4 argument-value grammar (recursive descent)
    # ------------------------------------------------------------------
    def _parse_value(self, s: str, i: int) -> tuple[Any, int]:
        """Parse one value starting at ``s[i]``; return (value, next_index)."""
        n = len(s)
        if s.startswith(QUOTE, i):
            inner_start = i + len(QUOTE)
            end = s.find(QUOTE, inner_start)
            if end == -1:
                return s[inner_start:], n
            return s[inner_start:end], end + len(QUOTE)

        c = s[i]
        if c == "{":
            return self._parse_object(s, i)
        if c == "[":
            return self._parse_array(s, i)
        if c in ",}]":
            raise ValueError(f"Unexpected delimiter {c!r} at position {i}")

        # Bare token: number / bool / null (read up to the next delimiter).
        j = i
        while j < n and s[j] not in ",}]" and not s.startswith(QUOTE, j):
            j += 1
        return self._coerce_scalar(s[i:j].strip()), j

    def _parse_object(self, s: str, i: int) -> tuple[dict, int]:
        """Parse ``{k:v,...}`` starting at the opening brace ``s[i] == '{'``."""
        obj: dict[str, Any] = {}
        n = len(s)
        i += 1  # consume '{'
        while i < n:
            while i < n and s[i] in ", ":
                i += 1
            if i >= n or s[i] == "}":
                break
            colon = s.find(":", i)
            if colon == -1:
                break
            key = s[i:colon].strip()
            if (
                key.startswith(QUOTE)
                and key.endswith(QUOTE)
                and len(key) >= 2 * len(QUOTE)
            ):
                key = key[len(QUOTE) : -len(QUOTE)]
            value_start = colon + 1
            value, i = self._parse_value(s, value_start)
            if i <= value_start:
                raise ValueError(f"Parser made no progress at position {value_start}")
            obj[key] = value
        if i < n and s[i] == "}":
            i += 1
        return obj, i

    def _parse_array(self, s: str, i: int) -> tuple[list, int]:
        """Parse ``[v,...]`` starting at the opening bracket ``s[i] == '['``."""
        arr: list[Any] = []
        n = len(s)
        i += 1  # consume '['
        while i < n:
            while i < n and s[i] in ", ":
                i += 1
            if i >= n or s[i] == "]":
                break
            value_start = i
            value, i = self._parse_value(s, value_start)
            if i <= value_start:
                raise ValueError(f"Parser made no progress at position {value_start}")
            arr.append(value)
        if i < n and s[i] == "]":
            i += 1
        return arr, i

    @staticmethod
    def _coerce_scalar(token: str) -> Any:
        if token == "true":
            return True
        if token == "false":
            return False
        if token in ("null", "None", ""):
            return None
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            pass
        return token

    def _parse_call(self, block_body: str) -> tuple[str | None, dict]:
        """Best-effort parsing for an unfinished call at stream finalization."""
        if not block_body.startswith(CALL_PREFIX):
            return None, {}
        brace = block_body.find("{", len(CALL_PREFIX))
        if brace == -1:
            return block_body[len(CALL_PREFIX) :].strip() or None, {}
        name = block_body[len(CALL_PREFIX) : brace].strip()
        arguments, _ = self._parse_object(block_body, brace)
        return (name or None), arguments

    def _parse_completed_call(self, block_body: str) -> tuple[str, dict]:
        """Strictly parse one delimiter-complete ``call:NAME{...}`` frame."""
        if not block_body.startswith(CALL_PREFIX):
            raise ValueError("Tool call is missing the call: prefix")

        brace = block_body.find("{", len(CALL_PREFIX))
        if brace == -1:
            raise ValueError("Tool call is missing its argument object")
        name = block_body[len(CALL_PREFIX) : brace].strip()
        if not name:
            raise ValueError("Tool call has an empty function name")

        stack: list[str] = []
        in_quote = False
        outer_end: int | None = None
        i = brace
        while i < len(block_body):
            if block_body.startswith(QUOTE, i):
                in_quote = not in_quote
                i += len(QUOTE)
                continue
            if in_quote:
                i += 1
                continue

            char = block_body[i]
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                expected = "{" if char == "}" else "["
                if not stack or stack[-1] != expected:
                    raise ValueError(f"Mismatched delimiter {char!r} at position {i}")
                stack.pop()
                if not stack:
                    outer_end = i
                    break
            i += 1

        if in_quote:
            raise ValueError("Tool call has an unterminated quote token")
        if outer_end is None or stack:
            raise ValueError("Tool call has an unterminated argument object")
        if block_body[outer_end + 1 :].strip():
            raise ValueError("Tool call has trailing text after its argument object")

        arguments, consumed = self._parse_object(block_body, brace)
        if consumed != outer_end + 1:
            raise ValueError(
                "Tool call argument parser did not consume the complete object"
            )
        return name, arguments

    @staticmethod
    def _find_block_boundary(
        text: str, body_start: int, *, final: bool = False
    ) -> tuple[int, bool] | None:
        """Find the next structural frame boundary.

        The boolean is true for a complete END boundary and false for a nested
        START that truncates the current frame. Markers inside a paired quote
        are always literal. An unpaired quote blocks ordinary streaming
        classification; only final scanning may recover structural markers
        after it.
        """
        i = body_start
        while i < len(text):
            if text.startswith(QUOTE, i):
                quote_end = text.find(QUOTE, i + len(QUOTE))
                if quote_end != -1:
                    i = quote_end + len(QUOTE)
                    continue
                if not final:
                    return None
                i += len(QUOTE)
                continue
            if text.startswith(TOOL_CALL_START, i):
                return i, False
            if text.startswith(TOOL_CALL_END, i):
                return i, True
            i += 1
        return None

    @staticmethod
    def _iter_top_level_markers(text: str, start: int = 0):
        """Yield structural START/END markers outside top-level quotes."""
        i = start
        while i < len(text):
            if text.startswith(QUOTE, i):
                quote_end = text.find(QUOTE, i + len(QUOTE))
                if quote_end == -1:
                    return
                i = quote_end + len(QUOTE)
                continue
            if text.startswith(TOOL_CALL_START, i):
                yield i, TOOL_CALL_START
                i += len(TOOL_CALL_START)
                continue
            if text.startswith(TOOL_CALL_END, i):
                yield i, TOOL_CALL_END
                i += len(TOOL_CALL_END)
                continue
            i += 1

    @classmethod
    def _find_top_level_start(cls, text: str, start: int) -> int | None:
        for marker_at, marker in cls._iter_top_level_markers(text, start):
            if marker == TOOL_CALL_START:
                return marker_at
        return None

    @classmethod
    def _clean_top_level_visible(cls, text: str) -> str:
        """Remove unmatched structural ENDs while preserving quoted markers."""
        parts: list[str] = []
        cursor = 0
        for marker_at, marker in cls._iter_top_level_markers(text):
            if marker != TOOL_CALL_END:
                continue
            parts.append(text[cursor:marker_at])
            cursor = marker_at + len(TOOL_CALL_END)
        parts.append(text[cursor:])
        return "".join(parts)

    @classmethod
    def _iter_block_spans(cls, text: str, *, final: bool = False):
        """Yield (start, end_after, body, complete) for START-anchored blocks."""
        idx = 0
        n = len(text)
        while True:
            start = cls._find_top_level_start(text, idx)
            if start is None:
                return
            body_start = start + len(TOOL_CALL_START)
            boundary = cls._find_block_boundary(text, body_start, final=final)
            if boundary is None:
                yield start, n, text[body_start:n], False
                return
            boundary_at, complete = boundary
            boundary_after = (
                boundary_at + len(TOOL_CALL_END) if complete else boundary_at
            )
            yield start, boundary_after, text[body_start:boundary_at], complete
            idx = boundary_after

    @classmethod
    def _iter_blocks(cls, text: str, *, final: bool = False):
        """Yield (block_body, complete) for each ``<|tool_call>`` block."""
        for _, _, body, complete in cls._iter_block_spans(text, final=final):
            yield body, complete

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        top_level_markers = list(self._iter_top_level_markers(model_output))
        has_start = any(marker == TOOL_CALL_START for _, marker in top_level_markers)
        has_end = any(marker == TOOL_CALL_END for _, marker in top_level_markers)
        if not has_start and not has_end:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        if not has_start:
            content = self._clean_top_level_visible(model_output)
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[],
                content=content or None,
            )

        tool_calls: list[ToolCall] = []
        visible_parts: list[str] = []
        cursor = 0
        for start, end_after, body, complete in self._iter_block_spans(
            model_output, final=True
        ):
            visible_parts.append(
                self._clean_top_level_visible(model_output[cursor:start])
            )
            if not complete:
                # A truncated frame is structural output, not assistant content.
                cursor = end_after
                if end_after == len(model_output):
                    break
                continue
            try:
                name, arguments = self._parse_completed_call(body)
            except (IndexError, ValueError) as exc:
                logger.warning("Skipping malformed Gemma4 tool call: %s", exc)
            else:
                tool_call = ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
                tool_calls.append(tool_call)
            cursor = end_after

        visible_parts.append(self._clean_top_level_visible(model_output[cursor:]))
        content = "".join(visible_parts)
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content or None,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def _buffer_delta_text(self, delta_text: str) -> str:
        """Hold back a partial trailing special token until it completes."""
        combined = self.buffered_delta_text + delta_text
        for tag in (TOOL_CALL_START, TOOL_CALL_END, QUOTE):
            if combined.endswith(tag):
                self.buffered_delta_text = ""
                return combined
        for tag in (TOOL_CALL_START, TOOL_CALL_END, QUOTE):
            for i in range(1, len(tag)):
                if combined.endswith(tag[:i]):
                    self.buffered_delta_text = combined[-i:]
                    return combined[:-i]
        self.buffered_delta_text = ""
        return combined

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        buffered_prefix = self.buffered_delta_text
        delta_text = self._buffer_delta_text(delta_text)
        self._last_current_text = current_text
        visible_previous_text = (
            previous_text[: -len(buffered_prefix)]
            if buffered_prefix and previous_text.endswith(buffered_prefix)
            else previous_text
        )
        buffered_suffix = self.buffered_delta_text
        visible_current_text = (
            current_text[: -len(buffered_suffix)]
            if buffered_suffix and current_text.endswith(buffered_suffix)
            else current_text
        )

        try:
            previous_complete_count = sum(
                complete for _, complete in self._iter_blocks(previous_text)
            )
            current_complete_count = sum(
                complete for _, complete in self._iter_blocks(current_text)
            )

            tool_calls: list[DeltaToolCall] = []
            for index in range(previous_complete_count, current_complete_count):
                try:
                    tool_call = self._emit_completed_call(current_text, index)
                except (IndexError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed streaming Gemma4 tool call at index %d: %s",
                        index,
                        exc,
                    )
                    continue
                if tool_call is not None:
                    tool_calls.append(tool_call)
            visible_before = self._visible_content(visible_previous_text)
            visible_after = self._visible_content(visible_current_text)
            visible_delta = (
                visible_after[len(visible_before) :]
                if visible_after.startswith(visible_before)
                else visible_after
            )
            if tool_calls:
                return DeltaMessage(
                    content=visible_delta or None,
                    tool_calls=tool_calls,
                )

            return DeltaMessage(content=visible_delta) if visible_delta else None
        except Exception:
            logger.exception("Error in Gemma4 streaming tool call extraction")
            return None

    @classmethod
    def _visible_content(cls, text: str, *, final: bool = False) -> str:
        parts: list[str] = []
        cursor = 0
        for start, end_after, _, complete in cls._iter_block_spans(text, final=final):
            parts.append(cls._clean_top_level_visible(text[cursor:start]))
            if not complete:
                cursor = end_after
                if end_after == len(text):
                    return "".join(parts)
                continue
            cursor = end_after
        parts.append(cls._clean_top_level_visible(text[cursor:]))
        return "".join(parts)

    def _candidate_public_tool_index(self, raw_index: int) -> int:
        return self._raw_to_public_tool_index.get(
            raw_index, self._next_public_tool_index
        )

    def _record_public_tool_index(self, raw_index: int, public_index: int) -> None:
        if raw_index in self._raw_to_public_tool_index:
            return
        if public_index != self._next_public_tool_index:
            raise ValueError(
                f"Unexpected public tool index {public_index}; "
                f"expected {self._next_public_tool_index}"
            )
        self._raw_to_public_tool_index[raw_index] = public_index
        self._next_public_tool_index += 1

    def get_remaining_unstreamed_args(self) -> str:
        # Partial calls are finalized by finish_streaming(). The generic vLLM
        # fallback serializes an empty parser state as "{}", fabricating
        # arguments for a stream that stopped immediately after the opening brace.
        return ""

    def finish_streaming(self) -> DeltaMessage | None:
        text = self._last_current_text
        buffered_suffix = self.buffered_delta_text
        visible_text = (
            text[: -len(buffered_suffix)]
            if buffered_suffix and text.endswith(buffered_suffix)
            else text
        )
        provisional_visible = self._visible_content(visible_text)
        final_visible = self._visible_content(visible_text, final=True)
        buffered_content = (
            final_visible[len(provisional_visible) :]
            if final_visible.startswith(provisional_visible)
            else final_visible
        )

        blocks = list(self._iter_blocks(visible_text, final=True))
        open_call = bool(blocks and not blocks[-1][1])
        if not open_call:
            buffered_content += buffered_suffix
        self.buffered_delta_text = ""

        tool_calls: list[DeltaToolCall] = []
        final_complete_count = sum(complete for _, complete in blocks)
        for complete_index in range(final_complete_count):
            try:
                tool_call = self._emit_completed_call(
                    visible_text, complete_index, final=True
                )
            except (IndexError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed finalized Gemma4 tool call at index %d: %s",
                    complete_index,
                    exc,
                )
                continue
            if tool_call is not None:
                tool_calls.append(tool_call)

        if open_call:
            body = blocks[-1][0]
            brace = body.find("{", len(CALL_PREFIX))
            if brace == -1:
                if not tool_calls and not buffered_content:
                    return None
            else:
                try:
                    name, arguments = self._parse_call(body)
                except (IndexError, ValueError):
                    logger.exception("Malformed unfinished Gemma4 tool call")
                else:
                    empty_object_complete = body[brace + 1 :].strip() == "}"
                    tool_index = len(blocks) - 1
                    if name and tool_index not in self._raw_to_public_tool_index:
                        arguments_json = (
                            json.dumps(arguments, ensure_ascii=False)
                            if arguments or empty_object_complete
                            else None
                        )
                        public_index = self._candidate_public_tool_index(tool_index)
                        tool_call = DeltaToolCall(
                            index=public_index,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=name,
                                arguments=arguments_json,
                            ).model_dump(exclude_none=True),
                        )
                        self._record_public_tool_index(tool_index, public_index)
                        self.streamed_args_for_tool.append(arguments_json or "")
                        self.prev_tool_call_arr.append(
                            {
                                "name": name,
                                "arguments": arguments,
                            }
                        )
                        tool_calls.append(tool_call)

        if not tool_calls and not buffered_content:
            return None
        if not tool_calls:
            return DeltaMessage(content=buffered_content)
        return DeltaMessage(
            content=buffered_content or None,
            tool_calls=tool_calls,
        )

    def _emit_completed_call(
        self,
        current_text: str,
        complete_index: int,
        *,
        final: bool = False,
    ) -> DeltaToolCall | None:
        blocks = [
            (raw_index, body)
            for raw_index, (body, complete) in enumerate(
                self._iter_blocks(current_text, final=final)
            )
            if complete
        ]
        if not (0 <= complete_index < len(blocks)):
            return None
        raw_index, body = blocks[complete_index]
        if raw_index in self._raw_to_public_tool_index:
            return None  # already emitted
        name, arguments = self._parse_completed_call(body)
        args_json = json.dumps(arguments, ensure_ascii=False)

        public_index = self._candidate_public_tool_index(raw_index)
        function = DeltaFunctionCall(name=name, arguments=args_json)
        tool_call = DeltaToolCall(
            index=public_index,
            type="function",
            id=make_tool_call_id(),
            function=function.model_dump(exclude_none=True),
        )

        self._record_public_tool_index(raw_index, public_index)
        self.streamed_args_for_tool.append(args_json)
        self.prev_tool_call_arr.append(
            {
                "name": name,
                "arguments": arguments,
            }
        )
        return tool_call
