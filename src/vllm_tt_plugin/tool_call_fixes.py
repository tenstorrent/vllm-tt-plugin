# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Frontend monkeypatches for Qwen tool-call argument errors.

Each ``_install_*`` function fixes one upstream vLLM defect and follows the
conventions of the ``_install_*`` patches in ``platform.py``: it is idempotent
through a ``_tt_original_*`` marker on the patched owner, and it raises when its
target is missing, so a vLLM upgrade that moves the target fails at startup.

- P1: ``$ref``/``allOf`` in tool parameter schemas hide the property types from
  argument coercion (vLLM #46924).
- P2: ``finish_reason`` is "tool_calls" even when generation stopped on
  "length" (vLLM #46303, #59324).
- P3: the final conversion drops a last ``<parameter>`` that has no
  ``</parameter>`` (vLLM #57699).
- P4: a ``["string", ...]`` union turns "007" into 7 (vLLM #62252).
- P5: streamed argument conversion is O(n) per delta, so a long argument costs
  O(n^2) per stream (incremental ``_compute_arg_delta``).
- P6: ``count_reasoning_tokens`` re-parses the whole output per delta when
  thinking is disabled (the count is always 0 there).
- P7: an argument value that contains a literal ``</parameter>`` or
  ``</function>`` (XML, HTML, code) is cut at that string. A closer ends a
  parameter or a call only where the template's next tag follows it (see
  ``_scan_params`` and ``_install_function_end_lookahead_patch``).

TODO: remove each patch once the matching upstream fix is in the pinned vLLM.
"""

import copy
import functools
import inspect
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

_MAX_DEREF_DEPTH = 32
_SCHEMA_CACHE_SIZE = 64


def install_tool_call_fixes() -> None:
    """Install the tool-call patches. Safe to call more than once."""
    _install_tool_schema_deref_patch()
    _install_tool_finish_reason_patch()
    _install_unclosed_parameter_patch()
    _install_string_union_patch()
    _install_arg_delta_patch()
    _install_reasoning_count_patch()
    _install_function_end_lookahead_patch()
    logger.info(
        "tool-call fixes active: P1 P2 P3 P4 P5(arg delta fast path) "
        "P6(reasoning count short-circuit) P7(literal closers in values)"
    )


# ---------------------------------------------------------------------------
# P1: dereference $ref / allOf before coercion looks up property types.
# ---------------------------------------------------------------------------


def _resolve_pointer(root: Any, ref: str) -> Any:
    node = root
    for part in ref[1:].split("/"):
        if part == "":
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _merge_schemas(base: dict, other: dict) -> dict:
    """Merge `other` into a copy of `base`: properties per key, required union."""
    merged = dict(base)
    for key, value in other.items():
        if key not in merged:
            merged[key] = value
        elif key == "properties" and isinstance(value, dict):
            props = dict(merged[key]) if isinstance(merged[key], dict) else {}
            for prop_name, prop_schema in value.items():
                existing = props.get(prop_name)
                if isinstance(existing, dict) and isinstance(prop_schema, dict):
                    props[prop_name] = _merge_schemas(existing, prop_schema)
                else:
                    props.setdefault(prop_name, prop_schema)
            merged[key] = props
        elif key == "required" and isinstance(value, list):
            current = merged[key] if isinstance(merged[key], list) else []
            merged[key] = current + [r for r in value if r not in current]
        # Any other key (including "type") keeps the first value.
    return merged


def _deref_schema(node: Any, root: Any, stack: tuple[str, ...], depth: int = 0) -> Any:
    """Return a copy of `node` with local $ref and allOf resolved."""
    if not isinstance(node, dict) or depth > _MAX_DEREF_DEPTH:
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        rest = {k: v for k, v in node.items() if k != "$ref"}
        target = _resolve_pointer(root, ref) if ref.startswith("#") else None
        if ref in stack or not isinstance(target, dict):
            return _deref_schema(rest, root, stack, depth + 1)
        return _deref_schema({**target, **rest}, root, stack + (ref,), depth + 1)

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "allOf":
            continue
        if key in ("properties", "patternProperties") and isinstance(value, dict):
            out[key] = {
                name: _deref_schema(sub, root, stack, depth + 1)
                for name, sub in value.items()
            }
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            out[key] = _deref_schema(value, root, stack, depth + 1)
        elif key in ("items", "prefixItems", "anyOf", "oneOf") and isinstance(
            value, list
        ):
            out[key] = [_deref_schema(sub, root, stack, depth + 1) for sub in value]
        else:
            out[key] = copy.deepcopy(value)
    members = node.get("allOf")
    if isinstance(members, list):
        for member in members:
            if isinstance(member, dict):
                out = _merge_schemas(out, _deref_schema(member, root, stack, depth + 1))
    return out


def _has_ref_or_all_of(node: Any) -> bool:
    if isinstance(node, dict):
        return (
            "$ref" in node
            or "allOf" in node
            or any(map(_has_ref_or_all_of, node.values()))
        )
    if isinstance(node, list):
        return any(map(_has_ref_or_all_of, node))
    return False


def _install_tool_schema_deref_patch() -> None:
    """Resolve ``$ref``/``allOf`` in the properties the parser engine coerces by.

    ``parser_engine`` binds ``find_tool_properties`` by name at import, so the
    module attribute is the target.
    """
    from vllm.parser.engine import parser_engine
    from vllm.tool_parsers import utils as tool_utils

    if hasattr(parser_engine, "_tt_original_find_tool_properties"):
        return
    original = parser_engine.find_tool_properties
    parser_engine._tt_original_find_tool_properties = original
    _is_function_tool = tool_utils._is_function_tool
    _extract_tool_info = tool_utils._extract_tool_info
    iter_info = tool_utils.iter_response_function_tool_info
    response_tool_types = (tool_utils.FunctionTool, tool_utils.NamespaceTool)
    # id(parameters) -> (parameters, dereferenced properties or None).
    cache: dict[int, tuple[Any, dict | None]] = {}

    def find_parameters(tools: Any, tool_name: str) -> Any:
        for tool in tools or []:
            if isinstance(tool, response_tool_types):
                for name, params in iter_info(tool):
                    if name == tool_name:
                        return params
                continue
            if not _is_function_tool(tool):
                continue
            name, params = _extract_tool_info(tool)
            if name == tool_name:
                return params
        return None

    def find_tool_properties(tools: Any, tool_name: str) -> dict[str, Any]:
        # The schema comes from the client. A malformed one must not fail the
        # request, so dereference errors fall back to the upstream lookup.
        try:
            params = find_parameters(tools, tool_name)
            if isinstance(params, dict):
                hit = cache.get(id(params))
                if hit is None or hit[0] is not params:
                    props = None
                    if _has_ref_or_all_of(params):
                        resolved = _deref_schema(params, params, ())
                        props = resolved.get("properties")
                        props = props if isinstance(props, dict) else {}
                    if len(cache) >= _SCHEMA_CACHE_SIZE:
                        cache.clear()
                    hit = cache[id(params)] = (params, props)
                if hit[1] is not None:
                    return hit[1]
        except Exception as exc:
            logger.debug("Schema dereference failed for %s: %s", tool_name, exc)
        return original(tools, tool_name)

    parser_engine.find_tool_properties = find_tool_properties


# ---------------------------------------------------------------------------
# P2: "tool_calls" only when the engine stopped on "stop".
# ---------------------------------------------------------------------------


async def _aclose(generator: Any) -> None:
    aclose = getattr(generator, "aclose", None)
    if aclose is not None:
        await aclose()


def _tee_finish_reasons(
    result_generator: AsyncIterator[Any], rec: dict[int, str]
) -> AsyncGenerator[Any, None]:
    """Yield the engine outputs unchanged and record each finish_reason.

    Closing the tee closes the engine generator at once. Leaving an async
    ``for`` early does not close the iterated generator by itself.
    """

    async def tee() -> AsyncGenerator[Any, None]:
        try:
            async for res in result_generator:
                for output in res.outputs:
                    if output.finish_reason is not None:
                        rec[output.index] = output.finish_reason
                yield res
        finally:
            await _aclose(result_generator)

    return tee()


def _truthful_reason(rec: dict[int, str], index: Any) -> str | None:
    reason = rec.get(index)
    return reason if reason is not None and reason != "stop" else None


def _fix_sse_chunk(chunk: str, rec: dict[int, str]) -> str:
    if not chunk.startswith("data: ") or '"finish_reason":"tool_calls"' not in chunk:
        return chunk
    obj = json.loads(chunk[len("data: ") :])
    changed = False
    for choice in obj.get("choices", ()):
        if choice.get("finish_reason") != "tool_calls":
            continue
        reason = _truthful_reason(rec, choice.get("index"))
        if reason is not None:
            choice["finish_reason"] = reason
            changed = True
    if not changed:
        return chunk
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return f"data: {payload}\n\n"


def _install_tool_finish_reason_patch() -> None:
    """Report "length" instead of "tool_calls" for a truncated tool call.

    Ports the rule of vLLM #46303/#59324 as a class-attribute patch of the two
    chat generators, which decide ``finish_reason`` inline.
    """
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    if hasattr(OpenAIServingChat, "_tt_original_chat_completion_stream_generator"):
        return
    orig_stream = OpenAIServingChat.chat_completion_stream_generator
    orig_full = OpenAIServingChat.chat_completion_full_generator
    for orig in (orig_stream, orig_full):
        if "result_generator" not in inspect.signature(orig).parameters:
            raise RuntimeError(f"{orig.__name__} has no result_generator parameter")
    OpenAIServingChat._tt_original_chat_completion_stream_generator = orig_stream
    OpenAIServingChat._tt_original_chat_completion_full_generator = orig_full

    @functools.wraps(orig_stream)
    async def stream_generator(self, request, result_generator, *args, **kwargs):
        rec: dict[int, str] = {}
        tee = _tee_finish_reasons(result_generator, rec)
        inner = orig_stream(self, request, tee, *args, **kwargs)
        try:
            async for chunk in inner:
                yield _fix_sse_chunk(chunk, rec)
        finally:
            # A client disconnect closes this generator at the yield above.
            await _aclose(inner)
            await _aclose(tee)

    @functools.wraps(orig_full)
    async def full_generator(self, request, result_generator, *args, **kwargs):
        rec: dict[int, str] = {}
        tee = _tee_finish_reasons(result_generator, rec)
        response = await orig_full(self, request, tee, *args, **kwargs)
        # An ErrorResponse has no choices.
        for choice in getattr(response, "choices", ()):
            if choice.finish_reason == "tool_calls":
                reason = _truthful_reason(rec, choice.index)
                if reason is not None:
                    choice.finish_reason = reason
        return response

    OpenAIServingChat.chat_completion_stream_generator = stream_generator
    OpenAIServingChat.chat_completion_full_generator = full_generator


# ---------------------------------------------------------------------------
# P3: keep a last <parameter> that has no </parameter>.
# ---------------------------------------------------------------------------


def _strip_one_closer(value: str) -> str | None:
    """Drop one trailing ``</function>`` or ``</tool_call>`` and its whitespace.

    Same result as removing a match of ``\\s*</\\s*(?:function|tool_call)\\s*>\\s*$``
    but linear: that regex is quadratic on a long whitespace run. Returns None
    when the value has no trailing closer. No whitespace may sit between "<"
    and "/".
    """
    stripped = value.rstrip()
    if not stripped.endswith(">"):
        return None
    head = stripped[:-1].rstrip()
    for name in ("function", "tool_call"):
        if head.endswith(name):
            before = head[: -len(name)].rstrip()
            if before.endswith("/") and before[:-1].endswith("<"):
                return before[:-2].rstrip()
    return None


class _ParamScan:
    """Result of ``_scan_params``.

    ``closed``: (name, value) per parameter that ended, values untrimmed.
    ``open_name`` / ``open_start`` / ``open_end``: the parameter still open at
    the end of the text (``open_end`` < len(text) when a possible closer at
    ``open_end`` is held back), or ``open_name`` None.
    """

    __slots__ = ("closed", "open_name", "open_start", "open_end")


@functools.cache
def _scan_patterns():
    import regex  # vLLM's qwen3 parser uses it too; partial matching is needed

    return (
        regex.compile(r"<\s*parameter\s*=\s*([^>]*)>"),
        regex.compile(r"(?<=\n)<\s*parameter\s*="),  # a dropped closer
        regex.compile(r"<\s*/\s*parameter\s*>"),
        # What follows a real </parameter>: the next parameter, or the end of
        # the call and the start of the next call.
        regex.compile(
            r"\s*(?:<\s*parameter\s*=|<\s*/\s*function\s*>\s*"
            r"(?:<\s*/?\s*tool_call\s*>|<\s*function\s*=))"
        ),
        # ... or only closers up to the end of the call's text.
        regex.compile(r"\s*(?:<\s*/\s*(?:function|tool_call)\s*>\s*)*\Z"),
    )


def _scan_params(raw: str, partial: bool) -> _ParamScan:
    """Split Qwen3 XML argument text into parameters.

    The template writes ``<parameter=NAME>`` newline, VALUE, newline,
    ``</parameter>`` newline per parameter and closes the call with
    ``</function>`` and ``</tool_call>``. A ``</parameter>`` ends the value
    only when the next parameter or the end of the call follows it (whitespace
    allowed); any other ``</parameter>`` is part of the value. With
    ``partial`` (streaming), a ``</parameter>`` whose follower has not arrived
    yet is held back: the value stops before it and stays open.

    A value with no real closer ends at a ``<parameter=`` that starts a line
    (the upstream regex ends it at any ``<parameter=``: a dropped closer), else
    it runs to the end of the text (an unclosed last parameter). A
    ``<parameter=`` inside a line stays in the value.

    Every decision depends only on the text up to a bounded lookahead past
    the tag, so a value streamed by its prefix never has to be taken back.
    """
    open_re, line_open_re, close_re, next_re, end_re = _scan_patterns()
    scan = _ParamScan()
    scan.closed = []
    scan.open_name = None
    scan.open_start = scan.open_end = 0
    n = len(raw)
    opener = open_re.search(raw)
    while opener is not None:
        name, start = opener.group(1), opener.end()
        # A line-start "<parameter=" ends the value even before its name and
        # ">" arrive, as upstream: the decision never waits for later text.
        implicit = line_open_re.search(raw, start)
        limit = n if implicit is None else implicit.start()
        following = None if implicit is None else open_re.search(raw, limit)
        end = None
        for closer in close_re.finditer(raw, start, limit):
            after = closer.end()
            nxt = next_re.match(raw, after, partial=partial)
            if nxt is not None and not nxt.partial:
                end, following = closer.start(), open_re.search(raw, after)
                break
            at_end = end_re.match(raw, after, partial=partial) is not None
            if partial and (nxt is not None or at_end):
                if name:  # held back: the follower decides
                    scan.open_name = name
                    scan.open_start, scan.open_end = start, closer.start()
                return scan
            if at_end:
                end, following = closer.start(), None
                break
        if end is None:
            if implicit is None:
                if name:
                    scan.open_name = name
                    scan.open_start, scan.open_end = start, n
                return scan
            end = limit
        scan.closed.append((name, raw[start:end]))
        opener = following
    return scan


def _strip_closers(value: str) -> str:
    while (stripped := _strip_one_closer(value)) is not None:
        value = stripped
    return value


def _install_unclosed_parameter_patch() -> None:
    """Replace the Qwen3 argument converter with one built on ``_scan_params``.

    Keeps an unclosed last parameter in the final argument JSON (P3) and keeps
    literal ``</parameter>`` text inside a value (P7). For well-formed template
    output the JSON is the same as the upstream converter's.

    ``qwen3_config`` is cached and stores the converter it was built with, so
    its cache is cleared after the module attribute is replaced.
    """
    from vllm.parser import qwen3

    if hasattr(qwen3, "_tt_original_qwen3_arg_converter"):
        return
    original = qwen3._qwen3_arg_converter
    qwen3._tt_original_qwen3_arg_converter = original
    trim = qwen3._trim_wrapping_newlines

    def arg_converter(raw_args: str, partial: bool) -> str:
        scan = _scan_params(raw_args, partial)
        params: dict[str, object] = {}
        for name, value in scan.closed:
            params[name] = trim(value)
        if scan.open_name is not None:
            value = raw_args[scan.open_start : scan.open_end]
            params[scan.open_name] = trim(_strip_closers(value))
        return json.dumps(params, ensure_ascii=False)

    qwen3._qwen3_arg_converter = arg_converter
    qwen3.qwen3_config.cache_clear()


# ---------------------------------------------------------------------------
# P5: incremental streamed argument conversion.
# ---------------------------------------------------------------------------


class _ArgAnchor:
    """Per tool slot: where the open string parameter has been emitted to."""

    __slots__ = ("slot", "args_len", "emitted_end", "trim_start", "last_lt")


def _install_arg_delta_patch() -> None:
    """Make ``ParserEngine._compute_arg_delta`` O(delta) for an open string value.

    The original converts the whole argument text on every delta. For an open
    parameter whose type is string, the text streamed so far is
    ``json.dumps(trim(value))[1:-1]`` and JSON escaping is per character, so a
    new delta only appends the escape of the new characters. A trailing "\\n"
    is held back (the converter trims it). Any closer, duplicate key,
    non-string key or mismatch runs the original once and re-anchors.
    Must be installed after P3: it only applies when the engine uses the P3
    converter.
    """
    from vllm.parser import qwen3
    from vllm.parser.engine import parser_engine

    owner = parser_engine.ParserEngine
    if hasattr(owner, "_tt_original_compute_arg_delta"):
        return
    original = owner._compute_arg_delta
    owner._tt_original_compute_arg_delta = original
    regex = qwen3.re  # the module qwen3 uses for its patterns
    close_re = regex.compile(r"<\s*/\s*parameter\s*>|<\s*parameter\s*=")
    pending_re = regex.compile(r"<[\s/aemoprt]*\Z")  # a tag start not yet complete

    def esc(text: str) -> str:
        return json.dumps(text, ensure_ascii=False)[1:-1]

    def try_anchor(self, idx, slot, args, anchors) -> None:
        # The open parameter as the P3/P7 converter sees it: a literal closer
        # earlier in the value is part of it, a held-back closer is not.
        scan = _scan_params(args, True)
        name = scan.open_name
        if name is None or scan.open_end != len(args):
            return
        value_start = scan.open_start
        if value_start == len(args):
            return
        if slot.string_keys is not None and name not in slot.string_keys:
            return
        if name in [closed for closed, _ in scan.closed]:
            return  # duplicate key: the dict position moves
        if _strip_one_closer(args) is not None:
            return
        trim_start = value_start + (1 if args[value_start] == "\n" else 0)
        end = max(trim_start, len(args) - (1 if args.endswith("\n") else 0))
        streamed = slot.streamed_json
        text = esc(args[trim_start:end])
        key_prefix = json.dumps(name, ensure_ascii=False) + ': "'
        if not (
            streamed.endswith(text)
            and streamed[: len(streamed) - len(text)].endswith(key_prefix)
        ):
            return
        anchor = _ArgAnchor()
        anchor.slot = slot
        anchor.args_len = len(args)
        anchor.emitted_end = end
        anchor.trim_start = trim_start
        anchor.last_lt = args.rfind("<", value_start)
        anchors[idx] = anchor

    def compute_arg_delta(self, idx, raw_delta):
        if (
            self._arg_converter is not qwen3._qwen3_arg_converter
            or not hasattr(qwen3, "_tt_original_qwen3_arg_converter")
            or not self._stream_arg_deltas
            or self._arg_structural_chars is not None
        ):
            return original(self, idx, raw_delta)
        slot = self._tool_slots[idx]
        args = slot.args
        anchors = self.__dict__.setdefault("_tt_anchor", {})
        anchor = anchors.get(idx)
        if (
            anchor is not None
            and anchor.slot is slot
            and anchor.args_len == len(args) - len(raw_delta)
        ):
            old_len = anchor.args_len
            last_lt = anchor.last_lt
            start = (
                last_lt
                if last_lt >= 0 and pending_re.match(args, last_lt, old_len)
                else old_len
            )
            if close_re.search(args, start) is None and (
                ">" not in raw_delta or _strip_one_closer(args) is None
            ):
                found = args.rfind("<", old_len)
                if found >= 0:
                    anchor.last_lt = found
                end = len(args) - (1 if args.endswith("\n") else 0)
                end = max(end, anchor.trim_start)
                anchor.args_len = len(args)
                if end <= anchor.emitted_end:
                    return None
                diff = esc(args[anchor.emitted_end : end])
                anchor.emitted_end = end
                slot.streamed_json += diff
                return diff
        result = original(self, idx, raw_delta)
        anchors.pop(idx, None)
        try_anchor(self, idx, slot, args, anchors)
        return result

    owner._compute_arg_delta = compute_arg_delta


# ---------------------------------------------------------------------------
# P6: skip the reasoning-token re-parse when thinking is disabled.
# ---------------------------------------------------------------------------


def _install_reasoning_count_patch() -> None:
    """Return 0 from ``count_reasoning_tokens`` when thinking is disabled.

    Without a valid streaming count the adapter decodes and re-parses every
    generated token on each call. With ``enable_thinking=false`` the engine
    starts in CONTENT state, which has no transition into reasoning, so the
    count is always 0.
    """
    from vllm.parser.engine import adapters

    owner = adapters.ParserEngineReasoningAdapter
    if hasattr(owner, "_tt_original_count_reasoning_tokens"):
        return
    original = owner.count_reasoning_tokens
    owner._tt_original_count_reasoning_tokens = original

    def count_reasoning_tokens(self, token_ids):
        if (
            not self._streaming_count_valid
            and getattr(self._parser_engine, "thinking_enabled", True) is False
        ):
            return 0
        return original(self, token_ids)

    owner.count_reasoning_tokens = count_reasoning_tokens


# ---------------------------------------------------------------------------
# P7: a </function> inside a value does not end the call.
# ---------------------------------------------------------------------------

_CALL_FOLLOWERS = frozenset(("TOOL_END", "TOOL_START", "FUNC_PREFIX"))


def _install_function_end_lookahead_patch() -> None:
    """End a Qwen3 tool call at ``</function>`` only when a call tag follows.

    Upstream, the engine's ``(TOOL_ARGS, FUNC_END)`` transition ends the call
    at the first ``</function>``, also one inside a value; the rest of the
    value is dropped. Here a ``</function>`` in TOOL_ARGS is kept in the
    argument text and remembered. The call ends when the next terminal is
    ``</tool_call>``, ``<tool_call>`` or ``<function=`` with only whitespace
    since that ``</function>``, or when the stream finishes (the engine's
    ``finish`` already ends an open call). Any other text after it makes it
    part of the value. The argument converter drops the trailing closers.

    Applies only to engines whose config uses the patched Qwen3 converter,
    so it must be installed after P3.
    """
    from vllm.parser import qwen3
    from vllm.parser.engine import streaming_parser_engine as spe
    from vllm.parser.engine.events import EventType
    from vllm.parser.engine.parser_engine_config import ParserState, Transition

    owner = spe.StreamingParserEngine
    if hasattr(owner, "_tt_original_on_terminal"):
        return
    orig_on_terminal = owner._on_terminal
    orig_emit = owner._emit_for_state
    owner._tt_original_on_terminal = orig_on_terminal
    owner._tt_original_emit_for_state = orig_emit
    tool_args = ParserState.TOOL_ARGS
    end_call = Transition(ParserState.TOOL_BETWEEN, (EventType.TOOL_CALL_END,))

    def applies(self) -> bool:
        cfg = self.config
        return (
            hasattr(qwen3, "_tt_original_qwen3_arg_converter")
            and cfg.arg_converter is qwen3._qwen3_arg_converter
            and cfg.arg_converter is not qwen3._tt_original_qwen3_arg_converter
            and cfg.terminal_literal("FUNC_END") == qwen3.FUNC_END
            and (tool_args, "FUNC_END") in cfg.transitions
        )

    def on_terminal(self, terminal, value, token_count=0):
        if self.state is not tool_args or self.skip_tool_parsing or not applies(self):
            return orig_on_terminal(self, terminal, value, token_count)
        if terminal == "FUNC_END":
            events = orig_emit(self, value, token_count)  # stays in the args
            self._tt_func_end_pending = True
            return events
        if self.__dict__.get("_tt_func_end_pending") and terminal in _CALL_FOLLOWERS:
            self._tt_func_end_pending = False
            events = self._apply_transition(end_call, "", 0)
            events.extend(orig_on_terminal(self, terminal, value, token_count))
            return events
        self._tt_func_end_pending = False
        return orig_on_terminal(self, terminal, value, token_count)

    def emit_for_state(self, text, token_count=0):
        if self.state is tool_args and text and not text.isspace():
            self._tt_func_end_pending = False
        return orig_emit(self, text, token_count)

    owner._on_terminal = on_terminal
    owner._emit_for_state = emit_for_state


# ---------------------------------------------------------------------------
# P4: keep a string when the schema allows "string" and the conversion is lossy.
# ---------------------------------------------------------------------------


def _install_string_union_patch() -> None:
    """Keep "007" a string for a union that allows "string".

    A scalar conversion stays only when it round-trips to the raw text. Object
    and array results stay as converted.
    """
    from vllm.parser.engine import parser_engine
    from vllm.tool_parsers import utils as tool_utils

    if hasattr(parser_engine, "_tt_original_coerce_to_schema_type"):
        return
    original = parser_engine.coerce_to_schema_type
    parser_engine._tt_original_coerce_to_schema_type = original
    aliases = tool_utils._TYPE_ALIASES

    def coerce_to_schema_type(value: str, schema_type: str | list[str]) -> Any:
        result = original(value, schema_type)
        if not isinstance(value, str) or not (
            result is None or isinstance(result, (bool, int, float))
        ):
            return result
        types = [schema_type] if isinstance(schema_type, str) else schema_type
        names = {aliases.get(t.strip().lower(), t.strip().lower()) for t in types}
        if "string" in names and json.dumps(result) != value.strip():
            return value
        return result

    parser_engine.coerce_to_schema_type = coerce_to_schema_type
