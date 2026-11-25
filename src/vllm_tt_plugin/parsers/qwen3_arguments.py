# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Splits Qwen3 XML tool-call arguments only at the chat template's tags.

The Qwen3 chat template writes a tool call as::

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

vLLM's Qwen3 parser engine (``vllm.parser.qwen3``, used by the ``qwen3_coder``
and ``qwen3_xml`` tool parsers) splits that text with three defects in vLLM
0.33.0:

- ``_qwen3_arg_converter`` drops a last parameter that has no
  ``</parameter>`` from the final arguments (vLLM issue 57699). This happens
  when the model leaves out that tag or generation stops inside the value.
  Non-streamed arguments lose the parameter; streamed arguments end as
  unterminated JSON.
- ``_PARAM_RE`` ends a value at its first ``</parameter>``, so a value that
  contains that text (XML, HTML, or source code) is cut there.
- The ``(TOOL_ARGS, FUNC_END)`` transition ends the call at the first
  ``</function>``, also one inside a value, and the rest of the value is lost.

``install_patches`` replaces ``_qwen3_arg_converter`` with
``convert_arguments`` and ``_qwen3_arg_value_extender`` with
``extend_open_value``, then clears the ``qwen3_config`` cache, which stores
both. ``StreamingParserEngine`` then keeps a ``</function>`` in the arguments
until a call boundary follows it. Every configuration that ``qwen3_config``
builds afterwards uses the replacements: ``qwen3_coder``, ``qwen3_xml``, and
the other parsers built on ``qwen3_config``. For arguments that the template
wrote without such text, the arguments are the same as upstream's.

TODO: remove once the pinned vLLM fixes these defects.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field

import regex
from vllm.parser import qwen3
from vllm.parser.engine.events import SemanticEvent
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
)
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine

__all__ = ("convert_arguments", "extend_open_value", "install_patches")

_OPEN_RE = regex.compile(r"<\s*parameter\s*=\s*([^>]*)>")
# A "<parameter=" at a line start ends a value that has no real "</parameter>".
_LINE_OPEN_RE = regex.compile(r"(?<=\n)<\s*parameter\s*=")
_CLOSE_RE = regex.compile(r"<\s*/\s*parameter\s*>")
# Text after a real "</parameter>": the next parameter, or the end of the call
# followed by the start or end of a tool-call block.
_NEXT_RE = regex.compile(
    r"\s*(?:<\s*parameter\s*=|<\s*/\s*function\s*>\s*"
    r"(?:<\s*/?\s*tool_call\s*>|<\s*function\s*=))"
)
# ... or only call closers up to the end of the argument text.
_END_RE = regex.compile(r"\s*(?:<\s*/\s*(?:function|tool_call)\s*>\s*)*\Z")
# Tags whose arrival can change how ``_scan`` splits an open value.
_TAG_RE = regex.compile(
    r"<\s*/\s*parameter\s*>|<\s*parameter\s*=|<\s*/\s*(?:function|tool_call)\s*>"
)

# Instance attribute of a ``StreamingParserEngine``: True while the last
# non-whitespace argument text of the open call is a ``</function>``.
_FUNCTION_END_PENDING = "_tt_function_end_pending"
# Terminals that end the call when they follow a pending ``</function>``.
_CALL_BOUNDARIES = frozenset(("TOOL_END", "TOOL_START", "FUNC_PREFIX"))


def install_patches() -> None:
    """Replaces the Qwen3 argument functions and patches ``StreamingParserEngine``.

    Each patch stores the replaced functions in ``_tt_original_*`` attributes
    of the patched module or class; a second call finds them and changes
    nothing.

    Raises
    ------
    RuntimeError
        If upstream ``_qwen3_arg_converter`` or ``_qwen3_arg_value_extender``
        no longer takes the parameters that the parser engine passes to the
        replacements.
    """
    _install_converter_patch()
    _install_function_end_patch()


def convert_arguments(text: str, partial: bool) -> str:
    """Converts the argument text of one Qwen3 tool call to a JSON object.

    Parameters
    ----------
    text
        Text between ``<function=NAME>`` and the end of the call, as the
        parser engine collected it. It may end with ``</function>`` and
        ``</tool_call>`` tags, which are not part of any value.
    partial
        True while the call is still streaming. A ``</parameter>`` whose
        following text has not arrived then stays out of the value, and the
        value stays open.

    Returns
    -------
    str
        JSON object mapping each parameter name to its value text, in the
        order of first appearance. One newline at each end of a value is the
        template's markup and is removed. The last parameter is included also
        when its value has no ``</parameter>``.

    Examples
    --------
    >>> from vllm_tt_plugin.parsers.qwen3_arguments import convert_arguments
    >>> convert_arguments("<parameter=a>\\nx</parameter>y\\n</parameter>\\n", False)
    '{"a": "x</parameter>y"}'
    >>> convert_arguments("<parameter=a>\\nhello\\n</function>\\n", False)
    '{"a": "hello"}'
    """
    scan = _scan(text, partial)
    params: dict[str, str] = {}
    for name, value in scan.closed:
        params[name] = qwen3._trim_wrapping_newlines(value)
    if scan.open_name is not None:
        value = _strip_closers(text[scan.open_start : scan.open_end])
        params[scan.open_name] = qwen3._trim_wrapping_newlines(value)
    return json.dumps(params, ensure_ascii=False)


def extend_open_value(text: str) -> ArgValueExtender | None:
    """Returns a function that streams new text of the open last value.

    The parser engine calls this after each streamed conversion, and while
    the returned function gives text, it skips ``convert_arguments`` for the
    following deltas. The function follows the same rules as upstream
    ``_param_value_extender``, but it gives up on every tag in ``_TAG_RE``, so
    the engine converts again whenever ``_scan`` could split the value
    differently.

    Parameters
    ----------
    text
        Argument text converted so far.

    Returns
    -------
    ArgValueExtender or None
        None when ``text`` does not end inside the value of its last
        parameter, that parameter name appeared before, or the value ends
        with a call closer. Otherwise a function that maps
        each raw delta to the JSON-escaped text that ``convert_arguments``
        would append to that value, or to None when the delta may change
        anything else. It holds back a trailing newline, which
        ``convert_arguments`` removes from a value end.
    """
    scan = _scan(text, partial=True)
    name = scan.open_name
    if name is None or scan.open_end != len(text):
        return None
    # A repeated key keeps its first position, so its value is not last.
    if any(closed == name for closed, _ in scan.closed):
        return None
    value = text[scan.open_start :]
    if _strip_closer(value) is not None:
        return None

    window = _tag_window(value)
    started = bool(value)
    held = "\n" if value.endswith("\n") and value != "\n" else ""

    def extend(delta: str) -> str | None:
        nonlocal window, started, held
        tail = window + delta
        if _TAG_RE.search(tail):
            return None
        window = _tag_window(tail)
        chunk = held + delta
        if not started and chunk:
            started = True
            chunk = chunk.removeprefix("\n")
        held = "\n" if chunk.endswith("\n") else ""
        chunk = chunk[: len(chunk) - len(held)]
        return json.dumps(chunk, ensure_ascii=False)[1:-1]

    return extend


@dataclass(slots=True)
class _Scan:
    """Parameters that ``_scan`` found in Qwen3 argument text.

    Attributes
    ----------
    closed
        ``(name, value)`` per parameter whose value ended, in text order.
        Values keep the template's wrapping newlines.
    open_name
        Name of the parameter whose value is still open at the end of the
        text, or None.
    open_start, open_end
        Slice of the text that holds the open value. ``open_end`` is less
        than the text length while a ``</parameter>`` at ``open_end`` waits
        for the text that decides whether it ends the value.
    """

    closed: list[tuple[str, str]] = field(default_factory=list)
    open_name: str | None = None
    open_start: int = 0
    open_end: int = 0


def _scan(text: str, partial: bool) -> _Scan:
    """Splits Qwen3 argument text into parameter values.

    A ``</parameter>`` ends a value only when ``_NEXT_RE`` matches the text
    after it, or ``_END_RE`` does (whitespace and call closers up to the end).
    Any other ``</parameter>`` is part of the value. With ``partial``, a
    ``</parameter>`` that the text after it can still make real is held
    back: the value stops before it and stays open.

    A value with no real ``</parameter>`` ends at the next ``<parameter=``
    that starts a line (the model left out the closer). A ``<parameter=``
    inside a line stays in the value. A value that does not end runs to the
    end of the text.

    Each decision depends only on text up to a bounded distance after a tag,
    so a value streamed by its prefix never has to be taken back.
    """
    scan = _Scan()
    opener = _OPEN_RE.search(text)
    while opener is not None:
        name, start = opener.group(1), opener.end()
        implicit = _LINE_OPEN_RE.search(text, start)
        limit = len(text) if implicit is None else implicit.start()
        following = None if implicit is None else _OPEN_RE.search(text, limit)
        end = None
        for closer in _CLOSE_RE.finditer(text, start, limit):
            after = closer.end()
            next_tag = _NEXT_RE.match(text, after, partial=partial)
            if next_tag is not None and not next_tag.partial:
                end, following = closer.start(), _OPEN_RE.search(text, after)
                break
            at_end = _END_RE.match(text, after, partial=partial) is not None
            if partial and (next_tag is not None or at_end):
                if name:
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
                    scan.open_start, scan.open_end = start, len(text)
                return scan
            end = limit
        scan.closed.append((name, text[start:end]))
        opener = following
    return scan


def _strip_closer(value: str) -> str | None:
    """Removes one trailing ``</function>`` or ``</tool_call>`` from a value.

    Returns the text before that tag, or None when the value does not end
    with one. Whitespace after the tag goes with it; whitespace before it
    stays, so the template's newline is removed exactly once, by the
    newline trim. Runs in linear time on long whitespace runs.
    """
    stripped = value.rstrip()
    if not stripped.endswith(">"):
        return None
    head = stripped[:-1].rstrip()
    for name in ("function", "tool_call"):
        if head.endswith(name):
            before = head[: -len(name)].rstrip()
            if before.endswith("/"):
                before = before[:-1].rstrip()
                if before.endswith("<"):
                    return before[:-1]
    return None


def _strip_closers(value: str) -> str:
    """Removes every trailing ``</function>`` and ``</tool_call>`` from a value."""
    while (stripped := _strip_closer(value)) is not None:
        value = stripped
    return value


def _tag_window(text: str) -> str:
    """Returns the end of ``text`` that may still grow into a ``_TAG_RE`` tag."""
    start = text.rfind("<")
    if start == -1:
        return ""
    tail = text[start:]
    return tail if _TAG_RE.fullmatch(tail, partial=True) else ""


def _install_converter_patch() -> None:
    """Makes ``qwen3_config`` build configurations with this module's functions."""
    if hasattr(qwen3, "_tt_original_qwen3_arg_converter"):
        return
    # The engine catches TypeError from a converter and drops the delta, so a
    # signature change upstream would fail silently per request.
    for name, expected in (
        ("_qwen3_arg_converter", ["raw_args", "partial"]),
        ("_qwen3_arg_value_extender", ["raw_args"]),
    ):
        found = list(inspect.signature(getattr(qwen3, name)).parameters)
        if found != expected:
            raise RuntimeError(
                f"vllm.parser.qwen3.{name} takes {found}, expected {expected}; "
                "update vllm_tt_plugin.parsers.qwen3_arguments."
            )
    qwen3._tt_original_qwen3_arg_converter = qwen3._qwen3_arg_converter
    qwen3._tt_original_qwen3_arg_value_extender = qwen3._qwen3_arg_value_extender
    qwen3._qwen3_arg_converter = convert_arguments
    qwen3._qwen3_arg_value_extender = extend_open_value
    # Cached configurations keep the functions they were built with.
    qwen3.qwen3_config.cache_clear()


def _install_function_end_patch() -> None:
    """Ends a call at ``</function>`` only when a call boundary follows it.

    In ``TOOL_ARGS``, a ``</function>`` goes into the argument text and is
    marked pending. When the next terminal is ``</tool_call>``,
    ``<tool_call>``, or ``<function=`` with only whitespace since that
    ``</function>``, the engine first applies the configuration's
    ``(TOOL_ARGS, FUNC_END)`` transition, then handles the terminal. Other
    text clears the mark, and the ``</function>`` stays in the value. At the
    end of the output, the engine's ``finish`` ends the open call and
    ``convert_arguments`` drops the trailing tags.

    Applies only to an engine whose configuration converts arguments with
    ``convert_arguments`` and declares that transition; other engines run the
    original methods unchanged.
    """
    if hasattr(StreamingParserEngine, "_tt_original_on_terminal"):
        return
    on_terminal = StreamingParserEngine._on_terminal
    emit_for_state = StreamingParserEngine._emit_for_state
    StreamingParserEngine._tt_original_on_terminal = on_terminal
    StreamingParserEngine._tt_original_emit_for_state = emit_for_state
    function_end = (ParserState.TOOL_ARGS, "FUNC_END")

    def on_terminal_tt(
        self: StreamingParserEngine, terminal: str, value: str, token_count: int = 0
    ) -> list[SemanticEvent]:
        if self.state is not ParserState.TOOL_ARGS:
            # Arguments of the next call start with nothing pending.
            self.__dict__.pop(_FUNCTION_END_PENDING, None)
            return on_terminal(self, terminal, value, token_count)
        if self.skip_tool_parsing or not _keeps_function_end(self.config):
            return on_terminal(self, terminal, value, token_count)
        if terminal == "FUNC_END":
            self.__dict__[_FUNCTION_END_PENDING] = True
            return emit_for_state(self, value, token_count)
        if (
            self.__dict__.pop(_FUNCTION_END_PENDING, False)
            and terminal in _CALL_BOUNDARIES
        ):
            transition = self.config.transitions[function_end]
            events = self._apply_transition(transition, "", 0)
            events.extend(on_terminal(self, terminal, value, token_count))
            return events
        return on_terminal(self, terminal, value, token_count)

    def emit_for_state_tt(
        self: StreamingParserEngine, text: str, token_count: int = 0
    ) -> list[SemanticEvent]:
        if self.state is ParserState.TOOL_ARGS and text and not text.isspace():
            self.__dict__.pop(_FUNCTION_END_PENDING, None)
        return emit_for_state(self, text, token_count)

    StreamingParserEngine._on_terminal = on_terminal_tt
    StreamingParserEngine._emit_for_state = emit_for_state_tt


def _keeps_function_end(config: ParserEngineConfig) -> bool:
    """Tells whether ``_install_function_end_patch`` applies to ``config``."""
    return (
        config.arg_converter is convert_arguments
        and config.terminal_literal("FUNC_END") == qwen3.FUNC_END
        and (ParserState.TOOL_ARGS, "FUNC_END") in config.transitions
    )
