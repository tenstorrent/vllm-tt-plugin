# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Typing of tool-call arguments whose schema property declares no type.

The DSML converter of the DeepSeek parsers already decodes a parameter written with
``string="false"`` (or without the attribute) as JSON, so ``<parameter name="n"
string="false">42</parameter>`` becomes the integer 42. ``ParserEngine._coerce_value``
then applies the tool schema; for a property WITHOUT ``type`` (no ``type`` / ``enum`` /
``anyOf`` / ``oneOf`` / ``allOf``) ``extract_types_from_schema`` answers ``["string"]``
and the generic scalar branch re-serialises the decoded value and "coerces" it to that
default string, so 42, true and null reach the client as "42", "true" and "null".

An untyped property has nothing to coerce to: a decoded scalar is kept as the parser
decoded it. A value written with ``string="true"`` is still a string (it never reaches
the scalar branch), and typed schemas go through the unchanged vLLM code.
Disable with ``VLLM_TT_UNTYPED_SCALARS=0``.
"""

from __future__ import annotations

import os
from typing import Any

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

_PATCHED_ATTR = "_tt_untyped_scalars_patched"
_TYPE_KEYS = ("type", "enum", "anyOf", "oneOf", "allOf")


def enabled() -> bool:
    return os.environ.get("VLLM_TT_UNTYPED_SCALARS", "1").strip().lower() not in ("0", "false", "off", "no")


def is_untyped(schema: Any) -> bool:
    return isinstance(schema, dict) and not any(k in schema for k in _TYPE_KEYS)


def wrap_coerce_value(original):
    """Wrap ``ParserEngine._coerce_value`` (a staticmethod ``(value, schema) -> (value, changed)``)."""
    if getattr(original, _PATCHED_ATTR, False):
        return original

    def _coerce_value(value: object, schema: dict) -> tuple[object, bool]:
        if (
            not isinstance(value, (str, dict, list))
            and is_untyped(schema)
            and enabled()
        ):
            return value, False
        return original(value, schema)

    setattr(_coerce_value, _PATCHED_ATTR, True)
    return _coerce_value


def install() -> None:
    try:
        from vllm.parser.engine.parser_engine import ParserEngine

        current = ParserEngine.__dict__["_coerce_value"].__func__
        ParserEngine._coerce_value = staticmethod(wrap_coerce_value(current))
    except Exception as exc:  # pragma: no cover - a vLLM layout without the parser engine
        logger.debug("untyped scalar typing not installed: %s", exc)
