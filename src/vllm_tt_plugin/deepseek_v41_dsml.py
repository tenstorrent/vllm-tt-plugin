# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""V4.1 DSML syntax, independent of vLLM transport and sampling."""

import json
import re

START = "<｜DSML｜ calls>"
END = "</｜DSML｜ calls>"
EOS = "<｜end▁of▁sentence｜>"
INVOKE = re.compile(
    r'<｜DSML｜ invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜ invoke>', re.DOTALL
)
PARAM = re.compile(
    r'<｜DSML｜ parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>'
    r"(.*?)</｜DSML｜ parameter>",
    re.DOTALL,
)


def parse_calls(text, *, allow_partial_eos=False):
    """Return content and full calls, preserving JSON types and string bytes.

    Incomplete or malformed markup raises ValueError rather than fabricating
    empty arguments. Callers retain the raw completion for diagnostics.
    """
    start = text.find(START)
    if start < 0:
        return text.removesuffix(EOS), []
    end = text.find(END, start + len(START))
    if end < 0:
        raise ValueError("Incomplete V4.1 DSML calls block")
    body = text[start + len(START) : end]
    calls, cursor = [], 0
    for invocation in INVOKE.finditer(body):
        if body[cursor : invocation.start()].strip():
            raise ValueError("Invalid V4.1 invoke markup")
        arguments, param_cursor = {}, 0
        params = invocation.group(2)
        for parameter in PARAM.finditer(params):
            if params[param_cursor : parameter.start()].strip():
                raise ValueError("Invalid V4.1 parameter markup")
            name, is_string, value = parameter.groups()
            if name in arguments:
                raise ValueError(f"Duplicate DSML parameter: {name}")
            arguments[name] = value if is_string == "true" else json.loads(value)
            param_cursor = parameter.end()
        if params[param_cursor:].strip():
            raise ValueError("Unparsed V4.1 parameter markup")
        calls.append((invocation.group(1), arguments))
        cursor = invocation.end()
    if body[cursor:].strip() or not calls:
        raise ValueError("Invalid or empty V4.1 calls block")
    suffix = text[end + len(END) :]
    if suffix.removesuffix(EOS).strip() and not (
        allow_partial_eos and EOS.startswith(suffix)
    ):
        raise ValueError("Unexpected content after V4.1 calls")
    return text[:start] or None, calls


def safe_content_prefix(text):
    """Hide even partial opening tags while streaming ordinary content."""
    start = text.find(START)
    if start >= 0:
        return text[:start]
    for marker in (START, EOS):
        for overlap in range(min(len(marker) - 1, len(text)), 0, -1):
            if text.endswith(marker[:overlap]):
                return text[:-overlap]
    return text.removesuffix(EOS)
