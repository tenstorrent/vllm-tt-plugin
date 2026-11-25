# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Types parser-engine tool-call arguments by the complete tool schema.

vLLM's parser engine (``vllm.parser.engine.parser_engine``, used by the
``qwen3_coder`` tool parser and the other engine parsers) receives argument
values as text and converts each value to the JSON type that the tool schema
declares for its parameter. In vLLM 0.33.0 that conversion has two defects:

- ``find_tool_properties`` returns the raw ``properties`` of the tool
  parameters. A parameter declared through a local ``$ref`` (Pydantic writes
  one for every nested model) or through ``allOf`` therefore has no type, and
  its value stays a JSON string (vLLM issue 46924, plugin issue 349).
- ``coerce_to_schema_type`` tries the integer, number, boolean, and null types
  before ``string``. For a union such as ``["string", "integer"]`` it turns the
  value ``007`` into ``7`` (vLLM issue 62252).

``install_patches`` replaces the two names in the ``parser_engine`` module,
which imports them from ``vllm.tool_parsers.utils``. Parsers that import those
functions into their own namespace keep the upstream behavior.

TODO: remove each patch once the pinned vLLM fixes its issue.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from typing import Any

from vllm.parser.engine import parser_engine
from vllm.tool_parsers import utils as tool_utils
from vllm.tool_parsers.utils import Tool

__all__ = ("install_patches", "keeps_raw_string", "resolve_properties")

# Bounds the nesting depth of ``_resolve`` on recursive or very deep schemas.
_MAX_DEPTH = 32

# Schema keys whose value maps names to subschemas.
_SCHEMA_MAPS = ("properties", "patternProperties")
# Schema keys whose value is one subschema, or a list of subschemas.
_SCHEMA_ITEMS = ("items", "additionalProperties", "prefixItems", "anyOf", "oneOf")


def install_patches() -> None:
    """Installs the ``$ref``/``allOf`` resolution and the string-union fix.

    Each patch stores the replaced function in a ``_tt_original_*`` attribute
    of ``vllm.parser.engine.parser_engine``; a second call finds that marker
    and changes nothing.
    """
    _install_properties_patch()
    _install_string_union_patch()


def resolve_properties(parameters: dict[str, Any]) -> dict[str, Any]:
    """Returns the parameter schemas of a tool with local references resolved.

    Parameters
    ----------
    parameters
        JSON schema of the tool's parameters, as the request supplied it. It is
        not modified.

    Returns
    -------
    dict
        The ``properties`` of ``parameters`` after ``_resolve`` replaced each
        local ``$ref`` (``#/...``) with its target and merged each ``allOf``,
        including ``allOf`` at the schema root. Empty when the resolved schema
        has no ``properties`` object. A reference that is not local, points
        nowhere, or refers to a schema that contains it is dropped, so that
        subschema is typed by its other keys as upstream types it.

    Examples
    --------
    >>> from vllm_tt_plugin.parsers.tool_schema import resolve_properties
    >>> resolve_properties(
    ...     {
    ...         "$defs": {"Point": {"type": "object"}},
    ...         "properties": {"at": {"$ref": "#/$defs/Point"}},
    ...     }
    ... )
    {'at': {'type': 'object'}}
    """
    properties = _resolve(parameters, parameters, ()).get("properties")
    return properties if isinstance(properties, dict) else {}


def keeps_raw_string(value: str, schema_type: str | list[str], result: Any) -> bool:
    """Tells whether a coerced scalar must stay the model's original text.

    Parameters
    ----------
    value
        Raw argument text that the model wrote.
    schema_type
        JSON schema type name or names of the parameter, before alias
        normalization (``"str"``, ``"int"``, ...).
    result
        What upstream ``coerce_to_schema_type`` returned for ``value``.

    Returns
    -------
    bool
        True when the schema allows ``"string"``, ``result`` is a JSON scalar
        (null, boolean, or number), and serializing ``result`` does not give
        back ``value`` (without surrounding whitespace). The string is then
        the only schema type that keeps the value unchanged.

    Examples
    --------
    >>> from vllm_tt_plugin.parsers.tool_schema import keeps_raw_string
    >>> keeps_raw_string("007", ["string", "integer"], 7)
    True
    >>> keeps_raw_string("7", ["string", "integer"], 7)
    False
    """
    if not (result is None or isinstance(result, (int, float))):
        return False
    types = [schema_type] if isinstance(schema_type, str) else schema_type
    aliases = tool_utils._TYPE_ALIASES
    names = {aliases.get(t.strip().lower(), t.strip().lower()) for t in types}
    return "string" in names and json.dumps(result) != value.strip()


def _install_properties_patch() -> None:
    """Makes the parser engine read parameter types through ``resolve_properties``.

    The replacement returns upstream's result for a schema without ``$ref`` or
    ``allOf``, so only schemas that use them change.
    """
    if hasattr(parser_engine, "_tt_original_find_tool_properties"):
        return
    original = parser_engine.find_tool_properties
    parser_engine._tt_original_find_tool_properties = original

    def find_tool_properties(
        tools: list[Tool] | None, tool_name: str
    ) -> dict[str, Any]:
        parameters = _find_parameters(tools, tool_name)
        if isinstance(parameters, dict) and _uses_ref_or_all_of(parameters):
            return resolve_properties(parameters)
        return original(tools, tool_name)

    parser_engine.find_tool_properties = find_tool_properties


def _install_string_union_patch() -> None:
    """Keeps a value as written when ``keeps_raw_string`` says the coercion loses it."""
    if hasattr(parser_engine, "_tt_original_coerce_to_schema_type"):
        return
    original: Callable[[str, str | list[str]], Any] = (
        parser_engine.coerce_to_schema_type
    )
    parser_engine._tt_original_coerce_to_schema_type = original

    def coerce_to_schema_type(value: str, schema_type: str | list[str]) -> Any:
        result = original(value, schema_type)
        return value if keeps_raw_string(value, schema_type, result) else result

    parser_engine.coerce_to_schema_type = coerce_to_schema_type


def _find_parameters(tools: Sequence[Tool] | None, name: str) -> dict[str, Any] | None:
    """Returns the parameters schema of the function tool ``name``, or None.

    Walks ``tools`` the way upstream ``find_tool_properties`` does: Responses
    tools through ``iter_response_function_tool_info``, chat tools through
    ``_extract_tool_info``. The first tool with that name wins.
    """
    responses_tools = (
        tool_utils.FunctionTool,
        tool_utils.NamespaceTool,
        tool_utils.CustomTool,
    )
    for tool in tools or ():
        if isinstance(tool, responses_tools):
            infos = tool_utils.iter_response_function_tool_info(tool)
        elif tool_utils._is_function_tool(tool):
            infos = [tool_utils._extract_tool_info(tool)]
        else:
            continue
        for tool_name, parameters in infos:
            if tool_name == name:
                return parameters
    return None


def _uses_ref_or_all_of(node: Any) -> bool:
    """Tells whether a schema contains a ``$ref`` or ``allOf`` key at any depth."""
    if isinstance(node, dict):
        return (
            "$ref" in node
            or "allOf" in node
            or any(map(_uses_ref_or_all_of, node.values()))
        )
    if isinstance(node, list):
        return any(map(_uses_ref_or_all_of, node))
    return False


def _resolve(node: Any, root: dict, refs: tuple[str, ...], depth: int = 0) -> Any:
    """Returns a copy of ``node`` with local ``$ref`` replaced and ``allOf`` merged.

    Parameters
    ----------
    node
        Subschema to resolve. A non-dict value is returned as it is.
    root
        Schema that ``#/...`` pointers are relative to.
    refs
        References being resolved on the path to ``node``. A reference to one
        of them is a cycle and is dropped.
    depth
        Nesting depth of this call. At ``_MAX_DEPTH`` the node is returned
        unresolved.

    Notes
    -----
    Keys next to a ``$ref`` override the same keys of its target. ``allOf``
    members merge as ``_merge`` describes. Only subschemas under the keys in
    ``_SCHEMA_MAPS`` and ``_SCHEMA_ITEMS`` are resolved; other values are
    deep-copied.
    """
    if not isinstance(node, dict) or depth >= _MAX_DEPTH:
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        rest = {key: value for key, value in node.items() if key != "$ref"}
        target = _resolve_pointer(root, ref) if ref.startswith("#") else None
        if ref in refs or not isinstance(target, dict):
            return _resolve(rest, root, refs, depth + 1)
        return _resolve({**target, **rest}, root, (*refs, ref), depth + 1)

    resolved: dict[str, Any] = {}
    for key, value in node.items():
        if key == "allOf":
            continue
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            resolved[key] = {
                name: _resolve(sub, root, refs, depth + 1)
                for name, sub in value.items()
            }
        elif key in _SCHEMA_ITEMS and isinstance(value, dict):
            resolved[key] = _resolve(value, root, refs, depth + 1)
        elif key in _SCHEMA_ITEMS and isinstance(value, list):
            resolved[key] = [_resolve(sub, root, refs, depth + 1) for sub in value]
        else:
            resolved[key] = copy.deepcopy(value)
    members = node.get("allOf")
    if isinstance(members, list):
        for member in members:
            if isinstance(member, dict):
                resolved = _merge(resolved, _resolve(member, root, refs, depth + 1))
    return resolved


def _resolve_pointer(root: dict, ref: str) -> Any:
    """Returns the node that the JSON pointer fragment ``ref`` names, or None."""
    node: Any = root
    for part in ref[1:].split("/"):
        if not part:
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdecimal() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _merge(base: dict, other: dict) -> dict:
    """Returns ``base`` merged with an ``allOf`` member ``other``.

    ``properties`` merge per name, recursively when both sides are schemas.
    ``required`` lists are joined without duplicates. For any other key, the
    value in ``base`` wins, so the first declared ``type`` is kept.
    """
    merged = dict(base)
    for key, value in other.items():
        if key not in merged:
            merged[key] = value
        elif key == "properties" and isinstance(value, dict):
            properties = dict(merged[key]) if isinstance(merged[key], dict) else {}
            for name, schema in value.items():
                current = properties.get(name)
                if isinstance(current, dict) and isinstance(schema, dict):
                    properties[name] = _merge(current, schema)
                else:
                    properties.setdefault(name, schema)
            merged[key] = properties
        elif key == "required" and isinstance(value, list):
            current = merged[key] if isinstance(merged[key], list) else []
            merged[key] = current + [name for name in value if name not in current]
    return merged
