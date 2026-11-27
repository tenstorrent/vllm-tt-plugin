# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks argument types that vLLM's parser engine derives from tool schemas.

Each case runs the upstream ``qwen3_coder`` parser class after
``install_patches``, on the complete output and on the streamed output. The
model writes every argument as text; the tool schema decides its JSON type.
"""

import copy
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import pytest

from tests.support import parser_patches
from tests.support.tool_calls import Call, call_text, extract, stream
from vllm_tt_plugin.parsers import tool_schema

Parse = Callable[[dict[str, dict], str], list[Call]]

POINT = {
    "type": "object",
    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
}


class Case(NamedTuple):
    """One tool parameters schema, the argument text, and the parsed value."""

    parameters: dict
    text: str
    expected: Any


@pytest.fixture(autouse=True)
def installed() -> Iterator[None]:
    """Installs the patches for one test and restores upstream vLLM afterward."""
    restore = parser_patches.snapshot()
    tool_schema.install_patches()
    yield
    restore()


@pytest.fixture(
    params=[pytest.param(extract, id="one-shot"), pytest.param(stream, id="streamed")]
)
def parse(request: pytest.FixtureRequest) -> Parse:
    """Parses a model output on the non-streaming or the streaming path."""
    return request.param


def _parse_arg(parse: Parse, parameters: dict, text: str) -> Any:
    """Returns the parsed value of the single argument ``arg`` of tool ``f``."""
    calls = parse({"f": parameters}, call_text("f", {"arg": text}))
    assert [call.name for call in calls] == ["f"]
    return calls[0].arguments["arg"]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            Case(
                {
                    "$defs": {"Point": POINT},
                    "properties": {"arg": {"$ref": "#/$defs/Point"}},
                },
                '{"x": 1, "y": 2}',
                {"x": 1, "y": 2},
            ),
            id="ref-to-object",
        ),
        pytest.param(
            Case(
                {
                    "$defs": {"Count": {"type": "integer"}},
                    "properties": {
                        "arg": {"$ref": "#/$defs/Count", "description": "n"}
                    },
                },
                "12",
                12,
            ),
            id="ref-with-sibling-keys",
        ),
        pytest.param(
            Case(
                {
                    "definitions": {"Count": {"type": "integer"}},
                    "properties": {"arg": {"$ref": "#/definitions/Count"}},
                },
                "3",
                3,
            ),
            id="definitions-ref",
        ),
        pytest.param(
            Case(
                {
                    "$defs": {"Point": POINT},
                    "properties": {"arg": {"allOf": [{"$ref": "#/$defs/Point"}]}},
                },
                '{"x": 1, "y": 2}',
                {"x": 1, "y": 2},
            ),
            id="allof-ref-property",
        ),
        pytest.param(
            Case(
                {
                    "$defs": {"Point": POINT},
                    "properties": {
                        "arg": {"anyOf": [{"$ref": "#/$defs/Point"}, {"type": "null"}]}
                    },
                },
                '{"x": 1, "y": 2}',
                {"x": 1, "y": 2},
            ),
            id="optional-ref",
        ),
        pytest.param(
            Case(
                {
                    "allOf": [
                        {"type": "object", "properties": {"arg": {"type": "integer"}}}
                    ]
                },
                "5",
                5,
            ),
            id="allof-at-root",
        ),
        pytest.param(
            Case(
                {
                    "$defs": {
                        "Node": {
                            "type": "object",
                            "properties": {
                                "children": {
                                    "type": "array",
                                    "items": {"$ref": "#/$defs/Node"},
                                }
                            },
                        }
                    },
                    "properties": {"arg": {"$ref": "#/$defs/Node"}},
                },
                '{"children": [{"children": []}]}',
                {"children": [{"children": []}]},
            ),
            id="recursive-ref",
        ),
    ],
)
def test_reference_gives_argument_type(parse: Parse, case: Case):
    """A parameter typed through ``$ref`` or ``allOf`` gets that JSON type."""
    assert _parse_arg(parse, case.parameters, case.text) == case.expected


def test_unresolvable_reference_keeps_string(parse: Parse):
    """A reference that points nowhere leaves the value a string, as upstream."""
    parameters = {"properties": {"arg": {"$ref": "#/$defs/Missing"}}}
    assert _parse_arg(parse, parameters, "12") == "12"


@pytest.mark.parametrize(
    ("types", "text", "expected"),
    [
        pytest.param(["string", "integer"], "007", "007", id="leading-zeros"),
        pytest.param(["string", "number"], "1.50", "1.50", id="trailing-zero"),
        pytest.param(["str", "int"], "+5", "+5", id="type-aliases"),
        pytest.param(["string", "integer"], "42", 42, id="exact-integer"),
        pytest.param(["string", "boolean"], "true", True, id="exact-boolean"),
        pytest.param(["string", "null"], "null", None, id="exact-null"),
        pytest.param(["integer"], "007", 7, id="no-string-type"),
    ],
)
def test_string_union_keeps_value_as_written(
    parse: Parse, types: list[str], text: str, expected: Any
):
    """A union with ``string`` converts a scalar only when nothing is lost."""
    parameters = {"properties": {"arg": {"type": types}}}
    assert _parse_arg(parse, parameters, text) == expected


def test_resolve_properties_leaves_request_schema_unchanged():
    """The request's schema is read, never modified."""
    parameters = {
        "$defs": {"Point": POINT},
        "allOf": [{"properties": {"at": {"$ref": "#/$defs/Point"}}}],
    }
    before = copy.deepcopy(parameters)

    assert tool_schema.resolve_properties(parameters) == {"at": POINT}
    assert parameters == before
