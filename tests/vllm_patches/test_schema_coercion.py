# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Schema-directed coercion of tool-call arguments.

Raw-text formats (Qwen3-Coder/MiMo XML, `<parameter=x>5</parameter>`) carry
no type information, so the parser reads each value through the tool schema,
including schemas without a direct `type` (`$ref` into `$defs`, recursive
refs, `not`, `if/then/else`, `const`/`enum`, `{}`).  Formats that encode the
type (JSON arguments, Gemma4 `<|"|>..<|"|>`, DeepSeek `string="true"`) must
keep a value the model encoded as a string: under `{}` an explicit `"5"`
stays `"5"`.  A coerced value must validate against the property schema:
`5` under `not: {type: integer}` must not come back as an integer.

Runs against an installed vLLM; every streaming case asserts the streamed
arguments are byte-identical to the non-streamed ones.
"""

from __future__ import annotations

import json

import pytest

from tests.vllm_patches import requires_patched_vllm
from tests.vllm_patches.coercion_harness import (
    EXPLICIT_FORMATS,
    FORMATS,
    RAW_FORMATS,
    TYPED_FORMATS,
    Raw,
    Str,
    Typed,
    canon,
    request,
    tools_for,
)

pytestmark = requires_patched_vllm

CHUNKS = [1, 3, 7, 4096]

DEFS = {
    "Point": {
        "type": "object",
        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
    },
    "Category": {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "children": {"type": "array", "items": {"$ref": "#/$defs/Category"}},
        },
        "required": ["label", "children"],
    },
    "NoDate": {"type": "null"},
    "Alias": {"$ref": "#/$defs/Point"},
    "LoopA": {"$ref": "#/$defs/LoopB"},
    "LoopB": {"$ref": "#/$defs/LoopA"},
    "Str": {"type": "string"},
    "Int": {"type": "integer"},
}
DEFINITIONS = {"Legacy": {"type": "array", "items": {"type": "integer"}}}
TREE = '{"label": "Garden Tools", "children": [{"label": "Rakes", "children": []}]}'
IF_THEN_ELSE = {
    "if": {"type": "integer"},
    "then": {"type": "integer", "minimum": -100000000},
    "else": {"type": "string"},
}
IF_MIN = {"if": {"type": "integer"}, "then": {"minimum": 0}, "else": {"type": "string"}}
NOT_INT = {"not": {"type": "integer"}}
NULLABLE_ANY = {"anyOf": [{}, {"type": "null"}]}
TAG = ("tag", {"type": "string"}, "t {1}")


def _tools(prop_schema: dict) -> list[dict]:
    return tools_for(
        {
            "type": "object",
            "properties": {"x": prop_schema, TAG[0]: TAG[1]},
            "$defs": DEFS,
            "definitions": DEFINITIONS,
        }
    )


def _params(value, last: bool):
    x = ("x", value)
    tag = (TAG[0], Str(TAG[2]) if not isinstance(value, Raw) else Raw(TAG[2]))
    return [tag, x] if last else [x, tag]


def _check(fmt_name: str, schema: dict, value, expected, last: bool, chunk):
    fmt = FORMATS[fmt_name]
    tools = _tools(schema)
    if fmt_name == "qwen3_coder" or fmt_name == "mimo":
        params = _params(value, last)
    else:
        params = [(TAG[0], Str(TAG[2])), ("x", value)]
        if not last:
            params.reverse()
    text = fmt.render(params)
    ns = fmt.non_streaming(tools, text)
    assert canon(json.loads(ns)) == canon({"x": expected, TAG[0]: TAG[2]}), ns
    if chunk is not None:
        streamed = fmt.streaming(tools, text, chunk)
        assert streamed == ns


# ── The reviewer's two counter-cases, exactly ───────────────────────


@pytest.mark.parametrize("chunk", [None, *CHUNKS])
@pytest.mark.parametrize("fmt", EXPLICIT_FORMATS)
def test_explicit_string_5_under_empty_schema_stays_string(fmt, chunk):
    """Gemma4 `<|"|>5<|"|>`, DeepSeek `string="true"`, JSON `"5"` under `{}`."""
    _check(fmt, {}, Str("5"), "5", True, chunk)


@pytest.mark.parametrize("chunk", [None, *CHUNKS])
@pytest.mark.parametrize(
    "fmt,value",
    [(f, Raw("5")) for f in RAW_FORMATS]
    + [(f, Str("5")) for f in EXPLICIT_FORMATS]
    + [("gemma4", Raw("5"))]
    + [(f, Typed(5)) for f in TYPED_FORMATS],
    ids=lambda v: getattr(v, "__class__", type(v)).__name__
    if not isinstance(v, str)
    else v,
)
def test_5_under_not_integer_is_not_an_integer(fmt, value, chunk):
    """An integer-valued number is an integer in JSON Schema: only "5" fits."""
    if FORMATS[fmt].raw_json_stream and isinstance(value, Typed) and chunk:
        pytest.skip("no-converter JSON parsers stream the raw arguments")
    _check(fmt, NOT_INT, value, "5", True, chunk)


# ── Raw-text formats (Qwen3-Coder / MiMo) ────────────────────────────

RAW_CASES: list[tuple[str, dict, str, object]] = [
    ("ref_object", {"$ref": "#/$defs/Point"}, '{"x": 1, "y": 2}', {"x": 1, "y": 2}),
    (
        "ref_object_nested_strings_typed",
        {"$ref": "#/$defs/Point"},
        '{"x": "1", "y": 2}',
        {"x": 1, "y": 2},
    ),
    ("ref_recursive", {"$ref": "#/$defs/Category"}, TREE, json.loads(TREE)),
    (
        "ref_null_with_sibling",
        {"$ref": "#/$defs/NoDate", "description": "null"},
        "null",
        None,
    ),
    ("ref_chain", {"$ref": "#/$defs/Alias"}, '{"x": 3}', {"x": 3}),
    ("ref_cycle_json_fallback", {"$ref": "#/$defs/LoopA"}, "[1, 2]", [1, 2]),
    ("ref_definitions", {"$ref": "#/definitions/Legacy"}, "[3, 4]", [3, 4]),
    ("ref_dangling_json_fallback", {"$ref": "#/$defs/Nope"}, "7", 7),
    ("ref_to_string_stays_string", {"$ref": "#/$defs/Str"}, "[1]", "[1]"),
    ("ref_to_integer", {"$ref": "#/$defs/Int"}, "42", 42),
    ("allOf_ref", {"allOf": [{"$ref": "#/$defs/Point"}]}, '{"x": 1}', {"x": 1}),
    (
        "anyOf_ref_or_null__null",
        {"anyOf": [{"$ref": "#/$defs/Point"}, {"type": "null"}]},
        "null",
        None,
    ),
    (
        "anyOf_ref_or_null__object",
        {"anyOf": [{"$ref": "#/$defs/Point"}, {"type": "null"}]},
        '{"y": 9}',
        {"y": 9},
    ),
    (
        "oneOf_ref_or_int",
        {"oneOf": [{"$ref": "#/$defs/Point"}, {"type": "integer"}]},
        "7",
        7,
    ),
    ("not_string__int", {"not": {"type": "string"}}, "-2500", -2500),
    ("not_string__object", {"not": {"type": "string"}}, '{"a": 1}', {"a": 1}),
    ("not_integer__int_stays_string", NOT_INT, "5", "5"),
    ("not_integer__integral_float_stays_string", NOT_INT, "5.0", "5.0"),
    ("not_integer__fraction", NOT_INT, "5.5", 5.5),
    ("not_integer__text", NOT_INT, "abc", "abc"),
    ("not_number__int_stays_string", {"not": {"type": "number"}}, "5", "5"),
    (
        "number_not_integer__unsatisfiable_keeps_raw",
        {"type": "number", "not": {"type": "integer"}},
        "5",
        "5",
    ),
    (
        "allOf_int_or_str_not_int",
        {"allOf": [{"type": ["integer", "string"]}, NOT_INT]},
        "5",
        "5",
    ),
    ("if_then_else__int", IF_THEN_ELSE, "-2500", -2500),
    ("if_then_else__string", IF_THEN_ELSE, "12.50", "12.50"),
    ("if_min__then_holds", IF_MIN, "7", 7),
    ("if_min__then_fails_keeps_string", IF_MIN, "-5", "-5"),
    ("const_int", {"const": 42}, "42", 42),
    ("const_int_as_float", {"const": 5.0}, "5", 5),
    ("const_bool", {"const": True}, "true", True),
    ("const_null", {"const": None}, "null", None),
    ("const_string_digits_stays_string", {"const": "5"}, "5", "5"),
    ("enum_ints", {"enum": [1, 2, 3]}, "2", 2),
    ("enum_str_5_or_int_6__5", {"enum": ["5", 6]}, "5", "5"),
    ("enum_str_5_or_int_6__6", {"enum": ["5", 6]}, "6", 6),
    ("enum_null_or_str__null", {"enum": [None, "a"]}, "null", None),
    ("nullable_type_list__null", {"type": ["integer", "null"]}, "null", None),
    ("nullable_type_list__int", {"type": ["integer", "null"]}, "5", 5),
    ("empty__object", {}, '{"a": [1, true, null]}', {"a": [1, True, None]}),
    ("empty__int", {}, "5", 5),
    ("empty__negative_float", {}, "-3.5", -3.5),
    ("empty__null", {"description": "anything"}, "null", None),
    ("empty__bool", {}, "false", False),
    ("empty__plain_text", {}, "hello world", "hello world"),
    ("empty__json_string_stays_raw", {}, '"quoted"', '"quoted"'),
    ("empty__nan_stays_string", {}, "NaN", "NaN"),
    ("empty__invalid_json_stays_string", {}, '{"a": ', '{"a": '),
    # The benchmark's `nullable/{}` shape: an unconstrained branch.
    ("anyOf_empty_or_null__int", NULLABLE_ANY, "25", 25),
    ("anyOf_empty_or_null__null", NULLABLE_ANY, "null", None),
    ("anyOf_empty_or_null__array", NULLABLE_ANY, "[1, null]", [1, None]),
    ("anyOf_empty_or_null__text", NULLABLE_ANY, "abc", "abc"),
    ("not_const__json_fallback_validated", {"not": {"const": 5}}, "5", "5"),
    ("string__looks_like_object", {"type": "string"}, '{"a": 1}', '{"a": 1}'),
    ("string__looks_like_int", {"type": "string"}, "123", "123"),
    ("string__looks_like_null", {"type": "string"}, "null", "null"),
    (
        "anyOf_string_or_int__text",
        {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        "abc",
        "abc",
    ),
]


@pytest.mark.parametrize("chunk", [None, *CHUNKS])
@pytest.mark.parametrize("last", [False, True], ids=["first", "last"])
@pytest.mark.parametrize(
    "schema,raw,expected", [c[1:] for c in RAW_CASES], ids=[c[0] for c in RAW_CASES]
)
@pytest.mark.parametrize("fmt", RAW_FORMATS)
def test_raw_text_formats(fmt, schema, raw, expected, last, chunk):
    _check(fmt, schema, Raw(raw), expected, last, chunk)


# ── Formats that encode types: explicit strings ──────────────────────

EXPLICIT_CASES: list[tuple[str, dict, str, object]] = [
    # Under a schema without type information an explicit string is kept.
    ("empty__5", {}, "5", "5"),
    ("empty__true", {}, "true", "true"),
    ("empty__null", {}, "null", "null"),
    ("empty__float", {}, "-3.5", "-3.5"),
    ("empty__object_text", {}, '{"a": 1}', '{"a": 1}'),
    ("empty__array_text", {}, "[1]", "[1]"),
    ("description_only__null", {"description": "anything"}, "null", "null"),
    ("ref_dangling__7", {"$ref": "#/$defs/Nope"}, "7", "7"),
    ("ref_cycle__array_text", {"$ref": "#/$defs/LoopA"}, "[1, 2]", "[1, 2]"),
    ("not_const__5", {"not": {"const": 5}}, "5", "5"),
    ("anyOf_empty_or_null__25", NULLABLE_ANY, "25", "25"),
    # A string the schema allows is kept.
    ("not_integer__5", NOT_INT, "5", "5"),
    ("if_then_else__string_branch_valid", IF_THEN_ELSE, "-2500", "-2500"),
    ("const_string", {"const": "42"}, "42", "42"),
    ("enum_str_5_or_int_6", {"enum": ["5", 6]}, "5", "5"),
    ("ref_to_string", {"$ref": "#/$defs/Str"}, "[1]", "[1]"),
    ("string_type", {"type": "string"}, "123", "123"),
    # A string the schema forbids is repaired into a value that validates,
    # as v0.31.0 already does for a declared `type`.
    ("ref_to_integer", {"$ref": "#/$defs/Int"}, "42", 42),
    ("ref_object", {"$ref": "#/$defs/Point"}, '{"x": 1}', {"x": 1}),
    ("not_string__int", {"not": {"type": "string"}}, "-2500", -2500),
    ("const_int", {"const": 42}, "42", 42),
    ("ref_null", {"$ref": "#/$defs/NoDate"}, "null", None),
    # Declared types: v0.31.0's own coercion, unchanged.
    ("declared_integer", {"type": "integer"}, "42", 42),
    (
        "declared_int_or_string_unchanged_upstream",
        {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        "5",
        5,
    ),
]


@pytest.mark.parametrize("chunk", [None, *CHUNKS])
@pytest.mark.parametrize("last", [False, True], ids=["first", "last"])
@pytest.mark.parametrize(
    "schema,text,expected",
    [c[1:] for c in EXPLICIT_CASES],
    ids=[c[0] for c in EXPLICIT_CASES],
)
@pytest.mark.parametrize("fmt", EXPLICIT_FORMATS)
def test_explicit_strings(fmt, schema, text, expected, last, chunk):
    if FORMATS[fmt].raw_json_stream and chunk and expected != text:
        pytest.skip("no-converter JSON parsers stream the raw arguments")
    _check(fmt, schema, Str(text), expected, last, chunk)


# ── Formats that encode types: typed values ──────────────────────────

TYPED_CASES: list[tuple[str, dict, object, object]] = [
    ("empty__int_kept", {}, 5, 5),
    ("empty__object_kept", {}, {"a": "1"}, {"a": "1"}),
    ("not_integer__int_to_string", NOT_INT, 5, "5"),
    ("not_integer__fraction_kept", NOT_INT, 5.5, 5.5),
    ("ref_to_integer", {"$ref": "#/$defs/Int"}, 5, 5),
    ("ref_to_string", {"$ref": "#/$defs/Str"}, 5, "5"),
    ("const_string", {"const": "5"}, 5, "5"),
    ("ref_object_nested", {"$ref": "#/$defs/Point"}, {"x": "1"}, {"x": 1}),
    (
        "ref_recursive",
        {"$ref": "#/$defs/Category"},
        json.loads(TREE),
        json.loads(TREE),
    ),
    ("if_then_else__int", IF_THEN_ELSE, -2500, -2500),
    ("declared_string", {"type": "string"}, 12345, "12345"),
]


@pytest.mark.parametrize("chunk", [None, *CHUNKS])
@pytest.mark.parametrize("last", [False, True], ids=["first", "last"])
@pytest.mark.parametrize(
    "schema,value,expected",
    [c[1:] for c in TYPED_CASES],
    ids=[c[0] for c in TYPED_CASES],
)
@pytest.mark.parametrize("fmt", TYPED_FORMATS)
def test_typed_values(fmt, schema, value, expected, last, chunk):
    if FORMATS[fmt].raw_json_stream and chunk and canon(expected) != canon(value):
        pytest.skip("no-converter JSON parsers stream the raw arguments")
    _check(fmt, schema, Typed(value), expected, last, chunk)


# Gemma4 bare values (`num:5`) reach the engine as strings, like explicit
# ones; only the schema can type them.
GEMMA4_BARE = [
    ("ref_to_integer", {"$ref": "#/$defs/Int"}, "5", 5),
    ("declared_integer", {"type": "integer"}, "5", 5),
    ("not_integer", NOT_INT, "5", "5"),
    ("empty_unchanged_upstream", {}, "5", "5"),
]


@pytest.mark.parametrize("chunk", [None, *CHUNKS])
@pytest.mark.parametrize(
    "schema,text,expected",
    [c[1:] for c in GEMMA4_BARE],
    ids=[c[0] for c in GEMMA4_BARE],
)
def test_gemma4_bare_values(schema, text, expected, chunk):
    _check("gemma4", schema, Raw(text), expected, True, chunk)


# ── Streaming shape ──────────────────────────────────────────────────


def _stream_prefix(fmt_name: str, schema: dict, value, cut_marker: str) -> str:
    fmt = FORMATS[fmt_name]
    tools = _tools(schema)
    text = fmt.render([("x", value), (TAG[0], Str(TAG[2]))])
    return fmt.streaming(tools, text[: text.index(cut_marker)], 1)


@pytest.mark.parametrize("fmt", RAW_FORMATS)
def test_ref_string_param_streams_before_it_closes(fmt):
    got = _stream_prefix(
        fmt, {"$ref": "#/$defs/Str"}, Raw(("word " * 40).strip()), "</parameter>"
    )
    assert "word word word" in got


@pytest.mark.parametrize("fmt", RAW_FORMATS)
def test_untyped_raw_value_is_not_streamed_as_string_prefix(fmt):
    pieces = FORMATS[fmt].streaming_pieces(
        _tools({}), FORMATS[fmt].render([(TAG[0], Raw("t")), ("x", Raw("12345"))]), 1
    )
    assert '"x": "' not in "".join(pieces)
    assert json.loads("".join(pieces)) == {"tag": "t", "x": 12345}


@pytest.mark.parametrize("fmt", EXPLICIT_FORMATS + RAW_FORMATS)
def test_streamable_keys(fmt):
    """String values under `{}` may stream early unless they are raw text.

    (Private API: `_tool_string_keys` decides which values stream before
    they close.)  An explicit string under `{}` is never reinterpreted, so
    it keeps streaming as before; raw text under `{}` may decode to JSON,
    and a `not: integer` value depends on the whole text, so both wait.
    """
    f = FORMATS[fmt]
    tools = tools_for(
        {
            "type": "object",
            "properties": {
                "empty": {},
                "s": {"$ref": "#/$defs/Str"},
                "n": NOT_INT,
                "i": {"$ref": "#/$defs/Int"},
            },
            "$defs": DEFS,
        }
    )
    parser = f.engine()(f.tokenizer(), request(tools).tools, **f.engine_kwargs)
    keys = parser._tool_string_keys("f")
    want = {"s"} if fmt in RAW_FORMATS else {"s", "empty"}
    assert keys == want


@pytest.mark.parametrize("fmt", RAW_FORMATS + ["gemma4", "deepseek_v4"])
def test_not_integer_value_is_not_streamed_early(fmt):
    """Its reading depends on the whole value, so it is held until it closes."""
    f = FORMATS[fmt]
    value = Raw("5") if fmt in RAW_FORMATS else Str("5")
    pieces = f.streaming_pieces(_tools(NOT_INT), f.render([("x", value)]), 1)
    assert json.loads("".join(pieces)) == {"x": "5"}
    assert not any('"x": "' in p and not p.endswith('"}') for p in pieces[:-1])


# ── `$defs` survive request handling ─────────────────────────────────


def test_required_tool_choice_keeps_defs():
    from vllm.tool_parsers.utils import get_json_schema_from_tools

    tools = _tools({"$ref": "#/$defs/Point"})
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
    )

    parsed = [ChatCompletionToolsParam(**t) for t in tools]
    schema = get_json_schema_from_tools(tool_choice="required", tools=parsed)
    assert schema is not None
    assert "Point" in json.dumps(schema)
    assert parsed[0].function.parameters["$defs"]["Point"] == DEFS["Point"]
