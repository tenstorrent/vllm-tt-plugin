# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Argument coercion for tools whose whole `parameters` schema is a combinator.

A root of `{"$ref": "#/$defs/Params"}`, `{"allOf": [...]}`, `{"anyOf": [...]}`,
`{"oneOf": [...]}` or `{"type": "object", "anyOf": [...]}` has no root
`properties`; the parser used to skip coercion for it, so raw-text formats
(MiMo/Qwen3-Coder) returned every value as a string: `48213` -> `"48213"`.

Runs against an installed vLLM; streaming cases assert the streamed
arguments are byte-identical to the non-streamed ones.
"""

from __future__ import annotations

import json

import pytest

from tests.vllm_patches import requires_patched_vllm
from tests.vllm_patches.coercion_harness import (
    FORMATS,
    RAW_FORMATS,
    Raw,
    Str,
    Typed,
    canon,
    tools_for,
)

pytestmark = requires_patched_vllm

CHUNKS = [1, 2, 3, 5, 8, 100_000]


def _obj(props: dict, required: list[str] | None = None, closed: bool = True):
    out: dict = {"type": "object", "properties": props}
    out["required"] = list(props) if required is None else required
    if closed:
        out["additionalProperties"] = False
    return out


def _two_way(kw: str, typed_key: str, typed: dict) -> dict:
    """The fuzz harness's root-anyOf/oneOf shape: typed key vs string code."""
    return {kw: [_obj({typed_key: typed}), _obj({"code": {"type": "string"}})]}


TYPED = {
    "integer": ({"type": "integer"}, "-300", -300),
    "number": ({"type": "number"}, "-18.5", -18.5),
    "boolean": ({"type": "boolean"}, "true", True),
    "null": ({"type": "null"}, "null", None),
}

SHAPES = {
    "root-anyOf": lambda s: _two_way("anyOf", "v", s),
    "root-oneOf": lambda s: _two_way("oneOf", "v", s),
    "root-object+anyOf": lambda s: {"type": "object", **_two_way("anyOf", "v", s)},
    "root-allOf1": lambda s: {"allOf": [_obj({"v": s})]},
    "root-ref": lambda s: {"$ref": "#/$defs/P", "$defs": {"P": _obj({"v": s})}},
    "root-ref-chain": lambda s: {
        "$ref": "#/$defs/A",
        "$defs": {"A": {"$ref": "#/$defs/B"}, "B": _obj({"v": s})},
    },
    "root-anyOf-of-refs": lambda s: {
        "anyOf": [{"$ref": "#/$defs/ById"}, {"$ref": "#/$defs/ByCode"}],
        "$defs": {
            "ById": _obj({"v": s}),
            "ByCode": _obj({"code": {"type": "string"}}),
        },
    },
    "root-oneOf-nested-ref-prop": lambda s: {
        "oneOf": [
            _obj({"v": {"$ref": "#/$defs/V"}}),
            _obj({"code": {"type": "string"}}),
        ],
        "$defs": {"V": s},
    },
    "root-allOf-two": lambda s: {
        "allOf": [_obj({"v": s}, closed=False), _obj({"note": {"type": "string"}}, [])]
    },
    "root-allOf-intersect": lambda s: {
        "allOf": [
            _obj({"v": {"type": [s["type"], "string"]}}, closed=False),
            _obj({"v": s}, closed=False),
        ]
    },
    "root-props+anyOf": lambda s: {
        "type": "object",
        "properties": {"note": {"type": "string"}},
        "anyOf": [_obj({"v": s}, closed=False), _obj({"code": {"type": "string"}})],
    },
    "root-anyOf-nested-allOf": lambda s: {
        "anyOf": [
            {"allOf": [{"$ref": "#/$defs/P"}]},
            _obj({"code": {"type": "string"}}),
        ],
        "$defs": {"P": _obj({"v": s})},
    },
}
WITH_NOTE = ("root-allOf-two", "root-props+anyOf")

CASES = [
    (shape, typ, where)
    for shape in SHAPES
    for typ in TYPED
    for where in (("alone", "first", "last") if shape in WITH_NOTE else ("alone",))
]
IDS = [f"{s}/{t}/{w}" for s, t, w in CASES]


def _case(shape: str, typ: str, where: str, wrap):
    schema, raw, value = TYPED[typ]
    tools = tools_for(SHAPES[shape](schema))
    extra = [("note", "n {1}")] if shape in WITH_NOTE and where != "alone" else []
    params = [("v", raw)]
    params = extra + params if where == "last" else params + extra
    want = {"v": value, **dict(extra)}
    return tools, [(k, wrap(v)) for k, v in params], want


def _run(fmt_name: str, tools, params, want, chunk):
    fmt = FORMATS[fmt_name]
    text = fmt.render(params)
    ns = fmt.non_streaming(tools, text)
    assert canon(json.loads(ns)) == canon(want), ns
    if chunk is not None:
        assert fmt.streaming(tools, text, chunk) == ns


@pytest.mark.parametrize("chunk", [None, *CHUNKS])
@pytest.mark.parametrize("shape,typ,where", CASES, ids=IDS)
@pytest.mark.parametrize("fmt", RAW_FORMATS)
def test_raw_text_root_combinators(fmt, shape, typ, where, chunk):
    tools, params, want = _case(shape, typ, where, Raw)
    _run(fmt, tools, params, want, chunk)


@pytest.mark.parametrize("chunk", [None, 1, 7])
@pytest.mark.parametrize("shape,typ,where", CASES, ids=IDS)
@pytest.mark.parametrize("fmt", ["deepseek_v4", "deepseek_v32"])
def test_typed_root_combinators(fmt, shape, typ, where, chunk):
    """DeepSeek `string="false"` values under a combinator root keep their type."""
    tools, params, want = _case(shape, typ, where, Raw)
    params = [(k, v if k == "v" else Str(v.text)) for k, v in params]
    _run(fmt, tools, params, want, chunk)


STRING_CASES = [
    (
        "anyOf_other_branch_string",
        _two_way("anyOf", "v", {"type": "integer"}),
        [("code", "007")],
        {"code": "007"},
    ),
    (
        "oneOf_other_branch_string_looks_like_null",
        _two_way("oneOf", "v", {"type": "null"}),
        [("code", "null")],
        {"code": "null"},
    ),
    (
        "ref_string_prop",
        {
            "$ref": "#/$defs/P",
            "$defs": {"P": _obj({"s": {"type": "string"}})},
        },
        [("s", "123")],
        {"s": "123"},
    ),
    (
        "unknown_key_left_alone",
        _two_way("anyOf", "v", {"type": "integer"}),
        [("other", "5")],
        {"other": "5"},
    ),
]


@pytest.mark.parametrize("chunk", [None, *CHUNKS])
@pytest.mark.parametrize(
    "params,emitted,want",
    [c[1:] for c in STRING_CASES],
    ids=[c[0] for c in STRING_CASES],
)
@pytest.mark.parametrize("fmt", RAW_FORMATS)
def test_string_branches(fmt, params, emitted, want, chunk):
    _run(fmt, tools_for(params), [(k, Raw(v)) for k, v in emitted], want, chunk)


CONFLICT = {
    "anyOf": [
        _obj({"id": {"type": "integer"}}, ["id"]),
        _obj({"id": {"type": "string"}, "code": {"type": "string"}}, ["id", "code"]),
    ]
}
AMBIGUOUS = {
    "oneOf": [
        _obj({"id": {"type": "integer"}, "a": {"type": "string"}}, []),
        _obj({"id": {"type": "string"}, "b": {"type": "string"}}, []),
    ]
}
CONFLICT_CASES = [
    ("pick_int_branch", CONFLICT, [("id", "42")], {"id": 42}),
    (
        "pick_string_branch_id_first",
        CONFLICT,
        [("id", "007"), ("code", "x")],
        {"id": "007", "code": "x"},
    ),
    (
        "pick_string_branch_id_last",
        CONFLICT,
        [("code", "x"), ("id", "007")],
        {"id": "007", "code": "x"},
    ),
    ("ambiguous_union_json_fallback_int", AMBIGUOUS, [("id", "7")], {"id": 7}),
    ("ambiguous_union_json_fallback_text", AMBIGUOUS, [("id", "abc")], {"id": "abc"}),
    ("disambiguated_by_a", AMBIGUOUS, [("a", "z"), ("id", "7")], {"a": "z", "id": 7}),
    (
        "disambiguated_by_b_later",
        AMBIGUOUS,
        [("id", "7"), ("b", "z")],
        {"id": "7", "b": "z"},
    ),
]


@pytest.mark.parametrize("chunk", [None, *CHUNKS])
@pytest.mark.parametrize(
    "params,emitted,want",
    [c[1:] for c in CONFLICT_CASES],
    ids=[c[0] for c in CONFLICT_CASES],
)
@pytest.mark.parametrize("fmt", RAW_FORMATS)
def test_conflicting_key_branch_choice(fmt, params, emitted, want, chunk):
    _run(fmt, tools_for(params), [(k, Raw(v)) for k, v in emitted], want, chunk)


EXPLICIT_CONFLICT_CASES = [
    # An ambiguous key is unconstrained: an explicit string stays a string.
    ("ambiguous_explicit_string_kept", AMBIGUOUS, [("id", Str("7"))], {"id": "7"}),
    ("ambiguous_typed_kept", AMBIGUOUS, [("id", Typed(7))], {"id": 7}),
    ("int_branch_repairs_explicit_string", CONFLICT, [("id", Str("42"))], {"id": 42}),
]


@pytest.mark.parametrize("chunk", [None, 1, 7])
@pytest.mark.parametrize(
    "params,emitted,want",
    [c[1:] for c in EXPLICIT_CONFLICT_CASES],
    ids=[c[0] for c in EXPLICIT_CONFLICT_CASES],
)
@pytest.mark.parametrize("fmt", ["deepseek_v4", "deepseek_v41", "kimi_k2"])
def test_explicit_conflicting_key(fmt, params, emitted, want, chunk):
    if FORMATS[fmt].raw_json_stream and chunk:
        pytest.skip("no-converter JSON parsers stream the raw arguments")
    _run(fmt, tools_for(params), emitted, want, chunk)


@pytest.mark.parametrize("fmt", RAW_FORMATS)
def test_conflicting_key_never_streams_as_string_prefix(fmt):
    f = FORMATS[fmt]
    pieces = f.streaming_pieces(
        tools_for(AMBIGUOUS), f.render([("id", Raw("7")), ("b", Raw("zz"))]), 1
    )
    joined = "".join(pieces)
    assert json.loads(joined) == {"id": "7", "b": "zz"}
    assert '"7"' not in "".join(pieces[:-1])


@pytest.mark.parametrize("fmt", RAW_FORMATS)
def test_string_key_in_every_branch_still_streams_incrementally(fmt):
    f = FORMATS[fmt]
    tools = tools_for(
        {
            "anyOf": [
                _obj({"text": {"type": "string"}, "n": {"type": "integer"}}, []),
                _obj({"text": {"$ref": "#/$defs/S"}}, []),
            ],
            "$defs": {"S": {"type": "string"}},
        }
    )
    text = f.render([("text", Raw(("word " * 40).strip()))])
    got = f.streaming(tools, text[: text.index("</parameter>")], 1)
    assert "word word word" in got


@pytest.mark.parametrize("fmt", RAW_FORMATS)
def test_root_ref_cycle_is_left_alone(fmt):
    f = FORMATS[fmt]
    tools = tools_for({"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "#/$defs/A"}}})
    text = f.render([("v", Raw("5"))])
    ns = f.non_streaming(tools, text)
    assert json.loads(ns) == {"v": "5"}
    assert f.streaming(tools, text, 3) == ns


@pytest.mark.parametrize("fmt", RAW_FORMATS)
def test_plain_object_schema_unchanged(fmt):
    f = FORMATS[fmt]
    tools = tools_for(_obj({"v": {"type": "integer"}, "s": {"type": "string"}}))
    text = f.render([("s", Raw("9")), ("v", Raw("9"))])
    ns = f.non_streaming(tools, text)
    assert ns == '{"s": "9", "v": 9}'
    for chunk in CHUNKS:
        assert f.streaming(tools, text, chunk) == ns
