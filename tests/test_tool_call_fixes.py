# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Host tests for the tool-call monkeypatches (P1 to P4).

The tests drive the real vLLM `qwen3_coder` engine tool parser with the local
Qwen3.8 tokenizer, both in one shot and as streamed token deltas.
"""

import asyncio
import copy
import json
import os
from types import SimpleNamespace

import pytest
from jsonschema import Draft7Validator

import vllm_tt_plugin
import vllm_tt_plugin.tool_call_fixes as fixes

TOKENIZER_PATH = os.environ.get(
    "QWEN38_TOKENIZER_PATH", "/mnt/tt-data/sdawle/models/Qwen3.8-27B"
)
WORKTREE = os.environ.get(
    "TT_PLUGIN_WORKTREE", "/home/ttuser/atupe/qwen-or/wt/plugin-toolfix"
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(TOKENIZER_PATH), reason=f"{TOKENIZER_PATH} is not available"
)


def test_plugin_comes_from_worktree():
    # The installed editable plugin is the main checkout; the tests must not
    # silently exercise it.
    assert vllm_tt_plugin.__file__.startswith(WORKTREE)
    assert fixes.__file__.startswith(WORKTREE)


@pytest.fixture(autouse=True)
def installed_fixes():
    # tests/conftest.py restores the patched attributes after every test.
    fixes.install_tool_call_fixes()


def uninstall_fixes() -> None:
    """Put the upstream functions back, to show what the old code does."""
    from vllm.parser import qwen3
    from vllm.parser.engine import parser_engine, streaming_parser_engine

    engine = streaming_parser_engine.StreamingParserEngine
    for owner, attr, marker in (
        (engine, "_on_terminal", "_tt_original_on_terminal"),
        (engine, "_emit_for_state", "_tt_original_emit_for_state"),
        (parser_engine, "find_tool_properties", "_tt_original_find_tool_properties"),
        (parser_engine, "coerce_to_schema_type", "_tt_original_coerce_to_schema_type"),
        (qwen3, "_qwen3_arg_converter", "_tt_original_qwen3_arg_converter"),
    ):
        setattr(owner, attr, getattr(owner, marker))
    qwen3.qwen3_config.cache_clear()


@pytest.fixture(scope="module")
def tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(TOKENIZER_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request(tools: dict[str, dict]):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    return ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {"type": "function", "function": {"name": name, "parameters": schema}}
            for name, schema in tools.items()
        ],
    )


def make_parser(tokenizer, request):
    from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser

    return Qwen3EngineToolParser(tokenizer, request.tools)


def call_text(name: str, params: dict[str, str], close: str = "</parameter>\n") -> str:
    body = "".join(f"<parameter={k}>\n{v}\n{close}" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def extract(tokenizer, request, text: str) -> list[tuple[str, str]]:
    parser = make_parser(tokenizer, request)
    result = parser.extract_tool_calls(text, request)
    return [(c.function.name, c.function.arguments) for c in result.tool_calls]


def stream(tokenizer, request, text: str, step: int) -> list[tuple[str, str]]:
    """Feed `text` in deltas of `step` tokens; return (name, arguments) per index."""
    parser = make_parser(tokenizer, request)
    ids = tokenizer.encode(text, add_special_tokens=False)
    calls: dict[int, list[str]] = {}

    def collect(delta) -> None:
        for tc in (delta.tool_calls if delta else None) or ():
            entry = calls.setdefault(tc.index, ["", ""])
            if tc.function and tc.function.name:
                entry[0] = tc.function.name
            if tc.function and tc.function.arguments:
                entry[1] += tc.function.arguments

    prev_text, prev_ids = "", []
    for i in range(0, len(ids), step):
        cur_ids = ids[: i + step]
        cur_text = tokenizer.decode(cur_ids)
        delta = parser.extract_tool_calls_streaming(
            prev_text,
            cur_text,
            cur_text[len(prev_text) :],
            prev_ids,
            cur_ids,
            ids[i : i + step],
            request,
        )
        collect(delta)
        prev_text, prev_ids = cur_text, cur_ids
    collect(parser.finish_streaming())
    return [(calls[k][0], calls[k][1]) for k in sorted(calls)]


def check_all(tokenizer, tools, text, expected: list[dict], validate=True):
    """Assert one-shot and streamed extraction both give `expected` arguments."""
    request = make_request(tools)
    once = extract(tokenizer, request, text)
    assert [json.loads(a) for _, a in once] == expected
    for step in (1, 2, 3, 4):
        streamed = stream(tokenizer, request, text, step)
        assert [n for n, _ in streamed] == [n for n, _ in once], step
        for (_, s_args), (_, o_args) in zip(streamed, once):
            assert json.loads(s_args) == json.loads(o_args), step
            assert s_args == o_args, step
    if validate:
        for (name, args), want in zip(once, expected):
            Draft7Validator(tools[name]).validate(json.loads(args))
            assert json.loads(args) == want


def obj(properties: dict, **extra) -> dict:
    return {"type": "object", "properties": properties, **extra}


# ---------------------------------------------------------------------------
# P1: $ref / allOf
# ---------------------------------------------------------------------------

REF_TOOLS = {
    "configure": {
        "type": "object",
        "properties": {
            "cfg": {"$ref": "#/$defs/Cfg"},
            "n": {"$ref": "#/$defs/Count"},
        },
        "$defs": {
            "Cfg": obj({"a": {"type": "integer"}, "b": {"type": "boolean"}}),
            "Count": {"type": "integer"},
        },
    }
}
REF_TEXT = call_text("configure", {"cfg": '{"a": 3, "b": true}', "n": "5"})
REF_EXPECTED = [{"cfg": {"a": 3, "b": True}, "n": 5}]

TOP_REF_TOOLS = {
    "run": {
        "$ref": "#/definitions/Args",
        "definitions": {
            "Args": obj(
                {
                    "count": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                }
            )
        },
    }
}
TOP_REF_TEXT = call_text("run", {"count": "7", "tags": '["x", "y"]'})
TOP_REF_EXPECTED = [{"count": 7, "tags": ["x", "y"]}]

ALLOF_TOOLS = {
    "update": {
        "allOf": [
            {"$ref": "#/$defs/Base"},
            obj({"force": {"type": "boolean"}}, required=["force"]),
        ],
        "$defs": {"Base": obj({"id": {"type": "integer"}}, required=["id"])},
    }
}
ALLOF_TEXT = call_text("update", {"id": "42", "force": "true"})
ALLOF_EXPECTED = [{"id": 42, "force": True}]

NESTED_TOOLS = {
    "bulk": {
        "type": "object",
        "properties": {"rows": {"$ref": "#/$defs/Rows"}},
        "$defs": {
            "Rows": {"type": "array", "items": {"$ref": "#/$defs/Row"}},
            "Row": obj({"id": {"type": "integer"}}),
        },
    }
}
NESTED_TEXT = call_text("bulk", {"rows": '[{"id": 1}, {"id": 2}]'})
NESTED_EXPECTED = [{"rows": [{"id": 1}, {"id": 2}]}]

CYCLE_TOOLS = {
    "walk": {
        "type": "object",
        "properties": {
            "node": {"$ref": "#/$defs/Node"},
            "loop": {"$ref": "#/$defs/Loop"},
            "k": {"type": "integer"},
        },
        "$defs": {
            "Node": obj({"next": {"$ref": "#/$defs/Node"}, "v": {"type": "integer"}}),
            "Loop": {"$ref": "#/$defs/Loop"},
        },
    }
}
CYCLE_TEXT = call_text(
    "walk", {"node": '{"v": 1, "next": {"v": 2}}', "loop": "abc", "k": "9"}
)
CYCLE_EXPECTED = [{"node": {"v": 1, "next": {"v": 2}}, "loop": "abc", "k": 9}]

P1_CASES = {
    "ref": (REF_TOOLS, REF_TEXT, REF_EXPECTED),
    "top_ref": (TOP_REF_TOOLS, TOP_REF_TEXT, TOP_REF_EXPECTED),
    "allof": (ALLOF_TOOLS, ALLOF_TEXT, ALLOF_EXPECTED),
    "nested": (NESTED_TOOLS, NESTED_TEXT, NESTED_EXPECTED),
    "cycle": (CYCLE_TOOLS, CYCLE_TEXT, CYCLE_EXPECTED),
}


@pytest.mark.parametrize("case", sorted(P1_CASES))
def test_p1_ref_allof_fixed(tokenizer, case):
    tools, text, expected = P1_CASES[case]
    check_all(tokenizer, tools, text, expected, validate=case != "cycle")


@pytest.mark.parametrize("case", sorted(set(P1_CASES) - {"cycle"}))
def test_p1_old_behavior_is_wrong(tokenizer, case):
    # Without the fix the typed values stay strings: the test sees the bug.
    uninstall_fixes()
    tools, text, expected = P1_CASES[case]
    request = make_request(tools)
    ((_, args),) = extract(tokenizer, request, text)
    assert json.loads(args) != expected[0]
    assert any(isinstance(v, str) for v in json.loads(args).values())


def test_p1_old_behavior_cycle_has_no_type_info(tokenizer):
    uninstall_fixes()
    request = make_request(CYCLE_TOOLS)
    ((_, args),) = extract(tokenizer, request, CYCLE_TEXT)
    assert json.loads(args)["node"] == '{"v": 1, "next": {"v": 2}}'


def test_p1_deref_unit_does_not_mutate_and_resolves_everything():
    root = {
        "type": "object",
        "properties": {
            "a": {"$ref": "#/$defs/A~1B", "description": "sibling wins"},
            "arr": {"type": "array", "items": [{"$ref": "#/definitions/I"}]},
            "pre": {"type": "array", "prefixItems": [{"$ref": "#/definitions/I"}]},
            "map": {
                "type": "object",
                "additionalProperties": {"$ref": "#/definitions/I"},
            },
            "u": {"anyOf": [{"$ref": "#/definitions/I"}, {"type": "null"}]},
            "m": {
                "allOf": [
                    obj({"x": {"type": "integer"}}, required=["x"]),
                    obj({"x": {"description": "d"}, "y": {"type": "string"}}),
                    {"type": "string", "required": ["y", "x"]},
                ]
            },
        },
        "$defs": {"A/B": {"type": "integer", "description": "target"}},
        "definitions": {"I": {"type": "number"}},
    }
    before = copy.deepcopy(root)
    props = fixes._deref_schema(root, root, ())["properties"]
    assert root == before
    assert props["a"] == {"type": "integer", "description": "sibling wins"}
    assert props["arr"]["items"] == [{"type": "number"}]
    assert props["pre"]["prefixItems"] == [{"type": "number"}]
    assert props["map"]["additionalProperties"] == {"type": "number"}
    assert props["u"]["anyOf"] == [{"type": "number"}, {"type": "null"}]
    merged = props["m"]
    assert merged["type"] == "object"
    assert merged["required"] == ["x", "y"]
    assert merged["properties"]["x"] == {"type": "integer", "description": "d"}
    assert merged["properties"]["y"] == {"type": "string"}


def test_p1_deref_unit_unknown_ref_and_cycle():
    root = {"properties": {"a": {"$ref": "#/nope", "type": "string"}}}
    assert fixes._deref_schema(root, root, ())["properties"]["a"] == {"type": "string"}
    root = {"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "#/$defs/A", "type": "x"}}}
    assert fixes._deref_schema(root, root, ())["type"] == "x"


def test_p1_schema_without_refs_uses_original(tokenizer):
    from vllm.parser.engine import parser_engine

    request = make_request({"f": obj({"a": {"type": "integer"}})})
    assert parser_engine.find_tool_properties(request.tools, "f") == {
        "a": {"type": "integer"}
    }
    assert parser_engine.find_tool_properties(request.tools, "missing") == {}
    assert parser_engine.find_tool_properties(None, "f") == {}


# ---------------------------------------------------------------------------
# P3: unclosed last parameter
# ---------------------------------------------------------------------------

CONTENT_TOOLS = {
    "write": obj({"content": {"type": "string"}, "mode": {"type": "string"}})
}


UNCLOSED_TEXT = (
    "<tool_call>\n<function=write>\n<parameter=content>\nabc\n</function>\n</tool_call>"
)


def test_p3_unclosed_last_parameter_kept(tokenizer):
    text = UNCLOSED_TEXT
    check_all(tokenizer, CONTENT_TOOLS, text, [{"content": "abc"}])


def test_p3_unclosed_after_closed_parameter(tokenizer):
    text = (
        "<tool_call>\n<function=write>\n<parameter=mode>\nw\n</parameter>\n"
        "<parameter=content>\nline1\nline2\n</function>\n</tool_call>"
    )
    check_all(
        tokenizer, CONTENT_TOOLS, text, [{"mode": "w", "content": "line1\nline2"}]
    )


def test_p3_missing_function_end_too(tokenizer):
    text = "<tool_call>\n<function=write>\n<parameter=content>\nabc\n</tool_call>"
    request = make_request(CONTENT_TOOLS)
    ((_, args),) = extract(tokenizer, request, text)
    assert json.loads(args) == {"content": "abc"}


def test_p3_old_behavior_drops_value(tokenizer):
    uninstall_fixes()
    text = UNCLOSED_TEXT
    request = make_request(CONTENT_TOOLS)
    ((_, args),) = extract(tokenizer, request, text)
    assert json.loads(args) == {}


@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_p3_truncated_value_keeps_partial_and_valid_json(tokenizer, step):
    text = "<tool_call>\n<function=write>\n<parameter=content>\nabc def ghi"
    request = make_request(CONTENT_TOOLS)
    ((_, once),) = extract(tokenizer, request, text)
    assert json.loads(once) == {"content": "abc def ghi"}
    ((_, streamed),) = stream(tokenizer, request, text, step)
    assert json.loads(streamed) == {"content": "abc def ghi"}
    assert streamed == once


# ---------------------------------------------------------------------------
# P4: string unions
# ---------------------------------------------------------------------------

UNION_TOOLS = {
    "lookup": obj(
        {
            "key": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            "opt": {"type": ["string", "null"]},
            "zip": {"type": "string"},
        }
    )
}


def test_p4_string_unions(tokenizer):
    text = call_text("lookup", {"key": "007", "opt": "null", "zip": "02139"})
    check_all(
        tokenizer, UNION_TOOLS, text, [{"key": "007", "opt": None, "zip": "02139"}]
    )
    text = call_text("lookup", {"key": "7", "opt": "x", "zip": "1"})
    check_all(tokenizer, UNION_TOOLS, text, [{"key": 7, "opt": "x", "zip": "1"}])


@pytest.mark.parametrize(
    "kind, raw",
    [
        ("object", '{"a":1}'),
        ("object", '{"a": 1}'),
        ("array", "[1,2]"),
        ("array", "[1, 2]"),
    ],
)
def test_p4_object_array_unions_still_convert(tokenizer, kind, raw):
    # Only scalar conversions need the lossless check.
    from vllm.parser.engine import parser_engine

    assert parser_engine.coerce_to_schema_type(raw, [kind, "string"]) == json.loads(raw)
    tools = {"u": obj({"v": {"anyOf": [{"type": kind}, {"type": "string"}]}})}
    check_all(tokenizer, tools, call_text("u", {"v": raw}), [{"v": json.loads(raw)}])


def test_p4_old_behavior_loses_leading_zeros(tokenizer):
    uninstall_fixes()
    request = make_request(UNION_TOOLS)
    text = call_text("lookup", {"key": "007", "opt": "null", "zip": "02139"})
    ((_, args),) = extract(tokenizer, request, text)
    assert json.loads(args)["key"] == 7


def test_p4_coerce_unit():
    from vllm.parser.engine import parser_engine

    coerce = parser_engine.coerce_to_schema_type
    assert coerce("007", ["string", "integer"]) == "007"
    assert coerce("7", ["string", "integer"]) == 7
    assert coerce(" 7 ", ["string", "integer"]) == 7
    assert coerce("1.0", ["string", "number"]) == "1.0"
    assert coerce("2.5", ["string", "number"]) == 2.5
    assert coerce("1", ["string", "boolean"]) == "1"
    assert coerce("true", ["string", "boolean"]) is True
    assert coerce("null", ["string", "null"]) is None
    assert coerce("007", "integer") == 7
    assert coerce("[1,2]", ["string", "array"]) == [1, 2]
    assert coerce("[1, 2]", ["string", "array"]) == [1, 2]


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------

PLAIN_TOOLS = {
    "plain": obj(
        {
            "s": {"type": "string"},
            "i": {"type": "integer"},
            "x": {"type": "number"},
            "b": {"type": "boolean"},
            "e": {"enum": ["red", "green"]},
            "arr": {"type": "array", "items": {"type": "integer"}},
            "o": obj({"k": {"type": "string"}}),
            "text": {"type": "string"},
        }
    ),
    "other": obj({"q": {"type": "string"}, "n": {"type": "integer"}}),
}


def test_regression_plain_types(tokenizer):
    text = call_text(
        "plain",
        {
            "s": "hello",
            "i": "12",
            "x": "2.5",
            "b": "false",
            "e": "red",
            "arr": "[1, 2, 3]",
            "o": '{"k": "v"}',
            "text": "line1\nline2",
        },
    )
    expected = [
        {
            "s": "hello",
            "i": 12,
            "x": 2.5,
            "b": False,
            "e": "red",
            "arr": [1, 2, 3],
            "o": {"k": "v"},
            "text": "line1\nline2",
        }
    ]
    check_all(tokenizer, PLAIN_TOOLS, text, expected)


def test_regression_two_tool_calls(tokenizer):
    text = (
        call_text("plain", {"s": "a", "i": "1"})
        + "\n"
        + call_text("other", {"q": "z", "n": "2"})
    )
    check_all(tokenizer, PLAIN_TOOLS, text, [{"s": "a", "i": 1}, {"q": "z", "n": 2}])


@pytest.mark.parametrize("value", ["2024", "true", "[1, 2]", "null", "1.5"])
def test_regression_json_looking_strings_stay_strings(tokenizer, value):
    text = call_text("plain", {"s": value})
    check_all(tokenizer, PLAIN_TOOLS, text, [{"s": value}])


# ---------------------------------------------------------------------------
# P2: finish_reason
# ---------------------------------------------------------------------------


def out(index, finish_reason):
    return SimpleNamespace(index=index, finish_reason=finish_reason)


def res(*outputs):
    return SimpleNamespace(outputs=list(outputs))


async def agen(items):
    for item in items:
        yield item


def sse(obj_) -> str:
    return "data: " + json.dumps(obj_, separators=(",", ":")) + "\n\n"


def chunk(finish_reasons: dict[int, str | None]) -> str:
    return sse(
        {
            "id": "c",
            "choices": [
                {"index": i, "delta": {}, "finish_reason": r}
                for i, r in finish_reasons.items()
            ],
        }
    )


@pytest.fixture
def serving(monkeypatch):
    """Install fake generators on OpenAIServingChat, then wrap them."""
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    async def fake_stream(self, request, result_generator, request_id="r"):
        async for r in result_generator:
            for o in r.outputs:
                reason = "tool_calls" if o.finish_reason is not None else None
                yield chunk({o.index: reason})
        yield "data: [DONE]\n\n"

    async def fake_full(self, request, result_generator, request_id="r"):
        indexes = set()
        async for r in result_generator:
            indexes.update(o.index for o in r.outputs)
        if request == "error":
            return SimpleNamespace(error="boom")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(index=i, finish_reason="tool_calls")
                for i in sorted(indexes)
            ]
        )

    for name in ("chat_completion_stream_generator", "chat_completion_full_generator"):
        original = getattr(OpenAIServingChat, f"_tt_original_{name}")
        setattr(OpenAIServingChat, name, original)
        delattr(OpenAIServingChat, f"_tt_original_{name}")
    monkeypatch.setattr(
        OpenAIServingChat, "chat_completion_stream_generator", fake_stream
    )
    monkeypatch.setattr(OpenAIServingChat, "chat_completion_full_generator", fake_full)
    fixes.install_tool_call_fixes()
    return OpenAIServingChat


def run_stream(cls, results, **kwargs):
    async def go():
        gen = cls.chat_completion_stream_generator(None, "req", agen(results), **kwargs)
        return [s async for s in gen]

    return asyncio.run(go())


@pytest.mark.parametrize(
    "reason, expected",
    [("length", "length"), ("stop", "tool_calls"), (None, "tool_calls")],
)
def test_p2_stream_finish_reason(serving, reason, expected):
    results = [res(out(0, None)), res(out(0, reason))]
    chunks = run_stream(serving, results)
    # The fake emits tool_calls for any output with a finish_reason, so with
    # reason None both chunks stay untouched.
    final = json.loads(chunks[-2][len("data: ") :])
    if reason is None:
        assert chunks[-2] == chunk({0: None})
    else:
        assert final["choices"][0]["finish_reason"] == expected
    assert chunks[-1] == "data: [DONE]\n\n"


def test_p2_stream_keyword_result_generator_and_n2(serving):
    async def go():
        gen = serving.chat_completion_stream_generator(
            None,
            "req",
            result_generator=agen([res(out(0, "stop"), out(1, "length"))]),
        )
        return [s async for s in gen]

    chunks = asyncio.run(go())
    reasons = {
        json.loads(c[6:])["choices"][0]["index"]: json.loads(c[6:])["choices"][0][
            "finish_reason"
        ]
        for c in chunks[:-1]
    }
    assert reasons == {0: "tool_calls", 1: "length"}


def test_p2_stream_other_strings_are_byte_identical(serving):
    results = [res(out(0, None)), res(out(0, None))]
    chunks = run_stream(serving, results)
    assert chunks == [chunk({0: None}), chunk({0: None}), "data: [DONE]\n\n"]


def test_p2_stream_rewrite_helper_is_selective():
    rec = {0: "length"}
    plain = sse(
        {
            "choices": [
                {"index": 0, "delta": {"content": "tool_calls"}, "finish_reason": None}
            ]
        }
    )
    assert fixes._fix_sse_chunk(plain, rec) == plain
    assert fixes._fix_sse_chunk("data: [DONE]\n\n", rec) == "data: [DONE]\n\n"
    stop = chunk({0: "stop"})
    assert fixes._fix_sse_chunk(stop, rec) == stop
    fixed = fixes._fix_sse_chunk(chunk({0: "tool_calls"}), rec)
    assert fixed == chunk({0: "length"})
    unicode_chunk = sse(
        {
            "choices": [
                {"index": 0, "delta": {"content": "é"}, "finish_reason": "tool_calls"}
            ]
        }
    )
    assert "é" in fixes._fix_sse_chunk(unicode_chunk, rec)


@pytest.mark.parametrize(
    "reason, expected",
    [("length", "length"), ("stop", "tool_calls"), (None, "tool_calls")],
)
def test_p2_full_response_is_fixed(serving, reason, expected):
    async def go():
        return await serving.chat_completion_full_generator(
            None, "req", agen([res(out(0, reason))])
        )

    response = asyncio.run(go())
    assert response.choices[0].finish_reason == expected


def test_p2_full_error_response_and_n2(serving):
    async def go(request):
        return await serving.chat_completion_full_generator(
            None, request, agen([res(out(0, "stop"), out(1, "length"))])
        )

    response = asyncio.run(go("req"))
    assert [c.finish_reason for c in response.choices] == ["tool_calls", "length"]
    assert asyncio.run(go("error")).error == "boom"


def test_p2_aclose_closes_original_and_engine_generators(serving):
    closed = []

    async def source():
        try:
            yield res(out(0, None))
            yield res(out(0, "stop"))
        finally:
            closed.append("source")

    async def go():
        gen = serving.chat_completion_stream_generator(None, "req", source())
        first = await gen.__anext__()
        assert first == chunk({0: None})
        await gen.aclose()
        # The source finally must have run before aclose() returned.
        assert closed == ["source"]

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def patched_targets():
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.parser import qwen3
    from vllm.parser.engine import parser_engine, streaming_parser_engine

    engine = streaming_parser_engine.StreamingParserEngine
    return (
        parser_engine.find_tool_properties,
        parser_engine.coerce_to_schema_type,
        qwen3._qwen3_arg_converter,
        engine._on_terminal,
        engine._emit_for_state,
        OpenAIServingChat.chat_completion_stream_generator,
        OpenAIServingChat.chat_completion_full_generator,
    )


def test_install_is_idempotent():
    first = patched_targets()
    fixes.install_tool_call_fixes()
    fixes.install_tool_call_fixes()
    assert patched_targets() == first


def test_install_raises_when_target_is_missing(monkeypatch):
    from vllm.parser.engine import parser_engine

    monkeypatch.delattr(parser_engine, "_tt_original_coerce_to_schema_type")
    monkeypatch.delattr(parser_engine, "coerce_to_schema_type")
    with pytest.raises(AttributeError, match="coerce_to_schema_type"):
        fixes.install_tool_call_fixes()


def test_register_installs_the_fixes(monkeypatch):
    import vllm_tt_plugin.model_registry as model_registry

    calls = []
    monkeypatch.setitem(__import__("sys").modules, "ttnn", SimpleNamespace())
    for name in (
        "register_tt_models",
        "_install_diffusion_gemma_architecture_patch",
        "_install_tt_async_spec_method_patch",
        "_install_torch_accelerator_cleanup_patch",
    ):
        monkeypatch.setattr(model_registry, name, lambda *a, **k: None)
    monkeypatch.setattr(
        model_registry, "_should_pre_register_tt_test_models_from_cli", lambda: False
    )
    monkeypatch.setattr(
        model_registry, "install_tool_call_fixes", lambda: calls.append(1)
    )
    model_registry.register_tt_models_from_plugin()
    assert calls == [1]
