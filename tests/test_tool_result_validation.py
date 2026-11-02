# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests of the tool-result id validation and of the untyped-scalar typing."""

import asyncio
import json
from http import HTTPStatus
from types import SimpleNamespace

import pytest
import vllm.config  # noqa: F401  # finish vLLM init before the plugin package

from vllm_tt_plugin import tool_arg_typing, tool_result_validation as trv


def _call(cid, name="f"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _asst(*ids):
    return {"role": "assistant", "content": None, "tool_calls": [_call(i) for i in ids]}


def _tool(cid, content="r"):
    msg = {"role": "tool", "content": content}
    if cid is not None:
        msg["tool_call_id"] = cid
    return msg


USER = {"role": "user", "content": "hi"}


def _err(messages):
    return trv.find_tool_result_error(messages)


# ---- accepted flows ----------------------------------------------------------------


def test_plain_conversation_without_tools():
    assert _err([USER, {"role": "assistant", "content": "x"}, USER]) is None


def test_single_call_result():
    assert _err([USER, _asst("a"), _tool("a")]) is None


def test_parallel_results_any_order():
    assert _err([USER, _asst("a", "b", "c"), _tool("c"), _tool("a"), _tool("b")]) is None


def test_partial_results_are_accepted():
    assert _err([USER, _asst("a", "b"), _tool("b")]) is None
    assert _err([USER, _asst("a", "b"), _tool("b"), {"role": "assistant", "content": "ok"}, USER]) is None


def test_several_rounds():
    msgs = [USER, _asst("a"), _tool("a"), _asst("b", "c"), _tool("b"), _tool("c"), USER, _asst("d"), _tool("d")]
    assert _err(msgs) is None


def test_result_of_an_earlier_turn_is_accepted():
    # a late result for round 1 after round 2 was declared
    assert _err([USER, _asst("a"), _asst("b"), _tool("b"), _tool("a")]) is None


def test_a_client_that_restarts_ids_each_round():
    msgs = [USER, _asst("call_0"), _tool("call_0"), _asst("call_0"), _tool("call_0")]
    assert _err(msgs) is None


def test_message_objects_and_tuples():
    msgs = (
        SimpleNamespace(role="user", content="x"),
        SimpleNamespace(role="assistant", tool_calls=[SimpleNamespace(id="a")]),
        SimpleNamespace(role="tool", tool_call_id="a"),
    )
    assert _err(msgs) is None


def test_non_list_messages_are_left_alone():
    assert _err(None) is None
    assert _err("x") is None


# ---- refused flows -----------------------------------------------------------------


def test_unknown_id():
    message, param = _err([USER, _asst("a"), _tool("zzz")])
    assert "zzz" in message and "does not match" in message
    assert param == "messages[2].tool_call_id"


def test_result_before_the_call():
    assert _err([USER, _tool("a"), _asst("a")]) is not None


def test_no_assistant_message_at_all():
    assert _err([USER, _tool("a")]) is not None


@pytest.mark.parametrize("cid", [None, "", "  "])
def test_missing_or_empty_id(cid):
    message, param = _err([USER, _asst("a"), _tool(cid)])
    assert "non-empty tool_call_id" in message
    assert param == "messages[2].tool_call_id"


def test_non_string_id():
    assert _err([USER, _asst("a"), {"role": "tool", "content": "r", "tool_call_id": 5}]) is not None


def test_duplicate_result():
    message, _ = _err([USER, _asst("a", "b"), _tool("a"), _tool("a")])
    assert "duplicate" in message


def test_duplicate_across_user_turn():
    assert _err([USER, _asst("a"), _tool("a"), USER, _tool("a")]) is not None


def test_assistant_call_without_an_id_cannot_be_answered():
    msgs = [USER, {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "f"}}]}, _tool("a")]
    assert _err(msgs) is not None


def test_first_error_wins():
    message, param = _err([USER, _asst("a"), _tool("x"), _tool(None)])
    assert "'x'" in message and param == "messages[2].tool_call_id"


# ---- the endpoint wrapper ----------------------------------------------------------


class _Owner:
    def __init__(self):
        self.calls = 0

    def create_error_response(self, message, err_type="BadRequestError", status_code=HTTPStatus.BAD_REQUEST, param=None):
        return SimpleNamespace(message=message, err_type=err_type, status_code=status_code, param=param)


async def _endpoint(self, request, raw_request=None):
    self.calls += 1
    return "ok"


def _run(coro):
    return asyncio.run(coro)


def test_wrapper_refuses_before_the_engine_call():
    owner = _Owner()
    wrapped = trv.wrap_endpoint(_endpoint)
    out = _run(wrapped(owner, SimpleNamespace(messages=[USER, _asst("a"), _tool("q")], stream=True)))
    assert out.status_code == HTTPStatus.BAD_REQUEST and out.err_type == "BadRequestError"
    assert owner.calls == 0


def test_wrapper_passes_a_valid_request_streaming_or_not():
    owner = _Owner()
    wrapped = trv.wrap_endpoint(_endpoint)
    for stream in (True, False):
        assert _run(wrapped(owner, SimpleNamespace(messages=[USER, _asst("a"), _tool("a")], stream=stream))) == "ok"
    assert owner.calls == 2


def test_wrapper_is_installed_once():
    once = trv.wrap_endpoint(_endpoint)
    assert trv.wrap_endpoint(once) is once


def test_env_switch(monkeypatch):
    monkeypatch.setenv("VLLM_TT_TOOL_RESULT_VALIDATE", "0")
    owner = _Owner()
    out = _run(trv.wrap_endpoint(_endpoint)(owner, SimpleNamespace(messages=[USER, _tool("q")])))
    assert out == "ok"


def test_install_patches_the_chat_endpoint():
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    trv.install()
    assert getattr(OpenAIServingChat.create_chat_completion, "_tt_tool_result_wrapped", False)


# ---- untyped scalar typing ---------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    from vllm.parser.engine.parser_engine import ParserEngine

    tool_arg_typing.install()
    return ParserEngine


def _convert(raw):
    from vllm.parser.deepseek_v41 import deepseek_v41_config

    cfg = deepseek_v41_config(thinking=False)
    return json.loads(cfg.arg_converter(raw, False))


def _param(name, value, is_str):
    return f'<｜DSML｜ parameter name="{name}" string="{is_str}">{value}</｜DSML｜ parameter>\n'


UNTYPED = {"n": {}, "flag": {"description": "a flag"}, "name": {}, "nothing": {}}


def test_untyped_decoded_scalars_keep_their_json_type(engine):
    args = _convert(_param("n", "42", "false") + _param("flag", "true", "false") + _param("name", "abc", "true") + _param("nothing", "null", "false"))
    assert args == {"n": 42, "flag": True, "name": "abc", "nothing": None}
    out, changed = engine._coerce_dict(args, UNTYPED)
    assert out == {"n": 42, "flag": True, "name": "abc", "nothing": None}
    assert type(out["n"]) is int and out["flag"] is True and out["nothing"] is None


def test_untyped_string_flag_true_stays_a_string(engine):
    args = _convert(_param("n", "42", "true") + _param("flag", "true", "true"))
    out, _ = engine._coerce_dict(args, UNTYPED)
    assert out == {"n": "42", "flag": "true"}


def test_untyped_floats_and_containers(engine):
    out, _ = engine._coerce_dict({"n": 1.5, "a": [1, "x"], "o": {"k": 2}}, {"n": {}, "a": {}, "o": {}})
    assert out == {"n": 1.5, "a": [1, "x"], "o": {"k": 2}}


@pytest.mark.parametrize(
    ("value", "schema", "expected"),
    [
        (42, {"type": "string"}, "42"),
        (True, {"type": "string"}, "true"),
        ("42", {"type": "integer"}, 42),
        ("3.5", {"type": "number"}, 3.5),
        ("true", {"type": "boolean"}, True),
        (42, {"type": "integer"}, 42),
        (42, {"enum": ["a", "b"]}, "42"),
        (42, {"anyOf": [{"type": "string"}]}, "42"),
        ("[1, 2]", {"type": "array"}, [1, 2]),
    ],
)
def test_typed_schemas_are_unchanged(engine, value, schema, expected):
    out, _ = engine._coerce_value(value, schema)
    assert out == expected and type(out) is type(expected)


def test_nested_untyped_inside_typed_object(engine):
    schema = {"type": "object", "properties": {"inner": {}}}
    out, _ = engine._coerce_value({"inner": 7}, schema)
    assert out == {"inner": 7}


def test_install_is_idempotent_and_switchable(engine, monkeypatch):
    first = engine.__dict__["_coerce_value"]
    tool_arg_typing.install()
    assert engine.__dict__["_coerce_value"].__func__ is first.__func__
    monkeypatch.setenv("VLLM_TT_UNTYPED_SCALARS", "0")
    out, _ = engine._coerce_value(42, {})
    assert out == "42"  # stock vLLM behaviour
