# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The MiMo structural-tag grammar in the patched vLLM (until-0.32, upstream
#58019): xgrammar's builtin ``mimo`` tag, ``parallel_tool_calls=false``, and
``--tool-strict-level``; and the hard xgrammar>=0.2.8 requirement."""

import importlib
import importlib.metadata
import json
import sys
from unittest.mock import MagicMock

import pytest

from tests.vllm_patches import requires_patched_vllm

pytestmark = requires_patched_vllm

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": ["number", "null"]},
                "limit": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            },
            "required": ["path", "offset", "limit"],
            "additionalProperties": False,
        },
    },
}
LOOSE_TOOL = {
    "type": "function",
    "function": {**READ_TOOL["function"], "name": "peek", "strict": False},
}


def mimo_call(name: str, args: dict) -> str:
    """MiMo's chat template: string values raw, everything else JSON."""
    params = "".join(
        f"<parameter={k}>{v if isinstance(v, str) else json.dumps(v)}</parameter>"
        for k, v in args.items()
    )
    return f"<tool_call><function={name}>{params}</function></tool_call>"


OK = mimo_call("read", {"path": "a b.md", "offset": None, "limit": 50})


def grammar_for(tools, tool_choice="auto", strict_level=None, **kwargs):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.tool_parsers import ToolParserManager
    from vllm.tool_parsers.tool_strict_level import ToolStrictLevel
    from xgrammar import Grammar

    request = ChatCompletionRequest(
        model="mimo",
        messages=[{"role": "user", "content": "x"}],
        tools=tools,
        tool_choice=tool_choice,
        **kwargs,
    )
    parser = ToolParserManager.get_tool_parser("mimo")(MagicMock(), request.tools)
    level = ToolStrictLevel.from_name(strict_level or "auto")
    tag = parser.get_structural_tag(request, strict_level=level)
    if tag is None:
        return None
    return Grammar.from_structural_tag(tag.model_dump_json())


def accepts(grammar, text: str) -> bool:
    from xgrammar.testing import _is_grammar_accept_string

    return _is_grammar_accept_string(grammar, text)


def test_mimo_uses_the_xgrammar_mimo_tag():
    from vllm.tool_parsers import ToolParserManager

    assert ToolParserManager.get_tool_parser("mimo").structural_tag_model == "mimo"


def test_auto_with_a_strict_tool_constrains_mimo_calls():
    g = grammar_for([READ_TOOL])
    assert accepts(g, OK)
    assert accepts(g, "Plain answer, no tool.")
    assert accepts(g, OK + OK)
    bad = {
        "string for number": OK.replace(">50<", '>"50"<'),
        "omitted required": OK.replace("<parameter=offset>null</parameter>", ""),
        "unknown key": OK.replace("<parameter=limit>", "<parameter=lim>"),
        "unknown function": OK.replace("function=read", "function=write"),
        "qwen3-coder newlines": OK.replace(
            "<tool_call><function=read>", "<tool_call>\n<function=read>\n"
        ),
    }
    for why, text in bad.items():
        assert not accepts(g, text), why


@pytest.mark.parametrize("choice", ["required", "named"])
def test_required_and_named_force_a_compact_mimo_call(choice):
    tool_choice = (
        {"type": "function", "function": {"name": "read"}}
        if choice == "named"
        else "required"
    )
    g = grammar_for([READ_TOOL], tool_choice)
    assert accepts(g, OK)
    assert not accepts(g, "text only")


def test_parallel_tool_calls_false_allows_one_call():
    g = grammar_for([READ_TOOL], parallel_tool_calls=False)
    assert accepts(g, OK)
    assert not accepts(g, OK + OK)


def test_tool_strict_level_controls_non_strict_tools():
    loose = mimo_call("peek", {"path": 1, "offset": "x", "limit": None})
    # auto (0.31's default): a non-strict tool alone gets no grammar.
    assert grammar_for([LOOSE_TOOL]) is None
    # function: the envelope is constrained, the arguments are not.
    g = grammar_for([LOOSE_TOOL], strict_level="function")
    assert accepts(g, loose)
    assert not accepts(g, loose.replace("function=peek", "function=nope"))
    # parameter: the arguments follow the schema as if strict were set.
    g = grammar_for([LOOSE_TOOL], strict_level="parameter")
    assert not accepts(g, loose)
    assert accepts(g, OK.replace("function=read", "function=peek"))


def test_an_older_xgrammar_is_an_error_not_a_qwen3_coder_fallback(monkeypatch):
    real_version = importlib.metadata.version

    def version(name):
        return "0.2.7" if name == "xgrammar" else real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.delitem(sys.modules, "vllm.tool_parsers.mimo_tool_parser")
    with pytest.raises(ImportError, match=r"xgrammar>=0\.2\.8"):
        importlib.import_module("vllm.tool_parsers.mimo_tool_parser")
