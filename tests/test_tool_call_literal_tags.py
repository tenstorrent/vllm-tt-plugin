# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Host tests for P7: literal ``</parameter>`` / ``</function>`` in a value.

A coding agent that writes XML, HTML or code about tool calls puts these
strings inside an argument value. Upstream cuts the value at the first one,
in one shot and in streaming. The tests drive the real vLLM ``qwen3_coder``
engine with the Qwen3.8 tokenizer: one-shot, token-streamed, and
character-streamed with a delta boundary at every position of the tags.
"""

import json
import os
import random

import pytest

import vllm_tt_plugin.tool_call_fixes as fixes
from tests.test_tool_call_fixes import (
    TOKENIZER_PATH,
    check_all,
    extract,
    make_request,
    obj,
    stream,
    uninstall_fixes,
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(TOKENIZER_PATH), reason=f"{TOKENIZER_PATH} is not available"
)


@pytest.fixture(autouse=True)
def installed_fixes():
    # tests/conftest.py restores the patched attributes after every test.
    fixes.install_tool_call_fixes()


@pytest.fixture(scope="module")
def tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(TOKENIZER_PATH)


@pytest.fixture(scope="module")
def parser_cls():
    from vllm.parser import ParserManager

    return ParserManager.get_parser(
        tool_parser_name="qwen3_coder",
        reasoning_parser_name="qwen3",
        enable_auto_tools=True,
        model_name="Qwen3.8",
        is_harmony=False,
    )


TOOLS = {
    "write_file": obj({"path": {"type": "string"}, "content": {"type": "string"}}),
    "edit": obj(
        {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "line": {"type": "integer"},
            "ratio": {"type": "number"},
            "force": {"type": "boolean"},
            "opts": obj({"tag": {"type": "string"}, "n": {"type": "integer"}}),
            "tags": {"type": "array", "items": {"type": "string"}},
        }
    ),
    "noschema": {"type": "object", "properties": {}},
}


def call_text(name: str, params: dict[str, str]) -> str:
    """The template's rendering of one call."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


LITERAL_VALUES = [
    "</parameter>",
    "</function>",
    "a</parameter>b",
    "a</function>b",
    "<div>\n</parameter>\n</div>",
    "/* </function> */\nint x;",
    "x </parameter> </function> y",
    "a <parameter=q>v</parameter> b",
    "inline <parameter=q> text",
    "<p>done</parameter>",
    "ends with </function>",
    "</parameter></parameter>\n</parameter>x",
    "a\n</parameter>\n</function>\nmore",
    "a\n</parameter>\nmore\n</function>\nz",
    "</function>\n\nstill the value",
    "< /parameter > and < / function >",
    "<tool_call> text and </tool_call> text",
    "café 漢字 😀 </parameter> ü",
    "</param",
    "</func",
]

CODE = (
    "def emit(tag):\n"
    '    """Render <parameter=x> ... </parameter> as XML."""\n'
    '    out = f"<parameter={tag}>\\n"\n'
    '    out += "</parameter>\\n</function>"\n'
    "    if a < b and b > a:\n"
    '        return out + "\\t</function>"\n'
    "    return out\n"
)


@pytest.mark.parametrize("value", LITERAL_VALUES)
def test_literal_closer_in_string_value(tokenizer, value):
    text = call_text("write_file", {"path": "f.txt", "content": value})
    check_all(tokenizer, TOOLS, text, [{"path": "f.txt", "content": value}])


@pytest.mark.parametrize("value", LITERAL_VALUES[:8])
def test_literal_closer_in_first_and_middle_value(tokenizer, value):
    text = call_text("edit", {"old": value, "new": value + "2", "path": "p"})
    check_all(tokenizer, TOOLS, text, [{"old": value, "new": value + "2", "path": "p"}])


def test_multiline_code_value(tokenizer):
    text = call_text("write_file", {"path": "gen.py", "content": CODE})
    check_all(tokenizer, TOOLS, text, [{"path": "gen.py", "content": CODE}])


def test_json_typed_params_next_to_literal_tags(tokenizer):
    params = {
        "path": "a.xml",
        "old": "<a></parameter></a>",
        "line": "42",
        "ratio": "0.5",
        "force": "true",
        "opts": '{"tag": "</parameter>", "n": 3}',
        "tags": '["</function>", "x"]',
        "new": "</function>",
    }
    want = {
        "path": "a.xml",
        "old": "<a></parameter></a>",
        "line": 42,
        "ratio": 0.5,
        "force": True,
        "opts": {"tag": "</parameter>", "n": 3},
        "tags": ["</function>", "x"],
        "new": "</function>",
    }
    check_all(tokenizer, TOOLS, call_text("edit", params), [want])


def test_parallel_tool_calls_with_literal_tags(tokenizer):
    first = {"path": "a.html", "content": "<b>\n</parameter>\n</function>\n</b>"}
    second = {"path": "b.c", "content": "/* </function> */"}
    third = {"old": "x</parameter>", "new": "y", "line": "3"}
    text = (
        call_text("write_file", first)
        + "\n"
        + call_text("write_file", second)
        + "\n"
        + call_text("edit", third)
    )
    check_all(tokenizer, TOOLS, text, [first, second, {**third, "line": 3}])


def test_no_schema_tool_keeps_literals(tokenizer):
    params = {"a": "</parameter>", "b": "</function>"}
    request = make_request(TOOLS)
    text = call_text("noschema", params)
    ((_, once),) = extract(tokenizer, request, text)
    assert json.loads(once) == params
    for step in (1, 2, 3):
        ((_, streamed),) = stream(tokenizer, request, text, step)
        assert streamed == once


def test_upstream_truncates(tokenizer):
    """The old code cuts both cases: the bug this patch fixes."""
    uninstall_fixes()
    request = make_request(TOOLS)
    text = call_text("write_file", {"path": "f", "content": "<a>\n</parameter>\n</a>"})
    ((_, args),) = extract(tokenizer, request, text)
    assert json.loads(args)["content"] == "<a>"
    text = call_text("write_file", {"path": "f", "content": "/* </function> */ x"})
    ((_, args),) = extract(tokenizer, request, text)
    # the call ends at "</function>": the value is gone (upstream has no P3)
    assert json.loads(args).get("content", "") == ""


# ---------------------------------------------------------------------------
# Well-formed output is unchanged, malformed output keeps upstream's handling
# ---------------------------------------------------------------------------


def test_dropped_closer_before_line_start_parameter(tokenizer):
    # A "<parameter=" that starts a line still ends a value that has no
    # </parameter> (upstream behavior for a dropped closer).
    text = (
        "<tool_call>\n<function=edit>\n<parameter=old>\na\n"
        "<parameter=new>\nb\n</parameter>\n</function>\n</tool_call>"
    )
    check_all(tokenizer, TOOLS, text, [{"old": "a", "new": "b"}])


def test_missing_function_end(tokenizer):
    text = (
        "<tool_call>\n<function=write_file>\n<parameter=content>\n"
        "x</function>y\n</parameter>\n</tool_call>"
    )
    request = make_request(TOOLS)
    ((_, args),) = extract(tokenizer, request, text)
    assert json.loads(args) == {"content": "x</function>y"}


def test_function_end_without_tool_call_end(tokenizer):
    # </function> then </tool_call> never arrives: the stream end closes it.
    text = (
        "<tool_call>\n<function=write_file>\n<parameter=content>\n"
        "a</function>b\n</parameter>\n</function>\n"
    )
    request = make_request(TOOLS)
    ((_, once),) = extract(tokenizer, request, text)
    assert json.loads(once) == {"content": "a</function>b"}
    for step in (1, 2):
        ((_, streamed),) = stream(tokenizer, request, text, step)
        assert streamed == once


def test_consecutive_calls_without_tool_call_end(tokenizer):
    # "</function>\n<tool_call>" ends the first call, as upstream.
    text = (
        "<tool_call>\n<function=write_file>\n<parameter=content>\na\n</parameter>\n"
        "</function>\n<tool_call>\n<function=edit>\n<parameter=old>\nb\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    request = make_request(TOOLS)
    once = extract(tokenizer, request, text)
    assert [(n, json.loads(a)) for n, a in once] == [
        ("write_file", {"content": "a"}),
        ("edit", {"old": "b"}),
    ]


@pytest.mark.parametrize("seed", range(40))
def test_wellformed_output_matches_upstream(tokenizer, seed):
    """Without a literal tag in a value, P7 changes nothing."""
    rng = random.Random(seed)
    atoms = ["x", "\n", " ", "<b>", "</b>", "a < b", "c > d", "{", "}", '"', "é"]
    calls = []
    for _ in range(rng.randint(1, 3)):
        params = {
            k: "".join(rng.choice(atoms) for _ in range(rng.randint(0, 12)))
            for k in rng.sample(["path", "old", "new"], rng.randint(1, 3))
        }
        calls.append(call_text("edit", params))
    text = "\n".join(calls)
    request = make_request(TOOLS)
    new_once = extract(tokenizer, request, text)
    new_stream = stream(tokenizer, request, text, 1)
    uninstall_fixes()
    old_once = extract(tokenizer, request, text)
    assert new_once == old_once
    assert [a for _, a in new_stream] == [a for _, a in old_once]


# ---------------------------------------------------------------------------
# Character-level streaming: delta boundaries inside every tag
# ---------------------------------------------------------------------------

_SPECIAL = ("<tool_call>", "</tool_call>")


def _pieces_at(text: str, cuts: list[int]) -> list[str]:
    """Split `text` at `cuts`, never inside a special tool-call token."""
    spans = []
    for sp in _SPECIAL:
        start = text.find(sp)
        while start >= 0:
            spans.append((start, start + len(sp)))
            start = text.find(sp, start + 1)
    ok = [c for c in cuts if 0 < c < len(text) and not any(a < c < b for a, b in spans)]
    bounds = [0, *sorted(set(ok)), len(text)]
    return [text[a:b] for a, b in zip(bounds, bounds[1:])]


def _parse_pieces(parser_cls, tokenizer, pieces, thinking=False):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        tools=make_request(TOOLS).tools,
        tool_choice="auto",
    )
    parser = parser_cls(
        tokenizer,
        req.tools,
        chat_template_kwargs={"enable_thinking": thinking},
        model_config=None,
    )
    req = parser.adjust_request(req)
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
    pids = enc("<|im_start|>assistant\n") + enc(
        "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    )
    calls: dict[int, list[str]] = {}
    content = ""
    for j, piece in enumerate(pieces):
        dm = parser.parse_delta(
            delta_text=piece,
            delta_token_ids=enc(piece),
            request=req,
            prompt_token_ids=pids,
            finished=(j == len(pieces) - 1),
        )
        if dm is None:
            continue
        content += dm.content or ""
        for tc in dm.tool_calls or ():
            entry = calls.setdefault(tc.index, ["", ""])
            if tc.function and tc.function.name:
                entry[0] = tc.function.name
            if tc.function and tc.function.arguments:
                entry[1] += tc.function.arguments
    return [tuple(calls[k]) for k in sorted(calls)], content


SPLIT_TEXT = (
    call_text(
        "edit",
        {
            "path": "t.xml",
            "old": "<a>\n</parameter>\n</function>\nb</parameter>c",
            "new": "x </function> y <parameter=z> w",
            "line": "7",
        },
    )
    + "\n"
    + call_text("write_file", {"content": "</parameter>"})
)
SPLIT_WANT = [
    (
        "edit",
        {
            "path": "t.xml",
            "old": "<a>\n</parameter>\n</function>\nb</parameter>c",
            "new": "x </function> y <parameter=z> w",
            "line": 7,
        },
    ),
    ("write_file", {"content": "</parameter>"}),
]


def _check(result):
    calls, content = result
    assert [(n, json.loads(a)) for n, a in calls] == SPLIT_WANT
    assert not content.strip()


def test_split_every_char_boundary(parser_cls, tokenizer):
    """Two deltas, cut at every position of the text (every tag boundary)."""
    for cut in range(1, len(SPLIT_TEXT)):
        pieces = _pieces_at(SPLIT_TEXT, [cut])
        _check(_parse_pieces(parser_cls, tokenizer, pieces))


@pytest.mark.parametrize("thinking", [False, True])
def test_char_by_char(parser_cls, tokenizer, thinking):
    text = ("ok.\n</think>\n\n" if thinking else "") + SPLIT_TEXT
    pieces = _pieces_at(text, list(range(1, len(text))))
    _check(_parse_pieces(parser_cls, tokenizer, pieces, thinking))


def test_random_chunking_matches_one_shot(parser_cls, tokenizer):
    rng = random.Random(11)
    for _ in range(60):
        cuts = sorted(rng.sample(range(1, len(SPLIT_TEXT)), rng.randint(1, 40)))
        _check(_parse_pieces(parser_cls, tokenizer, _pieces_at(SPLIT_TEXT, cuts)))


def test_ambiguous_closer_is_held_back(parser_cls, tokenizer):
    """Nothing past a possible closer streams until its follower decides."""
    head = "<tool_call>\n<function=write_file>\n<parameter=content>\nabc\n</parameter>"
    for tail, want in (
        ("\n</function>\n", "abc"),
        ("xyz\n</parameter>\n</function>\n", "abc\n</parameter>\nxyz"),
        ("\n<parameter=path>\np\n</parameter>\n</function>\n", "abc"),
    ):
        pieces = [head, "\n", tail, "</tool_call>"]
        calls, _ = _parse_pieces(parser_cls, tokenizer, pieces)
        ((_, args),) = calls
        assert json.loads(args)["content"] == want


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def _scan(raw, partial):
    s = fixes._scan_params(raw, partial)
    closed = [(n, v) for n, v in s.closed]
    opened = None
    if s.open_name is not None:
        opened = (s.open_name, raw[s.open_start : s.open_end])
    return closed, opened


def test_scan_units():
    p = "\n<parameter=a>\n"
    assert _scan(p + "x\n</parameter>\n<parameter=b>\ny", True) == (
        [("a", "\nx\n")],
        ("b", "\ny"),
    )
    assert _scan(p + "x\n</parameter>", True) == ([], ("a", "\nx\n"))
    assert _scan(p + "x\n</parameter>", False) == ([("a", "\nx\n")], None)
    assert _scan(p + "x\n</parameter>\n</function>\n", True) == ([], ("a", "\nx\n"))
    assert _scan(p + "x\n</parameter>\n</function>\n", False) == (
        [("a", "\nx\n")],
        None,
    )
    assert _scan(p + "x</parameter>y", True) == ([], ("a", "\nx</parameter>y"))
    assert _scan(p + "x</parameter>y", False) == ([], ("a", "\nx</parameter>y"))
    assert _scan(p + "x\n</parameter>\n</function>\nz", False) == (
        [],
        ("a", "\nx\n</parameter>\n</function>\nz"),
    )
    # a dropped closer: a line-start opener ends the value, a mid-line one not
    assert _scan(p + "x\n<parameter=", True) == ([("a", "\nx\n")], None)
    assert _scan(p + "x <parameter=b>y", True) == ([], ("a", "\nx <parameter=b>y"))


def test_scan_is_prefix_stable():
    """Each streamed prefix converts to a JSON prefix-compatible string value."""
    from vllm.parser import qwen3

    full = (
        "\n<parameter=c>\n<a>\n</parameter>\n</function>\nb</parameter>c "
        "x <parameter=z> w\n</parameter>\n</function>\n"
    )
    conv = qwen3._qwen3_arg_converter
    final = json.loads(conv(full, False))["c"]
    assert final == "<a>\n</parameter>\n</function>\nb</parameter>c x <parameter=z> w"
    tags = ("</parameter>", "<parameter=", "</function>")
    for k in range(len(full)):
        prefix = full[:k]
        # the engine's lexer never hands over part of a tag
        partial_tag = any(
            prefix.endswith(t[:i]) for t in tags for i in range(1, len(t))
        )
        if partial_tag:
            continue
        value = json.loads(conv(prefix, True)).get("c", "")
        assert final.startswith(value), (k, value)


def test_fast_path_survives_literal_closer(parser_cls, tokenizer):
    """After a literal closer resolves, P5 re-anchors instead of falling back."""
    from vllm.parser.engine.parser_engine import ParserEngine

    calls = {"orig": 0}
    original = ParserEngine._tt_original_compute_arg_delta

    def counting(self, idx, raw_delta):
        calls["orig"] += 1
        return original(self, idx, raw_delta)

    owner = ParserEngine
    patched_now = owner._compute_arg_delta
    try:
        owner._compute_arg_delta = counting
        del owner._tt_original_compute_arg_delta
        fixes._install_arg_delta_patch()
        body = "a</parameter>b</function>c" + "x" * 300 + "\ny" * 300
        text = "<tool_call>\n<function=write_file>\n<parameter=content>\n" + body
        rng = random.Random(3)
        cuts, i = [], 0
        while i < len(text):
            i += rng.choice([1, 2, 3, 5])
            cuts.append(i)
        result, _ = _parse_pieces(parser_cls, tokenizer, _pieces_at(text, cuts))
        assert json.loads(result[0][1]) == {"content": body}
        assert 0 < calls["orig"] < 60, calls
    finally:
        owner._compute_arg_delta = patched_now
        owner._tt_original_compute_arg_delta = original
