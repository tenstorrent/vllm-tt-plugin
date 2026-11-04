# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Host tests for the streaming fast paths in tool_call_fixes (P3 linear strip,
P5 incremental ``_compute_arg_delta``, P6 reasoning-count short-circuit).

P5 is checked by a fuzz: the full list of streamed (index, name, arguments)
deltas of the patched engine must equal the list of the original engine.
"""

import os
import random
import re

import pytest

import vllm_tt_plugin.tool_call_fixes as fixes

TOKENIZER_PATH = os.environ.get(
    "QWEN38_TOKENIZER_PATH", "/mnt/tt-data/sdawle/models/Qwen3.8-27B"
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(TOKENIZER_PATH), reason=f"{TOKENIZER_PATH} is not available"
)

N_STREAMS = 600


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


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "w",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "count": {"type": "integer"},
                    "flag": {"type": "boolean"},
                    "opt": {"type": ["string", "null"]},
                    "data": {"type": "object"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "noschema",
            "description": "n",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def make_parser(parser_cls, tokenizer, thinking):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        tools=TOOLS,
        tool_choice="auto",
    )
    parser = parser_cls(
        tokenizer,
        req.tools,
        chat_template_kwargs={"enable_thinking": thinking},
        model_config=None,
    )
    return parser, parser.adjust_request(req)


def prompt_ids(tokenizer, thinking):
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
    tail = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return enc("<|im_start|>assistant\n") + enc(tail)


ATOMS = [
    "x", "def f(a, b):\n", "    return a < b and b > a\n", "if (a<b) { c = d->e; }\n",
    '"q"', "\\n", "\\", "é漢😀", "\t", "\n", "\n\n", " " * 40, "<", ">", "</",
    "</para", "</function", "</function>", "</tool_call>", "< /function >", "<p>",
    "<paramet", "</ parameter", "=", "<parameter", "list[int]", "{}", "[1,2]", "0042",
]  # fmt: skip


def gen_call(rng):
    fname = rng.choice(["write_file", "write_file", "noschema", "unknown_fn"])
    params = []
    for _ in range(rng.randint(1, 4)):
        k = rng.choice(
            [
                "path",
                "content",
                "content",
                "count",
                "flag",
                "opt",
                "data",
                "other",
                "content",
            ]
        )
        if k == "count":
            v = rng.choice(["7", "007", "x", " 12 ", "-3"])
        elif k == "flag":
            v = rng.choice(["true", "False", "yes"])
        elif k == "data":
            v = rng.choice(['{"a": 1}', '{"a": ', "[1]", "zz"])
        else:
            v = "".join(rng.choice(ATOMS) for _ in range(rng.randint(0, 25)))
        params.append((k, v))
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params[:-1])
    kind = rng.random()
    k, v = params[-1]
    if kind < 0.6:
        body += f"<parameter={k}>\n{v}\n</parameter>\n"
    elif kind < 0.8:
        body += f"<parameter={k}>\n{v}"  # unclosed last parameter
    else:
        body += f"<parameter={k}>\n{v}\n"
    tail = rng.choice(["</function>\n</tool_call>", "</function>", "", "</tool_call>"])
    return f"<tool_call>\n<function={fname}>\n{body}{tail}"


def run_stream(parser_cls, tokenizer, text, rng, thinking):
    parser, req = make_parser(parser_cls, tokenizer, thinking)
    pids = prompt_ids(tokenizer, thinking)
    if thinking:
        text = "think.\n</think>\n\n" + text
    pieces, i = [], 0
    while i < len(text):
        n = rng.choice([1, 1, 2, 3, 5, 9, 17])
        pieces.append(text[i : i + n])
        i += n
    out = []
    for j, piece in enumerate(pieces):
        ids = tokenizer.encode(piece, add_special_tokens=False)
        dm = parser.parse_delta(
            delta_text=piece,
            delta_token_ids=ids,
            request=req,
            prompt_token_ids=pids,
            finished=(j == len(pieces) - 1),
        )
        for tc in (dm.tool_calls or []) if dm is not None else []:
            fn = tc.function
            out.append(
                (tc.index, fn.name if fn else None, fn.arguments if fn else None)
            )
    return out


def test_markers_and_idempotent():
    from vllm.parser.engine import adapters, parser_engine

    owner = parser_engine.ParserEngine
    patched = owner._compute_arg_delta
    count = adapters.ParserEngineReasoningAdapter.count_reasoning_tokens
    assert hasattr(owner, "_tt_original_compute_arg_delta")
    assert hasattr(
        adapters.ParserEngineReasoningAdapter, "_tt_original_count_reasoning_tokens"
    )
    fixes.install_tool_call_fixes()
    assert owner._compute_arg_delta is patched
    assert adapters.ParserEngineReasoningAdapter.count_reasoning_tokens is count


def test_arg_delta_fuzz_equivalence(parser_cls, tokenizer):
    from vllm.parser.engine.parser_engine import ParserEngine

    patched = ParserEngine._compute_arg_delta
    original = ParserEngine._tt_original_compute_arg_delta
    assert patched is not original
    bad = []
    nonempty = 0
    try:
        for seed in range(N_STREAMS):
            rng0 = random.Random(seed)
            text = gen_call(rng0)
            if seed % 5 == 0:
                text += "\n" + gen_call(rng0)  # second tool call
            thinking = seed % 2 == 0
            ParserEngine._compute_arg_delta = original
            a = run_stream(
                parser_cls, tokenizer, text, random.Random(seed + 1), thinking
            )
            ParserEngine._compute_arg_delta = patched
            b = run_stream(
                parser_cls, tokenizer, text, random.Random(seed + 1), thinking
            )
            nonempty += bool(a)
            if a != b:
                bad.append((seed, text[:200], a[:6], b[:6]))
    finally:
        ParserEngine._compute_arg_delta = patched
    assert not bad, bad[:3]
    assert nonempty > N_STREAMS // 2


def test_arg_delta_fast_path_is_used(parser_cls, tokenizer):
    """The fast path must run, not silently fall back for every delta."""
    from vllm.parser.engine.parser_engine import ParserEngine

    calls = {"orig": 0}
    original = ParserEngine._tt_original_compute_arg_delta

    def counting(self, idx, raw_delta):
        calls["orig"] += 1
        return original(self, idx, raw_delta)

    ParserEngine._tt_original_compute_arg_delta = counting
    try:
        # patched() closes over the original it captured at install time, so
        # count through a fresh install instead.
        owner = ParserEngine
        patched_now = owner._compute_arg_delta
        owner._compute_arg_delta = counting
        del owner._tt_original_compute_arg_delta
        fixes._install_arg_delta_patch()
        body = "x" * 300 + "\ny" * 300
        text = "<tool_call>\n<function=write_file>\n<parameter=content>\n" + body
        run_stream(parser_cls, tokenizer, text, random.Random(3), False)
        assert 0 < calls["orig"] < 60, calls
        owner._compute_arg_delta = patched_now
    finally:
        ParserEngine._tt_original_compute_arg_delta = original


def _closer_cases():
    cases = [
        "",
        "abc",
        "abc</function>",
        "abc </function> ",
        "abc\n</function>\n</tool_call>\n",
        "a</ tool_call >", "a</function></tool_call></function>", "a< /function>",
        "a</func tion>", "a</function", "</function>", "   ", " " * 5000,
        "x" + " " * 5000, "x" + " " * 5000 + "</function>",
        " " * 5000 + "</tool_call>  ",
        "</function>x", "a</function>x</tool_call>", "a</tool_call> ",
        "a </ function　> ", "a   </function>", "café</function>", "a</tool_call>\x1c",
        "a< / function >", "a</functionx>", "a</Function>", "a</function>>",
        "a</function> </function> ", "x\n\n\n</tool_call>\n\n</function>",
        "</ function>", "< / function>", "<\n/\nfunction\n>", "a>b", "a<b>", "<>", ">",
        "/function>", "a </function> ", "a </tool_call>　",
    ]  # fmt: skip
    rng = random.Random(7)
    atoms = ["</function>", "</tool_call>", " ", "\n", "\t", " ", " ", "\x1f", "a",
             "<", ">", "/", "function", "tool_call", "</", "xxx"]  # fmt: skip
    for _ in range(20000):
        cases.append("".join(rng.choice(atoms) for _ in range(rng.randint(0, 14))))
    return cases


def test_closer_strip_matches_regex():
    regex = re.compile(r"\s*</\s*(?:function|tool_call)\s*>\s*$")

    def old(value):
        while regex.search(value):
            value = regex.sub("", value)
        return value

    def new(value):
        while (stripped := fixes._strip_one_closer(value)) is not None:
            value = stripped
        return value

    cases = _closer_cases()
    assert len(cases) >= 5000
    bad = [
        c
        for c in cases
        if old(c) != new(c)
        or bool(regex.search(c)) != (fixes._strip_one_closer(c) is not None)
    ]
    assert not bad, [repr(c) for c in bad[:3]]


def test_closer_strip_is_linear_on_whitespace():
    import time

    value = "x" + " " * 200_000
    start = time.perf_counter()
    assert fixes._strip_one_closer(value) is None
    assert time.perf_counter() - start < 0.5


def test_count_short_circuit_thinking_off(parser_cls, tokenizer):
    parser, _ = make_parser(parser_cls, tokenizer, thinking=False)
    adapter = parser._reasoning_parser
    ids = tokenizer.encode(
        "<think>a</think>b<tool_call>\n<function=noschema>\n</function>\n</tool_call>",
        add_special_tokens=False,
    )
    assert not adapter._streaming_count_valid
    assert adapter.count_reasoning_tokens(tuple(ids)) == 0
    assert adapter._counting_parser_engine is None  # no decode and re-parse
    original = type(adapter)._tt_original_count_reasoning_tokens
    assert original(adapter, tuple(ids)) == 0  # same value as the original


def test_count_unchanged_when_thinking_on(parser_cls, tokenizer):
    parser, _ = make_parser(parser_cls, tokenizer, thinking=True)
    adapter = parser._reasoning_parser
    original = type(adapter)._tt_original_count_reasoning_tokens
    ids = tokenizer.encode("<think>abc def</think>answer", add_special_tokens=False)
    expected = original(adapter, tuple(ids))
    assert expected > 0
    assert adapter.count_reasoning_tokens(tuple(ids)) == expected
    assert adapter._counting_parser_engine is not None  # slow path ran
