# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks Qwen3 tool-call arguments through vLLM's ``qwen3_coder`` parser.

Each case runs the upstream parser class after ``install_patches``, on the
complete output and on the output streamed one and four tokens per delta.
A streamed call's argument deltas must join into the same JSON as the
non-streamed arguments.
"""

import functools
import random
from collections.abc import Callable, Iterator

import pytest
from vllm.parser import qwen3

from tests.support import parser_patches
from tests.support.tool_calls import Call, call_text, extract, stream
from vllm_tt_plugin.parsers import qwen3_arguments

Parse = Callable[[dict[str, dict], str], list[Call]]

TOOLS = {
    "edit": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
    }
}
HEAD = "<tool_call>\n<function=edit>\n<parameter=path>\na.html\n</parameter>\n"


@pytest.fixture(autouse=True)
def installed() -> Iterator[None]:
    """Installs the patches for one test and restores upstream vLLM afterward."""
    restore = parser_patches.snapshot()
    qwen3_arguments.install_patches()
    yield
    restore()


@pytest.fixture(
    params=[
        pytest.param(extract, id="one-shot"),
        pytest.param(stream, id="streamed"),
        pytest.param(functools.partial(stream, step=4), id="streamed-4-tokens"),
    ]
)
def parse(request: pytest.FixtureRequest) -> Parse:
    """Parses a model output on the non-streaming or a streaming path."""
    return request.param


@pytest.mark.parametrize(
    ("tail", "text"),
    [
        pytest.param("\n</function>\n</tool_call>", "<p>hi</p>", id="closer-omitted"),
        pytest.param("\n</tool_call>", "<p>hi</p>", id="closers-omitted"),
        pytest.param("\n</function>\n</tool_call>", "x  ", id="trailing-spaces"),
        pytest.param("", "<p>h", id="generation-stopped"),
    ],
)
def test_keeps_last_parameter_without_closer(parse: Parse, tail: str, text: str):
    """A last value without ``</parameter>`` stays in the arguments."""
    output = f"{HEAD}<parameter=text>\n{text}{tail}"
    assert parse(TOOLS, output) == [Call("edit", {"path": "a.html", "text": text})]


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("<x></parameter></x>", id="inline-parameter-closer"),
        pytest.param("<x>\n</parameter>\n</x>", id="parameter-closer-line"),
        pytest.param("a</function>b", id="inline-function-closer"),
        pytest.param("<x>\n</function>\n</x>", id="function-closer-line"),
        pytest.param("a</parameter>\n</function>\nb", id="both-closers"),
        pytest.param("a <parameter=b> c", id="inline-parameter-opener"),
    ],
)
def test_keeps_literal_tags_in_value(parse: Parse, text: str):
    """Tag text that the template does not continue stays in the value."""
    output = call_text("edit", {"path": "a.html", "text": text})
    assert parse(TOOLS, output) == [Call("edit", {"path": "a.html", "text": text})]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        pytest.param(
            call_text("edit", {"path": "a", "text": "x\n\ny"})
            + "\n"
            + call_text("edit", {"path": "b", "text": ""}),
            [
                Call("edit", {"path": "a", "text": "x\n\ny"}),
                Call("edit", {"path": "b", "text": ""}),
            ],
            id="two-calls",
        ),
        pytest.param(
            f"{HEAD}</function>\n<function=edit>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>",
            [Call("edit", {"path": "a.html"}), Call("edit", {"path": "b"})],
            id="next-function-in-same-block",
        ),
        pytest.param(
            f"{HEAD}<parameter=text>\nx\n<parameter=more>\ny\n</parameter>\n</function>\n</tool_call>",
            [Call("edit", {"path": "a.html", "text": "x", "more": "y"})],
            id="closer-dropped-before-next-parameter",
        ),
    ],
)
def test_call_boundaries_match_template(
    parse: Parse, output: str, expected: list[Call]
):
    """Tags in the template's positions still end values and calls."""
    assert parse(TOOLS, output) == expected


@pytest.mark.parametrize("seed", range(4))
def test_streaming_matches_complete_output(seed: int):
    """Streamed and non-streamed arguments agree for random values and tags.

    Values without tags must also come back unchanged. Parameter names are
    unique, because a repeated key changes an earlier value and upstream
    streaming cannot express that.
    """
    rng = random.Random(seed)
    plain = ["x", " ", "\n", '"', "\\", "é", "<", ">", "/", "<div>", "</div>"]
    tags = [
        "</parameter>",
        "</function>",
        "</tool_call>",
        "</parameter>\n",
        "\n</function>\n",
    ]
    tools = {"f": {"type": "object", "properties": {}}}
    for case in range(40):
        atoms = plain + (tags if case % 2 else [])
        params = {
            key: "".join(rng.choice(atoms) for _ in range(rng.randint(0, 10)))
            for key in rng.sample("abc", rng.randint(1, 3))
        }
        output = call_text("f", params)
        if case % 3 == 0:
            output = output[
                : rng.randint(len("<tool_call>\n<function=f>\n"), len(output))
            ]
        complete = extract(tools, output)
        assert stream(tools, output) == complete, output
        assert stream(tools, output, step=3) == complete, output
        if case % 2 == 0 and case % 3 != 0:
            assert complete == [Call("f", params)], output


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("x{}", id="plain"),
        pytest.param("x{}</parameter>y{}", id="literal-closer"),
    ],
)
def test_plain_deltas_skip_conversion(monkeypatch: pytest.MonkeyPatch, value: str):
    """Streaming a ten times longer value scans the arguments as often.

    ``convert_arguments`` and ``extend_open_value`` both start with ``_scan``,
    which reads the whole argument text. Without the fast path, every delta
    of the value would scan once more.
    """
    scans = 0
    scan = qwen3_arguments._scan

    def counting_scan(text: str, partial: bool) -> qwen3_arguments._Scan:
        nonlocal scans
        scans += 1
        return scan(text, partial)

    monkeypatch.setattr(qwen3_arguments, "_scan", counting_scan)

    counts = []
    for length in (100, 1000):
        text = value.format("x" * length, "x" * length)
        output = call_text("edit", {"path": "a.html", "text": text})
        scans = 0
        assert stream(TOOLS, output) == [Call("edit", {"path": "a.html", "text": text})]
        counts.append(scans)
    assert counts[0] == counts[1]


def test_install_is_idempotent():
    """A second install keeps the upstream functions as the originals."""
    upstream = qwen3._tt_original_qwen3_arg_converter

    qwen3_arguments.install_patches()

    assert qwen3._qwen3_arg_converter is qwen3_arguments.convert_arguments
    assert qwen3._tt_original_qwen3_arg_converter is upstream
    assert upstream is not qwen3_arguments.convert_arguments


def test_install_rejects_changed_converter_signature(monkeypatch: pytest.MonkeyPatch):
    """An upstream converter with other parameters fails at install time.

    The parser engine swallows a ``TypeError`` from the converter, so a
    replacement with the wrong parameters would drop arguments silently.
    """

    def converter(raw_args: str, partial: bool, closed: object = None) -> str:
        return "{}"

    # As before the first install, with a converter that takes another argument.
    monkeypatch.delattr(qwen3, "_tt_original_qwen3_arg_converter")
    monkeypatch.setattr(qwen3, "_qwen3_arg_converter", converter)

    with pytest.raises(RuntimeError, match="_qwen3_arg_converter"):
        qwen3_arguments.install_patches()
