# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm_tt_plugin.deepseek_v41_dsml import END, EOS, START
from vllm_tt_plugin.deepseek_v41_tool_parser import DeepSeekV41ToolParser

OUTPUT = (
    "Checking.\n\n"
    + START
    + """
<｜DSML｜ invoke name="lookup">
<｜DSML｜ parameter name="id" string="true">001</｜DSML｜ parameter>
<｜DSML｜ parameter name="n" string="false">2</｜DSML｜ parameter>
</｜DSML｜ invoke>
"""
    + END
    + EOS
)


def parser():
    return DeepSeekV41ToolParser(SimpleNamespace(get_vocab=lambda: {}))


def test_nonstream_vllm_protocol():
    result = parser().extract_tool_calls(OUTPUT, SimpleNamespace())
    assert result.tools_called
    assert result.content == "Checking.\n\n"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "lookup"
    assert json.loads(call.function.arguments) == {"id": "001", "n": 2}


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_streaming_protocol_and_request_reuse(chunk_size):
    instance = parser()
    for _ in range(2):
        previous, content, calls = "", "", []
        for start in range(0, len(OUTPUT), chunk_size):
            current = OUTPUT[: start + chunk_size]
            delta = instance.extract_tool_calls_streaming(
                previous,
                current,
                current[len(previous) :],
                [],
                [],
                [],
                SimpleNamespace(),
            )
            previous = current
            if delta:
                content += delta.content or ""
                calls.extend(delta.tool_calls or [])
        assert content == "Checking.\n\n"
        assert len(calls) == 1
        assert calls[0].function.name == "lookup"
        assert json.loads(calls[0].function.arguments) == {"id": "001", "n": 2}
        assert calls[0].id.startswith("call_")


def test_required_tool_choice_does_not_enable_unsupported_grammar():
    instance = parser()
    request = SimpleNamespace(
        tools=[{}],
        tool_choice="required",
        skip_special_tokens=True,
        structured_outputs=None,
    )
    assert instance.adjust_request(request) is request
    assert not request.skip_special_tokens
    assert request.structured_outputs is None
    assert not instance.supports_required_and_named


def test_registered_tokenizer_and_hf_renderer_on_checkpoint():
    snapshot = os.getenv("V41_TEST_SNAPSHOT")
    if not snapshot:
        pytest.skip("set V41_TEST_SNAPSHOT for the installed-vLLM integration test")
    from vllm.renderers.hf import safe_apply_chat_template
    from vllm.tokenizers.registry import get_tokenizer

    from vllm_tt_plugin.deepseek_v41_tokenizer import load_encoder
    from vllm_tt_plugin.entrypoints import _register_tt_tokenizers

    _register_tt_tokenizers()
    tokenizer = get_tokenizer(snapshot, tokenizer_mode="deepseek_v41")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = [{"role": "user", "content": "Look up this account."}]
    template = (
        Path(__file__).parents[1]
        / "src/vllm_tt_plugin/chat_templates/deepseek_v41.jinja"
    ).read_text()
    actual = safe_apply_chat_template(
        None,
        tokenizer,
        messages,
        tools=tools,
        chat_template=template,
        tokenize=False,
        thinking=True,
        reasoning_effort="max",
    )
    expected = load_encoder(
        str(Path(snapshot) / "encoding/encoding.py")
    ).encode_messages(
        [{"role": "system", "content": "", "tools": tools}, *messages],
        thinking_mode="thinking",
        reasoning_effort="max",
    )
    assert actual == expected
    assert tokenizer.encode(actual) == tokenizer.encode(expected)
