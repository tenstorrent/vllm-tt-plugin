# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import copy
import os
from pathlib import Path

import pytest

from vllm_tt_plugin.deepseek_v41_tokenizer import (
    encode_chat,
    get_deepseek_v41_tokenizer,
    load_encoder,
)


@pytest.fixture
def encoder_path():
    path = os.getenv("V41_TEST_ENCODER")
    if not path:
        pytest.skip(
            "set V41_TEST_ENCODER to the pinned checkpoint encoding/encoding.py"
        )
    assert Path(path).is_file()
    return path


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Find an account",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    }
]
MESSAGES = [
    {"role": "system", "content": "Banking assistant."},
    {"role": "user", "content": "Find account 001."},
    {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Need a lookup.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"id":"001"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "Found."},
    {"role": "user", "content": "What is the balance?"},
]


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("effort", [None, "low", "high", "max", 1, 37, 100])
def test_tool_history_matches_checkpoint_and_preserves_inputs(
    encoder_path, thinking, effort
):
    original = copy.deepcopy(MESSAGES)
    expected_messages = copy.deepcopy(MESSAGES)
    expected_messages[0]["tools"] = TOOLS
    expected = load_encoder(encoder_path).encode_messages(
        expected_messages,
        thinking_mode="thinking" if thinking else "chat",
        reasoning_effort=effort,
    )
    actual = encode_chat(
        MESSAGES,
        tools=TOOLS,
        encoder_path=encoder_path,
        thinking=thinking,
        reasoning_effort=effort,
    )
    assert actual == expected
    assert "<｜DSML｜ calls>" in actual
    assert "<tool_result>Found.</tool_result>" in actual
    assert ("Need a lookup." in actual) == thinking
    assert original == MESSAGES


def test_encoder_hash_is_checked(tmp_path):
    path = tmp_path / "encoder.py"
    path.write_text("raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_encoder(str(path))


@pytest.mark.parametrize("symlink", [False, True])
def test_encoder_loads_extensionless_hf_cache_blob(encoder_path, tmp_path, symlink):
    blob = tmp_path / "92a67eab2a67d924a6a8279bc0f90e3f504ac40e"
    blob.write_bytes(Path(encoder_path).read_bytes())
    path = blob
    if symlink:
        path = tmp_path / "encoding.py"
        path.symlink_to(blob)
    expected = load_encoder(encoder_path).encode_messages(
        MESSAGES, thinking_mode="chat"
    )
    assert (
        load_encoder(str(path)).encode_messages(MESSAGES, thinking_mode="chat")
        == expected
    )


def test_tokenizer_retains_ids_and_uses_encoder(encoder_path):
    class Tokenizer:
        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return list(text.encode())

    original = Tokenizer()
    wrapped = get_deepseek_v41_tokenizer(original, encoder_path)
    prompt = wrapped.apply_chat_template(
        MESSAGES, tools=TOOLS, thinking=True, tokenize=False
    )
    assert wrapped.apply_chat_template(MESSAGES, tools=TOOLS, thinking=True) == list(
        prompt.encode()
    )
    assert original.__class__ is Tokenizer
