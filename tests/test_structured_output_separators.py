# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""JSON grammars that TT's ``disable_any_whitespace`` compiles through xgrammar."""

import json
from types import SimpleNamespace

import pytest
import xgrammar as xgr
from vllm.v1.structured_output import backend_xgrammar
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

from tests import test_dp_modes
from vllm_tt_plugin import platform

SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
)
SPACED = '{"name": "Al", "age": 3}'
COMPACT = '{"name":"Al","age":3}'

# One token per printable ASCII character, so a string maps to token ids
# without a real tokenizer.
VOCAB = [chr(code) for code in range(32, 127)] + ["</s>"]


@pytest.fixture
def char_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer_info = xgr.TokenizerInfo(
        VOCAB, vocab_type=xgr.VocabType.RAW, stop_token_ids=[len(VOCAB) - 1]
    )
    monkeypatch.setattr(
        xgr.TokenizerInfo,
        "from_huggingface",
        lambda _tokenizer, vocab_size: tokenizer_info,
    )


def accepts(disable_any_whitespace: bool, text: str) -> bool:
    backend = backend_xgrammar.XgrammarBackend(
        vllm_config=SimpleNamespace(
            structured_outputs_config=SimpleNamespace(
                disable_any_whitespace=disable_any_whitespace
            ),
            model_config=None,
            speculative_config=None,
        ),
        tokenizer=SimpleNamespace(
            get_vocab=lambda: {token: index for index, token in enumerate(VOCAB)}
        ),
        vocab_size=len(VOCAB),
    )
    grammar = backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA)
    tokens = [VOCAB.index(char) for char in text]
    return grammar.validate_tokens(tokens) == tokens


@pytest.mark.usefixtures("char_tokenizer")
def test_whitespace_free_json_keeps_one_space_after_separators() -> None:
    platform._install_tt_xgrammar_json_separators_patch()
    post_init = backend_xgrammar.XgrammarBackend.__post_init__
    platform._install_tt_xgrammar_json_separators_patch()
    assert backend_xgrammar.XgrammarBackend.__post_init__ is post_init

    assert accepts(True, SPACED)
    assert not accepts(True, COMPACT)
    assert not accepts(True, '{"name":  "Al", "age": 3}')
    # Without the flag the backend keeps upstream's free whitespace.
    assert accepts(False, SPACED)
    assert accepts(False, COMPACT)


@pytest.mark.usefixtures("char_tokenizer")
def test_config_hook_installs_the_separator_patch(
    monkeypatch: pytest.MonkeyPatch, vllm_config: SimpleNamespace
) -> None:
    assert not accepts(True, SPACED)

    dummy_model_class = type(
        "DummyModel", (), {"__module__": "models.tt_transformers.tt.generator_vllm"}
    )
    test_dp_modes.TestDPModes.register_dummy_model(
        monkeypatch, vllm_config, dummy_model_class
    )

    assert vllm_config.structured_outputs_config.disable_any_whitespace is True
    assert accepts(True, SPACED)
