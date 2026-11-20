# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""JSON Schema ``pattern`` search semantics under xgrammar in the patched vLLM
(common, xgrammar pattern search). xgrammar compiles every pattern as a
whole-string match (mlc-ai/xgrammar#838, fix #925 unreleased), so ``"^x_"``
admitted only the key ``"x_"``."""

import copy
import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.vllm_patches import requires_patched_vllm

pytestmark = requires_patched_vllm

SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "pattern": "^ab"},
        "counters": {
            "type": "object",
            "patternProperties": {"^x_": {"type": "integer"}},
            "additionalProperties": False,
        },
    },
    "required": ["code", "counters"],
    "additionalProperties": False,
}
GOOD = {"code": "abc", "counters": {"x_signups": -12, "x_refunds": 340}}
NARROW = {"code": "ab", "counters": {"x_": 1}}
BAD = [
    {"code": "xab", "counters": {"x_a": 1}},
    {"code": "abc", "counters": {"y_a": 1}},
    {"code": "abc", "counters": {"x_a": "1"}},
]


@pytest.fixture
def pattern_search(monkeypatch):
    def set_enabled(enabled: bool):
        monkeypatch.setenv("VLLM_XGRAMMAR_PATTERN_SEARCH", "1" if enabled else "0")

    return set_enabled


def accepts(grammar, text: str) -> bool:
    from xgrammar.testing import _is_grammar_accept_string

    return _is_grammar_accept_string(grammar, text)


PATTERNS = ["ab", "^ab", "ab$", "^ab$", "a|b", "^a|b$", "(a|b)c", "^x_", "x_.*"]
PATTERNS += ["[0-9]+", "^[0-9]+", "[^a]$", r"a\$", "[$^]", "(?i)^ab", "colou?r"]
STRINGS = ["", "a", "b", "ab", "abc", "xab", "xabx", "ba", "AB", "x_", "x_signups"]
STRINGS += ["y_x_", "123", "a123", "123a", "a$", "$", "^a", "color", "my colour"]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_rewritten_pattern_matches_json_schema_search(pattern):
    """xgrammar(rewritten) accepts s <=> re.search(pattern, s), as JSON Schema."""
    import xgrammar as xgr
    from vllm.utils.json_schema_pattern import search_regex_for_xgrammar

    rewritten = search_regex_for_xgrammar(pattern)
    grammar = xgr.Grammar.from_json_schema(
        json.dumps({"type": "string", "pattern": rewritten})
    )
    for s in STRINGS:
        expected = re.search(pattern, s) is not None
        assert accepts(grammar, json.dumps(s)) == expected, (pattern, rewritten, s)


@pytest.mark.parametrize("enabled", [True, False])
def test_strict_mimo_tool_schema(pattern_search, enabled):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.tool_parsers import ToolParserManager
    from xgrammar import Grammar

    from tests.vllm_patches.test_mimo_grammar import mimo_call

    pattern_search(enabled)
    tools = [
        {
            "type": "function",
            "function": {"name": "label", "strict": True, "parameters": SCHEMA},
        }
    ]
    original = copy.deepcopy(tools)
    request = ChatCompletionRequest(
        model="mimo", messages=[{"role": "user", "content": "x"}], tools=tools
    )
    parser = ToolParserManager.get_tool_parser("mimo")(MagicMock(), request.tools)
    grammar = Grammar.from_structural_tag(
        parser.get_structural_tag(request).model_dump_json()
    )
    assert tools == original  # the request's schema is not rewritten in place
    assert accepts(grammar, mimo_call("label", NARROW))
    assert accepts(grammar, mimo_call("label", GOOD)) == enabled
    for bad in BAD:
        assert not accepts(grammar, mimo_call("label", bad)), bad


@pytest.mark.parametrize("enabled", [True, False])
def test_json_schema_structured_output(pattern_search, enabled):
    import xgrammar as xgr
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams
    from vllm.v1.structured_output.backend_xgrammar import (
        _with_search_patterns,
        validate_xgrammar_grammar,
    )

    pattern_search(enabled)
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=SCHEMA))
    validate_xgrammar_grammar(params)
    assert params.structured_outputs.json == SCHEMA

    grammar = xgr.Grammar.from_json_schema(_with_search_patterns(json.dumps(SCHEMA)))
    assert accepts(grammar, json.dumps(NARROW))
    assert accepts(grammar, json.dumps(GOOD)) == enabled
    for bad in BAD:
        assert not accepts(grammar, json.dumps(bad)), bad


def test_xgrammar_backend_compiles_with_the_pinned_xgrammar(pattern_search):
    """vLLM 0.31.0's xgrammar backend, run end to end on the overridden xgrammar."""
    transformers = pytest.importorskip("transformers")
    from vllm.config import StructuredOutputsConfig
    from vllm.v1.structured_output.backend_types import StructuredOutputOptions
    from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
    except Exception as e:  # offline without a cached gpt2
        pytest.skip(f"gpt2 tokenizer unavailable: {e}")
    pattern_search(True)
    vllm_config = SimpleNamespace(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
        speculative_config=None,
    )
    backend = XgrammarBackend(vllm_config, tokenizer=tokenizer, vocab_size=50257)

    def accepts_text(text: str) -> bool:
        grammar = backend.compile_grammar(
            StructuredOutputOptions.JSON, json.dumps(SCHEMA)
        )
        return grammar.matcher.accept_string(text) and grammar.matcher.accept_token(
            50256
        )

    assert accepts_text(json.dumps(GOOD))
    assert not accepts_text(json.dumps(BAD[0]))
