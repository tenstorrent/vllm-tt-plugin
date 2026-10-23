# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest


def _fresh_parser():
    from vllm.parser.harmony import HarmonyParser

    # process_chunk reads only the streaming-parser state: skip the tokenizer-bound
    # __init__.
    parser = HarmonyParser.__new__(HarmonyParser)
    parser._parser = None
    parser._next_tool_call_index = 0
    parser._num_processed_messages = 0
    parser._current_message_tokens = []
    return parser


def test_tokens_after_stop_token_are_dropped_not_fatal():
    """ignore_eos generates past <|return|>: the streamed message must still parse."""
    from openai_harmony import HarmonyError
    from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

    from vllm_tt_plugin.platform import _install_tt_harmony_after_stop_patch

    encoding = get_encoding()
    message = encoding.encode(
        "<|channel|>final<|message|>hi<|return|>", allowed_special="all"
    )
    after_stop = encoding.encode("<|channel|>", allowed_special="all")

    import vllm.parser.harmony as harmony_module

    parser_cls = harmony_module.HarmonyParser
    original = getattr(parser_cls, "_tt_original_process_chunk", None)
    if original is not None:
        with pytest.raises(HarmonyError):
            original(_fresh_parser(), message + after_stop)

    _install_tt_harmony_after_stop_patch()
    parser = _fresh_parser()
    result = parser.process_chunk(message + after_stop)
    assert "".join(segment.delta for segment in result.segments) == "hi"
    assert parser.process_chunk(after_stop).segments == []
