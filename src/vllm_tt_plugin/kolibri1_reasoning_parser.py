# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Aleph-Alpha/Kolibri-1 served by the TT plugin.

Ported from Aleph Alpha's ``aleph-alpha-inference`` 1.0.0 (``reasoning.py``,
Apache-2.0). Kolibri uses the Qwen3 ``<think>``/``</think>`` grammar; the only
difference from vLLM's stock ``qwen3`` parser is how the starting state is
chosen. The Kolibri chat template switches thinking off when
``reasoning_effort == "none"`` and only falls back to ``enable_thinking`` when
no ``reasoning_effort`` is given, whereas the stock parser reads
``enable_thinking`` alone. A mismatch makes the non-streaming path file the
answer as reasoning (or the reverse). ``Kolibri1Parser`` derives the start
state the same way the template does.

A request with ``continue_final_message`` extends the final assistant
message. The template has already rendered that message's closed
``<think>...</think>`` block in the prompt, so the continuation is answer
content and the parser starts in content state.
"""

from collections.abc import Mapping
from typing import Any

from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.qwen3 import Qwen3Parser


def thinking_enabled(chat_template_kwargs: Mapping[str, Any] | None) -> bool:
    """Mirror the Kolibri template's switch.

    A ``reasoning_effort`` other than ``None`` wins and only ``"none"`` disables
    thinking. Without it, only a literal ``enable_thinking: false`` disables it,
    as the template tests ``enable_thinking is false``.
    """
    kwargs = chat_template_kwargs or {}
    effort = kwargs.get("reasoning_effort")
    if effort is not None:
        return effort != "none"
    return kwargs.get("enable_thinking") is not False


class Kolibri1Parser(Qwen3Parser):
    """Qwen3 grammar with the starting state chosen like the Kolibri 1 template."""

    def __init__(self, tokenizer, tools=None, **kwargs) -> None:
        # Parser-local copy: only the parser's starting state changes, not the
        # inputs the chat template rendered the prompt with.
        chat_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
        chat_kwargs["enable_thinking"] = thinking_enabled(
            chat_kwargs
        ) and not chat_kwargs.get("continue_final_message", False)
        kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, tools, **kwargs)


class Kolibri1ReasoningParser(ParserEngineReasoningAdapter):
    _parser_engine_cls = Kolibri1Parser
