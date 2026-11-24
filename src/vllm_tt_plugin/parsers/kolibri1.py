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

``Kolibri1Parser`` owns the per-request parsing state; the reasoning adapter
exposes it through vLLM's reasoning interface. ``thinking_enabled`` is the
standalone template-switch operation. Lazy names and the upstream Hermes tool
alias are owned by ``parsers.registration``. Replace this adapter when the pinned
vLLM supplies the same template-switch and continuation behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.qwen3 import Qwen3Parser

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

__all__ = ("Kolibri1Parser", "Kolibri1ReasoningParser", "thinking_enabled")


class Kolibri1Parser(Qwen3Parser):
    """Uses Qwen3's grammar with the Kolibri template's initial reasoning state.

    Notes
    -----
    Create one instance per request. The upstream engine owns mutable stream
    state; this adapter changes only its initial state and leaves the caller's
    chat-template options unchanged.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs: object,
    ) -> None:
        """Initializes request-local state without changing template inputs.

        Parameters
        ----------
        tokenizer
            Tokenizer exposing the model's special-token vocabulary.
        tools
            Tool definitions forwarded unchanged to the upstream parser.
        **kwargs
            Upstream parser options. ``chat_template_kwargs`` is a mapping of
            template inputs; a shallow copy controls the initial reasoning state.
            Other options pass through unchanged.
        """
        chat_kwargs = dict(
            cast(Mapping[str, object] | None, kwargs.get("chat_template_kwargs")) or {}
        )
        chat_kwargs["enable_thinking"] = thinking_enabled(
            chat_kwargs
        ) and not chat_kwargs.get("continue_final_message", False)
        kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, tools, **kwargs)


class Kolibri1ReasoningParser(ParserEngineReasoningAdapter):
    """Exposes a request-local Kolibri engine through vLLM's reasoning interface."""

    _parser_engine_cls: type[Qwen3Parser] = Kolibri1Parser


def thinking_enabled(chat_template_kwargs: Mapping[str, object] | None) -> bool:
    """Resolves the Kolibri template's thinking switch without mutating its inputs.

    Parameters
    ----------
    chat_template_kwargs
        Template inputs, or ``None`` for the default enabled state. A non-None
        ``reasoning_effort`` wins; only ``"none"`` disables it. Otherwise only
        the literal boolean ``enable_thinking=False`` disables thinking.

    Examples
    --------
    >>> from vllm_tt_plugin.parsers.kolibri1 import thinking_enabled
    >>> thinking_enabled(None)
    True
    >>> thinking_enabled({"reasoning_effort": "none", "enable_thinking": True})
    False
    >>> thinking_enabled({"reasoning_effort": "high", "enable_thinking": False})
    True
    """
    kwargs = chat_template_kwargs or {}
    effort = kwargs.get("reasoning_effort")
    if effort is not None:
        return effort != "none"
    return kwargs.get("enable_thinking") is not False
