# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""The streaming detokenizer primed with the prompt's tail, not the whole prompt:
``TT_DETOK_PRIME_TOKENS=<K>``.

vLLM's ``FastIncrementalDetokenizer`` (``vllm/v1/engine/detokenizer.py``) constructs
``tokenizers.decoders.DecodeStream(ids=request.prompt_token_ids)``: the stream keeps
every id it was given and, at
the first output token, decodes them all to text to form its prefix, then all of them
again with the new id
appended (``tokenizers/src/tokenizer/mod.rs``, ``step_decode_stream``). Two passes over
the prompt, 0.26 us an
id (0.27 s at 1M tokens), in the API server's output-handler loop, once per request,
before any output of the
same engine step is streamed; a cohort whose 32 first tokens leave one decode step
(``TT_SCHED_FINAL_STEP``)
pays it 32 times in a row, 9.7 s at 1M, during which the engine keeps decoding and the
client sees nothing. The stream needs only the last few ids to render the first output
token
(a multi-byte character spans at most a few ids; vLLM's own slow detokenizer primes with
7).

With the variable set, :func:`install` (called from the plugin's
``vllm.general_plugins`` entry point, so it runs
in the API server as well as the engine) replaces ``tokenizers.decoders.DecodeStream``
on its module with
:class:`BoundedDecodeStream`, which keeps the last ``K`` ids of what it is given and
delegates everything else to
the stock class. vLLM looks the class up on the module at construction time for this
purpose. The text emitted
is identical: the stream's prefix is what it decodes from its ids, and the delta after
it depends on the new id
and the ids immediately before it only, provided the tail holds a whole character before
any trailing special
ids (``K`` at least 4, the longest UTF-8 character in byte-level ids; the default 16
covers the chat template's
trailing specials as well). Unset or ``0``, nothing is installed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import tokenizers.decoders

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

ENV = "TT_DETOK_PRIME_TOKENS"
# Covers multi-byte characters and adjacent special tokens the decoder needs.
DEFAULT = 16
# A UTF-8 character is at most 4 bytes, so at most 4 byte-level ids: fewer can
# leave the tail's text ending in U+FFFD, the stream's degenerate case (its prefix
# never set, the window echoed at the first token).
MINIMUM = 4

_STOCK: type | None = None  # the class the module had before install


def value(environ: Mapping[str, str] | None = None) -> int:
    """The bound, or 0 when unset (the stock priming)."""
    env = os.environ if environ is None else environ
    v = env.get(ENV, "").strip()
    if v in ("", "0"):
        return 0
    try:
        k = int(v)
    except ValueError:
        raise ValueError(f"{ENV}={v!r}: a positive integer, 0 or unset") from None
    if k < MINIMUM:
        raise ValueError(
            f"{ENV}={v!r}: at least {MINIMUM} (a character's ids), 0 or unset"
        )
    return k


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return value(environ) > 0


class BoundedDecodeStream:
    """``tokenizers.decoders.DecodeStream`` primed with at most ``bound`` ids: the tail
    of what it is given."""

    bound: int = DEFAULT
    stock: type = (
        tokenizers.decoders.DecodeStream
    )  # the class delegated to; replaced by install()

    def __init__(
        self, ids: list[int] | None = None, skip_special_tokens: bool = False
    ) -> None:
        self.primed = 0 if ids is None else min(len(ids), self.bound)
        if ids is not None and len(ids) > self.bound:
            ids = ids[-self.bound :]
        self._stream = self.stock(ids=ids, skip_special_tokens=skip_special_tokens)

    # Keep the stock signature.
    def step(self, tokenizer: Any, id: Any) -> str | None:  # noqa: A002
        return self._stream.step(tokenizer, id)

    def __getattr__(self, name: str) -> Any:
        if name == "_stream":
            raise AttributeError(name)
        return getattr(self._stream, name)


def install(environ: Mapping[str, str] | None = None) -> int:
    """Install the bounded class on ``tokenizers.decoders`` when the variable is set;
    idempotent. Returns the
    bound installed, or 0 when the stock class is left (or restored, if a previous
    install is in place)."""
    global _STOCK
    k = value(environ)
    current = tokenizers.decoders.DecodeStream
    if k == 0:
        if _STOCK is not None:
            tokenizers.decoders.DecodeStream = _STOCK
            _STOCK = None
        return 0
    if _STOCK is None:
        _STOCK = current
        BoundedDecodeStream.stock = current
    BoundedDecodeStream.bound = k
    tokenizers.decoders.DecodeStream = BoundedDecodeStream
    logger.info(
        "TT detokenizer: DecodeStream primed with the prompt's last %d ids (%s)", k, ENV
    )
    return k


def uninstall() -> None:
    """Restore the stock class (tests)."""
    install({})
