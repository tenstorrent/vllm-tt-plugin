# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Token masks of a deferred ``argmax_ids`` verify (``supports_spec_device_mask``).

A model declaring ``supports_spec_device_mask`` draws the columns of a verify
row from the target distribution masked by one packed token mask per row and
column, which ``sample_deferred_decode`` receives after the forward
(docs/SPEC_DECODE_CONTRACT.md, section 4j). This module builds that mask from
the controls that rule tokens out:

- the grammar bitmask the scheduler fills while the forward runs
  (``spec_grammar_bitmask_for_tt_batch``);
- ``allowed_token_ids`` and the one-token ``bad_words``, the same at every
  column of every step of a request (``SpecMaskCache``);
- a longer ``bad_words`` entry, which bans its last token at a column whose
  history (committed output plus the row's first ``j`` drafts) ends with the
  rest of the word;
- a pending ``min_tokens``, which bans the stop ids at column ``j`` while the
  committed output plus ``j`` is below ``min_tokens``.

Each control only clears bits, so the masks combine by AND. The rules are the
ones the runner's ``logits`` walk applies
(``spec_accept.apply_speculative_token_filters``), so a masked draw and the
host walk certify the same distribution.

Principal entry points: ``static_token_mask`` at build time (every control but
the grammar, which only the sample-time ``GrammarOutput`` has),
``compose_token_mask`` at sample time, and ``mask_may_allow_nothing``, which
decides before the build whether a row may be masked on the device at all.

A mask row is ``[words]`` int32 in xgrammar's packing: bit ``v % 32`` of word
``v // 32`` is 1 where token ``v`` is allowed, ``words = ceil(vocab / 32)``,
and a row of -1 words constrains nothing.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm_tt_plugin.spec_admission import SUPPORTS_SPEC_DEVICE_MASK
from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_FUSED_SAMPLE

WORD_BITS = 32


def mask_words(vocab_size: int) -> int:
    """Words of one mask row: ``ceil(vocab_size / 32)``."""
    return -(-int(vocab_size) // WORD_BITS)


def unconstrained_mask(*lead: int, words: int) -> torch.Tensor:
    """The mask ``[*lead, words]`` int32 that constrains nothing (every word -1)."""
    return torch.full((*lead, words), -1, dtype=torch.int32)


def _vocab_ids(ids: Sequence[int], vocab_size: int) -> np.ndarray:
    array = np.asarray(list(ids), dtype=np.int64)
    return array[(array >= 0) & (array < vocab_size)]


def pack_allowed_ids(ids: Sequence[int], vocab_size: int) -> torch.Tensor:
    """The ``[words]`` int32 row that allows exactly ``ids``.

    Ids outside ``[0, vocab_size)`` are dropped.

    Examples
    --------
    >>> pack_allowed_ids([0, 33], 64).tolist()
    [1, 2]
    """
    bits = np.zeros(mask_words(vocab_size) * WORD_BITS, dtype=np.uint8)
    bits[_vocab_ids(ids, vocab_size)] = 1
    packed = np.packbits(bits, bitorder="little").view(np.int32)
    return torch.from_numpy(packed.copy())


def clear_ids_(row: torch.Tensor, ids: Sequence[int], vocab_size: int) -> None:
    """Clear, in place, the bits of ``ids`` in the contiguous ``[words]`` row ``row``.

    Examples
    --------
    >>> row = unconstrained_mask(words=2)
    >>> clear_ids_(row, [0, 32], 64)
    >>> row.tolist()
    [-2, -2]
    """
    array = _vocab_ids(ids, vocab_size)
    if not array.size:
        return
    words = row.numpy().view(np.uint32)
    bits = np.left_shift(np.uint32(1), (array % WORD_BITS).astype(np.uint32))
    np.bitwise_and.at(words, array // WORD_BITS, ~bits)


def _bad_word_token(word: Sequence[int], history: Sequence[int]) -> int | None:
    """The token ``word`` bans after ``history``, or ``None``.

    vLLM's rule (``vllm.v1.sample.ops.bad_words``): a word bans its last token
    when the tokens before it end the history; a one-token word always bans
    its token.
    """
    prefix = len(word) - 1
    if prefix > len(history):
        return None
    if prefix and list(history[-prefix:]) != list(word[:prefix]):
        return None
    return int(word[-1])


@dataclass(frozen=True)
class MaskRowControls:
    """What rules tokens out of one live verify row, besides its grammar.

    Attributes
    ----------
    req_id
        The row's request.
    allowed_ids
        ``allowed_token_ids``, or ``None`` when the request has none.
    bad_words
        ``bad_words`` token sequences.
    committed
        The request's committed output, the history a longer bad word is
        matched against.
    drafts
        The row's valid drafts this step.
    min_remaining
        ``min_tokens`` minus the committed output length; 0 once reached.
    stop_ids
        The ids a pending ``min_tokens`` holds back (``all_stop_token_ids``).
    """

    req_id: str
    allowed_ids: Sequence[int] | None = None
    bad_words: Sequence[Sequence[int]] = ()
    committed: Sequence[int] = ()
    drafts: Sequence[int] = ()
    min_remaining: int = 0
    stop_ids: Sequence[int] = ()

    @property
    def constrained(self) -> bool:
        """Whether any control applies to the row."""
        return bool(
            self.allowed_ids
            or self.bad_words
            or (self.min_remaining > 0 and self.stop_ids)
        )


class SpecMaskCache:
    """Per request, the packed row of the controls that never change.

    ``allowed_token_ids`` and the one-token ``bad_words`` of a request apply at
    every column of every step, so their row is packed once. The runner that
    builds steps owns the cache, calls it from that one thread, and drops a
    request's entry when the request finishes (``drop``). Callers copy the
    row; they never write it.
    """

    def __init__(self) -> None:
        self._rows: dict[str, torch.Tensor | None] = {}

    def static_row(
        self, controls: MaskRowControls, vocab_size: int
    ) -> torch.Tensor | None:
        """The cached ``[words]`` row of ``controls``; ``None`` without either."""
        if controls.req_id in self._rows:
            return self._rows[controls.req_id]
        singles = [int(word[0]) for word in controls.bad_words if len(word) == 1]
        row: torch.Tensor | None = None
        if controls.allowed_ids:
            row = pack_allowed_ids(controls.allowed_ids, vocab_size)
        if singles:
            if row is None:
                row = unconstrained_mask(words=mask_words(vocab_size))
            clear_ids_(row, singles, vocab_size)
        self._rows[controls.req_id] = row
        return row

    def drop(self, req_id: str) -> None:
        """Forget a finished request's row."""
        self._rows.pop(req_id, None)

    def prune(self, live: Collection[str]) -> None:
        """Forget the rows of every request not in ``live``."""
        for req_id in self._rows.keys() - set(live):
            del self._rows[req_id]

    def __len__(self) -> int:
        return len(self._rows)


def static_token_mask(
    rows: Sequence[MaskRowControls | None],
    cache: SpecMaskCache,
    *,
    width: int,
    vocab_size: int,
) -> torch.Tensor | None:
    """Every control of a verify's rows but the grammar, per column.

    Parameters
    ----------
    rows
        One entry per row of the verify block, ``None`` for a padding row.
    cache
        The runner's ``SpecMaskCache``.
    width
        The block's ``1 + K`` columns.
    vocab_size
        The vocabulary the mask covers.

    Returns
    -------
    torch.Tensor or None
        ``[len(rows), width, words]`` int32, all ones where nothing applies;
        ``None`` when no row has a control. Column ``j`` of a row is its
        cached row, without the last token of each longer bad word whose
        prefix ends ``committed + drafts[:j]``, and without the stop ids while
        ``j < min_remaining``. The history of column ``j`` holds
        ``min(j, len(drafts))`` drafts, as in the ``logits`` walk.
    """
    if not any(row is not None and row.constrained for row in rows):
        return None
    words = mask_words(vocab_size)
    mask = unconstrained_mask(len(rows), width, words=words)
    for index, controls in enumerate(rows):
        if controls is None or not controls.constrained:
            continue
        static = cache.static_row(controls, vocab_size)
        if static is not None:
            mask[index] = static
        longer = [word for word in controls.bad_words if len(word) > 1]
        for column in range(width):
            if longer:
                history = list(controls.committed) + list(controls.drafts)[:column]
                banned = [
                    token
                    for word in longer
                    if (token := _bad_word_token(word, history)) is not None
                ]
                clear_ids_(mask[index, column], banned, vocab_size)
            if controls.stop_ids and column < controls.min_remaining:
                clear_ids_(mask[index, column], controls.stop_ids, vocab_size)
    return mask


def compose_token_mask(
    static: torch.Tensor | None, grammar: torch.Tensor | None
) -> torch.Tensor | None:
    """The mask a deferred verify samples under: ``static`` AND ``grammar``.

    Both are ``[rows, 1+K, words]`` int32 over the same rows, or ``None``. A
    grammar narrower than ``static`` covers the tokenizer's ids only, and an id
    no grammar names is disallowed: every column of a row the grammar
    constrains anywhere is padded with zero words, as the ``logits`` walk pads
    it (``spec_accept._apply_grammar_bitmask``). Never edits its arguments.

    Examples
    --------
    >>> a = torch.tensor([[[3, -1]]], dtype=torch.int32)
    >>> b = torch.tensor([[[-1, 5]]], dtype=torch.int32)
    >>> compose_token_mask(a, b).tolist()
    [[[3, 5]]]
    """
    if static is None or grammar is None:
        return grammar if static is None else static
    grammar = grammar.to(torch.int32)
    width = int(grammar.shape[-1])
    words = int(static.shape[-1])
    if width > words:
        raise ValueError(
            f"a grammar bitmask of {width} words is wider than the {words} words "
            "of the step's token mask"
        )
    if width < words:
        pad = torch.zeros((*grammar.shape[:-1], words - width), dtype=torch.int32)
        constrained = (grammar != -1).flatten(1).any(dim=1).reshape(-1, 1, 1)
        grammar = torch.cat([grammar, torch.where(constrained, pad, pad - 1)], dim=-1)
    return static & grammar


def mask_may_allow_nothing(
    *,
    structured: bool,
    allowed_ids: Sequence[int] | None,
    bad_words: Sequence[Sequence[int]],
    stop_ids: Sequence[int],
    min_tokens_pending: bool,
    vocab_size: int,
) -> bool:
    """Whether a row's mask can allow no token at some column.

    Decided before the build, from the request's parameters alone, at the
    worst case: every bad word fires. A structured row with any other control
    is at risk too, because its grammar can narrow to the tokens the other
    controls rule out (a finished JSON value allows the end of sequence only,
    which a pending ``min_tokens`` bans). Such a row is not masked on the
    device; its step verifies in ``logits`` mode, whose host walk handles it as
    an ordinary step would.
    """
    banned = {int(word[-1]) for word in bad_words if word}
    if min_tokens_pending:
        banned.update(int(t) for t in stop_ids)
    statics = bool(allowed_ids) or bool(banned)
    if structured:
        return statics
    if allowed_ids:
        return not (set(_vocab_ids(allowed_ids, vocab_size).tolist()) - banned)
    return len(banned) >= vocab_size


def spec_masks_on_device(runner: Any) -> bool:
    """Whether a decode step of ``runner`` may verify under a device token mask.

    The model declares ``supports_spec_device_mask``, the launch speculates
    with an ``argmax_ids`` verify that ``fused_sample`` does not stand in for,
    the step is front-packed (a lane-DP verify keeps its token controls on the
    host) and the launch samples on the device.
    """
    capabilities = getattr(getattr(runner, "model", None), "model_capabilities", None)
    modes = getattr(runner, "_spec_accept_modes", ())
    return bool(
        (capabilities or {}).get(SUPPORTS_SPEC_DEVICE_MASK, False) is True
        and getattr(runner, "_num_speculative_tokens", 0)
        and ACCEPT_MODE_ARGMAX_IDS in modes
        and ACCEPT_MODE_FUSED_SAMPLE not in modes
        and not getattr(runner, "_is_lane_mode", False)
        and getattr(runner, "sample_on_device_mode", None) is not None
    )


def step_static_token_mask(
    runner: Any,
    row_req_ids: Sequence[str],
    drafts: torch.Tensor,
    num_valid: torch.Tensor,
) -> torch.Tensor | None:
    """The build-time part of a token-masked verify's mask, over the padded block.

    Parameters
    ----------
    runner
        The ``TTModelRunner`` building the step; its ``requests`` give each
        live row's parameters and committed output, and its ``vocab_size``
        the mask's width. Its ``SpecMaskCache`` (created on first use) keeps
        the static rows and loses the rows of requests it no longer holds.
    row_req_ids
        The live rows' requests, front-packed: row ``i`` is block row ``i``.
    drafts
        ``[rows, K]`` int32, the block's drafts (rows past the live ones are
        padding).
    num_valid
        ``[rows]`` int32, each row's valid drafts.

    Returns
    -------
    torch.Tensor or None
        ``static_token_mask`` over the ``rows`` rows and ``K + 1`` columns.
    """
    cache = runner.__dict__.get("_spec_mask_cache")
    if cache is None:
        cache = runner._spec_mask_cache = SpecMaskCache()
    cache.prune(runner.requests.keys())
    rows: list[MaskRowControls | None] = [None] * int(drafts.shape[0])
    for row, req_id in enumerate(row_req_ids):
        state = runner.requests[req_id]
        params = state.sampling_params
        output = list(state.output_token_ids)
        remaining = int(params.min_tokens) - len(output)
        bad_words = params.bad_words_token_ids or ()
        rows[row] = MaskRowControls(
            req_id=req_id,
            allowed_ids=params.allowed_token_ids or None,
            bad_words=bad_words,
            committed=output if any(len(word) > 1 for word in bad_words) else (),
            drafts=[int(t) for t in drafts[row, : int(num_valid[row])].tolist()],
            min_remaining=max(remaining, 0),
            stop_ids=sorted(params.all_stop_token_ids or ()) if remaining > 0 else (),
        )
    return static_token_mask(
        rows, cache, width=int(drafts.shape[1]) + 1, vocab_size=runner.vocab_size
    )


__all__ = [
    "MaskRowControls",
    "SpecMaskCache",
    "clear_ids_",
    "compose_token_mask",
    "mask_may_allow_nothing",
    "mask_words",
    "pack_allowed_ids",
    "spec_masks_on_device",
    "static_token_mask",
    "step_static_token_mask",
    "unconstrained_mask",
]
