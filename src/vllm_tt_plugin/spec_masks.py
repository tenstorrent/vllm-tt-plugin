# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The token mask of a ``fused_sample`` verify (``supports_spec_device_mask``).

A model that declares ``supports_spec_device_mask`` applies one packed token
mask per row and candidate column inside its fused step, before its top-k. The
plugin composes that mask from the controls that rule tokens out
(docs/SPEC_DECODE_CONTRACT.md, "Device token masks"):

- the grammar bitmask the scheduler filled (``spec_grammar_bitmask_for_tt_batch``);
- ``allowed_token_ids``: static per request, packed once and cached;
- ``bad_words``: a single-token word is static and cached, a longer word bans
  its last token at a column whose history (committed output + the first ``j``
  drafts) ends with the rest of the word;
- ``min_tokens``: the stop ids, at the columns ``j < remaining``.

Each control only clears bits, so the AND is exact and the order does not
matter. The rules are those of the plugin's own ``logits`` walk
(``spec_accept.apply_speculative_token_filters``), not vLLM's: the device
replaces that walk. A row of the mask is ``[words]`` int32, standard packing
(bit ``v % 32`` of word ``v // 32`` is 1 where token ``v`` is allowed; -1 words
mean no constraint), ``words = ceil(vocab / 32)``.

The split into a build-time part (``static_masks``: everything but the grammar,
which only the sample-time ``GrammarOutput`` has) and ``compose_step_mask``
(AND of the grammar rows into it) follows the step flow: ``execute_model``
enqueues the forward, ``sample_tokens`` composes the mask and finishes the step.

The counters are module-level, like ``spec_device_penalties``, because the
submission line reads them from the controller.
"""

from __future__ import annotations

import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

WORD_BITS = 32

# A masked logit the device writes is ``-2**99`` (its own constant, not read
# here); no real logit is below this. The payload of a masked top-N entry is
# turned back into ``-inf``, which is what the host path gives.
MASK_VALUE_FLOOR = -1e20


def mask_words(vocab_size: int) -> int:
    """Words of one mask row: ``ceil(vocab_size / 32)``."""
    return -(-int(vocab_size) // WORD_BITS)


def all_ones(*lead: int, words: int) -> torch.Tensor:
    """The no-constraint mask ``[*lead, words]`` int32 (every word -1)."""
    return torch.full((*lead, words), -1, dtype=torch.int32)


def is_unconstrained(mask: torch.Tensor) -> torch.Tensor:
    """``[...]`` bool: the row is -1 in every word (no token is ruled out)."""
    return (mask == -1).all(dim=-1)


def empty_rows(mask: torch.Tensor) -> torch.Tensor:
    """``[...]`` bool: the row allows no token (every word is 0)."""
    return (mask == 0).all(dim=-1)


def normalise_width(mask: torch.Tensor, words: int) -> torch.Tensor:
    """``mask`` ``[..., W]`` zero-padded to ``words``: an id no grammar names is
    disallowed.

    The plugin's ``logits`` walk pads the same way (``_grammar_disallowed``). A
    wider mask is an error: it names ids the model has no logit for."""
    width = int(mask.shape[-1])
    if width > words:
        raise ValueError(
            f"a grammar bitmask of {width} words is wider than the {words} words "
            "of the model's vocabulary"
        )
    mask = mask.to(torch.int32)
    if width == words:
        return mask
    pad = torch.zeros((*mask.shape[:-1], words - width), dtype=torch.int32)
    return torch.cat([mask, pad], dim=-1)


def clear_tail_(mask: torch.Tensor, vocab_size: int) -> None:
    """Clear, in place, the bits of ids at or past ``vocab_size``."""
    keep = int(vocab_size) % WORD_BITS
    if keep:
        mask[..., -1] &= (1 << keep) - 1


def _as_ids(ids: Sequence[int], vocab_size: int) -> np.ndarray:
    array = np.asarray(list(ids), dtype=np.int64)
    return array[(array >= 0) & (array < vocab_size)]


def pack_allowed_ids(ids: Sequence[int], vocab_size: int) -> torch.Tensor:
    """``[words]`` int32 mask that allows exactly ``ids`` (ids outside the vocabulary
    are dropped)."""
    words = mask_words(vocab_size)
    bits = np.zeros(words * WORD_BITS, dtype=np.uint8)
    bits[_as_ids(ids, vocab_size)] = 1
    packed = np.packbits(bits, bitorder="little").view(np.int32)
    return torch.from_numpy(packed.copy())


def clear_ids_(row: torch.Tensor, ids: Sequence[int], vocab_size: int) -> None:
    """Clear, in place, the bits of ``ids`` in one ``[words]`` mask row."""
    array = _as_ids(ids, vocab_size)
    if not array.size:
        return
    view = row.numpy().view(np.uint32)
    bit = np.left_shift(np.uint32(1), (array % WORD_BITS).astype(np.uint32))
    np.bitwise_and.at(view, array // WORD_BITS, ~bit)


def bad_word_last_token(word: Sequence[int], history: Sequence[int]) -> int | None:
    """The token a bad word bans after ``history``, or None.

    vLLM's rule (``apply_bad_words``): the word bans its last token when the
    tokens before it are the end of the history. A one-token word always bans
    its token."""
    prefix = len(word) - 1
    if len(word) > len(history) + 1:
        return None
    if prefix > 0 and list(history[-prefix:]) != list(word[:prefix]):
        return None
    return int(word[-1])


@dataclass(frozen=True)
class MaskRowSpec:
    """What rules tokens out of one live row besides its grammar, at build time.

    ``drafts`` are the row's valid drafts and ``committed`` the committed output
    (read only when a bad word is longer than one token). ``min_remaining`` is
    ``min_tokens`` minus the committed length, 0 once reached or without stop
    ids; ``stop_ids`` are the ids ``min_tokens`` holds back."""

    req_id: str
    allowed_ids: Sequence[int] | None = None
    bad_words: Sequence[Sequence[int]] = ()
    committed: Sequence[int] = ()
    min_remaining: int = 0
    stop_ids: Sequence[int] = ()
    drafts: Sequence[int] = ()

    @property
    def constrained(self) -> bool:
        return bool(
            self.allowed_ids
            or self.bad_words
            or (self.min_remaining > 0 and self.stop_ids)
        )


class SpecMaskCache:
    """Per request, the packed mask of what does not change between its steps.

    ``allowed_token_ids`` and the single-token bad words, as one ``[words]``
    row. A request's parameters never change, so the entry is kept until the
    request finishes (``drop``). The row is read-only for its users: they copy
    it into their own tensor. Single-threaded: the runner calls it from the
    thread that builds steps."""

    def __init__(self) -> None:
        self._rows: dict[str, torch.Tensor | None] = {}

    def static_row(self, spec: MaskRowSpec, vocab_size: int) -> torch.Tensor | None:
        """The cached row of ``spec`` (None when it has neither control)."""
        if spec.req_id in self._rows:
            return self._rows[spec.req_id]
        singles = [word[0] for word in spec.bad_words if len(word) == 1]
        row: torch.Tensor | None = None
        if spec.allowed_ids:
            row = pack_allowed_ids(spec.allowed_ids, vocab_size)
        if singles:
            if row is None:
                row = all_ones(words=mask_words(vocab_size))
            clear_ids_(row, singles, vocab_size)
        self._rows[spec.req_id] = row
        return row

    def drop(self, req_id: str) -> None:
        self._rows.pop(req_id, None)


@dataclass(frozen=True)
class StaticMasks:
    """The build-time part of a step's mask: every control but the grammar.

    ``mask`` is ``[live, width, words]`` int32 (all ones where nothing applies);
    ``empty`` lists the ``(row, column)`` positions that allow no token (the
    step then cannot run on the device: the walk would have nothing to draw)."""

    mask: torch.Tensor
    empty: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SpecMaskPlan:
    """A masked ``fused_sample`` step, decided at build time.

    ``static`` is every non-grammar control of the live rows. ``grammar_rows``
    are the live rows whose scheduler bitmask rows apply (a structured request
    whose grammar can constrain this step). ``reasoning_rows`` are structured
    rows that stay inside their open reasoning phase: the scheduler holds all
    ones for them, which ``TTModelRunner._spec_step_mask`` checks."""

    static: StaticMasks
    grammar_rows: tuple[int, ...] = ()
    reasoning_rows: tuple[int, ...] = ()


def static_masks(
    specs: Sequence[MaskRowSpec],
    cache: SpecMaskCache,
    *,
    width: int,
    vocab_size: int,
) -> StaticMasks:
    """Compose every live row's non-grammar controls, per column ``0..width-1``.

    Column ``j`` of a row: the cached static row (allowed ids, one-token bad
    words), AND NOT the last token of each longer bad word whose prefix ends the
    history ``committed + drafts[:j]``, AND NOT the stop ids while
    ``j < min_remaining``. The history of column ``j`` holds ``min(j, valid)``
    drafts, as in the ``logits`` walk."""
    words = mask_words(vocab_size)
    mask = all_ones(len(specs), width, words=words)
    for row, spec in enumerate(specs):
        if not spec.constrained:
            continue
        static = cache.static_row(spec, vocab_size)
        if static is not None:
            mask[row] = static
        longer = [word for word in spec.bad_words if len(word) > 1]
        for column in range(width):
            if longer:
                history = list(spec.committed) + list(spec.drafts)[:column]
                hits = [
                    token
                    for word in longer
                    if (token := bad_word_last_token(word, history)) is not None
                ]
                clear_ids_(mask[row, column], hits, vocab_size)
            if spec.stop_ids and column < max(int(spec.min_remaining), 0):
                clear_ids_(mask[row, column], spec.stop_ids, vocab_size)
    empty = tuple(
        (int(row), int(column)) for row, column in empty_rows(mask).nonzero().tolist()
    )
    return StaticMasks(mask, empty)


def compose_step_mask(
    static: StaticMasks | None,
    grammar: torch.Tensor | None,
    grammar_rows: Sequence[int],
    *,
    rows: int,
    width: int,
    vocab_size: int,
) -> torch.Tensor:
    """The mask of a step: ``[rows, width, words]`` int32 over the padded rows.

    ``static`` (live rows) is copied in; every row of ``grammar_rows`` that the
    scheduler constrains is ANDed with its scheduler row (``grammar``
    ``[live, width, W]``, ``W <= words``, zero-padded). A row, or a column,
    with no control stays all ones. A row that
    gets a constraint loses the bits past the vocabulary. Never edits its
    arguments."""
    words = mask_words(vocab_size)
    out = all_ones(rows, width, words=words)
    if static is not None:
        out[: static.mask.shape[0]] = static.mask
    if grammar is not None:
        for row in grammar_rows:
            # Like the ``logits`` walk (``_apply_grammar_bitmask``): a row with
            # no cleared bit in any column is skipped, any other row is
            # applied in every column, zero-padded to the vocabulary.
            if bool(is_unconstrained(grammar[row]).all()):
                continue
            out[row] &= normalise_width(grammar[row], words)
    constrained = ~is_unconstrained(out)
    if bool(constrained.any()):
        keep = all_ones(words=words)
        clear_tail_(keep, vocab_size)
        out[constrained] &= keep
    return out


def mask_plan_for_step(
    requests: Any,
    row_req_ids: Sequence[str],
    spec_drafts: torch.Tensor,
    num_valid_drafts: torch.Tensor,
    reasoning_ok: Collection[str],
    cache: SpecMaskCache,
    *,
    width: int,
    vocab_size: int,
) -> tuple[SpecMaskPlan | None, str | None]:
    """The mask plan of a ``fused_sample`` step, or why the device cannot run it.

    ``requests`` maps a request id to its ``CachedRequestState``. ``reasoning_ok``
    are the structured rows that stay inside their open reasoning phase
    (``TTModelRunner._grammar_fused_rows``): their grammar is not applied, so on
    their own they need no mask. Returns ``(None, None)`` for a step with no
    control to apply (the step stays a one-call step), ``(plan, None)`` for a
    masked step, and ``(None, "empty_row")`` when a row's non-grammar controls
    leave no token at some column (``StaticMasks.empty``): such a step runs in
    logits mode, which is decided before the step is submitted. Every other
    refusal is the predicate's (``spec_accept.mask_row_refusal``)."""
    specs: list[MaskRowSpec] = []
    grammar_rows: list[int] = []
    reasoning_rows: list[int] = []
    for row, req_id in enumerate(row_req_ids):
        state = requests[req_id]
        params = state.sampling_params
        if params.structured_outputs is not None:
            (reasoning_rows if req_id in reasoning_ok else grammar_rows).append(row)
        valid = int(num_valid_drafts[row])
        remaining = int(params.min_tokens) - len(state.output_token_ids)
        stop_ids = sorted(params.all_stop_token_ids) if remaining > 0 else []
        bad_words = params.bad_words_token_ids or ()
        specs.append(
            MaskRowSpec(
                req_id=req_id,
                allowed_ids=params.allowed_token_ids or None,
                bad_words=bad_words,
                committed=(
                    list(state.output_token_ids)
                    if any(len(word) > 1 for word in bad_words)
                    else ()
                ),
                min_remaining=remaining if stop_ids else 0,
                stop_ids=stop_ids,
                drafts=[int(token) for token in spec_drafts[row, :valid]],
            )
        )
    if not grammar_rows and not any(spec.constrained for spec in specs):
        return None, None
    static = static_masks(specs, cache, width=width, vocab_size=vocab_size)
    if static.empty:
        return None, "empty_row"
    return SpecMaskPlan(static, tuple(grammar_rows), tuple(reasoning_rows)), None


def step_mask(
    plan: SpecMaskPlan,
    grammar: torch.Tensor | None,
    row_req_ids: Sequence[str],
    *,
    rows: int,
    width: int,
    vocab_size: int,
) -> tuple[torch.Tensor | None, int]:
    """The mask to hand ``decode_forward_finish``, and how many live rows it constrains.

    ``grammar`` is the scheduler's bitmask in TT layout ``[live, width, W]``
    (None when the sample-time grammar output has none: the grammar rows are then
    unconstrained, as on the ``logits`` path, where a request an abort or a
    finish removed is absent from it; the sync runner path refuses an absent
    grammar output before it gets here). Raises ``RuntimeError`` when the worker
    and the scheduler disagree: a reasoning row whose scheduler rows are not all
    ones, or a composed row that allows no token. Returns ``(None, 0)`` when
    no row is constrained after all (every grammar row all ones), which the
    model reads as a normal step."""
    if grammar is not None and plan.reasoning_rows:
        held = (~is_unconstrained(grammar[list(plan.reasoning_rows)])).any(dim=1)
        offenders = [
            row_req_ids[row]
            for row, constrained in zip(plan.reasoning_rows, held.tolist())
            if constrained
        ]
        if offenders:
            raise RuntimeError(
                "a fused_sample verify was built for structured-output requests "
                f"{offenders} inside their reasoning phase, but the scheduler's "
                "grammar bitmask constrains their tokens: the worker held their "
                "reasoning phase open and the scheduler did not"
            )
    mask = compose_step_mask(
        plan.static,
        grammar,
        plan.grammar_rows,
        rows=rows,
        width=width,
        vocab_size=vocab_size,
    )
    live = len(row_req_ids)
    empty = empty_rows(mask[:live]).nonzero().tolist()
    if empty:
        raise RuntimeError(
            "the composed token mask allows no token at (row request, column) "
            f"{[(row_req_ids[row], column) for row, column in empty]}: the grammar "
            "row left no token to the request's other controls"
        )
    constrained = (~is_unconstrained(mask[:live])).any(dim=1)
    if not bool(constrained.any()):
        return None, 0
    return mask, int(constrained.sum())


class _MaskStats:
    """Steps that carried a mask, their rows, and the logits fallbacks by reason."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.steps = 0
            self.rows = 0
            self.fallbacks: dict[str, int] = {}

    def add(self, *, steps: int = 0, rows: int = 0) -> None:
        with self._lock:
            self.steps += int(steps)
            self.rows += int(rows)

    def fallback(self, reason: str) -> None:
        with self._lock:
            self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "steps": self.steps,
                "rows": self.rows,
                "fallbacks": dict(self.fallbacks),
            }


_STATS = _MaskStats()


def device_mask_stats() -> dict[str, Any]:
    """A copy of the counters: ``steps``, ``rows``, ``fallbacks`` (reason to count)."""
    return _STATS.snapshot()


def reset_device_mask_stats() -> None:
    _STATS.reset()


def count_mask_step(rows: int) -> None:
    """One masked step with ``rows`` constrained live rows."""
    _STATS.add(steps=1, rows=rows)


def count_mask_fallback(reason: str) -> None:
    """One step that would be a mask step but runs in logits mode, and why."""
    _STATS.fallback(reason)
