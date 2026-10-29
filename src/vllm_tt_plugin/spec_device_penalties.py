# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Presence, frequency and repetition penalties of a ``fused_sample`` verify.

A model that declares ``supports_spec_device_penalties`` keeps, on the device,
each penalty row's output-token counts and prompt mask, applies the penalties
to the verify logits inside the fused step and adds every commit to its counts
(docs/SPEC_DECODE_CONTRACT.md, "Device penalties"). The plugin therefore sends
only the per-row parameters on every such step, and the token lists of a row
whose device state is not current.

``SpecPenaltyTracker`` knows which requests the model's state is current for.
A request is current from the step that carried it with a ``rebuild`` entry
(or a later one) for as long as every verify it takes part in is a
``fused_sample`` verify: the model adds each of those steps' commits itself,
also on its own host fallbacks. Any other decode step of the request (a
``logits`` verify, an ordinary decode) commits tokens the model never sees, so
the request stops being current. It also stops being current when it finishes,
is preempted, changes runner slot, or its step is discarded by a forced reset.
The next ``fused_sample`` step then carries a ``rebuild`` entry (prompt ids,
output ids) for it. A chained step carries none: it can only follow a step that
carried the row, so every penalty row of a chain is current
(``TTModelRunner._chain_rows_blocker``).

The counters are module-level, like ``spec_device_logprobs``, because the
submission line reads them from the controller.
"""

import threading
from typing import Any

import torch

_NEUTRAL = (1.0, 0.0, 0.0)  # repetition, frequency, presence: no change to the logits


class _PenaltyStats:
    """Steps that carried a penalty row, and the rows whose state was rebuilt."""

    _FIELDS = ("steps", "rebuilds")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            for name in self._FIELDS:
                setattr(self, name, 0)

    def add(self, **counts: int) -> None:
        with self._lock:
            for name, value in counts.items():
                setattr(self, name, getattr(self, name) + int(value))

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {name: getattr(self, name) for name in self._FIELDS}


_STATS = _PenaltyStats()


def device_penalty_stats() -> dict[str, int]:
    """A copy of the counters: ``steps`` and ``rebuilds``."""
    return _STATS.snapshot()


def reset_device_penalty_stats() -> None:
    _STATS.reset()


class SpecPenaltyTracker:
    """The requests whose device penalty state is current, and each one's runner slot.

    Single-threaded: the runner calls it from the thread that builds and submits
    steps."""

    def __init__(self) -> None:
        self._slot: dict[str, int] = {}

    def is_current(self, req_id: str) -> bool:
        return req_id in self._slot

    def slot_of(self, req_id: str) -> int | None:
        return self._slot.get(req_id)

    def mark_current(self, req_id: str, slot: int) -> None:
        self._slot[req_id] = int(slot)

    def invalidate(self, req_ids: Any) -> None:
        """The model's state is no longer current for these requests (or they ended)."""
        for req_id in req_ids:
            self._slot.pop(req_id, None)

    def clear(self) -> None:
        self._slot.clear()


def non_neutral_rows(
    repetition: torch.Tensor, frequency: torch.Tensor, presence: torch.Tensor
) -> torch.Tensor:
    """``[rows]`` bool: the rows whose penalties change a logit."""
    return (
        (repetition != _NEUTRAL[0])
        | (frequency != _NEUTRAL[1])
        | (presence != _NEUTRAL[2])
    )


def _ids(row: torch.Tensor) -> torch.Tensor:
    """A token-id row with TT's -1 padding removed, as int32."""
    row = row.reshape(-1)
    return row[row >= 0].to(torch.int32).clone()


def device_penalties(
    tracker: SpecPenaltyTracker,
    row_req_ids: list[str],
    repetition: torch.Tensor,
    frequency: torch.Tensor,
    presence: torch.Tensor,
    prompt_tokens: torch.Tensor | None,
    output_tokens: torch.Tensor | None,
    *,
    chained: bool,
    rows: int,
) -> dict[str, Any] | None:
    """The ``penalties`` entry of a ``fused_sample`` verify's ``spec_sampling``.

    ``repetition``, ``frequency`` and ``presence`` are fp32 ``[live]``, one value
    per live row. None when no row has a penalty. Otherwise a dict of the three
    parameter tensors padded to the ``rows`` rows of the block (padding rows
    neutral), plus ``rebuild`` ``{slot: (prompt_ids, output_ids)}`` (int32 1-D)
    for each penalty row whose state is not current or sits in another slot. The
    penalty rows all become current at their slot. A ``chained`` step cannot
    rebuild (its rows' host lists are one step behind), so it raises for a row
    that is not current; a row that only changed slot needs nothing, because the
    model's rows follow their requests through ``slot_remap``.
    """
    live = len(row_req_ids)
    penalized = non_neutral_rows(repetition, frequency, presence)
    if not bool(penalized.any()):
        return None
    rebuild: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for slot in penalized.nonzero().reshape(-1).tolist():
        req_id = row_req_ids[slot]
        if chained:
            # The model's rows follow their requests through a slot remap, so
            # only a row it never counted breaks the chain.
            if not tracker.is_current(req_id):
                raise RuntimeError(
                    f"a chained verify carries penalty row {slot} ({req_id}) whose "
                    "device state is not current; _chain_rows_blocker admits only "
                    "rows the model's state follows"
                )
            continue
        if tracker.slot_of(req_id) == slot:
            continue
        if prompt_tokens is None or output_tokens is None:
            raise RuntimeError(
                f"penalty row {slot} ({req_id}) needs its prompt and output tokens "
                "to rebuild the model's state, and the step carries none"
            )
        rebuild[slot] = (_ids(prompt_tokens[slot]), _ids(output_tokens[slot]))
    for slot in penalized.nonzero().reshape(-1).tolist():
        tracker.mark_current(row_req_ids[slot], slot)

    def padded(values: torch.Tensor, neutral: float) -> torch.Tensor:
        out = torch.full((rows,), neutral, dtype=torch.float32)
        out[:live] = values[:live].to(torch.float32)
        return out

    payload: dict[str, Any] = {
        "repetition": padded(repetition, _NEUTRAL[0]),
        "frequency": padded(frequency, _NEUTRAL[1]),
        "presence": padded(presence, _NEUTRAL[2]),
    }
    if rebuild:
        payload["rebuild"] = rebuild
    _STATS.add(steps=1, rebuilds=len(rebuild))
    return payload
