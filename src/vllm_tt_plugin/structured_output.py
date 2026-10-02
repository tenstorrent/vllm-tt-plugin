# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


logger = init_tt_logger(__name__)

# Track missing request IDs we've already warned about, so each ID logs once.
_warned_missing_request_ids: set[str] = set()


def has_structured_outputs(
    requests: Mapping[str, CachedRequestState],
    scheduler_output: SchedulerOutput,
    bitmask: torch.Tensor | None,
) -> bool:
    """True if any request scheduled this step constrains its tokens via
    structured outputs: a grammar bitmask, pending structured tokens, or a
    scheduled request carrying ``structured_outputs`` sampling params."""
    if bitmask is not None or scheduler_output.pending_structured_output_tokens:
        return True
    return any(
        (req := requests.get(req_id)) is not None
        and req.sampling_params is not None
        and req.sampling_params.structured_outputs is not None
        for req_id in scheduler_output.num_scheduled_tokens
    )


def reorder_grammar_bitmask_for_tt_batch(
    *,
    bitmask: torch.Tensor,
    structured_output_request_ids: Sequence[str],
    row_req_ids: Sequence[str | None],
    batch_length: int,
) -> torch.Tensor:
    """Reorder scheduler bitmask rows into the TT batch layout.

    Warn once per process for each request ID that is missing a remapped row.
    """
    # region Reorder rows
    grammar_bitmask_length = bitmask.shape[1]
    reordered_bitmask = torch.full(
        (batch_length, grammar_bitmask_length),
        -1,
        dtype=bitmask.dtype,
        device=bitmask.device,
    )

    req_id_to_bitmask_row: dict[str, int] = {
        req_id: i for i, req_id in enumerate(structured_output_request_ids)
    }

    # Collect placeable and placed IDs to log missing IDs if left.
    placeable_ids = {
        req_id
        for req_id in row_req_ids[:batch_length]
        if req_id is not None and req_id in req_id_to_bitmask_row
    }
    placed_ids: set[str] = set()

    for local_row, req_id in enumerate(row_req_ids[:batch_length]):
        scheduler_bitmask_row = req_id_to_bitmask_row.get(req_id)
        if scheduler_bitmask_row is not None:
            reordered_bitmask[local_row, :] = bitmask[scheduler_bitmask_row, :]
            placed_ids.add(req_id)
    # endregion

    # region Log missing request IDs
    missing_ids = placeable_ids - placed_ids
    new_missing_ids = sorted(
        req_id for req_id in missing_ids if req_id not in _warned_missing_request_ids
    )
    if new_missing_ids:
        _warned_missing_request_ids.update(new_missing_ids)
        msg = (
            "Structured-output bitmask remap shortfall: %d new missing "
            "request ID%s did not receive a bitmask row:\n%s"
        )
        args = [
            len(new_missing_ids),
            "" if len(new_missing_ids) == 1 else "s",
            new_missing_ids,
        ]
        logger.warning(msg, *args)
    # endregion

    return reordered_bitmask


def spec_grammar_bitmask_for_tt_batch(
    *,
    bitmask: torch.Tensor,
    structured_output_request_ids: Sequence[str],
    rows_per_request: Mapping[str, int],
    row_req_ids: Sequence[str | None],
    batch_length: int,
    width: int | None,
) -> torch.Tensor:
    """Unpack a speculating launch's bitmask into the TT batch layout.

    On a launch with ``num_speculative_tokens`` set, vLLM's
    ``StructuredOutputManager.grammar_bitmask`` writes each structured request
    ``1 + len(scheduled_spec_decode_tokens[req])`` consecutive rows: row ``j``
    is the grammar after the request's first ``j`` scheduled drafts, and the
    last row is the bonus. An asynchronous launch schedules ``-1``
    placeholders there, and every step of it reserves them, prefill and plain
    decode included, so a request owns several rows even on a step that
    verifies nothing. Row 0 is always the grammar at the request's committed
    output, whatever was scheduled.

    ``width=None`` returns ``[batch_length, W]``, each request's row 0, for a
    step that commits one token per row. ``width=w`` returns
    ``[batch_length, w, W]``: column ``j`` is the request's row ``j`` while it
    has one and all ones past that. A row with no structured request is all
    ones throughout, which is -1.

    Raises when ``rows_per_request`` does not account for the bitmask exactly,
    because a request whose row count is wrong shifts the mask of every
    structured request after it.
    """
    words = int(bitmask.shape[1])
    shape = (batch_length, words) if width is None else (batch_length, width, words)
    unpacked = torch.full(shape, -1, dtype=bitmask.dtype, device=bitmask.device)

    first_row: dict[str, int] = {}
    offset = 0
    for req_id in structured_output_request_ids:
        count = rows_per_request.get(req_id)
        if count is None or count < 1:
            raise RuntimeError(
                f"structured request {req_id} has a grammar bitmask but this "
                f"step recorded {count!r} bitmask rows for it, so its rows "
                "cannot be located"
            )
        first_row[req_id] = offset
        offset += count
    if offset != int(bitmask.shape[0]):
        raise RuntimeError(
            f"the grammar bitmask has {int(bitmask.shape[0])} rows, but the "
            f"scheduled drafts account for {offset} across "
            f"{list(structured_output_request_ids)}; the scheduler and this "
            "step disagree about how many rows each request owns"
        )

    for local_row, req_id in enumerate(row_req_ids[:batch_length]):
        start = first_row.get(req_id) if req_id is not None else None
        if start is None:
            continue
        if width is None:
            unpacked[local_row] = bitmask[start]
            continue
        columns = min(width, rows_per_request[req_id])
        unpacked[local_row, :columns] = bitmask[start : start + columns]
    return unpacked
