# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


def scheduled_structured_output_request_ids(
    requests: Mapping[str, CachedRequestState],
    scheduler_output: SchedulerOutput,
) -> frozenset[str]:
    """Scheduled requests carrying structured-output parameters.

    Decode callers require rows for all returned IDs. Prefill callers can
    exclude intermediate chunks, which must not sample a token.
    """
    return frozenset(
        req_id
        for req_id in scheduler_output.num_scheduled_tokens
        if (req := requests.get(req_id)) is not None
        and req.sampling_params is not None
        and req.sampling_params.structured_outputs is not None
    )


def has_scheduled_structured_outputs(
    requests: Mapping[str, CachedRequestState],
    scheduler_output: SchedulerOutput,
) -> bool:
    """Whether a request scheduled this step carries structured parameters."""
    return bool(scheduled_structured_output_request_ids(requests, scheduler_output))


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
    return has_scheduled_structured_outputs(requests, scheduler_output)


def capture_structured_decode_request_ids(
    scheduled_request_ids: Collection[str],
    row_req_ids: Sequence[str | None],
) -> frozenset[str]:
    """Capture every scheduled structured decode row before submission."""
    captured_ids = frozenset(scheduled_request_ids)
    local_row_ids = {req_id for req_id in row_req_ids if req_id is not None}
    missing_ids = sorted(captured_ids - local_row_ids)
    if missing_ids:
        raise RuntimeError(
            "scheduled structured requests are absent from the submitted TT "
            f"decode rows: {missing_ids}"
        )
    return captured_ids


def reorder_grammar_bitmask_for_tt_batch(
    *,
    bitmask: torch.Tensor,
    structured_output_request_ids: Sequence[str],
    row_req_ids: Sequence[str | None],
    batch_length: int,
    expected_structured_output_request_ids: Collection[str] | None = None,
) -> torch.Tensor:
    """Reorder scheduler bitmask rows into TT row order.

    Supplying ``expected_structured_output_request_ids`` enables strict mode
    and requires complete source and destination coverage. With ``None``, only
    source IDs present in the submitted rows are mapped; other rows remain
    all-allowed for stale async outputs that will be discarded.
    """
    if bitmask.ndim != 2:
        raise ValueError(
            f"grammar bitmask must be rank 2, got shape={tuple(bitmask.shape)}"
        )
    if bitmask.shape[0] != len(structured_output_request_ids):
        raise ValueError(
            "grammar bitmask row count must match request IDs, got "
            f"rows={bitmask.shape[0]}, request_ids={len(structured_output_request_ids)}"
        )
    duplicate_source_ids = sorted(
        req_id
        for req_id, count in Counter(structured_output_request_ids).items()
        if count > 1
    )
    if duplicate_source_ids:
        raise ValueError(
            f"grammar output contains duplicate request IDs: {duplicate_source_ids}"
        )

    strict_identity = expected_structured_output_request_ids is not None
    local_rows = list(row_req_ids[:batch_length])
    if strict_identity and len(local_rows) != batch_length:
        raise ValueError(
            "grammar row identity must cover the full TT batch, got "
            f"rows={len(local_rows)}, batch_length={batch_length}"
        )
    local_rows.extend([None] * (batch_length - len(local_rows)))
    local_row_ids = {req_id for req_id in local_rows if req_id is not None}
    expected_local_ids = (
        set(expected_structured_output_request_ids)
        if strict_identity
        else set(structured_output_request_ids) & local_row_ids
    )
    absent_expected_ids = sorted(expected_local_ids - local_row_ids)
    if absent_expected_ids:
        raise RuntimeError(
            "captured structured request IDs are absent from the submitted TT "
            f"row identity: {absent_expected_ids}"
        )
    duplicate_expected_rows = sorted(
        req_id
        for req_id, count in Counter(
            req_id for req_id in local_rows if req_id in expected_local_ids
        ).items()
        if count > 1
    )
    if duplicate_expected_rows:
        raise RuntimeError(
            "structured requests appear more than once in the TT batch: "
            f"{duplicate_expected_rows}"
        )

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
    missing_ids = sorted(expected_local_ids - req_id_to_bitmask_row.keys())
    if missing_ids:
        raise RuntimeError(
            f"sample-time grammar output is missing TT batch request IDs: {missing_ids}"
        )

    for local_row, req_id in enumerate(local_rows):
        if req_id not in expected_local_ids:
            continue
        scheduler_bitmask_row = req_id_to_bitmask_row.get(req_id)
        if scheduler_bitmask_row is not None:
            reordered_bitmask[local_row, :] = bitmask[scheduler_bitmask_row, :]
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
