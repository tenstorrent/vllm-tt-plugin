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
