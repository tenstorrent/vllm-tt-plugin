# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Chained speculative verification (``supports_chained_spec_verify``).

Under asynchronous scheduling vLLM schedules step n + 1 before the engine
reads step n. A verify is otherwise drained before the next build, because
the next candidate block starts from the token and the state the verify
commits. A model declaring ``supports_chained_spec_verify`` builds the next
verify's rows on the device from its own unread result instead, so the runner
can submit step n + 1 over an unread verify n:

1. ``plan_chain`` decides, before the build, whether step n + 1 continues
   verify n row for row (``ChainPlan``), or drains it as before.
2. The build sends each live row as placeholders (``chained_rows``): tokens
   ``PLACEHOLDER_TOKEN_ID``, positions of a predecessor that accepted all its
   drafts, ``num_valid_drafts`` the most drafts the row may verify, count 1,
   and the predecessor's ``VerifyOutput`` as ``spec_predecessor``.
   ``MambaCacheGroups.step_state`` names each count's start block.
3. After the submission the runner reads verify n, commits it and calls the
   model's ``propose_draft_tokens`` for it, whose drafts are the ones step
   n + 1 verified: ``fill_chained_input`` writes the rows step n + 1 really
   ran into its ``TTModelInput``, from which its own accept walk then runs.

The rows are greedy only (``argmax_ids``), so the runner's walk of step n's
ids gives the count the device used; the model raises from that propose when
the runner's commit differs from its own.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from vllm_tt_plugin.scheduler import get_tt_forced_reset_discard_counts
from vllm_tt_plugin.spec_admission import SUPPORTS_CHAINED_SPEC_VERIFY
from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, PLACEHOLDER_TOKEN_ID
from vllm_tt_plugin.structured_output import has_structured_outputs

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_tt_plugin.model_input import TTModelInput


@dataclass(frozen=True)
class ChainPlan:
    """Step n + 1 continues the unread verify ``predecessor`` row for row."""

    # The predecessor's ``AsyncTTSpecDecodeOutput``.
    predecessor: Any


def plan_chain(
    runner: Any, scheduler_output: SchedulerOutput
) -> tuple[ChainPlan | None, str]:
    """Whether the step about to be built chains on the one unread verify.

    Read before ``_update_states``: the persistent batch still holds the
    predecessor's rows. Returns the plan, or ``None`` and the reason the step
    drains instead. A row chains only when every count the predecessor can
    commit gives the same build: the same rows at the same indices, the
    scheduler's full ``1 + K`` tokens with ``K`` placeholders, room for two
    more steps below ``max_model_len``, and, in mode ``align``, the same
    window column (``MambaCacheGroups.chain_window_safe``).
    """
    from vllm_tt_plugin.async_decode import AsyncTTSpecDecodeOutput

    if not (
        getattr(runner, "async_decode_scheduling", False)
        and getattr(runner, "_num_speculative_tokens", 0)
        and getattr(runner, "_spec_drafts_from_model", False)
    ):
        return None, "off"
    model = getattr(runner, "model", None)
    capabilities = getattr(model, "model_capabilities", None) or {}
    if capabilities.get(SUPPORTS_CHAINED_SPEC_VERIFY, False) is not True:
        return None, "off"
    with runner._steady_decode_lock:
        pending = list(runner._pending_async_steps)
        completed = list(runner._completed_decode_steps)
    if len(pending) != 1 or not isinstance(pending[0], AsyncTTSpecDecodeOutput):
        return None, "pending"
    predecessor = pending[0]
    if any(getattr(c, "source", None) is not predecessor for c in completed):
        return None, "pending"
    previous = predecessor.model_input
    if previous.spec_mode != ACCEPT_MODE_ARGMAX_IDS or previous.live_rows is not None:
        return None, "mode"
    if (
        scheduler_output.scheduled_new_reqs
        or scheduler_output.scheduled_cached_reqs.resumed_req_ids
        or scheduler_output.preempted_req_ids
        or get_tt_forced_reset_discard_counts(scheduler_output)
        or runner.async_decode.scheduler_output_has_prefill_work(scheduler_output)
    ):
        return None, "lifecycle"
    if scheduler_output.pending_structured_output_tokens or has_structured_outputs(
        runner.requests, scheduler_output, None
    ):
        return None, "grammar"
    previous_rows = list(previous.row_req_ids or ())
    batch = runner.input_batch
    if list(batch.req_ids[: batch.num_reqs]) != previous_rows:
        return None, "rows"
    finished = set(scheduler_output.finished_req_ids)
    rows = [r for r in previous_rows if r not in finished]
    # Finished rows leave the batch; only a suffix leaves the others in place.
    if not rows or previous_rows[: len(rows)] != rows:
        return None, "rows"
    if set(scheduler_output.num_scheduled_tokens) != set(rows):
        return None, "rows"
    num_drafts = int(runner._num_speculative_tokens)
    computed = dict(
        zip(
            scheduler_output.scheduled_cached_reqs.req_ids,
            scheduler_output.scheduled_cached_reqs.num_computed_tokens,
        )
    )
    max_model_len = int(runner.model_config.max_model_len)
    mamba = runner._mamba_groups
    for req_id in rows:
        if scheduler_output.num_scheduled_tokens[req_id] != 1 + num_drafts:
            return None, "width"
        offered = scheduler_output.scheduled_spec_decode_tokens.get(req_id) or ()
        if len(offered) != num_drafts or any(
            t != PLACEHOLDER_TOKEN_ID for t in offered
        ):
            return None, "width"
        if not runner._request_is_argmax_certifiable(req_id, device_sampled=False):
            return None, "sampling"
        length = int(batch.num_tokens[batch.req_id_to_index[req_id]])
        if length + 2 * (1 + num_drafts) > max_model_len:
            return None, "max_model_len"
        # vLLM counted the predecessor's drafts as accepted.
        if computed.get(req_id) != length + num_drafts:
            return None, "counts"
        if mamba is not None and not mamba.chain_window_safe(
            length + num_drafts, 1 + num_drafts
        ):
            return None, "mamba_boundary"
    return ChainPlan(predecessor=predecessor), "chained"


def chained_rows(
    input_positions: torch.Tensor, num_drafts: int, draft_cap: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(tokens, positions, drafts, num_valid_drafts, accepted_counts)`` of a
    chained step's live rows, every one a placeholder the model replaces.

    ``input_positions`` is each row's plain decode position: the predecessor's
    column-0 position, since its commit is not applied. The positions sent are
    those of a predecessor that accepted all its drafts, the ones vLLM
    scheduled; ``num_valid_drafts`` is the most the row may verify.
    """
    rows = int(input_positions.shape[0])
    width = 1 + num_drafts
    offsets = torch.arange(width, dtype=input_positions.dtype)
    positions = (input_positions + width).unsqueeze(1) + offsets.unsqueeze(0)
    tokens = torch.full((rows, width), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    drafts = torch.full((rows, num_drafts), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    num_valid = torch.full((rows,), int(draft_cap), dtype=torch.int32)
    counts = torch.ones(rows, dtype=torch.int32)
    return tokens, positions, drafts, num_valid, counts


def fill_chained_input(
    successor: TTModelInput,
    predecessor: TTModelInput,
    committed: torch.Tensor,
    counts: torch.Tensor,
    draft_token_ids: torch.Tensor,
    offered: list[int] | None,
    max_model_len: int,
) -> TTModelInput:
    """The chained step's input as the model ran it.

    ``committed`` and ``counts`` are the predecessor's walk, and
    ``draft_token_ids`` and ``offered`` its ``DraftOutput``: the drafts the
    model's drafter made from that commit, which the chained step verified.
    Each successor row takes the predecessor's last committed token at the
    position after its commit, then ``min(num_valid_drafts, offered)`` of
    those drafts, as ``_spec_candidate_block`` lays out a verify row.
    """
    num_drafts = int(draft_token_ids.shape[1])
    previous_rows = {r: i for i, r in enumerate(predecessor.row_req_ids or ())}
    tokens = successor.input_tokens.clone()
    positions = successor.input_positions.clone()
    drafts = successor.draft_token_ids.clone()
    num_valid = successor.num_valid_drafts.clone()
    accepted = successor.accepted_counts.clone()
    for row, req_id in enumerate(successor.row_req_ids or ()):
        prev = previous_rows.get(req_id)
        if prev is None:
            raise RuntimeError(
                f"request {req_id} is a row of a chained verify but not of its "
                f"predecessor's rows {list(previous_rows)}"
            )
        count = int(counts[prev])
        if not 1 <= count <= 1 + num_drafts:
            raise RuntimeError(
                f"request {req_id}'s predecessor committed {count} tokens"
            )
        limit = num_drafts if offered is None else int(offered[prev])
        n = min(int(num_valid[row]), limit)
        first = int(predecessor.input_positions[prev, 0]) + count
        if first + 1 + n > max_model_len:
            raise RuntimeError(
                f"request {req_id}'s chained verify reaches position "
                f"{first + n} at max_model_len {max_model_len}"
            )
        tokens[row] = PLACEHOLDER_TOKEN_ID
        tokens[row, 0] = committed[prev, count - 1]
        tokens[row, 1 : n + 1] = draft_token_ids[prev, :n]
        positions[row] = -1
        positions[row, : n + 1] = first + torch.arange(n + 1, dtype=positions.dtype)
        drafts[row] = PLACEHOLDER_TOKEN_ID
        drafts[row, :n] = draft_token_ids[prev, :n]
        num_valid[row] = n
        accepted[row] = count
    return replace(
        successor,
        input_tokens=tokens,
        input_positions=positions,
        draft_token_ids=drafts,
        num_valid_drafts=num_valid,
        accepted_counts=accepted,
    )


__all__ = [
    "ChainPlan",
    "chained_rows",
    "fill_chained_input",
    "plan_chain",
]
