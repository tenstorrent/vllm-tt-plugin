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
   verify n row for row (``ChainPlan``), or drains it as before. A row can
   move: when a predecessor row leaves, ``InputBatch.condense`` moves the
   last row into its place, and the build's ``slot_remap`` names, for each
   row, the predecessor row it continues.
2. The build sends each live row as placeholders (``chained_rows``): tokens
   ``PLACEHOLDER_TOKEN_ID``, positions of a predecessor that accepted all its
   drafts, ``num_valid_drafts`` the most drafts the row may verify, count 1,
   and the predecessor's ``VerifyOutput`` as ``spec_predecessor``.
   ``MambaCacheGroups.step_state`` names each count's start block, and the
   window vLLM allocated by the scheduled count, which a drained build of the
   step names for every count too.
3. After the submission the runner reads verify n, commits it and calls the
   model's ``propose_draft_tokens`` for it, whose drafts are the ones step
   n + 1 verified: ``fill_chained_input`` writes the rows step n + 1 really
   ran into its ``TTModelInput``, from which its own accept walk then runs.

A greedy row's ids are the target argmax. On a model that also declares
``spec_ids_are_sampled``, a sampled row chains when step n and step n + 1
both sample on the device; its ids are then the device sampler's draws, and
the model counts acceptance over them as the runner's walk does. Either way
the runner's walk of step n's ids gives the count the device used, and the
model raises from that propose when the runner's commit differs from its own.

A verify holding a row with top-k off and ``top_p`` below 1 on a model
declaring ``supports_spec_device_top_p`` may have rows drawn again from its
device logits when it is read, so no step chains on it (reason ``top_p``) and
no prefill runs behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from vllm_tt_plugin.model_input import spec_verify_holds_top_p_rows
from vllm_tt_plugin.scheduler import (
    get_tt_forced_reset_discard_counts,
    prefill_qualifies_behind_verify,
    scheduler_output_has_prefill_work,
)
from vllm_tt_plugin.spec_admission import (
    SUPPORTS_CHAINED_SPEC_VERIFY,
    SUPPORTS_PREFILL_OVER_PENDING_SPEC,
    SUPPORTS_SPEC_DEVICE_TOP_P,
)
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
    commit gives the same build: a request of the predecessor's rows, the
    scheduler's full ``1 + K`` tokens with ``K`` placeholders, and room for
    two more steps below ``max_model_len``.
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
    if _redraws_on_read(capabilities, previous, runner):
        return None, "top_p"
    if (
        scheduler_output.scheduled_new_reqs
        or scheduler_output.scheduled_cached_reqs.resumed_req_ids
        or scheduler_output.preempted_req_ids
        or get_tt_forced_reset_discard_counts(scheduler_output)
        or scheduler_output_has_prefill_work(scheduler_output)
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
    # A predecessor row whose request finished, or that the step does not
    # schedule (vLLM leaves out a request whose in-flight step reaches
    # max_tokens), leaves the batch, and ``InputBatch.condense`` moves the
    # last row into its place: the chained call's ``slot_remap`` names each
    # row's predecessor row. A request that is not a predecessor row drains.
    finished = set(scheduler_output.finished_req_ids)
    scheduled = set(scheduler_output.num_scheduled_tokens)
    rows = [r for r in previous_rows if r in scheduled and r not in finished]
    if not rows or set(rows) != scheduled:
        return None, "rows"
    num_drafts = int(runner._num_speculative_tokens)
    computed = dict(
        zip(
            scheduler_output.scheduled_cached_reqs.req_ids,
            scheduler_output.scheduled_cached_reqs.num_computed_tokens,
        )
    )
    max_model_len = int(runner.model_config.max_model_len)
    device_sampled = None
    for req_id in rows:
        if scheduler_output.num_scheduled_tokens[req_id] != 1 + num_drafts:
            return None, "width"
        offered = scheduler_output.scheduled_spec_decode_tokens.get(req_id) or ()
        if len(offered) != num_drafts or any(
            t != PLACEHOLDER_TOKEN_ID for t in offered
        ):
            return None, "width"
        if not runner._request_is_argmax_certifiable(req_id, device_sampled=False):
            if device_sampled is None:
                device_sampled = _samples_on_device(
                    runner, scheduler_output, previous, rows
                )
            if not device_sampled or not runner._request_is_argmax_certifiable(
                req_id, device_sampled=True
            ):
                return None, "sampling"
        length = int(batch.num_tokens[batch.req_id_to_index[req_id]])
        if length + 2 * (1 + num_drafts) > max_model_len:
            return None, "max_model_len"
        # vLLM counted the predecessor's drafts as accepted.
        if computed.get(req_id) != length + num_drafts:
            return None, "counts"
    return ChainPlan(predecessor=predecessor), "chained"


def prefill_over_pending(runner: Any, scheduler_output: SchedulerOutput) -> bool:
    """Whether the prefill step about to be built goes behind the one unread
    verify instead of draining it (``supports_prefill_over_pending_spec``).

    The device runs the verify, then the prefill, with no idle between them.
    The verify is read, committed and proposed after the prefill, at the
    next build, as any completed step is; the step after the prefill finds
    it read and does not chain on it. The rule for the step is
    ``TTScheduler``'s (``prefill_qualifies_behind_verify``), which returned
    the prefill without a release step; the verify must be an ``argmax_ids``
    verify the model returned unread, and the only pending step.
    """
    from vllm_tt_plugin.async_decode import AsyncTTSpecDecodeOutput

    if not (
        getattr(runner, "async_decode_scheduling", False)
        and getattr(runner, "_num_speculative_tokens", 0)
        and getattr(runner, "_spec_drafts_from_model", False)
    ):
        return False
    model = getattr(runner, "model", None)
    capabilities = getattr(model, "model_capabilities", None) or {}
    if capabilities.get(SUPPORTS_PREFILL_OVER_PENDING_SPEC, False) is not True:
        return False
    if not prefill_qualifies_behind_verify(scheduler_output):
        return False
    with runner._steady_decode_lock:
        pending = list(runner._pending_async_steps)
    if len(pending) != 1 or not isinstance(pending[0], AsyncTTSpecDecodeOutput):
        return False
    verify = pending[0]
    return (
        not verify.is_resolved()
        and verify.chain_filled
        and verify.model_input.spec_mode == ACCEPT_MODE_ARGMAX_IDS
        and verify.model_input.live_rows is None
        and verify.verify_output is not None
        and not _redraws_on_read(capabilities, verify.model_input, runner)
    )


def _redraws_on_read(
    capabilities: dict[str, Any], model_input: TTModelInput, runner: Any
) -> bool:
    """Whether the model may draw rows of this unread verify again when it is
    read (``supports_spec_device_top_p``: a row with top-k off and ``top_p``
    below 1 whose nucleus reaches past the device's candidates), from device
    logits that a chained verify or a prefill behind it would overwrite."""
    return capabilities.get(
        SUPPORTS_SPEC_DEVICE_TOP_P, False
    ) is True and spec_verify_holds_top_p_rows(model_input, int(runner.vocab_size))


def _samples_on_device(
    runner: Any,
    scheduler_output: SchedulerOutput,
    previous: TTModelInput,
    rows: list[str],
) -> bool:
    """Whether the unread verify and the step about to be built both sample
    on the device, so that a sampled row's ids are draws in both: the count
    the model continues from is its acceptance over the predecessor's draws,
    and the chained verify answers with draws the runner's walk certifies.

    Read before ``_update_states``, over the chained rows of a persistent
    batch that still holds the finished ones; the build decides again over
    the rows alone and raises if its decision does not certify every chained
    row (``TTModelRunner._spec_decode_rows``).
    """
    if not previous.perform_device_sampling:
        return False
    batch = runner.input_batch
    return bool(
        runner._decode_device_sampled(
            scheduler_output, None, [batch.req_id_to_index[r] for r in rows]
        )
    )


def check_chained_rows_certifiable(
    runner: Any, row_req_ids: list[str], device_sampled: bool
) -> None:
    """Raise unless the id walk of the step being built certifies every
    chained row under the step's own device-sampling decision
    (``TTModelRunner._decode_device_sampled``), which the submission uses;
    ``plan_chain`` decided over the persistent batch before
    ``_update_states``."""
    uncertified = [
        r
        for r in row_req_ids
        if not runner._request_is_argmax_certifiable(r, device_sampled)
    ]
    if uncertified:
        raise RuntimeError(
            f"chained verify rows {uncertified} are not certifiable by the id "
            f"walk of a step with device_sampled={device_sampled}"
        )


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
    "check_chained_rows_certifiable",
    "fill_chained_input",
    "plan_chain",
    "prefill_over_pending",
]
