# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, cast

import torch
import ttnn
from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsLists, ModelRunnerOutput

from vllm_tt_plugin.input_batch import SEED_NONE_SENTINEL
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.model_input import TTCompactedHostLogits
from vllm_tt_plugin.scheduler import (
    TT_VALID_DRAFTS_ATTR,
    get_tt_finishing_req_ids,
    get_tt_forced_reset_discard_counts,
)
from vllm_tt_plugin.spec_accept import (
    spec_chain_enabled,
    spec_chain_sampled_enabled,
    spec_prefill_overlap_enabled,
    support_fallback_stats,
)
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_FUSED_SAMPLE,
    ACCEPT_MODE_LOGITS,
    ACCEPT_MODE_TOPK,
    MODE_REQUIRED_FIELDS,
    PLACEHOLDER_TOKEN_ID,
    FusedReadout,
    VerifyOutput,
)
from vllm_tt_plugin.spec_device_logprobs import device_logprob_stats
from vllm_tt_plugin.spec_device_penalties import device_penalty_stats
from vllm_tt_plugin.spec_masks import count_mask_wait, device_mask_stats
from vllm_tt_plugin.spec_specials import count_step_specials, specials_report
from vllm_tt_plugin.stage_outputs import TTStageFrame, pop_tt_stage_outputs
from vllm_tt_plugin.structured_output import has_structured_outputs


def _verify_output_tensor(
    tt_out: Any, model_name: str, spec_mode: str
) -> torch.Tensor | VerifyOutput:
    """What a verify's mode returns, unwrapped from ``VerifyOutput``.

    A step that sent ``spec_mode`` asked the model to verify a candidate
    block, and a ``VerifyOutput`` is the only answer to that question. A model
    that declares ``supports_spec_decode`` and whose ``decode_forward`` does
    not implement the verify answers with a plain decode's return instead, and
    that return survives every check further down: device sampling produces a
    ``[B, 1]`` id tensor, which the accept walk reads as a verify claiming one
    token per row, so the run commits one token per step for its whole life
    and reports nothing. Refused by type here, where the answer arrives.

    The answer must be in the mode that was asked for. A model that answers in
    another one is refused by name rather than having its tensor read as the
    requested mode's: ``[B, 1+K, V]`` logits read as ids would commit
    vocabulary indices of float rows, and ``[B, 1+K]`` ids read as logits
    would be sampled over a block-width vocabulary. ``VerifyOutput`` has
    already checked that the field its mode declares is present.

    ``argmax_ids`` and ``logits`` each return one tensor. ``topk`` returns the
    ``VerifyOutput`` itself, and so does ``fused_sample`` (its ids and counts):
    its walk reads three fields and the lazy
    ``fetch_logits`` callable, and the output already pairs them.
    """
    if not isinstance(tt_out, VerifyOutput):
        raise TypeError(
            f"TT model {model_name} was asked to verify in spec_mode "
            f"{spec_mode!r} and its decode_forward returned "
            f"{type(tt_out).__name__}; a verify returns a VerifyOutput from "
            "vllm_tt_plugin.spec_decode, so a model whose decode_forward "
            "serves no verify must not declare supports_spec_decode"
        )
    if tt_out.spec_mode != spec_mode:
        raise NotImplementedError(
            f"TT decode asked for spec_mode {spec_mode!r} and the model "
            f"answered in {tt_out.spec_mode!r}, which carries "
            f"{list(MODE_REQUIRED_FIELDS[tt_out.spec_mode])}; a verify answers "
            "in the mode it was asked for"
        )
    if spec_mode == ACCEPT_MODE_ARGMAX_IDS:
        return tt_out.argmax_ids
    if spec_mode == ACCEPT_MODE_LOGITS:
        return tt_out.logits
    if spec_mode in (ACCEPT_MODE_TOPK, ACCEPT_MODE_FUSED_SAMPLE):
        return tt_out
    raise NotImplementedError(
        f"the runner drives no accept walk for spec_mode {spec_mode!r}"
    )


if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_tt_plugin.model_input import TTDecodeReloadPlan, TTModelInput
    from vllm_tt_plugin.model_runner import TTModelRunner

logger = init_tt_logger(__name__)

# How often the submission counters are reported. A benchmark diffs the last
# line before its measured interval against the last one after, so the cadence
# bounds the error on that diff; a step costs about a millisecond on a model
# with no device work, so 128 is a fraction of a second.
_SUBMISSION_LOG_INTERVAL = 128

# The ``spec_specials`` path name of each accept mode; no mode is an ordinary decode.
_SPECIALS_PATHS = {
    ACCEPT_MODE_FUSED_SAMPLE: "fused",
    ACCEPT_MODE_LOGITS: "logits",
    ACCEPT_MODE_TOPK: "topk",
    ACCEPT_MODE_ARGMAX_IDS: "ids",
}


@dataclass(frozen=True)
class TTDecodeSubmission:
    """Carries the raw result of decode submission until finalization."""

    tt_out: Any | None
    read_events: list[Any] | None
    batch_size_per_dp: list[int]
    sampling_params: Any
    perform_device_sampling: bool
    device_sampling_params: Any | None = None
    deferred_device_sampling: bool = False
    reload_plan: TTDecodeReloadPlan | None = None
    # The verify's opaque hidden handle, for a model whose drafter consumes it.
    # Carried rather than stored on the runner so it cannot outlive the step it
    # belongs to: the runner hands it straight back to ``propose_draft_tokens``
    # without interpreting its dtype, layout or tensor-parallel fracturing.
    spec_hidden: Any | None = None
    readback_rows: tuple[int, ...] | None = None
    # A stage model's stage frames from this forward, popped right after it.
    stage_outputs: dict[str, TTStageFrame] | None = None
    # A masked fused verify (``TTModelInput.spec_mask``) after its first half:
    # the model has the forward but not the token mask, so nothing may read it
    # until ``finish_masked_decode``.
    mask_pending: bool = False


@dataclass(frozen=True)
class TTFinalizedDecode:
    """Normalized decode result after TT event waits and host processing."""

    # A ``topk`` verify finalizes as its ``VerifyOutput``: the accept walk reads
    # three of its fields and its ``fetch_logits`` callable, so there is no one
    # tensor to normalize it to.
    tt_out: torch.Tensor | TTCompactedHostLogits | VerifyOutput
    tt_log_probs: torch.Tensor | None
    # Chain mode: the device-reported position of each row's column 0, read
    # with the deferred fused verify's handle. None for every other step.
    p0: torch.Tensor | None = None
    # The drafts the next chained step verifies (``FusedReadout.drafts``),
    # ``[B, K]`` int32 in runner row order; None when the model reports none.
    drafts: torch.Tensor | None = None


@dataclass(frozen=True)
class ChainResolved:
    """What the reader thread knows about a resolved fused step, by its tip id.

    A chained masked step needs it before the engine thread commits the step:
    ``drafts`` go to the scheduler (``take_draft_token_ids``) and ``prefixes``
    (the committed tokens of each row, in ``row_req_ids`` order) give the
    output the request's min_tokens and bad words must see at sample time.
    """

    tip: int
    row_req_ids: tuple[str, ...]
    prefixes: list[list[int]]
    drafts: torch.Tensor | None


# How many resolved steps the controller remembers. A chained masked step reads
# the record of the step before it, so one is enough; the rest is slack for a
# step that is read late.
_RESOLVED_KEEP = 4


@dataclass(frozen=True)
class ChainPlan:
    """What a chained fused build needs, decided once per step by the controller.

    ``tip`` is the id of the pending fused step being chained onto,
    ``chain_len`` each request's optimistic length (applied length plus K+1 per
    verify in flight) and ``hints`` how many drafts its block may carry.
    """

    tip: int
    chain_len: dict[str, int]
    hints: dict[str, int]


def _device_verify_fields(verify: VerifyOutput) -> dict[str, str]:
    """The fields ``verify``'s mode reads that are not host tensors, by type name.

    ``hidden`` is an opaque handle the walk never reads and ``fetch_logits`` a
    callable, so neither is a field here.
    """
    return {
        name: type(getattr(verify, name)).__name__
        for name in MODE_REQUIRED_FIELDS[verify.spec_mode]
        if not isinstance(getattr(verify, name), torch.Tensor)
    }


def _is_host_decode_output(tt_out: Any) -> bool:
    if isinstance(tt_out, torch.Tensor):
        return True
    if isinstance(tt_out, tuple):
        return all(
            tensor is None or isinstance(tensor, torch.Tensor) for tensor in tt_out
        )
    if isinstance(tt_out, VerifyOutput):
        # A verify's return reaches the accept walk as it is (``topk`` is the
        # mode that is not unwrapped to one tensor), so it is host output when
        # every field its mode reads already is.
        return not _device_verify_fields(tt_out)
    return False


@dataclass(frozen=True)
class SubmittedStepContext:
    """Immutable snapshot of the host state associated with one decode submit."""

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    submit_time_ns: int


@dataclass
class CompletedDecodeStep:
    """Sampled output that has completed but is not yet applied to host state.

    Most instances are async decode readbacks. Async-scheduled final prefills
    also use this lifecycle so a prefix reset can discard their published frame
    before it reaches runner state.
    """

    sampled_token_ids: torch.Tensor
    logprobs: LogprobsLists | None
    context: SubmittedStepContext
    completion_time_ns: int
    runner_output: ModelRunnerOutput | None = None
    # Kept for a speculating launch only, whose draftless decode steps owe the
    # next proposal at their commit: the drafter is called with this step's own
    # rows and positions, and the commit runs a step later than the build.
    model_input: TTModelInput | None = None


@dataclass
class CompletedSpecDecodeStep:
    """An accepted speculative step, not yet applied to host state.

    A speculative step cannot be finished where an ordinary one is. Its output
    is not a sampled token but a candidate block whose committed length is
    decided by the accept walk, and the walk's result then has to reach two
    places that only the engine thread may touch: the request's token history,
    and the next proposal. So the walk runs at completion and everything it
    produced waits here until the next step drains it.

    What it retains is what the commit and the proposal need after the runner
    has moved on: the accepted block, each row's count, the step's own
    ``model_input`` (its request identities, positions and drafts), and the
    verify's hidden handle. Every tensor here belongs to this step. The
    builder allocates the candidate block, the draft block and both side
    tensors fresh per step, so none of them is a view into storage the next
    step reuses.
    """

    committed: torch.Tensor
    counts: torch.Tensor
    prefixes: list[list[int]]
    model_input: TTModelInput
    spec_hidden: Any | None
    context: SubmittedStepContext
    completion_time_ns: int
    runner_output: ModelRunnerOutput | None = None
    # The published prefixes' logprobs, when the step's rows asked for them.
    logprobs: LogprobsLists | None = None
    # Chain mode: the device-reported column-0 position of each row.
    p0: torch.Tensor | None = None


class DeferredDecodeOutput(AsyncModelRunnerOutput):
    """Run the deferred device readback exactly once, from whichever caller
    reaches it first.

    The executor resolves the step through ``get_output``; the runner's drain
    resolves it through ``ensure_finalized``. Depending on the executor, these
    calls can share a thread or race across threads. ``_finalize_lock`` makes
    either ordering resolve the readback exactly once; a second readback
    of the same device submission corrupts the decode output. The completion
    event is set here, when the readback actually runs, not only inside
    ``get_output``. That is the invariant the drain depends on: vLLM 0.22's
    ``step_with_batch_queue`` schedules the next batch before resolving the
    prior future, so a drain that merely ``event.wait()``-ed would block forever
    on an event nothing else has reached yet.
    """

    _completion_event: threading.Event
    _finalize_lock: threading.Lock
    _finalized: bool
    _cached_output: Any
    _cached_exception: BaseException | None

    def _init_deferred(self) -> None:
        self._finalized = False
        self._cached_output = None
        self._cached_exception = None
        self._finalize_lock = threading.Lock()

    def ensure_finalized(self) -> Any:
        if self._finalized:
            return self._replay()
        with self._finalize_lock:
            if not self._finalized:
                # A failed readback still consumed this submission. Cache the
                # failure as terminal so a racing drain cannot perform the same
                # non-idempotent device readback a second time.
                try:
                    self._cached_output = self._get_output_impl()
                except BaseException as exc:
                    self._cached_exception = exc
                    raise
                finally:
                    self._finalized = True
                    self._completion_event.set()
        return self._replay()

    def _replay(self) -> Any:
        if self._cached_exception is not None:
            raise self._cached_exception
        return self._cached_output

    def is_resolved(self) -> bool:
        return self._completion_event.is_set()

    def get_output(self) -> Any:
        return self.ensure_finalized()

    def _get_output_impl(self) -> Any:
        raise NotImplementedError


class AsyncTTModelRunnerOutput(DeferredDecodeOutput):
    """Wrap a non-blocking single-process TT decode submission plus async read.

    Handles both a plain single-process decode and a lane-DP decode: when
    ``scheduled_rows`` is set the read-back goes through the merged lane batch
    (``TTLaneInputBatch.extract_output``), otherwise through ``_get_output_tokens``.
    """

    def __init__(
        self,
        controller: TTAsyncDecodeController,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
        completion_event: threading.Event,
        context: SubmittedStepContext,
        scheduled_rows: list[int] | None = None,
    ):
        self._controller = controller
        self._submission = submission
        self._model_input = model_input
        self._completion_event = completion_event
        self._context = context
        self._scheduled_rows = scheduled_rows
        self._init_deferred()

    def set_grammar_bitmask(self, bitmask: torch.Tensor) -> None:
        """Attach a sample-time grammar bitmask before the deferred read.

        The runner reorders the bitmask on the engine thread (where the batch
        layout still matches this step's forward) and calls this; the read on
        the output thread then applies it through ``model_input``.
        """
        self._model_input = replace(self._model_input, grammar_bitmask=[bitmask])

    def _get_output_impl(self) -> ModelRunnerOutput:
        completed = self._controller.complete_decode_step(
            submission=self._submission,
            model_input=self._model_input,
            context=self._context,
            scheduled_rows=self._scheduled_rows,
        )
        runner_output = self._controller.build_runner_output_from_completed_step(
            completed
        )
        self._controller.tag_valid_drafts(runner_output, self._model_input)
        completed.runner_output = runner_output
        self._controller.enqueue_completed_decode_step(completed)
        return runner_output


class AsyncTTSpecDecodeOutput(DeferredDecodeOutput):
    """A deferred speculative step: read back, then walk acceptance.

    Shares ``DeferredDecodeOutput``'s exactly-once resolution, because the same
    resolution entry points can race for a speculative step as for an ordinary
    one, and a second readback of the same submission corrupts it. Resolution
    differs: the accept walk instead of the sampler, and a published output
    whose rows carry a committed prefix each rather than one token apiece.

    It never mutates runner state. The walk reads only this step's own tensors,
    and the commit waits in the completed queue for the engine thread.
    """

    def __init__(
        self,
        controller: TTAsyncDecodeController,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
        completion_event: threading.Event,
        context: SubmittedStepContext,
    ):
        self._controller = controller
        self._submission = submission
        self._model_input = model_input
        self._completion_event = completion_event
        self._context = context
        self._init_deferred()

    def set_grammar_bitmask(
        self, bitmask: torch.Tensor, *, num_valid_drafts: torch.Tensor
    ) -> None:
        """Attach the per-column bitmask and the drafts it can constrain.

        ``bitmask`` is ``[rows, 1+K, W]``. ``num_valid_drafts`` is the
        submitted count with every structured row whose drafts the bitmask was
        not computed from set to 0, which
        ``TTModelRunner._grammar_validated_drafts`` decides; the walk reads
        both off ``model_input``. Only an ``argmax_ids`` verify has no use for
        a grammar, and no structured row reaches one.
        """
        if self._model_input.spec_mode == ACCEPT_MODE_ARGMAX_IDS:
            raise RuntimeError(
                "a grammar bitmask reached an argmax_ids verify, whose ids "
                "cannot be masked; a structured-output row makes its verify "
                "ask for logits"
            )
        self._model_input = replace(
            self._model_input,
            grammar_bitmask=[bitmask],
            num_valid_drafts=num_valid_drafts,
        )

    def finish_masked_step(self, mask: torch.Tensor | None) -> None:
        """Second half of a masked step, on the engine thread, before the
        output is handed to anything that can read it."""
        self._submission = self._controller.finish_masked_decode(
            self._submission, mask, async_read=True
        )

    def _get_output_impl(self) -> ModelRunnerOutput:
        completed = self._controller.complete_spec_decode_step(
            submission=self._submission,
            model_input=self._model_input,
            context=self._context,
        )
        runner_output = self._controller.runner.build_spec_runner_output(
            self._model_input.row_req_ids, completed.prefixes, completed.logprobs
        )
        self._controller.tag_valid_drafts(runner_output, self._model_input)
        completed.runner_output = runner_output
        self._controller.enqueue_completed_decode_step(completed)
        return runner_output


class TTAsyncDecodeController:
    """Own the TT async decode lifecycle for a `TTModelRunner`."""

    def __init__(self, runner: TTModelRunner):
        self.runner = runner
        # Decode residency is committed at submission, not readback. During
        # overlapped device sampling the host can intentionally remain one step
        # behind, so submitted-device history is the only safe reload authority.
        self._decode_chain_valid = False
        self._previous_device_sampling: bool | None = None
        self._previous_submission_verified = False
        self._submitted_page_tables: tuple[torch.Tensor, ...] | None = None
        self._legacy_contract_warning_emitted = False
        self.device_grammar_sample_count = 0
        # How many submissions found a step already outstanding, and how many
        # of those were not overlap-safe. The second is the serialization
        # claim: a verify must never be submitted over an outstanding step, so
        # this stays at zero for the life of the server.
        self._overlapped_submissions = 0
        self._overlapped_unsafe_submissions = 0
        # What was actually submitted, by kind. Read by the overhead
        # benchmark, which cannot take these from the speculative metrics: those
        # count the scheduler's lookahead reservation rather than a verify.
        self._ordinary_decode_submissions = 0
        self._verify_submissions = 0
        # The verifies among those that asked for logits, which is the one
        # observable that says a launch serving both modes chose per step.
        self._logits_verify_submissions = 0
        self._topk_verify_submissions = 0
        self._fused_verify_submissions = 0
        # Chain mode (VLLM_TT_SPEC_CHAIN). The newest fused step that was
        # submitted with a deferred handle, its rows and its tip id: the only
        # step a following verify may chain onto. Cleared by any later step.
        self._last_fused_step: DeferredDecodeOutput | None = None
        self._last_fused_rows: tuple[str, ...] = ()
        self._last_fused_tip: int | None = None
        # The newest few fused steps by tip id, and what the reader thread
        # learned when each resolved (``ChainResolved``). Both are written from
        # two threads (the engine thread submits, the reader resolves), so one
        # lock guards them; a record is stored before the output is returned.
        self._resolved_lock = threading.Lock()
        self._steps_by_tip: dict[int, DeferredDecodeOutput] = {}
        self._resolved: dict[int, ChainResolved] = {}
        # The fused step whose mask finish (``decode_forward_finish``) is still
        # to come: nothing may be built, chained or read until it is done.
        self._mask_pending_step: DeferredDecodeOutput | None = None
        # Whether the newest fused readout the reader resolved carried
        # ``FusedReadout.drafts`` (None: none resolved yet).
        self._latest_readout_drafts: bool | None = None
        # Fused steps submitted while another was in flight (so chained), and
        # the drains of a possible chain by reason. A break is counted when the
        # step that follows it is submitted, so a drain that needed no verify
        # is not one.
        self._chained_submissions = 0
        self._chain_breaks: Counter[str] = Counter()
        self._pending_break_reason: str | None = None
        # Prefill steps that did not drain a pending fused verify
        # (VLLM_TT_SPEC_PREFILL_OVERLAP).
        self._prefill_overlap_steps = 0
        # The verify a prefill was submitted behind: its rows left the
        # persistent batch at that prefill, so nothing may chain onto it.
        self._prefill_overlapped_step: DeferredDecodeOutput | None = None
        # Requests the scheduler reported finished while a verify for them may
        # still be in flight; guarded by the runner's steady-decode lock.
        self._late_finished: set[str] = set()
        # Real draft accounting (not the scheduler's placeholder reservation):
        # drafts a verify was built with, and drafts the accept walk took.
        self._valid_drafts_submitted = 0
        self._accepted_drafts = 0
        self._verify_rows = 0

    @staticmethod
    def _clone_page_tables(model_input: TTModelInput) -> tuple[torch.Tensor, ...]:
        return tuple(
            table.detach().clone() for table in model_input.block_tables_per_group
        )

    def note_prefill_submitted(self) -> None:
        """Invalidate decode-resident token/position state after prefill."""
        self._decode_chain_valid = False

    def decode_input_update_contract_version(self) -> int:
        """Return the adapter's negotiated reload-contract version."""
        return int(getattr(self.runner.model, "decode_input_update_contract", 0))

    def _page_tables_changed(self, model_input: TTModelInput) -> bool:
        current = model_input.block_tables_per_group
        previous = self._submitted_page_tables
        return (
            previous is None
            or len(previous) != len(current)
            or any(not torch.equal(old, new) for old, new in zip(previous, current))
        )

    def plan_decode_reload(self, model_input: TTModelInput) -> TTDecodeReloadPlan:
        """Return the explicit update commands for the next decode submit."""
        from vllm_tt_plugin.model_input import TTDecodeReloadPlan

        device_sampling = model_input.perform_device_sampling
        model_capabilities = getattr(self.runner.model, "model_capabilities", {}) or {}
        supports_resident_decode = bool(
            model_capabilities.get("supports_async_decode", False)
        )
        decode_trace_enabled = self.runner.trace_mode in ("all", "decode_only")
        sampling_mode_changed = (
            self._previous_device_sampling is not None
            and self._previous_device_sampling != device_sampling
        )
        # A verify commits a count the host decides after the forward and
        # writes none of its tokens into the resident inputs, so those still
        # hold what the last plain decode fed back. The plain decode after a
        # verify therefore starts a new chain, and so do the sampler's seed
        # counters, which the verify's commit moved past.
        verify_to_plain = (
            self._previous_submission_verified and model_input.spec_mode is None
        )
        transition = (
            not self._decode_chain_valid
            or model_input.decode_layout_changed
            or sampling_mode_changed
            or verify_to_plain
        )
        reload_inputs = (
            # A verify's candidate block is assembled from this step's drafts
            # and each row's last committed token, so it is new every step and
            # can never be the resident input. The resident mode is for the
            # plain decode, whose single token the device sampler writes into
            # the buffer the next decode reads.
            model_input.spec_mode is not None
            or not device_sampling
            or transition
            or not supports_resident_decode
            or not decode_trace_enabled
        )
        sampling_reset = device_sampling and transition
        return TTDecodeReloadPlan(
            reload_inputs=reload_inputs,
            reload_page_table=(
                not reload_inputs and self._page_tables_changed(model_input)
            ),
            reload_sampling_params=sampling_reset,
            reset_sampling_state=sampling_reset,
        )

    def commit_decode_submission(
        self,
        model_input: TTModelInput,
        reload_plan: TTDecodeReloadPlan,
    ) -> None:
        """Commit residency after the model accepts a decode submission."""
        self._decode_chain_valid = True
        self._previous_device_sampling = model_input.perform_device_sampling
        self._previous_submission_verified = model_input.spec_mode is not None
        if (
            reload_plan.reload_inputs
            or reload_plan.reload_page_table
            or self._submitted_page_tables is None
        ):
            self._submitted_page_tables = self._clone_page_tables(model_input)

    def scheduler_preserves_decode_layout(
        self, scheduler_output: SchedulerOutput
    ) -> bool:
        """Predict front-packed batch membership before state mutation."""
        current_req_ids = set(self.runner.input_batch.req_id_to_index)
        scheduled_req_ids = set(scheduler_output.num_scheduled_tokens)
        return current_req_ids == scheduled_req_ids

    @staticmethod
    def scheduler_output_has_prefill_work(
        scheduler_output: SchedulerOutput,
    ) -> bool:
        """Whether the scheduler output contains any context-phase work.

        This decision is made before ``_update_states``, so the persistent
        batch still describes the previous step. ``is_context_phase`` is the
        exact signal for a cached chunked-prefill continuation; token counts
        are ambiguous for a one-token final chunk and speculative decode.
        """
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if scheduler_output.scheduled_new_reqs or cached_reqs.resumed_req_ids:
            return True
        return any(
            cached_reqs.is_context_phase(req_id) for req_id in cached_reqs.req_ids
        )

    @staticmethod
    def suppressed_output_req_ids(scheduler_output: SchedulerOutput) -> set[str]:
        """Requests whose late output must not reach scheduler accounting.

        Ordinary preemption and resume deliberately do not appear here: their
        in-flight token is valid and AsyncScheduler must consume its output
        placeholder. Forced-reset frames also stay published so
        ``drop_stale_output`` causes the scheduler to consume the stale frame
        itself.
        """
        return set(scheduler_output.finished_req_ids)

    def capture_submitted_step_context(
        self, req_ids: list[str] | None = None
    ) -> SubmittedStepContext:
        """Snapshot the submitted requests for deferred async state apply.

        ``req_ids`` is the merged output order: the lane path passes its
        scheduled slots' requests (sparse rows), so the index map is built from
        their position. ``None`` takes the condensed front-packed batch, whose
        ``req_id_to_index`` already equals that position map.
        """
        runner = self.runner
        if req_ids is None:
            num_reqs = runner.input_batch.num_reqs
            req_ids = list(runner.input_batch.req_ids[:num_reqs])
            req_id_to_index = dict(runner.input_batch.req_id_to_index)
        else:
            req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
        return SubmittedStepContext(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            submit_time_ns=time.perf_counter_ns(),
        )

    def steady_decode_base_enabled(self) -> bool:
        runner = self.runner
        if not runner.async_decode_scheduling:
            return False
        if runner.trace_mode == "none":  # noqa: SIM103
            return False
        return True

    def steady_decode_scheduler_invariants_met(
        self,
        scheduler_output: SchedulerOutput,
        *,
        decode_layout_changed: bool | None = None,
    ) -> bool:
        runner = self.runner
        input_batch = runner.input_batch
        cached_reqs = scheduler_output.scheduled_cached_reqs
        legacy_prompt = bool(
            scheduler_output.scheduled_new_reqs or cached_reqs.resumed_req_ids
        )
        if legacy_prompt or runner._decode_layout_changed_since_last_decode:
            return False
        if self.decode_input_update_contract_version() >= 1:
            if self.scheduler_output_has_prefill_work(scheduler_output):
                return False
            if (
                not self._decode_chain_valid
                or self._previous_device_sampling is not True
            ):
                return False
            if decode_layout_changed is None:
                decode_layout_changed = not self.scheduler_preserves_decode_layout(
                    scheduler_output
                )
            if decode_layout_changed:
                return False
        # Structured outputs are detected from the scheduler state, not from a
        # prepared bitmask: grammar is applied at sample time, so no bitmask
        # exists yet when this runs. Either signal disables steady decode so the
        # grammar constraint is never skipped by an overlapped step.
        if scheduler_output.pending_structured_output_tokens or has_structured_outputs(
            runner.requests, scheduler_output, None
        ):
            return False
        if not input_batch.no_penalties:
            return False
        if not input_batch.no_allowed_token_ids:
            return False
        if input_batch.sampling.bad_words_token_ids:
            return False
        max_num_logprobs = input_batch.max_num_logprobs
        # Treat logprobs=0 as a real logprobs request so decode does not
        # bypass the slower path that preserves per-token logprob metadata.
        if max_num_logprobs is not None:
            return False
        if input_batch.sampling.has_active_logitsprocs():
            return False
        if runner.model_config.logits_processors:
            return False
        return runner.check_perform_device_sampling(
            is_decode=True,
            has_structured_outputs=False,
        )

    def can_attempt_steady_decode_from_scheduler(
        self,
        scheduler_output: SchedulerOutput,
    ) -> bool:
        if not self.steady_decode_base_enabled():
            return False
        return self.steady_decode_scheduler_invariants_met(scheduler_output)

    def can_attempt_steady_lane_decode_from_scheduler(
        self,
        scheduler_output: SchedulerOutput | None,
        *,
        decode_layout_changed: bool,
    ) -> bool:
        """Whether a merged lane-DP step can overlap steady decode.

        Unlike the front-packed variant, ``scheduler_output=None`` or a zero-token
        step counts as steady-eligible: the merged step still overlaps safely
        when no lane has decode work of its own.
        """
        if not self.steady_decode_base_enabled():
            return False
        if scheduler_output is None or scheduler_output.total_num_scheduled_tokens == 0:
            return (
                not decode_layout_changed
                if self.decode_input_update_contract_version() >= 1
                else True
            )
        return self.steady_decode_scheduler_invariants_met(
            scheduler_output,
            decode_layout_changed=decode_layout_changed,
        )

    def can_use_steady_decode_fast_path(self, model_input: TTModelInput) -> bool:
        if not self.steady_decode_base_enabled():
            return False
        if model_input.prompt_lens is not None:
            return False
        if not model_input.perform_device_sampling:
            return False
        if model_input.decode_layout_changed:
            return False
        if getattr(model_input, "defer_device_sampling", False):
            return False
        if (
            model_input.prompt_tokens is not None
            or model_input.output_tokens is not None
        ):
            return False
        if model_input.allowed_token_ids_mask_list[0] is not None:
            return False
        if model_input.bad_words_token_ids_list[0]:
            return False
        max_num_logprobs = model_input.max_num_logprobs[0]
        if max_num_logprobs is not None:  # noqa: SIM103
            return False
        if self.decode_input_update_contract_version() < 1:
            return True
        return self.plan_decode_reload(model_input).overlap_safe

    # region Chain mode (VLLM_TT_SPEC_CHAIN)

    def chain_enabled(self) -> bool:
        """Whether this launch may chain a fused verify onto the one in flight.

        The env gate, async scheduling with speculation, and a model that
        declares ``supports_async_spec_decode``, which by contract it does only
        when it runs chain and fused. Checked first so that with the gate off
        nothing below runs.
        """
        if not spec_chain_enabled():
            return False
        runner = self.runner
        if not getattr(runner, "async_decode_scheduling", False):
            return False
        if not getattr(runner, "_num_speculative_tokens", 0):
            return False
        capabilities = getattr(runner.model, "model_capabilities", {}) or {}
        return bool(capabilities.get("supports_async_spec_decode", False))

    def _chain_candidate_pending(self) -> bool:
        """Whether the newest deferred fused step is still in flight."""
        step = self._last_fused_step
        if step is None or step.is_resolved():
            return False
        with self.runner._steady_decode_lock:
            return step in self.runner._pending_async_steps

    def chain_evaluate(
        self, scheduler_output: SchedulerOutput | None
    ) -> tuple[str | None, ChainPlan | None]:
        """Decide whether the step about to be built may chain onto step n.

        ``(None, plan)`` when it may, else ``(reason, None)``. All of: chaining
        is on; exactly one verify is unresolved and it is the newest fused step
        submitted with a deferred handle; the scheduled requests are that
        step's rows less the ones reported finished (no join, no preemption, no
        prefill chunk, no forced reset); every row is fused-eligible with no
        grammar, logprobs, penalties or room problem
        (``TTModelRunner._chain_rows_blocker``). Evaluated twice per step on
        the same scheduler output, before the lifecycle update (to skip the
        drain) and after the apply (to build), and it must give the same
        answer: the second is stricter only in having applied more, which
        shrinks the optimistic lengths.
        """
        if not self.chain_enabled():
            return "disabled", None
        if scheduler_output is None:
            return "no_scheduler_output", None
        if self._mask_pending_step is not None:
            return "mask_pending", None
        runner = self.runner
        with runner._steady_decode_lock:
            unresolved = [
                step
                for step in runner._pending_async_steps
                if not step.is_resolved()
            ]
            completed = len(runner._completed_decode_steps)
        if not unresolved:
            return "no_pending", None
        if len(unresolved) > 1:
            return "multi_pending", None
        if unresolved[0] is not self._last_fused_step or self._last_fused_tip is None:
            return "pending_not_fused", None
        if get_tt_forced_reset_discard_counts(scheduler_output):
            return "forced_reset", None
        if self.scheduler_output_has_prefill_work(scheduler_output):
            return "prefill", None
        if unresolved[0] is self._prefill_overlapped_step:
            return "prefill_overlap", None
        # A request the scheduler left out only because its in-flight step is
        # sure to reach max_tokens is as good as finished: it is not coming
        # back, and the model drops it from the chain overlay.
        finished = set(scheduler_output.finished_req_ids) | set(
            get_tt_finishing_req_ids(scheduler_output)
        )
        expected = [r for r in self._last_fused_rows if r not in finished]
        if not expected or set(expected) != set(scheduler_output.num_scheduled_tokens):
            return "layout", None
        reason, plan = runner._chain_rows_blocker(
            expected, scheduler_output, len(unresolved) + completed
        )
        if reason is not None or plan is None:
            return reason or "layout", None
        chain_len, hints = plan
        return None, ChainPlan(self._last_fused_tip, chain_len, hints)

    def mask_pending(self) -> bool:
        """Whether a masked step was begun and its finish has not been called."""
        return self._mask_pending_step is not None

    def latest_readout_has_drafts(self) -> bool:
        """Whether the newest fused readout read carried ``FusedReadout.drafts``.

        False until the first one is read (the first step of a request is never
        chained, so that costs nothing in normal use) and again after a readout
        with none: a chained masked step needs the drafts of the step before it,
        and a model that declares ``supports_spec_device_mask`` returns them."""
        return bool(self._latest_readout_drafts)

    def chain_blocker(self, scheduler_output: SchedulerOutput | None) -> str | None:
        """The reason the next step cannot chain, or None when it can."""
        return self.chain_evaluate(scheduler_output)[0]

    def prefill_overlap_allowed(self, scheduler_output: SchedulerOutput | None) -> bool:
        """Whether a prefill step may be built and submitted over a pending verify.

        Asked only once ``chain_blocker`` has answered "prefill". The step must
        be all prefill with no resumed request and no multimodal input, the
        model must declare ``supports_prefill_over_pending_spec`` and
        VLLM_TT_SPEC_PREFILL_OVERLAP must not be 0. A resumed request still
        drains: ``apply_ready_completed_decode_steps`` assumes its completed
        steps are ordered oldest to newest for the forced-reset discard counts.
        """
        if scheduler_output is None or not self.chain_enabled():
            return False
        if not spec_prefill_overlap_enabled():
            return False
        runner = self.runner
        capabilities = getattr(runner.model, "model_capabilities", {}) or {}
        if not capabilities.get("supports_prefill_over_pending_spec", False):
            return False
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if cached_reqs.resumed_req_ids:
            return False
        if not all(
            cached_reqs.is_context_phase(req_id) for req_id in cached_reqs.req_ids
        ):
            return False
        if any(
            getattr(new_req, "mm_features", None)
            for new_req in scheduler_output.scheduled_new_reqs
        ):
            return False
        return not any(
            getattr(runner.requests.get(req_id), "mm_features", None)
            for req_id in cached_reqs.req_ids
        )

    def chain_forces_drain(self, scheduler_output: SchedulerOutput | None) -> bool:
        """Whether the chain blocker for this step means the verify must drain.

        A blocker forces the drain, except "prefill" when the prefill may be
        submitted over the pending verify (``prefill_overlap_allowed``).
        """
        reason = self.chain_blocker(scheduler_output)
        if reason is None:
            return False
        return not (reason == "prefill" and self.prefill_overlap_allowed(scheduler_output))

    def _note_break(self, reason: str | None) -> None:
        """Remember why a chain was refused; counted at the next submission."""
        if reason not in (None, "disabled", "no_pending", "no_scheduler_output"):
            self._pending_break_reason = reason

    def note_finished_requests(self, req_ids: Any) -> None:
        """Remember requests the scheduler just reported finished.

        A verify already in flight for one of them is discarded: the scheduler
        does not wait for it, so the step after the one that finished a
        request has already been submitted with its row. The output is blanked
        when that step resolves (``complete_spec_decode_step``) and its commit
        is skipped (``suppressed_output_req_ids`` and the row being gone from
        the batch). The blanking is best effort across threads, since the
        readback thread can win the race with the engine thread's next
        build; the scheduler ignores output for a finished request either way.
        """
        if not req_ids or not self.chain_enabled():
            return
        with self.runner._steady_decode_lock:
            self._late_finished.update(req_ids)

    def _blank_late_finished(
        self, model_input: TTModelInput, prefixes: list[list[int]]
    ) -> list[list[int]]:
        with self.runner._steady_decode_lock:
            if not self._late_finished:
                return prefixes
            late = set(self._late_finished)
        return [
            [] if req_id in late else prefix
            for req_id, prefix in zip(model_input.row_req_ids, prefixes)
        ]

    # endregion

    def enqueue_completed_decode_step(self, completed: CompletedDecodeStep) -> None:
        with self.runner._steady_decode_lock:
            self.runner._completed_decode_steps.append(completed)

    @staticmethod
    def _any_unresolved(steps: Any) -> bool:
        """Whether any pending step has not resolved yet (a step that cannot
        say is taken as unresolved)."""
        return any(
            not getattr(step, "is_resolved", lambda: False)() for step in steps
        )

    def register_pending_async_step(
        self,
        step: DeferredDecodeOutput,
        *,
        overlap_ok: bool,
    ) -> None:
        unsafe_overlap = False
        with self.runner._steady_decode_lock:
            # Counted here because this is the one place that knows both
            # facts: whether a step was already outstanding when this one was
            # submitted, and whether this one is a verify. Nothing else
            # reports overlap. A launch option does not prove it, a completed
            # request does not prove it, and per-step wall clock cannot
            # separate it from a faster model. The device suite reads these off
            # the log line below and asserts that ordinary steps overlapped
            # and that no verify ever did.
            # Only a step still unresolved counts: a resolved one is complete on
            # the device, and building over it only waits for its host apply.
            overlapped = self._any_unresolved(self.runner._pending_async_steps)
            if overlapped:
                self._overlapped_submissions += 1
                if not overlap_ok:
                    self._overlapped_unsafe_submissions += 1
                    unsafe_overlap = True
            self.runner._pending_async_steps.append(step)
            self.runner._pending_async_overlap_ok.append(overlap_ok)
        self._log_overlap_counters(force=unsafe_overlap)

    def count_decode_submission(self, model_input: TTModelInput) -> None:
        """Count what was actually submitted, and report it on a cadence.

        A measurement cannot take these from vLLM's speculative metrics. Those
        count what the scheduler reserved, and under asynchronous scheduling
        ``AsyncScheduler`` reserves ``[-1] * K`` for every scheduled request
        whether or not a draft was ever verified, so
        ``spec_decode_num_drafts_total`` moves on a launch that verifies
        nothing. What a cost-per-step comparison needs is how many verifies and
        how many ordinary decodes the runner actually sent, which only the
        runner knows.

        Counted in ``submit_decode`` because that is the single funnel both
        execution modes go through, so a synchronous run and an asynchronous
        one are counted by the same code. Reported every
        ``_SUBMISSION_LOG_INTERVAL`` submissions rather than per step: a
        benchmark diffs the last line before its interval against the last one
        after, and a line per step would cost more than the work it measures.

        The line repeats ``_overlapped_submissions``, which
        ``_log_overlap_counters`` also reports, because that one reports at
        every power of two: between 4096 and 8191 overlaps it prints nothing,
        so a diff across a measured interval cannot tell a launch that kept
        overlapping from one that stopped. Here the same cadence carries the
        overlap count and the submission count it divides by. The current
        submission is not registered as pending until later in
        ``submit_decode``, so the overlap count on a line covers the
        submissions before this one.
        """
        if model_input.spec_mode is None:
            self._ordinary_decode_submissions += 1
        else:
            self._verify_submissions += 1
            if model_input.spec_mode == ACCEPT_MODE_LOGITS:
                self._logits_verify_submissions += 1
            elif model_input.spec_mode == ACCEPT_MODE_TOPK:
                self._topk_verify_submissions += 1
            elif model_input.spec_mode == ACCEPT_MODE_FUSED_SAMPLE:
                self._fused_verify_submissions += 1
            rows = getattr(model_input, "row_req_ids", None)
            if getattr(model_input, "num_valid_drafts", None) is not None and rows:
                live = len(rows)
                self._verify_rows += live
                self._valid_drafts_submitted += int(
                    model_input.num_valid_drafts[:live].sum()
                )
        self._count_step_specials(model_input)
        total = self._ordinary_decode_submissions + self._verify_submissions
        if total % _SUBMISSION_LOG_INTERVAL == 0:
            # Parsed by tests/tt/spec/bench_spec_overhead.py and
            # tests/tt/spec/test_sampled_speculation.py, which match the line
            # up to the logits count; the topk count is appended after it.
            fallback = support_fallback_stats()
            logger.info(
                "TT submissions: %d ordinary decode, %d verify, "
                "%d overlapped an outstanding step, %d verify in logits mode, "
                "%d verify in topk mode, %d verify in fused mode; "
                "topk fallback: %d/%d positions overflow, %d fetches in %d support steps"
                "; real drafts: %d valid, %d accepted over %d verify rows"
                "%s%s",
                self._ordinary_decode_submissions,
                self._verify_submissions,
                self._overlapped_submissions,
                self._logits_verify_submissions,
                self._topk_verify_submissions,
                self._fused_verify_submissions,
                fallback["overflow_positions"],
                fallback["support_positions"],
                fallback["fetch_calls"],
                fallback["support_steps"],
                self._valid_drafts_submitted,
                self._accepted_drafts,
                self._verify_rows,
                self._chain_report(),
                self._device_logprobs_report()
                + self._device_penalties_report()
                + self._device_masks_report(),
            )
            # One line per worker for the production mix of special requests,
            # parsed by tests/spec/_specials_parse.py.
            logger.info(specials_report())

    def _count_step_specials(self, model_input: TTModelInput) -> None:
        """Count this step once per special feature a row carries, by path.

        The features were classified when each request was admitted
        (``TTCachedRequestState.specials``); this only takes their union over
        the step's rows."""
        requests = getattr(self.runner, "requests", None) or {}
        features: set[str] = set()
        for req_id in getattr(model_input, "row_req_ids", None) or ():
            state = requests.get(req_id)
            if state is not None:
                features.update(getattr(state, "specials", ()))
        count_step_specials(
            _SPECIALS_PATHS.get(model_input.spec_mode, "plain"), features
        )

    @staticmethod
    def _device_logprobs_report() -> str:
        """Device-logprob counters for the submission line (empty before step 1)."""
        stats = device_logprob_stats()
        if not stats["steps"]:
            return ""
        return (
            f"; device logprobs: {stats['steps']} steps, "
            f"{stats['rank_lower_bound']} rank lower bounds, "
            f"{stats['missing_id']} missing ids"
        )

    @staticmethod
    def _device_masks_report() -> str:
        """The device-mask counters for the submission line (empty before any)."""
        stats = device_mask_stats()
        if not stats["steps"] and not stats["fallbacks"]:
            return ""
        reasons = ", ".join(
            f"{reason}={count}" for reason, count in sorted(stats["fallbacks"].items())
        )
        chain = (
            f", chained {stats['chained']}, late {stats['late']}, "
            f"waits {stats['waits']}"
            if stats["chained"] or stats["late"] or stats["waits"]
            else ""
        )
        return (
            f"; device masks: {stats['steps']} steps, {stats['rows']} masked rows, "
            f"{sum(stats['fallbacks'].values())} logits fallbacks"
            + (f" ({reasons})" if reasons else "")
            + chain
        )

    @staticmethod
    def _device_penalties_report() -> str:
        """The device-penalty counters for the submission line (empty before any)."""
        stats = device_penalty_stats()
        if not stats["steps"]:
            return ""
        return (
            f"; device penalties: {stats['steps']} steps, "
            f"{stats['rebuilds']} state rebuilds"
        )

    def _chain_report(self) -> str:
        """The chain counters for the submission line, empty when chaining is off."""
        if not self.chain_enabled():
            return ""
        breaks = ", ".join(
            f"{reason}={count}" for reason, count in sorted(self._chain_breaks.items())
        )
        return (
            f"; chain: {self._chained_submissions} chained fused steps, "
            f"{sum(self._chain_breaks.values())} chain breaks"
            + (f" ({breaks})" if breaks else "")
            + f", prefill_overlap={self._prefill_overlap_steps}"
        )

    def _log_overlap_counters(self, *, force: bool = False) -> None:
        """Report the overlap counters, rarely enough to be readable.

        Logged rather than exposed as a metric because vLLM's metrics are the
        engine's and this is the runner's: the step it describes belongs to the
        worker process, which publishes none of its own. Safe overlaps report at
        every power of two. Every unsafe overlap reports immediately because the
        device regression reads the latest line as its serialization result.
        """
        total = self._overlapped_submissions
        if not force and (total == 0 or total & (total - 1)):
            return
        logger.info(
            "TT async decode: %d submission(s) overlapped an outstanding step, "
            "%d of them a step that was not overlap-safe",
            total,
            self._overlapped_unsafe_submissions,
        )

    def prune_finished_async_events(self) -> None:
        with self.runner._steady_decode_lock:
            while (
                self.runner._pending_async_steps
                and self.runner._pending_async_steps[0].is_resolved()
            ):
                self.runner._pending_async_steps.popleft()
                self.runner._pending_async_overlap_ok.popleft()
            if (
                self._late_finished
                and not self.runner._pending_async_steps
                and not self.runner._completed_decode_steps
            ):
                self._late_finished.clear()

    def drain_completed_decode_steps(self) -> list[CompletedDecodeStep]:
        completed: list[CompletedDecodeStep] = []
        with self.runner._steady_decode_lock:
            while self.runner._completed_decode_steps:
                completed.append(self.runner._completed_decode_steps.popleft())
        return completed

    def apply_ready_completed_decode_steps(
        self,
        *,
        suppress_output_req_ids: set[str] | None = None,
        forced_reset_discard_counts: dict[str, int] | None = None,
    ) -> None:
        """Apply completed outputs after the scheduler updates request state.

        Complete each pending step before the batch layout changes. Lane output
        extraction needs the layout used for that step. Apply the token only
        after the next ``SchedulerOutput`` updates request state. This update
        tells us which tokens are valid. Apply valid tokens before you build
        reload inputs from host state.

        A forced prefix-cache reset calls
        ``reset_prefix_cache(reset_running_requests=True)``. It preempts live
        requests and frees their KV blocks. vLLM resumes the requests from
        saved token history and drops their in-flight outputs
        (``drop_stale_output``).

        "Publish" means that the output stays visible to vLLM. "Apply" means
        that the runner adds the token to its request state. Use these rules:

        * Normal output for a live request: publish and apply.
        * Ordinary preemption or resume: publish and apply.
        * Forced-reset output: publish, but do not apply. vLLM discards it.
        * Request already finished or aborted: remove any late output and do
          not apply it.

        For example, request R has two completed decode results: result A and
        result B. vLLM accepted result A before the reset, but the runner has
        not applied its token. Result B was in flight at the reset, so
        ``forced_reset_discard_counts[R]`` is 1. Apply the token from result A.
        Publish result B, but do not apply its token; vLLM drops it.

        A late result can occur when an earlier result ends the request. It can
        also occur when a result is in flight during an abort. The result that
        ended the request was already accepted. Only the late result is
        removed.
        """
        # Prune before draining: completions are enqueued before their event
        # is set, so every step pruned here is already in the queue the drain
        # reads.
        self.prune_finished_async_events()
        completed_steps = self.drain_completed_decode_steps()
        suppressed = set(suppress_output_req_ids or ())
        discard_counts = forced_reset_discard_counts or {}

        # Results are ordered from oldest to newest. vLLM accepted the older
        # results before the reset. The discard count applies to the newest
        # results that were in flight during the reset.
        remaining_results: Counter[str] = Counter()
        for completed in completed_steps:
            if completed.runner_output is None:
                continue
            for req_id in completed.context.req_ids:
                req_idx = completed.runner_output.req_id_to_index.get(req_id)
                if (
                    req_idx is not None
                    and completed.runner_output.sampled_token_ids[req_idx]
                ):
                    remaining_results[req_id] += 1

        missing = {
            req_id: count - remaining_results[req_id]
            for req_id, count in discard_counts.items()
            if remaining_results[req_id] < count
        }
        if missing:
            raise RuntimeError(
                "Not enough completed outputs for the forced-reset discard "
                f"counts: {missing}"
            )

        for completed in completed_steps:
            published_req_ids: set[str] = set()
            if completed.runner_output is not None:
                for req_id in completed.context.req_ids:
                    req_idx = completed.runner_output.req_id_to_index.get(req_id)
                    if (
                        req_idx is not None
                        and completed.runner_output.sampled_token_ids[req_idx]
                    ):
                        published_req_ids.add(req_id)
            forced_reset_req_ids = {
                req_id
                for req_id in published_req_ids
                if 0 < remaining_results[req_id] <= discard_counts.get(req_id, 0)
            }
            self.apply_completed_decode_step(
                completed,
                suppress_output_req_ids=suppressed,
                skip_state_req_ids=suppressed | forced_reset_req_ids,
            )
            remaining_results.subtract(published_req_ids)

    def apply_resolved_before_build(self, scheduler_output: SchedulerOutput) -> None:
        """Apply what resolved after the pre-build apply, before a host build.

        ``chain_evaluate`` answers ``no_pending`` when the verify the earlier
        decision saw in flight has resolved in between (the readback thread
        races the engine thread). That step's completion is queued but was not
        applied, so build only after applying it.
        """
        with self.runner._steady_decode_lock:
            queued = bool(self.runner._completed_decode_steps)
        if queued:
            self.apply_ready_completed_decode_steps(
                suppress_output_req_ids=self.suppressed_output_req_ids(
                    scheduler_output
                ),
                forced_reset_discard_counts=get_tt_forced_reset_discard_counts(
                    scheduler_output
                ),
            )

    def apply_completed_decode_steps_before_build(
        self, scheduler_output: SchedulerOutput
    ) -> None:
        """Apply completed steps and serialize a newly visible verify.

        A plain completion can publish a proposal while this method applies it.
        The pre-build drain decision could not see that proposal. Recheck after
        the apply so every remaining plain step commits before the candidate
        block is built. A draftless next step keeps the existing overlap.
        """
        suppressed = self.suppressed_output_req_ids(scheduler_output)
        self.apply_ready_completed_decode_steps(
            suppress_output_req_ids=suppressed,
            forced_reset_discard_counts=get_tt_forced_reset_discard_counts(
                scheduler_output
            ),
        )
        if self._chain_candidate_pending():
            # The drain decision was taken before the lifecycle update and the
            # apply; with a verify still in flight the answer must be the same
            # now, or the build below would stack a step over it.
            reason = self.chain_blocker(scheduler_output)
            if reason is None:
                return
            if reason == "prefill" and self.prefill_overlap_allowed(scheduler_output):
                # The steps already completed are applied above; the verify
                # stays in flight and commits before the next decode build.
                self._prefill_overlapped_step = self._last_fused_step
                return
            self._note_break(reason)
            self.wait_for_all_pending_async_steps()
            self.apply_ready_completed_decode_steps(suppress_output_req_ids=suppressed)
            return
        if not self._next_step_verifies(scheduler_output):
            return
        self.wait_for_all_pending_async_steps()
        self.apply_ready_completed_decode_steps(suppress_output_req_ids=suppressed)

    def wait_for_all_pending_async_steps(self) -> None:
        """Finalize pending readbacks without applying them to runner state."""
        # Drive each pending readback to completion here rather than blocking on
        # its event: the engine has not popped these futures yet (and on 0.22
        # will not until after this returns), so nothing else will set the
        # events. ``ensure_finalized`` is idempotent, so the engine's later
        # ``get_output`` on the same step returns the cached result.
        with self.runner._steady_decode_lock:
            steps = list(self.runner._pending_async_steps)
        for step in steps:
            step.ensure_finalized()

    def must_drain_pending_async_steps(
        self,
        steady_decode_candidate: bool,
        scheduler_output: SchedulerOutput | None = None,
    ) -> bool:
        with self.runner._steady_decode_lock:
            if not self.runner._pending_async_steps:
                return False
            # A wholesale reset marks the already-submitted frames stale. They
            # must all be present before suffix discard counts are applied.
            forced_reset = scheduler_output is not None and bool(
                get_tt_forced_reset_discard_counts(scheduler_output)
            )
        if forced_reset:
            if self._chain_candidate_pending():
                self._note_break("forced_reset")
            return True
        if self._chain_candidate_pending():
            # A fused verify is in flight and this launch chains. Either the
            # next step chains onto it, which overlaps whatever else is true of
            # it, or nothing may be built over a verify and everything drains.
            reason = self.chain_blocker(scheduler_output)
            if reason is None:
                return False
            if reason == "prefill" and self.prefill_overlap_allowed(scheduler_output):
                # The prefill reads nothing the verify produces and the device
                # queue is in order: submit it behind the verify.
                self._prefill_overlap_steps += 1
                self._prefill_overlapped_step = self._last_fused_step
                return False
            self._note_break(reason)
            return True
        with self.runner._steady_decode_lock:
            if not steady_decode_candidate:
                return True
            if any(
                not overlap_ok for overlap_ok in self.runner._pending_async_overlap_ok
            ):
                return True
        # And the other direction, which the pending flags cannot express: the
        # step about to be built may itself be a verify, and a verify's
        # candidate block starts from each row's last committed token. An
        # outstanding step holds one of those tokens, so building over it would
        # verify a block starting one token behind. The flags above say whether
        # what is pending tolerates company; this says whether what comes next
        # does.
        return self._next_step_verifies(scheduler_output)

    def _next_step_verifies(self, scheduler_output: SchedulerOutput | None) -> bool:
        """Whether the step about to be built will send the model a verify.

        Decided from the scheduler output and the runner's own state, because
        the persistent batch has not been updated yet when this is asked. It
        errs towards True: a drain this did not need costs one step's overlap,
        while a verify built over an outstanding step commits the wrong tokens.
        """
        runner = self.runner
        if not runner._num_speculative_tokens or scheduler_output is None:
            return False
        if not runner._pending_async_steps:
            return False
        scheduled = scheduler_output.num_scheduled_tokens
        if runner._spec_supports_narrow_decode and not all(
            runner._request_is_speculable(req_id) for req_id in scheduled
        ):
            # ``_step_verifies`` sends such a step as the ordinary decode,
            # whatever its rows carry. A scheduled request that is not in the
            # persistent batch yet reads as speculable here, which errs
            # towards True below.
            return False
        drafts = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduled:
            offered = drafts.get(req_id) or ()
            # Read the way ``TTModelRunner._drafts_to_verify`` reads it, and
            # for the same reason: under asynchronous scheduling every
            # scheduled request holds ``[-1] * num_spec_tokens_to_schedule``,
            # which reserves lookahead rather than proposing anything. Taking
            # that list for a proposal would make every step look like a
            # verify, and every step would then drain, which is the whole
            # overlap gone for a launch that drafts nothing.
            if any(token != PLACEHOLDER_TOKEN_ID for token in offered):
                return True
            if runner._proposed_draft_token_ids.get(req_id):
                return True
            if runner._req_accepted_counts.get(req_id, 1) > 1:
                return True
        return False

    def complete_decode_step(
        self,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
        context: SubmittedStepContext,
        scheduled_rows: list[int] | None = None,
    ) -> CompletedDecodeStep:
        """Finalize a single-process async decode read into sampled tokens.

        When ``scheduled_rows`` is given this is a lane-DP step: the merged slot
        batch is read back via ``TTLaneInputBatch.extract_output``. Otherwise it
        is a plain single-process step read back via ``_get_output_tokens``.
        """
        finalized = self.finalize_decode(submission, sampling_rows=scheduled_rows)
        if finalized is None:
            sampled_token_ids = torch.empty((0, 1), dtype=torch.int32)
            logprobs = None
        elif scheduled_rows is not None:
            sampled_token_ids, logprobs = self.runner.lane_batch.extract_output(
                self.runner,
                finalized.tt_out,
                finalized.tt_log_probs,
                model_input,
                scheduled_rows,
                is_decode=True,
            )
        else:
            sampled_token_ids_per_dp, logprobs_per_dp = self.runner._get_output_tokens(
                tt_out=finalized.tt_out,
                tt_log_probs=finalized.tt_log_probs,
                sampling_params=submission.sampling_params,
                model_input=model_input,
                batch_size_per_dp=submission.batch_size_per_dp,
                perform_device_sampling=submission.perform_device_sampling,
                is_decode=True,
            )
            sampled_token_ids = sampled_token_ids_per_dp[0]
            logprobs_tensors = logprobs_per_dp[0] if logprobs_per_dp else None
            logprobs = logprobs_tensors.tolists() if logprobs_tensors else None
        return CompletedDecodeStep(
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            context=context,
            completion_time_ns=time.perf_counter_ns(),
            model_input=model_input,
        )

    def complete_spec_decode_step(
        self,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
        context: SubmittedStepContext,
    ) -> CompletedSpecDecodeStep:
        """Finalize a speculative read and walk acceptance over what came back.

        Runs on whichever thread resolved the deferred output, so it reads the
        step's own tensors and the immutable model config and nothing else. A
        ``logits`` walk also advances the step's request generators, which is
        why ``DeferredDecodeOutput`` resolves a step once. The commit and the
        next proposal are left for the engine thread.
        """
        finalized = self.finalize_decode(submission)
        if finalized is None:
            rows = len(model_input.row_req_ids)
            return CompletedSpecDecodeStep(
                committed=torch.empty((0, 0), dtype=torch.int32),
                counts=torch.empty((0,), dtype=torch.int32),
                prefixes=[[] for _ in range(rows)],
                model_input=model_input,
                spec_hidden=submission.spec_hidden,
                context=context,
                completion_time_ns=time.perf_counter_ns(),
            )
        walk = self.runner.walk_spec_acceptance(model_input, finalized.tt_out)
        live_rows = len(model_input.row_req_ids)
        self._accepted_drafts += max(
            0, int(walk.counts[:live_rows].sum()) - live_rows
        )
        if finalized.p0 is None:
            prefixes = self.runner.spec_committed_prefixes(
                model_input, walk.committed, walk.counts
            )
        else:
            # A chained step's submitted positions are only provisional: the
            # length cap starts from the position the device reports.
            prefixes = self.runner.spec_committed_prefixes(
                model_input, walk.committed, walk.counts, finalized.p0
            )
            prefixes = self._blank_late_finished(model_input, prefixes)
        self._store_resolved(submission, model_input, prefixes, finalized.drafts)
        return CompletedSpecDecodeStep(
            committed=walk.committed,
            counts=walk.counts,
            prefixes=prefixes,
            model_input=model_input,
            spec_hidden=submission.spec_hidden,
            context=context,
            completion_time_ns=time.perf_counter_ns(),
            logprobs=self.runner.spec_committed_logprobs(walk.logprobs, prefixes),
            p0=finalized.p0,
        )

    def _remember_step(self, tip: int, step: DeferredDecodeOutput) -> None:
        """Keep the newest fused steps by tip id, so ``wait_resolved`` can read one."""
        with self._resolved_lock:
            self._steps_by_tip[tip] = step
            while len(self._steps_by_tip) > _RESOLVED_KEEP:
                del self._steps_by_tip[min(self._steps_by_tip)]

    def _store_resolved(
        self,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
        prefixes: list[list[int]],
        drafts: torch.Tensor | None,
    ) -> None:
        """Record a resolved fused step for ``resolved_view``.

        Runs on whichever thread resolved the step, before its output is
        returned, so the engine core never sees an output whose record is
        missing. A step with no deferred handle has no tip and no record.
        """
        tip = getattr(self._deferred_handle(submission), "tip_id", None)
        if tip is None:
            return
        record = ChainResolved(
            tip=int(tip),
            row_req_ids=tuple(model_input.row_req_ids),
            prefixes=[list(prefix) for prefix in prefixes],
            drafts=drafts,
        )
        with self._resolved_lock:
            self._resolved[record.tip] = record
            while len(self._resolved) > _RESOLVED_KEEP:
                del self._resolved[min(self._resolved)]

    def resolved_view(self, tip: int) -> ChainResolved | None:
        """The record of the fused step ``tip``, or None while it is unresolved.

        A pure read: it never waits and changes nothing."""
        with self._resolved_lock:
            return self._resolved.get(tip)

    def wait_resolved(self, tip: int) -> ChainResolved:
        """The record of the fused step ``tip``, resolving the step if needed.

        Resolution goes through ``ensure_finalized``, so the step is read
        exactly once whoever gets there first (the reader thread or this one),
        and a failed read raises here as it does there. Raises if the step is
        not one this controller still remembers, or resolved without a record.
        """
        record = self.resolved_view(tip)
        if record is not None:
            return record
        with self._resolved_lock:
            step = self._steps_by_tip.get(tip)
        if step is None:
            raise RuntimeError(
                f"cannot wait for fused step {tip}: it is not one of the "
                f"newest {_RESOLVED_KEEP} submitted fused steps"
            )
        count_mask_wait()
        step.ensure_finalized()
        record = self.resolved_view(tip)
        if record is None:
            raise RuntimeError(f"fused step {tip} resolved without a record")
        return record

    def commit_view(self, req_id: str, tip: int) -> list[int]:
        """The output of ``req_id`` as it is once the step ``tip`` is committed.

        The runner is one step behind: the commit of ``tip`` is applied at the
        next build. A pure read: the applied output plus the step's committed
        prefix for the row, so min_tokens and multi-token bad words see what the
        commit will give them. Valid only while that commit is not applied, and
        only after the step is resolved (``wait_resolved``). A request that is
        not a row of ``tip`` has nothing to add.
        """
        record = self.resolved_view(tip)
        if record is None:
            raise RuntimeError(
                f"the commit view of fused step {tip} needs the step resolved "
                "(wait_resolved first)"
            )
        state = self.runner.requests[req_id]
        output = list(state.output_token_ids)
        if req_id in record.row_req_ids:
            output.extend(record.prefixes[record.row_req_ids.index(req_id)])
        return output

    def chain_handoff_drafts(
        self, tip: int, hints: dict[str, int]
    ) -> dict[str, list[int]]:
        """The drafts of the step ``tip`` for each request with a hint above 0.

        ``hints`` is how many drafts each structured row of the chained step
        verifies. Waits for the step when it is not resolved. A request with a
        hint and no drafts to give (the model returned none, or the request is
        not a row of ``tip``) raises: a structured row whose drafts reach the
        device but not the scheduler would walk them against an all-ones mask.
        """
        record = self.wait_resolved(tip)
        wanted = {req_id: int(hint) for req_id, hint in hints.items() if hint > 0}
        if not wanted:
            return {}
        if record.drafts is None:
            raise RuntimeError(
                f"fused step {tip} gave no drafts, but a chained masked step "
                f"verifies drafts for {sorted(wanted)}: a model that declares "
                "supports_spec_device_mask returns FusedReadout.drafts"
            )
        handed: dict[str, list[int]] = {}
        for req_id, hint in wanted.items():
            if req_id not in record.row_req_ids:
                raise RuntimeError(
                    f"request {req_id!r} verifies drafts in a step chained on "
                    f"{tip}, but is not one of its rows {list(record.row_req_ids)}"
                )
            row = record.row_req_ids.index(req_id)
            handed[req_id] = [int(token) for token in record.drafts[row, :hint]]
        return handed

    def tag_valid_drafts(
        self, runner_output: ModelRunnerOutput, model_input: TTModelInput
    ) -> None:
        """Report how many real drafts each row verified, for vLLM's metrics.

        ``AsyncScheduler`` offers K placeholders to every row, so the
        scheduler cannot tell a verify of three real drafts from a seed step
        or an ordinary decode that verified none. ``TTScheduler`` reads this
        and sets ``num_invalid_spec_tokens`` from it. A step that did not
        verify (no ``num_valid_drafts``) verified none.
        """
        if not self.runner._num_speculative_tokens or not model_input.row_req_ids:
            return
        rows = model_input.row_req_ids
        valid = model_input.num_valid_drafts
        counts = (
            [int(v) for v in valid[: len(rows)].tolist()]
            if valid is not None
            else [0] * len(rows)
        )
        setattr(runner_output, TT_VALID_DRAFTS_ATTR, dict(zip(rows, counts)))

    def build_runner_output_from_completed_step(
        self,
        completed: CompletedDecodeStep,
    ) -> ModelRunnerOutput:
        return self.runner._build_runner_output(
            sampled_token_ids=completed.sampled_token_ids,
            logprobs=completed.logprobs,
            req_ids=completed.context.req_ids,
            req_id_to_index=completed.context.req_id_to_index,
        )

    def apply_completed_decode_step(
        self,
        completed: CompletedDecodeStep | CompletedSpecDecodeStep,
        *,
        suppress_output_req_ids: set[str] | None = None,
        skip_state_req_ids: set[str] | None = None,
    ) -> None:
        suppressed = set(suppress_output_req_ids or ())
        skipped_state = set(skip_state_req_ids or ())
        # The batch-queue loop schedules the current step before consuming the
        # previous future. Scrub late results for request IDs already reported
        # finished before scheduler.update_from_output observes that cached
        # output. The result that finished the request was already accepted.
        # These results arrived after finish or raced with an abort.
        # Forced-reset results stay intact because the scheduler must see them
        # to drain ``num_stale_output_tokens``.
        if completed.runner_output is not None:
            assert self.runner.scheduler_config.async_scheduling, (
                "mutating a published runner output is only ordered correctly "
                "under the batch-queue step loop"
            )
            for req_id in suppressed:
                req_idx = completed.runner_output.req_id_to_index.get(req_id)
                if req_idx is not None:
                    completed.runner_output.sampled_token_ids[req_idx] = []
        if isinstance(completed, CompletedSpecDecodeStep):
            # The commit and the next proposal, on the engine thread, for the
            # step whose acceptance was walked at completion. The lifecycle
            # sets above apply unchanged: a request finished, aborted or
            # force-reset since submission is skipped here rather than having
            # its tokens written and a continuation of them drafted.
            self.runner.commit_spec_acceptance(
                completed.model_input,
                completed.committed,
                completed.counts,
                completed.spec_hidden,
                skip_req_ids=skipped_state,
                **(
                    {"first_positions": completed.p0}
                    if completed.p0 is not None
                    else {}
                ),
            )
            return
        self.runner._apply_sampled_tokens_to_state(
            sampled_token_ids=completed.sampled_token_ids,
            req_ids=completed.context.req_ids,
            skip_req_ids=skipped_state,
        )
        if self.runner._num_speculative_tokens and completed.model_input is not None:
            # A speculating launch's draftless decode step is an ordinary step
            # in every way but one: the proposer is asked once per step, so
            # this step owes the next proposal or the launch never drafts
            # again. It runs here, a step after the submission, because the
            # drafts continue the token that just came back.
            self.runner.propose_after_plain_step(
                completed.model_input,
                completed.sampled_token_ids,
                completed.context.req_ids,
                skip_req_ids=skipped_state,
            )

    @staticmethod
    def _deferred_handle(submission: TTDecodeSubmission) -> Any | None:
        """The handle of a fused verify submitted with ``defer_read``, else None."""
        out = submission.tt_out
        if (
            isinstance(out, VerifyOutput)
            and out.handle is not None
            and out.accepted_token_ids is None
        ):
            return out.handle
        return None

    def submit_async_decode(
        self,
        model_input: TTModelInput,
        *,
        steady_decode_fast_path: bool,
    ) -> DeferredDecodeOutput:
        event = threading.Event()
        context = self.capture_submitted_step_context()
        submission = self.submit_decode(
            model_input,
            read_from_device=False,
            async_read=True,
        )
        if submission.tt_out is None:
            event.set()
        step: DeferredDecodeOutput
        if model_input.spec_mode is not None:
            # A speculative step resolves through the accept walk, not the
            # sampler. It is never overlap-safe: the next candidate block is
            # built from this step's committed tokens, so the build has to wait
            # for this commit. This is the only thing that says so:
            # ``check_perform_device_sampling`` does not look at speculation,
            # so on a launch that samples on device a verify reaches
            # ``can_use_steady_decode_fast_path`` with every condition met and
            # comes back eligible.
            step = AsyncTTSpecDecodeOutput(
                controller=self,
                submission=submission,
                model_input=model_input,
                completion_event=event,
                context=context,
            )
            handle = self._deferred_handle(submission)
            tip = getattr(handle, "tip_id", None)
            if handle is not None and tip is not None:
                self._remember_step(int(tip), step)
            chained = model_input.chain_rows is not None
            if handle is None or tip is None:
                # Not chainable: nothing may be built over it, as before.
                self._last_fused_step = None
                self._last_fused_tip = None
                self._last_fused_rows = ()
                self.register_pending_async_step(step, overlap_ok=False)
                return step
            # Chainable: a following verify may overlap it. It may itself
            # overlap an outstanding step only when it was built chained; a
            # verify stacked over one without the chain is the unsafe overlap
            # the counter must keep at zero.
            with self.runner._steady_decode_lock:
                overlapped = self._any_unresolved(self.runner._pending_async_steps)
            self.register_pending_async_step(
                step, overlap_ok=chained or not overlapped
            )
            if chained:
                self._chained_submissions += 1
                self._pending_break_reason = None
            elif self._pending_break_reason is not None:
                self._chain_breaks[self._pending_break_reason] += 1
                self._pending_break_reason = None
            self._last_fused_step = step
            self._last_fused_tip = int(tip)
            self._last_fused_rows = tuple(model_input.row_req_ids)
            if getattr(submission, "mask_pending", False):
                # A masked step is a chain tip like any other once its finish
                # has been called; until then ``chain_evaluate`` refuses.
                self._mask_pending_step = step
            return step
        step = AsyncTTModelRunnerOutput(
            controller=self,
            submission=submission,
            model_input=model_input,
            completion_event=event,
            context=context,
        )
        self._last_fused_step = None
        self.register_pending_async_step(step, overlap_ok=steady_decode_fast_path)
        return step

    def submit_async_lane_decode(
        self,
        model_input: TTModelInput,
        context: SubmittedStepContext,
        scheduled_rows: list[int],
    ) -> AsyncTTModelRunnerOutput:
        """Submit a non-blocking single-process multi-lane decode step."""
        overlap_ok = self.can_use_steady_decode_fast_path(model_input)
        completion_event = threading.Event()
        submission = self.submit_decode(
            model_input,
            read_from_device=False,
            async_read=True,
            sampling_rows=scheduled_rows,
        )
        if submission.tt_out is None:
            completion_event.set()
        step = AsyncTTModelRunnerOutput(
            controller=self,
            submission=submission,
            model_input=model_input,
            completion_event=completion_event,
            context=context,
            scheduled_rows=scheduled_rows,
        )
        self.register_pending_async_step(step, overlap_ok=overlap_ok)
        return step

    def submit_decode(
        self,
        model_input: TTModelInput,
        *,
        read_from_device: bool,
        async_read: bool = False,
        sampling_rows: list[int] | None = None,
    ) -> TTDecodeSubmission:
        runner = self.runner
        batch_size_per_dp = model_input.unpadded_batch_size
        if not isinstance(batch_size_per_dp, list):
            batch_size_per_dp = [batch_size_per_dp]

        sampling_params = model_input.tt_sampling_params
        perform_device_sampling = model_input.perform_device_sampling
        deferred_device_sampling = bool(
            getattr(model_input, "defer_device_sampling", False)
        )
        readback_rows = None
        if (
            sampling_rows is not None
            and not perform_device_sampling
            and getattr(runner.model, "model_capabilities", {}).get(
                "supports_selective_host_readback", False
            )
            and runner.lane_batch.can_compact_host_sampling()
        ):
            readback_rows = tuple(sampling_rows)
            # Select rows before the adapter initiates a full output read.
            # Synchronous callers wait for these same events in finalize_decode.
            read_from_device = False
            async_read = True
        contract_version = self.decode_input_update_contract_version()
        if deferred_device_sampling and (
            not perform_device_sampling or model_input.prompt_lens is not None
        ):
            raise ValueError(
                "defer_device_sampling requires a device-sampled decode step; "
                f"perform_device_sampling={perform_device_sampling}, "
                f"is_decode={model_input.prompt_lens is None}"
            )
        if deferred_device_sampling and (read_from_device or async_read):
            raise ValueError(
                "deferred device sampling must retain device logits and cannot "
                f"read them during submission; read_from_device={read_from_device}, "
                f"async_read={async_read}"
            )
        self.count_decode_submission(model_input)
        if not any(bs > 0 for bs in batch_size_per_dp):
            return TTDecodeSubmission(
                tt_out=None,
                read_events=None,
                batch_size_per_dp=batch_size_per_dp,
                sampling_params=sampling_params,
                perform_device_sampling=perform_device_sampling,
                deferred_device_sampling=deferred_device_sampling,
                reload_plan=None,
            )

        kwargs: dict[str, Any] = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": runner.kv_caches,
            "start_pos": model_input.input_positions,
        }
        # Hybrid attention models route per-layer block tables; the
        # runner already populated ``block_tables_per_layer`` at
        # submission time when the kv_cache_config has multiple groups.
        # Legacy/uniform models leave it as ``None`` and never see the
        # kwarg.
        if model_input.block_tables_per_layer is not None:
            kwargs["page_tables_per_layer"] = model_input.block_tables_per_layer
        device_sampling_params = None
        # Speculative side tensors, sent only on a speculating decode step so a
        # model that never speculates keeps its present call shape. A verify
        # carries all three: a model needs the count to know which candidate
        # state to continue from, not only the draft count to know how much of
        # the block is real. An ordinary decode carries ``accepted_counts``
        # alone, and only while a row's previous step committed more than one
        # token, for the same reason.
        if model_input.num_valid_drafts is not None:
            kwargs["num_valid_drafts"] = model_input.num_valid_drafts
            kwargs["spec_mode"] = model_input.spec_mode
        if model_input.accepted_counts is not None:
            kwargs["accepted_counts"] = model_input.accepted_counts
        if model_input.spec_mode == ACCEPT_MODE_FUSED_SAMPLE:
            # Stage S2: a fused step with a sampled row carries the controls and
            # the pre-drawn uniforms the device accepts with. None (nothing
            # sent) for a greedy-only step, which is stage S1.
            from vllm_tt_plugin.model_runner import (
                fused_spec_sampling,
                spec_sampling_has_certificate_row,
            )

            spec_sampling = fused_spec_sampling(
                model_input,
                getattr(runner, "_spec_plan", None),
                vocab_size=getattr(runner, "vocab_size", None),
                device_top_p=getattr(runner, "_spec_device_top_p", False),
                penalty_tracker=(
                    runner._spec_penalty_tracker
                    if getattr(runner, "_spec_device_penalties", False)
                    else None
                ),
            )
            if spec_sampling is not None:
                kwargs["spec_sampling"] = spec_sampling
            if (
                async_read
                and self.chain_enabled()
                # A sampled step stays synchronous (stage S2, host overflow
                # redo) until deferred steps handle sampled overflow. A
                # certificate row (top_k 0) can fail and be redone on the
                # host, so it stays synchronous even when sampled steps chain.
                and (
                    spec_sampling is None
                    or (
                        spec_chain_sampled_enabled()
                        and not spec_sampling_has_certificate_row(spec_sampling)
                    )
                )
            ):
                # Chain mode: enqueue only, the reader thread waits on the
                # returned handle. Sent on every fused step, not only a chained
                # one, because the step after it may chain onto it.
                kwargs["defer_read"] = True
                if model_input.chain_rows is not None:
                    kwargs["chain_rows"] = model_input.chain_rows
                    kwargs["chain_tip"] = model_input.chain_tip
        if perform_device_sampling:
            sampling_param_dict = {
                field.name: (
                    getattr(sampling_params, field.name).tolist()
                    if getattr(sampling_params, field.name) is not None
                    else None
                )
                for field in fields(sampling_params)
            }
            sampling_param_dict["seed"] = [
                None if s == SEED_NONE_SENTINEL else s
                for s in sampling_param_dict["seed"]
            ]
            device_sampling_params = type(sampling_params)(**sampling_param_dict)
            if deferred_device_sampling:
                kwargs["defer_device_sampling"] = True
            else:
                kwargs["sampling_params"] = device_sampling_params
            if model_input.prompt_tokens is not None:
                assert model_input.output_tokens is not None
                kwargs["prompt_tokens"] = model_input.prompt_tokens
                kwargs["output_tokens"] = model_input.output_tokens
        # Standalone-plugin models already receive state-slot remaps in both
        # sampling modes. Preserve that behavior for legacy adapters too.
        if model_input.slot_remap is not None:
            kwargs["slot_remap"] = model_input.slot_remap
        if getattr(model_input, "finishing_slots", None):
            kwargs["finishing_slots"] = list(model_input.finishing_slots)

        # Versioned compatibility seam. Refactored tt-metal adapters opt into
        # the four explicit commands; legacy adapters keep their old call shape.
        reload_plan = None
        if contract_version >= 1:
            reload_plan = self.plan_decode_reload(model_input)
            kwargs.update(
                reload_inputs=reload_plan.reload_inputs,
                reload_page_table=reload_plan.reload_page_table,
                reload_sampling_params=reload_plan.reload_sampling_params,
                reset_sampling_state=reload_plan.reset_sampling_state,
            )
        elif perform_device_sampling:
            kwargs["reset_batch"] = model_input.decode_layout_changed
        if contract_version < 1 and not self._legacy_contract_warning_emitted:
            self._legacy_contract_warning_emitted = True
            if getattr(runner, "async_decode_scheduling", False):
                logger.warning(
                    "TT model %s does not advertise decode_input_update_contract "
                    ">= 1; preserving its legacy reset_batch reload behavior. "
                    "Async decode correctness is not guaranteed until the model "
                    "adapter implements the explicit contract.",
                    type(runner.model).__name__,
                )
            else:
                logger.info(
                    "TT model %s uses the legacy decode input reload contract "
                    "under synchronous scheduling; async overlap is disabled.",
                    type(runner.model).__name__,
                )

        enc_dec_kwargs: dict[str, Any] = {}
        if runner.request_specific_rope:
            if model_input.decode_layout_changed or any(
                req_id not in runner.previous_req_ids
                for req_id in runner.input_batch.req_ids
            ):
                enc_dec_kwargs = {
                    "rope_deltas_all_users": [
                        runner.requests[req_id].mrope_position_delta
                        for req_id in runner.input_batch.req_ids
                    ]
                }
            else:
                enc_dec_kwargs = {"rope_deltas_all_users": None}
            runner.previous_req_ids = set(runner.input_batch.req_ids)

        enable_trace = runner.trace_mode in ["all", "decode_only"]
        # A masked step takes the model's first half here and its second half in
        # ``finish_masked_decode``, once the sample-time mask exists.
        split = getattr(model_input, "spec_mask", None) is not None
        decode = (
            getattr(runner.model, "decode_forward_begin", None)
            if split
            else runner.model.decode_forward
        )
        if decode is None:
            raise AttributeError(
                f"TT model {type(runner.model).__name__} declares "
                "supports_spec_device_mask but has no decode_forward_begin / "
                "decode_forward_finish"
            )
        tt_out = decode(
            **kwargs,
            **enc_dec_kwargs,
            enable_trace=enable_trace,
            read_from_device=read_from_device,
        )
        spec_hidden = None
        deferred_fused = False
        requested_spec_mode = kwargs.get("spec_mode")
        if requested_spec_mode is not None:
            # A verify returns its mode's tensor inside a VerifyOutput, which
            # the read path below and every consumer above expect as a plain
            # host tensor. Unwrapped here, at the one boundary the model
            # returns through, rather than teaching each of them the type. A
            # ``topk`` verify stays a VerifyOutput, which the read path and
            # ``finalize_decode`` take for host output by its fields
            # (``_is_host_decode_output``) and hand to the walk unconverted.
            verify = tt_out
            tt_out = _verify_output_tensor(
                tt_out, type(runner.model).__name__, requested_spec_mode
            )
            # Safe after the unwrap, which refuses anything but a VerifyOutput.
            # A deferred fused verify (chain mode) has no host tensors: its
            # handle is what the reader waits on and what the proposal is
            # later handed back as ``hidden``.
            spec_hidden = verify.handle if verify.handle is not None else verify.hidden
            deferred_fused = (
                verify.handle is not None and verify.accepted_token_ids is None
            )

        elif isinstance(tt_out, VerifyOutput):
            raise TypeError(
                f"TT model {type(runner.model).__name__} returned a "
                "VerifyOutput from a decode step that asked for no verify; "
                "the runner sends spec_mode on every step it speculates on "
                "and on no other, so this step's return has no accepted "
                "count to be read against"
            )
        stage_outputs = None
        if runner.supports_stage_outputs:
            stage_outputs = pop_tt_stage_outputs(
                runner.model, model_input, runner.requests
            )
        # Input construction only proposed this layout/remap. Commit both at
        # the boundary where the model accepted the decode submission.
        runner.note_decode_layout_consumed()
        runner.note_decode_state_slots_settled()
        if reload_plan is not None:
            self.commit_decode_submission(model_input, reload_plan)
        else:
            # Preserve version-0 overlap behavior. The legacy adapter remains
            # reload-policy owner; this records only that submission succeeded.
            self._decode_chain_valid = True
            self._previous_device_sampling = perform_device_sampling
        read_events = None
        if async_read and deferred_fused:
            # The reader thread resolves the handle itself (``finalize_decode``).
            pass
        elif async_read:
            if hasattr(runner.model, "read_decode_output"):
                tt_out, read_events = cast(
                    tuple[Any, list[Any]],
                    runner.model.read_decode_output(
                        tt_out,
                        async_read=True,
                        **(
                            {"sample_rows": list(readback_rows)}
                            if readback_rows is not None
                            else {}
                        ),
                    ),
                )
            elif not _is_host_decode_output(tt_out):
                raise AttributeError(
                    "TT model must implement read_decode_output() "
                    "unless decode_forward() already returns host tensors"
                )
        return TTDecodeSubmission(
            tt_out=tt_out,
            read_events=read_events,
            batch_size_per_dp=batch_size_per_dp,
            sampling_params=sampling_params,
            perform_device_sampling=perform_device_sampling,
            device_sampling_params=device_sampling_params,
            deferred_device_sampling=deferred_device_sampling,
            reload_plan=reload_plan,
            spec_hidden=spec_hidden,
            readback_rows=readback_rows,
            stage_outputs=stage_outputs,
            mask_pending=split,
        )

    def finish_masked_decode(
        self,
        submission: TTDecodeSubmission,
        mask: torch.Tensor | None,
        *,
        async_read: bool,
    ) -> TTDecodeSubmission:
        """The second half of a masked fused verify: hand the model the mask.

        ``mask`` is int32 ``[rows, 1+K, W]`` (``spec_masks.step_mask``) or None
        for a step no row of which ended up constrained. The model enqueues the
        rest of the step. With ``defer_read`` (the first half asked for it) the
        result is the same handle, now readable by the reader thread; otherwise
        it is the verify's host output. Either way the submission is what
        ``submit_decode`` would have returned for a one-call step."""
        if not submission.mask_pending:
            raise RuntimeError(
                "finish_masked_decode: the submission is not a pending masked step"
            )
        model = self.runner.model
        out = model.decode_forward_finish(submission.tt_out, mask)
        self._mask_pending_step = None
        tt_out = _verify_output_tensor(
            out, type(model).__name__, ACCEPT_MODE_FUSED_SAMPLE
        )
        deferred = out.handle is not None and out.accepted_token_ids is None
        read_events = None
        if async_read and not deferred and hasattr(model, "read_decode_output"):
            tt_out, read_events = model.read_decode_output(tt_out, async_read=True)
        return replace(
            submission,
            tt_out=tt_out,
            read_events=read_events,
            spec_hidden=out.handle if out.handle is not None else out.hidden,
            mask_pending=False,
        )

    def complete_deferred_device_sampling(
        self,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
    ) -> TTFinalizedDecode:
        """Apply sample-time grammar on device, then finalize sampled tokens."""

        def fail_deferred() -> None:
            self._decode_chain_valid = False
            self._submitted_page_tables = None
            self.runner._poison_device_grammar()

        if not submission.deferred_device_sampling:
            raise ValueError(
                "decode submission is not awaiting deferred device sampling"
            )
        if submission.tt_out is None or submission.device_sampling_params is None:
            raise RuntimeError(
                "deferred device sampling is missing device logits or "
                "sampling parameters"
            )
        if len(model_input.grammar_bitmask) != 1:
            raise ValueError(
                "deferred device grammar expects one merged mask payload, got "
                f"{len(model_input.grammar_bitmask)}"
            )
        grammar_bitmask = model_input.grammar_bitmask[0]
        if grammar_bitmask is None:
            fail_deferred()
            raise RuntimeError(
                "structured decode reached deferred device sampling without "
                "a sample-time grammar bitmask"
            )

        sample_deferred = getattr(self.runner.model, "sample_deferred_decode", None)
        if not callable(sample_deferred):
            fail_deferred()
            raise AttributeError(
                "TT model declares device grammar support but does not implement "
                "sample_deferred_decode()"
            )
        try:
            sampled = sample_deferred(
                submission.tt_out,
                sampling_params=submission.device_sampling_params,
                grammar_bitmask=grammar_bitmask,
            )
            completed_submission = replace(
                submission,
                tt_out=sampled,
                read_events=None,
                deferred_device_sampling=False,
            )
            finalized = self.finalize_decode(completed_submission)
            if finalized is None:
                raise RuntimeError("deferred device sampling produced no output")
        except Exception:
            fail_deferred()
            raise
        self.device_grammar_sample_count += 1
        logger.debug(
            "TT device grammar sampling active count=%d",
            self.device_grammar_sample_count,
        )
        return finalized

    def _finalize_fused_handle(self, submission: TTDecodeSubmission) -> TTFinalizedDecode:
        """Resolve a deferred fused verify on the reader thread.

        ``model.read_decode_output(handle)`` (or ``handle.wait()``) blocks on
        the step's device event and returns ``FusedReadout`` (or the 3-tuple
        ``(accepted_token_ids, accepted_counts, p0)``). Touches no runner
        state. The result is the verify's ``VerifyOutput`` with its host fields
        filled in (including the logprob ``candidates``, when the readout has
        them), which the walk reads as it reads any fused verify, plus the
        device's ``p0``.
        """
        runner = self.runner
        verify = cast(VerifyOutput, submission.tt_out)
        handle = verify.handle
        reader = getattr(runner.model, "read_decode_output", None)
        raw = reader(handle) if reader is not None else handle.wait()
        candidates = None
        drafts = None
        if isinstance(raw, FusedReadout):
            ids, counts, p0 = raw.accepted_token_ids, raw.accepted_counts, raw.p0
            candidates = raw.candidates
            drafts = raw.drafts
        elif isinstance(raw, (tuple, list)) and len(raw) == 3:
            ids, counts, p0 = raw
        else:
            raise TypeError(
                f"TT model {type(runner.model).__name__} read a fused verify's "
                f"handle and returned {type(raw).__name__}; it returns a "
                "FusedReadout or (accepted_token_ids, accepted_counts, p0)"
            )
        self._latest_readout_drafts = drafts is not None
        if not isinstance(p0, torch.Tensor) or p0.dtype != torch.int32 or p0.dim() != 1:
            raise ValueError(
                "a fused verify's p0 must be a 1-D int32 host tensor, got "
                f"{type(p0).__name__} {getattr(p0, 'dtype', None)}"
            )
        out = VerifyOutput(
            spec_mode=ACCEPT_MODE_FUSED_SAMPLE,
            accepted_token_ids=ids,
            accepted_counts=counts,
            hidden=verify.hidden,
            handle=handle,
            candidates=candidates,
        )
        return TTFinalizedDecode(tt_out=out, tt_log_probs=None, p0=p0, drafts=drafts)

    def finalize_decode(
        self,
        submission: TTDecodeSubmission,
        *,
        sampling_rows: list[int] | None = None,
    ) -> TTFinalizedDecode | None:
        runner = self.runner
        if submission.tt_out is None:
            return None
        if submission.deferred_device_sampling:
            raise RuntimeError(
                "cannot finalize decode logits before deferred device sampling"
            )
        if getattr(submission, "mask_pending", False):
            raise RuntimeError(
                "cannot read a masked fused verify before its mask finish "
                "(decode_forward_finish)"
            )

        if self._deferred_handle(submission) is not None:
            return self._finalize_fused_handle(submission)

        if submission.readback_rows is not None and (
            sampling_rows is None or tuple(sampling_rows) != submission.readback_rows
        ):
            raise ValueError(
                f"Sampling rows {sampling_rows} differ from submitted readback "
                f"rows {submission.readback_rows}"
            )

        if submission.read_events is not None:
            for read_event in submission.read_events:
                ttnn.event_synchronize(read_event)
            tt_out = submission.tt_out
        else:
            tt_out = submission.tt_out

        is_host_output = _is_host_decode_output(tt_out)
        compact_rows = None
        if isinstance(tt_out, VerifyOutput) and not is_host_output:
            # ``process_decode_output_host`` converts a plain decode's device
            # output, one per data-parallel rank; it cannot read a VerifyOutput,
            # and the walk below needs the fields on the host. Refused by name
            # here rather than inside the model's hook.
            raise TypeError(
                f"TT model {type(runner.model).__name__} answered a verify in "
                f"spec_mode {tt_out.spec_mode!r} with fields that are not host "
                f"tensors {_device_verify_fields(tt_out)}; a VerifyOutput "
                "reaches the accept walk as it is, so decode_forward or "
                "read_decode_output must return its fields as host tensors"
            )
        if not is_host_output and hasattr(runner.model, "process_decode_output_host"):
            kwargs = {"is_tokens": submission.perform_device_sampling}
            if (
                sampling_rows is not None
                and not submission.perform_device_sampling
                and getattr(runner.model, "model_capabilities", {}).get(
                    "supports_compact_host_logits", False
                )
                and (
                    submission.readback_rows is not None
                    or runner.lane_batch.can_compact_host_sampling()
                )
            ):
                compact_rows = tuple(sampling_rows)
                kwargs["sample_rows"] = list(compact_rows)
            tt_out = runner.model.process_decode_output_host(tt_out, **kwargs)
        elif not is_host_output:
            raise AttributeError(
                "TT model must implement process_decode_output_host() "
                "unless decode output is already a torch tensor"
            )

        tt_log_probs = None
        assert isinstance(submission.sampling_params.enable_log_probs, torch.Tensor)
        if (
            submission.perform_device_sampling
            and submission.sampling_params.enable_log_probs.any()
        ):
            assert isinstance(tt_out, tuple) and len(tt_out) == 2
            tt_out, tt_log_probs = tt_out
        elif isinstance(tt_out, tuple):
            tt_out, _ = tt_out

        if compact_rows is not None:
            tt_out = TTCompactedHostLogits(logits=tt_out, rows=compact_rows)
        return TTFinalizedDecode(tt_out=tt_out, tt_log_probs=tt_log_probs)
