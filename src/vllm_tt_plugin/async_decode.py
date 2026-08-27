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
from vllm_tt_plugin.scheduler import get_tt_forced_reset_discard_counts
from vllm_tt_plugin.structured_output import has_structured_outputs

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_tt_plugin.model_input import TTDecodeReloadPlan, TTModelInput
    from vllm_tt_plugin.model_runner import TTModelRunner

logger = init_tt_logger(__name__)


@dataclass(frozen=True)
class TTDecodeSubmission:
    """Carries the raw result of decode submission until finalization."""

    tt_out: Any | None
    read_events: list[Any] | None
    batch_size_per_dp: list[int]
    sampling_params: Any
    perform_device_sampling: bool
    reload_plan: TTDecodeReloadPlan | None = None


@dataclass(frozen=True)
class TTFinalizedDecode:
    """Normalized decode result after TT event waits and host processing."""

    tt_out: torch.Tensor
    tt_log_probs: torch.Tensor | None


def _is_host_decode_output(tt_out: Any) -> bool:
    if isinstance(tt_out, torch.Tensor):
        return True
    if isinstance(tt_out, tuple):
        return all(
            tensor is None or isinstance(tensor, torch.Tensor) for tensor in tt_out
        )
    return False


@dataclass(frozen=True)
class SubmittedStepContext:
    """Immutable snapshot of the host state associated with one decode submit."""

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    submit_time_ns: int


@dataclass
class CompletedDecodeStep:
    """Decode output that has completed readback but is not yet applied."""

    sampled_token_ids: torch.Tensor
    logprobs: LogprobsLists | None
    context: SubmittedStepContext
    completion_time_ns: int
    runner_output: ModelRunnerOutput | None = None


class DeferredDecodeOutput(AsyncModelRunnerOutput):
    """Run the deferred device readback exactly once, from whichever caller
    reaches it first.

    Two callers race for the same step from different threads: vLLM's
    ``UniProcExecutor`` resolves it via ``get_output`` on its async-output
    thread when async scheduling is on, while the runner's drain
    (``TTAsyncDecodeController.wait_for_all_pending_async_steps``) resolves it
    via ``ensure_finalized`` on the engine thread. ``_finalize_lock`` makes the
    readback run exactly once across both threads; a second concurrent readback
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
        self._submitted_page_tables: tuple[torch.Tensor, ...] | None = None
        self._legacy_contract_warning_emitted = False

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
        transition = (
            not self._decode_chain_valid
            or model_input.decode_layout_changed
            or sampling_mode_changed
        )
        reload_inputs = (
            not device_sampling
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
        """Requests whose older output must not reach scheduler accounting.

        Ordinary preemption and resume deliberately do not appear here: their
        in-flight token is valid and AsyncScheduler must consume its output
        placeholder. Forced-reset frames also stay published so
        ``async_tokens_to_discard`` consumes the stale frame itself.
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
        if model_input.grammar_bitmask[0] is not None:
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

    def enqueue_completed_decode_step(self, completed: CompletedDecodeStep) -> None:
        with self.runner._steady_decode_lock:
            self.runner._completed_decode_steps.append(completed)

    def register_pending_async_step(
        self,
        step: DeferredDecodeOutput,
        *,
        overlap_ok: bool,
    ) -> None:
        with self.runner._steady_decode_lock:
            self.runner._pending_async_steps.append(step)
            self.runner._pending_async_overlap_ok.append(overlap_ok)

    def prune_finished_async_events(self) -> None:
        with self.runner._steady_decode_lock:
            while (
                self.runner._pending_async_steps
                and self.runner._pending_async_steps[0].is_resolved()
            ):
                self.runner._pending_async_steps.popleft()
                self.runner._pending_async_overlap_ok.popleft()

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
        completed_steps = self.drain_completed_decode_steps()
        suppressed = set(suppress_output_req_ids or ())
        discard_counts = forced_reset_discard_counts or {}

        # An output can already have reached AsyncScheduler while its runner
        # state apply is still queued. Such accepted frames precede the stale
        # frames counted at reset time, so skip only the newest N matching rows.
        remaining_rows: Counter[str] = Counter()
        for completed in completed_steps:
            if completed.runner_output is None:
                continue
            for req_id in completed.context.req_ids:
                req_idx = completed.runner_output.req_id_to_index.get(req_id)
                if (
                    req_idx is not None
                    and completed.runner_output.sampled_token_ids[req_idx]
                ):
                    remaining_rows[req_id] += 1

        missing = {
            req_id: count - remaining_rows[req_id]
            for req_id, count in discard_counts.items()
            if remaining_rows[req_id] < count
        }
        if missing:
            raise RuntimeError(
                "Forced-reset async discard boundary is missing completed "
                f"frames: {missing}"
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
            forced_reset_rows = {
                req_id
                for req_id in published_req_ids
                if 0 < remaining_rows[req_id] <= discard_counts.get(req_id, 0)
            }
            self.apply_completed_decode_step(
                completed,
                suppress_output_req_ids=suppressed,
                skip_state_req_ids=suppressed | forced_reset_rows,
            )
            remaining_rows.subtract(published_req_ids)
        self.prune_finished_async_events()

    def wait_for_all_pending_async_steps(self, *, apply_completed: bool = True) -> None:
        # Drive each pending readback to completion here rather than blocking on
        # its event: the engine has not popped these futures yet (and on 0.22
        # will not until after this returns), so nothing else will set the
        # events. ``ensure_finalized`` is idempotent, so the engine's later
        # ``get_output`` on the same step returns the cached result.
        with self.runner._steady_decode_lock:
            steps = list(self.runner._pending_async_steps)
        for step in steps:
            step.ensure_finalized()
        if apply_completed:
            self.apply_ready_completed_decode_steps()

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
            if scheduler_output is not None and get_tt_forced_reset_discard_counts(
                scheduler_output
            ):
                return True
            if not steady_decode_candidate:
                return True
            return any(
                not overlap_ok for overlap_ok in self.runner._pending_async_overlap_ok
            )

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
        finalized = self.finalize_decode(submission)
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
        )

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
        completed: CompletedDecodeStep,
        *,
        suppress_output_req_ids: set[str] | None = None,
        skip_state_req_ids: set[str] | None = None,
    ) -> None:
        suppressed = set(suppress_output_req_ids or ())
        skipped_state = set(skip_state_req_ids or ())
        # The batch-queue loop schedules the current step before consuming the
        # previous future. Scrub finished rows before
        # scheduler.update_from_output observes that cached output. Forced-reset
        # rows intentionally remain intact: AsyncScheduler must see them to
        # decrement ``async_tokens_to_discard``.
        if completed.runner_output is not None:
            assert self.runner.scheduler_config.async_scheduling, (
                "mutating a published runner output is only ordered correctly "
                "under the batch-queue step loop"
            )
            for req_id in suppressed:
                req_idx = completed.runner_output.req_id_to_index.get(req_id)
                if req_idx is not None:
                    completed.runner_output.sampled_token_ids[req_idx] = []
        self.runner._apply_sampled_tokens_to_state(
            sampled_token_ids=completed.sampled_token_ids,
            req_ids=completed.context.req_ids,
            skip_req_ids=skipped_state,
        )

    def submit_async_decode(
        self,
        model_input: TTModelInput,
        *,
        steady_decode_fast_path: bool,
    ) -> AsyncTTModelRunnerOutput:
        event = threading.Event()
        context = self.capture_submitted_step_context()
        submission = self.submit_decode(
            model_input,
            read_from_device=False,
            async_read=True,
        )
        if submission.tt_out is None:
            event.set()
        step = AsyncTTModelRunnerOutput(
            controller=self,
            submission=submission,
            model_input=model_input,
            completion_event=event,
            context=context,
        )
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
            model_input, read_from_device=False, async_read=True
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
    ) -> TTDecodeSubmission:
        runner = self.runner
        batch_size_per_dp = model_input.unpadded_batch_size
        if not isinstance(batch_size_per_dp, list):
            batch_size_per_dp = [batch_size_per_dp]

        sampling_params = model_input.tt_sampling_params
        perform_device_sampling = model_input.perform_device_sampling
        contract_version = self.decode_input_update_contract_version()
        if not any(bs > 0 for bs in batch_size_per_dp):
            return TTDecodeSubmission(
                tt_out=None,
                read_events=None,
                batch_size_per_dp=batch_size_per_dp,
                sampling_params=sampling_params,
                perform_device_sampling=perform_device_sampling,
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
            kwargs["sampling_params"] = type(sampling_params)(**sampling_param_dict)
            if model_input.prompt_tokens is not None:
                assert model_input.output_tokens is not None
                kwargs["prompt_tokens"] = model_input.prompt_tokens
                kwargs["output_tokens"] = model_input.output_tokens
        # Standalone-plugin models already receive state-slot remaps in both
        # sampling modes. Preserve that behavior for legacy adapters too.
        if model_input.slot_remap is not None:
            kwargs["slot_remap"] = model_input.slot_remap

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
            logger.warning(
                "TT model %s does not advertise decode_input_update_contract "
                ">= 1; preserving its legacy reset_batch reload behavior. "
                "Async decode correctness is not guaranteed until the model "
                "adapter implements the explicit contract.",
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
        tt_out = runner.model.decode_forward(
            **kwargs,
            **enc_dec_kwargs,
            enable_trace=enable_trace,
            read_from_device=read_from_device,
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
        if async_read:
            if hasattr(runner.model, "read_decode_output"):
                tt_out, read_events = cast(
                    tuple[Any, list[Any]],
                    runner.model.read_decode_output(tt_out, async_read=True),
                )
            else:
                is_host_tensor = isinstance(tt_out, torch.Tensor)
                is_host_tensor_tuple = isinstance(tt_out, tuple) and all(
                    tensor is None or isinstance(tensor, torch.Tensor)
                    for tensor in tt_out
                )
                if not (is_host_tensor or is_host_tensor_tuple):
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
            reload_plan=reload_plan,
        )

    def finalize_decode(
        self,
        submission: TTDecodeSubmission,
    ) -> TTFinalizedDecode | None:
        runner = self.runner
        if submission.tt_out is None:
            return None

        if submission.read_events is not None:
            for read_event in submission.read_events:
                ttnn.event_synchronize(read_event)
            tt_out = submission.tt_out
        else:
            tt_out = submission.tt_out

        is_host_output = _is_host_decode_output(tt_out)
        if not is_host_output and hasattr(runner.model, "process_decode_output_host"):
            tt_out = runner.model.process_decode_output_host(
                tt_out,
                is_tokens=submission.perform_device_sampling,
            )
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

        return TTFinalizedDecode(tt_out=tt_out, tt_log_probs=tt_log_probs)
