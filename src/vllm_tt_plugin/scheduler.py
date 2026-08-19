# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from enum import Enum
from typing import cast

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from vllm_tt_plugin.config import (
    get_tt_output_tokens_per_step,
    is_tt_block_output_model,
)
from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


class TTSchedulingMode(Enum):
    DEFAULT = "default"
    DECODE_ONLY = "decode_only"
    PREFILL_ONLY = "prefill_only"

    @classmethod
    def from_prefill_intent(cls, prefill_intent: int) -> "TTSchedulingMode":
        if prefill_intent == 0:
            return cls.DECODE_ONLY
        if prefill_intent == 1:
            return cls.PREFILL_ONLY
        raise ValueError(f"Invalid TT scheduling intent: {prefill_intent}")


class TTScheduler(AsyncScheduler):
    """Scheduler for the TT (Tenstorrent) platform.

    TT constraints:
    - No mixed prefill+decode batches: each batch is either all-prefill
      or all-decode.
    - Token-chunked prefill is allowed: a long prefill may be split across
      steps. After a partial chunk the base scheduler keeps the request in
      ``running`` with ``is_prefill_chunk=True``; later prefill steps schedule
      the next chunk until the prompt is fully computed. Those continuations
      count as prefill work here, never as decodes.

    The base scheduler holds temporarily blocked prefill requests (e.g. while
    a structured-output grammar compiles) in ``skipped_waiting`` rather than
    ``waiting``. Its prefill loop revisits that queue and promotes requests
    whose dependency is ready. Decode-only scheduling must hide both queues so
    that the base scheduler cannot admit a prefill into a decode step.

    Inherits from AsyncScheduler to get num_output_placeholders support.
    TT uses this scheduler in both sync and async execution modes:
    - with async_scheduling=False, it behaves as the single TT scheduler
      without execution overlap
    - with async_scheduling=True, placeholders allow decode requests to be
      re-scheduled before update_from_output processes the previous step's
      results, enabling host/device overlap
    - under async scheduling a preempted request keeps the tokens it had
      already scheduled but not yet received, and needs no TT-side handling
      for them. Those tokens are valid: the forward that produced them ran to
      completion before the preempt freed any block, device submits form a
      strict queue, and every async op is forced to complete before the next
      prefill, so no later write can reach the KV they were computed against.
      The base class appends them on arrival and the resumed prefill replays
      them. ``Request.async_tokens_to_discard`` serves the wholesale
      ``reset_prefix_cache`` teardown only; wiring ordinary preemption into it
      drops valid tokens and silently truncates the response.

    Supports ``set_forced_mode`` for lane coordination:
    - ``TTSchedulingMode.DECODE_ONLY`` forces decode-only (even if waiting
      queue is non-empty).
    - ``TTSchedulingMode.PREFILL_ONLY`` forces prefill-only (and may return an
      empty batch when there is no pending prefill work).
    - ``TTSchedulingMode.DEFAULT`` uses the default policy: prefer prefill
      when pending prefill work exists, but fall back to decode-only if
      prefill cannot make progress and running decode requests exist.
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_mode = TTSchedulingMode.DEFAULT
        self._output_tokens_per_step = get_tt_output_tokens_per_step(self.vllm_config)
        self._is_block_output_model = is_tt_block_output_model(self.vllm_config)
        if self._is_block_output_model:
            assert self.num_sampled_tokens_per_step == 1, (
                "Block-output accounting requires upstream to reserve exactly "
                "one sampled-token placeholder"
            )
            assert not self.kv_cache_manager.enable_caching, (
                "Block-output models bypass AsyncScheduler.cache_blocks and "
                "must disable prefix caching"
            )

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

    def _has_pending_prefill(self) -> bool:
        """Whether any request still needs prefill work.

        A request in ``skipped_waiting`` still needs a future prefill pass:
        that is where the base scheduler retries promotion after its dependency
        becomes ready. In decode-only mode this check also ensures both waiting
        queues are hidden from the base scheduler.

        A running ``is_prefill_chunk`` request is a partial prefill whose next
        chunk can only be scheduled by a prefill step.
        """
        return (
            bool(self.waiting)
            or bool(getattr(self, "skipped_waiting", False))
            or any(request.is_prefill_chunk for request in self.running)
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # NOTE: `throttle_prefills` accepted for interface compatibility with the base
        #        scheduler but unused - TT separates prefill/decode explicitly.
        has_pending_prefill = self._has_pending_prefill()
        # A partial prefill occupies ``running`` but cannot produce a decode
        # token, so it must not make the decode fallback below look viable.
        has_running_decode = any(
            not request.is_prefill_chunk for request in self.running
        )
        mode = self._forced_mode

        if mode == TTSchedulingMode.PREFILL_ONLY:
            # Forced mode is shared by every lane. Return an empty prefill
            # result unchanged so the coordinator can decide whether all lanes
            # should fall back to decode together.
            result = self._schedule_prefill_only()
            return self._finalize_scheduler_output(result)
        if mode == TTSchedulingMode.DECODE_ONLY:
            if has_pending_prefill:
                # Hide the waiting queues and partial prefills so the base
                # scheduler cannot admit prefill work.
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result)
            # No pending prefill: base scheduler naturally runs decode-only.
            result = super().schedule()
            return self._finalize_scheduler_output(result)

        # Default mode:
        # Prefer prefill whenever prefill work is pending, so new requests are
        # admitted and partial prefills advance.
        if has_pending_prefill:
            prefill_result = self._schedule_prefill_only()
            # If prefill cannot make progress (e.g. KV pressure), do not stall
            # decode. Fall back to decode-only so running requests can advance
            # and free capacity for a later prefill admission.
            if prefill_result.total_num_scheduled_tokens == 0 and has_running_decode:
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result)
            return self._finalize_scheduler_output(prefill_result)

        # No pending prefill work in default mode: run decode-only naturally.
        result = super().schedule()
        return self._finalize_scheduler_output(result)

    def _finalize_scheduler_output(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        return scheduler_output

    def _schedule_prefill_only(self) -> SchedulerOutput:
        """Schedule prefill work: waiting requests and partial continuations.

        Temporarily hides the running *decode* requests so the base scheduler's
        running loop only advances partial prefills, and its waiting loop
        admits new ones.  Adjusts max_num_running_reqs so the waiting loop
        respects the true capacity with the decodes hidden.
        """
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_max = self.max_num_running_reqs
        self.running = cast(list[Request], partial_prefills)
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
        return result

    def _schedule_decode_only(self) -> SchedulerOutput:
        """Schedule only running decode requests.

        Temporarily hides both the ``waiting`` and ``skipped_waiting`` queues
        so the base scheduler's waiting loop is a no-op and cannot promote a
        grammar-ready structured-output request into this decode step, and
        hides partial prefills so their next chunk is not scheduled into it
        either.  Any requests that get preempted during decode scheduling are
        merged back into the original queues afterwards.
        """
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_waiting = self.waiting
        saved_skipped = getattr(self, "skipped_waiting", None)
        self.waiting = create_request_queue(self.policy)
        if saved_skipped is not None:
            self.skipped_waiting = create_request_queue(self.policy)
        if partial_prefills:
            self.running = [r for r in self.running if not r.is_prefill_chunk]
        try:
            result = super().schedule()
        finally:
            if self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            if saved_skipped is not None:
                if self.skipped_waiting:
                    saved_skipped.prepend_requests(self.skipped_waiting)
                self.skipped_waiting = saved_skipped
            self.waiting = saved_waiting
            if partial_prefills:
                self.running.extend(partial_prefills)
        return result

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Backstop against a stale-canvas resume vLLM's AR reset cannot represent.

        The engine layer (``_install_block_output_reset_abort_patch``) aborts
        running block requests and notifies their clients before delegating,
        so a reset requested through ``EngineCoreProc`` succeeds. Live block
        requests reach this guard only from engines that cannot notify the
        owning clients (a bare in-process ``EngineCore``) or callers that
        bypassed the engine layer; refusing here beats silently removing a
        request someone is still waiting on.
        """
        if self._is_block_output_model and reset_running_requests and self.running:
            message = (
                "Cannot reset prefix cache while a block-output request is "
                "running; finish or abort the request first."
            )
            # pause_generation(mode="keep") reaches this through an unguarded
            # idle-state callback while intentionally retaining live requests.
            # Returning False preserves them without letting an exception
            # escape EngineCore and strand the callback's Future.
            if self.pause_state == PauseState.PAUSED_ALL:
                logger.error("%s", message)
                return False
            raise RuntimeError(message)
        return super().reset_prefix_cache(reset_running_requests, reset_connector)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Reserve the complete physical output emitted by each block step.

        The platform removes the upstream diffusion marker, so vLLM reserves
        its normal one sampled-token placeholder. The TT adapter returns one
        K-token canvas; reserve the remaining K-1 positions.
        """
        super()._update_after_schedule(scheduler_output)
        if not self._is_block_output_model:
            return
        extra_placeholders = (
            self._output_tokens_per_step - self.num_sampled_tokens_per_step
        )
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if not request.is_prefill_chunk:
                request.num_output_placeholders += extra_placeholders

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        """Commit one block and reconcile its full physical reservation."""
        if not self._is_block_output_model:
            return super()._update_request_with_output(request, new_token_ids)
        if request.async_tokens_to_discard:
            raise RuntimeError(
                "A stale async output reached synchronous block serving; "
                "block-output async scheduling and running prefix resets are "
                "unsupported"
            )
        if len(new_token_ids) != self._output_tokens_per_step:
            raise ValueError(
                "Model output width violates output_tokens_per_step: "
                f"{len(new_token_ids)} != {self._output_tokens_per_step}"
            )

        # Scheduler appends token-by-token and trims at EOS, stop tokens,
        # max_tokens, or max_model_len. The reservation is physical, so consume
        # all K placeholders even when the client-visible block is trimmed.
        # Calling Scheduler directly intentionally skips AsyncScheduler's
        # cache_blocks hook. Platform validation disables prefix caching for
        # block-output models, and __init__ asserts that invariant.
        new_token_ids, stopped = Scheduler._update_request_with_output(
            self, request, new_token_ids
        )
        request.num_output_placeholders -= self._output_tokens_per_step
        if request.num_output_placeholders < 0:
            raise RuntimeError(
                "Output placeholders underflowed after block-output reconciliation"
            )
        return new_token_ids, stopped
