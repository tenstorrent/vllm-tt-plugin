# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from enum import Enum
from typing import cast

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request

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
    - No chunked prefill: each prefill must be scheduled in full.

    The base scheduler holds prefill requests that are temporarily blocked
    (e.g. ``WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`` while a grammar compiles)
    in a separate ``skipped_waiting`` queue, not ``waiting``. Its waiting loop
    drains both queues and can promote a now-ready blocked request into the
    same step that schedules running decodes. To preserve the all-prefill or
    all-decode invariant, every place that inspects or hides pending prefill
    work must treat ``waiting`` and ``skipped_waiting`` together.

    Inherits from AsyncScheduler to get num_output_placeholders support.
    TT uses this scheduler in both sync and async execution modes:
    - with async_scheduling=False, it behaves as the single TT scheduler
      without execution overlap
    - with async_scheduling=True, placeholders allow decode requests to be
      re-scheduled before update_from_output processes the previous step's
      results, enabling host/device overlap

    Supports ``set_forced_mode`` for DP-gather coordination:
    - ``TTSchedulingMode.DECODE_ONLY`` forces decode-only (even if waiting
      queue is non-empty).
    - ``TTSchedulingMode.PREFILL_ONLY`` forces prefill-only (and may return an
      empty batch when waiting is empty).
    - ``TTSchedulingMode.DEFAULT`` uses the default policy: prefer prefill
      when waiting is non-empty, but fall back to decode-only if prefill
      cannot admit any request and running decode requests exist.
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_mode = TTSchedulingMode.DEFAULT

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

    def _has_pending_prefill(self) -> bool:
        """Whether any request is waiting to be prefilled.

        Includes ``skipped_waiting`` so a request blocked on grammar
        compilation still routes scheduling through the prefill-only path
        instead of leaking into a decode step.
        """
        return bool(self.waiting) or bool(self.skipped_waiting)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        has_waiting = self._has_pending_prefill()
        has_running = bool(self.running)
        mode = self._forced_mode

        if mode == TTSchedulingMode.PREFILL_ONLY:
            # If waiting is empty, this intentionally returns an empty batch.
            result = self._schedule_prefill_only(throttle_prefills)
            return self._finalize_scheduler_output(result)
        if mode == TTSchedulingMode.DECODE_ONLY:
            if has_waiting:
                # Hide waiting so base scheduler cannot admit prefill.
                result = self._schedule_decode_only(throttle_prefills)
                return self._finalize_scheduler_output(result)
            # No waiting requests: base scheduler naturally runs decode-only.
            result = super().schedule(throttle_prefills)
            return self._finalize_scheduler_output(result)

        # Default mode:
        # Prefer prefill whenever waiting is non-empty to admit new requests.
        if has_waiting:
            prefill_result = self._schedule_prefill_only(throttle_prefills)
            # If waiting is non-empty but prefill cannot be admitted (e.g. KV
            # pressure and no chunked prefill), do not stall decode progress.
            # Fall back to decode-only so running requests can advance and free
            # capacity for later full-prefill admission.
            if prefill_result.total_num_scheduled_tokens == 0 and has_running:
                result = self._schedule_decode_only(throttle_prefills)
                return self._finalize_scheduler_output(result)
            return self._finalize_scheduler_output(prefill_result)

        # No waiting requests in default mode: run decode-only naturally.
        result = super().schedule(throttle_prefills)
        return self._finalize_scheduler_output(result)

    def _finalize_scheduler_output(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        return scheduler_output

    def _schedule_prefill_only(
        self, throttle_prefills: bool = False
    ) -> SchedulerOutput:
        """Schedule only waiting (prefill) requests.

        Temporarily hides the running (decode) requests so the base
        scheduler's running loop iterates zero times and only the
        waiting loop executes.  Adjusts max_num_running_reqs so the
        waiting loop respects the true capacity.
        """
        saved_running = self.running
        saved_max = self.max_num_running_reqs
        self.running = cast(list[Request], [])
        self.max_num_running_reqs = max(0, saved_max - len(saved_running))
        try:
            result = super().schedule(throttle_prefills)
        finally:
            self.running = saved_running + self.running
            self.max_num_running_reqs = saved_max
        return result

    def _schedule_decode_only(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Schedule only running (decode) requests.

        Temporarily hides both the ``waiting`` and ``skipped_waiting`` queues
        so the base scheduler's waiting loop is a no-op and cannot promote a
        grammar-ready structured-output request into this decode step.  Any
        requests that get preempted during decode scheduling are merged back
        into the original queues afterwards.
        """
        saved_waiting = self.waiting
        saved_skipped_waiting = self.skipped_waiting
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)
        try:
            result = super().schedule(throttle_prefills)
        finally:
            if self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            if self.skipped_waiting:
                saved_skipped_waiting.prepend_requests(self.skipped_waiting)
            self.waiting = saved_waiting
            self.skipped_waiting = saved_skipped_waiting
        return result
