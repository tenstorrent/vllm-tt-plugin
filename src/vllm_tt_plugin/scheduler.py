# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from enum import Enum
from typing import cast

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

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

    # Host-only sampling controls and their neutral values: any of these
    # forces the step onto host sampling (check_perform_device_sampling),
    # which cannot construct a multi-token canvas and kills the engine.
    _BLOCK_HOST_ONLY_SAMPLING_NEUTRAL = (
        ("min_p", 0.0),
        ("min_tokens", 0),
        ("logit_bias", None),
        ("allowed_token_ids", None),
        ("bad_words", None),
        ("_bad_words_token_ids", None),
    )

    def add_request(self, request: Request) -> None:
        if self._is_block_output_model:
            existing = self.requests.get(request.request_id)
            if existing is not None and existing.streaming_queue is None:
                # A continuation (next input chunk or the closing sentinel) of
                # a session whose resumable flag the neutralization below
                # scrubbed. The base scheduler asserts on the missing
                # streaming_queue, which would tear down EngineCore. The
                # session cannot accept more input, so drop the message; the
                # live request finishes and notifies its client on its own.
                logger.warning(
                    "Dropping streaming-input continuation for request %s: "
                    "block-output models do not support resumable sessions",
                    request.request_id,
                )
                return
            self._truncate_unservable_block_prompt(request)
            self._align_block_output_max_tokens(request)
            self._neutralize_block_output_host_sampling(request)
        super().add_request(request)

    def _truncate_unservable_block_prompt(self, request: Request) -> None:
        """Contain a bypassed prompt that leaves no room for a whole canvas.

        Frontend validation rejects such prompts; a prebuilt EngineCoreRequest
        skips it, and neither upstream EngineCore.add_request nor the base
        scheduler re-checks prompt length. Admitted untouched, the prompt
        either can never be scheduled (chunked prefill is disabled and the
        prompt exceeds the token budget: parked in WAITING forever,
        head-of-line blocking every later request), overflows the worker's
        max_model_len-wide token buffer, or — even when it fits
        max_model_len — trips the adapter's own canvas-capacity validation,
        which raises out of execute_model in eager (no-upfront-capture) mode
        where there is no graceful stop-canvas rejection. Truncate to the
        largest tile-aligned prompt that still fits one whole canvas below
        the tile-floored max_model_len, so every admitted request is
        genuinely servable and finishes through the normal notifying path.
        Prefix caching is disabled for block models (__init__ asserts it), so
        the request's stale block hashes stay inert.
        """
        from vllm_tt_plugin.platform import _TT_TOKEN_TILE_SIZE

        if request.mm_features:
            # A text-only block model has a zero encoder budget: a feature at
            # offset 0 forces zero-token schedules forever (head-of-line
            # stall), and an interior offset carves a partial prefill chunk
            # that flips the step onto host sampling and kills the engine.
            # Dropped, the placeholder positions decode as ordinary tokens.
            logger.warning(
                "Request %s bypassed frontend validation with multimodal "
                "features a block-output model cannot encode; dropping them",
                request.request_id,
            )
            request.mm_features = []
        if request.prompt_token_ids is None and request.num_prompt_tokens > 0:
            # The frontend rejects prompt_embeds for every TT model; admitted
            # bare, the worker's request-state builder raises
            # NotImplementedError out of execute_model. Replace with
            # placeholder tokens; an embeds-only Request already carries
            # [0] * num_prompt_tokens in _all_token_ids, so the replacement
            # keeps every derived view consistent.
            logger.warning(
                "Request %s bypassed frontend validation with a "
                "prompt_embeds-only prompt the TT backend does not support; "
                "replacing with %d placeholder tokens",
                request.request_id,
                request.num_prompt_tokens,
            )
            request.prompt_token_ids = [0] * request.num_prompt_tokens
            request.prompt_embeds = None
        if request.num_prompt_tokens == 0:
            # The frontend rejects empty prompts; admitted bare, the waiting
            # loop schedules zero new tokens and upstream's num_new_tokens
            # assert tears down the engine. Pad to one placeholder token so
            # the request schedules and finishes through the normal path.
            logger.warning(
                "Request %s bypassed frontend validation with an empty "
                "prompt; padding to one placeholder token",
                request.request_id,
            )
            request.prompt_token_ids = [0]
            request.prompt_embeds = None
            request._all_token_ids.append(0)
            request.num_prompt_tokens = 1
            return
        tile = _TT_TOKEN_TILE_SIZE
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        aligned_max_model_len = max_model_len // tile * tile
        keep = (aligned_max_model_len - self._output_tokens_per_step) // tile * tile
        if request.num_prompt_tokens <= keep:
            return
        logger.warning(
            "Request %s bypassed frontend validation with a %d-token prompt "
            "that leaves no room for a whole %d-token canvas within "
            "max_model_len; truncating to %d tokens so the request can "
            "finish length-capped",
            request.request_id,
            request.num_prompt_tokens,
            self._output_tokens_per_step,
            keep,
        )
        if request.prompt_token_ids is not None:
            request.prompt_token_ids = request.prompt_token_ids[:keep]
        if getattr(request, "prompt_embeds", None) is not None:
            request.prompt_embeds = request.prompt_embeds[:keep]
        del request._all_token_ids[keep:]
        request.num_prompt_tokens = keep

    def _align_block_output_max_tokens(self, request: Request) -> None:
        """Clamp max_tokens so a bypassed EngineCoreRequest cannot overshoot.

        Frontend validation rejects an oversized limit. Prebuilt requests skip
        that path; the last canvas would then be applied past max_model_len and
        kill the engine. Shrink the logical cap to the largest whole-canvas
        budget that still fits. Lane mode reaches this through each lane.
        """
        from vllm_tt_plugin.platform import _fit_block_output_max_tokens

        prompt_ids = request.prompt_token_ids
        if prompt_ids is None or request.sampling_params is None:
            return
        fitted = _fit_block_output_max_tokens(
            len(prompt_ids),
            request.max_tokens,
            self._output_tokens_per_step,
            int(self.vllm_config.model_config.max_model_len),
        )
        if fitted == request.max_tokens:
            return
        # The prompt truncation above guarantees at least one whole canvas
        # fits, so a clamp always leaves a positive, servable budget.
        logger.debug(
            "Clamping block-output max_tokens from %s to %s for request %s "
            "so physical canvases fit max_model_len",
            request.max_tokens,
            fitted,
            request.request_id,
        )
        request.max_tokens = fitted
        request.sampling_params.max_tokens = fitted

    def _neutralize_block_output_host_sampling(self, request: Request) -> None:
        """Strip controls a block-output model cannot honor from a bypassed
        request: host-sampling forcers, structured outputs, and resumable
        streaming-input sessions.

        Frontend validation rejects these; a prebuilt EngineCoreRequest skips
        it, and any of them flips the step to host sampling mid-flight (which
        cannot construct a multi-token canvas) or parks the request forever.
        Raising here is no safer: an add_request exception also tears down
        EngineCore, so neutralize instead, the way the sampling controls are
        neutralized at the frontend.
        """
        params = request.sampling_params
        if params is None:
            return
        stripped = []
        if request.structured_output_request is not None:
            request.structured_output_request = None
            stripped.append("structured_outputs")
            # Request.__init__ parked the request on grammar compilation; with
            # the structured-output request gone nothing would ever promote it
            # out of skipped_waiting, so restore schedulability.
            if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
                request.status = RequestStatus.WAITING
        if params.structured_outputs is not None:
            params.structured_outputs = None
        if request.resumable:
            # A resumable session parks the stopped request to wait for more
            # input instead of finishing it, permanently leaking the
            # model-owned state slot. The frontend rejects it for block models.
            request.resumable = False
            stripped.append("resumable")
        for field, neutral in self._BLOCK_HOST_ONLY_SAMPLING_NEUTRAL:
            if getattr(params, field, neutral) not in (neutral, [], {}):
                setattr(params, field, neutral)
                stripped.append(field.lstrip("_"))
        if stripped:
            logger.warning(
                "Request %s bypassed frontend validation; stripped "
                "controls unsupported by block-output models: %s",
                request.request_id,
                ", ".join(stripped),
            )

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
