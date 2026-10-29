# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import os
from collections import deque
from collections.abc import Mapping
from enum import Enum
from typing import cast

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm_tt_plugin import sched_events
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin import sched_defer_final, sched_final_step
from vllm_tt_plugin.sched_defer_final import DeferFinalPolicy
from vllm_tt_plugin.sched_interleave import InterleavePolicy
from vllm_tt_plugin.sched_pack import PackPolicy
from vllm_tt_plugin.sched_release_first import ReleaseFirstPolicy

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


# HF ``model_type`` values whose generator resumes a prefill only at a multiple
# of the token budget (the V4.1 generator's chunk): ``_schedule_prefill_only``
# admits one new prefill a step and advances a partial prefill alone.
ALIGNED_CHUNK_MODEL_TYPES = {"deepseek_v41"}

# The ``TT_SCHED_*`` environment variables this tree implements. A scheduling
# policy is one variable, parsed at ``TTScheduler.__init__`` into one
# attribute (``None`` when unset), consulted in ``_prefer_decode`` and
# accounted in ``_finalize_scheduler_output``; a tree that adds a policy adds
# its name here in the same change (``TT_SCHED_PACK``, sched_pack: the
# admission of a listed layout of whole prompts in one prefill step). Any
# other ``TT_SCHED_*`` variable makes ``check_sched_flags`` raise, so a
# misspelt or unimplemented flag (``TT_SCHED_ADMISSION_GROUP`` is not
# implemented) does not serve the stock schedule under its name.
KNOWN_SCHED_FLAGS = frozenset(
    {
        "TT_SCHED_INTERLEAVE",
        "TT_SCHED_EVENT_LOG",
        "TT_SCHED_DEFER_FINAL",
        "TT_SCHED_RELEASE_FIRST",
        "TT_SCHED_FINAL_STEP",
        "TT_SCHED_PACK",
    }
)


def check_sched_flags(environ: Mapping[str, str] | None = None) -> None:
    """Raise ``ValueError`` on a ``TT_SCHED_*`` variable this tree does not know."""
    env = os.environ if environ is None else environ
    unknown = sorted(
        k for k in env if k.startswith("TT_SCHED_") and k not in KNOWN_SCHED_FLAGS
    )
    if unknown:
        raise ValueError(
            f"unknown TT scheduler flag(s) {', '.join(unknown)}; "
            f"this tree implements {', '.join(sorted(KNOWN_SCHED_FLAGS))}"
        )


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
        # see _schedule_prefill_only: the model types whose generator needs
        # every chunk of a prompt to start at a multiple of the token budget
        model_config = getattr(self.vllm_config, "model_config", None)
        model_type = getattr(
            getattr(model_config, "hf_config", None), "model_type", None
        )
        self._aligned_chunks = model_type in ALIGNED_CHUNK_MODEL_TYPES
        # The TT_SCHED_* flags this scheduler knows; any other one in the
        # environment is a typo or another tree's policy and is refused at
        # start-up rather than serving the stock schedule under its name.
        check_sched_flags()
        # TT_SCHED_INTERLEAVE (sched_interleave): decode bursts between
        # prefills in default mode; None unless the variable is set. Its
        # max_prompt=chunk is the token budget (one chunk: 5,120 served).
        scheduler_config = getattr(self.vllm_config, "scheduler_config", None)
        chunk = getattr(scheduler_config, "max_num_batched_tokens", None)
        self._interleave = InterleavePolicy.from_env(chunk_tokens=chunk)
        if self._interleave is not None:
            logger.info("TT scheduler interleaving: %s", self._interleave.describe())
        # TT_SCHED_DEFER_FINAL (sched_defer_final): every request's final
        # chunk deferred until the cohort's bulk prefills are done; None
        # unless the variable is set. The generator's TT_V41_DEFER_FINAL must
        # agree (it parks and restores the prefixes), or the start-up refuses.
        sched_defer_final.check_agreement()
        self._defer = DeferFinalPolicy.from_env(chunk_tokens=chunk, cohort=self.max_num_running_reqs)
        # TT_SCHED_FINAL_STEP (sched_final_step): the cohort's first tokens
        # from one decode step; needs the deferred finals and the generator's
        # parking off.
        sched_final_step.check_agreement(defer_enabled=self._defer is not None)
        if self._defer is not None:
            if self._interleave is not None:
                raise ValueError("TT_SCHED_DEFER_FINAL and TT_SCHED_INTERLEAVE do not compose")
            logger.info("TT scheduler deferred finals: %s", self._defer.describe())
        # TT_SCHED_RELEASE_FIRST (sched_release_first): a cohort's first
        # tokens released k at a time, or all at the drain's end; None unless
        # the variable is set, and refused without the deferred finals.
        self._release = ReleaseFirstPolicy.from_env(defer_enabled=self._defer is not None)
        if self._release is not None:
            logger.info("TT scheduler first-token release: %s", self._release.describe())
        # TT_SCHED_PACK (sched_pack): a prefill step with no partial prefill
        # admits the whole prompts of a listed layout at the head of the queue
        # (aligned chunks only: the generator packs them); None unless set.
        self._pack = PackPolicy.from_env(chunk_tokens=chunk)
        if self._pack is not None:
            assert self._aligned_chunks, (
                "TT_SCHED_PACK needs an aligned-chunk model (the V4.1 generator)"
            )
            logger.info("TT scheduler packing: %s", self._pack.describe())

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

    @property
    def _defer_policy(self):
        """The deferred-final policy, or None (a scheduler built without ``__init__``, as the unit tests do)."""
        return getattr(self, "_defer", None)

    @property
    def _release_policy(self):
        """The first-token release policy, or None (as above)."""
        return getattr(self, "_release", None)

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
            or (self._defer_policy is not None and (self._defer_policy.size() > 0 or self._defer_policy.draining))
        )

    @sched_events.guarded
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # NOTE: `throttle_prefills` accepted for interface compatibility with the base
        #        scheduler but unused - TT separates prefill/decode explicitly.
        if not sched_events.enabled():
            return self._schedule(throttle_prefills)
        result = self._schedule(throttle_prefills)
        self._log_schedule(result)
        return result

    def _log_schedule(self, result: SchedulerOutput) -> None:
        """One ``admit`` line per newly admitted request, one ``preempt`` line per preempted one, and one
        ``chunk`` line per prefill row (a new request or a partial prefill) with the chunk it was given
        (``TT_SCHED_EVENT_LOG``, see ``sched_events``)."""
        for new in result.scheduled_new_reqs:
            sched_events.emit("admit", id=new.req_id)
        for req_id in result.preempted_req_ids or ():
            sched_events.emit("preempt", id=req_id)
        for req_id, n in result.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None:
                continue
            # ``_update_after_schedule`` has already advanced the computed
            # count by this step's tokens: the chunk is [end - n, end). A
            # decode row starts at or past the prompt (one token, or 1 + K
            # with a speculative round's drafts); a prefill chunk starts
            # within it (a resumed preemption may replay past it, in more
            # than one token).
            end = request.num_computed_tokens
            if end - n >= request.num_prompt_tokens:
                continue  # a decode row
            sched_events.emit("chunk", id=req_id, start=end - n, end=end)

    def _schedule(self, throttle_prefills: bool) -> SchedulerOutput:
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
        # admitted and partial prefills advance, unless a policy asks for a
        # decode step first (TT_SCHED_INTERLEAVE: a bounded burst of the
        # running decodes between prefills).
        if has_pending_prefill:
            if self._prefer_decode(has_running_decode):
                result = self._schedule_decode_only()
                if result.total_num_scheduled_tokens > 0:
                    return self._finalize_scheduler_output(result)
                # nothing to decode after all (KV pressure): end the burst
                # and prefill as before
                self._finalize_scheduler_output(result)
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

    def _prefer_decode(self, has_running_decode: bool) -> bool:
        """Default mode only: run a decode step although prefill work is
        pending. ``has_running_decode`` false means there is nothing to
        decode and the answer is no whatever the policies say.

        The contract every scheduling policy keeps (``TT_SCHED_INTERLEAVE``
        here; ``TT_SCHED_PACK`` and any later one conform to it):

        - one clause per policy, in the order below (``TT_SCHED_DEFER_FINAL=hold``
          first: a decoding cohort admits nothing, ``sched_defer_final``); the
          first true clause wins; a policy whose variable is unset is ``None`` and adds nothing,
          so with every variable unset this returns ``False`` and
          ``_schedule`` is the stock prefill-first path;
        - a clause reads the scheduler's state (``self.running`` and its
          requests' ``is_prefill_chunk`` / ``num_prompt_tokens``,
          ``self.waiting``, ``self.requests``) and the policy's own, never the
          device, the clock of the step or another policy's state;
        - ``True`` means one ``_schedule_decode_only`` step (the waiting
          queues and partial prefills hidden, restored after); a step that
          schedules nothing falls through to the prefill step, so a clause
          can never stall prefill;
        - every scheduler output, from every path, reaches
          ``_finalize_scheduler_output``, where each policy's ``observe``
          accounts the scheduled work (rows, prompts completed) and never
          ``schedule()`` calls; a policy that packs several prompts into one
          prefill step completes several prompts in one output and the
          interleave count sees them all.
        """
        if not has_running_decode:
            return False
        defer = self._defer_policy
        if defer is not None:
            head = self.waiting.peek_request() if self.waiting else None
            if defer.hold_for_decoders(self.running, head):
                return True
        policy = self._interleave
        if policy is not None:
            ready = sum(1 for r in self.running if not r.is_prefill_chunk)
            partial = next((r for r in self.running if r.is_prefill_chunk), None)
            if partial is not None:
                next_prompt = partial.num_prompt_tokens
            elif self.waiting:
                next_prompt = self.waiting.peek_request().num_prompt_tokens
            else:
                next_prompt = None
            if policy.burst_due(ready, len(self.waiting), next_prompt):
                policy.start_burst()
                return True
        return False

    def _finalize_scheduler_output(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        # Count scheduled work, not schedule() calls: every path ends here.
        if self._interleave is not None:
            self._interleave.observe(scheduler_output, self.requests, len(self.waiting))
        if self._defer_policy is not None:
            self._defer_policy.observe(scheduler_output, self.requests)
            release = self._release_policy
            if release is not None:
                for req_id in scheduler_output.finished_req_ids or ():
                    release.drop(req_id)
                for req_id in scheduler_output.preempted_req_ids or ():
                    release.drop(req_id)
                for req_id, parked in self._defer_policy.last_finals:
                    if parked:  # a one-chunk prompt's prefill is its final: never held
                        release.expect(req_id, self._defer_policy.cohorts)
        if self._pack is not None:
            self._pack.observe(scheduler_output, self.requests, len(self.waiting))
        return scheduler_output

    def _pack_admit(self) -> int:
        """New requests a prefill step with no partial prefill admits under
        aligned chunks: one, or the prompts of the listed layout the FIFO head
        of the waiting queue forms (``TT_SCHED_PACK``)."""
        if self._pack is None or not self.waiting:
            return 1
        return self._pack.admit(self.waiting)

    def _schedule_prefill_only(self) -> SchedulerOutput:
        """Schedule prefill work: waiting requests and partial continuations.

        Temporarily hides the running *decode* requests so the base scheduler's
        running loop only advances partial prefills, and its waiting loop
        admits new ones.  Adjusts max_num_running_reqs so the waiting loop
        respects the true capacity with the decodes hidden.
        """
        if self._defer_policy is not None:
            return self._schedule_prefill_deferred()
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_max = self.max_num_running_reqs
        self.running = cast(list[Request], partial_prefills)
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        # Aligned chunks (``ALIGNED_CHUNK_MODEL_TYPES``, the V4.1 generator):
        # a partial prefill advances alone (the waiting queues hidden), and a
        # step with no partial prefill admits at most one new request (or the
        # whole prompts of a listed layout at the head of the queue under
        # ``TT_SCHED_PACK``: they fit the budget together and the generator
        # runs them as one packed chunk), so every chunk of a prompt starts at
        # a multiple of the token budget and no prompt is split at the
        # remainder another prompt left of it.
        aligned = self._aligned_chunks
        saved_waiting = saved_skipped = None
        if aligned:
            if partial_prefills:
                saved_waiting = self.waiting
                saved_skipped = getattr(self, "skipped_waiting", None)
                self.waiting = create_request_queue(self.policy)
                if saved_skipped is not None:
                    self.skipped_waiting = create_request_queue(self.policy)
            else:
                self.max_num_running_reqs = min(
                    self.max_num_running_reqs, self._pack_admit()
                )
        try:
            result = super().schedule()
        finally:
            if saved_waiting is not None:
                if self.waiting:
                    saved_waiting.prepend_requests(self.waiting)
                self.waiting = saved_waiting
                if saved_skipped is not None:
                    if self.skipped_waiting:
                        saved_skipped.prepend_requests(self.skipped_waiting)
                    self.skipped_waiting = saved_skipped
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
        return result

    def _schedule_prefill_deferred(self) -> SchedulerOutput:
        """``_schedule_prefill_only`` under ``TT_SCHED_DEFER_FINAL`` (sched_defer_final): the base scheduler sees,
        per step, exactly the request that should advance. A closed cohort drains its finals one a step in order
        (a held one-chunk prompt put back at the head of the queue and admitted alone; a parked partial un-hidden
        alone); else an unparked partial (a bulk in progress) advances alone; else, while the cohort is open, the
        head of the queue is admitted alone (a one-chunk prompt is held instead, and the step looks again); else
        the cohort closes (full, or a queue tail) and its first final runs. Parked partials and pure decodes are
        hidden from the base scheduler and counted in the capacity; the waiting queues are hidden except for the
        one admission."""
        policy = self._defer_policy
        assert policy is not None
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        saved_running = self.running
        saved_max = self.max_num_running_reqs
        saved_waiting = self.waiting
        saved_skipped = getattr(self, "skipped_waiting", None)
        hidden_queue = create_request_queue(self.policy)
        while True:
            parked_ids = policy.parked_ids
            partials = [r for r in saved_running if r.is_prefill_chunk]
            active = [r for r in partials if r.request_id not in parked_ids]
            visible: list[Request] = []
            admit: Request | None = None
            if policy.draining:
                head_id, held = policy.finals[0]
                if held is not None:
                    admit = held
                elif policy.final_step:
                    # every parked final of the cohort in one step: one token each (the last prompt token),
                    # which the runner routes to a decode step (sched_final_step)
                    ids = {rid for rid, h in policy.finals if h is None}
                    visible = [r for r in partials if r.request_id in ids]
                    if not visible:  # all gone (finished or preempted between steps): skip them
                        policy.finals = deque(x for x in policy.finals if x[1] is not None)
                        continue
                else:
                    visible = [r for r in partials if r.request_id == head_id]
                    if not visible:  # gone (finished or preempted between steps): skip it
                        policy.finals.popleft()
                        continue
                break
            if active:
                visible = active
                break
            occupied = policy.size() + len(pure_decodes)
            if saved_waiting and occupied < policy.cohort:
                head = saved_waiting.peek_request()
                if policy.one_chunk(head):
                    policy.hold(saved_waiting.pop_request())
                    continue
                admit = head
                break
            if policy.size() > 0:
                policy.close("full" if occupied >= policy.cohort else "tail")
                continue
            break  # nothing to prefill: an empty step (the caller falls back to decode)
        hidden = [r for r in saved_running if r not in visible]  # parked partials and pure decodes
        self.running = cast(list[Request], visible)
        self.max_num_running_reqs = len(visible) if visible else (1 if admit is not None else 0)
        saved_budget = getattr(self, "max_num_scheduled_tokens", None)
        budget = policy.budget(visible[0]) if len(visible) == 1 and not policy.draining else None
        if admit is not None and admit is not (saved_waiting.peek_request() if saved_waiting else None):
            hidden_queue.add_request(admit)  # a held one-chunk prompt: alone at the head
            self.waiting = hidden_queue
        elif admit is not None:
            self.waiting = saved_waiting  # the head of the real queue, admitted alone (max 1)
        else:
            self.waiting = hidden_queue
        if saved_skipped is not None:
            self.skipped_waiting = create_request_queue(self.policy)
        if budget is not None:
            self.max_num_scheduled_tokens = budget  # a stepped bulk's last chunk ends at n - 1
        try:
            result = super().schedule()
        finally:
            if self.waiting is not saved_waiting and self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            self.waiting = saved_waiting
            if saved_skipped is not None:
                if self.skipped_waiting:
                    saved_skipped.prepend_requests(self.skipped_waiting)
                self.skipped_waiting = saved_skipped
            self.running.extend(hidden)  # a visible request the base preempted stays out, as in the stock path
            self.max_num_running_reqs = saved_max
            if budget is not None:
                self.max_num_scheduled_tokens = saved_budget
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
            # A partial prefill in a decode batch would join its decode slot
            # with the prefix computed so far and answer on a truncated
            # context with no error (the generator joins ``pending`` rows at
            # their first decode step): it must never be scheduled here.
            hidden = {r.request_id for r in partial_prefills}
            leaked = hidden.intersection(result.num_scheduled_tokens)
            assert not leaked, f"partial prefill(s) in a decode step: {leaked}"
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

    # --- the event log (TT_SCHED_EVENT_LOG; sched_events) ------------------

    def add_request(self, request: Request) -> None:
        super().add_request(request)
        if sched_events.enabled():
            sched_events.emit(
                "arrive",
                id=request.request_id,
                prompt=request.num_prompt_tokens,
                max_tokens=request.max_tokens,
            )

    @sched_events.guarded
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict:
        if not sched_events.enabled():
            return self._intercept_outputs(super().update_from_output(scheduler_output, model_runner_output))
        before = [
            request
            for req_id in scheduler_output.num_scheduled_tokens
            if (request := self.requests.get(req_id)) is not None
            and request.num_output_tokens == 0
        ]
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for request in before:
            if request.num_output_tokens > 0:
                sched_events.emit("first_token", id=request.request_id)
        return self._intercept_outputs(outputs)

    def _intercept_outputs(self, outputs: dict) -> dict:
        """``TT_SCHED_RELEASE_FIRST``: the first tokens of a draining cohort held and released in groups."""
        release = self._release_policy
        if release is not None:
            defer = self._defer_policy
            before = release.released
            release.intercept(outputs, defer is not None and defer.draining)
            if sched_events.enabled() and release.released > before:
                sched_events.emit("release", n=release.releases[-1], groups=release.released)
        return outputs

    def _free_request(self, request: Request, *args, **kwargs):
        if sched_events.enabled():
            sched_events.emit(
                "finish",
                id=request.request_id,
                status=request.status.name,
                output=request.num_output_tokens,
            )
        return super()._free_request(request, *args, **kwargs)
