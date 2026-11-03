# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Deferred final chunks: ``TT_SCHED_DEFER_FINAL=1``.

Under the stock prefill-first schedule a wave's requests prefill one after
another and decode together, so a request's window from its first token to its
last holds the other requests' prefill. This policy makes the
first tokens of a cohort come out back to back: every request is admitted one
at a time as today but its prefill stops one chunk short of the prompt's end
(the *bulk*; the generator joins the prefix into the request's decode slot
and *parks* it, ``TT_V41_DEFER_FINAL`` on the model side), a prompt of one
chunk is held off the queue at admission, and when the cohort closes (the
slots are full, or the queue is empty with no bulk in progress) the parked
requests run their final chunk one a step in admission order (the generator
*restores* each prefix from its slot first), each emitting its first token;
decode starts after the last. A request arriving while a cohort drains waits
for the next cohort; one arriving while a cohort decodes has its bulk run
prefill-first as today and is parked into the next cohort. With
``TT_SCHED_DEFER_FINAL=hold`` that last rule changes: while any
request of the cohort is still decoding, no new prefill is admitted (the
scheduler takes a decode step instead), so the replacements of the requests
that finished first wait for the whole cohort to finish and then form the next
cohort together. Under the stock rule a cohort whose members finish a few steps
apart (the speculative round's ragged acceptance) has each early finisher's
replacement bulk run prefill-first between the others' decode steps, and at
long prompts those bulks fill the stragglers' windows exactly as the serial
path's prefills did. ``hold`` leaves a one-chunk prompt at the
head of the queue on the stock path (admitted between decode steps, its prefill
its final), so at the points where every prompt is one chunk the
replacements' prefills land in the cohort's decode windows as before. With
``TT_SCHED_DEFER_FINAL=hold-all`` a one-chunk
prompt is held for the cohort as a bulk is: the replacements wait for the
cohort to finish and form the next one together, which then drains as the
packs ``TT_SCHED_PACK`` lists, and with ``TT_SCHED_RELEASE_FIRST=cohort`` their
first tokens leave together at the drain's end. The burst that opens a run (a
client's first wave arrives over a few hundred milliseconds) would otherwise
close a cohort of one on the empty queue and decode it alone while the rest
wait: under ``hold-all`` a cohort smaller than the slots whose
every decoder is *young* (``YOUNG`` generated tokens or fewer) is reopened by a
one-chunk arrival, which is held with the arrivals after it, closed and
drained while the young decoders pause; the engine reads no clock for it.

The policy keeps three queues: ``parked`` (request ids of multi-chunk prompts
whose bulk is done; they stay in ``running`` as partial prefills and are
hidden from the base scheduler), ``held`` (one-chunk prompts popped from the
waiting queue; prepended back when their turn comes) and ``finals`` (the
closed cohort's order: ``(req_id, held request or None)``). The scheduler's
``_schedule_prefill_only`` consults ``next_action`` and shows the base
scheduler exactly what should advance this step; ``observe`` in
``_finalize_scheduler_output`` moves a request into ``parked`` when the step
it saw brought its computed tokens to the last chunk boundary before its
prompt's end, pops the head of ``finals`` when its chunk completed the prompt,
and drops finished or preempted requests from every queue. Unset, the
scheduler never consults the policy and nothing changes.
"""

from __future__ import annotations

import os
from collections import deque
from typing import TYPE_CHECKING

from vllm_tt_plugin import sched_final_step

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

ENV = "TT_SCHED_DEFER_FINAL"
MODEL_ENV = "TT_V41_DEFER_FINAL"  # the generator's flag; both must agree


# "1": the cohort rule; "hold": plus no admission while a cohort decodes; "hold-all":
# plus one-chunk
# prompts held for the cohort too
VALUES = ("1", "hold", "hold-all")


def value(environ=None) -> str:
    env = os.environ if environ is None else environ
    v = env.get(ENV, "0").strip()
    if v not in ("0", "") and v not in VALUES:
        raise ValueError(f"{ENV}={v!r}: one of {VALUES}, or unset")
    return v


def enabled(environ=None) -> bool:
    return value(environ) in VALUES


def check_agreement(environ=None) -> None:
    """The scheduler's and the generator's flags must agree: a policy that parks
    requests the generator does not
    restore (or the reverse) serves wrong tokens or asserts in the middle of a wave, so
    refuse at start-up."""
    env = os.environ if environ is None else environ
    sched = enabled(env)
    model = env.get(MODEL_ENV, "0").strip() == "1"
    if sched and sched_final_step.enabled(env):
        # TT_SCHED_FINAL_STEP: the generator parks nothing (its own check
        # refuses the model flag).
        return
    if sched != model:
        raise ValueError(
            f"{ENV}={'1' if sched else 'unset'} but "
            f"{MODEL_ENV}={'1' if model else 'unset'}: set both or neither"
        )


class DeferFinalPolicy:
    # ``hold-all``: a decoding cohort smaller than the slots is reopened by a one-chunk
    # arrival while every one of
    # its decoders has generated this many tokens or fewer (a burst's first request,
    # prefilled alone, has had at
    # most a few decode steps by the time the rest of the burst is in the queue)
    YOUNG = 8

    def __init__(
        self,
        chunk_tokens: int,
        cohort: int,
        hold_admission: bool = False,
        final_step: bool = False,
        hold_one_chunk: bool = False,
    ) -> None:
        if chunk_tokens <= 0 or cohort <= 0:
            raise ValueError((chunk_tokens, cohort))
        if hold_one_chunk and not hold_admission:
            raise ValueError(
                "hold_one_chunk needs hold_admission (TT_SCHED_DEFER_FINAL=hold-all)"
            )
        self.chunk = chunk_tokens
        # The slots: parked + held + decoding requests close a cohort at this size.
        self.cohort = cohort
        # TT_SCHED_DEFER_FINAL=hold: no admission while a cohort decodes.
        self.hold_admission = hold_admission
        self.hold_one_chunk = (
            hold_one_chunk  # TT_SCHED_DEFER_FINAL=hold-all: one-chunk prompts held too
        )
        # TT_SCHED_FINAL_STEP (sched_final_step): a multi-chunk request's bulk runs to n
        # - 1 and the cohort's
        # parked requests take their first token from one decode step instead of a final
        # chunk each
        self.final_step = final_step
        self.parked: deque[str] = deque()
        self.held: deque[Request] = deque()
        self.finals: deque[tuple[str, Request | None]] = deque()
        self.cohorts = 0
        self.closed_on: list[str] = []
        self.last_finals: list[
            tuple[str, bool]
        ] = []  # the finals the last observe popped: (req_id, parked?)

    @classmethod
    def from_env(
        cls, environ=None, chunk_tokens: int | None = None, cohort: int | None = None
    ):
        if not enabled(environ):
            return None
        v = value(environ)
        return cls(
            chunk_tokens or 5120,
            cohort or 32,
            hold_admission=v in ("hold", "hold-all"),
            final_step=sched_final_step.enabled(environ),
            hold_one_chunk=v == "hold-all",
        )

    def describe(self) -> str:
        return (
            f"defer_final(chunk={self.chunk}, cohort={self.cohort}, "
            f"hold_admission={self.hold_admission}, "
            f"final_step={self.final_step}, hold_one_chunk={self.hold_one_chunk})"
        )

    def hold_for_decoders(self, running, waiting_head) -> bool:
        """``_prefer_decode``'s clause (``hold``): take a decode step instead of
        admitting or advancing a prefill
        while a cohort decodes, i.e. when no bulk is in progress (an unparked partial)
        and no cohort is draining
        its finals; the caller has established that a decode row is running. Not for a
        one-chunk prompt at the
        head of the queue (``waiting_head``, or None): its prefill is its final, so
        deferring it buys nothing and
        holding it costs the cohort's whole decode (the first of a burst of 32 closed a
        cohort of one and decoded alone for 5 s); the stock path admits it between
        decode steps as the serial
        path does."""
        if not self.hold_admission or self.draining:
            return False
        parked = self.parked_ids
        if any(r.is_prefill_chunk and r.request_id not in parked for r in running):
            return False  # a bulk in progress advances
        if waiting_head is not None and self.one_chunk(waiting_head):
            if not self.hold_one_chunk:
                return False  # the stock rule above
            if self.young_cohort(running):
                # ``hold-all``: the burst's stragglers reopen the young cohort
                # (held, closed, drained).
                return False
        return True

    def young_cohort(self, running) -> bool:
        """``hold-all``: whether the running decoders are a cohort smaller than the
        slots whose members have all
        generated ``YOUNG`` tokens or fewer (the first of a burst, prefilled alone and
        decoding while the rest of
        the burst arrived), so a one-chunk arrival reopens it instead of waiting for it
        to finish."""
        decoders = [r for r in running if not r.is_prefill_chunk]
        return 0 < len(decoders) < self.cohort and all(
            r.num_computed_tokens - r.num_prompt_tokens <= self.YOUNG for r in decoders
        )

    # --- the request's shape
    def prefix(self, request) -> int:
        """The bulk's extent: the last chunk boundary before the prompt's end (0 for a
        one-chunk prompt), or under
        ``final_step`` the prompt's end less one token (a one-chunk prompt: 0 still, the
        stock path)."""
        n = request.num_prompt_tokens
        if self.final_step and sched_final_step.stepped(n, self.chunk):
            return n - 1
        return (n - 1) // self.chunk * self.chunk

    def budget(self, request) -> int | None:
        """``final_step``: the token budget that ends a stepped request's last bulk
        chunk at n - 1 when that is
        less than a chunk away; None when the stock budget serves."""
        if not self.final_step:
            return None
        left = self.prefix(request) - request.num_computed_tokens
        return left if 0 < left < self.chunk else None

    def one_chunk(self, request) -> bool:
        return request.num_prompt_tokens <= self.chunk

    # --- the queues
    @property
    def parked_ids(self) -> set[str]:
        return set(self.parked)

    @property
    def draining(self) -> bool:
        return bool(self.finals)

    def size(self) -> int:
        """The cohort so far: every parked or held request (``hold`` puts the id in
        ``parked`` too, for the order;
        counting both closed a one-chunk cohort at half the slots)."""
        return len(self.parked)

    def hold(self, request) -> None:
        self.held.append(request)
        self.parked.append(request.request_id)  # the cohort's order holds both kinds

    def park(self, req_id: str) -> None:
        if req_id not in self.parked:
            self.parked.append(req_id)

    def close(self, why: str) -> None:
        held = {r.request_id: r for r in self.held}
        self.finals = deque((rid, held.get(rid)) for rid in self.parked)
        self.parked = deque()
        self.held = deque()
        self.cohorts += 1
        self.closed_on.append(why)

    def drop(self, req_id: str) -> None:
        if req_id in self.parked:
            self.parked.remove(req_id)
        self.held = deque(r for r in self.held if r.request_id != req_id)
        self.finals = deque(x for x in self.finals if x[0] != req_id)

    # --- the accounting
    def observe(self, output: SchedulerOutput, requests: dict) -> None:
        """Called on every scheduler output: park the rows whose chunk ended at their
        bulk's extent, pop a final
        once its chunk (or, under ``final_step``, its one-token step) completed the
        prompt, drop finished and
        preempted requests."""
        self.last_finals = []
        for req_id in output.finished_req_ids or ():
            self.drop(req_id)
        for req_id in output.preempted_req_ids or ():
            self.drop(req_id)
        completed: set[str] = set()
        for req_id, n in output.num_scheduled_tokens.items():
            request = requests.get(req_id)
            if request is None:
                continue
            end = request.num_computed_tokens  # advanced by this step's tokens already
            prompt = request.num_prompt_tokens
            if end - n >= prompt:
                continue  # a decode row
            if (
                self.finals
                and end >= prompt
                and any(x[0] == req_id for x in self.finals)
            ):
                completed.add(req_id)  # the final completed: its first token is out
            elif (
                end == self.prefix(request)
                and end < prompt
                and req_id not in self.parked
            ):
                self.park(req_id)
        if completed:
            self.last_finals = [
                (rid, held is None) for rid, held in self.finals if rid in completed
            ]
            self.finals = deque(x for x in self.finals if x[0] not in completed)
