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
prefill-first as today and is parked into the next cohort.

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

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

ENV = "TT_SCHED_DEFER_FINAL"
MODEL_ENV = "TT_V41_DEFER_FINAL"  # the generator's flag; both must agree


def enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(ENV, "0").strip() == "1"


def check_agreement(environ=None) -> None:
    """The scheduler's and the generator's flags must agree: a policy that parks requests the generator does not
    restore (or the reverse) serves wrong tokens or asserts in the middle of a wave, so refuse at start-up."""
    env = os.environ if environ is None else environ
    sched = env.get(ENV, "0").strip() == "1"
    model = env.get(MODEL_ENV, "0").strip() == "1"
    if sched != model:
        raise ValueError(f"{ENV}={'1' if sched else 'unset'} but {MODEL_ENV}={'1' if model else 'unset'}: set both or neither")


class DeferFinalPolicy:
    def __init__(self, chunk_tokens: int, cohort: int) -> None:
        if chunk_tokens <= 0 or cohort <= 0:
            raise ValueError((chunk_tokens, cohort))
        self.chunk = chunk_tokens
        self.cohort = cohort  # the slots: parked + held + decoding requests close a cohort when they reach it
        self.parked: deque[str] = deque()
        self.held: deque[Request] = deque()
        self.finals: deque[tuple[str, Request | None]] = deque()
        self.cohorts = 0
        self.closed_on: list[str] = []

    @classmethod
    def from_env(cls, environ=None, chunk_tokens: int | None = None, cohort: int | None = None):
        if not enabled(environ):
            return None
        return cls(chunk_tokens or 5120, cohort or 32)

    def describe(self) -> str:
        return f"defer_final(chunk={self.chunk}, cohort={self.cohort})"

    # --- the request's shape ------------------------------------------------------------------------------------
    def prefix(self, request) -> int:
        """The last chunk boundary before the prompt's end: the bulk's extent (0 for a one-chunk prompt)."""
        return (request.num_prompt_tokens - 1) // self.chunk * self.chunk

    def one_chunk(self, request) -> bool:
        return request.num_prompt_tokens <= self.chunk

    # --- the queues ---------------------------------------------------------------------------------------------
    @property
    def parked_ids(self) -> set[str]:
        return set(self.parked)

    @property
    def draining(self) -> bool:
        return bool(self.finals)

    def size(self) -> int:
        return len(self.parked) + len(self.held)

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

    # --- the accounting ---------------------------------------------------------------------------------------
    def observe(self, output: SchedulerOutput, requests: dict) -> None:
        """Called on every scheduler output: park the rows whose chunk ended at their prompt's last boundary, pop
        the head final once its chunk completed the prompt, drop finished and preempted requests."""
        for req_id in output.finished_req_ids or ():
            self.drop(req_id)
        for req_id in output.preempted_req_ids or ():
            self.drop(req_id)
        for req_id, n in output.num_scheduled_tokens.items():
            request = requests.get(req_id)
            if request is None:
                continue
            end = request.num_computed_tokens  # advanced by this step's tokens already
            prompt = request.num_prompt_tokens
            if end - n >= prompt:
                continue  # a decode row
            if self.finals and self.finals[0][0] == req_id and end >= prompt:
                self.finals.popleft()  # the final completed: its first token is out
            elif end == self.prefix(request) and end < prompt and req_id not in self.parked:
                self.park(req_id)
