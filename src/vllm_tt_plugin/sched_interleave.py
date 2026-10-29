# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Decode bursts between prefills: ``TT_SCHED_INTERLEAVE=<spec>``.

The TT scheduler is prefill-first: while anything waits or a prefill is
partial, every step is a prefill step, and the running decodes wait. Under a
closed-loop client (``c`` requests in flight, each replaced as it finishes)
that makes every cohort of ``c`` requests prefill in a row and finish together,
so the next cohort arrives together and waits the same row of prefills again. This
policy lets the scheduler
run a bounded burst of decode steps although prefill work is pending: the
running requests then finish at staggered times, their replacements arrive a
few at a time, and a request's time to first token is a few prefills instead
of a cohort of them (the simulated and scored variants).

The spec is ``k=v`` pairs separated by commas (a bare integer is ``burst``):

- ``every=N`` (default 4): a burst once ``N`` prompts' prefills have completed
  since the last decode step (a one-call prompt, or the final chunk of a
  chunked one), so a chunked prompt is never split by a burst;
- ``burst=K`` (default 1): decode steps per burst;
- ``ready=R`` (default 1): at least ``R`` running decode requests;
- ``max_prompt=T``: no burst while the next prefill work (the partial prefill,
  else the head of the waiting queue) is longer than ``T`` prompt tokens: the
  length-aware off switch. ``T`` equal to the token budget (``max_prompt=chunk``
  resolves to ``scheduler_config.max_num_batched_tokens``, 5,120 served) keeps
  the bursts to one-call prompts: a prompt that needs several chunks is never
  interrupted, and a run of such prompts (a long-context benchmark point) is
  scheduled exactly as with the variable unset;
- ``queue=Q``: a burst after every completed prompt while the waiting queue
  holds at most ``Q`` requests, ``every`` otherwise; ``split``: a burst once
  half the queue seen at the start of the prefill run has been prefilled;
- ``work_ms=W``: a burst once ``W`` ms have passed since the last decode step
  while prefill work ran (between the chunks of a long prompt too).

Unset, ``TTScheduler`` never consults the policy and nothing changes. The
burst is a ``_schedule_decode_only`` step: the waiting queues and any partial
prefill are hidden from the base scheduler for that step and restored after
it, as the KV-pressure fallback already does; a partial prefill keeps its
state in the generator (its ``pending`` entry and host-side pipeline) across
the step. Counting is of scheduled work seen in ``_finalize_scheduler_output``
(a step that scheduled nothing counts for nothing), never of ``schedule()``
calls.
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

ENV = "TT_SCHED_INTERLEAVE"
CHUNK = "chunk"  # max_prompt=chunk: the scheduler's token budget


def parse_spec(spec: str) -> dict[str, int | float | bool | str]:
    out: dict[str, int | float | bool | str] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        if "=" not in item:
            if item == "split":
                out["split"] = True
                continue
            out["burst"] = int(item)
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key == "work_ms":
            out[key] = float(value)
        elif key == "split":
            out[key] = value.strip().lower() in ("1", "true", "yes")
        elif key == "max_prompt" and value.strip() == CHUNK:
            out[key] = CHUNK
        elif key in ("every", "burst", "ready", "max_prompt", "queue"):
            out[key] = int(value)
        else:
            raise ValueError(f"{ENV}: unknown key {key!r} in {spec!r}")
    return out


class InterleavePolicy:
    """The policy's state and the two hooks ``TTScheduler`` calls."""

    def __init__(
        self,
        every: int = 4,
        burst: int = 1,
        ready: int = 1,
        max_prompt: int | None = None,
        queue: int | None = None,
        split: bool = False,
        work_ms: float | None = None,
    ) -> None:
        if every < 1 or burst < 1 or ready < 1:
            raise ValueError(f"{ENV}: every, burst and ready must be >= 1")
        self.every, self.burst, self.ready = every, burst, ready
        self.max_prompt, self.queue, self.split, self.work_ms = (
            max_prompt,
            queue,
            split,
            work_ms,
        )
        self.completed = 0  # prompts whose prefill completed since the last decode step
        self.credit = 0  # decode steps still owed in the current burst
        self.epoch = 0  # split: the queue seen when the prefill run began
        self.prefill_ms = 0.0  # work_ms: prefill time since the last decode step
        # the previous step's kind and the clock when it was scheduled
        self._last: tuple[str, float] | None = None
        self.bursts = 0  # bursts started (diagnostics)
        self.steps = 0  # decode steps taken inside bursts (diagnostics)

    @classmethod
    def from_env(
        cls,
        environ: dict[str, str] | None = None,
        chunk_tokens: int | None = None,
    ) -> InterleavePolicy | None:
        """The policy of ``TT_SCHED_INTERLEAVE``, or ``None`` when it is unset,
        empty or ``0``. ``chunk_tokens`` is what ``max_prompt=chunk`` resolves
        to (the scheduler's token budget); the alias without it is an error."""
        spec = (environ if environ is not None else os.environ).get(ENV)
        if spec is None or spec.strip() in ("", "0"):
            return None
        kwargs = parse_spec(spec)
        if kwargs.get("max_prompt") == CHUNK:
            if chunk_tokens is None:
                raise ValueError(f"{ENV}: max_prompt=chunk needs the token budget")
            kwargs["max_prompt"] = int(chunk_tokens)
        return cls(**kwargs)  # type: ignore[arg-type]

    def describe(self) -> str:
        """The parameters as one line (the scheduler logs it at start-up)."""
        gate = "none" if self.max_prompt is None else f"{self.max_prompt} prompt tokens"
        return (
            f"every {self.every} completed prefill(s), burst {self.burst}, "
            f"ready >= {self.ready}, max_prompt {gate}, queue {self.queue}, "
            f"split {self.split}, work_ms {self.work_ms}"
        )

    # --- the hooks ---------------------------------------------------------

    def burst_due(
        self,
        ready_decoders: int,
        waiting: int,
        next_prompt_tokens: int | None,
    ) -> bool:
        """Whether the scheduler should run a decode step now although prefill
        work is pending (``ready_decoders`` running decode requests,
        ``waiting`` requests queued, ``next_prompt_tokens`` the prompt length
        of the partial prefill or the head of the queue)."""
        if ready_decoders < self.ready:
            return False
        if self.credit > 0:
            return True  # a burst in progress
        if (
            self.max_prompt is not None
            and next_prompt_tokens is not None
            and next_prompt_tokens > self.max_prompt
        ):
            return False
        if self.work_ms is not None:
            ran = self.prefill_ms  # plus the step that ran since the last observe
            if self._last is not None and self._last[0] == "prefill":
                ran += (time.monotonic() - self._last[1]) * 1000
            if ran >= self.work_ms:
                return True
        if self.completed == 0:
            return False
        if self.split:
            return self.completed >= max(1, math.ceil(self.epoch / 2))
        n = 1 if self.queue is not None and waiting <= self.queue else self.every
        return self.completed >= n

    def observe(
        self,
        output: SchedulerOutput,
        requests: dict[str, Request],
        waiting: int,
    ) -> None:
        """Account the scheduled work of a step: a decode step ends the prefill
        run (and spends one unit of a burst's credit); a prefill step counts
        the prompts it completed."""
        now = time.monotonic()
        if self._last is not None and self._last[0] == "prefill":
            # the previous (prefill) step ran from its schedule until now
            self.prefill_ms += (now - self._last[1]) * 1000
        if output.total_num_scheduled_tokens == 0:
            self.credit = 0  # a burst that scheduled nothing is over
            self._last = None
            return
        decode_rows = 0
        completed = 0
        for req_id, n in output.num_scheduled_tokens.items():
            request = requests.get(req_id)
            if request is None:
                continue
            # ``_update_after_schedule`` has advanced the computed count by
            # this step's tokens: the row's chunk is [end - n, end). A decode
            # row starts at or past the prompt (one token, or 1 + K with a
            # speculative round's drafts); a prefill chunk starts inside it
            # and completes the prompt when it ends at or past it.
            end = request.num_computed_tokens
            if end - n >= request.num_prompt_tokens:
                decode_rows += 1
            elif end >= request.num_prompt_tokens:
                completed += 1
        if decode_rows:
            if self.credit > 0:
                self.credit -= 1
                self.steps += 1
            self.completed = 0
            self.prefill_ms = 0.0
            self._last = ("decode", now)
            return
        if self.completed == 0:
            self.epoch = waiting + len(output.num_scheduled_tokens)
        self.completed += completed
        self._last = ("prefill", now)

    def start_burst(self) -> None:
        """Called when the scheduler takes the burst's first step."""
        if self.credit == 0:
            self.credit = self.burst
            self.bursts += 1
