# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Release groups for a cohort's first tokens: ``TT_SCHED_RELEASE_FIRST``.

Under ``TT_SCHED_DEFER_FINAL`` (``sched_defer_final``) a cohort's final chunks run one a
step after its bulk
prefills, each yielding its request's first token, and no decode step runs until the
last final is done. A
request's first token therefore leaves the engine before its second by up to the
remaining finals' duration, and AIPerf's per-user decode rate, whose clock
starts at the first token, holds
that wait. This policy withholds the ``EngineCoreOutput`` carrying a parked
(multi-chunk) request's first token
until ``k`` of the cohort's finals have produced theirs, or the cohort's drain is over,
and then returns the held
outputs together in one step's outputs, in the order they were produced. ``k = cohort``
(the value ``cohort``)
aligns every first token of the cohort at its last final; a smaller ``k`` trades a part
of the alignment for a
smaller rise of the mean TTFT. A held one-chunk prompt's prefill
is its final and its output is never held, unless ``TT_SCHED_DEFER_FINAL=hold-all`` held
it for the cohort
(then every first token of the cohort, packed or single, leaves at the drain's end); a
held output whose request finished is released at once; a request
dropped while held (aborted, preempted) has its output released with the next release.
Unset, the scheduler never
consults the policy and nothing changes.

The scheduler is an ``AsyncScheduler``: ``schedule()`` for step N+1 runs before
``update_from_output`` for step N,
so the defer policy's ``observe`` (schedule time) tells this one which finals were
scheduled (``expect``), and
``intercept`` (output time) matches their first tokens as they arrive. The release at
the drain's end happens in
the ``update_from_output`` where the last expected token arrives, before any decode
output of the cohort can
exist (``hold`` admits nothing and schedules no decode until the drain ends).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs

ENV = "TT_SCHED_RELEASE_FIRST"
COHORT = "cohort"


def value(environ=None) -> str:
    env = os.environ if environ is None else environ
    v = env.get(ENV, "").strip()
    if v in ("", "0"):
        return ""
    if v != COHORT and not (v.isdigit() and int(v) > 1):
        raise ValueError(
            f"{ENV}={v!r}: an integer above 1 (finals released together), "
            f"{COHORT!r}, or unset"
        )
    return v


def enabled(environ=None) -> bool:
    return value(environ) != ""


class ReleaseFirstPolicy:
    def __init__(self, group: int | None) -> None:
        """``group`` finals released together; None: at the cohort's drain's end."""
        if group is not None and group < 2:
            raise ValueError(group)
        self.group = group
        # req_id -> cohort number: a final scheduled, its first token not yet seen.
        self.expected: dict[str, int] = {}
        self.held: list[
            tuple[int, EngineCoreOutput]
        ] = []  # (client index, output) in arrival order
        self.released = 0  # groups released
        self.releases: list[int] = []  # the size of each release, for the record

    @classmethod
    def from_env(cls, environ=None, defer_enabled: bool = False):
        v = value(environ)
        if not v:
            return None
        if not defer_enabled:
            raise ValueError(
                f"{ENV} needs TT_SCHED_DEFER_FINAL: "
                "without deferred finals there is no cohort to align"
            )
        return cls(None if v == COHORT else int(v))

    def describe(self) -> str:
        return f"release_first(group={'cohort' if self.group is None else self.group})"

    # --- schedule time
    def expect(self, req_id: str, cohort: int) -> None:
        """A parked request's final was scheduled: its first token is withheld when it
        arrives."""
        self.expected[req_id] = cohort

    def drop(self, req_id: str) -> None:
        self.expected.pop(req_id, None)

    # --- output time
    def intercept(self, outputs: dict[int, EngineCoreOutputs], draining: bool) -> None:
        """Pull the expected first tokens out of this step's ``outputs`` (in place) and
        put the held ones back when
        a group is complete (``group`` held) or the drain is over (no final left to
        schedule, none expected)."""
        for client, eco in outputs.items():
            keep = []
            for out in eco.outputs:
                if (
                    out.request_id in self.expected
                    and out.new_token_ids
                    and not out.finished
                ):
                    del self.expected[out.request_id]
                    self.held.append((client, out))
                else:
                    if out.request_id in self.expected and out.finished:
                        del self.expected[out.request_id]
                    keep.append(out)
            eco.outputs = keep
        due = self.held and (
            (self.group is not None and len(self.held) >= self.group)
            or (not draining and not self.expected)
        )
        if due:
            self.release(outputs)

    def release(self, outputs: dict[int, EngineCoreOutputs]) -> None:
        from vllm.v1.engine import EngineCoreOutputs

        for client, out in self.held:
            eco = outputs.get(client)
            if eco is None:
                outputs[client] = eco = EngineCoreOutputs()
            eco.outputs.append(out)
        self.releases.append(len(self.held))
        self.released += 1
        self.held = []
