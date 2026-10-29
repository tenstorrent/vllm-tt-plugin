# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Prompt packing admission: ``TT_SCHED_PACK=<layouts>``.

The DeepSeek-V4.1 generator (``TT_V41_PACK=1`` with ``TT_V41_PACK_LAYOUTS``)
runs several whole short prompts of a listed layout as one prefill chunk when
one ``prefill_forward`` hands it those prompts as its first rows. Under aligned chunks
the TT scheduler admits one
new prefill a step (``_schedule_prefill_only``); this policy raises that cap to
the length of the longest listed layout whose prompt lengths are a prefix of
the FIFO waiting queue, so the base scheduler admits exactly those whole
prompts (its token budget holds them: a layout fits the chunk) and the
generator packs them. The value is the generator's list, ``<k>x<length>`` or
``<l1>+<l2>+...`` items separated by commas (``4x1028,3x1028,2x1028,2x2052``):
the same list in both places, so no scheduler-generator contract mismatch
(a prefix the scheduler admits that the generator does not list runs row by
row, which is correct and only slower; a layout the generator lists that the
scheduler does not never forms).

What it never does: wait for arrivals to fill a pack (the longest layout the
queue's head satisfies *now*, else one request as before); pack behind a
partial prefill (that step advances the partial alone, as before); admit a
partial next request (the cap is on requests, not tokens, and a layout's
prompts are whole and within the budget); touch the decode steps. A prompt
whose length is in no layout is admitted alone, as before. Unset,
``TTScheduler`` never consults the policy and nothing changes. The policy
reads scheduler state only (the waiting queue's head and the requests'
``num_prompt_tokens``), as the ``_prefer_decode`` contract asks, and composes
with ``TT_SCHED_INTERLEAVE``, whose ``observe`` counts every prompt the packed
step completed.
"""

from __future__ import annotations

import os
from itertools import islice
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

ENV = "TT_SCHED_PACK"


def parse_layouts(spec: str) -> list[tuple[int, ...]]:
    """``spec`` -> the layouts as length tuples, longest (in prompts) first,
    then by total tokens."""
    layouts: list[tuple[int, ...]] = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        if "x" in item:
            k, n = item.split("x", 1)
            lengths = (int(n),) * int(k)
        else:
            lengths = tuple(int(x) for x in item.split("+"))
        if len(lengths) < 2 or any(n <= 0 for n in lengths):
            raise ValueError(
                f"{ENV} {item!r}: a layout packs two or more prompts of positive length"
            )
        if lengths not in layouts:
            layouts.append(lengths)
    if not layouts:
        raise ValueError(f"{ENV}: no layout in {spec!r}")
    layouts.sort(key=lambda s: (-len(s), -sum(s)))
    return layouts


class PackPolicy:
    def __init__(self, layouts: list[tuple[int, ...]], chunk_tokens: int | None = None):
        self.layouts = list(layouts)
        if chunk_tokens is not None:
            bad = [s for s in self.layouts if sum(s) > chunk_tokens]
            if bad:
                raise ValueError(
                    f"{ENV}: layouts {bad} exceed the token budget {chunk_tokens}"
                )
        self.kmax = max(len(s) for s in self.layouts)
        self.packs = 0  # packed admissions (two or more prompts in one step)
        self.packed_prompts = 0

    @classmethod
    def from_env(
        cls, environ=None, chunk_tokens: int | None = None
    ) -> PackPolicy | None:
        env = os.environ if environ is None else environ
        spec = env.get(ENV, "").strip()
        if not spec or spec == "0":
            return None
        return cls(parse_layouts(spec), chunk_tokens)

    def describe(self) -> str:
        return f"{ENV}: layouts {[list(s) for s in self.layouts]}"

    def match(self, lengths) -> tuple[int, ...] | None:
        """The first layout whose lengths are a prefix of ``lengths`` (the queue's
        head, in FIFO order)."""
        lengths = list(lengths)
        for layout in self.layouts:
            k = len(layout)
            if len(lengths) >= k and all(lengths[i] == layout[i] for i in range(k)):
                return layout
        return None

    def admit(self, waiting) -> int:
        """How many new requests a prefill step with no partial prefill admits:
        the length of the longest listed layout the FIFO head of ``waiting``
        forms, else 1."""
        head = [r.num_prompt_tokens for r in islice(iter(waiting), self.kmax)]
        layout = self.match(head)
        return 1 if layout is None else len(layout)

    def observe(
        self, scheduler_output: SchedulerOutput, requests, waiting: int
    ) -> None:
        """Count the packed admissions of ``scheduler_output`` (its new requests,
        two or more)."""
        new = len(getattr(scheduler_output, "scheduled_new_reqs", None) or ())
        if new >= 2:
            self.packs += 1
            self.packed_prompts += new
