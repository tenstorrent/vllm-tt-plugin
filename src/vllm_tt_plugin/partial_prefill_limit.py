# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which multi-chunk prefills one TT prefill step advances.

A request is *multi-chunk* when the part of its prompt that is not cached
needs more than one prefill chunk. vLLM's ``Scheduler.schedule`` gives every
running partial prefill and every admitted waiting request a chunk in each
step, so several long prompts on one engine advance together and all reach
their first token late, while every running decode waits for each of those
steps (a TT step cannot mix prefill and decode rows).
``additional_config['tt']['max_concurrent_partial_prefills']`` sets a limit
``L``. ``PartialPrefillLimit.select`` then picks the requests a prefill step
holds back:

- at most ``L`` multi-chunk requests advance, oldest first: running partial
  prefills in ``running`` order, then waiting requests in queue order;
- a request that fits in one chunk is never held back;
- while a multi-chunk request that has a place under ``L`` waits next to
  waiting single-chunk requests, it is held back so that they go first, for
  at most ``SHORT_FIRST_MAX_STEPS`` consecutive prefill steps.

``TTScheduler._schedule_prefill_only`` decides which request is multi-chunk,
hides the selected requests from its base ``schedule()`` pass and restores
them before it returns. This module does not touch scheduler queues.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from vllm.v1.request import Request

# Consecutive prefill steps in which waiting single-chunk requests go ahead of
# a waiting multi-chunk request. A bound, so that a steady stream of short
# prompts cannot starve a long one.
SHORT_FIRST_MAX_STEPS = 8


@dataclass(frozen=True)
class HeldPrefills:
    """Requests one prefill step holds back.

    Attributes
    ----------
    running
        Running partial prefills, in ``running`` order.
    waiting
        Waiting requests, in queue order.
    """

    running: tuple[Request, ...] = ()
    waiting: tuple[Request, ...] = ()


class PartialPrefillLimit:
    """Selects the multi-chunk prefills each prefill step holds back.

    Parameters
    ----------
    limit
        Multi-chunk requests one prefill step advances, at least 1.

    Notes
    -----
    The owning ``TTScheduler`` calls ``select`` once per prefill pass. The
    instance counts the consecutive passes that held a waiting multi-chunk
    request back for single-chunk ones; any other pass resets that count.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"a partial prefill limit must be at least 1, got {limit}")
        self.limit = limit
        self._short_first_steps = 0

    def select(
        self,
        partial_prefills: Sequence[Request],
        waiting: Sequence[Request],
        is_multi_chunk: Callable[[Request], bool],
    ) -> HeldPrefills:
        """Pick the requests this prefill step holds back.

        Parameters
        ----------
        partial_prefills
            Running requests whose prompt is partly computed, in ``running``
            order.
        waiting
            The waiting queue, in queue order.
        is_multi_chunk
            Whether a request's uncached remainder needs more than one chunk.

        Returns
        -------
        HeldPrefills
            The requests to hide from this step's base ``schedule()`` pass.
        """
        slots = self.limit
        held_running = []
        for request in partial_prefills:
            if is_multi_chunk(request):
                if slots:
                    slots -= 1
                else:
                    held_running.append(request)
        held_waiting = []
        placed: list[Request] = []
        has_single_chunk = False
        for request in waiting:
            if not is_multi_chunk(request):
                has_single_chunk = True
            elif slots:
                slots -= 1
                placed.append(request)
            else:
                held_waiting.append(request)
        if (
            placed
            and has_single_chunk
            and self._short_first_steps < SHORT_FIRST_MAX_STEPS
        ):
            self._short_first_steps += 1
            held_waiting = [r for r in waiting if r in placed or r in held_waiting]
        else:
            self._short_first_steps = 0
        return HeldPrefills(tuple(held_running), tuple(held_waiting))
