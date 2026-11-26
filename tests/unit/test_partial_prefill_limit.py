# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``PartialPrefillLimit`` picks the multi-chunk prefills a prefill step holds
back under ``additional_config['tt']['max_concurrent_partial_prefills']``."""

from dataclasses import dataclass

import pytest

from vllm_tt_plugin.partial_prefill_limit import (
    SHORT_FIRST_MAX_STEPS,
    PartialPrefillLimit,
)


@dataclass(eq=False)
class _Prompt:
    """A request stand-in: a name, and whether its uncached remainder needs
    more than one chunk."""

    name: str
    multi_chunk: bool


def _long(name: str) -> _Prompt:
    return _Prompt(name, multi_chunk=True)


def _short(name: str) -> _Prompt:
    return _Prompt(name, multi_chunk=False)


def _held(
    limit: PartialPrefillLimit, running: list[_Prompt], waiting: list[_Prompt]
) -> tuple[list[str], list[str]]:
    held = limit.select(running, waiting, lambda prompt: prompt.multi_chunk)
    return [p.name for p in held.running], [p.name for p in held.waiting]


def test_the_oldest_multi_chunk_prefills_advance():
    # Running partial prefills come first, in running order, then waiting
    # requests in queue order.
    running = [_long("r1"), _long("r2"), _long("r3")]
    assert _held(PartialPrefillLimit(2), running, [_long("w1")]) == (["r3"], ["w1"])
    waiting = [_long("w1"), _long("w2")]
    assert _held(PartialPrefillLimit(2), [_long("r1")], waiting) == ([], ["w2"])


def test_single_chunk_requests_are_never_held():
    # The final chunk of r0 and the short prompt w1 advance beside r1, which
    # takes the one place; no waiting multi-chunk request has a place, so w1
    # does not go ahead of anything.
    running = [_short("r0"), _long("r1"), _long("r2")]
    held = _held(PartialPrefillLimit(1), running, [_short("w1"), _long("w2")])
    assert held == (["r2"], ["w2"])


def test_waiting_single_chunk_requests_go_first():
    limit = PartialPrefillLimit(1)
    assert _held(limit, [], [_long("w1"), _short("w2")]) == ([], ["w1"])
    # With no single-chunk request waiting, w1 takes its place.
    assert _held(limit, [], [_long("w1")]) == ([], [])


def test_single_chunk_requests_go_first_for_a_bounded_number_of_steps():
    limit = PartialPrefillLimit(1)
    waiting = [_long("w1"), _short("w2")]

    held = [_held(limit, [], waiting)[1] for _ in range(SHORT_FIRST_MAX_STEPS + 1)]

    assert held == [["w1"]] * SHORT_FIRST_MAX_STEPS + [[]]
    # The step that let w1 through starts the count again.
    assert _held(limit, [], waiting) == ([], ["w1"])


def test_a_step_without_the_choice_starts_the_count_again():
    limit = PartialPrefillLimit(1)
    both = [_long("w1"), _short("w2")]
    for _ in range(SHORT_FIRST_MAX_STEPS - 1):
        _held(limit, [], both)
    _held(limit, [], [_long("w1")])

    held = [_held(limit, [], both)[1] for _ in range(SHORT_FIRST_MAX_STEPS)]

    assert held == [["w1"]] * SHORT_FIRST_MAX_STEPS


def test_the_limit_is_at_least_one():
    with pytest.raises(ValueError, match="at least 1, got 0"):
        PartialPrefillLimit(0)
