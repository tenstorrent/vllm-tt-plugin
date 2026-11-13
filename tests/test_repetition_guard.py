# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for the single-token repetition guard."""

import random
from types import SimpleNamespace

from vllm_tt_plugin.model_runner import _guard_repetition
from vllm_tt_plugin.repetition_guard import MAX_IDENTICAL_TOKEN_RUN, RepetitionGuard

STOP = 106


def _publish(guard, req_id, steps, stop=STOP):
    """Feed token lists step by step as the runner publishes them."""
    out = []
    for tokens in steps:
        out.extend(guard.apply(req_id, list(tokens), stop))
    return out


def test_output_below_the_limit_is_unchanged():
    guard = RepetitionGuard()
    rng = random.Random(0)
    tokens = [rng.randrange(1, 200) for _ in range(5000)]
    assert _publish(guard, "r", [[t] for t in tokens]) == tokens


def test_a_run_of_exactly_the_limit_is_delivered_whole():
    guard = RepetitionGuard()
    seq = [7] * MAX_IDENTICAL_TOKEN_RUN + [8, 9]
    assert _publish(guard, "r", [[t] for t in seq]) == seq


def test_the_token_after_the_limit_is_replaced_by_the_stop_token():
    guard = RepetitionGuard()
    out = _publish(guard, "r", [[7]] * (MAX_IDENTICAL_TOKEN_RUN + 1))
    assert out == [7] * MAX_IDENTICAL_TOKEN_RUN + [STOP]


def test_a_block_of_tokens_is_cut_at_the_offending_token():
    # A speculative step commits several tokens at once.
    guard = RepetitionGuard()
    assert guard.apply("r", [5] * (MAX_IDENTICAL_TOKEN_RUN - 2), STOP) == [5] * (
        MAX_IDENTICAL_TOKEN_RUN - 2
    )
    assert guard.apply("r", [5, 5, 5, 5, 9], STOP) == [5, 5, STOP]


def test_the_run_is_counted_across_steps_and_resets_on_a_different_token():
    guard = RepetitionGuard()
    steps = [[3] * 600, [4], [3] * 600, [3] * 424, [3]]
    out = _publish(guard, "r", steps)
    # 600 x3, 4, then 1024 x3 delivered whole, the 1025th replaced
    assert out == [3] * 600 + [4] + [3] * 1024 + [STOP]


def test_requests_are_counted_independently_and_drop_forgets():
    guard = RepetitionGuard()
    guard.apply("a", [1] * 1000, STOP)
    assert guard.apply("b", [1] * 1000, STOP) == [1] * 1000
    guard.drop("a")
    assert guard.apply("a", [1] * 1000, STOP) == [1] * 1000


def test_without_a_stop_token_nothing_is_counted_or_changed():
    guard = RepetitionGuard()
    assert guard.apply("r", [2] * 5000, None) == [2] * 5000


def test_legitimate_patterns_never_reach_the_limit():
    # The longest run in 25K real generations is 62 tokens. Whitespace padding,
    # dashes, table borders and base64 are short runs or alternate between tokens.
    guard = RepetitionGuard()
    seq = [10] * 62 + [11, 12] * 3000 + [13] * 900 + [14, 15, 16] * 2000 + [17] * 1000
    assert _publish(guard, "r", [seq[i : i + 7] for i in range(0, len(seq), 7)]) == seq


def _runner(ignore_eos=False, stop_ids=(1, STOP)):
    params = SimpleNamespace(ignore_eos=ignore_eos, all_stop_token_ids=set(stop_ids))
    runner = SimpleNamespace(
        requests={"r": SimpleNamespace(sampling_params=params)},
        _repetition_guard=RepetitionGuard(),
    )
    return runner


def _runner_steps(runner, n):
    out = []
    for _ in range(n):
        out.extend(_guard_repetition(runner, ["r"], [[7]])[0])
    return out


def test_runner_publishes_the_request_stop_id_after_the_limit():
    runner = _runner()
    out = _runner_steps(runner, MAX_IDENTICAL_TOKEN_RUN + 1)
    assert out == [7] * MAX_IDENTICAL_TOKEN_RUN + [1]  # min of the request's stop ids


def test_runner_never_touches_an_ignore_eos_request():
    runner = _runner(ignore_eos=True)
    assert _runner_steps(runner, MAX_IDENTICAL_TOKEN_RUN + 50) == [7] * (
        MAX_IDENTICAL_TOKEN_RUN + 50
    )


def test_runner_skips_requests_without_stop_ids_and_empty_rows():
    runner = _runner(stop_ids=())
    assert _runner_steps(runner, MAX_IDENTICAL_TOKEN_RUN + 5) == [7] * (
        MAX_IDENTICAL_TOKEN_RUN + 5
    )
    runner = _runner()
    lists = [[], [3]]
    assert _guard_repetition(runner, ["gone", "r"], lists) is lists
