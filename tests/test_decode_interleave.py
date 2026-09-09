# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Unit tests for the decode-interleave scheduling policy.

TT executes a step as all-prefill or all-decode, so a prompt the base scheduler
splits into N chunks occupies N consecutive prefill steps and every running
decode waits for the whole run. ``TTDecodeInterleavePolicy`` bounds that run by
spending a step on decode once the bound is reached. These tests drive the
policy directly and drive ``TTScheduler.schedule`` over a scripted mix of one
chunked prefill plus running decodes.
"""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from vllm_tt_plugin.config import get_tt_decode_interleave_config
from vllm_tt_plugin.scheduler import (
    TTDecodeInterleavePolicy,
    TTScheduler,
    TTSchedulingMode,
)


def _config(**tt_keys):
    return SimpleNamespace(additional_config={"tt": dict(tt_keys)} if tt_keys else {})


def _policy(**tt_keys):
    return TTDecodeInterleavePolicy(_config(**tt_keys))


def _run_phases(policy, steps, *, has_pending_prefill=True, has_running_decode=True):
    """Return the phase ("P" or "D") the policy picks for ``steps`` steps.

    Stands in for a scheduler whose prefill work never runs out, which is the
    shape of a long multi-chunk prefill alongside running decodes.
    """
    phases = []
    for _ in range(steps):
        # With nothing pending the scheduler runs decode without consulting the
        # policy; ``wants_decode_step`` answers only whether to take a step away
        # from pending prefill work.
        is_decode = not has_pending_prefill or policy.wants_decode_step(
            has_pending_prefill=has_pending_prefill,
            has_running_decode=has_running_decode,
        )
        policy.record_step(is_decode=is_decode)
        phases.append("D" if is_decode else "P")
    return "".join(phases)


# --- Defaults and configuration -------------------------------------------


def test_policy_is_enabled_by_default():
    enabled, prefill_steps, decode_steps = get_tt_decode_interleave_config(_config())
    assert enabled is True
    assert prefill_steps >= 1
    assert decode_steps >= 1


def test_config_keys_override_the_defaults():
    assert get_tt_decode_interleave_config(
        _config(
            decode_interleave_enabled=False,
            decode_interleave_prefill_steps=7,
            decode_interleave_decode_steps=3,
        )
    ) == (False, 7, 3)


@pytest.mark.parametrize(
    "tt_keys",
    [
        {"decode_interleave_enabled": 1},
        {"decode_interleave_prefill_steps": 0},
        {"decode_interleave_prefill_steps": -1},
        {"decode_interleave_prefill_steps": True},
        {"decode_interleave_prefill_steps": 2.5},
        {"decode_interleave_decode_steps": 0},
        {"decode_interleave_decode_steps": "2"},
    ],
)
def test_invalid_config_values_raise(tt_keys):
    with pytest.raises(ValueError):
        get_tt_decode_interleave_config(_config(**tt_keys))


# --- Cadence ---------------------------------------------------------------


def test_disabled_policy_never_picks_decode():
    policy = _policy(decode_interleave_enabled=False)
    assert _run_phases(policy, 8) == "PPPPPPPP"


@pytest.mark.parametrize(
    ("prefill_steps", "decode_steps", "expected"),
    [
        (1, 1, "PDPDPDPD"),
        (2, 1, "PPDPPDPP"),
        (4, 1, "PPPPDPPP"),
        (1, 2, "PDDPDDPD"),
        (2, 3, "PPDDDPPD"),
    ],
)
def test_cadence_matches_the_configured_bounds(prefill_steps, decode_steps, expected):
    policy = _policy(
        decode_interleave_prefill_steps=prefill_steps,
        decode_interleave_decode_steps=decode_steps,
    )
    assert _run_phases(policy, len(expected)) == expected


def test_first_step_of_a_prefill_run_is_never_stolen():
    # A request arriving while decodes run must not wait: the counters start a
    # fresh prefill run, so the very next step prefills.
    policy = _policy(decode_interleave_prefill_steps=1)
    assert _run_phases(policy, 4, has_pending_prefill=False) == "DDDD"
    assert _run_phases(policy, 1)[0] == "P"


def test_no_interleave_without_a_request_a_decode_step_can_advance():
    # Only a partial-prefill continuation is running: a decode step samples
    # nothing for it, so stealing a prefill step would produce an empty step.
    policy = _policy(decode_interleave_prefill_steps=1)
    assert _run_phases(policy, 6, has_running_decode=False) == "PPPPPP"


def test_prefill_gets_a_step_within_the_bounded_window():
    # The two-sided guarantee: with prefill work always pending, no window of
    # prefill_steps + decode_steps consecutive steps is decode-only.
    prefill_steps, decode_steps = 2, 3
    policy = _policy(
        decode_interleave_prefill_steps=prefill_steps,
        decode_interleave_decode_steps=decode_steps,
    )
    phases = _run_phases(policy, 60)
    window = prefill_steps + decode_steps
    assert "D" * (decode_steps + 1) not in phases
    assert "P" * (prefill_steps + 1) not in phases
    for start in range(len(phases) - window + 1):
        assert "P" in phases[start : start + window]


def test_a_zero_token_decode_step_still_spends_its_allowance():
    # Upstream's running loop skips a decode request whose async placeholders
    # already reached max_tokens, so a decode-only step can schedule nothing.
    # The allowance must still be consumed, or the policy re-picks decode every
    # step and the engine livelocks on empty steps.
    policy = _policy(
        decode_interleave_prefill_steps=1, decode_interleave_decode_steps=1
    )
    policy.record_step(is_decode=False)
    assert policy.wants_decode_step(has_pending_prefill=True, has_running_decode=True)
    policy.record_step(is_decode=True)
    assert not policy.wants_decode_step(
        has_pending_prefill=True, has_running_decode=True
    )


# --- TTScheduler integration ----------------------------------------------


def _running(is_prefill_chunk=False):
    return SimpleNamespace(is_prefill_chunk=is_prefill_chunk)


def _scheduler(*, running=(), waiting=0, mode=TTSchedulingMode.DEFAULT, **tt_keys):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    for _ in range(waiting):
        scheduler.waiting.add_request(
            SimpleNamespace(status=RequestStatus.WAITING, num_output_placeholders=0)
        )
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.running = list(running)
    scheduler.max_num_running_reqs = 8
    scheduler._forced_mode = mode
    scheduler._decode_interleave = _policy(**tt_keys)
    return scheduler


def _record_phases(scheduler, monkeypatch, steps):
    """Drive ``schedule`` and return the phase each step chose.

    Both passes are stubbed to report one scheduled token, so the zero-progress
    fallback (prefill schedules nothing -> decode) stays out of the way and the
    recorded phases are the policy's own choices.
    """
    phases = []

    def one_token(kind):
        def stub():
            phases.append(kind)
            out = SchedulerOutput.make_empty()
            out.num_scheduled_tokens = {"r": 1}
            out.total_num_scheduled_tokens = 1
            return out

        return stub

    monkeypatch.setattr(scheduler, "_schedule_prefill_only", one_token("P"))
    monkeypatch.setattr(scheduler, "_schedule_decode_only", one_token("D"))
    for _ in range(steps):
        scheduler.schedule()
    return "".join(phases)


def test_long_chunked_prefill_yields_decode_steps_at_the_configured_cadence(
    monkeypatch,
):
    # One partial prefill continuation that a single step cannot finish, plus a
    # running decode: exactly the starvation shape a multi-chunk prompt creates.
    scheduler = _scheduler(
        running=[_running(is_prefill_chunk=True), _running()],
        decode_interleave_prefill_steps=2,
        decode_interleave_decode_steps=1,
    )

    assert _record_phases(scheduler, monkeypatch, 9) == "PPDPPDPPD"


def test_disabled_policy_keeps_every_step_on_prefill(monkeypatch):
    scheduler = _scheduler(
        running=[_running(is_prefill_chunk=True), _running()],
        decode_interleave_enabled=False,
    )

    assert _record_phases(scheduler, monkeypatch, 6) == "PPPPPP"


def test_partial_prefill_alone_is_never_interleaved(monkeypatch):
    # No genuine decode is running, so a decode step could only be empty.
    scheduler = _scheduler(
        running=[_running(is_prefill_chunk=True)],
        decode_interleave_prefill_steps=1,
    )

    assert _record_phases(scheduler, monkeypatch, 5) == "PPPPP"


def test_interleaved_decode_step_hides_prefill_work(monkeypatch):
    # The interleaved step must go through the decode-only path, which hides
    # both waiting queues and the partial prefills, so the base scheduler
    # cannot admit prefill work into a decode step.
    continuation = _running(is_prefill_chunk=True)
    decode = _running()
    scheduler = _scheduler(
        running=[continuation, decode],
        waiting=1,
        decode_interleave_prefill_steps=1,
    )
    seen = []

    def fake_base_schedule(self, throttle_prefills=False):
        seen.append((bool(self.waiting), list(self.running)))
        return SchedulerOutput.make_empty()

    def prefill_one_token():
        # Nonzero so the zero-progress fallback stays out of this test.
        out = SchedulerOutput.make_empty()
        out.num_scheduled_tokens = {"r": 1}
        out.total_num_scheduled_tokens = 1
        return out

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base_schedule)
    monkeypatch.setattr(scheduler, "_schedule_prefill_only", prefill_one_token)

    # Step 1 prefills (fresh run), step 2 is the interleaved decode step.
    scheduler.schedule()
    scheduler.schedule()

    assert seen == [(False, [decode])]
    assert scheduler.running == [decode, continuation]
    assert bool(scheduler.waiting)


def test_zero_progress_fallback_still_reaches_decode(monkeypatch):
    # KV pressure: the prefill pass schedules nothing. That fallback is
    # independent of the interleave policy and must keep working with the
    # policy disabled.
    scheduler = _scheduler(
        running=[_running()],
        waiting=1,
        decode_interleave_enabled=False,
    )
    calls = []
    monkeypatch.setattr(
        scheduler,
        "_schedule_prefill_only",
        lambda: calls.append("P") or SchedulerOutput.make_empty(),
    )
    monkeypatch.setattr(
        scheduler,
        "_schedule_decode_only",
        lambda: calls.append("D") or SchedulerOutput.make_empty(),
    )

    scheduler.schedule()

    assert calls == ["P", "D"]


def test_forced_mode_bypasses_the_policy(monkeypatch):
    # A lane-driven scheduler never reaches DEFAULT mode; the coordinator owns
    # the shared decision. Forced prefill must stay prefill however long the run.
    scheduler = _scheduler(
        running=[_running(is_prefill_chunk=True), _running()],
        mode=TTSchedulingMode.PREFILL_ONLY,
        decode_interleave_prefill_steps=1,
    )

    def fail_decode():
        raise AssertionError("forced prefill must remain coordinated across lanes")

    def prefill_one_token():
        out = SchedulerOutput.make_empty()
        out.num_scheduled_tokens = {"r": 1}
        out.total_num_scheduled_tokens = 1
        return out

    monkeypatch.setattr(scheduler, "_schedule_decode_only", fail_decode)
    monkeypatch.setattr(scheduler, "_schedule_prefill_only", prefill_one_token)

    for _ in range(5):
        scheduler.schedule()
