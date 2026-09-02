# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from types import SimpleNamespace

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode


def _running(is_prefill_chunk=False):
    """A stand-in for a running request, seen only through the fields TT reads."""
    return SimpleNamespace(is_prefill_chunk=is_prefill_chunk)


def _scheduler(*, running=(), waiting=0, skipped_waiting=0, mode):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    for _ in range(waiting):
        scheduler.waiting.add_request(
            SimpleNamespace(
                status=RequestStatus.WAITING,
                num_output_placeholders=0,
            )
        )
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    for _ in range(skipped_waiting):
        scheduler.skipped_waiting.add_request(object())
    scheduler.running = list(running)
    scheduler.max_num_running_reqs = 8
    scheduler._forced_mode = mode
    return scheduler


def test_forced_prefill_does_not_fallback_to_decode_per_lane(monkeypatch):
    scheduler = _scheduler(
        running=[_running()], waiting=1, mode=TTSchedulingMode.PREFILL_ONLY
    )

    monkeypatch.setattr(scheduler, "_schedule_prefill_only", SchedulerOutput.make_empty)

    def fail_local_decode_fallback():
        raise AssertionError("forced prefill must remain coordinated across lanes")

    monkeypatch.setattr(scheduler, "_schedule_decode_only", fail_local_decode_fallback)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 0


def test_forced_decode_hides_and_restores_skipped_waiting(monkeypatch):
    scheduler = _scheduler(
        running=[_running()], skipped_waiting=1, mode=TTSchedulingMode.DECODE_ONLY
    )

    saved_waiting = scheduler.waiting
    saved_skipped_waiting = scheduler.skipped_waiting
    skipped_request = scheduler.skipped_waiting.peek_request()
    visible_queues = []

    def fake_base_schedule(self, throttle_prefills=False):
        visible_queues.append((bool(self.waiting), bool(self.skipped_waiting)))
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base_schedule)

    scheduler.schedule()

    assert visible_queues == [(False, False)]
    assert scheduler.waiting is saved_waiting
    assert scheduler.skipped_waiting is saved_skipped_waiting
    assert scheduler.skipped_waiting.peek_request() is skipped_request


def test_running_continuation_alone_still_schedules_prefill(monkeypatch):
    # Nothing waiting; the only work is a partial prefill in `running`. Only a
    # prefill step can advance it, so DEFAULT mode must not pick decode.
    scheduler = _scheduler(
        running=[_running(is_prefill_chunk=True)], mode=TTSchedulingMode.DEFAULT
    )

    calls = []
    monkeypatch.setattr(
        scheduler,
        "_schedule_prefill_only",
        lambda: calls.append("prefill") or SchedulerOutput.make_empty(),
    )

    def fail_decode_fallback():
        raise AssertionError("a decode step cannot advance a partial prefill")

    monkeypatch.setattr(scheduler, "_schedule_decode_only", fail_decode_fallback)

    scheduler.schedule()

    assert calls == ["prefill"]


def test_prefill_only_hides_decodes_but_keeps_continuations(monkeypatch):
    continuation = _running(is_prefill_chunk=True)
    decode = _running()
    scheduler = _scheduler(
        running=[decode, continuation], waiting=1, mode=TTSchedulingMode.PREFILL_ONLY
    )
    seen = {}

    def fake_base_schedule(self, throttle_prefills=False):
        seen["running"] = list(self.running)
        seen["max_num_running_reqs"] = self.max_num_running_reqs
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base_schedule)

    scheduler.schedule()

    assert seen["running"] == [continuation]
    # One slot is held by the hidden decode, so the waiting loop sees 8 - 1.
    assert seen["max_num_running_reqs"] == 7
    # Restored: the hidden decodes are appended back after the base pass.
    assert scheduler.running == [continuation, decode]
    assert scheduler.max_num_running_reqs == 8


def test_decode_only_hides_continuations_and_restores_them(monkeypatch):
    continuation = _running(is_prefill_chunk=True)
    decode = _running()
    scheduler = _scheduler(
        running=[decode, continuation], mode=TTSchedulingMode.DECODE_ONLY
    )
    seen = {}

    def fake_base_schedule(self, throttle_prefills=False):
        seen["running"] = list(self.running)
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base_schedule)

    scheduler.schedule()

    assert seen["running"] == [decode]
    assert scheduler.running == [decode, continuation]
