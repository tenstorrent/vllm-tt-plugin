# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode


def test_forced_decode_hides_and_restores_skipped_waiting(monkeypatch):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting.add_request(object())
    scheduler.running = [object()]
    scheduler._forced_mode = TTSchedulingMode.DECODE_ONLY

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
