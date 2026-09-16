# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import queue
import threading
import time
from types import SimpleNamespace

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from vllm_tt_plugin.platform import (
    _drain_idle_admission_burst,
    _install_idle_admission_coalescing_patch,
    _uninstall_idle_admission_coalescing_patch,
)
from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode


def test_idle_admission_burst_drains_immediate_and_inflight_siblings():
    pending = queue.Queue()
    handled = []
    engine = SimpleNamespace(
        input_queue=pending,
        is_running=lambda: True,
        _handle_client_request=lambda *request: handled.append(request),
    )
    pending.put(("add", "first-sibling"))

    def send_tail():
        time.sleep(0.001)
        for index in range(30):
            pending.put(("add", f"tail-{index}"))

    producer = threading.Thread(target=send_tail)
    producer.start()
    drained = _drain_idle_admission_burst(engine, quiet_s=0.004, max_s=0.02)
    producer.join()

    assert drained == 31
    assert len(handled) == 31


def test_idle_admission_burst_stops_when_engine_stops():
    pending = queue.Queue()
    pending.put(("add", "must-not-run"))
    handled = []
    engine = SimpleNamespace(
        input_queue=pending,
        is_running=lambda: False,
        _handle_client_request=lambda *request: handled.append(request),
    )

    assert _drain_idle_admission_burst(engine, quiet_s=0.004, max_s=0.02) == 0
    assert handled == []
    assert pending.qsize() == 1


def test_idle_admission_burst_hard_deadline_bounds_sustained_arrivals():
    pending = queue.Queue()
    for index in range(100):
        pending.put(("add", index))
    handled = []

    def handle(*request):
        handled.append(request)
        time.sleep(0.001)

    engine = SimpleNamespace(
        input_queue=pending,
        is_running=lambda: True,
        _handle_client_request=handle,
    )
    started = time.monotonic()
    drained = _drain_idle_admission_burst(engine, quiet_s=0.004, max_s=0.005)
    elapsed = time.monotonic() - started

    assert 0 < drained < 100
    assert pending.qsize() == 100 - drained
    assert elapsed < 0.03


def test_idle_admission_burst_honors_shutdown_between_requests():
    pending = queue.Queue()
    pending.put(("add", "first"))
    pending.put(("add", "must-remain"))
    state = {"running": True}
    handled = []

    def handle(*request):
        handled.append(request)
        state["running"] = False

    engine = SimpleNamespace(
        input_queue=pending,
        is_running=lambda: state["running"],
        _handle_client_request=handle,
    )

    assert _drain_idle_admission_burst(engine, quiet_s=0.004, max_s=0.02) == 1
    assert handled == [("add", "first")]
    assert pending.qsize() == 1


def test_idle_admission_patch_targets_multiprocessing_cores_and_is_reversible():
    import vllm.v1.engine.core as engine_core

    _uninstall_idle_admission_coalescing_patch()
    owner = engine_core.EngineCoreProc
    dp = engine_core.DPEngineCoreProc
    original = owner._process_input_queue
    assert not hasattr(engine_core.EngineCore, "_process_input_queue")
    assert "_process_input_queue" in owner.__dict__
    assert "_process_input_queue" not in dp.__dict__

    try:
        _install_idle_admission_coalescing_patch()
        installed = owner._process_input_queue
        assert installed is not original
        # DP inherits the one wrapper; it must not receive a second wrapper.
        assert dp._process_input_queue is installed
        assert "_process_input_queue" not in dp.__dict__

        _install_idle_admission_coalescing_patch()
        assert owner._process_input_queue is installed
        assert dp._process_input_queue is installed
    finally:
        _uninstall_idle_admission_coalescing_patch()

    assert owner._process_input_queue is original
    assert dp._process_input_queue is original
    assert "_process_input_queue" not in dp.__dict__


def test_idle_admission_wrapper_calls_upstream_once_then_drains(monkeypatch):
    import vllm.v1.engine.core as engine_core

    _uninstall_idle_admission_coalescing_patch()
    calls = []

    def upstream(engine):
        calls.append("upstream")
        engine.work = True

    monkeypatch.setattr(engine_core.EngineCoreProc, "_process_input_queue", upstream)
    _install_idle_admission_coalescing_patch()
    pending = queue.Queue()
    pending.put(("add", "sibling"))
    handled = []
    engine = SimpleNamespace(
        work=False,
        has_work=lambda: engine.work,
        is_running=lambda: True,
        input_queue=pending,
        _handle_client_request=lambda *request: handled.append(request),
    )

    try:
        engine_core.DPEngineCoreProc._process_input_queue(engine)
    finally:
        _uninstall_idle_admission_coalescing_patch()

    assert calls == ["upstream"]
    assert handled == [("add", "sibling")]


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
