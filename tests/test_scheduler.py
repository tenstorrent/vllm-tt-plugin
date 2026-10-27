# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest
from vllm.config import VllmConfig
from vllm.config.aux_output import AuxOutputConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from tests.test_block_scheduler import _request, _runner_output
from tests.test_block_scheduler import _scheduler as _real_scheduler
from vllm_tt_plugin.scheduler import (
    TTDecodeInterleavePolicy,
    TTScheduler,
    TTSchedulingMode,
)


def _running(is_prefill_chunk=False):
    """A stand-in for a running request, seen only through the fields TT reads."""
    return SimpleNamespace(is_prefill_chunk=is_prefill_chunk)


def _scheduler(*, running=(), waiting=0, kv_holding_waiting=0, mode):
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
    scheduler.kv_holding_waiting = create_request_queue(scheduler.policy)
    for _ in range(kv_holding_waiting):
        scheduler.kv_holding_waiting.add_request(object())
    scheduler.running = list(running)
    scheduler.max_num_running_reqs = 8
    scheduler.max_num_active_reqs = 8
    scheduler._forced_mode = mode
    # Production defaults; the decode-interleave cadence itself is covered in
    # tests/test_decode_interleave.py.
    scheduler._decode_interleave = TTDecodeInterleavePolicy(
        SimpleNamespace(additional_config={})
    )
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


@pytest.mark.parametrize(
    ("prefill_preemptions", "decode_preemptions"),
    [(None, None), ({"prefill"}, None), (None, {"decode"}), ({"prefill"}, {"decode"})],
)
def test_decode_fallback_preserves_cleanup_from_both_outputs(
    monkeypatch, prefill_preemptions, decode_preemptions
):
    scheduler = _scheduler(
        running=[_running()], waiting=1, mode=TTSchedulingMode.DEFAULT
    )
    prefill = SchedulerOutput.make_empty()
    prefill.finished_req_ids = {"finished-prefill", "shared"}
    prefill.preempted_req_ids = prefill_preemptions
    prefill.free_encoder_mm_hashes = ["image-prefill"]
    decode = SchedulerOutput.make_empty()
    decode.num_scheduled_tokens = {"running": 1}
    decode.total_num_scheduled_tokens = 1
    decode.finished_req_ids = {"finished-decode", "shared"}
    decode.preempted_req_ids = decode_preemptions
    decode.free_encoder_mm_hashes = ["image-decode"]
    monkeypatch.setattr(scheduler, "_schedule_prefill_only", lambda: prefill)
    monkeypatch.setattr(scheduler, "_schedule_decode_only", lambda: decode)

    result = scheduler.schedule()

    assert result.num_scheduled_tokens == {"running": 1}
    assert result.total_num_scheduled_tokens == 1
    assert result.finished_req_ids == {"finished-prefill", "finished-decode", "shared"}
    assert (result.preempted_req_ids or set()) == (prefill_preemptions or set()) | (
        decode_preemptions or set()
    )
    assert result.free_encoder_mm_hashes == ["image-prefill", "image-decode"]


def test_forced_decode_hides_and_restores_kv_holding_waiting(monkeypatch):
    scheduler = _scheduler(
        running=[_running()], kv_holding_waiting=1, mode=TTSchedulingMode.DECODE_ONLY
    )

    saved_waiting = scheduler.waiting
    saved_kv_holding_waiting = scheduler.kv_holding_waiting
    kv_holding_request = scheduler.kv_holding_waiting.peek_request()
    visible_queues = []

    def fake_base_schedule(self, throttle_prefills=False):
        visible_queues.append((bool(self.waiting), bool(self.kv_holding_waiting)))
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base_schedule)

    scheduler.schedule()

    assert visible_queues == [(False, False)]
    assert scheduler.waiting is saved_waiting
    assert scheduler.kv_holding_waiting is saved_kv_holding_waiting
    assert scheduler.kv_holding_waiting.peek_request() is kv_holding_request


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
        seen["max_num_active_reqs"] = self.max_num_active_reqs
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base_schedule)

    scheduler.schedule()

    assert seen["running"] == [continuation]
    # One slot is held by the hidden decode, so the waiting loop sees 8 - 1.
    assert seen["max_num_running_reqs"] == 7
    assert seen["max_num_active_reqs"] == 7
    # Restored: the hidden decodes are appended back after the base pass.
    assert scheduler.running == [continuation, decode]
    assert scheduler.max_num_running_reqs == 8
    assert scheduler.max_num_active_reqs == 8


def test_a_scheduler_that_needs_auxiliary_outputs_refuses_to_start(monkeypatch):
    # vLLM admits --enable-return-routed-experts for MoE models with prefix
    # caching, and its update_from_output then asserts an auxiliary output for
    # every request that emits tokens; the TT model runner produces none. The
    # local test model is dense, so skip the MoE admission check.
    monkeypatch.setattr(
        VllmConfig, "_verify_aux_output_compatibility", lambda self: None
    )
    monkeypatch.setattr(AuxOutputConfig, "enabled", property(lambda self: True))

    with pytest.raises(ValueError, match="auxiliary model outputs"):
        _real_scheduler(output_width=1)


def _real_scheduler_with_running_decode() -> TTScheduler:
    """A real two-slot scheduler whose request ``A`` has finished prefill."""
    scheduler = _real_scheduler(output_width=1, max_num_seqs=2)
    scheduler.add_request(_request(4, request_id="A"))
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _runner_output(prefill, [7]))
    return scheduler


def test_prefill_admission_counts_the_running_decodes_it_hides():
    # A running decode holds one of the two slots. The prefill pass hides it
    # from the base scheduler, so the base waiting loop must still stop after
    # one newcomer; admitting both overflows the runner's persistent batch.
    scheduler = _real_scheduler_with_running_decode()
    scheduler.add_request(_request(4, request_id="B"))
    scheduler.add_request(_request(4, request_id="C"))

    step = scheduler.schedule()

    assert set(step.num_scheduled_tokens) == {"B"}
    assert {request.request_id for request in scheduler.running} == {"A", "B"}
    assert [request.request_id for request in scheduler.waiting] == ["C"]


def _hold_in_kv_holding_waiting(scheduler: TTScheduler, request_id: str):
    """Queue a request where the base scheduler keeps one that holds KV."""
    request = _request(4, request_id=request_id)
    scheduler.add_request(request)
    scheduler.waiting.remove_request(request)
    scheduler.kv_holding_waiting.add_request(request)
    return request


def test_a_kv_holding_request_is_pending_prefill_work():
    # The base scheduler admits from kv_holding_waiting before waiting, so a
    # request there needs a prefill step even when waiting is empty.
    scheduler = _real_scheduler_with_running_decode()
    _hold_in_kv_holding_waiting(scheduler, "B")

    step = scheduler.schedule()

    assert set(step.num_scheduled_tokens) == {"B"}


def test_decode_step_does_not_admit_a_kv_holding_request():
    scheduler = _real_scheduler_with_running_decode()
    held = _hold_in_kv_holding_waiting(scheduler, "B")
    scheduler.set_forced_mode(TTSchedulingMode.DECODE_ONLY)

    step = scheduler.schedule()

    assert set(step.num_scheduled_tokens) == {"A"}
    assert list(scheduler.kv_holding_waiting) == [held]


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
