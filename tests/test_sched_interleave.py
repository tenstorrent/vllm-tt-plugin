# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""TT_SCHED_INTERLEAVE: decode bursts between prefills (sched_interleave)."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

from vllm_tt_plugin.sched_interleave import InterleavePolicy, parse_spec
from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode


# --- the spec -----------------------------------------------------------------


def test_parse_spec():
    assert parse_spec("every=4") == {"every": 4}
    assert parse_spec("2") == {"burst": 2}
    spec = "every=2,burst=2,ready=8,max_prompt=5120,queue=4,split,work_ms=2500"
    assert parse_spec(spec) == {
        "every": 2,
        "burst": 2,
        "ready": 8,
        "max_prompt": 5120,
        "queue": 4,
        "split": True,
        "work_ms": 2500.0,
    }
    with pytest.raises(ValueError):
        parse_spec("bogus=1")


def test_from_env_unset_or_zero_is_none():
    assert InterleavePolicy.from_env({}) is None
    assert InterleavePolicy.from_env({"TT_SCHED_INTERLEAVE": "0"}) is None
    assert InterleavePolicy.from_env({"TT_SCHED_INTERLEAVE": ""}) is None
    policy = InterleavePolicy.from_env({"TT_SCHED_INTERLEAVE": "every=4"})
    assert policy is not None and policy.every == 4 and policy.burst == 1


def test_bad_values_rejected():
    with pytest.raises(ValueError):
        InterleavePolicy(every=0)
    with pytest.raises(ValueError):
        InterleavePolicy(burst=0)


# --- the policy's accounting ----------------------------------------------------


def _req(req_id, prompt, computed):
    return SimpleNamespace(
        request_id=req_id, num_prompt_tokens=prompt, num_computed_tokens=computed
    )


def _output(rows):
    """A SchedulerOutput stand-in: ``rows`` maps req_id -> scheduled tokens."""
    return SimpleNamespace(
        num_scheduled_tokens=dict(rows),
        total_num_scheduled_tokens=sum(rows.values()),
    )


def test_every_counts_completed_prompts_not_chunks():
    policy = InterleavePolicy(every=2)
    requests = {
        "a": _req("a", 1028, 1028),  # one-call prompt, complete
        "b": _req("b", 12000, 5120),  # first chunk of a chunked prompt
        "c": _req("c", 12000, 10240),  # second chunk
        "d": _req("d", 12000, 12000),  # final chunk
    }
    policy.observe(_output({"a": 1028}), requests, waiting=5)
    assert policy.completed == 1
    assert not policy.burst_due(ready_decoders=4, waiting=5, next_prompt_tokens=1028)
    policy.observe(_output({"b": 5120}), requests, waiting=4)
    policy.observe(_output({"c": 5120}), requests, waiting=4)
    assert policy.completed == 1, "an intermediate chunk completes nothing"
    assert not policy.burst_due(4, 4, 12000)
    policy.observe(_output({"d": 1760}), requests, waiting=4)
    assert policy.completed == 2
    assert policy.burst_due(4, 4, 1028)


def test_decode_step_resets_the_count_and_spends_credit():
    policy = InterleavePolicy(every=1, burst=2)
    requests = {"a": _req("a", 1028, 1028), "x": _req("x", 1028, 1040)}
    policy.observe(_output({"a": 1028}), requests, waiting=1)
    assert policy.burst_due(1, 1, 1028)
    policy.start_burst()
    assert policy.credit == 2 and policy.bursts == 1
    policy.observe(_output({"x": 1}), requests, waiting=1)  # a decode row
    assert policy.credit == 1 and policy.completed == 0
    assert policy.burst_due(1, 1, 1028), "the burst's second step is still owed"
    policy.observe(_output({"x": 1}), requests, waiting=1)
    assert policy.credit == 0
    assert not policy.burst_due(1, 1, 1028), "nothing completed since the burst"


def test_empty_step_ends_a_burst():
    policy = InterleavePolicy(every=1, burst=3)
    policy.observe(_output({"a": 1028}), {"a": _req("a", 1028, 1028)}, waiting=1)
    policy.start_burst()
    policy.observe(_output({}), {}, waiting=1)
    assert policy.credit == 0


def test_ready_and_max_prompt_gates():
    policy = InterleavePolicy(every=1, ready=8, max_prompt=5120)
    policy.observe(_output({"a": 1028}), {"a": _req("a", 1028, 1028)}, waiting=1)
    assert not policy.burst_due(7, 1, 1028), "too few ready decoders"
    assert policy.burst_due(8, 1, 1028)
    assert not policy.burst_due(8, 1, 12000), "the next prompt is chunked"
    assert policy.burst_due(8, 1, None), "unknown length: not gated"


def test_queue_gate_bursts_every_prompt_while_the_queue_is_short():
    policy = InterleavePolicy(every=4, queue=2)
    requests = {"a": _req("a", 1028, 1028)}
    policy.observe(_output({"a": 1028}), requests, waiting=10)
    assert not policy.burst_due(4, 10, 1028)
    assert policy.burst_due(4, 2, 1028)


def test_split_bursts_at_half_the_queue():
    policy = InterleavePolicy(split=True)
    requests = {k: _req(k, 1028, 1028) for k in "abcdefgh"}
    # the run starts with 7 waiting behind the first prefill: epoch 8
    policy.observe(_output({"a": 1028}), requests, waiting=7)
    assert policy.epoch == 8
    for k in "bc":
        policy.observe(_output({k: 1028}), requests, waiting=5)
        assert not policy.burst_due(3, 5, 1028)
    policy.observe(_output({"d": 1028}), requests, waiting=4)
    assert policy.burst_due(4, 4, 1028)


def test_work_ms_gate_counts_prefill_time_only(monkeypatch):
    import vllm_tt_plugin.sched_interleave as mod

    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    policy = InterleavePolicy(every=99, work_ms=1000.0)
    requests = {"a": _req("a", 12000, 5120), "x": _req("x", 1028, 1040)}
    policy.observe(_output({"x": 1}), requests, waiting=1)  # a decode step at t=100
    clock[0] = 150.0  # idle for 50 s
    policy.observe(_output({"a": 5120}), requests, waiting=1)  # a chunk at t=150
    assert not policy.burst_due(1, 1, 12000), "idle time is not prefill time"
    clock[0] = 150.4
    assert not policy.burst_due(1, 1, 12000)
    clock[0] = 151.1
    assert policy.burst_due(1, 1, 12000), "1.1 s of prefill has run"


# --- the scheduler's hook ---------------------------------------------------------


def _running(req_id="r", is_prefill_chunk=False, prompt=1028):
    return SimpleNamespace(
        request_id=req_id, is_prefill_chunk=is_prefill_chunk, num_prompt_tokens=prompt
    )


def _waiting(req_id="w", prompt=1028):
    return SimpleNamespace(request_id=req_id, num_prompt_tokens=prompt)


def _scheduled(rows):
    """A real SchedulerOutput with ``rows`` (req_id -> tokens) scheduled: a
    prefill mock must schedule something, or the KV-pressure fallback of the
    stock scheduler turns the step into a decode step."""
    out = SchedulerOutput.make_empty()
    out.num_scheduled_tokens = dict(rows)
    out.total_num_scheduled_tokens = sum(rows.values())
    return out


def _scheduler(*, running=(), waiting=(), policy=None):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    for request in waiting:
        scheduler.waiting.add_request(request)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.running = list(running)
    scheduler.max_num_running_reqs = 8
    scheduler._forced_mode = TTSchedulingMode.DEFAULT
    scheduler._interleave = policy
    scheduler.requests = {}
    return scheduler


def test_flag_unset_keeps_prefill_first(monkeypatch):
    scheduler = _scheduler(running=[_running()], waiting=[_waiting()], policy=None)
    calls = []
    monkeypatch.setattr(
        scheduler,
        "_schedule_prefill_only",
        lambda: calls.append("prefill") or _scheduled({"w": 1028}),
    )
    monkeypatch.setattr(
        scheduler,
        "_schedule_decode_only",
        lambda: calls.append("decode") or SchedulerOutput.make_empty(),
    )
    scheduler.schedule()
    assert calls == ["prefill"]


def test_burst_due_takes_a_decode_step_ahead_of_the_queue(monkeypatch):
    policy = InterleavePolicy(every=1)
    policy.completed = 1  # one prompt completed since the last decode step
    scheduler = _scheduler(running=[_running()], waiting=[_waiting()], policy=policy)
    calls = []

    monkeypatch.setattr(
        scheduler,
        "_schedule_decode_only",
        lambda: calls.append("decode") or _scheduled({"r": 1}),
    )
    monkeypatch.setattr(
        scheduler,
        "_schedule_prefill_only",
        lambda: calls.append("prefill") or _scheduled({"w": 1028}),
    )
    scheduler.requests = {"r": _req("r", 1028, 1040), "w": _req("w", 1028, 1028)}
    scheduler.schedule()
    assert calls == ["decode"]
    assert policy.completed == 0 and policy.credit == 0 and policy.bursts == 1
    # the next step is a prefill again: nothing completed since the burst
    scheduler.schedule()
    assert calls == ["decode", "prefill"]
    assert policy.completed == 1, "the prefill completed one prompt"


def test_burst_not_due_without_a_ready_decoder(monkeypatch):
    policy = InterleavePolicy(every=1)
    policy.completed = 5
    partial = _running("p", is_prefill_chunk=True, prompt=12000)
    scheduler = _scheduler(running=[partial], waiting=[_waiting()], policy=policy)
    calls = []
    monkeypatch.setattr(
        scheduler,
        "_schedule_prefill_only",
        lambda: calls.append("prefill") or _scheduled({"w": 1028}),
    )
    monkeypatch.setattr(
        scheduler,
        "_schedule_decode_only",
        lambda: calls.append("decode") or SchedulerOutput.make_empty(),
    )
    scheduler.schedule()
    assert calls == ["prefill"], "a partial prefill is not a decoder"


def test_forced_modes_ignore_the_policy(monkeypatch):
    policy = InterleavePolicy(every=1)
    policy.completed = 5
    scheduler = _scheduler(running=[_running()], waiting=[_waiting()], policy=policy)
    scheduler._forced_mode = TTSchedulingMode.PREFILL_ONLY
    calls = []
    monkeypatch.setattr(
        scheduler,
        "_schedule_prefill_only",
        lambda: calls.append("prefill") or _scheduled({"w": 1028}),
    )
    monkeypatch.setattr(
        scheduler,
        "_schedule_decode_only",
        lambda: calls.append("decode") or SchedulerOutput.make_empty(),
    )
    scheduler.schedule()
    assert calls == ["prefill"]


def test_empty_burst_falls_back_to_prefill(monkeypatch):
    policy = InterleavePolicy(every=1, burst=3)
    policy.completed = 1
    scheduler = _scheduler(running=[_running()], waiting=[_waiting()], policy=policy)
    calls = []
    monkeypatch.setattr(
        scheduler,
        "_schedule_decode_only",
        lambda: calls.append("decode") or SchedulerOutput.make_empty(),
    )
    monkeypatch.setattr(
        scheduler,
        "_schedule_prefill_only",
        lambda: calls.append("prefill") or _scheduled({"w": 1028}),
    )
    scheduler.schedule()
    assert calls == ["decode", "prefill"]
    assert policy.credit == 0, "a burst that scheduled nothing is over"


def test_max_prompt_reads_the_partial_before_the_queue(monkeypatch):
    policy = InterleavePolicy(every=1, max_prompt=5120)
    policy.completed = 1
    partial = _running("p", is_prefill_chunk=True, prompt=12000)
    scheduler = _scheduler(
        running=[_running(), partial], waiting=[_waiting(prompt=1028)], policy=policy
    )
    calls = []
    monkeypatch.setattr(
        scheduler,
        "_schedule_prefill_only",
        lambda: calls.append("prefill") or _scheduled({"w": 1028}),
    )
    monkeypatch.setattr(
        scheduler,
        "_schedule_decode_only",
        lambda: calls.append("decode") or SchedulerOutput.make_empty(),
    )
    scheduler.schedule()
    assert calls == ["prefill"], "the partial's 12,000 tokens gate the burst"


def test_decode_only_step_never_schedules_a_partial(monkeypatch):
    partial = _running("p", is_prefill_chunk=True, prompt=12000)
    scheduler = _scheduler(running=[_running(), partial])
    monkeypatch.setattr(
        AsyncScheduler,
        "schedule",
        lambda self, throttle_prefills=False: _scheduled({"p": 5120}),
    )
    with pytest.raises(AssertionError, match="partial prefill"):
        scheduler._schedule_decode_only()
    assert partial in scheduler.running, "restored even on the assertion"
