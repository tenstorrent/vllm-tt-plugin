# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""FCFS limit for multi-chunk prefills (max_concurrent_partial_prefills)."""

from types import SimpleNamespace

import pytest
import vllm  # noqa: F401  (resolve the platform plugin before importing ours)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

from vllm_tt_plugin.config import get_tt_max_concurrent_partial_prefills
from vllm_tt_plugin.scheduler import (
    TTDecodeInterleavePolicy,
    TTScheduler,
    TTSchedulingMode,
)

CHUNK = 4096


def _cfg(**tt):
    return SimpleNamespace(additional_config={"tt": dict(tt)} if tt else {})


class _Req:
    def __init__(self, rid, prompt, chunk=False):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self.num_tokens = prompt
        self.num_computed_tokens = 0
        self.is_prefill_chunk = chunk
        self.num_output_placeholders = 0
        self.mm_features = []
        self.has_encoder_inputs = False


def _scheduler(limit, waiting=(), running=(), cap=8, **tt):
    s = TTScheduler.__new__(TTScheduler)
    s.policy = SchedulingPolicy.FCFS
    s.waiting = create_request_queue(s.policy)
    for r in waiting:
        s.waiting.add_request(r)
    s.kv_holding_waiting = create_request_queue(s.policy)
    s.running = list(running)
    s.max_num_running_reqs = cap
    s.max_num_active_reqs = cap
    s._forced_mode = TTSchedulingMode.DEFAULT
    s._max_partial_prefills = limit
    s._prefill_chunk_cap_table = ()
    s._prefill_chunk_alignment = 0
    s.max_num_scheduled_tokens = 525312
    s.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=525312,
        max_num_scheduled_tokens=None,
        long_prefill_token_threshold=CHUNK,
    )
    for k, v in tt.items():
        setattr(s, k, v)
    return s


def _install_fake_base(monkeypatch, seen, preempt=None):
    """Fake base loop: every visible request gets one chunk, like vLLM here."""

    def fake_base(self, throttle_prefills=False):
        running_ids = [r.request_id for r in self.running]
        for r in list(self.running):
            r.num_computed_tokens = min(
                r.num_prompt_tokens, r.num_computed_tokens + CHUNK
            )
        admitted = []
        while self.waiting and len(self.running) < self.max_num_running_reqs:
            r = self.waiting.pop_request()
            r.num_computed_tokens = min(r.num_prompt_tokens, CHUNK)
            admitted.append(r.request_id)
            self.running.append(r)
        for r in self.running:
            r.is_prefill_chunk = r.num_computed_tokens < r.num_prompt_tokens
        seen.append(running_ids + admitted)
        out = SchedulerOutput.make_empty()
        out.total_num_scheduled_tokens = len(running_ids + admitted)
        return out

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base)


def test_config_default_and_validation():
    assert get_tt_max_concurrent_partial_prefills(_cfg()) == 0
    assert (
        get_tt_max_concurrent_partial_prefills(_cfg(max_concurrent_partial_prefills=2))
        == 2
    )
    for bad in (-1, 1.5, True, "1"):
        with pytest.raises(ValueError):
            get_tt_max_concurrent_partial_prefills(
                _cfg(max_concurrent_partial_prefills=bad)
            )


def test_limit_one_runs_longs_one_at_a_time_and_shorts_always(monkeypatch):
    longs = [_Req(f"L{i}", 3 * CHUNK + 100) for i in range(3)]  # 4 chunks each
    s = _scheduler(1, waiting=[longs[0], _Req("s0", 100), longs[1], longs[2]])
    seen = []
    _install_fake_base(monkeypatch, seen)
    for i in range(1, 9):
        s._schedule_prefill_only()
        if i == 1:
            s.waiting.add_request(_Req("s1", 200))  # a short arrives later
    longs_per_step = [[x for x in step if x.startswith("L")] for step in seen]
    # L0 has 4 chunks. While it needs more than one, it advances alone.
    assert longs_per_step[:3] == [["L0"], ["L0"], ["L0"]]
    # Its 4th chunk fits one chunk, so it frees the slot: L1 starts with it.
    assert longs_per_step[3] == ["L0", "L1"]
    # Afterwards L1 runs alone until its own last chunk, which lets L2 in.
    assert longs_per_step[4:7] == [["L1"], ["L1"], ["L1", "L2"]]
    # Shorts never wait: s0 in step 1, s1 (arrived after step 1) in step 2.
    assert "s0" in seen[0] and "s1" in seen[1]


def test_last_chunk_counts_as_fits():
    r = _Req("A", 2 * CHUNK + 10, chunk=True)
    r.num_computed_tokens = CHUNK + 20  # rest = CHUNK - 10 + ... fits one chunk
    other = _Req("B", 10 * CHUNK, chunk=True)
    other.num_computed_tokens = CHUNK
    s = _scheduler(1, running=[other, r])
    hidden, slots = s._hide_running_multi_chunk()
    assert hidden == [] and slots == 0
    assert [x.request_id for x in s.running] == ["B", "A"]


def test_limit_zero_is_unchanged(monkeypatch):
    longs = [_Req(f"L{i}", 5 * CHUNK) for i in range(3)]
    s = _scheduler(0, waiting=longs + [_Req("s", 10)])
    seen = []
    _install_fake_base(monkeypatch, seen)
    s._schedule_prefill_only()
    assert seen == [["L0", "L1", "L2", "s"]]


def test_fcfs_running_first_then_waiting(monkeypatch):
    run = []
    for i in range(3):
        r = _Req(f"R{i}", 10 * CHUNK, chunk=True)
        r.num_computed_tokens = CHUNK
        run.append(r)
    s = _scheduler(2, waiting=[_Req("W0", 9 * CHUNK), _Req("w", 50)], running=run)
    seen = []
    _install_fake_base(monkeypatch, seen)
    s._schedule_prefill_only()
    assert seen == [["R0", "R1", "w"]]
    # Hidden R2 is back in running order, after R0 R1; W0 stays queued.
    assert [r.request_id for r in s.running] == ["R0", "R1", "R2", "w"]
    assert [r.request_id for r in s.waiting] == ["W0"]


def test_waiting_takes_remaining_slots(monkeypatch):
    r = _Req("R0", 10 * CHUNK, chunk=True)
    r.num_computed_tokens = CHUNK
    s = _scheduler(
        2, waiting=[_Req("W0", 9 * CHUNK), _Req("W1", 9 * CHUNK)], running=[r]
    )
    seen = []
    _install_fake_base(monkeypatch, seen)
    s._schedule_prefill_only()
    assert seen == [["R0", "W0"]]
    assert [x.request_id for x in s.waiting] == ["W1"]


def test_hidden_running_keeps_state_and_is_not_preempted(monkeypatch):
    a = _Req("A", 10 * CHUNK, chunk=True)
    b = _Req("B", 10 * CHUNK, chunk=True)
    a.num_computed_tokens = b.num_computed_tokens = 3 * CHUNK
    s = _scheduler(1, running=[a, b])
    visible = []

    def fake_base(self, throttle_prefills=False):
        visible.append([r.request_id for r in self.running])
        # Emulate KV pressure: preempt the last visible running request.
        victim = self.running.pop()
        self.waiting.prepend_requests(_one(victim))
        return SchedulerOutput.make_empty()

    def _one(r):
        q = create_request_queue(SchedulingPolicy.FCFS)
        q.add_request(r)
        return q

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base)
    s._schedule_prefill_only()
    assert visible == [["A"]]
    assert b in s.running and b.num_computed_tokens == 3 * CHUNK


def test_interleave_policy_still_sees_hidden_partials():
    r1 = _Req("A", 10 * CHUNK, chunk=True)
    r2 = _Req("B", 10 * CHUNK, chunk=True)
    s = _scheduler(1, running=[r1, r2])
    s._hide_running_multi_chunk()
    s.running = [r1, r2]  # restored after the base call
    assert s._has_pending_prefill()
    policy = TTDecodeInterleavePolicy(_cfg(decode_interleave_prefill_steps=1))
    policy.record_step(is_decode=False, prefill_pending=True)
    assert policy.wants_decode_step(has_pending_prefill=True, has_running_decode=True)


def test_mixed_schedule_triggers_decode_step(monkeypatch):
    # End to end through _schedule_mode: with a pending hidden partial prefill
    # the interleave policy still picks a decode-only step.
    a = _Req("A", 10 * CHUNK, chunk=True)
    b = _Req("B", 10 * CHUNK, chunk=True)
    d = _Req("D", 100)
    s = _scheduler(1, running=[a, b, d])
    s._decode_interleave = TTDecodeInterleavePolicy(
        _cfg(decode_interleave_prefill_steps=1)
    )
    s._finishing_candidates = []
    s._pending_forced_reset_discard_counts = {}
    kinds = []
    monkeypatch.setattr(
        s,
        "_schedule_decode_only",
        lambda: (kinds.append("decode"), SchedulerOutput.make_empty())[1],
    )
    seen = []
    _install_fake_base(monkeypatch, seen)
    monkeypatch.setattr(s, "_finalize_scheduler_output", lambda out, is_decode: out)
    s._schedule_mode()
    kinds.append("prefill")
    s._schedule_mode()
    assert kinds == ["prefill", "decode"]


def test_with_long_prompt_alone_both_rules_apply(monkeypatch):
    longs = [_Req(f"L{i}", 5 * CHUNK) for i in range(3)]
    s = _scheduler(
        1,
        waiting=[longs[0], _Req("s", 100), longs[1], longs[2]],
        _long_prefill_threshold=CHUNK,
        _long_prefill_max_defer=2,
        _long_prefill_defer_steps=0,
        _long_prefill_hidden_ids=set(),
    )
    s._pending_prefill_tokens = lambda r, waiting=False: r.num_prompt_tokens
    seen = []
    _install_fake_base(monkeypatch, seen)
    s._schedule_prefill_only()
    s._schedule_prefill_only()
    # Step 1: the alone rule defers the longs, so only the short runs.
    # Step 2: only longs remain; the limit lets just L0 in.
    assert seen[0] == ["s"]
    assert seen[1] == ["L0"]
    assert [r.request_id for r in s.waiting] == ["L1", "L2"]


def test_waiting_cached_prefix_makes_a_long_prompt_fit(monkeypatch):
    a, b = _Req("A", 5 * CHUNK), _Req("B", 5 * CHUNK)
    s = _scheduler(1, waiting=[a, b])
    s._estimate_cached_prefix_tokens = lambda r: 4 * CHUNK + 1  # rest < one chunk
    seen = []
    _install_fake_base(monkeypatch, seen)
    s._schedule_prefill_only()
    assert seen == [["A", "B"]]
