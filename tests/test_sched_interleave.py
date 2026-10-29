# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""TT_SCHED_INTERLEAVE: decode bursts between prefills (sched_interleave)."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm_tt_plugin.sched_interleave import InterleavePolicy, parse_spec
from vllm_tt_plugin.scheduler import (
    KNOWN_SCHED_FLAGS,
    TTScheduler,
    TTSchedulingMode,
    check_sched_flags,
)

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


def test_max_prompt_chunk_is_the_token_budget():
    assert parse_spec("every=4,max_prompt=chunk") == {
        "every": 4,
        "max_prompt": "chunk",
    }
    env = {"TT_SCHED_INTERLEAVE": "every=4,max_prompt=chunk"}
    policy = InterleavePolicy.from_env(env, chunk_tokens=5120)
    assert policy is not None and policy.max_prompt == 5120
    assert "max_prompt 5120 prompt tokens" in policy.describe()
    with pytest.raises(ValueError, match="token budget"):
        InterleavePolicy.from_env(env)
    explicit = InterleavePolicy.from_env(
        {"TT_SCHED_INTERLEAVE": "every=4,max_prompt=5120"}
    )
    assert explicit is not None and explicit.max_prompt == 5120


# --- the TT_SCHED_* registry -----------------------------------------------------


def test_known_flags_pass_and_unknown_ones_raise():
    check_sched_flags({})
    check_sched_flags({"TT_SCHED_INTERLEAVE": "every=4", "TT_SCHED_EVENT_LOG": "/x"})
    check_sched_flags({"TT_SCHEDULER": "not ours", "TT_METAL_HOME": "/x"})  # no prefix
    assert {
        "TT_SCHED_INTERLEAVE",
        "TT_SCHED_EVENT_LOG",
        "TT_SCHED_DEFER_FINAL",
        "TT_SCHED_RELEASE_FIRST",
        "TT_SCHED_FINAL_STEP",
        "TT_SCHED_PACK",
    } == KNOWN_SCHED_FLAGS
    with pytest.raises(ValueError, match="TT_SCHED_ADMISSION_GROUP"):
        check_sched_flags(
            {"TT_SCHED_INTERLEAVE": "every=4", "TT_SCHED_ADMISSION_GROUP": "8"}
        )
    with pytest.raises(ValueError, match="TT_SCHED_INTERLAEVE"):
        check_sched_flags({"TT_SCHED_INTERLAEVE": "every=4"})
    check_sched_flags({"TT_SCHED_PACK": "4x1028"})  # sched_pack, this tree


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


def test_decode_rows_with_drafts_are_decode_rows():
    """A speculative round schedules 1 + K tokens on a decode row (the last
    token and K drafts): still a decode step, never a completed prompt."""
    policy = InterleavePolicy(every=1, burst=1)
    requests = {"a": _req("a", 1028, 1028), "x": _req("x", 1028, 1044)}
    policy.observe(_output({"a": 1028}), requests, waiting=1)
    assert policy.completed == 1
    policy.start_burst()
    policy.observe(_output({"x": 4}), requests, waiting=1)  # [1040, 1044): decode
    assert policy.completed == 0 and policy.credit == 0, "a decode step"
    policy.observe(_output({"x": 4}), requests, waiting=1)
    assert policy.completed == 0, "never a completed prompt"
    requests["r"] = _req("r", 1028, 1030)  # a resumed preemption replaying past it
    policy.observe(_output({"r": 1030}), requests, waiting=1)
    assert policy.completed == 1, "a prefill chunk that ends past the prompt"


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


# --- a closed loop of steps through the hook ---------------------------------------
#
# The stand-ins below drive ``TTScheduler.schedule`` through a sequence of steps
# the way the engine does, with ``_schedule_prefill_only`` and
# ``_schedule_decode_only`` replaced by mocks that keep the requests' computed
# counts: a prefill step takes the partial prefill, else the head of the queue,
# and advances it by one chunk (5,120 tokens); a decode step advances every
# ready decoder by one token. The trace is one letter a step: ``P`` or ``D``.

CHUNK = 5120


def _lreq(req_id, prompt, computed):
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
        is_prefill_chunk=computed < prompt,
    )


def _loop(scheduler, monkeypatch):
    def prefill():
        partial = next((r for r in scheduler.running if r.is_prefill_chunk), None)
        if partial is None:
            if not scheduler.waiting:
                return SchedulerOutput.make_empty()
            partial = scheduler.waiting.pop_request()
            scheduler.running.append(partial)
        n = min(CHUNK, partial.num_prompt_tokens - partial.num_computed_tokens)
        partial.num_computed_tokens += n
        partial.is_prefill_chunk = (
            partial.num_computed_tokens < partial.num_prompt_tokens
        )
        return _scheduled({partial.request_id: n})

    def decode():
        rows = {}
        for r in scheduler.running:
            if not r.is_prefill_chunk:
                r.num_computed_tokens += 1
                rows[r.request_id] = 1
        return _scheduled(rows) if rows else SchedulerOutput.make_empty()

    monkeypatch.setattr(scheduler, "_schedule_prefill_only", prefill)
    monkeypatch.setattr(scheduler, "_schedule_decode_only", decode)

    def trace(steps):
        out = []
        for _ in range(steps):
            if not scheduler._has_pending_prefill():
                break
            result = scheduler.schedule()
            kinds = set()
            for req_id, n in result.num_scheduled_tokens.items():
                request = scheduler.requests[req_id]
                decode = (
                    n == 1 and request.num_computed_tokens > request.num_prompt_tokens
                )
                kinds.add("D" if decode else "P")
            assert len(kinds) == 1, "a TT step is all-prefill or all-decode"
            out.append(kinds.pop())
        return "".join(out)

    return trace


def _closed_loop(prompts, *, decoders=1, policy):
    """A scheduler with ``decoders`` ready decode rows running and ``prompts``
    (a list of prompt lengths) waiting, in order."""
    running = [_lreq(f"d{i}", 1028, 1040) for i in range(decoders)]
    waiting = [_lreq(f"w{i}", p, 0) for i, p in enumerate(prompts)]
    scheduler = _scheduler(running=running, waiting=waiting, policy=policy)
    scheduler.requests = {r.request_id: r for r in running + waiting}
    return scheduler


def test_closed_loop_unset_is_prefill_first(monkeypatch):
    scheduler = _closed_loop([1028] * 8, policy=None)
    trace = _loop(scheduler, monkeypatch)
    assert trace(100) == "P" * 8, "no decode step while anything waits"


def test_closed_loop_every_4_bursts_once_per_four_prompts(monkeypatch):
    scheduler = _closed_loop([1028] * 8, policy=InterleavePolicy(every=4))
    trace = _loop(scheduler, monkeypatch)
    assert trace(100) == "PPPPDPPPP"
    assert scheduler._interleave.bursts == 1
    assert scheduler._interleave.completed == 4, (
        "a burst is owed at the next schedule()"
    )


def test_closed_loop_long_prompts_are_inert_under_max_prompt(monkeypatch):
    """The length-aware off switch: a run of prompts longer than one chunk (the
    8K to 1M c32 benchmark points) is scheduled exactly as with the flag
    unset, chunk after chunk with no burst, while decoders are ready."""
    long = [8196] * 8  # two chunks each
    baseline = _loop(_closed_loop(long, decoders=4, policy=None), monkeypatch)(100)
    policy = InterleavePolicy(every=4, max_prompt=CHUNK)
    scheduler = _closed_loop(long, decoders=4, policy=policy)
    assert _loop(scheduler, monkeypatch)(100) == baseline == "P" * 16
    assert policy.bursts == 0 and policy.completed == 8, "counted, never fired"


def test_closed_loop_without_max_prompt_long_prompts_do_burst(monkeypatch):
    scheduler = _closed_loop([8196] * 8, decoders=4, policy=InterleavePolicy(every=4))
    assert _loop(scheduler, monkeypatch)(100) == "P" * 8 + "D" + "P" * 8
    assert scheduler._interleave.bursts == 1, "every counts prompts, not chunks"


def test_closed_loop_mixed_lengths_under_max_prompt(monkeypatch):
    """Four short prompts complete, a long one is at the head: the gate holds
    the burst until the long prompt's chunks are done (it is never split),
    then the burst fires before the next short prompt."""
    policy = InterleavePolicy(every=4, max_prompt=CHUNK)
    scheduler = _closed_loop([1028] * 4 + [8196] + [1028] * 4, policy=policy)
    assert _loop(scheduler, monkeypatch)(100) == "PPPP" + "PP" + "D" + "PPPP"
    assert policy.bursts == 1


def test_closed_loop_short_chunk_prompts_still_burst_under_max_prompt(monkeypatch):
    """The served short points (1,028 / 2,052 / 4,100 tokens: one call each)
    keep the every=4 schedule with the switch on."""
    for prompt in (1028, 2052, 4100):
        policy = InterleavePolicy(every=4, max_prompt=CHUNK)
        scheduler = _closed_loop([prompt] * 8, policy=policy)
        assert _loop(scheduler, monkeypatch)(100) == "PPPPDPPPP", prompt
