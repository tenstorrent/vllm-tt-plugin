# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""TT_SCHED_DEFER_FINAL: deferred final chunks (sched_defer_final), on a stub of the
base scheduler's loop."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

from vllm_tt_plugin.sched_defer_final import DeferFinalPolicy, check_agreement
from vllm_tt_plugin.scheduler import (
    KNOWN_SCHED_FLAGS,
    TTScheduler,
    TTSchedulingMode,
    check_sched_flags,
)

CHUNK = 5120


def test_flags_and_agreement():
    assert "TT_SCHED_DEFER_FINAL" in KNOWN_SCHED_FLAGS
    check_sched_flags({"TT_SCHED_DEFER_FINAL": "1"})
    assert DeferFinalPolicy.from_env({}) is None
    assert DeferFinalPolicy.from_env({"TT_SCHED_DEFER_FINAL": "0"}) is None
    policy = DeferFinalPolicy.from_env(
        {"TT_SCHED_DEFER_FINAL": "1"}, chunk_tokens=CHUNK, cohort=32
    )
    assert (
        policy is not None
        and policy.chunk == CHUNK
        and policy.cohort == 32
        and not policy.hold_admission
    )
    policy = DeferFinalPolicy.from_env(
        {"TT_SCHED_DEFER_FINAL": "hold"}, chunk_tokens=CHUNK, cohort=32
    )
    assert (
        policy is not None
        and policy.hold_admission
        and "hold_admission=True" in policy.describe()
    )
    assert not policy.hold_one_chunk
    policy = DeferFinalPolicy.from_env(
        {"TT_SCHED_DEFER_FINAL": "hold-all"}, chunk_tokens=CHUNK, cohort=32
    )
    assert policy is not None and policy.hold_admission and policy.hold_one_chunk
    assert "hold_one_chunk=True" in policy.describe()
    with pytest.raises(ValueError, match="hold_one_chunk"):
        DeferFinalPolicy(CHUNK, 32, hold_admission=False, hold_one_chunk=True)
    with pytest.raises(ValueError, match="TT_SCHED_DEFER_FINAL"):
        DeferFinalPolicy.from_env({"TT_SCHED_DEFER_FINAL": "yes"})
    check_agreement({"TT_SCHED_DEFER_FINAL": "hold", "TT_V41_DEFER_FINAL": "1"})
    check_agreement({})
    check_agreement({"TT_SCHED_DEFER_FINAL": "1", "TT_V41_DEFER_FINAL": "1"})
    with pytest.raises(ValueError, match="TT_V41_DEFER_FINAL"):
        check_agreement({"TT_SCHED_DEFER_FINAL": "1"})
    with pytest.raises(ValueError, match="TT_SCHED_DEFER_FINAL"):
        check_agreement({"TT_V41_DEFER_FINAL": "1"})


def _req(req_id, prompt, computed=0):
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
        is_prefill_chunk=computed < prompt,
        status="WAITING",
    )


def _scheduled(rows):
    out = SchedulerOutput.make_empty()
    out.num_scheduled_tokens = dict(rows)
    out.total_num_scheduled_tokens = sum(rows.values())
    return out


def _scheduler(
    prompts,
    *,
    decoders=0,
    cohort=32,
    policy=True,
    hold=False,
    hold_all=False,
    generated=12,
):
    """A scheduler with ``decoders`` decode rows running (each ``generated`` tokens into
    its output) and
    ``prompts`` waiting; the base scheduler's ``schedule`` stubbed as the stock loop
    sees it: admit from
    ``waiting`` while ``len(running) < max_num_running_reqs`` (one chunk of the new
    request), else advance the
    first partial in ``running`` by one chunk, else decode every running decode row."""
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    running = [_req(f"d{i}", 1028, 1028 + generated) for i in range(decoders)]
    waiting = [_req(f"w{i}", p) for i, p in enumerate(prompts)]
    for request in waiting:
        scheduler.waiting.add_request(request)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.running = list(running)
    scheduler.max_num_running_reqs = cohort
    scheduler._forced_mode = TTSchedulingMode.DEFAULT
    scheduler._aligned_chunks = True
    scheduler._interleave = None
    scheduler._defer = (
        DeferFinalPolicy(
            CHUNK, cohort, hold_admission=hold or hold_all, hold_one_chunk=hold_all
        )
        if policy
        else None
    )
    scheduler.requests = {r.request_id: r for r in running + waiting}
    return scheduler


def _base_schedule(self):
    rows = {}
    partial = next((r for r in self.running if r.is_prefill_chunk), None)
    if (
        partial is None
        and self.waiting
        and len(self.running) < self.max_num_running_reqs
    ):
        partial = self.waiting.pop_request()
        self.running.append(partial)
    if partial is not None:
        n = min(CHUNK, partial.num_prompt_tokens - partial.num_computed_tokens)
        partial.num_computed_tokens += n
        partial.is_prefill_chunk = (
            partial.num_computed_tokens < partial.num_prompt_tokens
        )
        rows[partial.request_id] = n
    else:
        for r in self.running:
            if not r.is_prefill_chunk:
                r.num_computed_tokens += 1
                rows[r.request_id] = 1
    return _scheduled(rows) if rows else SchedulerOutput.make_empty()


@pytest.fixture
def base(monkeypatch):
    monkeypatch.setattr(AsyncScheduler, "schedule", _base_schedule)


def _trace(scheduler, steps, arrivals=None):
    """The step kinds: ``P<id>:<start>-<end>`` for a prefill row, ``D<n>`` for a decode
    of n rows; ``arrivals``
    maps a step index to prompts that arrive before it."""
    out = []
    for i in range(steps):
        for p in (arrivals or {}).get(i, []):
            r = _req(f"a{i}_{p}", p)
            scheduler.requests[r.request_id] = r
            scheduler.waiting.add_request(r)
        result = scheduler.schedule()
        if not result.num_scheduled_tokens:
            out.append("-")
            continue
        kinds = []
        for req_id, n in result.num_scheduled_tokens.items():
            r = scheduler.requests[req_id]
            if r.num_computed_tokens - n >= r.num_prompt_tokens:
                kinds.append("D")
            else:
                kinds.append(
                    f"P{req_id}:{r.num_computed_tokens - n}-{r.num_computed_tokens}"
                )
        out.append(f"D{len(kinds)}" if kinds[0] == "D" else kinds[0])
        assert all(k == "D" for k in kinds) or len(kinds) == 1, kinds
    return out


def test_policy_unset_is_prefill_first(base):
    scheduler = _scheduler([2 * CHUNK] * 3, policy=False)
    t = _trace(scheduler, 7)
    assert t == [
        "Pw0:0-5120",
        "Pw0:5120-10240",
        "Pw1:0-5120",
        "Pw1:5120-10240",
        "Pw2:0-5120",
        "Pw2:5120-10240",
        "D3",
    ]


def test_cohort_parks_in_order_and_drains_fifo(base):
    prompts = [3 * CHUNK - 100] * 4
    scheduler = _scheduler(prompts, cohort=4)
    t = _trace(scheduler, 14)
    bulk = [f"Pw{i}:{s}-{s + CHUNK}" for i in range(4) for s in (0, CHUNK)]
    finals = [f"Pw{i}:{2 * CHUNK}-{3 * CHUNK - 100}" for i in range(4)]
    assert t == bulk + finals + ["D4", "D4"], t
    assert scheduler._defer.closed_on == ["full"] and scheduler._defer.cohorts == 1


def test_one_chunk_prompt_parks_at_admission(base):
    scheduler = _scheduler([1028, 2 * CHUNK, 1028], cohort=8)
    t = _trace(scheduler, 6)
    # w0 held, w1's bulk, w2 held, the queue empty: the tail closes; finals in admission
    # order
    assert t == [
        "Pw1:0-5120",
        "Pw0:0-1028",
        f"Pw1:5120-{2 * CHUNK}",
        "Pw2:0-1028",
        "D3",
        "D3",
    ], t
    assert scheduler._defer.closed_on == ["tail"]


def test_queue_tail_closes_on_the_empty_queue(base):
    scheduler = _scheduler([2 * CHUNK] * 5, cohort=32)
    t = _trace(scheduler, 11)
    assert t[:5] == [f"Pw{i}:0-5120" for i in range(5)]
    assert t[5:10] == [f"Pw{i}:5120-10240" for i in range(5)]
    assert t[10] == "D5" and scheduler._defer.closed_on == ["tail"]


def test_arrival_after_the_close_waits_for_the_next_cohort(base):
    scheduler = _scheduler([2 * CHUNK] * 2, cohort=32)
    # the cohort closes at step 2 (queue empty); a prompt arriving at step 3 waits until
    # the finals are done
    t = _trace(scheduler, 9, arrivals={3: [2 * CHUNK]})
    assert t[:2] == ["Pw0:0-5120", "Pw1:0-5120"]
    assert t[2:4] == ["Pw0:5120-10240", "Pw1:5120-10240"], t
    # then the arrival: its bulk prefill-first (the stock rule), parked, the tail
    # closes, its final, decode of 3
    assert t[4:7] == ["Pa3_10240:0-5120", "Pa3_10240:5120-10240", "D3"], t


def test_cancellation_while_parked_shortens_the_finals(base):
    scheduler = _scheduler([2 * CHUNK] * 3, cohort=32)
    t = _trace(scheduler, 3)
    assert len(scheduler._defer.parked) == 3
    # w1 finishes (cancelled) between steps: the plugin's finished ids reach observe
    out = SchedulerOutput.make_empty()
    out.finished_req_ids = {"w1"}
    scheduler._finalize_scheduler_output(out)
    scheduler.running = [r for r in scheduler.running if r.request_id != "w1"]
    del scheduler.requests["w1"]
    t = _trace(scheduler, 3)
    assert t == ["Pw0:5120-10240", "Pw2:5120-10240", "D2"], t


def test_early_finish_replaced_into_the_next_cohort(base):
    scheduler = _scheduler([2 * CHUNK] * 2, cohort=32)
    t = _trace(scheduler, 5)
    assert t[-1] == "D2"
    # w0 finishes early and its replacement arrives: the bulk runs prefill-first between
    # w1's decode steps,
    # the tail closes, the final runs, both decode
    out = SchedulerOutput.make_empty()
    out.finished_req_ids = {"w0"}
    scheduler._finalize_scheduler_output(out)
    scheduler.running = [r for r in scheduler.running if r.request_id != "w0"]
    t = _trace(scheduler, 4, arrivals={0: [2 * CHUNK]})
    assert t == ["Pa0_10240:0-5120", "Pa0_10240:5120-10240", "D2", "D2"], t
    assert scheduler._defer.closed_on == ["tail", "tail"]


def test_forced_decode_only_ignores_the_policy(base):
    scheduler = _scheduler([2 * CHUNK] * 2, decoders=1, cohort=32)
    scheduler._forced_mode = TTSchedulingMode.DECODE_ONLY
    t = _trace(scheduler, 2)
    assert t == ["D1", "D1"]


def test_hold_keeps_replacements_out_until_the_cohort_finishes(base):
    """TT_SCHED_DEFER_FINAL=hold: the early finisher's replacement waits for the whole
    cohort."""
    scheduler = _scheduler([2 * CHUNK] * 2, cohort=32, hold=True)
    t = _trace(scheduler, 5)
    assert t[-1] == "D2"
    out = SchedulerOutput.make_empty()
    out.finished_req_ids = {"w0"}
    scheduler._finalize_scheduler_output(out)
    scheduler.running = [r for r in scheduler.running if r.request_id != "w0"]
    # the replacement arrives while w1 decodes: decode steps only, the bulk waits
    t = _trace(scheduler, 3, arrivals={0: [2 * CHUNK]})
    assert t == ["D1", "D1", "D1"], t
    assert scheduler._defer.closed_on == ["tail"]
    # w1 finishes: nothing decodes, the replacement is admitted, its cohort closes on
    # the tail, it decodes
    out = SchedulerOutput.make_empty()
    out.finished_req_ids = {"w1"}
    scheduler._finalize_scheduler_output(out)
    scheduler.running = [r for r in scheduler.running if r.request_id != "w1"]
    t = _trace(scheduler, 4)
    assert t == ["Pa0_10240:0-5120", "Pa0_10240:5120-10240", "D1", "D1"], t
    assert scheduler._defer.closed_on == ["tail", "tail"]


def test_hold_forms_the_next_cohort_from_every_replacement(base):
    """Three of a cohort of four finish at different steps; their replacements wait
    (decode steps only), and
    once the straggler finishes they and a fourth arrival prefill as one cohort of four:
    bulks, finals back to
    back, decode together."""
    scheduler = _scheduler([2 * CHUNK] * 4, cohort=4, hold=True)
    t = _trace(scheduler, 13)
    assert t[-1] == "D4" and scheduler._defer.closed_on == ["full"]
    for i, rid in enumerate(("w0", "w2", "w3")):
        out = SchedulerOutput.make_empty()
        out.finished_req_ids = {rid}
        scheduler._finalize_scheduler_output(out)
        scheduler.running = [r for r in scheduler.running if r.request_id != rid]
        t = _trace(
            scheduler, 3, arrivals={i: [2 * CHUNK]}
        )  # a replacement arrives before step i
        assert t == [f"D{3 - i}"] * 3, (
            i,
            t,
        )  # the ones still decoding; the arrival waits
    assert len(scheduler.waiting) == 3
    out = SchedulerOutput.make_empty()
    out.finished_req_ids = {"w1"}
    scheduler._finalize_scheduler_output(out)
    scheduler.running = [r for r in scheduler.running if r.request_id != "w1"]
    t = _trace(
        scheduler, 9, arrivals={3: [2 * CHUNK]}
    )  # a fourth arrives while the cohort is open
    assert t[:4] == [f"Pa{i}_10240:0-5120" for i in range(4)], t
    assert t[4:8] == [f"Pa{i}_10240:5120-10240" for i in range(4)], t
    assert t[8] == "D4" and scheduler._defer.closed_on == ["full", "full"]


def test_hold_is_off_under_the_stock_value(base):
    scheduler = _scheduler([2 * CHUNK] * 2, cohort=32)
    assert not scheduler._defer.hold_admission
    assert not scheduler._defer.hold_for_decoders(scheduler.running, None)


def test_one_chunk_cohort_counts_every_held_prompt_once(base):
    """One-chunk prompts closed at half the slots when held ids were counted twice."""
    scheduler = _scheduler([1028] * 8, cohort=8)
    t = _trace(scheduler, 9)
    assert t[:8] == [f"Pw{i}:0-1028" for i in range(8)], t
    assert t[8] == "D8" and scheduler._defer.closed_on == ["full"]


def test_hold_yields_to_a_one_chunk_head(base):
    """under hold a one-chunk prompt is admitted between decode steps as the stock rule
    does (its
    prefill is its final), so the first of a burst does not decode alone while the rest
    wait."""
    scheduler = _scheduler([1028], cohort=32, hold=True)
    t = _trace(scheduler, 2)
    assert t == ["Pw0:0-1028", "D1"], t  # a tail cohort of one
    t = _trace(scheduler, 5, arrivals={0: [1028], 1: [1029], 2: [1030]})
    # each arrival is held, the tail closes, its final runs between w0's decode steps,
    # then all decode
    assert t == ["Pa0_1028:0-1028", "Pa1_1029:0-1029", "Pa2_1030:0-1030", "D4", "D4"], t
    # a multi-chunk arrival while they decode waits (the hold rule)
    t = _trace(scheduler, 2, arrivals={0: [2 * CHUNK]})
    assert t == ["D4", "D4"], t


# --- TT_SCHED_DEFER_FINAL=hold-all: one-chunk prompts held for the cohort too ---------


def test_hold_all_keeps_one_chunk_replacements_out_until_the_cohort_finishes(base):
    """The replacements of a decoding cohort of one-chunk prompts wait (decode steps
    only) and form the next
    cohort together once the last decoder finishes: a one-chunk head no longer
    yields."""
    scheduler = _scheduler([1028] * 3, decoders=2, cohort=4, hold_all=True)
    t = _trace(scheduler, 3)
    assert t == ["D2", "D2", "D2"], (
        t
    )  # the decoders are old (12 tokens): nothing admitted
    assert len(scheduler.waiting) == 3 and scheduler._defer.size() == 0
    for rid in ("d0", "d1"):
        out = SchedulerOutput.make_empty()
        out.finished_req_ids = {rid}
        scheduler._finalize_scheduler_output(out)
        scheduler.running = [r for r in scheduler.running if r.request_id != rid]
    # a fourth arrives as the last decoder finishes: the four close the cohort (full)
    # and drain in order
    t = _trace(scheduler, 5, arrivals={0: [1028]})
    assert t == ["Pw0:0-1028", "Pw1:0-1028", "Pw2:0-1028", "Pa0_1028:0-1028", "D4"], t
    assert scheduler._defer.closed_on == ["full"]


def test_hold_all_reopens_a_young_tail_cohort_for_a_burst(base):
    """The first of a burst closes a cohort of one on the empty queue and is prefilled
    alone; the rest arrive
    while it is young (its first decode steps): they reopen the cohort, are held, closed
    on the tail and drained
    while it pauses, and all decode together. An old small cohort (the straggler of a
    wave) is not reopened."""
    scheduler = _scheduler([1028], cohort=32, hold_all=True)
    t = _trace(scheduler, 2)
    assert t == ["Pw0:0-1028", "D1"] and scheduler._defer.closed_on == ["tail"], t
    t = _trace(scheduler, 5, arrivals={0: [1028, 1029, 1030]})
    assert t == ["Pa0_1028:0-1028", "Pa0_1029:0-1029", "Pa0_1030:0-1030", "D4", "D4"], t
    assert scheduler._defer.closed_on == ["tail", "tail"]
    # the cohort is past YOUNG tokens: a later arrival waits for it (the hold rule)
    for r in scheduler.running:
        r.num_computed_tokens = r.num_prompt_tokens + DeferFinalPolicy.YOUNG + 1
    t = _trace(scheduler, 2, arrivals={0: [1031]})
    assert t == ["D4", "D4"] and len(scheduler.waiting) == 1, t


def test_hold_all_reopens_only_below_the_slots(base):
    """A young cohort that fills the slots is not reopened: the arrival waits as under
    hold."""
    scheduler = _scheduler([1028], decoders=4, cohort=4, hold_all=True, generated=1)
    t = _trace(scheduler, 2)
    assert t == ["D4", "D4"], t


def test_hold_keeps_the_stock_rule_for_a_one_chunk_head(base):
    """Under hold (not hold-all) a one-chunk head is still admitted between decode
    steps."""
    scheduler = _scheduler([1028], decoders=1, cohort=32, hold=True, generated=1)
    t = _trace(scheduler, 2)
    assert t == ["Pw0:0-1028", "D2"], t
