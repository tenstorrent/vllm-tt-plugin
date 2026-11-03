# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""TT_SCHED_PACK: the admission of a listed layout of whole prompts (sched_pack)."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm_tt_plugin.sched_interleave import InterleavePolicy
from vllm_tt_plugin.sched_pack import PackPolicy, parse_layouts
from vllm_tt_plugin.scheduler import (
    KNOWN_SCHED_FLAGS,
    TTScheduler,
    TTSchedulingMode,
    check_sched_flags,
)

LAYOUTS = "4x1028,3x1028,2x1028,2x2052"

# --- the spec -----------------------------------------------------------------


def test_parse_layouts_longest_first():
    assert parse_layouts(LAYOUTS) == [
        (1028, 1028, 1028, 1028),
        (1028, 1028, 1028),
        (2052, 2052),
        (1028, 1028),
    ]
    assert parse_layouts("1028+2052, 2x1028") == [(1028, 2052), (1028, 1028)]
    with pytest.raises(ValueError):
        parse_layouts("1x1028")
    with pytest.raises(ValueError):
        parse_layouts("")


def test_from_env_unset_or_zero_is_none():
    assert PackPolicy.from_env({}) is None
    assert PackPolicy.from_env({"TT_SCHED_PACK": "0"}) is None
    assert PackPolicy.from_env({"TT_SCHED_PACK": ""}) is None
    policy = PackPolicy.from_env({"TT_SCHED_PACK": LAYOUTS}, chunk_tokens=5120)
    assert policy is not None and policy.kmax == 4
    with pytest.raises(ValueError, match="token budget"):
        PackPolicy.from_env({"TT_SCHED_PACK": "2x4100"}, chunk_tokens=5120)


def test_registry_knows_the_flag():
    assert "TT_SCHED_PACK" in KNOWN_SCHED_FLAGS
    check_sched_flags({"TT_SCHED_PACK": LAYOUTS, "TT_SCHED_INTERLEAVE": "every=8"})
    with pytest.raises(ValueError, match="TT_SCHED_PCAK"):
        check_sched_flags({"TT_SCHED_PCAK": LAYOUTS})


# --- the admission rule -----------------------------------------------------------


def _waiting(req_id="w", prompt=1028):
    return SimpleNamespace(request_id=req_id, num_prompt_tokens=prompt)


def _queue(*prompts):
    queue = create_request_queue(SchedulingPolicy.FCFS)
    for i, n in enumerate(prompts):
        queue.add_request(_waiting(f"w{i}", n))
    return queue


def test_admit_is_the_longest_listed_prefix():
    policy = PackPolicy(parse_layouts(LAYOUTS))
    assert policy.admit(_queue(1028, 1028, 1028, 1028, 1028)) == 4
    assert policy.admit(_queue(1028, 1028, 1028)) == 3
    assert policy.admit(_queue(1028, 1028, 2052)) == 2
    assert policy.admit(_queue(2052, 2052, 1028)) == 2
    assert policy.admit(_queue(1028)) == 1, "never waits for a second arrival"
    assert policy.admit(_queue(2052, 1028, 1028)) == 1, "the head is not packable"
    assert policy.admit(_queue(4100, 1028, 1028, 1028)) == 1, "a long head runs alone"
    assert policy.admit(_queue(1027, 1028, 1028)) == 1, "an unlisted length"
    assert policy.admit(_queue()) == 1


def test_observe_counts_packed_admissions():
    policy = PackPolicy(parse_layouts(LAYOUTS))
    policy.observe(SimpleNamespace(scheduled_new_reqs=[1, 2, 3, 4]), {}, 0)
    policy.observe(SimpleNamespace(scheduled_new_reqs=[1]), {}, 0)
    policy.observe(SimpleNamespace(scheduled_new_reqs=[]), {}, 0)
    assert policy.packs == 1 and policy.packed_prompts == 4


# --- the scheduler's prefill step --------------------------------------------------


def _running(req_id="r", is_prefill_chunk=False, prompt=1028):
    return SimpleNamespace(
        request_id=req_id, is_prefill_chunk=is_prefill_chunk, num_prompt_tokens=prompt
    )


def _scheduler(*, running=(), waiting=(), pack=None, interleave=None, aligned=True):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    for request in waiting:
        scheduler.waiting.add_request(request)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.running = list(running)
    scheduler.max_num_running_reqs = 32
    scheduler._forced_mode = TTSchedulingMode.DEFAULT
    scheduler._interleave = interleave
    scheduler._pack = pack
    scheduler._aligned_chunks = aligned
    scheduler.requests = {}
    return scheduler


def _capture_cap(scheduler, monkeypatch, seen):
    """The base scheduler's ``schedule`` replaced by a probe of the running cap
    it is called with (what bounds the waiting loop's admissions)."""

    def schedule(self, throttle_prefills=False):
        seen.append(self.max_num_running_reqs)
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", schedule)


def test_prefill_step_caps_admission_at_the_layout(monkeypatch):
    policy = PackPolicy(parse_layouts(LAYOUTS))
    decoders = [_running(f"r{i}") for i in range(8)]
    scheduler = _scheduler(
        running=decoders, waiting=[_waiting(f"w{i}") for i in range(6)], pack=policy
    )
    seen = []
    _capture_cap(scheduler, monkeypatch, seen)
    scheduler._schedule_prefill_only()
    assert seen == [4], "four whole 1,028-token prompts, never a partial fifth"
    assert scheduler.max_num_running_reqs == 32 and len(scheduler.running) == 8


def test_prefill_step_admits_one_when_the_flag_is_unset(monkeypatch):
    scheduler = _scheduler(
        running=[_running()], waiting=[_waiting(f"w{i}") for i in range(6)], pack=None
    )
    seen = []
    _capture_cap(scheduler, monkeypatch, seen)
    scheduler._schedule_prefill_only()
    assert seen == [1]


def test_prefill_step_admits_one_when_the_head_does_not_pack(monkeypatch):
    policy = PackPolicy(parse_layouts(LAYOUTS))
    scheduler = _scheduler(
        running=[_running()],
        waiting=[_waiting("w0", 4100), _waiting("w1"), _waiting("w2")],
        pack=policy,
    )
    seen = []
    _capture_cap(scheduler, monkeypatch, seen)
    scheduler._schedule_prefill_only()
    assert seen == [1], "the head of the queue runs alone rather than waiting"


def test_partial_prefill_advances_alone_whatever_waits(monkeypatch):
    policy = PackPolicy(parse_layouts(LAYOUTS))
    partial = _running("p", is_prefill_chunk=True, prompt=12000)
    scheduler = _scheduler(
        running=[_running(), partial],
        waiting=[_waiting(f"w{i}") for i in range(4)],
        pack=policy,
    )
    hidden = []

    def schedule(self, throttle_prefills=False):
        hidden.append(len(self.waiting))
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", schedule)
    scheduler._schedule_prefill_only()
    assert hidden == [0], "the waiting queue is hidden behind a partial prefill"
    assert len(scheduler.waiting) == 4


def test_cap_never_exceeds_the_free_slots(monkeypatch):
    policy = PackPolicy(parse_layouts(LAYOUTS))
    decoders = [_running(f"r{i}") for i in range(30)]
    scheduler = _scheduler(
        running=decoders, waiting=[_waiting(f"w{i}") for i in range(4)], pack=policy
    )
    seen = []
    _capture_cap(scheduler, monkeypatch, seen)
    scheduler._schedule_prefill_only()
    assert seen == [2], "32 slots, 30 decoding: two of the four"


def test_composes_with_interleave_counting_every_packed_prompt():
    """A packed step completes k prompts in one output: the interleave count
    sees them all (the contract of ``_prefer_decode``)."""
    interleave = InterleavePolicy(every=4)
    pack = PackPolicy(parse_layouts(LAYOUTS))
    scheduler = _scheduler(running=[_running()], pack=pack, interleave=interleave)
    scheduler.requests = {
        f"w{i}": SimpleNamespace(
            request_id=f"w{i}", num_prompt_tokens=1028, num_computed_tokens=1028
        )
        for i in range(4)
    }
    out = SchedulerOutput.make_empty()
    out.num_scheduled_tokens = {f"w{i}": 1028 for i in range(4)}
    out.total_num_scheduled_tokens = 4 * 1028
    out.scheduled_new_reqs = [SimpleNamespace(req_id=f"w{i}") for i in range(4)]
    scheduler._finalize_scheduler_output(out)
    assert interleave.completed == 4, "one packed step completed four prompts"
    assert pack.packs == 1 and pack.packed_prompts == 4
    assert interleave.burst_due(ready_decoders=1, waiting=4, next_prompt_tokens=1028)


# --- the deferred drain (TT_SCHED_DEFER_FINAL=hold with TT_SCHED_PACK) ----------------

from vllm_tt_plugin.sched_defer_final import DeferFinalPolicy  # noqa: E402

CHUNK = 5120


def _held(req_id, prompt, computed=0):
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
        is_prefill_chunk=computed < prompt,
        status="WAITING",
    )


def _deferred_scheduler(prompts, *, pack, cohort=8, decoders=0):
    """The defer tests' scheduler (``test_sched_defer_final``) with the pack policy set."""
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    running = [_held(f"d{i}", 1028, 1040) for i in range(decoders)]
    waiting = [_held(f"w{i}", p) for i, p in enumerate(prompts)]
    for request in waiting:
        scheduler.waiting.add_request(request)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.running = list(running)
    scheduler.max_num_running_reqs = cohort
    scheduler._forced_mode = TTSchedulingMode.DEFAULT
    scheduler._aligned_chunks = True
    scheduler._interleave = None
    scheduler._pack = pack
    scheduler._defer = DeferFinalPolicy(CHUNK, cohort, hold_admission=True)
    scheduler.requests = {r.request_id: r for r in running + waiting}
    return scheduler


def _base_schedule_whole(self, throttle_prefills=False):
    """The base loop as the deferred path sees it, admitting whole prompts up to the running cap (the token budget
    holds a listed layout), else one chunk of the first partial, else a decode of every running decode row."""
    rows = {}
    new = []
    partial = next((r for r in self.running if r.is_prefill_chunk), None)
    if partial is None:
        while self.waiting and len(self.running) < self.max_num_running_reqs:
            r = self.waiting.pop_request()
            self.running.append(r)
            new.append(r)
            n = min(CHUNK, r.num_prompt_tokens)
            r.num_computed_tokens = n
            r.is_prefill_chunk = n < r.num_prompt_tokens
            rows[r.request_id] = n
    elif partial is not None:
        n = min(CHUNK, partial.num_prompt_tokens - partial.num_computed_tokens)
        partial.num_computed_tokens += n
        partial.is_prefill_chunk = partial.num_computed_tokens < partial.num_prompt_tokens
        rows[partial.request_id] = n
    if not rows:
        for r in self.running:
            if not r.is_prefill_chunk:
                r.num_computed_tokens += 1
                rows[r.request_id] = 1
    out = SchedulerOutput.make_empty()
    out.num_scheduled_tokens = rows
    out.total_num_scheduled_tokens = sum(rows.values())
    out.scheduled_new_reqs = new
    return out


def _steps(scheduler, n):
    """Per step: the sorted ids of the rows it scheduled."""
    out = []
    for _ in range(n):
        result = scheduler.schedule()
        out.append(sorted(result.num_scheduled_tokens))
    return out


def test_deferred_drain_packs_the_held_prompts(monkeypatch):
    """Under TT_SCHED_DEFER_FINAL=hold a one-chunk prompt is held into the cohort and drained one a step; with
    TT_SCHED_PACK the drain admits the leading held prompts that form a listed layout in one step."""
    monkeypatch.setattr(AsyncScheduler, "schedule", _base_schedule_whole)
    pack = PackPolicy(parse_layouts(LAYOUTS))
    scheduler = _deferred_scheduler([1028] * 8, pack=pack, cohort=8)
    t = _steps(scheduler, 3)
    # the eight held prompts close the cohort (full); the drain is two packs of four, then the decode of all eight
    assert t == [["w0", "w1", "w2", "w3"], ["w4", "w5", "w6", "w7"], [f"w{i}" for i in range(8)]], t
    assert scheduler._defer.closed_on == ["full"] and not scheduler._defer.draining
    assert pack.packs == 2 and pack.packed_prompts == 8


def test_deferred_drain_packs_only_a_listed_prefix(monkeypatch):
    """A tail cohort of 1028, 1028, 2052, 2052, 1028: the first pair, then the pair of 2,052s, then one alone."""
    monkeypatch.setattr(AsyncScheduler, "schedule", _base_schedule_whole)
    pack = PackPolicy(parse_layouts(LAYOUTS))
    scheduler = _deferred_scheduler([1028, 1028, 2052, 2052, 1028], pack=pack, cohort=32)
    t = _steps(scheduler, 4)
    assert t == [["w0", "w1"], ["w2", "w3"], ["w4"], [f"w{i}" for i in range(5)]], t
    assert scheduler._defer.closed_on == ["tail"]


def test_deferred_drain_never_packs_a_parked_final(monkeypatch):
    """A parked (multi-chunk) final between held prompts ends the pack: held w0, w1 pack; w2's final alone; w3."""
    monkeypatch.setattr(AsyncScheduler, "schedule", _base_schedule_whole)
    pack = PackPolicy(parse_layouts(LAYOUTS))
    scheduler = _deferred_scheduler([1028, 1028, 2 * CHUNK + 100, 1028], pack=pack, cohort=32)
    t = _steps(scheduler, 6)
    # w0 and w1 held, w2's two bulk chunks (it parks after them), w3 held, the tail closes: w0 + w1 packed, w2's
    # final alone, w3 alone, then the cohort decodes
    assert t == [["w2"], ["w2"], ["w0", "w1"], ["w2"], ["w3"], ["w0", "w1", "w2", "w3"]], t


def test_deferred_drain_without_the_pack_flag_is_one_a_step(monkeypatch):
    monkeypatch.setattr(AsyncScheduler, "schedule", _base_schedule_whole)
    scheduler = _deferred_scheduler([1028] * 4, pack=None, cohort=4)
    t = _steps(scheduler, 5)
    assert t == [["w0"], ["w1"], ["w2"], ["w3"], ["w0", "w1", "w2", "w3"]], t
