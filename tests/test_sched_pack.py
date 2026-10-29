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
