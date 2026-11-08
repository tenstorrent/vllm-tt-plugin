# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Prefill-aware DP load reporting and long-prompt-alone admission."""

from types import SimpleNamespace

import pytest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

from vllm_tt_plugin.config import (
    get_tt_dp_load_config,
    get_tt_long_prefill_alone_config,
)
from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode


def _cfg(**tt):
    return SimpleNamespace(additional_config={"tt": dict(tt)} if tt else {})


class _Req:
    """Hashable by identity, like vLLM's Request."""

    def __init__(self, rid, prompt, computed, cached, chunk):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self.num_computed_tokens = computed
        self.num_cached_tokens = cached
        self.is_prefill_chunk = chunk
        self.num_output_placeholders = 0


def _req(rid, prompt, computed=0, cached=-1, chunk=False):
    return _Req(rid, prompt, computed, cached, chunk)


def _scheduler(*, waiting=(), kv_holding=(), running=(), **tt):
    tt.setdefault("dp_load_prefill_min_tokens", 0)  # tests opt in to the floor
    s = TTScheduler.__new__(TTScheduler)
    s.policy = SchedulingPolicy.FCFS
    s.waiting = create_request_queue(s.policy)
    for r in waiting:
        s.waiting.add_request(r)
    s.kv_holding_waiting = create_request_queue(s.policy)
    for r in kv_holding:
        s.kv_holding_waiting.add_request(r)
    s.running = list(running)
    s.max_num_running_reqs = 8
    s.max_num_active_reqs = 8
    s._forced_mode = TTSchedulingMode.DEFAULT
    cfg = _cfg(**tt)
    (
        s._dp_load_prefill_aware,
        s._dp_load_tokens_per_unit,
        s._dp_load_min_tokens,
    ) = get_tt_dp_load_config(cfg)
    (
        s._long_prefill_threshold,
        s._long_prefill_max_defer,
    ) = get_tt_long_prefill_alone_config(cfg)
    s._long_prefill_defer_steps = 0
    s._long_prefill_hidden_ids = set()
    return s


# --- config ---------------------------------------------------------------


def test_defaults_are_off():
    assert get_tt_dp_load_config(_cfg()) == (False, 2048, 8192)
    assert get_tt_long_prefill_alone_config(_cfg()) == (0, 8)


@pytest.mark.parametrize(
    "tt",
    [
        {"dp_load_prefill_aware": 1},
        {"dp_load_prefill_tokens_per_unit": 0},
        {"dp_load_prefill_tokens_per_unit": True},
        {"dp_load_prefill_min_tokens": -1},
        {"dp_load_prefill_min_tokens": 1.5},
        {"dp_load_prefill_min_tokens": True},
        {"long_prefill_alone_threshold": -1},
        {"long_prefill_alone_threshold": 1.5},
        {"long_prefill_alone_threshold": True},
        {"long_prefill_alone_max_defer_steps": 0},
    ],
)
def test_bad_values_are_rejected(tt):
    getter = (
        get_tt_dp_load_config
        if next(iter(tt)).startswith("dp_load")
        else get_tt_long_prefill_alone_config
    )
    with pytest.raises(ValueError):
        getter(_cfg(**tt))


# --- get_request_counts ---------------------------------------------------


def test_counts_unchanged_when_flag_off():
    s = _scheduler(
        waiting=[_req("a", 100_000)], running=[_req("r", 50_000, chunk=True)]
    )
    assert s.get_request_counts() == (1, 1)


def test_counts_add_penalty_when_on():
    s = _scheduler(
        waiting=[_req("a", 10_000)],
        kv_holding=[_req("b", 1_000)],
        running=[_req("r", 8_192, computed=4_096, chunk=True), _req("d", 9_999)],
        dp_load_prefill_aware=True,
    )
    # pending = 10000 + 1000 + 4096 (decode request is ignored) = 15096
    # ceil(15096 / 2048) = 8; base counts (running=2, waiting=1+1)
    assert s.get_request_counts() == (2, 2 + 8)


def test_penalty_math_uses_cached_tokens_and_rounds_up():
    s = _scheduler(
        waiting=[
            _req("a", 5_000, cached=4_000),
            _req("b", 3_000, computed=500, cached=100),
        ],
        dp_load_prefill_aware=True,
        dp_load_prefill_tokens_per_unit=1000,
    )
    # pending = 1000 + 2500 = 3500 -> ceil = 4
    assert s.get_request_counts() == (0, 2 + 4)


def test_penalty_is_zero_with_nothing_pending():
    s = _scheduler(dp_load_prefill_aware=True)
    assert s.get_request_counts() == (0, 0)
    s = _scheduler(waiting=[_req("a", 100, computed=100)], dp_load_prefill_aware=True)
    assert s.get_request_counts() == (0, 1)


def test_below_min_tokens_adds_nothing():
    s = _scheduler(
        waiting=[_req("a", 8_191), _req("b", 1_000)],
        kv_holding=[_req("c", 5_000)],
        running=[_req("r", 8_000, chunk=True)],
        dp_load_prefill_aware=True,
        dp_load_prefill_min_tokens=8192,
    )
    assert s.get_request_counts() == (1, 3)  # base counts only


def test_at_or_above_min_tokens_adds_ceil_pending_over_unit():
    s = _scheduler(
        waiting=[_req("a", 8_192)],
        dp_load_prefill_aware=True,
        dp_load_prefill_min_tokens=8192,
    )
    assert s.get_request_counts() == (0, 1 + 4)
    s = _scheduler(
        waiting=[_req("a", 10_000)],
        dp_load_prefill_aware=True,
        dp_load_prefill_min_tokens=8192,
    )
    assert s.get_request_counts() == (0, 1 + 5)  # ceil(10000 / 2048)


def test_mixed_only_long_requests_count():
    s = _scheduler(
        waiting=[_req("s", 3_000), _req("L", 20_000)],
        kv_holding=[_req("s2", 8_000)],
        running=[_req("r", 12_000, computed=2_000, chunk=True)],
        dp_load_prefill_aware=True,
        dp_load_prefill_min_tokens=8192,
    )
    # counts: 20000 and 10000 (running); 3000 and 8000 are below the floor.
    # ceil(30000 / 2048) = 15; base (1, 3)
    assert s.get_request_counts() == (1, 3 + 15)


def test_min_tokens_uses_cache_aware_pending():
    h = _hashes("a", 1000)
    s, _ = _cached_sched(
        h[:600], dp_load_prefill_aware=True, dp_load_prefill_min_tokens=8192
    )
    s.waiting.add_request(_hreq("A", 1000 * BS, h))
    # 16000 tokens, 9600 cached -> 6400 pending < 8192 -> no penalty.
    assert s.get_request_counts() == (0, 1)


# --- long-prompt-alone admission -----------------------------------------


def _drive(s, steps, monkeypatch, per_step_cap=10**9):
    """Run _schedule_prefill_only with a fake base scheduler.

    The fake admits every request left in ``waiting`` (up to the cap), which
    is what the real waiting loop does when budget allows. Returns the admitted
    request ids per step.
    """
    seen = []

    def fake_base(self, throttle_prefills=False):
        admitted = []
        while self.waiting and len(admitted) < per_step_cap:
            admitted.append(self.waiting.pop_request().request_id)
        seen.append(admitted)
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base)
    for _ in range(steps):
        s._schedule_prefill_only()
    return seen


def test_flag_off_admits_in_fcfs_order(monkeypatch):
    s = _scheduler(waiting=[_req("L", 100_000), _req("s1", 100), _req("s2", 200)])
    assert _drive(s, 1, monkeypatch) == [["L", "s1", "s2"]]


def test_shorts_first_then_long_alone(monkeypatch):
    s = _scheduler(
        waiting=[
            _req("L1", 100_000),
            _req("s1", 100),
            _req("L2", 50_000),
            _req("s2", 200),
        ],
        long_prefill_alone_threshold=16_384,
    )
    # Step 1: shorts only. Step 2: only longs remain, FCFS among them.
    assert _drive(s, 2, monkeypatch) == [["s1", "s2"], ["L1", "L2"]]
    assert not s.waiting


def test_unscheduled_shorts_keep_order_and_longs_return(monkeypatch):
    s = _scheduler(
        waiting=[_req("L", 100_000), _req("s1", 100), _req("s2", 200), _req("s3", 300)],
        long_prefill_alone_threshold=1000,
    )
    # Cap of 1 per step: the shorts drain FCFS, then the long.
    assert _drive(s, 4, monkeypatch, per_step_cap=1) == [["s1"], ["s2"], ["s3"], ["L"]]


def test_all_long_or_all_short_is_unchanged(monkeypatch):
    s = _scheduler(
        waiting=[_req("L1", 100_000), _req("L2", 90_000)],
        long_prefill_alone_threshold=1000,
    )
    assert _drive(s, 1, monkeypatch) == [["L1", "L2"]]
    s = _scheduler(
        waiting=[_req("s1", 10), _req("s2", 20)], long_prefill_alone_threshold=1000
    )
    assert _drive(s, 1, monkeypatch) == [["s1", "s2"]]


def test_cached_prefix_makes_a_long_prompt_short(monkeypatch):
    s = _scheduler(
        waiting=[_req("A", 100_000, cached=99_500), _req("s", 100)],
        long_prefill_alone_threshold=1000,
    )
    assert _drive(s, 1, monkeypatch) == [["A", "s"]]


def test_starvation_bound(monkeypatch):
    # One short per step arrives while a long waits; the long must run after
    # at most max_defer short-only steps.
    s = _scheduler(
        waiting=[_req("L", 100_000), _req("s0", 10)],
        long_prefill_alone_threshold=1000,
        long_prefill_alone_max_defer_steps=3,
    )
    seen = []

    def fake_base(self, throttle_prefills=False):
        admitted = [self.waiting.pop_request().request_id]
        seen.append(admitted)
        self.waiting.add_request(_req(f"s{len(seen)}", 10))
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base)
    for _ in range(6):
        s._schedule_prefill_only()
    # The fake queues its new arrival during the step, so only the long
    # request's slot (after exactly max_defer short-only steps) is asserted.
    assert [x[0] for x in seen][:4] == ["s0", "s1", "s2", "L"]
    assert all(x[0] != "L" for x in seen[4:])


def test_preempted_request_prepended_during_step_stays_first(monkeypatch):
    s = _scheduler(
        waiting=[_req("L", 100_000), _req("s1", 100)],
        long_prefill_alone_threshold=1000,
    )

    def fake_base(self, throttle_prefills=False):
        self.waiting.pop_request()  # s1 admitted
        pre = _req("P", 50)
        q = create_request_queue(self.policy)
        q.add_request(pre)
        self.waiting.prepend_requests(q)
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_base)
    s._schedule_prefill_only()
    assert [r.request_id for r in s.waiting] == ["P", "L"]


# --- read-only prefix-cache estimate ---------------------------------------

BS = 16


class _Mgr:
    def __init__(self, pool, enable=True):
        self.block_pool = pool
        self.enable_caching = enable
        self.prefix_cache_stats = object()

    def prefix_cache_lookup_enabled(self, request):
        return self.enable_caching and not request.skip_reading_prefix_cache


def _hashes(tag, n):
    return [f"{tag}{i:04d}".encode().ljust(32, b"_") for i in range(n)]


def _cached_sched(cached_hashes, *, enable=True, **tt):
    pool = BlockPool(4096, True, BS)
    for i, h in enumerate(cached_hashes):
        pool.cached_block_hash_to_block.insert(
            make_block_hash_with_group_id(h, 0), pool.blocks[i + 1]
        )
    s = _scheduler(**tt)
    s.kv_cache_manager = _Mgr(pool, enable)
    s.block_size = BS
    return s, pool


def _hreq(rid, prompt, hashes, skip=False, **kw):
    r = _req(rid, prompt, **kw)
    r.block_hashes = hashes
    r.skip_reading_prefix_cache = skip
    return r


def _pool_state(pool, mgr):
    return (
        [b.ref_cnt for b in pool.blocks],
        [b.block_id for b in pool.blocks if b.prev_free_block is None],
        pool.free_block_queue.num_free_blocks,
        {b.block_id: (b.prev_free_block, b.next_free_block) for b in pool.blocks},
        dict(pool.cached_block_hash_to_block._cache),
        mgr.prefix_cache_stats,
    )


def test_estimate_counts_leading_cached_blocks_capped():
    h = _hashes("a", 100)
    s, _ = _cached_sched(h[:60])
    assert s._estimate_cached_prefix_tokens(_hreq("r", 100 * BS, h)) == 60 * BS
    # Fully cached: last token must be recomputed -> one block less.
    s2, _ = _cached_sched(h)
    assert s2._estimate_cached_prefix_tokens(_hreq("r", 100 * BS, h)) == 99 * BS
    # A gap stops the walk (chained hashes).
    s3, _ = _cached_sched(h[:5] + h[6:20])
    assert s3._estimate_cached_prefix_tokens(_hreq("r", 100 * BS, h)) == 5 * BS


def test_estimate_zero_when_disabled_skipped_or_missing():
    h = _hashes("a", 20)
    s, _ = _cached_sched(h, enable=False)
    assert s._estimate_cached_prefix_tokens(_hreq("r", 20 * BS, h)) == 0
    s, _ = _cached_sched(h)
    assert s._estimate_cached_prefix_tokens(_hreq("r", 20 * BS, h, skip=True)) == 0
    assert s._estimate_cached_prefix_tokens(_req("x", 20 * BS)) == 0
    bare = _scheduler()
    assert bare._estimate_cached_prefix_tokens(_hreq("r", 20 * BS, h)) == 0
    s.block_size = BS * 2  # hash size mismatch
    assert s._estimate_cached_prefix_tokens(_hreq("r", 20 * BS, h)) == 0


def test_estimate_has_no_side_effects():
    h = _hashes("a", 30)
    s, pool = _cached_sched(h[:20])
    before = _pool_state(pool, s.kv_cache_manager)
    for _ in range(3):
        assert s._estimate_cached_prefix_tokens(_hreq("r", 30 * BS, h)) == 20 * BS
    after = _pool_state(pool, s.kv_cache_manager)
    assert before == after
    assert before[-1] is after[-1]


def test_cached_long_prompt_adds_no_penalty():
    h = _hashes("a", 2000)
    s, _ = _cached_sched(h[:1990], dp_load_prefill_aware=True)
    s.waiting.add_request(_hreq("A", 2000 * BS, h))
    # 32000 prompt tokens, 31840 cached -> 160 pending -> 1 unit + 1 waiting.
    assert s.get_request_counts() == (0, 2)
    cold, _ = _cached_sched([], dp_load_prefill_aware=True)
    cold.waiting.add_request(_hreq("A", 2000 * BS, h))
    assert cold.get_request_counts() == (0, 1 + 16)


def test_partially_cached_penalty():
    h = _hashes("a", 1000)
    s, _ = _cached_sched(h[:500], dp_load_prefill_aware=True)
    s.waiting.add_request(_hreq("A", 1000 * BS, h))
    # 8000 pending tokens / 2048 -> 4 units.
    assert s.get_request_counts() == (0, 1 + 4)


def test_cached_long_not_deferred_cold_long_is_(monkeypatch):
    h = _hashes("a", 2000)
    s, _ = _cached_sched(h[:1990], long_prefill_alone_threshold=1000)
    s.waiting.add_request(_hreq("A", 2000 * BS, h))
    s.waiting.add_request(_req("s", 100))
    assert _drive(s, 1, monkeypatch) == [["A", "s"]]
    cold, _ = _cached_sched([], long_prefill_alone_threshold=1000)
    cold.waiting.add_request(_hreq("A", 2000 * BS, h))
    cold.waiting.add_request(_req("s", 100))
    assert _drive(cold, 2, monkeypatch) == [["s"], ["A"]]


def test_caching_disabled_counts_full_prompt():
    h = _hashes("a", 100)
    s, _ = _cached_sched(h, enable=False, dp_load_prefill_aware=True)
    s.waiting.add_request(_hreq("A", 100 * BS, h))
    assert s.get_request_counts() == (0, 1 + 1)  # 1600 tokens -> 1 unit


def test_running_partial_prefill_ignores_estimate():
    h = _hashes("a", 1000)
    s, _ = _cached_sched(h, dp_load_prefill_aware=True)
    r = _hreq("R", 1000 * BS, h, computed=1000, chunk=True)
    s.running = [r]
    assert s.get_request_counts() == (1, 0 + -(-(16000 - 1000) // 2048))
