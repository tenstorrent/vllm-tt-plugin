"""Row-aware admission (``prefill_rows``): the scheduler admits at most ``prefill_row_capacity()`` waiting prompts into a prefill step; off / no provider = the static cap only."""

import types

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

from vllm_tt_plugin import prefill_rows
from vllm_tt_plugin.scheduler import TTScheduler


@pytest.fixture(autouse=True)
def _reset():
    yield
    prefill_rows.set_prefill_capacity_provider(None)


def _r(n, computed=0, name=None):
    return types.SimpleNamespace(num_tokens=n, num_computed_tokens=computed, name=name or n)


def _stub(waiting, partial=(), budget=16384, cap=4, row_aware=True):
    s = types.SimpleNamespace(
        policy=SchedulingPolicy.FCFS,
        max_num_scheduled_tokens=budget,
        _prefill_chunk_alignment=512,
        _max_partial_prefills=cap,
        _prefill_row_aware=row_aware,
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=1048576)),
    )
    s.waiting = create_request_queue(s.policy)
    for r in waiting:
        s.waiting.add_request(r)
    s._hold_back_beyond_partial_cap = lambda p: TTScheduler._hold_back_beyond_partial_cap(s, p)
    held = TTScheduler._hold_back_partial_prefills(s, list(partial))
    return [r.name for r in s.waiting], [r.name for r in held]


def _long(k):
    return [_r(131076, name=f"L{i}") for i in range(k)]


def test_capacity_limits_the_admitted_prompts():
    prefill_rows.set_prefill_capacity_provider(lambda: 2)
    assert _stub(_long(4)) == (["L0", "L1"], ["L2", "L3"])


def test_zero_capacity_holds_everything_back():
    prefill_rows.set_prefill_capacity_provider(lambda: 0)
    assert _stub(_long(2)) == ([], ["L0", "L1"])


def test_capacity_counts_prompts_that_fit_the_budget_too():
    """a short prompt that finishes in its step still prefills in a row (it would evict the in-progress prompt of that row)"""
    prefill_rows.set_prefill_capacity_provider(lambda: 1)
    assert _stub([_r(300, name="s1"), _r(300, name="s2")]) == (["s1"], ["s2"])


def test_static_cap_still_applies():
    prefill_rows.set_prefill_capacity_provider(lambda: 4)
    assert _stub(_long(4), partial=[_r(131076, 4096), _r(131076, 4096)], cap=3) == (["L0"], ["L1", "L2", "L3"])


def test_off_or_without_provider_is_the_static_cap():
    prefill_rows.set_prefill_capacity_provider(lambda: 0)
    assert _stub(_long(5), row_aware=False) == (["L0", "L1", "L2", "L3"], ["L4"])  # switch off: the provider is not even asked
    prefill_rows.set_prefill_capacity_provider(None)
    assert _stub(_long(5), row_aware=True) == (["L0", "L1", "L2", "L3"], ["L4"])  # nothing registered
    prefill_rows.set_prefill_capacity_provider(lambda: None)
    assert _stub(_long(5), row_aware=True) == (["L0", "L1", "L2", "L3"], ["L4"])  # the model does not say


def test_broken_provider_does_not_stop_the_scheduler():
    def boom():
        raise RuntimeError("x")

    prefill_rows.set_prefill_capacity_provider(boom)
    assert prefill_rows.prefill_row_capacity() is None


def test_config_switch(monkeypatch):
    from vllm_tt_plugin.config import get_tt_prefill_row_aware

    cfg = lambda tt: types.SimpleNamespace(additional_config={"tt": tt} if tt is not None else {})  # noqa: E731
    monkeypatch.delenv("TT_PREFILL_ROW_AWARE", raising=False)
    assert get_tt_prefill_row_aware(cfg(None)) is False
    assert get_tt_prefill_row_aware(cfg({"prefill_row_aware": True})) is True
    monkeypatch.setenv("TT_PREFILL_ROW_AWARE", "1")
    assert get_tt_prefill_row_aware(cfg(None)) is True
    assert get_tt_prefill_row_aware(cfg({"prefill_row_aware": False})) is False  # explicit config wins
