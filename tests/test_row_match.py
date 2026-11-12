"""TTScheduler._match_waiting_to_free_rows: row-matching admission (DSV41_ROW_MATCH_K, default off)."""

import types

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

import vllm_tt_plugin.scheduler as sched
from vllm_tt_plugin.scheduler import TTScheduler

ENV = ("DSV41_ROW_MATCH_K", "DSV41_ROW_MATCH_WAIT_S", "DSV41_ROW_MATCH_DELTA_S", "DSV41_ROW_MATCH_FILE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def _req(rid, n=100000, computed=0):
    return types.SimpleNamespace(request_id=rid, num_tokens=n, num_computed_tokens=computed, all_token_ids=[1] * 4, prompt_token_ids=[1] * 4)


def _sched(waiting, cap=1):
    s = types.SimpleNamespace(policy=SchedulingPolicy.FCFS, finished_req_ids=None, max_num_running_reqs=cap, running=[])
    s.waiting = create_request_queue(s.policy)
    for r in waiting:
        s.waiting.add_request(r)
    s._row_match_params = types.MethodType(TTScheduler._row_match_params, s)
    return s


def _peeker(monkeypatch, answers):
    calls = []

    def peek(rid, tokens, n, finished_ids=None, touch=False):
        calls.append(rid)
        return answers.get(rid)

    monkeypatch.setattr(sched, "peek_model_home", peek)
    return calls


def _step(s, now=0.0):
    TTScheduler._match_waiting_to_free_rows(s, now=now)
    return [r.request_id for r in s.waiting]


ONE_FREE_ROW2 = [0, 0, 1, 0]


def test_off_by_default_touches_nothing(monkeypatch):
    calls = _peeker(monkeypatch, {"a": (100000, 0, ONE_FREE_ROW2)})
    assert _step(_sched([_req("a"), _req("b")])) == ["a", "b"] and calls == []


def test_free_user_goes_to_the_request_whose_prefix_is_on_that_row(monkeypatch):
    monkeypatch.setenv("DSV41_ROW_MATCH_K", "8")
    _peeker(monkeypatch, {"a": (200000, 0, ONE_FREE_ROW2), "b": (50000, 2, ONE_FREE_ROW2), "c": (0, -1, ONE_FREE_ROW2)})
    assert _step(_sched([_req("a"), _req("b"), _req("c")])) == ["b", "a", "c"]


def test_head_with_no_copy_stays_first(monkeypatch):
    monkeypatch.setenv("DSV41_ROW_MATCH_K", "8")
    _peeker(monkeypatch, {"a": (200000, 2, ONE_FREE_ROW2), "b": (50000, 2, ONE_FREE_ROW2)})
    assert _step(_sched([_req("a"), _req("b")])) == ["a", "b"]


def test_cheapest_copy_wins_when_nobody_is_local(monkeypatch):
    monkeypatch.setenv("DSV41_ROW_MATCH_K", "8")
    _peeker(monkeypatch, {"a": (600000, 0, ONE_FREE_ROW2), "b": (40000, 1, ONE_FREE_ROW2)})
    assert _step(_sched([_req("a"), _req("b")])) == ["b", "a"]


def test_a_head_that_waited_W_is_never_passed_over(monkeypatch):
    monkeypatch.setenv("DSV41_ROW_MATCH_K", "8")
    monkeypatch.setenv("DSV41_ROW_MATCH_WAIT_S", "30")
    _peeker(monkeypatch, {"a": (600000, 0, ONE_FREE_ROW2), "b": (50000, 2, ONE_FREE_ROW2)})
    s = _sched([_req("a"), _req("b")])
    assert _step(s, 0.0) == ["b", "a"]
    assert _step(s, 31.0) == ["b", "a"]  # (a's first sighting was t=0; order is b,a now: head b)
    s = _sched([_req("a"), _req("b")])
    s._row_match_state = {"a": [0.0, None, None]}
    assert _step(s, 31.0) == ["a", "b"]


def test_delta_keeps_small_savings_out(monkeypatch):
    monkeypatch.setenv("DSV41_ROW_MATCH_K", "8")
    monkeypatch.setenv("DSV41_ROW_MATCH_DELTA_S", "5")
    _peeker(monkeypatch, {"a": (100000, 0, ONE_FREE_ROW2), "b": (0, -1, ONE_FREE_ROW2)})  # a's copy 2.6 s < 5
    assert _step(_sched([_req("a"), _req("b")])) == ["a", "b"]


def test_capacity_two_takes_two_users(monkeypatch):
    monkeypatch.setenv("DSV41_ROW_MATCH_K", "8")
    free = [0, 0, 1, 1]
    _peeker(monkeypatch, {"a": (300000, 0, free), "b": (60000, 2, free), "c": (60000, 3, free), "d": (60000, 2, free)})
    assert _step(_sched([_req("a"), _req("b"), _req("c"), _req("d")], cap=2)) == ["b", "c", "a", "d"]


def test_window_limits_the_look(monkeypatch):
    monkeypatch.setenv("DSV41_ROW_MATCH_K", "2")
    _peeker(monkeypatch, {"a": (300000, 0, ONE_FREE_ROW2), "b": (0, -1, ONE_FREE_ROW2), "c": (50000, 2, ONE_FREE_ROW2)})
    assert _step(_sched([_req("a"), _req("b"), _req("c")])) == ["b", "a", "c"]


def test_no_capacity_or_single_request_is_a_no_op(monkeypatch):
    monkeypatch.setenv("DSV41_ROW_MATCH_K", "8")
    calls = _peeker(monkeypatch, {})
    assert _step(_sched([_req("a"), _req("b")], cap=0)) == ["a", "b"] and calls == []
    assert _step(_sched([_req("a")])) == ["a"] and calls == []


def test_started_requests_are_not_moved_and_unknown_cost_nothing(monkeypatch):
    monkeypatch.setenv("DSV41_ROW_MATCH_K", "8")
    _peeker(monkeypatch, {"b": (50000, 2, ONE_FREE_ROW2)})
    assert _step(_sched([_req("a", computed=512), _req("b")])) == ["a", "b"]


def test_switch_file(monkeypatch, tmp_path):
    f = tmp_path / "rm"
    monkeypatch.setenv("DSV41_ROW_MATCH_K", "8")
    monkeypatch.setenv("DSV41_ROW_MATCH_FILE", str(f))
    _peeker(monkeypatch, {"a": (200000, 0, ONE_FREE_ROW2), "b": (50000, 2, ONE_FREE_ROW2)})
    f.write_text("0\n")
    assert _step(_sched([_req("a"), _req("b")])) == ["a", "b"]
    f.write_text("1\n")
    assert _step(_sched([_req("a"), _req("b")])) == ["b", "a"]
    f.unlink()
    assert _step(_sched([_req("a"), _req("b")])) == ["b", "a"]


def test_replay_of_the_simulator_decisions(monkeypatch):
    """program/rowaffine/sim.py (K=8, W=60, delta 0.3): the first pick of the plugin function equals the simulator's pick in at least 85 % of 150 recorded admission decisions
    (the simulator ranks full placements, the plugin the peek answer: the deepest hit, its row and the free users per row)."""
    import json, os

    snaps = json.load(open(os.path.join(os.path.dirname(__file__), "row_match_sim_snapshots.json")))
    monkeypatch.setenv("DSV41_ROW_MATCH_K", "8")
    monkeypatch.setenv("DSV41_ROW_MATCH_WAIT_S", "60")
    same = 0
    for sn in snaps:
        answers = {str(c["id"]): (c["hit"], c["home"], sn["free"]) for c in sn["cands"]}
        _peeker(monkeypatch, answers)
        s = _sched([_req(str(c["id"])) for c in sn["cands"]], cap=1)
        s._row_match_state = {str(c["id"]): [1000.0 - c["waited"], None, None] for c in sn["cands"]}
        order = _step(s, 1000.0)
        same += order[0] == str(sn["pick"])
    assert same >= 0.85 * len(snaps), (same, len(snaps))
