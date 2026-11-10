"""TTScheduler._hold_for_home_row: bounded hold of a request whose resident prefix sits on a row with no free user (DSV41_HOME_ROW_HOLD_S, default off)."""

import types

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

import vllm_tt_plugin.scheduler as sched
from vllm_tt_plugin.scheduler import TTScheduler

ENV = ("DSV41_HOME_ROW_HOLD_S", "DSV41_HOME_ROW_HOLD_MIN", "DSV41_HOME_ROW_HOLD_FACTOR", "DSV41_HOME_ROW_HOLD_RATE", "DSV41_HOME_ROW_HOLD_FILE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def _req(rid, n=400000, computed=0):
    return types.SimpleNamespace(request_id=rid, num_tokens=n, num_computed_tokens=computed, all_token_ids=[1] * 4, prompt_token_ids=[1] * 4)


def _sched(waiting):
    s = types.SimpleNamespace(policy=SchedulingPolicy.FCFS, finished_req_ids=None)
    s.waiting = create_request_queue(s.policy)
    for r in waiting:
        s.waiting.add_request(r)
    s._home_row_hold_params = types.MethodType(TTScheduler._home_row_hold_params, s)
    return s


def _peeker(monkeypatch, answers):
    calls = []

    def peek(rid, tokens, n, finished_ids=None, touch=False):
        calls.append(rid)
        return answers.get(rid)

    monkeypatch.setattr(sched, "peek_model_home", peek)
    return calls


def _step(s, now):
    held = TTScheduler._hold_for_home_row(s, now=now)
    return [r.request_id for r in held], [r.request_id for r in s.waiting]


BUSY = (300000, 2, [0, 1, 0, 3])  # resident 300k on row 2, which has no free user
FREE = (300000, 2, [0, 1, 1, 3])


def test_off_by_default_touches_nothing(monkeypatch):
    calls = _peeker(monkeypatch, {"a": BUSY})
    s = _sched([_req("a"), _req("b")])
    assert _step(s, 0.0) == ([], ["a", "b"]) and calls == []


def test_holds_a_busy_home_row_request_and_lets_the_others_pass(monkeypatch):
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_S", "100")
    _peeker(monkeypatch, {"a": BUSY, "b": None, "c": (0, -1, [1, 1, 1, 1])})
    s = _sched([_req("a"), _req("b"), _req("c")])
    assert _step(s, 0.0) == (["a"], ["b", "c"])


def test_bound_is_min_of_T_and_the_cost_rule_and_never_held_again(monkeypatch):
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_S", "100")  # cost rule: 0.4 * 300000 / 2000 = 60 s
    _peeker(monkeypatch, {"a": BUSY})
    s = _sched([_req("a")])
    assert _step(s, 0.0)[0] == ["a"]
    s.waiting.add_request(_req("a"))  # (the caller puts it back)
    s.waiting = create_request_queue(s.policy)
    r = _req("a")
    s.waiting.add_request(r)
    assert _step(s, 59.0)[0] == ["a"]
    s.waiting = create_request_queue(s.policy)
    s.waiting.add_request(r)
    assert _step(s, 60.5) == ([], ["a"])  # bound over: admitted cold
    s.waiting = create_request_queue(s.policy)
    s.waiting.add_request(r)
    assert _step(s, 61.0) == ([], ["a"])  # and never held again, whatever the model says


def test_T_caps_the_hold(monkeypatch):
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_S", "10")
    _peeker(monkeypatch, {"a": BUSY})
    s = _sched([_req("a")])
    assert _step(s, 0.0)[0] == ["a"]
    s.waiting = create_request_queue(s.policy)
    s.waiting.add_request(_req("a"))
    assert _step(s, 10.5) == ([], ["a"])


def test_ends_early_when_the_home_row_frees_a_user(monkeypatch):
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_S", "100")
    answers = {"a": BUSY}
    _peeker(monkeypatch, answers)
    s = _sched([_req("a")])
    assert _step(s, 0.0)[0] == ["a"]
    answers["a"] = FREE
    s.waiting = create_request_queue(s.policy)
    s.waiting.add_request(_req("a"))
    assert _step(s, 1.0) == ([], ["a"])  # (plan_hit then reserves the free user of the home row)


def test_small_hits_started_and_unknown_requests_are_not_held(monkeypatch):
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_S", "100")
    _peeker(monkeypatch, {"small": (8192, 1, [1, 0, 1, 1]), "x": (300000, 0, [0, 0, 0, 0])})
    s = _sched([_req("small"), _req("started", computed=512), _req("nohook")])
    assert _step(s, 0.0) == ([], ["small", "started", "nohook"])


def test_hit_min_env(monkeypatch):
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_S", "100")
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_MIN", "400000")
    _peeker(monkeypatch, {"a": BUSY})
    assert _step(_sched([_req("a")]), 0.0) == ([], ["a"])


def test_two_held_requests_keep_fifo_order_and_go_back_in_front(monkeypatch):
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_S", "100")
    _peeker(monkeypatch, {"a": BUSY, "c": BUSY})
    s = _sched([_req("a"), _req("b"), _req("c"), _req("d")])
    held, rest = _step(s, 0.0)
    assert (held, rest) == (["a", "c"], ["b", "d"])
    # the scheduler admits b, d; the held ones return in front of what is left
    s.waiting = create_request_queue(s.policy)
    s.waiting.add_request(_req("e"))
    s.waiting.prepend_requests(reversed([_req(h) for h in held]))
    assert [r.request_id for r in s.waiting] == ["a", "c", "e"]


def test_state_of_vanished_requests_is_dropped(monkeypatch):
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_S", "100")
    _peeker(monkeypatch, {"a": BUSY})
    s = _sched([_req("a")])
    _step(s, 0.0)
    s.waiting = create_request_queue(s.policy)
    s.waiting.add_request(_req("z"))
    _step(s, 1.0)
    assert "a" not in s._home_hold


def test_switch_file_turns_it_off_live(monkeypatch, tmp_path):
    f = tmp_path / "hold"
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_S", "100")
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_FILE", str(f))
    _peeker(monkeypatch, {"a": BUSY})
    f.write_text("0\n")
    assert _step(_sched([_req("a")]), 0.0) == ([], ["a"])
    f.write_text("1\n")
    assert _step(_sched([_req("a")]), 0.0)[0] == ["a"]
    f.unlink()  # missing file: the env decides
    assert _step(_sched([_req("a")]), 0.0)[0] == ["a"]


def test_a_free_user_an_earlier_request_will_reserve_is_not_free_for_the_next(monkeypatch):
    monkeypatch.setenv("DSV41_HOME_ROW_HOLD_S", "100")
    one_free = [0, 0, 1, 0]
    _peeker(monkeypatch, {"small": (8192, 2, one_free), "giant": (300000, 2, one_free)})
    # queue order: the small one reserves the only free user of row 2 first -> the giant is held
    assert _step(_sched([_req("small", 9000), _req("giant")]), 0.0) == (["giant"], ["small"])
    # alone, the giant finds the user free and is planned on its home row
    assert _step(_sched([_req("giant")]), 0.0) == ([], ["giant"])
    # a request ahead without any hit takes nothing
    _peeker(monkeypatch, {"cold": (0, -1, one_free), "giant": (300000, 2, one_free)})
    assert _step(_sched([_req("cold"), _req("giant")]), 0.0) == ([], ["cold", "giant"])
