"""TTScheduler first prefill step of an idle engine is capped at ``prefill_first_step_tokens`` (a lone request must not take the whole token budget as its first step: requests that arrive a few
ms later can only join the NEXT step, so the first one would get a head start of that step). Not applied while anything runs, nor when the cap is 0."""

import types

from vllm_tt_plugin import scheduler as S
from vllm_tt_plugin.scheduler import TTScheduler


class _Fake(TTScheduler):
    def __init__(self, first, running=(), waiting=("a",), budget=8192):
        self._first_step_tokens = first
        self.running = list(running)
        self.waiting = list(waiting)
        self.max_num_scheduled_tokens = budget
        self.max_num_running_reqs = self.max_num_active_reqs = 4
        self.seen = []

    def _hold_back_partial_prefills(self, partial):
        self.seen.append(self.max_num_scheduled_tokens)
        return []


def _run(fake, monkeypatch):
    monkeypatch.setattr(S.TTScheduler.__mro__[1], "schedule", lambda self, *a, **k: "out")
    assert fake._schedule_prefill_only() == "out"
    return fake.seen[0]


def test_caps_the_first_step_of_an_idle_engine(monkeypatch):
    f = _Fake(512)
    assert _run(f, monkeypatch) == 512
    assert f.max_num_scheduled_tokens == 8192  # restored


def test_no_cap_when_something_runs(monkeypatch):
    r = types.SimpleNamespace(is_prefill_chunk=True)
    assert _run(_Fake(512, running=[r]), monkeypatch) == 8192


def test_no_cap_when_zero(monkeypatch):
    assert _run(_Fake(0), monkeypatch) == 8192


def test_cap_never_raises_the_budget(monkeypatch):
    assert _run(_Fake(512, budget=256), monkeypatch) == 256
