"""TTScheduler._schedule_prefill_only hides the running decodes by lowering max_num_running_reqs; vLLM >= 0.31 admits waiting requests against max_num_active_reqs (a copy of
max_num_seqs made at construction), so both limits have to shrink by the hidden decodes while the base scheduler runs, and both are restored afterwards. Without it a queue longer than
the slot count admitted past it and the base ``assert len(running) <= max_num_running_reqs`` killed the engine (5th concurrent request of a B=4 server)."""

import types

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_tt_plugin.scheduler import TTScheduler


def _req(prefill_chunk):
    return types.SimpleNamespace(is_prefill_chunk=prefill_chunk)


class _Fake(TTScheduler):
    def __init__(self, max_seqs, running):
        self.running = running
        self.max_num_running_reqs = max_seqs
        self.max_num_active_reqs = max_seqs
        self.waiting = types.SimpleNamespace(add_request=lambda r: None)

    def _hold_back_partial_prefills(self, partial_prefills):
        return []


@pytest.mark.parametrize("n_decodes,n_partial", [(0, 0), (1, 0), (3, 1), (4, 0)])
def test_both_admission_limits_shrink_while_the_base_scheduler_runs_and_are_restored(monkeypatch, n_decodes, n_partial):
    seen = {}

    def fake_schedule(self):
        seen["running"] = self.max_num_running_reqs
        seen["active"] = self.max_num_active_reqs
        seen["n_running"] = len(self.running)
        return "out"

    monkeypatch.setattr(AsyncScheduler, "schedule", fake_schedule)
    running = [_req(False) for _ in range(n_decodes)] + [_req(True) for _ in range(n_partial)]
    s = _Fake(4, running)
    assert s._schedule_prefill_only() == "out"
    assert seen["running"] == 4 - n_decodes and seen["active"] == 4 - n_decodes  # both shrink
    assert seen["n_running"] == n_partial  # (the decodes are hidden)
    assert s.max_num_running_reqs == 4 and s.max_num_active_reqs == 4  # restored
    assert len(s.running) == n_decodes + n_partial


def test_a_scheduler_without_max_num_active_reqs_(monkeypatch):
    """vLLM < 0.31 has no max_num_active_reqs: only max_num_running_reqs shrinks."""
    monkeypatch.setattr(AsyncScheduler, "schedule", lambda self: self.max_num_running_reqs)
    s = _Fake(4, [_req(False), _req(False)])
    del s.max_num_active_reqs
    assert s._schedule_prefill_only() == 2 and not hasattr(s, "max_num_active_reqs") and s.max_num_running_reqs == 4
