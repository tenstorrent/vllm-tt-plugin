# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deferred warmup units (background_warmup): between busy steps after an interval, and all at once while idle."""
import queue
import time

import pytest

from vllm_tt_plugin import background_warmup as bw


class FakeModel:
    def __init__(self, units):
        self.left = units
        self.calls = 0

    def background_warmup_step(self):
        self.calls += 1
        self.left -= 1
        return self.left > 0

    def background_warmup_pending(self):
        return self.left


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(bw, "_state", {"last": None, "busy_units": 0, "done": False})
    monkeypatch.setattr(bw, "READINESS_FILE", None)


def busy_steps(model, seconds, interval_s):
    """A steadily loaded engine: one 10 ms step after another, never idle."""
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        bw.busy_step(model, interval_s)
        time.sleep(0.01)


def test_a_never_idle_engine_runs_one_unit_per_interval_and_finishes():
    model = FakeModel(4)
    bw.started()
    busy_steps(model, 0.65, interval_s=0.1)
    assert model.calls == 4 and model.left == 0  # ~6 intervals: the 4 units, then nothing more


def test_the_interval_spaces_the_units_and_counts_from_the_end_of_warmup():
    model = FakeModel(10)
    bw.started()
    busy_steps(model, 0.5, interval_s=0.2)
    assert 1 <= model.calls <= 2


def test_off_with_interval_zero_and_before_warmup_ended():
    model = FakeModel(4)
    busy_steps(model, 0.2, interval_s=0.05)  # started() not called: still in warmup
    assert model.calls == 0
    bw.started()
    busy_steps(model, 0.2, interval_s=0)
    assert model.calls == 0


def test_idle_units_run_all_at_once_and_share_the_state_with_busy_units():
    class Core:
        def __init__(self, model):
            self.model, self.input_queue, self.model_executor = model, queue.Queue(), self

        def has_work(self):
            return False

        def is_running(self):
            return True

        def collective_rpc(self, method):
            assert method == "background_warmup_step"
            return [bw.step(self.model)]

    model = FakeModel(5)
    bw.started()
    bw.wrap(lambda self: None)(Core(model))
    assert model.calls == 5
    bw.busy_step(model, interval_s=0.0001)  # nothing left: no more units
    assert model.calls == 5
