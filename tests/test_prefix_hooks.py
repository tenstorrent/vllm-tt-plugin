# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``prefix_hooks``: the registry the scheduler asks for a model-owned prefix hit (stdlib only: no vLLM needed)."""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin import prefix_hooks as ph
from vllm_tt_plugin.config import get_tt_prefix_hit_policy, store_tt_prefix_hit_policy


@pytest.fixture(autouse=True)
def _no_hook():
    ph.set_prefix_hook(None)
    yield
    ph.set_prefix_hook(None)


class Hook:
    def __init__(self, hit=1024, row=1, fail=False):
        self.hit, self.row, self.fail, self.calls, self.cancelled = hit, row, fail, [], []

    def plan_hit(self, request_id, tokens, num_tokens, max_hit=None):
        self.calls.append((request_id, list(tokens), num_tokens, max_hit))
        if self.fail:
            raise RuntimeError("boom")
        return self.hit, self.row

    def cancel(self, request_id):
        self.cancelled.append(request_id)


def test_no_hook_means_no_hit():
    assert ph.get_prefix_hook() is None
    assert ph.plan_model_hit("a", [1, 2, 3], 3, max_hit=3) == 0
    ph.cancel_model_hit("a")  # no hook: nothing to do


def test_hit_is_clamped_to_vllms_hit_and_never_negative():
    h = Hook(hit=4096)
    ph.set_prefix_hook(h)
    assert ph.plan_model_hit("a", [1] * 10, 6000, max_hit=2048) == 2048
    assert ph.plan_model_hit("a", [1] * 10, 6000, max_hit=None) == 4096
    assert h.calls[0] == ("a", [1] * 10, 6000, 2048)
    ph.set_prefix_hook(Hook(hit=-5))
    assert ph.plan_model_hit("a", [1], 600, max_hit=600) == 0


def test_finished_ids_reach_a_hook_that_takes_them_and_only_when_given():
    seen = []

    class H2(Hook):
        def plan_hit(self, request_id, tokens, num_tokens, max_hit=None, finished_ids=()):
            seen.append(list(finished_ids))
            return 512, 0

    ph.set_prefix_hook(H2())
    assert ph.plan_model_hit("a", [1], 600, max_hit=600, finished_ids={"x", "y"}) == 512
    assert sorted(seen[0]) == ["x", "y"]
    ph.plan_model_hit("a", [1], 600, max_hit=600, finished_ids=None)
    assert seen[1] == []
    h = Hook(hit=512)  # (a hook whose plan_hit has no finished_ids parameter is only called without it)
    ph.set_prefix_hook(h)
    assert ph.plan_model_hit("a", [1], 600, max_hit=600, finished_ids=None) == 512


def test_zero_max_hit_does_not_ask_the_model():
    h = Hook()
    ph.set_prefix_hook(h)
    assert ph.plan_model_hit("a", [1], 600, max_hit=0) == 0 and h.calls == []


def test_a_failing_hook_is_a_miss():
    ph.set_prefix_hook(Hook(fail=True))
    assert ph.plan_model_hit("a", [1], 600, max_hit=600) == 0


def test_cancel_accepts_one_id_or_many_and_survives_a_failing_hook():
    h = Hook()
    ph.set_prefix_hook(h)
    ph.cancel_model_hit("a")
    ph.cancel_model_hit(["b", "c"])
    assert h.cancelled == ["a", "b", "c"]
    ph.set_prefix_hook(SimpleNamespace(plan_hit=None, cancel=lambda rid: 1 / 0))
    ph.cancel_model_hit("z")  # logged, not raised
    ph.set_prefix_hook(SimpleNamespace(plan_hit=None))  # a hook without cancel
    ph.cancel_model_hit("z")


def test_policy_model_is_accepted_and_garbage_is_not():
    cfg = SimpleNamespace(additional_config={})
    store_tt_prefix_hit_policy(cfg, "model")
    assert get_tt_prefix_hit_policy(cfg) == "model"
    with pytest.raises(ValueError):  # allow-pytest.raises: stdlib-only module test
        store_tt_prefix_hit_policy(cfg, "models")


def test_peek_model_home_is_none_without_hook_or_peek_and_survives_a_failing_hook():
    assert ph.peek_model_home("r", [1, 2], 2) is None
    ph.set_prefix_hook(Hook())  # a hook without peek_home
    assert ph.peek_model_home("r", [1, 2], 2) is None

    class Peeking(Hook):
        def peek_home(self, request_id, tokens, num_tokens, finished_ids=()):
            if self.fail:
                raise RuntimeError("boom")
            return 4096, 2, [1, 0, 0, 2]

    ph.set_prefix_hook(Peeking())
    assert ph.peek_model_home("r", [1, 2], 2) == (4096, 2, [1, 0, 0, 2])
    ph.set_prefix_hook(Peeking(fail=True))
    assert ph.peek_model_home("r", [1, 2], 2) is None
