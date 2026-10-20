# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The TT Omni AR scheduler keeps TT's per-step state across Omni's seams."""

import pytest

pytest.importorskip("vllm_omni")

import vllm.config  # noqa: E402,F401  # finish vLLM init before the plugin package
from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm_omni.core.sched.omni_ar_scheduler import (  # noqa: E402
    OmniARAsyncScheduler,
    OmniARScheduler,
)
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin  # noqa: E402

from tests import test_block_scheduler  # noqa: E402
from vllm_tt_plugin.config import resolve_tt_scheduler_cls  # noqa: E402
from vllm_tt_plugin.omni.scheduler import TTOmniARScheduler  # noqa: E402
from vllm_tt_plugin.omni.worker import TTOmniARWorker  # noqa: E402
from vllm_tt_plugin.scheduler import (  # noqa: E402
    _TT_OUTPUT_FRAME_REQ_IDS_ATTR,
    TTScheduler,
    set_tt_forced_reset_discard_counts,
)


def _bare(pending=None):
    """The scheduler's own state only: what the seams under test read."""
    scheduler = object.__new__(TTOmniARScheduler)
    scheduler._is_block_output_model = False
    scheduler._committing_block_step_decisions = {}
    scheduler._pending_async_output_frames = dict(pending or {})
    return scheduler


def test_omni_runs_around_tt_and_tt_around_the_base_scheduler():
    assert TTOmniARScheduler.__mro__[1:7] == (
        OmniARAsyncScheduler,
        OmniARScheduler,
        OmniSchedulerMixin,
        TTScheduler,
        AsyncScheduler,
        Scheduler,
    )


def test_tt_state_survives_omnis_rewrap():
    base = SchedulerOutput.make_empty()
    setattr(base, _TT_OUTPUT_FRAME_REQ_IDS_ATTR, ["a"])
    set_tt_forced_reset_discard_counts(base, {"a": 2})

    wrapped = _bare()._wrap_omni_scheduler_output(base)

    assert wrapped is not base
    assert getattr(wrapped, _TT_OUTPUT_FRAME_REQ_IDS_ATTR) == ["a"]
    assert wrapped._tt_forced_reset_discard_counts == {"a": 2}


def _omni_update(monkeypatch, calls, raises=False):
    def update(self, scheduler_output, model_runner_output):
        calls.append(scheduler_output)
        if raises:
            raise RuntimeError("omni update failed")
        return {}

    monkeypatch.setattr(OmniARScheduler, "update_from_output", update)


def test_the_step_releases_only_the_frames_it_reserved(monkeypatch):
    calls: list = []
    _omni_update(monkeypatch, calls)
    scheduler = _bare(pending={"a": 2, "b": 1, "c": 1})
    step = SchedulerOutput.make_empty()
    setattr(step, _TT_OUTPUT_FRAME_REQ_IDS_ATTR, ["a", "b"])

    assert scheduler.update_from_output(step, object()) == {}

    assert calls == [step], "Omni's update runs once"
    assert scheduler._pending_async_output_frames == {"a": 1, "c": 1}


def test_the_frames_are_released_when_omnis_update_raises(monkeypatch):
    _omni_update(monkeypatch, [], raises=True)
    scheduler = _bare(pending={"a": 1})
    step = SchedulerOutput.make_empty()
    setattr(step, _TT_OUTPUT_FRAME_REQ_IDS_ATTR, ["a"])

    with pytest.raises(RuntimeError, match="omni update failed"):
        scheduler.update_from_output(step, object())

    assert scheduler._pending_async_output_frames == {}


def _built(monkeypatch, **kwargs):
    """A TTOmniARScheduler built the way the block-scheduler tests build theirs."""
    monkeypatch.setattr(test_block_scheduler, "TTScheduler", TTOmniARScheduler)
    return test_block_scheduler._scheduler(**kwargs)


def test_a_one_token_stage_builds(monkeypatch):
    assert isinstance(_built(monkeypatch, output_width=1), TTOmniARScheduler)


def test_block_output_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="output_tokens_per_step=4"):
        _built(monkeypatch, output_width=4)


def test_speculative_decoding_is_refused(monkeypatch):
    original = test_block_scheduler.store_tt_output_tokens_per_step

    def with_spec(config, width):
        config.speculative_config = object()
        original(config, width)

    monkeypatch.setattr(
        test_block_scheduler, "store_tt_output_tokens_per_step", with_spec
    )
    with pytest.raises(ValueError, match="speculative decoding"):
        _built(monkeypatch, output_width=1)


def test_lane_mode_is_refused_when_the_scheduler_is_paired():
    with pytest.raises(ValueError, match="lane mode"):
        resolve_tt_scheduler_cls(OmniARAsyncScheduler, TTOmniARWorker, True)
