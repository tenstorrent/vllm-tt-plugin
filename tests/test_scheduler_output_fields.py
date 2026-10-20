# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The TT fields on ``SchedulerOutput`` and the output-frame release hook."""

from dataclasses import fields

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

from tests.spec.test_spec_final_audit import _chunked_scheduler
from tests.test_block_scheduler import _request, _runner_output, _scheduler
from tests.test_lane_scheduler import FakeLane, _make_coordinator
from vllm_tt_plugin.scheduler import (
    TT_SCHEDULER_OUTPUT_FIELDS,
    TTScheduler,
    carry_tt_scheduler_output_fields,
    set_tt_scheduler_output_field,
)


def _block_step():
    scheduler = _scheduler()
    scheduler.add_request(_request(8))
    return scheduler.schedule()


def _forced_reset_step():
    scheduler, request = _chunked_scheduler()
    intermediate = scheduler.schedule()
    scheduler.update_from_output(intermediate, _runner_output(intermediate, []))
    scheduler.schedule()
    scheduler.reset_prefix_cache(reset_running_requests=True)
    return scheduler.schedule()


def _lane_step():
    return _make_coordinator([FakeLane(running=1), FakeLane()]).schedule()


def _tt_fields(scheduler_output):
    return {name for name in vars(scheduler_output) if name.startswith("_tt_")}


@pytest.mark.parametrize("step", [_block_step, _forced_reset_step, _lane_step])
def test_the_tt_fields_a_step_sets_survive_a_rebuild_from_dataclass_fields(step):
    scheduler_output = step()
    attached = _tt_fields(scheduler_output)
    assert attached, "the step attaches TT fields"
    assert attached <= set(TT_SCHEDULER_OUTPUT_FIELDS)

    rebuilt = SchedulerOutput(
        **{
            field.name: getattr(scheduler_output, field.name)
            for field in fields(SchedulerOutput)
        }
    )
    assert not _tt_fields(rebuilt), "a rebuild from dataclass fields drops them"
    carry_tt_scheduler_output_fields(scheduler_output, rebuilt)

    assert {name: getattr(rebuilt, name) for name in _tt_fields(rebuilt)} == {
        name: getattr(scheduler_output, name) for name in attached
    }


def test_the_inventory_covers_every_tt_field_a_step_sets():
    seen = set().union(
        *(_tt_fields(step()) for step in (_block_step, _forced_reset_step, _lane_step))
    )

    assert seen == set(TT_SCHEDULER_OUTPUT_FIELDS)


def test_a_field_outside_the_inventory_is_refused():
    scheduler_output = SchedulerOutput.make_empty()

    with pytest.raises(ValueError, match="_tt_new_field"):
        set_tt_scheduler_output_field(scheduler_output, "_tt_new_field", 1)
    assert not _tt_fields(scheduler_output)


def _counting_release(monkeypatch):
    released = []
    release = TTScheduler.tt_release_output_frames

    def counting(self, scheduler_output):
        released.append(scheduler_output)
        release(self, scheduler_output)

    monkeypatch.setattr(TTScheduler, "tt_release_output_frames", counting)
    return released


def test_a_commit_releases_its_steps_frames_once(monkeypatch):
    released = _counting_release(monkeypatch)
    scheduler = _scheduler(output_width=1)
    request = _request(4)
    scheduler.add_request(request)
    step = scheduler.schedule()
    assert scheduler._pending_async_output_frames == {request.request_id: 1}

    scheduler.update_from_output(step, _runner_output(step, [7]))

    assert released == [step]
    assert scheduler._pending_async_output_frames == {}


def test_a_failed_commit_still_releases_its_steps_frames_once(monkeypatch):
    released = _counting_release(monkeypatch)
    scheduler = _scheduler(output_width=1)
    scheduler.add_request(_request(4))
    step = scheduler.schedule()

    def failing(self, scheduler_output, model_runner_output):
        raise RuntimeError("commit failed")

    monkeypatch.setattr(AsyncScheduler, "update_from_output", failing)
    with pytest.raises(RuntimeError, match="commit failed"):
        scheduler.update_from_output(step, _runner_output(step, [7]))

    assert released == [step]
    assert scheduler._pending_async_output_frames == {}
