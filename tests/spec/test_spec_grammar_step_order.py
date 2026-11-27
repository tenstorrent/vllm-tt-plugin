# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The vLLM call order a token-masked chained verify relies on.

A chained verify n + 1 is submitted at ``execute_model(n + 1)``, before the
runner reads verify n, and samples under its grammar bitmask at
``sample_tokens(n + 1)``. That bitmask must already follow verify n's commit:
column 0 of a structured row is the grammar after the tokens n committed.
vLLM 0.33 gives that order under asynchronous scheduling:

1. ``AsyncScheduler._update_after_schedule`` marks a step that schedules a
   decoding structured request whose previous output is still in flight
   (``pending_structured_output_tokens``);
2. ``EngineCore.step_with_batch_queue`` then defers that step's
   ``sample_tokens``: it resolves step n, applies it
   (``update_from_output``, which advances each grammar), takes the drafts back
   (``take_draft_token_ids``), and only then computes step n + 1's bitmask and
   calls ``sample_tokens(n + 1)``.

These tests run vLLM's own code for both, over a stand-in scheduler and
executor that record the calls, so a vLLM upgrade that changes either part
fails here before a masked chain commits a token its grammar forbids.
"""

from __future__ import annotations

import queue
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore


class _Recorder:
    """The stand-in scheduler and executor of one engine; ``calls`` in order."""

    def __init__(self, steps: int, structured_from: int) -> None:
        self.calls: list[tuple[str, int]] = []
        self.steps = steps
        self.structured_from = structured_from
        self.scheduled = 0
        self.bitmask_step: dict[int, int] = {}

    # Scheduler
    def has_requests(self) -> bool:
        return self.scheduled < self.steps

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        del throttle_prefills
        self.scheduled += 1
        step = self.scheduled
        output = SchedulerOutput.make_empty()
        output.total_num_scheduled_tokens = 1
        output.num_scheduled_tokens = {"j": 1}
        output.pending_structured_output_tokens = step >= self.structured_from
        output.step = step
        self.calls.append(("schedule", step))
        return output

    def get_grammar_bitmask(self, scheduler_output):
        self.calls.append(("bitmask", scheduler_output.step))
        return SimpleNamespace(step=scheduler_output.step)

    def update_from_output(self, scheduler_output, model_output):
        if model_output.step != scheduler_output.step:
            raise AssertionError("an output was applied to another step")
        self.calls.append(("update", scheduler_output.step))
        return {}

    def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output):
        self.calls.append(("validate_drafts", scheduler_output.step))

    # Executor
    def execute_model(self, scheduler_output, non_block: bool = False):
        self.calls.append(("execute", scheduler_output.step))
        self._executed = scheduler_output.step
        future: Future = Future()
        future.set_result(None)
        return future

    def sample_tokens(self, grammar_output, non_block: bool = False):
        step = grammar_output.step
        self.calls.append(("sample", step))
        future: Future = Future()
        future.set_result(SimpleNamespace(step=step))
        return future

    def take_draft_token_ids(self):
        self.calls.append(("take_drafts", self._executed))
        return SimpleNamespace(req_ids=["j"], draft_token_ids=[[1, 2, 3]])


def _engine(recorder: _Recorder) -> EngineCore:
    """An ``EngineCore`` whose batch queue is two deep, as async scheduling makes it."""
    engine = object.__new__(EngineCore)
    engine.batch_queue = deque()
    engine.batch_queue_size = 2
    engine.scheduler = recorder
    engine.model_executor = recorder
    engine.is_mm_encoder_only = False
    engine.is_pooling_model = False
    engine.check_for_draft_tokens = True
    engine.aborts_queue = queue.Queue()
    engine.log_stats = False
    engine._weight_version = None
    engine.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_logging_iteration_details=False)
    )
    return engine


def test_a_structured_step_samples_after_the_previous_step_is_applied():
    """For every step n + 1 that holds a decoding structured request: its
    forward runs before step n is applied, its bitmask is computed after, and
    its sample_tokens comes after its own execute_model and after step n's
    sample_tokens (the executor's queue order)."""
    recorder = _Recorder(steps=6, structured_from=2)
    engine = _engine(recorder)
    while recorder.has_requests() or engine.batch_queue:
        engine.step_with_batch_queue()
    calls = recorder.calls
    at = {call: index for index, call in enumerate(calls)}
    for step in range(2, 7):
        assert at[("execute", step)] < at[("update", step - 1)], step
        assert at[("update", step - 1)] < at[("take_drafts", step)], step
        assert at[("take_drafts", step)] < at[("validate_drafts", step)], step
        assert at[("validate_drafts", step)] < at[("bitmask", step)], step
        assert at[("bitmask", step)] < at[("sample", step)], step
        assert at[("sample", step - 1)] < at[("execute", step)], step
    # The step before any structured request samples right after its forward.
    assert calls[:4] == [("schedule", 1), ("execute", 1), ("bitmask", 1), ("sample", 1)]


def test_the_async_scheduler_marks_a_step_whose_structured_output_is_in_flight(
    monkeypatch,
):
    """``pending_structured_output_tokens`` is set exactly for a structured
    request that is past its prefill and still has an unapplied output."""
    monkeypatch.setattr(Scheduler, "_update_after_schedule", lambda self, output: None)

    def request(structured: bool, placeholders: int, prefill: bool = False):
        return SimpleNamespace(
            is_prefill_chunk=prefill,
            use_structured_output=structured,
            num_output_placeholders=placeholders,
            spec_token_ids=[],
        )

    cases = {
        "in-flight": (request(True, 1), True),
        "applied": (request(True, 0), False),
        "plain": (request(False, 4), False),
        "prefill-chunk": (request(True, 1, prefill=True), False),
    }
    for name, (req, expected) in cases.items():
        scheduler = object.__new__(AsyncScheduler)
        scheduler.requests = {"r": req}
        scheduler.num_sampled_tokens_per_step = 1
        scheduler.use_v2_model_runner = False
        output = SchedulerOutput.make_empty()
        output.num_scheduled_tokens = {"r": 1}
        output.scheduled_spec_decode_tokens = {}
        output.num_spec_tokens_to_schedule = 3
        AsyncScheduler._update_after_schedule(scheduler, output)
        assert output.pending_structured_output_tokens is expected, name
