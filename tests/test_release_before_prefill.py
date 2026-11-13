# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A prefill's tokens reach the client before the next step runs, and a
decode's tokens before a prefill runs.

Under async scheduling vLLM's ``EngineCore.step_with_batch_queue`` commits a
step's output only after it has scheduled and executed the next step, and
``TTModelRunner`` runs a prefill to the end inside ``execute_model``. The
engine tests here drive the real ``TTScheduler`` through the real
``step_with_batch_queue`` with an executor that, like the TT runner, finishes
every step inside ``execute_model``, and record the order in which steps
execute and tokens reach ``update_from_output``.
"""

import threading
from collections import Counter, deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import RequestStatus

from tests.test_block_scheduler import (
    BLOCK_SIZE,
    MAX_MODEL_LEN,
    _request,
    _runner_output,
)
from tests.test_block_scheduler import _scheduler as _real_scheduler
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import (
    get_tt_forced_reset_discard_counts,
    scheduler_output_has_prefill_work,
)

TOKEN = 7


def _done(value) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


def _runner(max_num_reqs: int) -> TTModelRunner:
    """The request bookkeeping of ``TTModelRunner``, without a model.

    ``_update_states`` and ``TTAsyncDecodeController`` are the real ones: they
    hold each request's state, and apply each step's sampled tokens at the
    next step's build, after forced-reset discard counts and finished requests
    are known.
    """
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.tt_per_lane_max_num_seqs = max_num_reqs
    runner._req_state_slot = {}
    runner._pending_state_slot_settle = None
    runner.requests = {}
    runner.encoder_cache = {}
    runner.supports_stage_inputs = False
    runner.supports_stage_outputs = False
    runner._decode_layout_changed_since_last_decode = False
    runner.model = SimpleNamespace(release_request=lambda slot: None)
    runner.input_batch = InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=64,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    runner.model_config = SimpleNamespace(max_model_len=MAX_MODEL_LEN)
    runner._output_tokens_per_step = 1
    runner._is_adaptive_block_output = False
    runner._num_speculative_tokens = 0
    runner._steady_decode_lock = threading.Lock()
    runner._pending_async_steps = deque()
    runner._pending_async_overlap_ok = deque()
    runner._completed_decode_steps = deque()
    runner.async_decode = TTAsyncDecodeController(runner)
    return runner


class _SynchronousExecutor:
    """Finishes each step inside ``execute_model``, as ``TTModelRunner`` does
    for a prefill, over the runner's real request bookkeeping, and logs it.

    Logs ``("prefill" | "decode" | "empty", req_ids)`` for each step,
    ``("discard", (req_id,))`` for forced-reset discard counts a step brings,
    and ``("resume", (req_id,))`` for a request a step resumes. A step is a
    prefill when ``TTModelRunner`` runs it as one. ``applied`` counts the
    tokens the runner adds to each request.
    """

    def __init__(self, log: list, max_num_reqs: int) -> None:
        self.log = log
        self.runner = _runner(max_num_reqs)
        self.applied: Counter[str] = Counter()
        self._executed: tuple[str, ...] = ()
        self._intermediate: np.ndarray | None = None
        apply = self.runner._apply_sampled_tokens_to_state

        def count_applied(sampled_token_ids, req_ids=None, skip_req_ids=None):
            for req_id in req_ids:
                if req_id not in (skip_req_ids or ()):
                    self.applied[req_id] += 1
            apply(sampled_token_ids, req_ids=req_ids, skip_req_ids=skip_req_ids)

        self.runner._apply_sampled_tokens_to_state = count_applied

    def execute_model(self, scheduler_output, non_block=False):
        runner = self.runner
        cached = scheduler_output.scheduled_cached_reqs
        unknown = set(cached.req_ids) - set(runner.requests)
        assert not unknown, f"the runner never saw requests {sorted(unknown)}"
        for req_id in get_tt_forced_reset_discard_counts(scheduler_output):
            self.log.append(("discard", (req_id,)))
        # What TTModelRunner.build_model_input does before it builds inputs.
        runner._update_states(scheduler_output)
        runner.async_decode.apply_completed_decode_steps_before_build(scheduler_output)
        for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
            if req_id in cached.resumed_req_ids:
                self.log.append(("resume", (req_id,)))
                scheduled = scheduler_output.num_scheduled_tokens[req_id]
                assert runner.requests[req_id].num_tokens == computed + scheduled, (
                    f"the runner holds {runner.requests[req_id].num_tokens} tokens "
                    f"of {req_id}; the scheduler resumes it with "
                    f"{computed + scheduled}"
                )
        req_ids = tuple(scheduler_output.num_scheduled_tokens)
        if not req_ids:
            self.log.append(("empty", ()))
            return _done(EMPTY_MODEL_RUNNER_OUTPUT)
        is_prefill = scheduler_output_has_prefill_work(scheduler_output)
        self.log.append(("prefill" if is_prefill else "decode", req_ids))
        self._executed = req_ids
        # As TTModelRunner marks a prefill row that ends before its last token.
        self._intermediate = (
            np.array(
                [
                    runner.requests[r].num_computed_tokens
                    + scheduler_output.num_scheduled_tokens[r]
                    < runner.requests[r].num_tokens
                    for r in req_ids
                ]
            )
            if is_prefill
            else None
        )
        return _done(None)

    def sample_tokens(self, grammar_output, non_block=False):
        req_ids = list(self._executed)
        sampled = torch.full((len(req_ids), 1), TOKEN, dtype=torch.int32)
        if self._intermediate is not None and self._intermediate.any():
            output = self.runner._build_chunked_prefill_output(
                req_ids=req_ids,
                sampled_token_ids=sampled,
                logprobs=None,
                intermediate_mask=self._intermediate,
                defer_state_apply=True,
            )
        else:
            output = self.runner.defer_state_apply_and_build_runner_output(
                sampled, req_ids=req_ids
            )
        return _done(output)


def _engine(scheduler, log: list) -> SimpleNamespace:
    """What ``step_with_batch_queue`` and ``pause_scheduler`` read off
    ``EngineCore``, with the queue depth the engine derives from the
    scheduler's config."""
    real_update = scheduler.update_from_output
    released: Counter[str] = Counter()

    def update_from_output(scheduler_output, model_runner_output):
        outputs = real_update(scheduler_output, model_runner_output)
        for engine_outputs in outputs.values():
            for output in engine_outputs.outputs:
                if output.new_token_ids:
                    released[output.request_id] += len(output.new_token_ids)
                    log.append(("release", (output.request_id,)))
        return outputs

    scheduler.update_from_output = update_from_output
    queue_size = scheduler.vllm_config.max_concurrent_batches

    def finish_pause(clear_cache: bool) -> None:
        # EngineCore._reset_caches, which a pause with clear_cache runs.
        if clear_cache and not scheduler.reset_prefix_cache(
            reset_running_requests=True, reset_connector=True
        ):
            raise RuntimeError("reset_prefix_cache refused")

    return SimpleNamespace(
        scheduler=scheduler,
        model_executor=_SynchronousExecutor(
            log, scheduler.scheduler_config.max_num_seqs
        ),
        released=released,
        batch_queue=deque(maxlen=queue_size),
        batch_queue_size=queue_size,
        is_mm_encoder_only=False,
        is_pooling_model=False,
        check_for_draft_tokens=False,
        _should_throttle_prefills=lambda: False,
        log_error_detail=lambda scheduler_output: nullcontext(),
        capture_iteration_details=lambda scheduler_output: nullcontext(),
        _process_aborts_queue=lambda: None,
        _attach_iteration_details=lambda outputs, details: None,
        _finish_pause=finish_pause,
    )


def _serve(scheduler, arrivals: dict[int, list], between=None) -> SimpleNamespace:
    """Run the engine loop until every request finishes.

    ``arrivals`` maps an engine iteration to the requests added before it, and
    ``between(engine)`` runs before every iteration, as
    ``EngineCoreProc._process_input_queue`` handles client calls there.
    Returns the engine; ``engine.log`` is the log.
    """
    log: list = []
    engine = _engine(scheduler, log)
    engine.log = log
    assert engine.batch_queue_size == 2
    for iteration in range(100):
        for request in arrivals.get(iteration, ()):
            scheduler.add_request(request)
        if between is not None:
            between(engine)
        if scheduler.has_requests() or engine.batch_queue:
            EngineCore.step_with_batch_queue(engine)
        elif iteration > max(arrivals) and scheduler.pause_state == (
            PauseState.UNPAUSED
        ):
            return engine
    raise AssertionError(f"requests did not finish: {log}")


def _held_at_prefills(log: list) -> list[dict[str, int]]:
    """For each prefill, the tokens executed but not yet released, by request."""
    pending: dict[str, int] = {}
    held = []
    for kind, req_ids in log:
        if kind == "prefill":
            held.append({r: n for r, n in pending.items() if n})
        if kind in ("prefill", "decode"):
            for req_id in req_ids:
                pending[req_id] = pending.get(req_id, 0) + 1
        elif kind == "release":
            pending[req_ids[0]] -= 1
    return held


def _async_scheduler(max_num_seqs: int = 2):
    return _real_scheduler(
        output_width=1, max_num_seqs=max_num_seqs, async_scheduling=True
    )


def test_a_prefill_runs_after_the_previous_prefill_is_released():
    # A is prefilled alone; B arrives while A's prefill is still in the queue.
    log = _serve(
        _async_scheduler(),
        {0: [_request(3, request_id="A")], 1: [_request(3, request_id="B")]},
    ).log

    prefill_b = log.index(("prefill", ("B",)))
    assert log.index(("release", ("A",))) < prefill_b
    assert _held_at_prefills(log) == [{}, {}]
    assert log[:prefill_b].count(("empty", ())) == 1


def test_a_prefill_runs_after_the_previous_decode_is_released():
    # B arrives while A decodes, so the step before B's prefill is A's decode.
    log = _serve(
        _async_scheduler(),
        {0: [_request(6, request_id="A")], 3: [_request(3, request_id="B")]},
    ).log

    prefill_b = log.index(("prefill", ("B",)))
    assert log[:prefill_b].count(("decode", ("A",))) >= 1
    assert _held_at_prefills(log) == [{}, {}]


def test_no_empty_step_while_a_waiting_request_cannot_be_admitted():
    # One slot: B waits for the whole of A's decode, and every prefill pass in
    # that time schedules nothing. On TT an empty step drains the outstanding
    # decode and takes every row out of the persistent batch, so empty steps
    # between A's decodes would end their overlap.
    log = _serve(
        _async_scheduler(max_num_seqs=1),
        {0: [_request(5, request_id="A")], 1: [_request(3, request_id="B")]},
    ).log

    decodes_of_a = [i for i, entry in enumerate(log) if entry == ("decode", ("A",))]
    assert len(decodes_of_a) == 4
    between = log[decodes_of_a[0] : decodes_of_a[-1]]
    assert ("empty", ()) not in between
    assert _held_at_prefills(log) == [{}, {}]


def test_synchronous_scheduling_returns_the_prefill():
    # As with a batch queue under pipeline parallelism: A's prefill is not
    # committed yet when B's is scheduled.
    scheduler = _real_scheduler(output_width=1, max_num_seqs=2)
    scheduler.add_request(_request(3, request_id="A"))
    scheduler.schedule()
    scheduler.add_request(_request(3, request_id="B"))

    step = scheduler.schedule()

    assert set(step.num_scheduled_tokens) == {"B"}


def test_no_empty_step_after_a_committed_step():
    scheduler = _async_scheduler()
    scheduler.add_request(_request(3, request_id="A"))
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _runner_output(prefill, [TOKEN]))
    scheduler.add_request(_request(3, request_id="B"))

    step = scheduler.schedule()

    assert set(step.num_scheduled_tokens) == {"B"}


def test_the_prefill_follows_one_empty_step():
    scheduler = _async_scheduler()
    scheduler.add_request(_request(3, request_id="A"))
    scheduler.schedule()
    scheduler.add_request(_request(3, request_id="B"))

    empty = scheduler.schedule()
    # Whether or not A's prefill has been committed by now.
    prefill = scheduler.schedule()

    assert empty.total_num_scheduled_tokens == 0
    assert set(prefill.num_scheduled_tokens) == {"B"}


def test_the_empty_step_carries_cleanup_ahead_of_the_prefill():
    scheduler = _real_scheduler(output_width=1, max_num_seqs=3, async_scheduling=True)
    scheduler.add_request(_request(3, request_id="A"))
    scheduler.add_request(_request(3, request_id="X"))
    scheduler.schedule()
    scheduler.finish_requests("X", RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(_request(3, request_id="B"))

    empty = scheduler.schedule()
    prefill = scheduler.schedule()

    assert empty.total_num_scheduled_tokens == 0
    assert empty.finished_req_ids == {"X"}
    assert set(prefill.num_scheduled_tokens) == {"B"}
    assert prefill.finished_req_ids == set()


def test_cleanup_reaches_the_decode_when_no_prefill_is_admitted():
    # One slot, held by A: the empty step's pass drains C's abort, the prefill
    # pass admits nothing, and the call returns A's decode instead.
    scheduler = _async_scheduler(max_num_seqs=1)
    scheduler.add_request(_request(8, request_id="A"))
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _runner_output(prefill, [TOKEN]))
    scheduler.add_request(_request(3, request_id="B"))
    scheduler.add_request(_request(3, request_id="C"))
    scheduler.schedule()
    scheduler.finish_requests("C", RequestStatus.FINISHED_ABORTED)

    step = scheduler.schedule()

    assert set(step.num_scheduled_tokens) == {"A"}
    assert step.finished_req_ids == {"C"}


def _held_from_prefills(log: list) -> list[dict[str, int]]:
    """For each prefill or decode step, the tokens that earlier prefill steps
    computed and that are not released yet, by request."""
    pending: Counter[str] = Counter()
    held = []
    for kind, req_ids in log:
        if kind in ("prefill", "decode"):
            held.append({r: n for r, n in pending.items() if n})
        if kind == "prefill":
            pending.update(req_ids)
        elif kind == "release" and pending[req_ids[0]]:
            pending[req_ids[0]] -= 1
    return held


def test_a_prefill_s_tokens_are_released_before_the_next_step_runs():
    # A's prefill is followed by A's decodes; B's by a decode of both.
    log = _serve(
        _async_scheduler(),
        {0: [_request(6, request_id="A")], 3: [_request(3, request_id="B")]},
    ).log

    assert log.index(("release", ("A",))) < log.index(("decode", ("A",)))
    assert not any(_held_from_prefills(log))


def test_no_empty_step_between_two_decodes():
    log = _serve(_async_scheduler(), {0: [_request(6, request_id="A")]}).log

    decodes = [i for i, (kind, _) in enumerate(log) if kind == "decode"]
    assert len(decodes) == 5
    assert ("empty", ()) not in log[decodes[0] : decodes[-1]]


def test_only_the_last_chunk_of_a_prompt_is_followed_by_an_empty_step():
    # A 200-token prompt at 64 tokens per step: three chunks that produce no
    # token, then the last one, which does.
    scheduler = _async_scheduler()
    scheduler.scheduler_config.enable_chunked_prefill = True
    scheduler.max_num_scheduled_tokens = 64
    log = _serve(scheduler, {0: [_request(3, request_id="A", prompt_length=200)]}).log

    executed = [kind for kind, _ in log if kind in ("prefill", "decode", "empty")]
    assert executed[:6] == ["prefill"] * 4 + ["empty", "decode"]
    assert log.index(("release", ("A",))) < log.index(("decode", ("A",)))


# A prefill kept by one schedule() call reaches the runner on the next call.
# Between the two, the engine commits the held step and handles client calls.


def _when_a_prefill_is_kept(action):
    """``between`` hook: run ``action(engine)`` once, between the call that
    keeps a prefill and the call that returns it."""
    ran: list[bool] = []

    def between(engine) -> None:
        if not ran and engine.scheduler._deferred_prefill is not None:
            ran.append(True)
            action(engine)

    return between, ran


def _no_unreleased_token_applied(engine) -> None:
    """The runner added no token to a request that the client did not get."""
    for req_id, count in engine.model_executor.applied.items():
        assert count <= engine.released[req_id], (
            f"the runner applied {count} tokens to {req_id}; the client got "
            f"{engine.released[req_id]}"
        )


def _prefill_after_a_decode():
    """B arrives while A decodes, so B's prefill is kept behind A's decode."""
    return {0: [_request(6, request_id="A")], 3: [_request(3, request_id="B")]}


def test_a_reset_while_a_prefill_is_kept_discards_its_frame_after_it_runs():
    # The reset preempts B, whose kept prefill reserved a frame. The prefill
    # still runs; the discard count for its frame must come after it.
    def reset(engine) -> None:
        assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)

    between, ran = _when_a_prefill_is_kept(reset)
    engine = _serve(_async_scheduler(), _prefill_after_a_decode(), between)
    log = engine.log

    assert ran
    assert log.index(("prefill", ("B",))) < log.index(("discard", ("B",)))
    assert log.index(("discard", ("B",))) < log.index(("resume", ("B",)))
    assert engine.released == {"A": 6, "B": 3}
    _no_unreleased_token_applied(engine)


@pytest.mark.parametrize("pause", [False, True], ids=["abort", "pause-abort"])
def test_an_abort_while_a_prefill_is_kept(pause):
    released_a: list[int] = []

    def abort(engine) -> None:
        released_a.append(engine.released["A"])
        if pause:
            EngineCore.pause_scheduler(engine, mode="abort", clear_cache=False)
            EngineCore.resume_scheduler(engine)
        else:
            engine.scheduler.finish_requests("B", RequestStatus.FINISHED_ABORTED)

    between, ran = _when_a_prefill_is_kept(abort)
    engine = _serve(_async_scheduler(), _prefill_after_a_decode(), between)

    assert ran
    assert engine.released["B"] == 0
    assert engine.released["A"] == (released_a[0] if pause else 6)
    assert not engine.model_executor.runner.requests
    _no_unreleased_token_applied(engine)


@pytest.mark.parametrize("clear_cache", [False, True])
@pytest.mark.parametrize("abort_a", [False, True], ids=["", "abort-A"])
def test_a_keep_pause_while_a_prefill_is_kept(clear_cache, abort_a):
    # Paused from the window for three engine iterations. An abort while
    # paused makes the engine step, and so call schedule(), during the pause.
    paused = {"iterations": -1, "released_a": 0}

    def between(engine) -> None:
        if paused["iterations"] < 0:
            if engine.scheduler._deferred_prefill is not None:
                paused["iterations"] = 0
                engine.log.append(("pause", ()))
                EngineCore.pause_scheduler(engine, mode="keep", clear_cache=clear_cache)
            return
        paused["iterations"] += 1
        if paused["iterations"] == 1 and abort_a:
            paused["released_a"] = engine.released["A"]
            engine.scheduler.finish_requests("A", RequestStatus.FINISHED_ABORTED)
        if paused["iterations"] == 3:
            engine.log.append(("unpause", ()))
            EngineCore.resume_scheduler(engine)

    engine = _serve(_async_scheduler(), _prefill_after_a_decode(), between)
    log = engine.log

    pause, unpause = log.index(("pause", ())), log.index(("unpause", ()))
    assert not [entry for entry in log[pause:unpause] if entry[0] == "prefill"]
    assert log.index(("prefill", ("B",))) > unpause
    if clear_cache:
        assert log.index(("discard", ("B",))) < log.index(("resume", ("B",)))
    assert engine.released == {"A": paused["released_a"] if abort_a else 6, "B": 3}
    _no_unreleased_token_applied(engine)
