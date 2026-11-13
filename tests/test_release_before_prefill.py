# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A step's output tokens reach the client before a prefill runs.

Under async scheduling vLLM's ``EngineCore.step_with_batch_queue`` commits a
step's output only after it has scheduled and executed the next step, and
``TTModelRunner`` runs a prefill to the end inside ``execute_model``. The
engine tests here drive the real ``TTScheduler`` through the real
``step_with_batch_queue`` with an executor that, like the TT runner, finishes
every step inside ``execute_model``, and record the order in which steps
execute and tokens reach ``update_from_output``.
"""

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from tests.test_block_scheduler import _request, _runner_output
from tests.test_block_scheduler import _scheduler as _real_scheduler

TOKEN = 7


def _sampled(req_ids) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=list(req_ids),
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[TOKEN] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _done(value) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


class _SynchronousExecutor:
    """Finishes each step inside ``execute_model`` and logs it.

    Logs ``("prefill" | "decode" | "empty", req_ids)``. Chunked prefill is off
    in these tests, so a step with a new request is a whole prefill.
    """

    def __init__(self, log: list) -> None:
        self.log = log
        self._executed: tuple[str, ...] = ()

    def execute_model(self, scheduler_output, non_block=False):
        req_ids = tuple(scheduler_output.num_scheduled_tokens)
        if not req_ids:
            self.log.append(("empty", ()))
            return _done(EMPTY_MODEL_RUNNER_OUTPUT)
        kind = "prefill" if scheduler_output.scheduled_new_reqs else "decode"
        self.log.append((kind, req_ids))
        self._executed = req_ids
        return _done(None)

    def sample_tokens(self, grammar_output, non_block=False):
        return _done(_sampled(self._executed))


def _engine(scheduler, log: list) -> SimpleNamespace:
    """What ``step_with_batch_queue`` reads off ``EngineCore``, with the queue
    depth the engine derives from the scheduler's config."""
    real_update = scheduler.update_from_output

    def update_from_output(scheduler_output, model_runner_output):
        outputs = real_update(scheduler_output, model_runner_output)
        for engine_outputs in outputs.values():
            for output in engine_outputs.outputs:
                if output.new_token_ids:
                    log.append(("release", (output.request_id,)))
        return outputs

    scheduler.update_from_output = update_from_output
    queue_size = scheduler.vllm_config.max_concurrent_batches
    return SimpleNamespace(
        scheduler=scheduler,
        model_executor=_SynchronousExecutor(log),
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
    )


def _serve(scheduler, arrivals: dict[int, list]) -> list:
    """Run the engine loop until every request finishes; ``arrivals`` maps an
    engine iteration to the requests added before it, as
    ``EngineCoreProc._process_input_queue`` adds them."""
    log: list = []
    engine = _engine(scheduler, log)
    assert engine.batch_queue_size == 2
    for iteration in range(100):
        for request in arrivals.get(iteration, ()):
            scheduler.add_request(request)
        if scheduler.has_requests() or engine.batch_queue:
            EngineCore.step_with_batch_queue(engine)
        elif iteration > max(arrivals):
            return log
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
    )

    prefill_b = log.index(("prefill", ("B",)))
    assert log.index(("release", ("A",))) < prefill_b
    assert _held_at_prefills(log) == [{}, {}]
    assert log[:prefill_b].count(("empty", ())) == 1


def test_a_prefill_runs_after_the_previous_decode_is_released():
    # B arrives while A decodes, so the step before B's prefill is A's decode.
    log = _serve(
        _async_scheduler(),
        {0: [_request(6, request_id="A")], 3: [_request(3, request_id="B")]},
    )

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
    )

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
