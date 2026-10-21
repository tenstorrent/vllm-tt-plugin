# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The TT Omni AR scheduler under Omni's request handling, end to end on CPU.

A real ``TTOmniARScheduler`` runs inside a stand-in engine loop: ``schedule``, a
runner that samples one token for each row that completes its prompt or decodes,
then ``update_from_output``. ``depth`` is how many scheduled steps may be in flight
before the oldest commits: 1 is a synchronous engine, 2 is async scheduling's
batch queue. Each step's rows are classified from the request state before
``schedule``, not from anything TT records, so the checks do not trust the code
under test.
"""

import random
import threading
from collections import Counter, deque
from unittest.mock import patch

import pytest

pytest.importorskip("vllm_omni")

import torch  # noqa: E402
from vllm.config import (  # noqa: E402
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.device_communicators.shm_broadcast import (  # noqa: E402
    MessageQueue,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.executor.uniproc_executor import UniProcExecutor  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402
from vllm_omni.core.sched.output import (  # noqa: E402
    OmniNewRequestData,
    OmniSchedulerOutput,
)
from vllm_omni.request import OmniRequest  # noqa: E402

from tests.test_block_scheduler import (  # noqa: E402
    LOCAL_MODEL_CONFIG,
    _stub_model_resolution,
)
from vllm_tt_plugin.config import (  # noqa: E402
    resolve_tt_supports_stage_outputs,
    store_tt_output_tokens_per_step,
)
from vllm_tt_plugin.omni.model_runner import to_omni_output  # noqa: E402
from vllm_tt_plugin.omni.scheduler import TTOmniARScheduler  # noqa: E402
from vllm_tt_plugin.scheduler import (  # noqa: E402
    _TT_OUTPUT_FRAME_REQ_IDS_ATTR,
    get_tt_forced_reset_discard_counts,
)
from vllm_tt_plugin.stage_outputs import (  # noqa: E402
    TTStageFrame,
    set_tt_stage_outputs,
)

BLOCK_SIZE = 16
MAX_MODEL_LEN = 2048
# TT's default prefill-chunk alignment, so a split prompt ends on its grid.
CHUNK = 128
TOKEN = 7


class _ChunkedModel:
    model_capabilities = {"supports_chunked_prefill": True}


def _omni_scheduler(
    *,
    stage_outputs=True,
    max_num_seqs=1,
    chunked=False,
    num_blocks=None,
    async_scheduling=False,
):
    """A stage-output model, as the talker is, runs one sequence; without stage
    outputs the same scheduler batches, chunks prompts and interleaves."""
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=CHUNK if chunked else MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=chunked,
        async_scheduling=False,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    with (
        _stub_model_resolution(),
        patch(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            return_value=(_ChunkedModel, None),
        ),
    ):
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )
    config.scheduler_config.async_scheduling = async_scheduling
    store_tt_output_tokens_per_step(config, 1)
    if stage_outputs:
        resolve_tt_supports_stage_outputs(
            config, {"supports_stage_outputs": True}, object, False
        )
    if num_blocks is None:
        num_blocks = max_num_seqs * MAX_MODEL_LEN // BLOCK_SIZE + 1
    cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return TTOmniARScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(req_id, *, prompt_len=8, max_tokens=4, info=None):
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    return OmniRequest(
        request_id=req_id,
        prompt_token_ids=[1] * prompt_len,
        sampling_params=params,
        pooling_params=None,
        additional_information=info,
    )


class _ChunkAdapter:
    """Omni's chunk-transfer adapter, reduced to what its scheduler calls.

    A gated request waits for its first upstream chunk: it leaves the waiting queue
    before the step is scheduled and returns after, as ``OmniChunkTransferAdapter``
    parks it.
    """

    receives_chunks = True

    def __init__(self):
        self.gated: set[str] = set()
        self.parked = 0
        self.saved: list[tuple[str, dict]] = []
        self.segment_finished_requests: set[str] = set()
        self.requests_num_chunks_sent: dict[str, int] = {}
        self._waiting: list = []

    def process_pending_chunks(self, waiting, running, *, scheduler_requests=None):
        self._waiting = [r for r in waiting if r.request_id in self.gated]
        waiting.remove_requests(self._waiting)
        self.parked += len(self._waiting)

    def restore_queues(self, waiting, running, scheduler_requests=None):
        for request in self._waiting:
            waiting.add_request(request)
        self._waiting = []

    def postprocess_scheduler_output(self, scheduler_output, requests=None):
        pass

    def save_async(self, payload, request, is_segment_finished=False, **kwargs):
        if payload is not None:
            self.saved.append((request.request_id, payload))

    @staticmethod
    def _confirmed_num_computed_tokens(request):
        return request.num_computed_tokens

    def collect_timed_out_request_ids(self, timeout_s):
        return set()

    def collect_failed_send_request_ids(self):
        return {}

    def has_active_sender(self, external_req_id):
        return False

    def finish_requests(self, *args, **kwargs):
        pass

    def cleanup_receiver(self, request_id):
        pass

    def cleanup_sender(self, external_req_id):
        pass


def _codes(req_id):
    return [ord(req_id[-1])] * 16


class _Engine:
    def __init__(self, scheduler, depth=1):
        self.scheduler = scheduler
        self.depth = depth
        self.in_flight: deque = deque()
        self.phases: list[str] = []
        self.preempted_in_flight: list[str] = []

    def _expected_frames(self):
        return dict(
            Counter(
                req_id
                for _, rows in self.in_flight
                for req_id, (_, samples) in rows.items()
                if samples
            )
        )

    def schedule(self):
        scheduler = self.scheduler
        before = {
            req_id: (
                request.status == RequestStatus.RUNNING
                and not request.is_prefill_chunk,
                request.num_computed_tokens,
                request.num_tokens,
            )
            for req_id, request in scheduler.requests.items()
        }
        scheduler_output = scheduler.schedule()
        # A row decodes when its request was running with its prompt computed; a
        # row samples when this step reaches the end of what it has to compute.
        rows = {}
        for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            decodes, computed, num_tokens = before[req_id]
            rows[req_id] = (decodes, computed + scheduled >= num_tokens)
            if decodes:
                assert scheduled == 1, (req_id, scheduled)
        kinds = {decodes for decodes, _ in rows.values()}
        assert len(kinds) <= 1, f"mixed prefill and decode rows: {rows}"
        self.phases.append("-" if not rows else "D" if kinds == {True} else "P")
        for req_id in scheduler_output.preempted_req_ids or ():
            if any(req_id in r for _, r in self.in_flight):
                self.preempted_in_flight.append(req_id)
        if rows:
            self.in_flight.append((scheduler_output, rows))
        assert scheduler._pending_async_output_frames == self._expected_frames()
        return scheduler_output

    def commit(self, raises=None):
        scheduler_output, rows = self.in_flight.popleft()
        # The runner's row order need not follow the scheduler's.
        req_ids = list(rows)[::-1]
        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[TOKEN] if rows[r][1] else [] for r in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        set_tt_stage_outputs(
            output,
            {
                req_id: TTStageFrame(frame_index=0, data=_codes(req_id))
                for req_id, (decodes, _) in rows.items()
                if decodes
            },
        )
        result = self.scheduler.update_from_output(
            scheduler_output, to_omni_output(output)
        )
        assert self.scheduler._pending_async_output_frames == self._expected_frames()
        return result

    def step(self):
        scheduler_output = self.schedule()
        if self.in_flight and (
            len(self.in_flight) >= self.depth
            or not scheduler_output.num_scheduled_tokens
        ):
            return self.commit()
        return {}

    def drain(self, max_steps=2000):
        for _ in range(max_steps):
            if not (self.scheduler.has_unfinished_requests() or self.in_flight):
                return
            self.step()
        raise AssertionError("the engine did not drain")


# --- Batch purity -----------------------------------------------------------


_TALKER = {}
_BATCHED = {"stage_outputs": False, "max_num_seqs": 3, "chunked": True}


@pytest.mark.parametrize("config", [_TALKER, _BATCHED], ids=["talker", "batched"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("seed", range(4))
def test_batch_purity_under_arrivals_aborts_and_gated_requests(seed, depth, config):
    rng = random.Random(seed)
    scheduler = _omni_scheduler(num_blocks=30, async_scheduling=depth > 1, **config)
    adapter = _ChunkAdapter()
    scheduler.chunk_transfer_adapter = adapter
    engine = _Engine(scheduler, depth)
    aborted_in_flight = 0
    for step in range(300):
        if rng.random() < 0.3:
            scheduler.add_request(
                _request(
                    f"r{step}",
                    prompt_len=rng.randint(1, 400),
                    max_tokens=rng.randint(1, 24),
                )
            )
        live = sorted(scheduler.requests)
        if live and rng.random() < 0.05:
            # Half the aborts land on a request with a step in flight.
            in_flight = sorted({r for _, rows in engine.in_flight for r in rows})
            victim = rng.choice(in_flight if in_flight and rng.random() < 0.5 else live)
            aborted_in_flight += any(victim in rows for _, rows in engine.in_flight)
            scheduler.finish_requests(victim, RequestStatus.FINISHED_ABORTED)
        if live and rng.random() < 0.2:
            adapter.gated ^= {rng.choice(live)}
        engine.step()
    adapter.gated.clear()
    engine.drain()

    assert scheduler._pending_async_output_frames == {}
    assert {"P", "D"} <= set(engine.phases), "the run never mixed phases"
    assert adapter.parked, "no request was ever gated"
    if depth > 1:
        assert aborted_in_flight, "no abort hit a frame in flight"
        if config is _BATCHED:
            assert engine.preempted_in_flight, "no preemption hit a frame in flight"


# --- Omni behaviour preserved -----------------------------------------------


def test_requests_aborted_in_the_queues_are_dropped_before_scheduling():
    scheduler = _omni_scheduler()
    engine = _Engine(scheduler)
    scheduler.add_request(_request("a", max_tokens=8))
    engine.step()
    scheduler.add_request(_request("b"))
    # A client disconnect marks the request without removing it from its queue.
    for request in scheduler.requests.values():
        request.status = RequestStatus.FINISHED_ABORTED

    scheduler_output = engine.schedule()

    assert scheduler_output.num_scheduled_tokens == {}
    assert not scheduler.running
    assert not scheduler.waiting


def test_a_gated_request_waits_for_its_chunk_and_keeps_its_place():
    scheduler = _omni_scheduler()
    adapter = _ChunkAdapter()
    scheduler.chunk_transfer_adapter = adapter
    engine = _Engine(scheduler)
    scheduler.add_request(_request("a"))
    adapter.gated.add("a")

    assert engine.schedule().num_scheduled_tokens == {}
    assert [r.request_id for r in scheduler.waiting] == ["a"]

    adapter.gated.clear()
    assert list(engine.schedule().num_scheduled_tokens) == ["a"]


def test_new_requests_reach_the_runner_with_their_payload():
    scheduler = _omni_scheduler()
    info = {"text": "hello"}
    scheduler.add_request(_request("a", info=info))

    (new_req,) = _Engine(scheduler).schedule().scheduled_new_reqs

    assert isinstance(new_req, OmniNewRequestData)
    assert new_req.additional_information is info


# --- Exactly-once release ---------------------------------------------------


def _two_steps_in_flight(**kwargs):
    scheduler = _omni_scheduler(async_scheduling=True, **kwargs)
    engine = _Engine(scheduler, depth=2)
    scheduler.add_request(_request("a", max_tokens=16))
    engine.schedule()
    engine.schedule()
    assert scheduler._pending_async_output_frames == {"a": 2}
    return scheduler, engine


def test_frames_are_released_as_their_steps_commit():
    scheduler, engine = _two_steps_in_flight()

    engine.commit()
    assert scheduler._pending_async_output_frames == {"a": 1}
    engine.commit()
    assert scheduler._pending_async_output_frames == {}


def test_an_abort_keeps_the_frames_in_flight_until_their_steps_commit():
    scheduler, engine = _two_steps_in_flight()

    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert scheduler._pending_async_output_frames == {"a": 2}

    engine.commit()
    engine.commit()
    assert scheduler._pending_async_output_frames == {}
    assert engine.schedule().num_scheduled_tokens == {}


def test_a_forced_reset_publishes_its_discards_and_releases_each_frame_once():
    scheduler, engine = _two_steps_in_flight(stage_outputs=False)

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = engine.schedule()

    assert isinstance(resumed, OmniSchedulerOutput)
    assert get_tt_forced_reset_discard_counts(resumed) == {"a": 2}
    engine.commit()
    engine.commit()
    engine.commit()
    assert scheduler._pending_async_output_frames == {}


def test_a_stage_output_request_refuses_a_forced_reset_and_keeps_its_frames():
    scheduler, engine = _two_steps_in_flight()

    with pytest.raises(RuntimeError, match="cannot resume"):
        scheduler.reset_prefix_cache(reset_running_requests=True)

    assert scheduler._pending_async_output_frames == {"a": 2}
    engine.commit()
    engine.commit()
    assert scheduler._pending_async_output_frames == {}


def test_a_preemption_keeps_the_frames_in_flight_until_their_steps_commit():
    # Room for two prompts but not for both to decode past their first block.
    scheduler = _omni_scheduler(
        stage_outputs=False, max_num_seqs=2, async_scheduling=True, num_blocks=4
    )
    engine = _Engine(scheduler, depth=2)
    for req_id in ("a", "b"):
        scheduler.add_request(_request(req_id, prompt_len=BLOCK_SIZE, max_tokens=32))

    engine.drain()

    assert engine.preempted_in_flight
    assert scheduler._pending_async_output_frames == {}


# The release after an exception inside Omni's update_from_output is
# tests/omni/test_scheduler.py::test_the_frames_are_released_when_omnis_update_raises.


# --- Interleave bounds ------------------------------------------------------


def test_a_long_prompt_yields_decode_steps_at_the_default_cadence():
    scheduler = _omni_scheduler(stage_outputs=False, max_num_seqs=2, chunked=True)
    engine = _Engine(scheduler)
    scheduler.add_request(_request("decode", max_tokens=64))
    engine.step()
    engine.step()
    assert engine.phases[-1] == "D"

    scheduler.add_request(_request("long", prompt_len=8 * CHUNK, max_tokens=4))
    long = scheduler.requests["long"]
    start = len(engine.phases)
    while long.num_computed_tokens < long.num_prompt_tokens:
        engine.step()

    # Two prefill steps, then one decode step, until the prompt is in.
    assert "".join(engine.phases[start:]) == "PPDPPDPPDPP"


# --- TT fields across Omni's wrap and both executors ------------------------


class _RecordingWorker:
    def __init__(self):
        self.received = []

    def execute_model(self, scheduler_output):
        self.received.append(scheduler_output)


def _through_uniproc(scheduler_output):
    executor = object.__new__(UniProcExecutor)
    executor.driver_worker = _RecordingWorker()
    executor.execute_model(scheduler_output)
    (received,) = executor.driver_worker.received
    return received


def _through_multiproc(scheduler_output):
    # The shared-memory queue MultiprocExecutor broadcasts each step through.
    writer = MessageQueue(n_reader=1, n_local_reader=1)
    reader = MessageQueue.create_from_handle(writer.export_handle(), 0)
    ready = threading.Thread(target=reader.wait_until_ready)
    ready.start()
    writer.wait_until_ready()
    ready.join()
    try:
        writer.enqueue(("execute_model", (scheduler_output,), {}, 0))
        _, (received,), _, _ = reader.dequeue()
    finally:
        reader.shutdown()
        writer.shutdown()
    return received


@pytest.mark.parametrize("send", [_through_uniproc, _through_multiproc])
def test_tt_fields_reach_the_worker_through_either_executor(send):
    scheduler, engine = _two_steps_in_flight(stage_outputs=False)
    scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = engine.schedule()

    received = send(resumed)

    assert isinstance(received, OmniSchedulerOutput)
    assert getattr(received, _TT_OUTPUT_FRAME_REQ_IDS_ATTR) == ("a",)
    assert get_tt_forced_reset_discard_counts(received) == {"a": 2}


# --- Side-output alignment --------------------------------------------------


def test_each_request_gets_its_own_frame():
    # A stage-output model runs one sequence today; the rows are aligned by request
    # id, which this checks on a two-row step.
    scheduler = _omni_scheduler(stage_outputs=False, max_num_seqs=2)
    adapter = _ChunkAdapter()
    scheduler.chunk_transfer_adapter = adapter
    engine = _Engine(scheduler)
    for req_id in ("a", "b"):
        scheduler.add_request(_request(req_id, max_tokens=8))
    engine.step()
    adapter.saved.clear()

    outputs = engine.step()
    assert engine.phases[-1] == "D"

    emitted = {
        output.request_id: output.multimodal_output["codes.audio"].tolist()
        for client in outputs.values()
        for output in client.outputs
    }
    assert emitted == {"a": [_codes("a")], "b": [_codes("b")]}
    saved = {
        req_id: payload["codes.audio"].tolist() for req_id, payload in adapter.saved
    }
    assert saved == emitted
