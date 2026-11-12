# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Admission of structured-output requests across lanes.

A request with a structured-output grammar waits in its lane with status
``WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`` until the grammar compiles, and vLLM's
prefill pass skips it until then. These tests drive a lane coordinator over real
per-lane ``TTScheduler`` instances through a burst of requests and check that
grammar requests spread over the lanes and run as concurrently as unconstrained
ones. On vLLM 0.30 such a request sat in ``skipped_waiting``, which the lane
picker did not count, so a burst of them was bound to lane 0.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.config import store_tt_output_tokens_per_step
from vllm_tt_plugin.lane_scheduler import TTLaneCoordinator
from vllm_tt_plugin.scheduler import TTDecodeInterleavePolicy, TTScheduler

LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"
BLOCK_SIZE = 64
NUM_LANES = 4
PER_LANE_MAX = 4
NUM_REQUESTS = NUM_LANES * PER_LANE_MAX
PROMPT_LEN = 32
MAX_TOKENS = 20


class _StubModel:
    pass


class _FakeGrammar:
    """Accepts every token and never terminates; the scheduler only needs these."""

    def accept_tokens(self, request_id, tokens):
        return True

    def validate_tokens(self, tokens):
        return tokens

    def is_terminated(self):
        return False

    def rollback(self, num_tokens):
        pass

    def reset(self):
        pass


def _lane_scheduler() -> TTScheduler:
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    max_model_len = 256
    model_config.max_model_len = max_model_len
    scheduler_config = SchedulerConfig(
        max_num_seqs=PER_LANE_MAX,
        max_num_batched_tokens=4096,
        max_model_len=max_model_len,
        enable_chunked_prefill=False,
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
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch(
            "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            return_value=(_StubModel, None),
        ),
    ):
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )
    store_tt_output_tokens_per_step(config, 1)
    num_blocks = 4 * NUM_REQUESTS + 2
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
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(config),
    )


def _coordinator() -> TTLaneCoordinator:
    coordinator = TTLaneCoordinator.__new__(TTLaneCoordinator)
    coordinator.lanes = [_lane_scheduler() for _ in range(NUM_LANES)]
    coordinator._decode_interleave = TTDecodeInterleavePolicy(
        SimpleNamespace(additional_config={"tt": {}})
    )
    coordinator.num_lanes = NUM_LANES
    coordinator._per_lane_max = PER_LANE_MAX
    coordinator._prefill_gather_s = 0.0
    coordinator._prefill_gather_quiet_s = 0.01
    coordinator.log_stats = False
    coordinator.structured_output_manager = None
    coordinator.connector = None
    coordinator.ec_connector = None
    coordinator._req_to_lane = {}
    coordinator._req_to_row = {}
    coordinator._free_slots_by_lane = [
        list(range(PER_LANE_MAX)) for _ in range(NUM_LANES)
    ]
    return coordinator


def _request(request_id: str, structured: bool) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        ignore_eos=True,
        structured_outputs=(
            StructuredOutputsParams(json={"type": "object"}) if structured else None
        ),
    )
    sampling_params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id=request_id,
        prompt_token_ids=[1] * PROMPT_LEN,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    if structured:
        assert request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    return request


def _runner_output(coordinator, scheduler_output) -> ModelRunnerOutput:
    requests = {}
    for lane in coordinator.lanes:
        requests.update(lane.requests)
    req_ids = list(scheduler_output.num_scheduled_tokens)
    sampled = []
    for req_id in req_ids:
        request = requests[req_id]
        # num_computed_tokens already counts this step's tokens (scheduled).
        done_prompt = request.num_computed_tokens >= request.num_prompt_tokens
        sampled.append([5] if done_prompt else [])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def simulate(structured: bool, grammar_delay_steps: int, max_steps: int = 400):
    """Burst of NUM_REQUESTS arrivals, then step until every request finishes.

    Returns ``(admit_step, running_per_step, lane_of_req)``: the step each
    request was first scheduled, the number of running requests after each
    step, and the lane each request was bound to.
    """
    coordinator = _coordinator()
    requests = [_request(f"r{i}", structured) for i in range(NUM_REQUESTS)]
    for request in requests:
        coordinator.add_request(request)
    lane_of_req = dict(coordinator._req_to_lane)
    admit_step: dict[str, int] = {}
    running_per_step: list[int] = []
    for step in range(max_steps):
        if structured and step == grammar_delay_steps:
            for request in requests:
                request.structured_output_request.grammar = _FakeGrammar()
        out = coordinator.schedule()
        for req_id in out.num_scheduled_tokens:
            admit_step.setdefault(req_id, step)
        coordinator.update_from_output(out, _runner_output(coordinator, out))
        running_per_step.append(sum(len(lane.running) for lane in coordinator.lanes))
        if len(admit_step) == NUM_REQUESTS and not any(
            lane.running or lane.waiting or lane.kv_holding_waiting
            for lane in coordinator.lanes
        ):
            break
    return admit_step, running_per_step, lane_of_req


@pytest.mark.parametrize("grammar_delay_steps", [0, 3])
def test_grammar_requests_spread_over_lanes_like_unconstrained_ones(
    grammar_delay_steps,
):
    """Requests waiting on their grammar must count toward a lane's load.

    Otherwise every lane scores 0 during a burst of grammar requests, the tie
    goes to lane 0, and lane 0 runs them PER_LANE_MAX at a time while the
    other lanes idle (the vLLM 0.30 behaviour, with ``skipped_waiting``).
    """
    _, _, plain_lanes = simulate(False, 0)
    _, _, grammar_lanes = simulate(True, grammar_delay_steps)
    for lanes in (plain_lanes, grammar_lanes):
        counts = [list(lanes.values()).count(lane) for lane in range(NUM_LANES)]
        assert counts == [PER_LANE_MAX] * NUM_LANES


@pytest.mark.parametrize("grammar_delay_steps", [0, 3])
def test_grammar_ready_requests_are_admitted_as_promptly_as_unconstrained(
    grammar_delay_steps,
):
    plain_admit, plain_running, _ = simulate(False, 0)
    grammar_admit, grammar_running, _ = simulate(True, grammar_delay_steps)

    # Every unconstrained request is admitted in the first step and all
    # NUM_REQUESTS run together.
    assert set(plain_admit.values()) == {0}
    assert max(plain_running) == NUM_REQUESTS
    # A grammar request is admitted on the first step its grammar is ready,
    # never later, and all NUM_REQUESTS run together too.
    assert set(grammar_admit.values()) == {grammar_delay_steps}
    assert max(grammar_running) == NUM_REQUESTS
    assert len(grammar_running) == len(plain_running) + grammar_delay_steps
