# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

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
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.config import store_tt_output_tokens_per_step
from vllm_tt_plugin.scheduler import TTScheduler

BLOCK_SIZE = 16
CANVAS = 16
MAX_MODEL_LEN = 256


def _scheduler(
    output_width: int = CANVAS, *, diffusion_checkpoint: bool = False
) -> TTScheduler:
    model_config = ModelConfig(
        model="Qwen/Qwen2-0.5B-Instruct",
        dtype="float16",
        seed=42,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=False,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    config.scheduler_config.async_scheduling = False
    if diffusion_checkpoint:
        # A real DiffusionGemma checkpoint flags is_diffusion through
        # hf_config.canvas_length (num_sampled_tokens_per_step becomes 0).
        # The platform hook clears the upstream auto-created DiffusionConfig,
        # so config.diffusion_config stays None here — the post-hook state.
        config.model_config.hf_config.canvas_length = output_width
    store_tt_output_tokens_per_step(config, output_width)
    num_blocks = MAX_MODEL_LEN // BLOCK_SIZE + 2
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
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(max_tokens: int, *, ignore_eos: bool = True) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )
    sampling_params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id="req-0",
        prompt_token_ids=[1] * 32,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _runner_output(
    scheduler_output: SchedulerOutput, tokens: list[int]
) -> ModelRunnerOutput:
    req_id = next(iter(scheduler_output.num_scheduled_tokens))
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[tokens],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_scheduler_reserves_and_reconciles_full_canvas():
    scheduler = _scheduler()
    request = _request(CANVAS * 2)
    scheduler.add_request(request)

    prefill = scheduler.schedule()

    assert prefill.num_scheduled_tokens == {"req-0": 32}
    assert request.num_output_placeholders == CANVAS
    assert request.num_computed_tokens == 32

    outputs = scheduler.update_from_output(
        prefill, _runner_output(prefill, list(range(CANVAS)))
    )
    assert outputs[0].outputs[0].new_token_ids == list(range(CANVAS))
    assert request.num_output_placeholders == 0

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-0": CANVAS}
    assert request.num_output_placeholders == CANVAS
    assert request.num_computed_tokens == 32 + CANVAS


def test_max_tokens_trims_canvas_and_consumes_physical_reservation():
    scheduler = _scheduler()
    request = _request(3)
    scheduler.add_request(request)
    prefill = scheduler.schedule()

    outputs = scheduler.update_from_output(
        prefill, _runner_output(prefill, list(range(CANVAS)))
    )

    assert outputs[0].outputs[0].new_token_ids == [0, 1, 2]
    assert list(request.output_token_ids) == [0, 1, 2]
    assert request.num_output_placeholders == 0
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_eos_trims_canvas_and_consumes_physical_reservation():
    scheduler = _scheduler()
    request = _request(CANVAS, ignore_eos=False)
    scheduler.add_request(request)
    prefill = scheduler.schedule()

    outputs = scheduler.update_from_output(
        prefill, _runner_output(prefill, [0, 2, *range(2, CANVAS)])
    )

    assert outputs[0].outputs[0].new_token_ids == [0, 2]
    assert request.num_output_placeholders == 0
    assert request.status == RequestStatus.FINISHED_STOPPED


def test_running_block_request_rejects_prefix_cache_reset():
    scheduler = _scheduler()
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    scheduler.schedule()

    assert scheduler.running == [request]
    assert (
        scheduler.reset_prefix_cache(
            reset_running_requests=True,
            reset_connector=False,
        )
        is False
    )
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING


def test_block_output_rejects_stale_async_frame():
    scheduler = _scheduler()
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    request.async_tokens_to_discard = 1

    with pytest.raises(RuntimeError, match="stale async output"):
        scheduler.update_from_output(
            prefill,
            _runner_output(prefill, list(range(CANVAS))),
        )


def test_block_output_rejects_wrong_width():
    scheduler = _scheduler()
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    prefill = scheduler.schedule()

    with pytest.raises(ValueError, match=r"15 != 16"):
        scheduler.update_from_output(
            prefill,
            _runner_output(prefill, list(range(CANVAS - 1))),
        )


def test_block_output_rejects_placeholder_underflow():
    scheduler = _scheduler()
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    request.num_output_placeholders = CANVAS - 1

    with pytest.raises(RuntimeError, match="placeholders underflowed"):
        scheduler.update_from_output(
            prefill,
            _runner_output(prefill, list(range(CANVAS))),
        )


def test_k1_delegates_to_upstream_async_scheduler():
    scheduler = _scheduler(output_width=1)
    request = _request(2)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    cache_calls = []
    scheduler.kv_cache_manager.cache_blocks = lambda *args: cache_calls.append(args)

    outputs = scheduler.update_from_output(prefill, _runner_output(prefill, [7]))

    assert outputs[0].outputs[0].new_token_ids == [7]
    assert request.num_output_placeholders == 0
    assert cache_calls


def test_diffusion_checkpoint_books_exactly_one_canvas():
    """Regression for the canvas-as-spec double booking: upstream 0.24 turns
    an uncleared DiffusionConfig into num_spec_tokens=canvas_length, stacking
    256 speculative placeholders on the plugin's physical reservation until
    long generations park unfinished. With the platform hook clearing it, a
    diffusion-flagged config must book exactly one canvas per step."""
    from vllm.config.diffusion import DiffusionConfig

    scheduler = _scheduler(diffusion_checkpoint=True)

    assert scheduler.num_sampled_tokens_per_step == 0
    assert scheduler.num_spec_tokens == 0
    # The live property chain the platform hook must keep broken: an
    # uncleared DiffusionConfig resurrects canvas-as-spec accounting.
    assert scheduler.vllm_config.num_speculative_tokens == 0
    scheduler.vllm_config.diffusion_config = DiffusionConfig(canvas_length=CANVAS)
    assert scheduler.vllm_config.num_speculative_tokens == CANVAS
    scheduler.vllm_config.diffusion_config = None

    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    prefill = scheduler.schedule()

    assert prefill.num_scheduled_tokens == {"req-0": 32}
    assert prefill.num_spec_tokens_to_schedule == 0
    assert list(request.spec_token_ids) == []
    assert request.num_output_placeholders == CANVAS

    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))
    assert request.num_output_placeholders == 0

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-0": CANVAS}
    assert decode.num_spec_tokens_to_schedule == 0
    assert request.num_output_placeholders == CANVAS

    scheduler.update_from_output(decode, _runner_output(decode, list(range(CANVAS))))
    assert request.num_output_placeholders == 0
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
