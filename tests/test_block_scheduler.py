# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from pathlib import Path
from types import SimpleNamespace

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
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore
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
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"


def _scheduler(
    output_width: int = CANVAS, *, diffusion_checkpoint: bool = False
) -> TTScheduler:
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
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
        # Reproduce the platform hook's post-update state, including
        # invalidation of ModelConfig.is_diffusion's cached True value.
        config.model_config.hf_config.canvas_length = output_width
        config.model_config.__dict__.pop("is_diffusion", None)
        assert config.model_config.is_diffusion is True
        delattr(config.model_config.hf_config, "canvas_length")
        config.model_config.__dict__.pop("is_diffusion", None)
        assert config.model_config.is_diffusion is False
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


def _scheduled(
    max_tokens: int = CANVAS * 2,
    *,
    output_width: int = CANVAS,
    ignore_eos: bool = True,
) -> tuple[TTScheduler, Request, SchedulerOutput]:
    scheduler = _scheduler(output_width)
    request = _request(max_tokens, ignore_eos=ignore_eos)
    scheduler.add_request(request)
    return scheduler, request, scheduler.schedule()


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


@pytest.mark.parametrize(
    ("max_tokens", "ignore_eos", "canvas", "kept", "status"),
    [
        (3, True, list(range(CANVAS)), [0, 1, 2], RequestStatus.FINISHED_LENGTH_CAPPED),
        (
            CANVAS,
            False,
            [0, 2, *range(2, CANVAS)],
            [0, 2],
            RequestStatus.FINISHED_STOPPED,
        ),
    ],
    ids=["max_tokens", "eos"],
)
def test_trimmed_canvas_consumes_physical_reservation(
    max_tokens, ignore_eos, canvas, kept, status
):
    scheduler, request, prefill = _scheduled(max_tokens, ignore_eos=ignore_eos)

    outputs = scheduler.update_from_output(prefill, _runner_output(prefill, canvas))

    assert outputs[0].outputs[0].new_token_ids == kept
    assert list(request.output_token_ids) == kept
    assert request.num_output_placeholders == 0
    assert request.status == status


def test_running_block_request_rejects_prefix_cache_reset():
    # Direct scheduler-level backstop: the engine-layer patch aborts running
    # block requests first, so reaching this guard means it was bypassed.
    scheduler, request, _ = _scheduled()

    assert scheduler.running == [request]
    with pytest.raises(RuntimeError, match="Cannot reset prefix cache"):
        scheduler.reset_prefix_cache(
            reset_running_requests=True,
            reset_connector=False,
        )
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING


def _patched_engine(scheduler: TTScheduler, sent: list) -> SimpleNamespace:
    from vllm_tt_plugin.platform import _install_block_output_reset_abort_patch

    _install_block_output_reset_abort_patch()
    return SimpleNamespace(scheduler=scheduler, _send_abort_outputs=sent.append)


def test_engine_level_reset_aborts_running_block_requests():
    scheduler, request, _ = _scheduled()
    sent: list = []
    engine = _patched_engine(scheduler, sent)

    assert (
        EngineCore.reset_prefix_cache(
            engine, reset_running_requests=True, reset_connector=False
        )
        is True
    )
    assert scheduler.running == []
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert sent == [[("req-0", 0)]]


def test_engine_level_keep_pause_reset_preserves_block_requests():
    scheduler, request, _ = _scheduled()
    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    sent: list = []
    engine = _patched_engine(scheduler, sent)

    assert (
        EngineCore.reset_prefix_cache(
            engine, reset_running_requests=True, reset_connector=False
        )
        is False
    )
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING
    assert sent == []


def test_deferred_keep_reset_returns_false_without_preempting_block_request():
    scheduler, request, _ = _scheduled()
    scheduler.set_pause_state(PauseState.PAUSED_ALL)

    assert (
        scheduler.reset_prefix_cache(
            reset_running_requests=True,
            reset_connector=False,
        )
        is False
    )
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING
    assert request.async_tokens_to_discard == 0


def test_ar_prefix_cache_reset_delegates_to_upstream_preemption():
    scheduler, request, _ = _scheduled(output_width=1)

    assert scheduler.reset_prefix_cache(
        reset_running_requests=True,
        reset_connector=False,
    )
    assert scheduler.running == []
    assert request.status == RequestStatus.PREEMPTED
    assert request.async_tokens_to_discard == 1
    assert request.num_output_placeholders == 0


@pytest.mark.parametrize(
    ("mutate", "width", "exc", "match"),
    [
        pytest.param(
            lambda r: setattr(r, "async_tokens_to_discard", 1),
            CANVAS,
            RuntimeError,
            "stale async output",
            id="stale-async-frame",
        ),
        pytest.param(None, CANVAS - 1, ValueError, r"15 != 16", id="wrong-width"),
        pytest.param(
            lambda r: setattr(r, "num_output_placeholders", CANVAS - 1),
            CANVAS,
            RuntimeError,
            "placeholders underflowed",
            id="placeholder-underflow",
        ),
    ],
)
def test_block_output_update_guards(mutate, width, exc, match):
    scheduler, request, prefill = _scheduled()
    if mutate is not None:
        mutate(request)

    with pytest.raises(exc, match=match):
        scheduler.update_from_output(
            prefill, _runner_output(prefill, list(range(width)))
        )


def test_k1_delegates_to_upstream_async_scheduler():
    scheduler, request, prefill = _scheduled(2, output_width=1)
    cache_calls = []
    scheduler.kv_cache_manager.cache_blocks = lambda *args: cache_calls.append(args)

    outputs = scheduler.update_from_output(prefill, _runner_output(prefill, [7]))

    assert outputs[0].outputs[0].new_token_ids == [7]
    assert request.num_output_placeholders == 0
    assert cache_calls


def test_diffusion_checkpoint_books_exactly_one_canvas():
    """After the platform removes the diffusion marker, upstream contributes
    one normal sampled-token placeholder and the plugin reserves only K-1 more."""
    from vllm.config.diffusion import DiffusionConfig

    scheduler = _scheduler(diffusion_checkpoint=True)

    assert scheduler.vllm_config.model_config.is_diffusion is False
    assert scheduler.num_sampled_tokens_per_step == 1
    assert scheduler.num_spec_tokens == 0
    assert scheduler.vllm_config.num_speculative_tokens == 0
    # The platform must still clear a DiffusionConfig created before its hook:
    # retaining one independently resurrects canvas-as-spec accounting.
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
    assert request.num_computed_tokens == 32

    cache_calls = []
    scheduler.kv_cache_manager.cache_blocks = lambda *args: cache_calls.append(args)
    outputs = scheduler.update_from_output(
        prefill, _runner_output(prefill, list(range(CANVAS)))
    )
    assert outputs[0].outputs[0].new_token_ids == list(range(CANVAS))
    assert request.num_output_placeholders == 0
    assert cache_calls == []

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-0": CANVAS}
    assert decode.num_spec_tokens_to_schedule == 0
    assert request.num_output_placeholders == CANVAS
    assert request.num_computed_tokens == 32 + CANVAS

    scheduler.update_from_output(decode, _runner_output(decode, list(range(CANVAS))))
    assert request.num_output_placeholders == 0
    assert cache_calls == []
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
