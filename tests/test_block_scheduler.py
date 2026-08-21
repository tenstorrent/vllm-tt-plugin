# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from contextlib import contextmanager
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
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore, EngineCoreProc
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


class _StubModel:
    """No model_capabilities: the platform hook resolves
    output_tokens_per_step=1; tests inject the block width afterwards."""


@contextmanager
def _stub_model_resolution():
    """Resolve the stub instead of tt-metal's TTQwen2ForCausalLM while the
    platform hook runs inside VllmConfig.__post_init__ (mirrors
    test_block_request_validation._patch_model_resolution)."""
    with (
        # Fresh-process semantics: don't leave this file's configs as the
        # platform's process-level admission handle across tests.
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
        yield


def _scheduler(
    output_width: int = CANVAS, *, diffusion_checkpoint: bool = False
) -> TTScheduler:
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
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
    with _stub_model_resolution():
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


def test_add_request_clamps_max_tokens_that_would_overshoot_max_model_len():
    """A prebuilt request's leftover max_tokens must not schedule a canvas
    past max_model_len; that path raises in the runner and kills the engine."""
    scheduler = _scheduler()
    request = _request(max_tokens=MAX_MODEL_LEN)

    scheduler.add_request(request)

    # prompt=32, max_model_len=256, canvas=16 → 224 tokens of whole canvases.
    assert request.max_tokens == 224
    assert request.sampling_params.max_tokens == 224


def test_add_request_strips_host_sampling_controls_from_bypassed_request():
    """A prebuilt EngineCoreRequest skips frontend validation; any of these
    controls flips the step onto host sampling, which cannot construct a
    multi-token canvas and would kill the engine."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(
        max_tokens=CANVAS,
        ignore_eos=True,
        min_p=0.2,
        min_tokens=1,
        logit_bias={2: 1.0},
        allowed_token_ids=[1, 2],
        bad_words=["bad"],
        structured_outputs=StructuredOutputsParams(json_object=True),
    )
    params.update_from_generation_config({}, eos_token_id=2)
    # The tokenized form is what actually flips the worker onto host sampling
    # (InputBatch reads bad_words_token_ids, not the strings); with
    # skip_tokenizer_init it stays unset unless seeded here.
    params._bad_words_token_ids = [[7]]
    request = Request(
        request_id="bypass-0",
        prompt_token_ids=[1] * 32,
        sampling_params=params,
        pooling_params=None,
        resumable=True,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    assert request.use_structured_output
    assert request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

    scheduler.add_request(request)

    assert not request.use_structured_output
    assert params.structured_outputs is None
    assert params.min_p == 0.0
    assert params.min_tokens == 0
    assert params.logit_bias is None
    assert params.allowed_token_ids is None
    assert params.bad_words is None
    assert params._bad_words_token_ids is None
    # A resumable session would park the stopped request forever and leak the
    # model-owned state slot.
    assert request.resumable is False
    # With the structured-output request gone, nothing could ever promote the
    # request out of skipped_waiting; it must be schedulable immediately.
    assert request.status == RequestStatus.WAITING
    scheduled = scheduler.schedule()
    assert scheduled.num_scheduled_tokens == {"bypass-0": 32}


# Largest tile-aligned prompt that still fits one whole canvas: the
# truncation target for unservable bypassed prompts (mml=256, K=16 -> 224).
SERVABLE_PROMPT = (MAX_MODEL_LEN // 32 * 32 - CANVAS) // 32 * 32


@pytest.mark.parametrize(
    "prompt_len",
    [MAX_MODEL_LEN - 15, MAX_MODEL_LEN + 40],
    ids=["dead-zone-band", "beyond-max-model-len"],
)
def test_unservable_bypassed_prompt_is_truncated_and_served(prompt_len):
    """A bypassed prompt with no room for a whole canvas is otherwise fatal:
    parked forever when it exceeds the token budget, overflowing the worker's
    max_model_len-wide buffer when it doesn't, or — even when it fits
    max_model_len — killing the engine via the adapter's own capacity check
    in eager mode, which raises instead of returning a clippable canvas.
    Truncation makes the request genuinely servable end to end."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=64, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="unservable-0",
        prompt_token_ids=[1] * prompt_len,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.num_prompt_tokens == SERVABLE_PROMPT
    assert len(request.prompt_token_ids) == SERVABLE_PROMPT
    assert request.num_tokens == SERVABLE_PROMPT
    # Whole canvases still fitting after the tile-aligned truncation.
    assert request.max_tokens == 32

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"unservable-0": SERVABLE_PROMPT}
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))
    assert request.status == RequestStatus.RUNNING

    decode = scheduler.schedule()
    scheduler.update_from_output(decode, _runner_output(decode, list(range(CANVAS))))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.running == []


def test_continuation_of_scrubbed_resumable_session_is_dropped():
    """Scrubbing resumable admits the first chunk with streaming_queue=None,
    and the streaming protocol always sends a same-id follow-up (the next
    chunk or the closing sentinel); the base scheduler's duplicate-id assert
    on the missing queue would tear down EngineCore."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=CANVAS, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    first = Request(
        request_id="stream-0",
        prompt_token_ids=[1] * 32,
        sampling_params=params,
        pooling_params=None,
        resumable=True,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    scheduler.add_request(first)
    assert first.resumable is False
    assert first.streaming_queue is None

    sentinel_params = SamplingParams(max_tokens=1)
    sentinel_params.update_from_generation_config({}, eos_token_id=2)
    sentinel = Request(
        request_id="stream-0",
        prompt_token_ids=[0],
        sampling_params=sentinel_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    scheduler.add_request(sentinel)  # must not raise

    assert scheduler.requests["stream-0"] is first
    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"stream-0": 32}


def test_multimodal_features_are_dropped_from_bypassed_request():
    """A text-only block model has a zero encoder budget: an mm feature at
    offset 0 parks the request in WAITING forever (head-of-line stall), and
    an interior offset carves a partial prefill chunk that flips the step
    onto host sampling and kills the engine."""
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="mm-0",
        prompt_token_ids=[1] * 32,
        sampling_params=params,
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="img-0",
                mm_position=PlaceholderRange(offset=0, length=16),
            )
        ],
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    assert request.has_encoder_inputs

    scheduler.add_request(request)

    assert request.mm_features == []
    assert not request.has_encoder_inputs

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"mm-0": 32}
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.running == []


def test_embeds_only_bypassed_prompt_is_replaced_with_placeholders():
    """The frontend rejects prompt_embeds for every TT model; admitted bare,
    the worker's request-state builder raises NotImplementedError out of
    execute_model and kills the engine."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="embeds-0",
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(8, 4),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.prompt_token_ids == [0] * 8
    assert request.prompt_embeds is None
    assert request.num_prompt_tokens == 8
    # The max_tokens clamp no longer early-returns on the missing token ids.
    assert request.max_tokens == 3

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"embeds-0": 8}
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.running == []


def test_oversized_embeds_only_prompt_is_replaced_and_truncated():
    """Embeds replacement and prompt truncation mutate the same four fields
    in sequence; a refactor breaking only the composition would admit an
    oversized or internally inconsistent request while the single-step tests
    stay green."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=64, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="embeds-big-0",
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(MAX_MODEL_LEN + 40, 4),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.prompt_token_ids == [0] * SERVABLE_PROMPT
    assert request.prompt_embeds is None
    assert request.num_prompt_tokens == SERVABLE_PROMPT
    assert request.num_tokens == SERVABLE_PROMPT
    assert request.max_tokens == 32

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"embeds-big-0": SERVABLE_PROMPT}
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))
    decode = scheduler.schedule()
    scheduler.update_from_output(decode, _runner_output(decode, list(range(CANVAS))))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.running == []


def test_empty_bypassed_prompt_is_padded_and_served():
    """The frontend rejects empty prompts; admitted bare, the waiting loop
    schedules zero new tokens and upstream's num_new_tokens assert tears
    down the engine."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="empty-0",
        prompt_token_ids=[],
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.prompt_token_ids == [0]
    assert request.num_prompt_tokens == 1
    assert request.num_tokens == 1

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"empty-0": 1}
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.running == []


def test_zero_max_tokens_bypassed_request_finishes_after_first_canvas():
    """A hand-crafted prebuilt request can carry max_tokens=0 (SamplingParams
    itself forbids it); the stop check must finish it length-capped on its
    first canvas instead of generating forever."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    params.max_tokens = 0
    request = Request(
        request_id="zero-0",
        prompt_token_ids=[1] * 32,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    assert request.max_tokens == 0

    scheduler.add_request(request)
    assert request.max_tokens == 0

    prefill = scheduler.schedule()
    outputs = scheduler.update_from_output(
        prefill, _runner_output(prefill, list(range(CANVAS)))
    )

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    # The stopping token is the only client-visible overshoot of the zero
    # budget; the request finishes through the normal output path.
    assert outputs[0].outputs[0].new_token_ids == [0]
    assert request.num_output_placeholders == 0
    assert scheduler.running == []


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


def test_engine_reset_patch_requires_resolved_output_width():
    scheduler, _, _ = _scheduled()
    scheduler.vllm_config.additional_config.clear()
    engine = _patched_engine(scheduler, [])

    with pytest.raises(RuntimeError, match="was not initialized"):
        EngineCore.reset_prefix_cache(
            engine, reset_running_requests=True, reset_connector=False
        )


def test_engine_without_abort_notifier_refuses_reset():
    # A bare in-process EngineCore lacks _send_abort_outputs: aborting there
    # would silently remove a request its caller is still waiting on, so the
    # reset must fall through to the scheduler guard's raise instead.
    scheduler, request, _ = _scheduled()
    from vllm_tt_plugin.platform import _install_block_output_reset_abort_patch

    _install_block_output_reset_abort_patch()
    engine = SimpleNamespace(scheduler=scheduler)

    with pytest.raises(RuntimeError, match="Cannot reset prefix cache"):
        EngineCore.reset_prefix_cache(
            engine, reset_running_requests=True, reset_connector=False
        )
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING


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


def _pause_guarded_engine(scheduler: TTScheduler) -> SimpleNamespace:
    from vllm_tt_plugin.platform import _install_block_output_pause_guard_patch

    _install_block_output_pause_guard_patch()
    return SimpleNamespace(scheduler=scheduler)


def test_keep_pause_with_clear_cache_is_refused_up_front():
    # The keep-mode reset runs from an idle callback whose result upstream
    # discards, so the only honest failure is a synchronous one before any
    # pause state changes.
    scheduler, request, _ = _scheduled()
    engine = _pause_guarded_engine(scheduler)

    with pytest.raises(ValueError, match="clear_cache=False or mode='abort'"):
        EngineCore.pause_scheduler(engine, mode="keep", clear_cache=True)

    assert scheduler.pause_state == PauseState.UNPAUSED
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING


def test_keep_pause_patch_requires_resolved_output_width():
    scheduler, _, _ = _scheduled()
    scheduler.vllm_config.additional_config.clear()
    engine = _pause_guarded_engine(scheduler)

    with pytest.raises(RuntimeError, match="was not initialized"):
        EngineCore.pause_scheduler(engine, mode="keep", clear_cache=True)


def test_keep_pause_guard_covers_engine_core_proc():
    # EngineCoreProc overrides pause_scheduler, so the guard must wrap it too.
    scheduler, request, _ = _scheduled()
    engine = _pause_guarded_engine(scheduler)

    with pytest.raises(ValueError, match="live block-output request"):
        EngineCoreProc.pause_scheduler(engine, mode="keep", clear_cache=True)

    assert scheduler.running == [request]


def test_keep_pause_without_clear_cache_pauses_block_requests():
    scheduler, request, _ = _scheduled()
    engine = _pause_guarded_engine(scheduler)

    assert EngineCore.pause_scheduler(engine, mode="keep", clear_cache=False) is None

    assert scheduler.pause_state == PauseState.PAUSED_ALL
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING


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
        pytest.param(
            None,
            CANVAS - 1,
            ValueError,
            r"15 != 16",
            id="narrow-output",
        ),
        pytest.param(
            None,
            CANVAS + 1,
            ValueError,
            r"17 != 16",
            id="wide-output",
        ),
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
    scheduler = _scheduler(diffusion_checkpoint=True)

    assert scheduler.vllm_config.model_config.is_diffusion is False
    assert scheduler.num_sampled_tokens_per_step == 1
    assert scheduler.num_spec_tokens == 0
    assert scheduler.vllm_config.num_speculative_tokens == 0

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
