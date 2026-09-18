# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import time
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

from vllm_tt_plugin.config import (
    store_tt_adaptive_block_output,
    store_tt_block_kv_extent_tokens,
    store_tt_output_tokens_per_step,
)
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import (
    TTScheduler,
    get_tt_block_step_decisions,
    get_tt_forced_reset_discard_counts,
)

BLOCK_SIZE = 128
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
    output_width: int = CANVAS,
    *,
    diffusion_checkpoint: bool = False,
    max_model_len: int = MAX_MODEL_LEN,
    async_scheduling: bool = False,
    adaptive: bool = False,
    max_num_seqs: int = 1,
    kv_extent: int = 0,
) -> TTScheduler:
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = max_model_len
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_model_len,
        max_model_len=max_model_len,
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
    config.scheduler_config.async_scheduling = async_scheduling
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
    if adaptive:
        store_tt_adaptive_block_output(config, True)
    if kv_extent:
        store_tt_block_kv_extent_tokens(config, kv_extent)
    num_blocks = max_model_len // BLOCK_SIZE + 2
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


def _request(
    max_tokens: int,
    *,
    ignore_eos: bool = True,
    request_id: str = "req-0",
    prompt_length: int = 32,
) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )
    sampling_params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * prompt_length,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _scheduled(
    max_tokens: int = CANVAS * 2,
    *,
    output_width: int = CANVAS,
    ignore_eos: bool = True,
    async_scheduling: bool = False,
) -> tuple[TTScheduler, Request, SchedulerOutput]:
    scheduler = _scheduler(output_width, async_scheduling=async_scheduling)
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


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_prefill_fallback_releases_finished_runner_state(async_scheduling):
    """KV pressure must not consume completion notifications before decode."""
    scheduler = _scheduler(
        output_width=1, max_num_seqs=2, async_scheduling=async_scheduling
    )
    # Four KV blocks include the reserved null block. A needs two usable
    # blocks and B needs one, so finishing B cannot yet admit C's two blocks.
    scheduler.add_request(_request(2, request_id="A", prompt_length=129))
    scheduler.add_request(_request(1, request_id="B"))
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.tt_per_lane_max_num_seqs = 2
    runner._req_state_slot = {}
    runner._pending_state_slot_settle = None
    runner.requests = {}
    runner.encoder_cache = {}
    runner._decode_layout_changed_since_last_decode = False
    released_slots = []
    runner.model = SimpleNamespace(release_request=released_slots.append)
    runner.input_batch = InputBatch(
        max_num_reqs=2,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=64,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"A": 129, "B": 32}
    runner._update_states(prefill)
    assert runner._alloc_prefill_state_slots(["A", "B"]) == [0, 1]
    scheduler.update_from_output(
        prefill,
        ModelRunnerOutput(
            req_ids=["A", "B"],
            req_id_to_index={"A": 0, "B": 1},
            sampled_token_ids=[[2], [2]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert scheduler.finished_req_ids == {"B"}
    scheduler.add_request(_request(10, request_id="C", prompt_length=129))

    decode = scheduler.schedule()

    assert decode.num_scheduled_tokens == {"A": 1}
    assert decode.finished_req_ids == {"B"}
    runner._update_states(decode)
    assert set(runner.requests) == {"A"}
    assert runner._req_state_slot == {"A": 0}
    assert released_slots == [1]

    scheduler.update_from_output(decode, _runner_output(decode, [2]))
    scheduler.add_request(_request(10, request_id="D"))
    next_prefill = scheduler.schedule()
    assert next_prefill.num_scheduled_tokens == {"C": 129, "D": 32}
    assert next_prefill.finished_req_ids == {"A"}
    runner._update_states(next_prefill)
    assert set(runner.requests) == {"C", "D"}
    assert released_slots == [1, 0]
    assert runner._alloc_prefill_state_slots(["C", "D"]) == [0, 1]


def test_scheduler_records_the_actual_widest_decode_batch():
    scheduler = _scheduler(output_width=1, max_num_seqs=2)
    first = _request(max_tokens=8, request_id="req-0")
    second = _request(max_tokens=8, request_id="req-1")
    scheduler.add_request(first)
    scheduler.add_request(second)
    prefill = scheduler.schedule()
    req_ids = list(prefill.num_scheduled_tokens)
    output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[[7], [8]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(prefill, output)

    decode = scheduler.schedule()

    assert len(decode.num_scheduled_tokens) == 2
    assert scheduler._widest_decode_batch_size == 2


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
        presence_penalty=0.5,
        frequency_penalty=0.5,
        repetition_penalty=1.1,
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
    # Non-neutral penalties make every block step build session-length
    # penalty tensors it then discards.
    assert params.presence_penalty == 0.0
    assert params.frequency_penalty == 0.0
    assert params.repetition_penalty == 1.0
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


@pytest.mark.parametrize(
    ("output_width", "max_model_len", "expected_prompt", "expected_max_tokens"),
    [
        # keep = (256 - 64) // 32 * 32 = 192; remaining 64 = one 64-canvas.
        pytest.param(64, 256, 192, 64, id="width-above-tile"),
        # aligned mml = 250 // 32 * 32 = 224; keep = (224 - 16) // 32 * 32
        # = 192; remaining 32 = two 16-canvases.
        pytest.param(CANVAS, 250, 192, 32, id="unaligned-max-model-len"),
    ],
)
def test_truncation_respects_width_and_alignment(
    output_width, max_model_len, expected_prompt, expected_max_tokens
):
    """Regimes where the width term and tile alignment actually matter; the
    default fixture (canvas 16, aligned limit) is insensitive to both."""
    scheduler = _scheduler(output_width, max_model_len=max_model_len)
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=64, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="regime-0",
        prompt_token_ids=[1] * 300,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.num_prompt_tokens == expected_prompt
    assert request.max_tokens == expected_max_tokens


def test_plain_prefix_cache_reset_leaves_running_block_requests_alone():
    """A plain /reset_prefix_cache (reset_running_requests=False) must
    delegate upstream instead of raising or preempting live block work."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))
    assert request.status == RequestStatus.RUNNING

    result = scheduler.reset_prefix_cache(
        reset_running_requests=False, reset_connector=False
    )

    assert result is False
    assert request.status == RequestStatus.RUNNING
    assert scheduler.running == [request]


def test_mixed_token_embeds_bypassed_prompt_drops_the_embeds():
    """Mixed token+embeds prompts skip the embeds-only branch; the embeds
    must still be scrubbed rather than pinned for the request lifetime."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="mixed-0",
        prompt_token_ids=[1, 0, 3, 0, 5, 6, 7, 8],
        prompt_embeds=torch.zeros(8, 4),
        prompt_is_token_ids=[True, False, True, False, True, True, True, True],
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.prompt_embeds is None
    assert request.prompt_is_token_ids is None
    assert request.prompt_token_ids == [1, 0, 3, 0, 5, 6, 7, 8]


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
    assert sent == [[request]]


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


def test_ordinary_async_preemption_keeps_inflight_token_for_resume():
    scheduler, request, submitted = _scheduled(output_width=1, async_scheduling=True)
    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.monotonic())

    assert request.num_output_placeholders == 1
    outputs = scheduler.update_from_output(submitted, _runner_output(submitted, [7]))
    assert outputs[0].outputs[0].new_token_ids == [7]
    assert list(request.output_token_ids) == [7]
    assert request.num_output_placeholders == 0

    resumed = scheduler.schedule()
    assert request.request_id in resumed.scheduled_cached_reqs.resumed_req_ids


def test_preempted_request_waits_for_inflight_output_before_resume():
    scheduler, request, submitted = _scheduled(output_width=1, async_scheduling=True)
    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.monotonic())

    blocked = scheduler.schedule()
    assert blocked.total_num_scheduled_tokens == 0
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_output_placeholders == 1

    older = scheduler.update_from_output(submitted, _runner_output(submitted, [7]))
    assert older[0].outputs[0].new_token_ids == [7]
    assert request.num_output_placeholders == 0

    resumed = scheduler.schedule()
    assert request.request_id in resumed.scheduled_cached_reqs.resumed_req_ids
    assert request.num_output_placeholders == 1

    following = scheduler.update_from_output(resumed, _runner_output(resumed, [8]))
    assert following[0].outputs[0].new_token_ids == [8]
    assert list(request.output_token_ids) == [7, 8]
    assert request.num_output_placeholders == 0


def test_forced_reset_discards_stale_frame_before_following_valid_frame():
    scheduler, request, submitted = _scheduled(output_width=1, async_scheduling=True)

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = scheduler.schedule()

    assert get_tt_forced_reset_discard_counts(resumed) == {request.request_id: 1}
    stale = scheduler.update_from_output(submitted, _runner_output(submitted, [7]))
    assert stale[0].outputs == []
    assert request.async_tokens_to_discard == 0
    assert list(request.output_token_ids) == []

    valid = scheduler.update_from_output(resumed, _runner_output(resumed, [8]))
    assert valid[0].outputs[0].new_token_ids == [8]
    assert list(request.output_token_ids) == [8]
    assert request.num_output_placeholders == 0


def test_forced_reset_counts_one_speculative_forward_as_one_output_frame():
    scheduler, request, prefill = _scheduled(output_width=1, async_scheduling=True)
    scheduler.update_from_output(prefill, _runner_output(prefill, [7]))
    scheduler.num_spec_tokens = 3
    scheduler.num_lookahead_tokens = 3
    request.spec_token_ids = [-1] * 3
    submitted = scheduler.schedule()

    assert request.num_output_placeholders == 4
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = scheduler.schedule()

    assert get_tt_forced_reset_discard_counts(resumed) == {request.request_id: 1}
    assert request.async_tokens_to_discard == 1

    stale = scheduler.update_from_output(
        submitted, _runner_output(submitted, [8, 9, 10, 11])
    )
    assert stale[0].outputs == []
    assert request.async_tokens_to_discard == 0

    valid = scheduler.update_from_output(resumed, _runner_output(resumed, [12]))
    assert valid[0].outputs[0].new_token_ids == [12]
    assert list(request.output_token_ids) == [7, 12]


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


# ── Adaptive block-output (tt_adaptive_block_output) ─────────────────────────


def _adaptive_anchor(scheduler, request, token=5):
    """Drive the prefill step: adaptive prefills commit ONE anchor token."""
    submitted = scheduler.schedule()
    assert get_tt_block_step_decisions(submitted)[request.request_id] is False
    assert request.num_output_placeholders == 1
    outputs = scheduler.update_from_output(
        submitted, _runner_output(submitted, [token])
    )
    assert outputs[0].outputs[0].new_token_ids == [token]
    assert request.num_output_placeholders == 0
    return outputs


def test_adaptive_prefill_commits_single_anchor_then_solo_decode_blocks():
    """Prefill is a plain one-token step; the following solo decode reserves
    and commits the full block. On the old code the prefill itself reserved
    the block and a 1-token anchor commit was rejected."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    _adaptive_anchor(scheduler, request)

    submitted = scheduler.schedule()
    assert get_tt_block_step_decisions(submitted)[request.request_id] is True
    assert request.num_output_placeholders == CANVAS

    block = list(range(10, 10 + CANVAS))
    outputs = scheduler.update_from_output(submitted, _runner_output(submitted, block))
    assert outputs[0].outputs[0].new_token_ids == block
    assert request.num_output_placeholders == 0


def test_adaptive_batched_decode_commits_single_tokens():
    """Two decodes in one step each get ONE placeholder and commit one
    baseline token."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    req_a = _request(CANVAS * 2, request_id="req-a")
    req_b = _request(CANVAS * 2, request_id="req-b")
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    # Batched prefill step: both commit their anchors.
    submitted = scheduler.schedule()
    assert len(submitted.num_scheduled_tokens) == 2
    anchor_output = ModelRunnerOutput(
        req_ids=["req-a", "req-b"],
        req_id_to_index={"req-a": 0, "req-b": 1},
        sampled_token_ids=[[5], [6]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(submitted, anchor_output)

    # Batched DECODE step: still plain one-token baseline for both.
    submitted = scheduler.schedule()
    for req in (req_a, req_b):
        assert get_tt_block_step_decisions(submitted)[req.request_id] is False
        assert req.num_output_placeholders == 1
    decode_output = ModelRunnerOutput(
        req_ids=["req-a", "req-b"],
        req_id_to_index={"req-a": 0, "req-b": 1},
        sampled_token_ids=[[7], [9]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    outputs = scheduler.update_from_output(submitted, decode_output)
    committed = {o.request_id: o.new_token_ids for o in outputs[0].outputs}
    assert committed == {"req-a": [7], "req-b": [9]}
    assert req_a.num_output_placeholders == 0
    assert req_b.num_output_placeholders == 0


def test_adaptive_batch_prefilled_request_never_blocks():
    """A request whose PREFILL ran batched never gets a block -- not even once
    it is the only request left.

    The model captures drafter taps only during a SOLO prefill and never
    re-arms afterwards (taps come only from a prefill), so a batch-prefilled
    request holds no speculative session for its whole life and serves every
    solo decode as plain width-1 baseline. Reserving the block here is what
    broke the benchmark sweep as concurrency drained back to one request: the
    scheduler reserved CANVAS and the model emitted 1.

    This replaces an earlier test that asserted the survivor DID block again,
    which described a capability the model does not have.
    """
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    req_a = _request(CANVAS * 4, request_id="req-a")
    req_b = _request(1, request_id="req-b", ignore_eos=False)
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    submitted = scheduler.schedule()  # batched prefill -> taps for neither
    assert len(submitted.num_scheduled_tokens) == 2
    assert scheduler._spec_session_owner is None
    anchor_output = ModelRunnerOutput(
        req_ids=["req-a", "req-b"],
        req_id_to_index={"req-a": 0, "req-b": 1},
        sampled_token_ids=[[5], [2]],  # req-b hits max_tokens=1 and finishes
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(submitted, anchor_output)
    assert req_b.is_finished()

    resumed = scheduler.schedule()
    assert len(resumed.num_scheduled_tokens) == 1
    assert get_tt_block_step_decisions(resumed)[req_a.request_id] is False
    assert req_a.num_output_placeholders == 1
    # ... and the width-1 commit the model actually emits reconciles cleanly.
    outputs = scheduler.update_from_output(resumed, _runner_output(resumed, [7]))
    assert outputs[0].outputs[0].new_token_ids == [7]
    assert req_a.num_output_placeholders == 0


def test_adaptive_batched_decode_drops_the_session_permanently():
    """The sweep-tail case, with both requests prefilled SOLO so each owned
    the session in turn.

    A batched decode makes the model release its session, and it never
    re-arms. So when the peer finishes, the survivor is solo AND spec-eligible
    but owns nothing -- it must stay at width 1.
    """
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    req_a = _request(CANVAS * 4, request_id="req-a")
    scheduler.add_request(req_a)
    _adaptive_anchor(scheduler, req_a)  # solo prefill -> req-a owns the session
    assert scheduler._spec_session_owner == "req-a"

    # req-b prefills solo: the model's SINGLE session is re-seated to req-b.
    req_b = _request(2, request_id="req-b", ignore_eos=False)
    scheduler.add_request(req_b)
    prefill_b = scheduler.schedule()
    assert len(prefill_b.num_scheduled_tokens) == 1, "TT steps are never mixed"
    assert get_tt_block_step_decisions(prefill_b)["req-b"] is False
    scheduler.update_from_output(prefill_b, _runner_output(prefill_b, [5]))
    assert scheduler._spec_session_owner == "req-b"

    # Both decode together: the model releases the session for good.
    batched = scheduler.schedule()
    assert len(batched.num_scheduled_tokens) == 2
    assert scheduler._spec_session_owner is None
    decisions = get_tt_block_step_decisions(batched)
    assert decisions["req-a"] is False and decisions["req-b"] is False
    scheduler.update_from_output(
        batched,
        ModelRunnerOutput(
            req_ids=["req-a", "req-b"],
            req_id_to_index={"req-a": 0, "req-b": 1},
            sampled_token_ids=[[6], [2]],  # req-b stops on EOS
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert req_b.is_finished()

    # req-a is alone again -- and still session-less.
    resumed = scheduler.schedule()
    assert len(resumed.num_scheduled_tokens) == 1
    assert get_tt_block_step_decisions(resumed)["req-a"] is False
    assert req_a.num_output_placeholders == 1


def test_adaptive_aborted_owner_does_not_hand_its_session_to_a_peer():
    """req-b's prefill re-seats the model's single session. If req-b is
    aborted before it ever decodes, req-a's next solo decode must NOT inherit
    req-b's taps: speculating from another prompt's residuals (and another
    prompt's length) yields wrong TOKENS, not merely a wrong width. The model
    refuses it independently via _spec_pending_is_mine; the scheduler must
    agree so the widths still match.
    """
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    req_a = _request(CANVAS * 4, request_id="req-a")
    scheduler.add_request(req_a)
    _adaptive_anchor(scheduler, req_a)
    assert scheduler._spec_session_owner == "req-a"

    req_b = _request(CANVAS * 4, request_id="req-b")
    scheduler.add_request(req_b)
    prefill_b = scheduler.schedule()
    scheduler.update_from_output(prefill_b, _runner_output(prefill_b, [5]))
    assert scheduler._spec_session_owner == "req-b"

    scheduler.finish_requests("req-b", RequestStatus.FINISHED_ABORTED)

    resumed = scheduler.schedule()
    assert len(resumed.num_scheduled_tokens) == 1
    assert get_tt_block_step_decisions(resumed)["req-a"] is False
    assert req_a.num_output_placeholders == 1


def test_adaptive_commit_without_scheduling_decision_raises():
    """A committing request the placeholder pass never stamped is a broken
    invariant, not a silent block-path default."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    request = _request(CANVAS * 2)
    with pytest.raises(RuntimeError, match="without a scheduling decision"):
        scheduler._update_request_with_output(request, list(range(CANVAS)))


def test_adaptive_block_decision_survives_async_schedule_lag():
    """The async batch queue runs schedule() for the NEXT step (mutating
    Request state) before update_from_output commits the PREVIOUS step. The
    per-step decision must ride its own SchedulerOutput, not a Request slot the
    next schedule overwrites -- the exact interleave that crashed the engine
    with "1 != 64" when the prefill anchor's width-1 commit hit the decode
    step's block stamp."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    request = _request(CANVAS * 3)
    scheduler.add_request(request)

    # Step P: schedule the prefill (anchor, width-1, non-block).
    prefill = scheduler.schedule()
    assert get_tt_block_step_decisions(prefill)[request.request_id] is False

    # Emulate the batch queue: the anchor's output is NOT committed yet. Run
    # schedule() for the FIRST DECODE step first, overwriting live Request
    # state (num_computed_tokens, num_output_placeholders) the way async does.
    scheduler.update_from_output(prefill, _runner_output(prefill, [5]))
    decode = scheduler.schedule()
    assert get_tt_block_step_decisions(decode)[request.request_id] is True
    # A stale single-slot would now read True for BOTH steps; the SchedulerOutput
    # maps stay independent.
    assert get_tt_block_step_decisions(prefill)[request.request_id] is False

    # Commit the decode block: reads the decode SchedulerOutput's decision and
    # accepts the full-width block, not the anchor's width.
    block = list(range(10, 10 + CANVAS))
    outputs = scheduler.update_from_output(decode, _runner_output(decode, block))
    assert outputs[0].outputs[0].new_token_ids == block
    assert request.num_output_placeholders == 0


def test_adaptive_block_width_mismatch_raises():
    """If the scheduler reserves a block but the model returns a different
    width (gate disagreement), fail loudly rather than leak placeholders."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    _adaptive_anchor(scheduler, request)
    submitted = scheduler.schedule()
    assert get_tt_block_step_decisions(submitted)[request.request_id] is True
    with pytest.raises(ValueError, match="block gates disagree"):
        # Solo decode reserved a block; a width-1 output contradicts it.
        scheduler.update_from_output(submitted, _runner_output(submitted, [7]))


def test_adaptive_over_frontier_prompt_never_blocks():
    """A prompt over tt_adaptive_block_max_prompt_tokens is served as plain
    baseline for its whole lifetime: width-1 reservation even on solo decode.
    On the old code the solo decode reserved the full block and the model's
    width-1 baseline output killed the engine."""
    from vllm_tt_plugin.config import store_tt_adaptive_block_max_prompt_tokens

    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    # frontier below this request's 32-token prompt
    store_tt_adaptive_block_max_prompt_tokens(scheduler.vllm_config, 16)
    scheduler._adaptive_block_max_prompt = 16
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    _adaptive_anchor(scheduler, request)

    submitted = scheduler.schedule()  # solo decode -- but over the frontier
    assert get_tt_block_step_decisions(submitted)[request.request_id] is False
    assert request.num_output_placeholders == 1
    outputs = scheduler.update_from_output(submitted, _runner_output(submitted, [9]))
    assert outputs[0].outputs[0].new_token_ids == [9]
    assert request.num_output_placeholders == 0


def test_adaptive_under_frontier_prompt_still_blocks():
    from vllm_tt_plugin.config import store_tt_adaptive_block_max_prompt_tokens

    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    store_tt_adaptive_block_max_prompt_tokens(scheduler.vllm_config, 64)
    scheduler._adaptive_block_max_prompt = 64
    request = _request(CANVAS * 2)  # 32-token prompt <= 64
    scheduler.add_request(request)
    _adaptive_anchor(scheduler, request)

    submitted = scheduler.schedule()
    assert get_tt_block_step_decisions(submitted)[request.request_id] is True
    assert request.num_output_placeholders == CANVAS


def test_adaptive_solo_decode_by_a_non_owner_drops_the_session():
    """A solo decode step whose single request does not own the session must
    clear ownership. Async scheduling reaches this: upstream skips a request
    that has hit max_tokens (guarded on num_output_placeholders), so the OWNER
    can drop out of a step while its session is still armed, leaving a peer
    alone. On the old code ownership survived ("a SOLO decode leaves ownership
    alone"), the model kept an armed session it would have served the peer
    from, and the peer committed the owner's speculated block against a single
    reserved placeholder.
    """
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    req_a = _request(CANVAS * 2, request_id="req-a")
    scheduler.add_request(req_a)
    _adaptive_anchor(scheduler, req_a)
    assert scheduler._spec_session_owner == "req-a"

    # req-b is running and decoding; req-a is still alive but not in this step.
    req_b = _request(CANVAS * 2, request_id="req-b")
    scheduler.requests[req_b.request_id] = req_b
    req_b.num_computed_tokens = req_b.num_prompt_tokens + 1

    scheduler._mirror_spec_session(
        SimpleNamespace(num_scheduled_tokens={"req-b": 1}), solo=True
    )
    assert scheduler._spec_session_owner is None


def test_adaptive_solo_decode_by_the_owner_keeps_the_session():
    """The companion of the above: the owner's own solo decode step is the
    steady state and must not disturb ownership."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    req_a = _request(CANVAS * 2, request_id="req-a")
    scheduler.add_request(req_a)
    _adaptive_anchor(scheduler, req_a)
    assert scheduler._spec_session_owner == "req-a"

    req_a.num_computed_tokens = req_a.num_prompt_tokens + 1
    scheduler._mirror_spec_session(
        SimpleNamespace(num_scheduled_tokens={"req-a": 1}), solo=True
    )
    assert scheduler._spec_session_owner == "req-a"


def test_adaptive_frontier_is_measured_on_the_replayed_length():
    """A resumed prefill replays the prompt AND the generated tokens, and the
    model measures its capture frontier on that replayed length (the runner's
    prompt_lens is input_positions + chunk_lens). The scheduler must measure
    the same quantity: on the old code it compared num_prompt_tokens, which
    never grows, so a preempted-and-resumed request crossed the frontier on
    the model side only -- the model served plain baseline while the scheduler
    reserved a block, and the width check killed the engine core.
    """
    from vllm_tt_plugin.config import store_tt_adaptive_block_max_prompt_tokens

    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    store_tt_adaptive_block_max_prompt_tokens(scheduler.vllm_config, 64)
    scheduler._adaptive_block_max_prompt = 64
    request = _request(CANVAS * 2)  # 32-token prompt, under the frontier
    scheduler.add_request(request)
    _adaptive_anchor(scheduler, request)
    assert scheduler._spec_session_owner == request.request_id

    # Resumed from preemption: the replay spans prompt + 68 generated tokens,
    # so the model sees prompt_lens=100 and drops the session at 100 > 64.
    replayed = 100
    request.num_computed_tokens = replayed
    scheduler._mirror_spec_session(
        SimpleNamespace(num_scheduled_tokens={request.request_id: replayed}),
        solo=True,
    )
    assert scheduler._spec_session_owner is None


# ── KV pages must cover the whole adaptive block, not just one token ─────────


def test_block_output_reserves_lookahead_for_the_whole_block():
    """A block-output decode writes past the position upstream allocated for.

    schedule() calls allocate_slots with num_lookahead_tokens, which is 0
    without a vLLM speculative_config, and raising num_output_placeholders
    afterwards accounts for pending output tokens without allocating pages.
    The model then runs several verify iterations against vLLM-owned KV and
    refresh_page_tables pads the missing columns with zero, so the verify
    reads and writes the null block.
    """
    sched = _scheduler(output_width=CANVAS)
    assert sched.num_lookahead_tokens >= CANVAS, (
        "lookahead must cover at least the emitted block width"
    )
    assert sched.num_lookahead_tokens == 2 * CANVAS


def test_adaptive_block_reserves_the_same_lookahead():
    sched = _scheduler(output_width=CANVAS, adaptive=True, max_num_seqs=4)
    assert sched.num_lookahead_tokens == 2 * CANVAS


def test_allocate_slots_is_asked_for_the_block_footprint(monkeypatch):
    """Victor's step 1, asserted where it happens.

    schedule() passes num_lookahead_tokens straight to allocate_slots. With no
    vLLM speculative_config that value is 0, so the first decode after prefill
    reserves for ONE token while the step goes on to commit a whole block and
    verify past it. Spying on the call is what discriminates: checking the
    resulting page count does not, because the prefill already covers this
    range and the shortfall only appears at a later page crossing.
    """
    sched = _scheduler(output_width=CANVAS, adaptive=True, max_num_seqs=4)
    seen: list[int] = []
    real = sched.kv_cache_manager.allocate_slots

    def spy(request, num_new_tokens, *a, **k):
        seen.append(int(k.get("num_lookahead_tokens", 0) or 0))
        return real(request, num_new_tokens, *a, **k)

    monkeypatch.setattr(sched.kv_cache_manager, "allocate_slots", spy)

    req = _request(max_tokens=4 * CANVAS)
    sched.add_request(req)
    out = sched.schedule()  # prefill
    sched.update_from_output(out, _runner_output(out, [1]))
    seen.clear()
    sched.schedule()  # first decode: this is the call that under-reserved

    assert seen, "the first decode must allocate"
    assert max(seen) >= CANVAS, (
        f"allocate_slots asked for lookahead {seen}, but one block step commits "
        f"CANVAS={CANVAS} tokens and verifies past them"
    )


def test_releasing_another_request_leaves_the_session_owners_block_intact():
    """vllm-tt-plugin#118.2: adaptive serving admits several live requests while
    the paired adapter keeps ONE spec session, so releasing a request that does
    NOT own that session must leave the owner's width alone.

    Before the adapters' ``release_request`` compared the released slot against
    the session owner, any release cleared the global session: the owner then
    decoded ONE baseline token against the block-width reservation this
    scheduler had already made for it, and ``_update_request_with_output``
    rejected the step. This pins the scheduler half of that contract -- the
    width RESERVED for the owner and the width ACCEPTED from it, together --
    across an abort of a different live request.
    """
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    req_a = _request(CANVAS * 2, request_id="req-a")
    req_b = _request(CANVAS * 2, request_id="req-b")

    # A prefills and takes the session.
    scheduler.add_request(req_a)
    _adaptive_anchor(scheduler, req_a, token=5)
    assert scheduler._spec_session_owner == "req-a"

    # B prefills. A prefill re-seats the session, so B becomes the owner while
    # A stays live -- exactly the state where a release can hit the wrong one.
    scheduler.add_request(req_b)
    submitted = scheduler.schedule()
    assert list(submitted.num_scheduled_tokens) == ["req-b"], (
        "a TT step is never mixed prefill+decode, so B's prefill must be solo"
    )
    scheduler.update_from_output(submitted, _runner_output(submitted, [6]))
    assert scheduler._spec_session_owner == "req-b"

    # The client aborts A before B's first decode.
    scheduler.finish_requests("req-a", RequestStatus.FINISHED_ABORTED)
    assert "req-a" not in scheduler.requests
    assert scheduler._spec_session_owner == "req-b", (
        "releasing a NON-owner must not move the session off its owner"
    )

    # B still gets the whole block reserved, and its full-width output is taken.
    submitted = scheduler.schedule()
    assert list(submitted.num_scheduled_tokens) == ["req-b"]
    assert get_tt_block_step_decisions(submitted)["req-b"] is True
    assert req_b.num_output_placeholders == CANVAS

    block = list(range(20, 20 + CANVAS))
    outputs = scheduler.update_from_output(submitted, _runner_output(submitted, block))
    assert outputs[0].outputs[0].new_token_ids == block
    assert req_b.num_output_placeholders == 0


def test_a_block_width_output_on_a_baseline_step_is_refused():
    """vllm-tt-plugin#118 review (r4045325441): when the adaptive scheduler
    stamps a step width 1 -- batched, or a prompt over the spec frontier -- the
    model owes exactly one baseline token.

    The baseline branch used to delegate straight to super(), which appends
    whatever it is handed. A model returning a BLOCK there would commit K tokens
    against a single reserved placeholder, and the disagreement would surface
    later as a placeholder leak or a corrupted continuation rather than at its
    cause. Refuse it, naming the request and both widths.
    """
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    req_a = _request(CANVAS * 2, request_id="req-a")
    req_b = _request(CANVAS * 2, request_id="req-b")
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)

    # Batched prefill: both commit their anchors, so neither owns the session
    # and the next step is stamped width 1 for both.
    submitted = scheduler.schedule()
    assert len(submitted.num_scheduled_tokens) == 2
    scheduler.update_from_output(
        submitted,
        ModelRunnerOutput(
            req_ids=["req-a", "req-b"],
            req_id_to_index={"req-a": 0, "req-b": 1},
            sampled_token_ids=[[5], [6]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    submitted = scheduler.schedule()
    for req in (req_a, req_b):
        assert get_tt_block_step_decisions(submitted)[req.request_id] is False
        assert req.num_output_placeholders == 1

    # req-a returns a full block on a step scheduled for one token.
    with pytest.raises(ValueError, match="violates the scheduled baseline width"):
        scheduler.update_from_output(
            submitted,
            ModelRunnerOutput(
                req_ids=["req-a", "req-b"],
                req_id_to_index={"req-a": 0, "req-b": 1},
                sampled_token_ids=[list(range(CANVAS)), [7]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )


def test_a_narrow_block_with_a_wide_verify_reserves_the_physical_extent():
    """Twice the output width is not a bound when verify is wider (#118 finding 2).

    Victor's configuration and arithmetic: GEMMA4_DFLASH_SERVE_BLOCK=2 with
    GEMMA4_DFLASH_VERIFY=7 emits an output width of 2 and uses eight physical
    verification rows. A 123-token prompt in 128-token blocks then decodes with
    ``123 + 1 + 2 * 2 = 128`` reserved positions, which is exactly one block
    (0..127), while the decoder writes verification positions 123..130 -- so
    128, 129 and 130 have no request block behind them.

    Twice the width only covers the extent while the block is at least as wide
    as the verification, which is true at the shipped default (64 vs 8) and
    false here. The model declares what it actually touches, and the scheduler
    honours whichever bound is larger.
    """
    # Undeclared: the old behaviour, which is short for this configuration.
    bare = _scheduler(output_width=2, adaptive=True, max_num_seqs=4)
    assert bare.num_lookahead_tokens == 4, "2 * output width"
    assert bare.num_lookahead_tokens < 2 + 8, (
        "this is the under-reservation: the step touches block + verify rows"
    )

    # Declared: block (2) + physical verification rows (V=7 -> N=8).
    sched = _scheduler(output_width=2, adaptive=True, max_num_seqs=4, kv_extent=2 + 8)
    assert sched.num_lookahead_tokens == 10, (
        "the declared physical extent must win when it exceeds twice the width"
    )
    # 123 + 1 + 10 = 134 > 128, so a second block is allocated and positions
    # 128..130 are backed.
    assert 123 + 1 + sched.num_lookahead_tokens > 128


def test_the_default_block_width_is_unchanged_by_the_declaration():
    """At the shipped default the multiple already dominates, so nothing moves.

    Guards against the extent declaration quietly inflating the reservation for
    every request on the default configuration: 2 * 64 = 128 already exceeds
    64 + 8, so the declared value must not change the answer.
    """
    sched = _scheduler(
        output_width=CANVAS, adaptive=True, max_num_seqs=4, kv_extent=CANVAS + 8
    )
    assert sched.num_lookahead_tokens == 2 * CANVAS
