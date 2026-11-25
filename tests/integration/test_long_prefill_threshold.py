# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``long_prefill_token_threshold`` on a TT prefill step beside running decodes.

vLLM's ``Scheduler.schedule`` caps each prefill chunk at
``long_prefill_token_threshold`` so that a long prompt cannot starve the other
requests, and lifts the cap when ``get_request_counts`` reports one request
only. ``TTScheduler._schedule_prefill_only`` runs that base pass with the
running decodes removed from ``running``. The decodes still wait for the whole
prefill step, so they must still count as requests the prefill can starve.

These tests drive a real ``TTScheduler`` over a real ``KVCacheManager`` in the
order of the synchronous ``EngineCore.step``: schedule a step, then commit the
output the TT model runner gives for it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

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
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.scheduler import TTScheduler

BLOCK_SIZE = 16
MAX_MODEL_LEN = 512
MAX_NUM_SEQS = 4
# Token budget of one step (``max_num_batched_tokens``).
BUDGET = 384
# A multiple of the default ``prefill_chunk_alignment`` (128), so a capped
# chunk already ends on the alignment grid.
THRESHOLD = 128
SHORT_PROMPT = 16
# Shorter than the budget, so one step could prefill all of it, and longer
# than two thresholds, so its second chunk is capped too.
LONG_PROMPT = 300
# Longer than the budget, so only the budget splits it.
LONE_PROMPT = 500
SAMPLED_TOKEN = 7
LOCAL_MODEL_CONFIG = Path(__file__).parent.parent / "model_configs" / "qwen2"


class _ChunkedPrefillModel:
    """Stands in for a TT model class whose generator can resume a prefill."""

    model_capabilities = {"supports_chunked_prefill": True}


@contextmanager
def _stub_model_resolution() -> Iterator[None]:
    """Resolve ``_ChunkedPrefillModel`` instead of a tt-metal class in
    ``VllmConfig``, so the platform hook keeps chunked prefill and the
    threshold."""
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
            return_value=(_ChunkedPrefillModel, None),
        ),
    ):
        yield


def _scheduler() -> TTScheduler:
    """A real synchronous ``TTScheduler`` with chunked prefill and the threshold."""
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=BUDGET,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=THRESHOLD,
        async_scheduling=False,
        is_encoder_decoder=False,
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
    num_blocks = MAX_NUM_SEQS * MAX_MODEL_LEN // BLOCK_SIZE + 1
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


def _request(request_id: str, prompt_len: int) -> Request:
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * prompt_len,
        sampling_params=params,
        pooling_params=None,
    )


def _step(scheduler: TTScheduler) -> SchedulerOutput:
    """Schedule one step and commit it as the TT model runner reports it.

    A request whose scheduled tokens complete its prompt samples
    ``SAMPLED_TOKEN``; an intermediate prefill chunk samples nothing.
    """
    step = scheduler.schedule()
    req_ids = list(step.num_scheduled_tokens)
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: row for row, req_id in enumerate(req_ids)},
            sampled_token_ids=[
                [] if scheduler.requests[req_id].is_prefill_chunk else [SAMPLED_TOKEN]
                for req_id in req_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return step


def _with_running_decodes(count: int) -> TTScheduler:
    """A scheduler whose ``count`` short requests are in steady decode."""
    scheduler = _scheduler()
    for index in range(count):
        scheduler.add_request(_request(f"decode-{index}", SHORT_PROMPT))
    _step(scheduler)
    # A decode step ends the prefill run, so the decode-interleave policy gives
    # the next two prefill steps to the prompts below.
    decode = _step(scheduler)
    assert set(decode.num_scheduled_tokens) == {
        f"decode-{index}" for index in range(count)
    }
    return scheduler


def test_a_new_long_prompt_beside_running_decodes_is_capped():
    scheduler = _with_running_decodes(3)
    scheduler.add_request(_request("long", LONG_PROMPT))

    step = _step(scheduler)

    assert step.num_scheduled_tokens == {"long": THRESHOLD}
    # The hidden decodes count only during the prefill pass; lane routing and
    # metrics read the true queues between steps.
    assert scheduler.get_request_counts() == (4, 0)


def test_a_continuation_beside_running_decodes_is_capped():
    # A second new prompt makes the first chunk a two-request pass, which the
    # base scheduler caps whether or not decodes are hidden. The next step
    # advances ``long`` alone beside the decodes, ``short`` among them.
    scheduler = _with_running_decodes(2)
    scheduler.add_request(_request("long", LONG_PROMPT))
    scheduler.add_request(_request("short", SHORT_PROMPT))
    first = _step(scheduler)
    assert first.num_scheduled_tokens == {"long": THRESHOLD, "short": SHORT_PROMPT}

    second = _step(scheduler)

    assert second.num_scheduled_tokens == {"long": THRESHOLD}


def test_a_lone_long_prompt_keeps_the_whole_budget():
    # Nothing else is running or waiting, so there is nobody to starve and
    # vLLM lifts the cap, for the first chunk and for the continuation alike.
    scheduler = _scheduler()
    scheduler.add_request(_request("long", LONE_PROMPT))

    chunks = [_step(scheduler).num_scheduled_tokens for _ in range(2)]

    assert chunks == [{"long": BUDGET}, {"long": LONE_PROMPT - BUDGET}]
