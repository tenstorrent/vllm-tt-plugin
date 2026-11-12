# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Running requests that a forced prefix-cache reset resumes get new page tables.

``reset_prefix_cache(reset_running_requests=True)`` preempts every running
request and resumes it in the same step, from position 0 on newly allocated
blocks. vLLM sends such a request as a cached request in ``resumed_req_ids``
with its whole new block list. The test drives a real ``TTScheduler`` with its
``KVCacheManager`` and a real ``TTModelRunner`` over an attention-only model,
and compares each step's page-table row with the scheduler's own blocks for
the request.
"""

from __future__ import annotations

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
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler
from vllm_tt_plugin.worker import _available_kv_cache_memory_bytes_for_num_blocks

BLOCK_SIZE = 64
MAX_MODEL_LEN = 2048
NUM_BLOCKS = 64
# tests/model_configs/qwen2: one layer, 2 KV heads of 8.
SPECS = {
    "model.layers.0.self_attn": FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
}
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"


class _StubModel:
    """Stands in for the TT model class while the platform hook runs."""


class _Model:
    """The KV allocation hook and the forward entry points the runner calls."""

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return ["kv"]

    def prefill_forward(self, **kwargs):
        pass

    def decode_forward(self, **kwargs):
        pass


def _vllm_config() -> VllmConfig:
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
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
            scheduler_config=SchedulerConfig(
                max_num_seqs=4,
                max_num_batched_tokens=MAX_MODEL_LEN,
                max_model_len=MAX_MODEL_LEN,
                enable_chunked_prefill=False,
                async_scheduling=False,
                is_encoder_decoder=False,
            ),
            model_config=model_config,
            cache_config=CacheConfig(
                block_size=BLOCK_SIZE, gpu_memory_utilization=0.9, cache_dtype="auto"
            ),
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )
    config.cache_config.num_gpu_blocks = NUM_BLOCKS
    # The engine core resolves it before the planner runs; TT supports LBNHC.
    config.cache_config.kv_cache_layout = KVCacheLayout.LBNHC.name
    return config


class _Harness:
    """Drives ``TTScheduler`` and ``TTModelRunner`` through whole steps."""

    def __init__(self):
        config = _vllm_config()
        available = _available_kv_cache_memory_bytes_for_num_blocks(
            config, SPECS, NUM_BLOCKS
        )
        [kv_cache_config] = get_kv_cache_configs(config, [SPECS], [available])
        assert kv_cache_config.num_blocks == NUM_BLOCKS
        self.scheduler = TTScheduler(
            vllm_config=config,
            kv_cache_config=kv_cache_config,
            block_size=BLOCK_SIZE,
            hash_block_size=BLOCK_SIZE,
            log_stats=True,
            structured_output_manager=StructuredOutputManager(config),
        )
        self.runner = TTModelRunner(
            vllm_config=config,
            mesh_device=None,
            trace_mode="none",
            enable_model_warmup=False,
            num_devices=1,
        )
        self.runner.model = _Model()
        self.runner.initialize_kv_cache(kv_cache_config)
        self.num_steps = 0
        self.resumed: set[str] = set()

    def add(self, req_id: str, prompt_len: int, max_tokens: int) -> None:
        init_none_hash(sha256)
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, eos_token_id=2)
        self.scheduler.add_request(
            Request(
                request_id=req_id,
                prompt_token_ids=list(range(10, 10 + prompt_len)),
                sampling_params=params,
                pooling_params=None,
                block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
            )
        )

    def step(self) -> bool:
        out = self.scheduler.schedule()
        self.resumed |= out.scheduled_cached_reqs.resumed_req_ids
        model_input = self.runner.build_model_input(out, None)
        if model_input is None:
            return bool(self.scheduler.requests)
        manager = self.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
        sampled = []
        for row, req_id in enumerate(model_input.row_req_ids):
            held = [b.block_id for b in manager.req_to_blocks[req_id]]
            page_table = model_input.block_tables[row].tolist()
            assert page_table[: len(held)] == held, (req_id, page_table, held)
            assert not any(page_table[len(held) :]), (req_id, page_table, held)
            request = self.scheduler.requests[req_id]
            end = request.num_computed_tokens
            sampled.append(
                [20_000 + self.num_steps] if end >= request.num_tokens else []
            )
        emitted = [(r, t) for r, t in zip(model_input.row_req_ids, sampled) if t]
        if emitted:
            self.runner._apply_sampled_tokens_to_state(
                torch.tensor([t for _, t in emitted], dtype=torch.int32),
                req_ids=[r for r, _ in emitted],
            )
        self.scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=list(model_input.row_req_ids),
                req_id_to_index={r: i for i, r in enumerate(model_input.row_req_ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        self.num_steps += 1
        return True


def test_forced_reset_resumes_running_requests_on_their_new_blocks():
    """Requests a and b decode, then the prefix cache is reset with
    ``reset_running_requests``. vLLM resumes both from position 0 in the next
    step, on new blocks. Their persistent-batch rows still hold the freed
    blocks; a runner that keeps the rows appends the new blocks behind the
    freed ones, and the page table sends attention to freed pages."""
    h = _Harness()
    h.add("a", 300, max_tokens=40)
    h.add("b", 150, max_tokens=40)
    for _ in range(4):
        h.step()
    assert {"a", "b"} <= set(h.runner.input_batch.req_id_to_index)

    assert h.scheduler.reset_prefix_cache(reset_running_requests=True)
    for _ in range(200):
        if not h.step():
            break

    assert not h.scheduler.requests
    assert h.resumed == {"a", "b"}
