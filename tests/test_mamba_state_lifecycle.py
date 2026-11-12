# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Qwen3.8 GDN state in vLLM ``MambaSpec`` blocks over whole request lifetimes.

The fake model has the per-device layout of Qwen3.8-27B at TP=8: 64 decoder
layers, every fourth one full attention (16 layers, 1 KV head of 256 in
bfloat8_b, 544 B per token and layer) and the other 48 gated delta network
(GDN) layers, each with conv state 3 x 1280 bf16 and recurrent state
6 x 128 x 128 fp32. The attention block size is 1024 tokens; cache mode is
``none`` without speculative decoding.

Each test builds the KV cache config with vLLM's own planner, then drives a
real ``TTScheduler`` with its ``KVCacheManager`` and a real ``TTModelRunner``
step by step. The pool is small, so requests finish, get aborted, are
preempted and resume, and their rows move while others run. Every step checks
the runner's ``mamba_state`` two ways:

- against upstream: each row's block is the one upstream
  ``mamba_get_block_table_tensor`` selects from the scheduler's own block list
  for that request (``KVCacheManager``), not from the runner's copy;
- against the state the blocks hold: ``StateLedger`` replays what the model
  contract asks of the recurrent kernels (read the start state from the block
  unless the row starts at position 0, write the end state into the same
  block, padding rows write anything into block 0), so a row must find the
  state of exactly the tokens it has computed, whoever owned the block before.
"""

from __future__ import annotations

from pathlib import Path
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
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.test_mamba_kv_cache import (
    _kv_cache_config,
    _RecordingModel,
    _vllm_config,
)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler

NUM_LAYERS = 64
BLOCK_SIZE = 1024
# 2 (K and V) x 256 x 1088 / 1024 bytes: bfloat8_b packs 1024 values into 1088
# bytes. bfloat8_b has no torch dtype, so the attention spec states its bytes
# through ``state_content_bytes``.
ATTENTION_BYTES_PER_TOKEN = 544
GDN_SHAPES = ((3, 1280), (6, 128, 128))
GDN_DTYPES = (torch.bfloat16, torch.float32)
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 4
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"
# The state a block holds after padding rows wrote into it, and after the
# model zeroed it.
GARBAGE = ("garbage",)
ZEROED = ("zeroed",)


def _is_attention(layer: int) -> bool:
    return layer % 4 == 3


def _qwen38_specs() -> dict:
    attention = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.bfloat16,
        state_content_bytes=ATTENTION_BYTES_PER_TOKEN,
    )
    gdn = MambaSpec(
        # Upstream ``MambaModelConfig`` resolves ``mamba_block_size`` to
        # max_model_len in mode none.
        block_size=MAX_MODEL_LEN,
        shapes=GDN_SHAPES,
        dtypes=GDN_DTYPES,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="none",
    )
    return {
        (
            f"model.layers.{i}.self_attn"
            if _is_attention(i)
            else f"model.layers.{i}.linear_attn"
        ): (attention if _is_attention(i) else gdn)
        for i in range(NUM_LAYERS)
    }


class _StubModel:
    """Stands in for the TT model class while the platform hook runs."""


def _scheduler(kv_cache_config, max_num_batched_tokens: int) -> TTScheduler:
    """A real ``TTScheduler`` over a real ``KVCacheManager``, prefix caching off."""
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        async_scheduling=False,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE, gpu_memory_utilization=0.9, cache_dtype="auto"
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
    cache_config.enable_prefix_caching = False
    cache_config.mamba_cache_mode = "none"
    cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    # The stub declares no chunked-prefill support, so the platform hook turned
    # chunked prefill off; these tests split long prompts into chunks.
    config.scheduler_config.enable_chunked_prefill = True
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    # As ``EngineCore`` resolves them from the groups.
    block_size, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, config)
    cache_config.block_size = min(
        g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups
    )
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        hash_block_size=hash_block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _prompt(seed: int, length: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 30_000, (length,), generator=g).tolist()


class StateLedger:
    """The state each (group, block) holds, as the token prefix it covers."""

    def __init__(self) -> None:
        self.states: dict[tuple[int, int], tuple] = {}
        # Every block a GDN group held for a request.
        self.gdn_blocks: set[int] = set()

    def run_row(self, group: int, block: int, tokens: list[int], start: int, end: int):
        """One row's recurrent kernel: read the start state, write the end state."""
        self.gdn_blocks.add(block)
        if start > 0:
            held = self.states.get((group, block))
            assert held == tuple(tokens[:start]), (
                f"group {group} block {block} holds the state after "
                f"{'?' if held is None else len(held)} tokens, but the row "
                f"starts at position {start}"
            )
        self.states[(group, block)] = tuple(tokens[:end])

    def run_padding_row(self, group: int):
        self.states[(group, 0)] = GARBAGE

    def zero(self, groups: list[int], block_ids: list[int]):
        """``zero_kv_cache_blocks``: the page of each block in every layer slot."""
        for g in groups:
            for block in block_ids:
                self.states[(g, block)] = ZEROED


class _Harness:
    """Drives ``TTScheduler`` and ``TTModelRunner`` through whole steps."""

    def __init__(self, num_blocks: int, max_num_batched_tokens: int = 2048):
        cfg = _vllm_config()
        cfg.model_config.max_model_len = MAX_MODEL_LEN
        cfg.model_config.original_max_model_len = MAX_MODEL_LEN
        cfg.scheduler_config.max_num_seqs = MAX_NUM_SEQS
        self.kv_cache_config = _kv_cache_config(cfg, _qwen38_specs(), num_blocks)
        self.scheduler = _scheduler(self.kv_cache_config, max_num_batched_tokens)
        self.model = _RecordingModel()
        self.runner = TTModelRunner(
            vllm_config=cfg,
            mesh_device=None,
            trace_mode="none",
            enable_model_warmup=False,
            num_devices=1,
        )
        self.runner.model = self.model
        self.runner.initialize_kv_cache(self.kv_cache_config)
        self.groups = list(self.runner._mamba_groups.group_indices)
        self.ledger = StateLedger()
        self.preempted: set[str] = set()
        self.resumed_from_zero: set[str] = set()
        self.rows_seen: dict[str, set[int]] = {}
        self.zeroed: list[int] = []
        self.output_base = 50_000
        self.num_steps = 0
        self.num_added = 0

    def add(self, req_id: str, prompt_len: int, max_tokens: int) -> None:
        init_none_hash(sha256)
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, eos_token_id=2)
        # Distinct prompts, so a row that reads another request's block
        # cannot find the state it expects there.
        self.num_added += 1
        self.scheduler.add_request(
            Request(
                request_id=req_id,
                prompt_token_ids=_prompt(self.num_added, prompt_len),
                sampling_params=params,
                pooling_params=None,
                block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
            )
        )

    def scheduler_blocks(self, req_id: str, group: int) -> list[int]:
        manager = self.scheduler.kv_cache_manager.coordinator.single_type_managers[
            group
        ]
        return [b.block_id for b in manager.req_to_blocks.get(req_id, [])]

    def step(self) -> bool:
        out = self.scheduler.schedule()
        self.preempted.update(out.preempted_req_ids or ())
        if not out.total_num_scheduled_tokens:
            self.runner.build_model_input(out, None)
            return bool(self.scheduler.requests)
        self.model.calls.clear()
        model_input = self.runner.build_model_input(out, None)
        zeroed = [
            b
            for name, kwargs in self.model.calls
            if name == "zero_kv_cache_blocks"
            for b in kwargs["block_ids"]
        ]
        self.ledger.zero(self.groups, zeroed)
        self.zeroed += zeroed
        row_req_ids = model_input.row_req_ids
        positions = torch.as_tensor(model_input.input_positions).reshape(-1)
        state = model_input.mamba_state
        assert [g.layer_names for g in state.groups] == [
            tuple(self.kv_cache_config.kv_cache_groups[g].layer_names)
            for g in self.groups
        ]
        # ``schedule`` has already advanced ``Request.num_computed_tokens``.
        computed = {
            new.req_id: new.num_computed_tokens for new in out.scheduled_new_reqs
        }
        cached = out.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        sampled: list[list[int]] = []
        for row, req_id in enumerate(row_req_ids):
            request = self.scheduler.requests[req_id]
            tokens = list(request.all_token_ids)
            start = int(positions[row])
            end = start + out.num_scheduled_tokens[req_id]
            assert start == computed[req_id]
            self.rows_seen.setdefault(req_id, set()).add(row)
            if req_id in self.preempted and start == 0:
                self.resumed_from_zero.add(req_id)
            for i, g in enumerate(self.groups):
                block = int(state.groups[i].windows[row, 0])
                # Upstream's choice from the scheduler's own allocation.
                held = self.scheduler_blocks(req_id, g)
                upstream = mamba_get_block_table_tensor(
                    torch.tensor([held], dtype=torch.int32),
                    torch.tensor([end]),
                    self.kv_cache_config.kv_cache_groups[g].kv_cache_spec,
                    "none",
                )
                assert held == [block] == upstream[0].tolist(), (req_id, g, held)
                self.ledger.run_row(g, block, tokens, start, end)
            emits = end >= request.num_tokens
            sampled.append([self.output_base + self.num_steps] if emits else [])
        for i, g in enumerate(self.groups):
            padding = state.groups[i].windows[len(row_req_ids) :, 0]
            assert set(padding.tolist()) <= {0}
            self.ledger.run_padding_row(g)
        # The runner keeps its own copy of each request's tokens, as its
        # sampling path does after a forward.
        emitted = [(r, t) for r, t in zip(row_req_ids, sampled) if t]
        if emitted:
            self.runner._apply_sampled_tokens_to_state(
                torch.tensor([t for _, t in emitted], dtype=torch.int32),
                req_ids=[r for r, _ in emitted],
            )
        self.scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=list(row_req_ids),
                req_id_to_index={r: i for i, r in enumerate(row_req_ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        self.num_steps += 1
        return True

    def run(self, max_steps: int = 400, abort: dict[int, str] | None = None) -> None:
        abort = dict(abort or {})
        for _ in range(max_steps):
            if self.num_steps in abort:
                self.scheduler.finish_requests(
                    abort.pop(self.num_steps), RequestStatus.FINISHED_ABORTED
                )
            if not self.step():
                return
        raise AssertionError("scenario did not finish")


def test_qwen38_layers_form_four_groups_of_16_with_one_page():
    """vLLM groups the 48 GDN and 16 attention layers into 4 groups of 16.

    The GDN state (400,896 B) is smaller than the 1024-token attention page
    (557,056 B), so vLLM pads the GDN page to the attention page and keeps the
    attention block size.
    """
    h = _Harness(num_blocks=32)
    groups = h.kv_cache_config.kv_cache_groups
    assert [type(g.kv_cache_spec) for g in groups] == [MambaSpec] * 3 + [
        FullAttentionSpec
    ]
    assert all(len(g.layer_names) == 16 for g in groups)
    assert {g.kv_cache_spec.page_size_bytes for g in groups} == {
        BLOCK_SIZE * ATTENTION_BYTES_PER_TOKEN
    }
    assert groups[3].kv_cache_spec.block_size == BLOCK_SIZE
    assert h.runner._mamba_groups.attention_group_index == 3


def test_requests_finish_and_their_blocks_serve_later_requests():
    """Completion: freed GDN blocks come back to new requests.

    Two waves of requests share a pool that holds one wave. A block that held
    a finished request's GDN state can go to another request's GDN group,
    which starts from zero state, or to an attention group, which vLLM lists
    for zeroing. Chunked prefill splits the 3000-token prompt.
    """
    h = _Harness(num_blocks=24)
    for i, prompt_len in enumerate((3000, 700, 1500, 300)):
        h.add(f"first-{i}", prompt_len, max_tokens=8)
    h.run()
    for i, prompt_len in enumerate((900, 2500, 400)):
        h.add(f"second-{i}", prompt_len, max_tokens=8)

    h.run()

    assert not h.scheduler.requests
    assert set(h.ledger.gdn_blocks) & set(h.zeroed), (
        "no GDN block went back to attention"
    )


def test_abort_preemption_and_row_movement_keep_each_state_with_its_request():
    """Abort, preemption and row movement in one run.

    The pool holds 14 blocks: three 1000-token requests with one attention
    block and three GDN blocks each, so the fourth waits until request r2 is
    aborted mid-decode. Decoding past the 1024-token boundary needs a second
    attention block per request, and the pool cannot give every request one:
    the scheduler preempts. A preempted request loses its blocks and resumes as
    a prefill from position 0. The abort, the preemption and finishing
    requests move the remaining rows.
    """
    h = _Harness(num_blocks=15)
    for i in range(MAX_NUM_SEQS):
        h.add(f"r{i}", 1000, max_tokens=60 + 10 * i)

    h.run(abort={12: "r2"})

    assert not h.scheduler.requests
    assert h.preempted, "the pool was large enough to avoid preemption"
    assert h.preempted <= h.resumed_from_zero | {"r2"}
    assert any(len(rows) > 1 for rows in h.rows_seen.values()), "no row moved"


@pytest.mark.parametrize("prompt_len", [1, 1023, 1024, 1025])
def test_state_block_is_fixed_across_block_boundaries(prompt_len):
    """In mode none a request's GDN block does not change when its attention
    blocks grow, whatever the prompt length."""
    h = _Harness(num_blocks=16)
    h.add("a", prompt_len, max_tokens=40)
    blocks: set[tuple[int, int]] = set()
    while h.scheduler.requests:
        held = [h.scheduler_blocks("a", g) for g in h.groups]
        h.step()
        if "a" in h.scheduler.requests:
            blocks.add(tuple(h.scheduler_blocks("a", g)[0] for g in h.groups))
            assert all(len(row) <= 1 for row in held)
    assert len(blocks) == 1
