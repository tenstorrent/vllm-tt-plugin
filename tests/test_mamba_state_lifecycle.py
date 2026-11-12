# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Qwen3.8 GDN state in vLLM ``MambaSpec`` blocks over whole request lifetimes.

The fake model has the per-device layout of Qwen3.8-27B at TP=8: 64 decoder
layers, every fourth one full attention (16 layers, 1 KV head of 256 in
bfloat8_b, 544 B per token and layer) and the other 48 gated delta network
(GDN) layers, each with conv state 3 x 1280 bf16 and recurrent state
6 x 128 x 128 fp32. The attention block size is 1024 tokens. Cache mode
``none`` runs without prefix caching; cache mode ``align`` runs with it and
1024-token GDN blocks. Neither uses speculative decoding.

Each test builds the KV cache config with vLLM's own planner, then drives a
real ``TTScheduler`` with its ``KVCacheManager`` and a real ``TTModelRunner``
step by step. The pool is small, so requests finish, get aborted, are
preempted and resume, and their rows move while others run. Every step checks
the runner's ``mamba_state`` three ways:

- against vLLM's allocation: each row's start and destination blocks are the
  ones in the scheduler's own block list for that request (``KVCacheManager``),
  not in the runner's copy, and each prefill checkpoint is the one
  ``MambaManager`` reserved for the step. ``MambaManager._checkpoints`` is a
  private vLLM 0.31 structure that only this test reads; it pins the runner's
  use of the public ``compute_mamba_prefill_checkpoints`` to the scheduler;
- against upstream: in mode ``none`` the block
  ``mamba_get_block_table_tensor`` selects, in mode ``align`` the source and
  destination of the copy ``preprocess_mamba`` makes before the forward;
- against the state the blocks hold: ``StateLedger`` replays what the model
  contract asks of the recurrent kernels (block operations first, then read the
  start state from the start block unless the row has none, write a prefill
  checkpoint and the end state, padding rows write anything into block 0), so
  a row must find the state of exactly the tokens it has computed, whoever
  owned the block before. In mode ``align`` every block vLLM keeps in its
  prefix cache must also hold the state of exactly the tokens its hash covers.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
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
    get_block_hash,
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker import mamba_utils

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


def _qwen38_specs(mode: str = "none", checkpoint_alignment: int | None = None):
    attention = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.bfloat16,
        state_content_bytes=ATTENTION_BYTES_PER_TOKEN,
    )
    gdn = MambaSpec(
        # Upstream ``MambaModelConfig`` resolves ``mamba_block_size`` to
        # max_model_len in mode none; mode align needs the attention block
        # size.
        block_size=MAX_MODEL_LEN if mode == "none" else BLOCK_SIZE,
        shapes=GDN_SHAPES,
        dtypes=GDN_DTYPES,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode=mode,
        num_prefill_checkpoint_blocks=int(checkpoint_alignment is not None),
        prefill_checkpoint_alignment=checkpoint_alignment,
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


def _scheduler(
    kv_cache_config,
    max_num_batched_tokens: int,
    mode: str = "none",
    prefix_match_unit: int | None = None,
    prefill_chunk_alignment: int | None = None,
) -> TTScheduler:
    """A real ``TTScheduler`` over a real ``KVCacheManager``.

    Prefix caching is on exactly in mode ``align``.
    """
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
    cache_config.enable_prefix_caching = mode == "align"
    cache_config.mamba_cache_mode = mode
    cache_config.prefix_match_unit = prefix_match_unit
    cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    if prefill_chunk_alignment is not None:
        config.additional_config = {
            "tt": {"prefill_chunk_alignment": prefill_chunk_alignment}
        }
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

    def read(self, group: int, block: int, tokens: list[int], start: int) -> None:
        """A row's start state: none at position 0, else block ``block``'s."""
        if start == 0:
            assert block == 0, f"a row at position 0 reads block {block}"
            return
        held = self.states.get((group, block))
        assert held == tuple(tokens[:start]), (
            f"group {group} block {block} holds the state after "
            f"{'?' if held is None or held in (GARBAGE, ZEROED) else len(held)} "
            f"tokens ({held if held in (GARBAGE, ZEROED) else '...'}), but the "
            f"row starts at position {start}"
        )

    def write(self, group: int, block: int, tokens: list[int], end: int) -> None:
        self.gdn_blocks.add(block)
        self.states[(group, block)] = tuple(tokens[:end])

    def run_padding_row(self, group: int):
        self.states[(group, 0)] = GARBAGE

    def zero(self, groups: list[int], block_ids: list[int]):
        """``zero_kv_cache_blocks``: the page of each block in every layer slot."""
        for g in groups:
            for block in block_ids:
                self.states[(g, block)] = ZEROED

    def copy(self, groups: list[int], copies: list[tuple[int, int]]):
        """``copy_kv_cache_blocks``: the page of each block in every layer slot."""
        for g in groups:
            for src, dst in copies:
                self.states[(g, dst)] = self.states.get((g, src))


class UpstreamAlignRunner:
    """The Mamba bookkeeping of upstream's GPU model runner in mode ``align``.

    Calls upstream ``preprocess_mamba`` and records, instead of copying, the
    block-table columns of each copy it makes before the forward: the column
    of the request's state after its last step, and the column the step
    updates in place. ``mamba_state_idx`` is the map upstream keeps across
    steps.
    """

    def __init__(self, kv_cache_config, group_indices) -> None:
        self.kv_cache_config = kv_cache_config
        self.mamba_state_idx: dict[str, int] = {}
        self.copy_bufs = SimpleNamespace(
            mamba_group_ids=list(group_indices),
            mamba_spec=kv_cache_config.kv_cache_groups[group_indices[0]].kv_cache_spec,
            offset=0,
        )

    def columns(self, scheduler_output, row_req_ids, requests):
        """Request id -> (source column, destination column) of the step.

        Without a copy the step reads and writes its running column, and a
        request at position 0 reads nothing (source -1).
        """
        copied: dict[str, tuple[int, int]] = {}

        def record(_bufs, _config, _funcs, _groups, src, dst, _bias, req_state, _ctx):
            copied[req_state.req_id] = (src, dst)

        with (
            patch.object(mamba_utils, "collect_mamba_copy_meta", record),
            patch.object(mamba_utils, "do_mamba_copy_block", lambda _bufs: None),
        ):
            mamba_utils.preprocess_mamba(
                scheduler_output,
                self.kv_cache_config,
                SimpleNamespace(enable_prefix_caching=True),
                self.mamba_state_idx,
                SimpleNamespace(
                    req_ids=list(row_req_ids),
                    num_accepted_tokens_cpu=np.ones(len(row_req_ids), np.int32),
                ),
                {
                    r: SimpleNamespace(
                        req_id=r,
                        num_computed_tokens=requests[r].num_computed_tokens,
                        block_ids=requests[r].block_ids,
                    )
                    for r in row_req_ids
                },
                {},
                {},
                self.copy_bufs,
            )
        result = {}
        for r in row_req_ids:
            running = self.mamba_state_idx[r]
            if r in copied:
                result[r] = copied[r]
            else:
                started = requests[r].num_computed_tokens > 0
                result[r] = (running if started else -1, running)
        return result


class _Harness:
    """Drives ``TTScheduler`` and ``TTModelRunner`` through whole steps."""

    def __init__(
        self,
        num_blocks: int,
        max_num_batched_tokens: int = 2048,
        *,
        mode: str = "none",
        checkpoint_alignment: int | None = None,
        prefix_match_unit: int | None = None,
    ):
        cfg = _vllm_config(mode)
        cfg.model_config.max_model_len = MAX_MODEL_LEN
        cfg.model_config.original_max_model_len = MAX_MODEL_LEN
        cfg.scheduler_config.max_num_seqs = MAX_NUM_SEQS
        cfg.cache_config.prefix_match_unit = prefix_match_unit
        self.mode = mode
        self.kv_cache_config = _kv_cache_config(
            cfg, _qwen38_specs(mode, checkpoint_alignment), num_blocks
        )
        self.scheduler = _scheduler(
            self.kv_cache_config, max_num_batched_tokens, mode, prefix_match_unit
        )
        self.hash_block_size = self.scheduler.hash_block_size
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
        self.upstream = (
            UpstreamAlignRunner(self.kv_cache_config, self.groups)
            if mode == "align"
            else None
        )
        self.ledger = StateLedger()
        self.requests: dict[str, Request] = {}
        # Prefix hash -> the tokens it covers, from every request's hashes.
        self.hash_tokens: dict[bytes, tuple[int, ...]] = {}
        self.preempted: set[str] = set()
        self.resumed_from_zero: set[str] = set()
        self.rows_seen: dict[str, set[int]] = {}
        self.zeroed: list[int] = []
        self.copies: list[tuple[int, int]] = []
        # Request id -> start position of each step that began from a block
        # vLLM kept for the prefix cache (new and resumed requests).
        self.hits: dict[str, list[int]] = {}
        # (request id, position) of every prefill checkpoint written.
        self.checkpoints: list[tuple[str, int]] = []
        # Steps whose rows read one block and wrote another.
        self.crossings = 0
        self.cached_states_checked = 0
        self.output_base = 50_000
        self.num_steps = 0
        self.num_added = 0

    def add(
        self,
        req_id: str,
        prompt_len: int = 0,
        max_tokens: int = 8,
        prompt: list[int] | None = None,
    ) -> None:
        init_none_hash(sha256)
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, eos_token_id=2)
        # Distinct prompts, so a row that reads another request's block
        # cannot find the state it expects there.
        self.num_added += 1
        request = Request(
            request_id=req_id,
            prompt_token_ids=prompt or _prompt(self.num_added, prompt_len),
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(self.hash_block_size, sha256),
        )
        self.requests[req_id] = request
        self.scheduler.add_request(request)

    def manager(self, group: int):
        return self.scheduler.kv_cache_manager.coordinator.single_type_managers[group]

    def scheduler_blocks(self, req_id: str, group: int) -> list[int]:
        return [b.block_id for b in self.manager(group).req_to_blocks.get(req_id, [])]

    def vllm_checkpoint(self, req_id: str, group: int, start: int, end: int):
        """``(position, block)`` of the prefill checkpoint vLLM reserved for
        this step, or ``None``. ``MambaManager`` keeps a request's entry until
        a later allocation drops it, so an entry outside the step is older."""
        manager = self.manager(group)
        entry = getattr(manager, "_checkpoints", {}).get(req_id)
        if entry is None or not start < entry[0] < end:
            return None
        position, column = entry
        return position, manager.req_to_blocks[req_id][column].block_id

    def record_hashes(self) -> None:
        for request in self.requests.values():
            tokens = request.all_token_ids
            for j, block_hash in enumerate(request.block_hashes):
                self.hash_tokens.setdefault(
                    block_hash, tuple(tokens[: (j + 1) * self.hash_block_size])
                )

    def check_cached_states(self) -> None:
        """Every GDN block vLLM keeps under a prefix hash holds the state of
        exactly the tokens that hash covers."""
        self.record_hashes()
        for block in self.scheduler.kv_cache_manager.block_pool.blocks:
            key = block.block_hash
            if key is None or get_group_id(key) not in self.groups:
                continue
            tokens = self.hash_tokens[get_block_hash(key)]
            held = self.ledger.states.get((get_group_id(key), block.block_id))
            assert held == tokens, (
                f"cached GDN block {block.block_id} of group {get_group_id(key)} "
                f"is keyed on {len(tokens)} tokens but holds "
                f"{held if held is None or held in (GARBAGE, ZEROED) else len(held)}"
            )
            self.cached_states_checked += 1

    def step(self) -> bool:
        out = self.scheduler.schedule()
        self.preempted.update(out.preempted_req_ids or ())
        if not out.total_num_scheduled_tokens:
            self.runner.build_model_input(out, None)
            return bool(self.scheduler.requests)
        self.model.calls.clear()
        model_input = self.runner.build_model_input(out, None)
        # Block operations in the order the runner hands them to the model.
        for name, kwargs in self.model.calls:
            if name == "zero_kv_cache_blocks":
                self.ledger.zero(self.groups, kwargs["block_ids"])
                self.zeroed += kwargs["block_ids"]
            elif name == "copy_kv_cache_blocks":
                self.ledger.copy(self.groups, kwargs["copies"])
                self.copies += kwargs["copies"]
        row_req_ids = model_input.row_req_ids
        positions = torch.as_tensor(model_input.input_positions).reshape(-1)
        state = model_input.mamba_state
        assert [g.layer_names for g in state.groups] == [
            tuple(self.kv_cache_config.kv_cache_groups[g].layer_names)
            for g in self.groups
        ]
        upstream = (
            self.upstream.columns(out, row_req_ids, self.runner.requests)
            if self.upstream is not None
            else None
        )
        # ``schedule`` has already advanced ``Request.num_computed_tokens``.
        computed = {
            new.req_id: new.num_computed_tokens for new in out.scheduled_new_reqs
        }
        cached = out.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        fresh = {new.req_id for new in out.scheduled_new_reqs} | cached.resumed_req_ids
        rows = []
        for row, req_id in enumerate(row_req_ids):
            request = self.scheduler.requests[req_id]
            start = int(positions[row])
            end = start + out.num_scheduled_tokens[req_id]
            assert start == computed[req_id]
            self.rows_seen.setdefault(req_id, set()).add(row)
            if req_id in self.preempted and start == 0:
                self.resumed_from_zero.add(req_id)
            if req_id in fresh and start > 0:
                self.hits.setdefault(req_id, []).append(start)
            position = (
                0
                if state.checkpoint_positions is None
                else int(state.checkpoint_positions[row, 0])
            )
            if position:
                self.checkpoints.append((req_id, position))
            blocks = []
            for i, g in enumerate(self.groups):
                group = state.groups[i]
                src = int(group.start_blocks[row])
                dst = int(group.windows[row, 0])
                checkpoint = (
                    (position, int(group.checkpoint_blocks[row, 0]))
                    if position
                    else None
                )
                self.check_blocks(req_id, g, start, end, src, dst, checkpoint)
                if upstream is not None:
                    row_ids = self.runner.requests[req_id].block_ids[g]
                    src_column, dst_column = upstream[req_id]
                    assert src == (row_ids[src_column] if src_column >= 0 else 0)
                    assert dst == row_ids[dst_column]
                blocks.append((g, src, dst, checkpoint))
            rows.append((request, start, end, blocks))
        # The kernels read every start state before any row writes.
        read_only = set()
        written = set()
        for request, start, _end, blocks in rows:
            for g, src, dst, checkpoint in blocks:
                self.ledger.read(g, src, list(request.all_token_ids), start)
                if src and src != dst:
                    read_only.add(src)
                written.add(dst)
                if checkpoint:
                    written.add(checkpoint[1])
        assert not read_only & written, (
            f"blocks {sorted(read_only & written)} are a start state and written"
        )
        self.crossings += bool(read_only)
        sampled: list[list[int]] = []
        for request, start, end, blocks in rows:
            tokens = list(request.all_token_ids)
            for g, _src, dst, checkpoint in blocks:
                if checkpoint:
                    self.ledger.write(g, checkpoint[1], tokens, checkpoint[0])
                self.ledger.write(g, dst, tokens, end)
            emits = end >= request.num_tokens
            sampled.append([self.output_base + self.num_steps] if emits else [])
        for i, g in enumerate(self.groups):
            group = state.groups[i]
            padding = len(row_req_ids)
            assert set(group.windows[padding:, 0].tolist()) <= {0}
            assert set(group.start_blocks[padding:].tolist()) <= {0}
            self.ledger.run_padding_row(g)
        if self.mode == "align":
            self.check_cached_states()
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

    def check_blocks(self, req_id, group, start, end, src, dst, checkpoint) -> None:
        """A row's blocks are the ones vLLM allocated and, in mode none,
        upstream ``mamba_get_block_table_tensor`` selects."""
        held = self.scheduler_blocks(req_id, group)
        if self.mode == "none":
            upstream = mamba_get_block_table_tensor(
                torch.tensor([held], dtype=torch.int32),
                torch.tensor([end]),
                self.kv_cache_config.kv_cache_groups[group].kv_cache_spec,
                "none",
            )
            assert held == [dst] == upstream[0].tolist(), (req_id, group, held)
            assert src == (dst if start else 0)
        else:
            assert dst == held[(end - 1) // BLOCK_SIZE], (req_id, group, held)
            assert src == (held[(start - 1) // BLOCK_SIZE] if start else 0), (
                req_id,
                group,
                held,
            )
        assert checkpoint == self.vllm_checkpoint(req_id, group, start, end), (
            req_id,
            group,
            checkpoint,
        )

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


@pytest.mark.parametrize("prefix_match_unit", [None, 128])
@pytest.mark.parametrize("max_num_batched_tokens", [1024, 2048, 4096])
def test_prefill_checkpoints_match_the_blocks_mambamanager_reserves(
    prefix_match_unit, max_num_batched_tokens
):
    """The runner finds every prefill checkpoint vLLM reserves, and no other.

    The runner applies vLLM's public rule, ``compute_mamba_prefill_checkpoints``;
    ``_Harness.check_blocks`` compares each row with the checkpoint vLLM 0.31's
    real ``MambaManager`` reserved for the step (``_checkpoints``) and with the
    block in the scheduler's own block list. Prompts at and around block
    boundaries, with and without prefix hits, in steps that hold whole prompts
    and steps that chunk them.
    """
    h = _Harness(
        num_blocks=48,
        max_num_batched_tokens=max_num_batched_tokens,
        mode="align",
        checkpoint_alignment=64,
        prefix_match_unit=prefix_match_unit,
    )
    shared = _prompt(100, 3000)
    for i, prompt_len in enumerate((1, 1023, 1024, 1025, 2048, 2500, 3000)):
        h.add(f"cold-{i}", prompt=_prompt(200 + i, prompt_len), max_tokens=4)
    h.run()
    for i, prompt_len in enumerate((2500, 3000, 2100)):
        h.add(f"warm-{i}", prompt=shared[:prompt_len], max_tokens=4)
        h.run()

    assert h.hits, "no request hit the prefix cache"
    # A checkpoint needs a chunk that ends the prefill and passes a whole block
    # boundary after its start, which a 1024-token step never holds.
    assert bool(h.checkpoints) == (max_num_batched_tokens > BLOCK_SIZE)


def test_prefix_match_unit_off_the_prefill_chunk_grid_fails_at_startup():
    """With a 64-token ``prefix_match_unit`` vLLM ends prefill chunks on
    multiples of 64, which the default 128-token chunk grid floors back to
    the chunk start; the scheduler refuses to start rather than stall."""
    kv_cache_config = _Harness(num_blocks=32, mode="align").kv_cache_config

    with pytest.raises(ValueError, match="prefix_match_unit=64"):
        _scheduler(kv_cache_config, 2048, "align", prefix_match_unit=64)
    _scheduler(kv_cache_config, 2048, "align", prefix_match_unit=128)
