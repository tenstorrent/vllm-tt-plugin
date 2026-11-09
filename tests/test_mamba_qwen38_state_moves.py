# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Qwen3.8-27B GDN state in vLLM ``MambaSpec`` blocks, driven by vLLM 0.31.

The fake model has the per-device layout of Qwen3.8-27B at TP=8: 64 decoder
layers, every fourth one full attention (16 layers, 1 KV head of 256 in
bfloat8_b, 544 B per token and layer) and the other 48 gated delta network
(GDN) layers, each with conv state 3 x 1280 bf16 and recurrent state
6 x 128 x 128 fp32. MTP speculative decoding proposes K=3 drafts. The block
size is 1024 tokens.

Each test builds the KV cache config with vLLM's own planner, then drives a
real ``TTScheduler`` and its ``KVCacheManager`` step by step and checks the
block ids against ``tests/mamba_runner_reference.py``, a reference of the
upstream GPU model runner's Mamba state moves: the state window of each row
(``mamba_get_block_table_tensor``), the copy from the previous to the current
block-table column (``preprocess_mamba``) and the boundary copy after draft
acceptance (``postprocess_mamba_align_gpu``). ``StateLedger`` replays the
recurrent kernels' reads and writes, so every read must find the state of
exactly the tokens the row has computed, also after a prefix-cache hit.

Without drafts (K=0) the TT runner's own bookkeeping runs too
(``TTModelRunner.build_model_input``) and must produce the reference's block
ids. With K=3 the TT runner's ``MambaCacheGroups`` runs next to the reference
and must make the same moves; a second ledger checks its reads in the TT state
layout, where each block holds one full state.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
    SpeculativeConfig,
    VllmConfig,
)
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.mamba_runner_reference import (
    BlockLedger,
    StateCopy,
    StateLedger,
    UpstreamMambaRunner,
    label,
    state_window,
)
from tests.test_mamba_kv_cache import (
    MAX_BATCH,
    _kv_cache_config,
    _new_request,
    _RecordingModel,
    _step,
    _vllm_config,
)
from vllm_tt_plugin import scheduler as scheduler_module
from vllm_tt_plugin.config import store_tt_spec_plan
from vllm_tt_plugin.mamba_state import MambaCacheGroups, mamba_cache_groups
from vllm_tt_plugin.model_input import TTModelInput
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.scheduler import TTScheduler, get_tt_mamba_checkpoints
from vllm_tt_plugin.spec_admission import (
    MODEL_OWNED_DRAFT_METHOD,
    MODEL_OWNED_DRAFT_SENTINEL,
)
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    DRAFTER_STATE_INTERNAL,
    SpecPlan,
)

NUM_LAYERS = 64
BLOCK_SIZE = 1024
K = 3
# 2 (K and V) x 256 x 1088 / 1024 bytes: bfloat8_b packs 1024 values into 1088
# bytes. bfloat8_b has no torch dtype, so the attention spec states its bytes
# through ``state_content_bytes``.
ATTENTION_BYTES_PER_TOKEN = 544
PAGE_BYTES = BLOCK_SIZE * ATTENTION_BYTES_PER_TOKEN
GDN_SHAPES = ((3, 1280), (6, 128, 128))
GDN_DTYPES = (torch.bfloat16, torch.float32)
GDN_STATE_BYTES = 3 * 1280 * 2 + 6 * 128 * 128 * 4
MAX_MODEL_LEN = 8192
NUM_BLOCKS = 96
MAX_NUM_SEQS = 4
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"


def _is_attention(layer: int) -> bool:
    return layer % 4 == 3


def _qwen38_specs(
    mode: str,
    num_spec: int = K,
    block_size: int = BLOCK_SIZE,
    gdn_block_size: int | None = None,
    checkpoint: bool = False,
) -> dict:
    """``checkpoint``: GDN groups with one prefill checkpoint block, as a model
    whose prefill writes the state at vLLM's checkpoint declares it."""
    attention = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.bfloat16,
        state_content_bytes=ATTENTION_BYTES_PER_TOKEN,
    )
    gdn = MambaSpec(
        # Upstream ``MambaModelConfig`` resolves ``mamba_block_size`` to
        # max_model_len in mode none and to the attention block size in mode
        # align; the TT model sets the spec itself.
        block_size=gdn_block_size or (MAX_MODEL_LEN if mode == "none" else block_size),
        shapes=GDN_SHAPES,
        dtypes=GDN_DTYPES,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode=mode,
        num_speculative_blocks=num_spec,
        num_prefill_checkpoint_blocks=int(checkpoint),
        prefill_checkpoint_alignment=128 if checkpoint else None,
    )
    return {
        (
            f"model.layers.{i}.self_attn"
            if _is_attention(i)
            else f"model.layers.{i}.linear_attn"
        ): (attention if _is_attention(i) else gdn)
        for i in range(NUM_LAYERS)
    }


def _planner_config(mode: str, retention: int | None = 0, speculative_config=None):
    cfg = _vllm_config(mode)
    cfg.model_config.max_model_len = MAX_MODEL_LEN
    cfg.model_config.original_max_model_len = MAX_MODEL_LEN
    cfg.cache_config.prefix_cache_retention_interval = retention
    cfg.speculative_config = speculative_config
    return cfg


def _plan(num_spec: int) -> SpecPlan:
    return SpecPlan(
        effective_k=num_spec,
        lanes_per_request=1,
        extra_bytes_per_seq=0,
        extra_bytes_per_token=0,
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
        drafter_state=DRAFTER_STATE_INTERNAL,
        supports_narrow_decode=False,
    )


class _StubModel:
    """Stands in for the TT model class while the platform hook runs."""

    model_capabilities = {
        "supports_chunked_prefill": True,
        "supports_prefix_caching": True,
    }


def _engine_config(
    mode: str,
    retention: int | None,
    max_num_batched_tokens: int,
    prefix_match_unit: int | None = None,
    model_class: type = _StubModel,
    mamba_block_size: int | None = None,
    max_model_len: int = MAX_MODEL_LEN,
    block_size: int = BLOCK_SIZE,
    long_prefill_token_threshold: int = 0,
    tt_config: dict | None = None,
) -> VllmConfig:
    """A ``VllmConfig`` resolved by the TT platform hook for ``model_class``.

    The cache mode a hybrid model gets from vLLM's config resolution
    (``MambaModelConfig``) is set before the hook runs: mode align comes with
    prefix caching. ``tt_config`` is ``additional_config["tt"]``.
    """
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = max_model_len
    scheduler_config = SchedulerConfig(
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=long_prefill_token_threshold,
        async_scheduling=False,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=mode == "align",
        mamba_cache_mode=mode,
        prefix_cache_retention_interval=retention,
        prefix_match_unit=prefix_match_unit,
        mamba_block_size=mamba_block_size,
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
            return_value=(model_class, None),
        ),
    ):
        return VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            additional_config={"tt": dict(tt_config)} if tt_config else {},
        )


def _scheduler(
    kv_cache_config,
    mode: str,
    num_spec: int,
    retention: int | None,
    max_num_batched_tokens: int,
    prefix_match_unit: int | None = None,
    **engine_kwargs,
) -> TTScheduler:
    """A real ``TTScheduler`` over a real ``KVCacheManager``, started as
    ``EngineCore`` starts it: ``block_size`` is set from the groups and
    validated. The model declares chunked prefill and prefix caching.
    ``engine_kwargs`` go to ``_engine_config``.
    """
    config = _engine_config(
        mode, retention, max_num_batched_tokens, prefix_match_unit, **engine_kwargs
    )
    cache_config = config.cache_config
    cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    if num_spec:
        config.speculative_config = SpeculativeConfig(
            method=MODEL_OWNED_DRAFT_METHOD,
            model=MODEL_OWNED_DRAFT_SENTINEL,
            num_speculative_tokens=num_spec,
            target_model_config=config.model_config,
            target_parallel_config=config.parallel_config,
        )
        store_tt_spec_plan(config, _plan(num_spec))
    # As ``EngineCore`` resolves them from the groups.
    block_size, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, config)
    cache_config.block_size = min(
        g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups
    )
    config.validate_block_size()
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        hash_block_size=hash_block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _tt_runner(cfg, kv_cache_config, model, num_spec: int = 0) -> TTModelRunner:
    """A ``TTModelRunner`` over ``kv_cache_config``; with ``num_spec`` drafts
    it speculates with a host n-gram proposer."""
    plan = _plan(num_spec) if num_spec else None
    if num_spec:
        cfg.speculative_config = SimpleNamespace(
            method="ngram", num_speculative_tokens=num_spec
        )
    with patch("vllm_tt_plugin.model_runner.get_tt_spec_plan", return_value=plan):
        runner = TTModelRunner(
            vllm_config=cfg,
            mesh_device=None,
            trace_mode="none",
            enable_model_warmup=False,
            num_devices=1,
        )
    runner.model = model
    runner.initialize_kv_cache(kv_cache_config)
    return runner


def _prompt(seed: int, length: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 30_000, (length,), generator=g).tolist()


@dataclass
class _Row:
    req_id: str
    num_computed: int
    num_scheduled: int
    drafts: list[int]
    windows: dict[int, list[int]]


@dataclass
class _Step:
    num_scheduled: dict[str, int]
    new_hits: dict[str, int]
    # Block-table rows of the requests new in this step, per group.
    new_block_ids: dict[str, list[list[int]]]
    rows: list[_Row]
    pre_copies: list[StateCopy]
    post_copies: list[StateCopy]
    # The upstream block a row's recurrent kernel reads its initial temporal
    # state from, per group.
    initial_blocks: dict[str, dict[int, int]] = field(default_factory=dict)
    # ``TTModelRunner.build_model_input``: the Mamba state copies handed to
    # ``copy_kv_cache_blocks`` and window column 0 per row and group.
    runner_copies: list[tuple[int, int]] | None = None
    runner_state_ids: dict[str, dict[int, int]] = field(default_factory=dict)
    # With drafts: the runner's windows and start-state counts, and the copies
    # ``commit_spec_acceptance`` hands to ``copy_kv_cache_blocks``.
    runner_windows: dict[str, dict[int, list[int]]] = field(default_factory=dict)
    runner_counts: dict[str, int] = field(default_factory=dict)
    runner_post: list[tuple[int, int]] | None = None
    runner_input: TTModelInput | None = None
    # The TT runner's ``MambaCacheGroups``: copies before the forward and after
    # acceptance, each row's window per group and its start-state count.
    groups_pre: list[tuple[int, int]] | None = None
    groups_post: list[tuple[int, int]] | None = None
    groups_windows: dict[str, dict[int, list[int]]] = field(default_factory=dict)
    groups_counts: dict[str, int] = field(default_factory=dict)
    # Request id -> the positions of the states its prefill chunk left inside.
    checkpoints: dict[str, list[int]] = field(default_factory=dict)


class _Harness:
    """Drives ``TTScheduler`` and the upstream reference through whole steps.

    Each step: the scheduler schedules; the upstream runner's view of every
    request (``req_states``: computed tokens and append-only block-table rows)
    is updated as the GPU runner's ``_update_states`` does; the reference runs
    its preprocess copies and selects each row's state window; the ledger
    checks the initial-state reads and records the writes; acceptance is
    simulated; the reference runs its postprocess copies; the scheduler gets
    the sampled tokens and the next drafts.

    With ``plugin=True`` the TT runner builds its model input from the same
    scheduler output. With ``plugin_groups`` (the default) the TT runner's
    ``MambaCacheGroups`` runs as ``TTModelRunner`` calls it (``begin_step``,
    ``step_inputs``, ``after_acceptance`` on a verify's commit), and a second
    ledger, in the TT state layout, checks its reads.
    """

    def __init__(
        self,
        mode: str,
        num_spec: int,
        retention: int | None = 0,
        max_num_batched_tokens: int = 2048,
        plugin: bool = False,
        plugin_groups: bool = True,
        prefix_match_unit: int | None = None,
        checkpoint: bool = False,
        block_size: int = BLOCK_SIZE,
    ):
        self.mode = mode
        self.num_spec = num_spec
        self.cfg = _planner_config(mode, retention)
        self.kv_cache_config = _kv_cache_config(
            self.cfg,
            _qwen38_specs(mode, num_spec, block_size=block_size, checkpoint=checkpoint),
            NUM_BLOCKS,
        )
        # A model that declares checkpoint blocks writes the checkpoint state;
        # False stands for one that ignores the plugin's checkpoint inputs.
        self.writes_checkpoints = True
        # Mode align holds the running block and K draft blocks, plus a block
        # passed until the next step frees it. With a checkpoint the step also
        # holds the checkpoint block, and vLLM keeps the previous chunk's K
        # draft blocks in place (``allocate_new_blocks`` relocates them only
        # without a checkpoint) until the next step frees them.
        self.max_live_state_blocks = 2 + num_spec + (1 + num_spec) * checkpoint
        self.scheduler = _scheduler(
            self.kv_cache_config,
            mode,
            num_spec,
            retention,
            max_num_batched_tokens,
            prefix_match_unit,
            block_size=block_size,
        )
        self.reference = UpstreamMambaRunner(self.kv_cache_config)
        self.groups = self.reference.group_indices
        self.ledger = StateLedger(self.groups)
        self.req_states: dict[str, SimpleNamespace] = {}
        # Width of the GPU runner's block-table row (``max_num_blocks_per_req``).
        spec = self.reference.spec
        self.width = spec.max_num_blocks_per_req(
            SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode=mode)),
            MAX_MODEL_LEN,
        )
        self.model = _RecordingModel() if plugin else None
        self.runner = (
            _tt_runner(self.cfg, self.kv_cache_config, self.model, num_spec)
            if plugin
            else None
        )
        self.plugin_groups = (
            mamba_cache_groups(
                self.kv_cache_config.kv_cache_groups,
                SimpleNamespace(
                    mamba_cache_mode=mode, prefix_cache_retention_interval=retention
                ),
                MAX_MODEL_LEN,
            )
            if plugin_groups
            else None
        )
        self.plugin_ledger = BlockLedger()
        # ``TTModelRunner._req_accepted_counts``: set by a verify's commit,
        # dropped by a step that does not verify.
        self.plugin_counts: dict[str, int] = {}
        self.steps: list[_Step] = []
        # Each request's tokens (prompt and output) after its last step.
        self.tokens: dict[str, list[int]] = {}
        self.draft_base = 70_000
        self.bonus_base = 50_000

    def add(
        self,
        req_id: str,
        prompt: list[int],
        max_tokens: int = 64,
        images: Sequence[tuple[int, int]] = (),
    ) -> None:
        """Queue a request; ``images`` are (offset, length) placeholder ranges."""
        init_none_hash(sha256)
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, eos_token_id=2)
        if images and not self.scheduler.max_num_encoder_input_tokens:
            # The text-only model config gives the scheduler no encoder budget.
            self.scheduler.max_num_encoder_input_tokens = MAX_MODEL_LEN
            self.scheduler.encoder_cache_manager = EncoderCacheManager(MAX_MODEL_LEN)
        self.scheduler.add_request(
            Request(
                request_id=req_id,
                prompt_token_ids=list(prompt),
                sampling_params=params,
                pooling_params=None,
                mm_features=[
                    MultiModalFeatureSpec(
                        data=None,
                        modality="image",
                        identifier=f"image-{offset}-{length}",
                        mm_position=PlaceholderRange(offset=offset, length=length),
                    )
                    for offset, length in images
                ]
                or None,
                block_hasher=get_request_block_hasher(
                    self.scheduler.hash_block_size, sha256
                ),
            )
        )

    def manager_blocks(self, req_id: str, group: int) -> list[int]:
        manager = self.scheduler.kv_cache_manager.coordinator.single_type_managers[
            group
        ]
        return [b.block_id for b in manager.req_to_blocks.get(req_id, [])]

    def live_state_blocks(self, req_id: str, group: int) -> int:
        return sum(1 for b in self.manager_blocks(req_id, group) if b != 0)

    def run(
        self,
        max_steps: int = 200,
        accept: Callable[[str, int, list[int]], int] = lambda r, s, d: len(d),
        until: Callable[[], bool] | None = None,
    ) -> None:
        for _ in range(max_steps):
            if until is not None and until():
                return
            if not self.step(accept):
                return
        raise AssertionError("scenario did not finish")

    def _groups_step_inputs(self, out, req_ids: list[str], record: _Step) -> None:
        """``MambaCacheGroups.step_inputs`` over this step's rows."""
        width = self.plugin_groups.max_num_blocks(MAX_MODEL_LEN)
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        tables = [
            torch.zeros((len(req_ids), width), dtype=torch.int32)
            for _ in range(num_groups)
        ]
        for g in self.groups:
            for i, r in enumerate(req_ids):
                row = self.req_states[r].block_ids[g]
                tables[g][i, : len(row)] = torch.tensor(row, dtype=torch.int32)
        windows, counts = self.plugin_groups.step_inputs(
            tables,
            req_ids,
            {r: self.req_states[r].num_computed_tokens for r in req_ids},
            out.num_scheduled_tokens,
            len(req_ids),
        )
        by_group = dict(zip(self.plugin_groups.group_indices, windows))
        for i, r in enumerate(req_ids):
            record.groups_windows[r] = {g: by_group[g][i].tolist() for g in self.groups}
            record.groups_counts[r] = 1 if counts is None else int(counts[i])

    def _runner_commit(self, model_input, sampled, req_ids, record: _Step) -> None:
        """``TTModelRunner.commit_spec_acceptance`` for a verify step, with
        this step's accepted tokens as the walk's result."""
        rows = model_input.input_positions.shape[0]
        committed = torch.zeros((rows, 1 + self.num_spec), dtype=torch.int32)
        counts = torch.ones(rows, dtype=torch.int32)
        by_req = dict(zip(req_ids, sampled))
        for row, req_id in enumerate(model_input.row_req_ids):
            tokens = by_req[req_id]
            committed[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
            counts[row] = len(tokens)
        self.model.calls.clear()
        # The scheduler gets this harness's drafts, so the runner proposes none.
        with patch.object(self.runner, "_propose_ngram_drafts"):
            self.runner.commit_spec_acceptance(model_input, committed, counts, None)
        record.runner_post = [
            pair
            for name, kwargs in self.model.calls
            if name == "copy_kv_cache_blocks"
            for pair in kwargs["copies"]
        ]

    def _write_checkpoint(self, out, req_id: str, tokens: list[int], record) -> None:
        """The states a prefill chunk leaves inside itself: upstream's kernel
        writes the state at each block boundary it passes into that column's
        block; the TT model writes where ``MambaCacheGroups.checkpoint_inputs``
        says. Only columns vLLM holds a block for matter."""
        entries = get_tt_mamba_checkpoints(out).get(req_id, ())
        for position, column in entries:
            for g in self.groups:
                block = self.req_states[req_id].block_ids[g][column]
                self.ledger.write(g, [block], [label(tokens[:position])], False)
        if not entries or self.plugin_groups is None or not self.writes_checkpoints:
            return
        blocks, positions = self.plugin_groups.checkpoint_inputs(
            get_tt_mamba_checkpoints(out), [req_id], self.req_states, 1
        )
        for entry, position in enumerate(positions[0].tolist()):
            if position:
                for group_blocks in blocks:
                    self.plugin_ledger.write(
                        [int(group_blocks[0, entry])], [label(tokens[:position])]
                    )
        record.checkpoints[req_id] = [p for p, _ in entries]

    def step(self, accept: Callable[[str, int, list[int]], int]) -> bool:
        scheduler = self.scheduler
        out = scheduler.schedule()
        if not out.num_scheduled_tokens:
            return False
        if out.preempted_req_ids:
            raise AssertionError(f"unexpected preemption {out.preempted_req_ids}")
        for req_id in out.finished_req_ids:
            self.req_states.pop(req_id, None)
            self.reference.forget(req_id)
            self.plugin_counts.pop(req_id, None)
        new_hits = {}
        new_block_ids = {}
        for new in out.scheduled_new_reqs:
            new_block_ids[new.req_id] = [list(ids) for ids in new.block_ids]
            self.req_states[new.req_id] = SimpleNamespace(
                num_computed_tokens=new.num_computed_tokens,
                block_ids=[list(ids) for ids in new.block_ids],
            )
            new_hits[new.req_id] = new.num_computed_tokens
        cached = out.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            state = self.req_states[req_id]
            state.num_computed_tokens = cached.num_computed_tokens[i]
            if cached.new_block_ids[i] is not None:
                for row, ids in zip(state.block_ids, cached.new_block_ids[i]):
                    row.extend(ids)

        zero = out.new_block_ids_to_zero or ()
        vllm_copies = [
            (c.src_block_id, c.dst_block_id) for c in out.kv_cache_block_copies or ()
        ]
        self.ledger.zero(zero)
        self.ledger.copy(
            [StateCopy(g, s, s, 0, d) for s, d in vllm_copies for g in self.groups]
        )
        self.plugin_ledger.zero(zero)
        self.plugin_ledger.copy(vllm_copies)
        req_ids = list(out.num_scheduled_tokens)
        record = _Step(
            dict(out.num_scheduled_tokens), new_hits, new_block_ids, [], [], []
        )
        if self.plugin_groups is not None:
            record.groups_pre = self.plugin_groups.begin_step(
                out,
                self.req_states,
                {r: self.req_states[r].num_computed_tokens for r in req_ids},
                self.plugin_counts,
            )
            self.plugin_ledger.copy(record.groups_pre)
            self._groups_step_inputs(out, req_ids, record)
        record.pre_copies = (
            self.reference.preprocess(out, req_ids, self.req_states)
            if self.mode == "align"
            else []
        )
        self.ledger.copy(record.pre_copies)

        if self.runner is not None:
            self.model.calls.clear()
            model_input = self.runner.build_model_input(out, None)
            record.runner_input = model_input
            copies = [
                pair
                for name, kwargs in self.model.calls
                if name == "copy_kv_cache_blocks"
                for pair in kwargs["copies"]
            ]
            record.runner_copies = copies[len(vllm_copies) :]
            state = model_input.mamba_state
            group_states = dict(
                zip(self.runner._mamba_groups.group_indices, state.groups)
            )
            for row, req_id in enumerate(model_input.row_req_ids or []):
                record.runner_state_ids[req_id] = {
                    g: int(group.windows[row, 0]) for g, group in group_states.items()
                }
                if state.accepted_counts is not None:
                    record.runner_windows[req_id] = {
                        g: group.windows[row].tolist()
                        for g, group in group_states.items()
                    }
                    record.runner_counts[req_id] = int(state.accepted_counts[row])

        accepted: dict[str, int] = {}
        verified: dict[str, int] = {}
        sampled: list[list[int]] = []
        for row_idx, req_id in enumerate(req_ids):
            request = scheduler.requests[req_id]
            state = self.req_states[req_id]
            nc = state.num_computed_tokens
            ns = out.num_scheduled_tokens[req_id]
            drafts = list(out.scheduled_spec_decode_tokens.get(req_id, ()))
            speculating = req_id in out.scheduled_spec_decode_tokens
            seq_len = nc + ns
            tokens = list(request.all_token_ids)
            expect = label(tokens[:nc])
            query = tokens[nc : nc + ns - len(drafts)] + drafts
            if speculating:
                labels = [
                    label(tokens[:nc] + query[: t + 1]) for t in range(len(query))
                ]
            else:
                labels = [label(tokens[:nc] + query)]
            windows = {}
            for g in self.groups:
                window = state_window(
                    state.block_ids[g], seq_len, self.reference.spec, self.width
                )
                windows[g] = window
                # Every block the window names is one the scheduler holds for
                # this request at that column.
                held = self.manager_blocks(req_id, g)
                start = (
                    0
                    if self.mode == "none"
                    else (seq_len - 1) // self.reference.block_size
                )
                assert held[start : start + len(window)] == window, (
                    req_id,
                    g,
                    held,
                    window,
                )
                assert 0 not in window, (req_id, g, window)
                num_accepted = self.reference.num_accepted.get(req_id, 1)
                shift = num_accepted - 1 if speculating else 0
                record.initial_blocks.setdefault(req_id, {})[g] = window[shift]
                if nc > 0:
                    got = self.ledger.initial_state(
                        g, window, num_accepted, speculating
                    )
                    assert got == (expect, expect), (req_id, g, nc, window)
                self.ledger.write(g, window, labels, speculating)
                if self.plugin_groups is not None:
                    plugin_window = record.groups_windows[req_id][g]
                    if nc > 0:
                        got = self.plugin_ledger.read(
                            plugin_window, record.groups_counts[req_id]
                        )
                        assert got == expect, ("tt", req_id, g, nc, plugin_window)
                    self.plugin_ledger.write(plugin_window, labels)
            self._write_checkpoint(out, req_id, tokens, record)
            record.rows.append(_Row(req_id, nc, ns, drafts, windows))

            emits_token = nc + ns - len(drafts) >= request.num_tokens
            if emits_token:
                a = max(0, min(accept(req_id, len(self.steps), drafts), len(drafts)))
                accepted[req_id] = a + 1
                sampled.append(
                    drafts[:a] + [self.bonus_base + 10 * len(self.steps) + row_idx]
                )
                if speculating:
                    verified[req_id] = a + 1
                    for g in self.groups:
                        self.ledger.reject(g, windows[g], a + 1)
                        if self.plugin_groups is not None:
                            self.plugin_ledger.reject(
                                record.groups_windows[req_id][g], a + 1
                            )
            else:
                accepted[req_id] = 1
                sampled.append([])

        if self.mode == "align" and self.num_spec:
            record.post_copies = self.reference.postprocess(
                out, req_ids, self.req_states, accepted
            )
            self.ledger.copy(record.post_copies)
        elif self.num_spec:
            for req_id, a in accepted.items():
                self.reference.num_accepted[req_id] = a
        if self.runner is not None and model_input.spec_mode is not None:
            self._runner_commit(model_input, sampled, req_ids, record)
        elif self.runner is not None:
            # A step that verifies nothing commits one token per row that
            # samples, as the runner's own sampling tail writes it.
            emitted = [(r, t) for r, t in zip(req_ids, sampled) if t]
            if emitted:
                self.runner._apply_sampled_tokens_to_state(
                    torch.tensor([[t[-1]] for _, t in emitted], dtype=torch.int32),
                    req_ids=[r for r, _ in emitted],
                )
        if self.plugin_groups is not None:
            for req_id in req_ids:
                self.plugin_counts.pop(req_id, None)
            self.plugin_counts.update(verified)
            record.groups_post = self.plugin_groups.after_acceptance(
                {
                    r: (self.req_states[r].num_computed_tokens, c)
                    for r, c in verified.items()
                },
                self.req_states,
            )
            self.plugin_ledger.copy(record.groups_post)

        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: i for i, r in enumerate(req_ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        if self.num_spec:
            running = [
                r for r, s in zip(req_ids, sampled) if s and r in scheduler.requests
            ]
            if running:
                scheduler.update_draft_token_ids(
                    DraftTokenIds(
                        running,
                        [
                            [
                                self.draft_base + 100 * len(self.steps) + 10 * i + j
                                for j in range(self.num_spec)
                            ]
                            for i in range(len(running))
                        ],
                    )
                )
        for req_id in req_ids:
            if req_id in scheduler.requests:
                self.tokens[req_id] = list(scheduler.requests[req_id].all_token_ids)
                for g in self.groups:
                    live = self.live_state_blocks(req_id, g)
                    if self.mode == "none":
                        assert live == 1 + self.num_spec, (req_id, g, live)
                    else:
                        assert live <= self.max_live_state_blocks, (req_id, g, live)
        self.steps.append(record)
        return True

    def finished(self, req_id: str) -> bool:
        return req_id not in self.scheduler.requests


# --- KV cache groups ---------------------------------------------------------


@pytest.mark.parametrize("mode", ["none", "align"])
def test_qwen38_layers_form_four_groups_of_16_with_equal_pages(mode):
    """vLLM groups the 48 GDN and 16 attention layers into 4 groups of 16.

    The GDN state (400,896 B) is smaller than the 1024-token attention page
    (557,056 B), and vLLM 0.31's planner pads the GDN page to the attention
    page itself: the model need not set ``page_size_padded``, and TT does not
    need upstream's ``Platform._align_hybrid_block_size``. The TT runner
    accepts the groups without drafts.
    """
    kv_cache_config = _kv_cache_config(
        _planner_config(mode), _qwen38_specs(mode, num_spec=0), NUM_BLOCKS
    )
    groups = kv_cache_config.kv_cache_groups
    assert [type(g.kv_cache_spec).__name__ for g in groups] == [
        "MambaSpec",
        "MambaSpec",
        "MambaSpec",
        "FullAttentionSpec",
    ]
    assert [len(g.layer_names) for g in groups] == [16, 16, 16, 16]
    assert GDN_STATE_BYTES == 400_896 < PAGE_BYTES == 557_056
    assert {g.kv_cache_spec.page_size_bytes for g in groups} == {PAGE_BYTES}
    gdn_block_size = MAX_MODEL_LEN if mode == "none" else BLOCK_SIZE
    assert [g.kv_cache_spec.block_size for g in groups] == [gdn_block_size] * 3 + [
        BLOCK_SIZE
    ]
    assert {t.block_stride for t in kv_cache_config.kv_cache_tensors} == {PAGE_BYTES}

    tt_groups = mamba_cache_groups(
        groups, SimpleNamespace(mamba_cache_mode=mode), MAX_MODEL_LEN
    )
    assert tt_groups.group_indices == (0, 1, 2)
    assert tt_groups.attention_group_index == 3
    assert tt_groups.block_size == gdn_block_size


def test_a_smaller_block_pads_the_attention_page_instead():
    """At 512 tokens the attention page is the smaller one, and vLLM pads it
    to the GDN state: 783 B per token instead of 544, so block sizes below 737
    tokens waste attention memory."""
    kv_cache_config = _kv_cache_config(
        _planner_config("align"),
        _qwen38_specs("align", num_spec=0, block_size=512),
        NUM_BLOCKS,
    )
    groups = kv_cache_config.kv_cache_groups
    assert [len(g.layer_names) for g in groups] == [16, 16, 16, 16]
    assert {g.kv_cache_spec.page_size_bytes for g in groups} == {GDN_STATE_BYTES}


@pytest.mark.parametrize(
    ("method", "eagle_groups"),
    [
        # vLLM 0.31 flags the group of the last registered layer for every
        # ``mtp`` drafter. Layer 63 is a full attention layer, so the attention
        # group is the one the block drop shortens.
        ("mtp", [False, False, False, True]),
        # The TT model-owned drafter registers as ``custom_class``, which is
        # not an eagle method: no group is flagged and no block is dropped.
        (MODEL_OWNED_DRAFT_METHOD, [False, False, False, False]),
    ],
)
def test_mtp_eagle_group_flag(method, eagle_groups):
    spec_config = SimpleNamespace(
        method=method,
        num_speculative_tokens=K,
        use_eagle=lambda: method == "mtp",
        use_eagle_block_drop=lambda: method == "mtp",
    )
    kv_cache_config = _kv_cache_config(
        _planner_config("align", speculative_config=spec_config),
        _qwen38_specs("align"),
        NUM_BLOCKS,
    )
    assert [g.is_eagle_group for g in kv_cache_config.kv_cache_groups] == eagle_groups


# --- Scheduler chunking -------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "chunks"),
    [
        # Mode align ends every prefill chunk but the last on a block boundary
        # (``_mamba_block_aligned_split``): column p holds the state after
        # (p + 1) * 1024 tokens.
        ("align", [1024, 1024, 1024, 428]),
        # Mode none: only TT's own prefill-chunk alignment (128 tokens) applies.
        ("none", [1408, 1408, 684]),
    ],
)
def test_prefill_chunk_ends_are_block_aligned_in_mode_align(mode, chunks):
    """In mode none ``EngineCore`` gives the scheduler the LCM of the group
    block sizes, 8192 here (the GDN block is max_model_len), and
    ``TTScheduler._align_prefill_chunk_end`` still ends chunks on its 128
    grid. Its only ``block_size`` use is the fallback for a running chunk that
    cannot reach that grid, which returns 0 for 1024 and 8192 alike, because
    a chunk with no 128 boundary inside it contains no 1024 boundary either.
    """
    h = _Harness(mode, num_spec=0, max_num_batched_tokens=1500)
    assert h.scheduler.block_size == (MAX_MODEL_LEN if mode == "none" else BLOCK_SIZE)
    h.add("a", _prompt(1, 3500), max_tokens=1)
    h.run()
    assert [s.num_scheduled["a"] for s in h.steps] == chunks


# --- Chunked prefill with prefix caching in mode align -------------------------


class _ChunkedOnlyModel:
    model_capabilities = {"supports_chunked_prefill": True}


@pytest.mark.parametrize(
    ("model_class", "mode", "chunked_mm_input_disabled"),
    [
        # vLLM rejects disable_chunked_mm_input in mode align
        # (``VllmConfig.validate_block_size``); TTScheduler keeps items whole.
        (_StubModel, "align", False),
        # The platform turns prefix caching off for a model that does not
        # declare it, which resets the mode to none; vLLM's flag applies again.
        (_ChunkedOnlyModel, "none", True),
    ],
)
def test_mode_align_engine_starts_with_chunked_prefill(
    model_class, mode, chunked_mm_input_disabled
):
    """A model that declares chunked prefill starts in the mode the platform
    resolves, and the scheduler aligns chunk ends in mode align."""
    config = _engine_config(
        "align", 0, 2048, model_class=model_class, mamba_block_size=BLOCK_SIZE
    )
    assert config.cache_config.mamba_cache_mode == mode
    assert config.scheduler_config.enable_chunked_prefill
    assert config.scheduler_config.disable_chunked_mm_input is chunked_mm_input_disabled
    config.validate_block_size()

    h = _Harness("align", num_spec=0)
    assert h.scheduler.cache_config.enable_prefix_caching
    assert h.scheduler.need_mamba_block_aligned_split
    assert h.scheduler._prefill_chunk_alignment == 128


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "chunks"),
    [
        (2500, [2048, 1024, 428]),
        # A block wider than a step: vLLM's split lets the chunk end inside the
        # block, TT floors it to its 128 grid, and the next chunk stops at the
        # block boundary (vLLM) that is also on the grid.
        (600, [512] * 6 + [428]),
    ],
)
def test_tt_chunk_grid_agrees_with_the_block_aligned_split(
    max_num_batched_tokens, chunks
):
    """vLLM's split ends chunks on 1024-token blocks, which are on TT's
    128-token grid, so ``TTScheduler._align_prefill_chunk_end`` never moves
    them, and every block boundary a chunk reaches is a chunk end."""
    h = _Harness("align", num_spec=0, max_num_batched_tokens=max_num_batched_tokens)
    h.add("a", _prompt(1, 3500), max_tokens=1)
    h.run()
    assert h.finished("a")
    assert [s.num_scheduled["a"] for s in h.steps] == chunks


def test_prefix_match_unit_off_the_chunk_grid_is_rejected():
    """With partial hash hits vLLM also stops a chunk at the prompt's last
    ``prefix_match_unit`` boundary. At 64 tokens that stop (2944 + 64 for a
    3050-token prompt) is off TT's 128 grid: the chunk end is floored back to
    the chunk start and the prefill never finishes, so startup refuses it.
    At 128 the stop is on the grid and a repeat of the prompt hits it."""
    with pytest.raises(ValueError, match="prefix_match_unit=64"):
        _Harness("align", num_spec=0, max_num_batched_tokens=8192, prefix_match_unit=64)

    h = _Harness(
        "align", num_spec=0, max_num_batched_tokens=8192, prefix_match_unit=128
    )
    prompt = _prompt(1, 3050)
    h.add("a", prompt, max_tokens=1)
    h.run(until=lambda: h.finished("a"))
    assert [s.num_scheduled["a"] for s in h.steps] == [2048, 896, 106]
    h.add("b", list(prompt), max_tokens=1)
    h.run()
    assert h.finished("b")
    assert h.steps[3].new_hits == {"b": 2944}


def _chunk_ends(h: _Harness, req_id: str) -> list[int]:
    ends, total = [], 0
    for step in h.steps:
        if req_id in step.num_scheduled:
            total += step.num_scheduled[req_id]
            ends.append(total)
    return ends


@pytest.mark.parametrize(
    ("images", "max_num_batched_tokens", "chunks"),
    [
        # The prompt is longer than a step. vLLM's split would end the first
        # chunk at 2048, inside the image; the chunk ends at the block boundary
        # before it instead. The rest fits a step and is one chunk.
        ([(1500, 1100)], 2048, [1024, 2048, 428]),
        # No block boundary before the image: the first chunk ends at its start,
        # the second takes the whole image, crossing the 1024 and 2048
        # boundaries without stopping. The rest fits a step and is one chunk,
        # past the prompt's last block boundary.
        ([(900, 1400)], 2048, [900, 1400, 1200]),
        # Two images back to back, each taken whole.
        ([(1000, 1000), (2000, 1000)], 2048, [2000, 1500]),
        # The prompt fits a step: one prefill step, although vLLM's split
        # would stop it at a block boundary.
        ([(2900, 400)], 4096, [3500]),
    ],
)
def test_multimodal_item_is_never_split_in_mode_align(
    images, max_num_batched_tokens, chunks
):
    """``TTScheduler`` schedules a multimodal prompt that fits a step as one
    chunk (``_mm_no_split_grant``) and moves every chunk end of a longer one
    out of the images (``_keep_multimodal_items_whole``). The GDN state reads
    stay exact (``StateLedger`` and ``BlockLedger``). A repeat of the prompt
    reads no prefix cache and prefills in the same chunks."""
    h = _Harness("align", num_spec=0, max_num_batched_tokens=max_num_batched_tokens)
    prompt = _prompt(1, 3500)
    h.add("a", prompt, max_tokens=1, images=images)
    h.run(until=lambda: h.finished("a"))
    assert [s.num_scheduled["a"] for s in h.steps] == chunks
    for end in _chunk_ends(h, "a"):
        assert not any(lo < end < lo + n for lo, n in images)
    h.add("b", list(prompt), max_tokens=1, images=images)
    h.run()
    assert h.finished("b")
    assert {r: n for s in h.steps for r, n in s.new_hits.items()} == {"a": 0, "b": 0}
    assert [s.num_scheduled["b"] for s in h.steps if "b" in s.num_scheduled] == chunks


def test_multimodal_item_longer_than_a_step_is_split_with_a_warning():
    """An item that skipped ``TTPlatform.validate_request`` and is longer than
    a whole step is split rather than stall the queue."""
    h = _Harness("align", num_spec=0, max_num_batched_tokens=2048)
    h.add("a", _prompt(1, 3500), max_tokens=1, images=[(500, 2500)])
    with patch.object(scheduler_module.logger, "warning") as warning:
        h.run()
    assert h.finished("a")
    assert [s.num_scheduled["a"] for s in h.steps] == [500, 524, 2048, 428]
    # One warning for the prompt longer than a step (``_mm_no_split_grant``,
    # once per scheduler) and one for the item.
    assert warning.call_count == 2


@pytest.mark.parametrize(
    ("mode", "length", "rejected"),
    [("align", 2049, True), ("align", 2048, False), ("none", 2049, False)],
)
def test_validate_request_rejects_an_item_longer_than_a_step(
    monkeypatch, mode, length, rejected
):
    """Mode align: ``TTPlatform.validate_request`` rejects an item that one
    prefill step (here ``max_num_batched_tokens``) cannot hold. Mode none
    keeps vLLM's own startup check under ``disable_chunked_mm_input``."""
    h = _Harness(mode, num_spec=0, max_num_batched_tokens=2048)
    monkeypatch.setattr(TTPlatform, "_tt_vllm_config", h.scheduler.vllm_config)
    engine_input = {
        "type": "multimodal",
        "prompt_token_ids": [1] * 3000,
        "mm_placeholders": {"image": [PlaceholderRange(offset=10, length=length)]},
    }
    params = SamplingParams(max_tokens=1)
    if rejected:
        with pytest.raises(VLLMValidationError, match="max_num_batched_tokens=2048"):
            TTPlatform.validate_request(engine_input, params)
    else:
        TTPlatform.validate_request(engine_input, params)


# --- TT runner versus upstream, no drafts --------------------------------------


@pytest.mark.parametrize("mode", ["none", "align"])
def test_tt_runner_picks_upstream_state_blocks_without_drafts(mode):
    """The TT runner hands the model the upstream state block and copies.

    Request A (3500 tokens) prefills in chunks of 2048, 1024 and 428 tokens;
    request B (1020 tokens) prefills in one chunk and decodes across the 1024
    boundary. In every step window column 0 of the TT runner's
    ``MambaGroupState.windows`` equals the upstream window column, and in mode
    align its
    ``copy_kv_cache_blocks`` pairs equal upstream ``preprocess_mamba``'s
    copies.
    """
    h = _Harness(mode, num_spec=0, max_num_batched_tokens=2048, plugin=True)
    h.add("a", _prompt(1, 3500), max_tokens=8)
    h.add("b", _prompt(2, 1020), max_tokens=8)
    h.run()
    assert h.finished("a") and h.finished("b")
    copies_seen = 0
    for step in h.steps:
        assert step.runner_copies == [(c.temporal_src, c.dst) for c in step.pre_copies]
        assert all(c.conv_src == c.temporal_src for c in step.pre_copies)
        copies_seen += len(step.pre_copies)
        for row in step.rows:
            assert step.runner_state_ids[row.req_id] == {
                g: window[0] for g, window in row.windows.items()
            }
    # Mode align: A's second and third chunks and B's decode step at 1025
    # tokens each start in a new column, in 3 GDN groups. A's first chunk
    # spans two columns but has no earlier state to move.
    assert copies_seen == (0 if mode == "none" else 3 * 3)


# --- Prefix hits ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("retention", "b_hit", "c_hit"),
    [
        # The default: only the replay boundary of the prompt keeps its state.
        # B shares 2100 tokens with A, so its 2048 boundary is not retained and
        # B hits nothing; C repeats A's prompt and hits A's last boundary.
        (0, 0, 3072),
        # Dense retention keeps every boundary state.
        (None, 2048, 3072),
    ],
)
def test_prefix_hits_land_only_on_retained_boundaries(retention, b_hit, c_hit):
    """A prefix hit hands over a block that holds exactly the hit's state.

    Request A prefills 3500 tokens and finishes. B shares its first 2100
    tokens, C all of them. The hit length and the state in the hit block
    (checked by ``StateLedger`` when B and C read it) follow
    ``prefix_cache_retention_interval``.
    """
    h = _Harness("align", num_spec=0, retention=retention)
    prompt_a = _prompt(1, 3500)
    h.add("a", prompt_a, max_tokens=2)
    h.run(until=lambda: h.finished("a"))
    h.add("b", prompt_a[:2100] + _prompt(3, 900), max_tokens=2)
    h.add("c", list(prompt_a), max_tokens=2)
    h.run()
    hits = {r: n for s in h.steps for r, n in s.new_hits.items()}
    assert hits == {"a": 0, "b": b_hit, "c": c_hit}


# --- K=3 drafts: upstream moves under the TT scheduler ------------------------


def _accept_pattern(req_id: str, step: int, drafts: list[int]) -> int:
    """Accept 0..K drafts in a fixed rotation, so every bias appears."""
    return (step + len(req_id)) % (len(drafts) + 1) if drafts else 0


def test_mode_none_holds_1_plus_k_state_blocks_and_reads_the_accepted_one():
    """Mode none, K=3: four state blocks per request and group, no copies.

    The harness asserts, every step, that each GDN group holds exactly 1 + K
    blocks per running request, that the window is those blocks, and that
    each row's initial state (temporal state from window column
    ``num_accepted - 1``, conv state from column 0 shifted by
    ``num_accepted - 1``) is the state after the row's computed tokens.
    """
    h = _Harness("none", num_spec=K, max_num_batched_tokens=2048)
    h.add("a", _prompt(1, 1500), max_tokens=40)
    h.add("bb", _prompt(2, 700), max_tokens=40)
    h.run(accept=_accept_pattern)
    assert h.finished("a") and h.finished("bb")
    drafted = [r for s in h.steps for r in s.rows if r.drafts]
    assert drafted
    assert {len(w) for r in drafted for w in r.windows.values()} == {1 + K}


def test_mode_align_moves_state_with_accepted_counts_and_boundary_copies():
    """Mode align, K=3: at most 2 + K state blocks, both copy kinds.

    Requests decode across block boundaries with partial acceptance. The
    harness asserts at most 2 + K live blocks per request and group, that
    every window block is the scheduler's block at that column, and that
    every read finds the right state after the preprocess copy (source
    columns shifted by the accepted drafts) and the postprocess boundary copy.
    The scenario must produce both kinds of copy with a nonzero shift.
    """
    h = _Harness("align", num_spec=K, retention=None, max_num_batched_tokens=2048)
    h.add("a", _prompt(1, 1010), max_tokens=60)
    h.add("bb", _prompt(2, 2030), max_tokens=60)
    h.run(accept=_accept_pattern)
    assert h.finished("a") and h.finished("bb")
    pre = [c for s in h.steps for c in s.pre_copies]
    post = [c for s in h.steps for c in s.post_copies]
    assert any(c.conv_row_offset > 0 for c in pre), pre
    assert any(c.conv_row_offset > 0 for c in post), post
    assert any(c.temporal_src != c.conv_src for c in pre + post)


def test_retention_0_runs_no_boundary_copy_and_every_read_stays_exact():
    """At retention 0 vLLM caches no decode-time boundary state, so the TT
    runner's ``MambaCacheGroups`` skips upstream's boundary copies after
    acceptance. Under the real scheduler and ``KVCacheManager`` the second
    ledger still finds every state the TT layout reads, across block
    boundaries with partial acceptance."""
    h = _Harness("align", num_spec=K, retention=0, max_num_batched_tokens=2048)
    assert h.plugin_groups.kept_boundary_interval == 0
    h.add("a", _prompt(1, 1010), max_tokens=60)
    h.add("bb", _prompt(2, 2030), max_tokens=60)
    h.run(accept=_accept_pattern)
    assert h.finished("a") and h.finished("bb")
    assert any(s.post_copies for s in h.steps), "upstream copies at boundaries"
    assert not any(s.groups_post for s in h.steps)


def test_sparse_retention_copies_only_the_boundaries_vllm_keeps():
    """Retention 2048 at block 1024: vLLM keeps A's decode-reached state at
    2048 but not at 3072. Upstream copies at both boundaries; the TT
    runner's ``MambaCacheGroups`` copies at 2048 only, every read stays
    exact, and B's prefix hit lands on 2048 and reads the copied state."""
    h = _Harness("align", num_spec=K, retention=2048)
    assert h.plugin_groups.kept_boundary_interval == 2048
    h.add("a", _prompt(1, 1900), max_tokens=1250)
    h.run(max_steps=1500, accept=_accept_pattern, until=lambda: h.finished("a"))
    upstream = [s for s in h.steps if s.post_copies]
    plugin = [s for s in h.steps if s.groups_post]
    assert len(upstream) == 2, "A passes 2048 and 3072"
    assert plugin == upstream[:1]
    h.add("b", h.tokens["a"][:3100], max_tokens=2)
    h.run(accept=_accept_pattern)
    hits = {r: n for s in h.steps for r, n in s.new_hits.items()}
    assert hits["b"] == 2048


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("variant", ["plugin", "no_zeroing", "every_block_written"])
def test_attention_never_reads_a_position_that_holds_state(mode, variant):
    """Blocks pass between the attention group and the GDN groups.

    Waves of requests finish and new ones take their blocks from vLLM's pool,
    so the real ``KVCacheManager`` gives the attention group blocks that a GDN
    group owned and blocks that only attention owned. The test replays what
    each position of each attention block holds:

    1. a block in a GDN group's row can hold recurrent state in every
       position, which reads as NaN in bf16;
    2. ``zero_kv_cache_blocks`` clears a whole block;
    3. a step writes the keys and values of its scheduled positions;
    4. the step's attention then reads every position of every block up to
       the one of its last position (masked positions included).

    No read may find state, and the runner (``MambaCacheGroups.blocks_to_zero``)
    zeroes only blocks that hold state. Both rules of the runner must matter:
    some listed blocks held no state, and some held state but their prefill
    wrote them whole. The variants are negative controls: a runner that zeroes
    nothing, and one that counts every prefill block as written whole, each
    lets a read find state.
    """
    h = _Harness(mode, num_spec=K, retention=0, plugin=True)
    groups = h.runner._mamba_groups
    if variant == "no_zeroing":
        groups.blocks_to_zero = lambda scheduler_output, requests: []
    elif variant == "every_block_written":
        groups._add_written_blocks = (
            lambda written, block_ids, first_columns, start, num_tokens: (
                written.update(
                    b for g in groups.attention_block_sizes for b in block_ids[g]
                )
            )
        )
    [attention] = groups.attention_block_sizes
    zeroed: list[int] = []
    real_zero = h.model.zero_kv_cache_blocks

    def zero_kv_cache_blocks(**kwargs):
        zeroed.extend(kwargs["block_ids"])
        return real_zero(**kwargs)

    h.model.zero_kv_cache_blocks = zero_kv_cache_blocks
    steps = []
    schedule = h.scheduler.schedule

    def schedule_and_record(*args, **kwargs):
        out = schedule(*args, **kwargs)
        starts = {new.req_id: new.num_computed_tokens for new in out.scheduled_new_reqs}
        cached = out.scheduled_cached_reqs
        starts.update(zip(cached.req_ids, cached.num_computed_tokens))
        # Taken before a request that finishes gives its blocks back.
        gdn = {
            b
            for req_id in h.scheduler.requests
            for g in h.groups
            for b in h.manager_blocks(req_id, g)
            if b
        }
        rows = {
            req_id: (start, start + out.num_scheduled_tokens[req_id])
            for req_id, start in starts.items()
        }
        attention_rows = {r: h.manager_blocks(r, attention) for r in rows}
        steps.append((out, gdn, rows, attention_rows))
        return out

    h.scheduler.schedule = schedule_and_record
    for i in range(16):
        h.add(f"r{i}", _prompt(100 + i, 1100 + 97 * i), max_tokens=6)

    state: dict[int, np.ndarray] = {}

    def holds(block: int) -> np.ndarray:
        return state.setdefault(block, np.zeros(BLOCK_SIZE, dtype=bool))

    needed = skipped_clear = skipped_written = 0
    for _ in range(400):
        zeroed.clear()
        if not h.step(_accept_pattern):
            break
        out, gdn, rows, attention_rows = steps[-1]
        listed = out.new_block_ids_to_zero or []
        for block in listed:
            if not holds(block).any():
                skipped_clear += 1
            elif block in zeroed:
                needed += 1
            else:
                skipped_written += 1
        for block in zeroed:
            assert holds(block).any(), f"block {block} held no state"
            holds(block)[:] = False
        for req_id, (start, end) in rows.items():
            for position in range(start, end):
                block = attention_rows[req_id][position // BLOCK_SIZE]
                holds(block)[position % BLOCK_SIZE] = False
        for req_id, (start, end) in rows.items():
            for block in attention_rows[req_id][: (end - 1) // BLOCK_SIZE + 1]:
                if holds(block).any():
                    assert variant != "plugin", (
                        f"{req_id} reads block {block} holding state at "
                        f"{np.flatnonzero(holds(block))[:4].tolist()}"
                    )
                    return
        for block in gdn:
            holds(block)[:] = True
    else:
        raise AssertionError("scenario did not finish")
    assert variant == "plugin", "no read found state: the scenario needs churn"
    assert all(h.finished(f"r{i}") for i in range(16))
    assert needed and skipped_clear and skipped_written, (
        needed,
        skipped_clear,
        skipped_written,
    )


@pytest.mark.parametrize(
    ("retention", "b_hit"),
    [
        # The default keeps the prompt's replay boundary (1024 here: A's
        # prefill stops there) but no decode-time boundary.
        (0, 1024),
        # Dense retention keeps the 2048 state that A's boundary copy wrote.
        (None, 2048),
    ],
)
def test_boundary_state_written_after_acceptance_serves_a_prefix_hit(retention, b_hit):
    """The boundary copy after acceptance is what a later prefix hit reads.

    Request A (1900 tokens) decodes with drafts past the 2048 boundary, so the
    state after 2048 tokens exists only through A's decode-time copies.
    Request B repeats A's first 2100 tokens (prompt and output). B hits at
    2048 only when retention keeps decode-time states, and then reads the
    state A's boundary copy put there.
    """
    h = _Harness("align", num_spec=K, retention=retention)
    h.add("a", _prompt(1, 1900), max_tokens=250)
    h.run(max_steps=400, accept=_accept_pattern, until=lambda: h.finished("a"))
    h.add("b", h.tokens["a"][:2100], max_tokens=2)
    h.run(accept=_accept_pattern)
    hits = {r: n for s in h.steps for r, n in s.new_hits.items()}
    assert hits["b"] == b_hit
    if b_hit == 2048:
        # B's hit block in column 1 of each GDN group is where one of A's
        # boundary copies wrote the state after 2048 tokens.
        b_blocks = next(s.new_block_ids["b"] for s in h.steps if "b" in s.new_block_ids)
        boundary_copies = {(c.group, c.dst): c for s in h.steps for c in s.post_copies}
        for g in h.groups:
            copy = boundary_copies[(g, b_blocks[g][1])]
            assert copy.temporal_src != copy.dst


def _last_writers(h: _Harness, prompt_lens: dict[str, int]) -> list[dict]:
    """Replay which kind of step last wrote each GDN block, through the
    plugin's copies; return, per step, that map before the step ran.

    A row writes its state window from column 0: one state in a prefill
    chunk, one per query token in a decode step. ``begin_step`` copies run
    before the writes and ``after_acceptance`` copies after them.
    """
    writers: dict[int, str] = {}
    before = []
    for step in h.steps:
        before.append(dict(writers))
        for src, dst in step.groups_pre or ():
            writers[dst] = writers.get(src)
        for row in step.rows:
            decode = row.num_computed >= prompt_lens[row.req_id]
            written = row.num_scheduled if decode else 1
            for window in step.groups_windows[row.req_id].values():
                for block in window[:written]:
                    writers[block] = "decode" if decode else "prefill"
        for src, dst in step.groups_post or ():
            writers[dst] = writers.get(src)
    return before


@pytest.mark.parametrize("retention", [0, None])
def test_multi_turn_follow_ups_hit_decode_written_states_only_with_retention(
    retention,
):
    """Turn 1 decodes across a block boundary; four follow-ups, one after the
    other, extend its prompt and output with new text.

    At retention 0 vLLM keeps a GDN state only at a request's replay boundary
    (``KVCacheCoordinator.get_replay_boundaries``: the prompt length minus 1)
    and at its junction (``KVCacheManager.get_computed_blocks``: the attention
    hit of a request whose GDN hit is shorter). Both end a prefill chunk, so a
    follow-up never reads a state that decoding wrote: the first follow-up
    recomputes past turn 1's replay boundary and caches the junction, and the
    other three hit that state, which the first follow-up's prefill wrote.
    Turn 1's prompts put a block boundary inside its first verify or just
    after it (1023 tokens: the state after the first query token is the
    boundary state). With dense retention the follow-ups hit the boundary
    states turn 1's decode left, which the plugin's boundary copies keep.
    Every read is exact either way (the harness ledgers).
    """
    h = _Harness("align", num_spec=K, retention=retention)
    prompt_lens: dict[str, int] = {}
    turn_1 = {"t1": 1023, "t2": 1022, "t3": 1021, "t4": 2047}
    for name, length in turn_1.items():
        prompt_lens[name] = length
        h.add(name, _prompt(length, length), max_tokens=60)
    h.run(max_steps=400, accept=_accept_pattern)
    for i in range(4):
        for name in turn_1:
            follow_up = f"{name}-f{i}"
            prompt = h.tokens[name] + _prompt(100 * i + len(name), 300)
            prompt_lens[follow_up] = len(prompt)
            h.add(follow_up, prompt, max_tokens=4)
        h.run(max_steps=400, accept=_accept_pattern)

    writers = _last_writers(h, prompt_lens)
    hit_writers = {}
    for step, before in zip(h.steps, writers):
        for req_id, hit in step.new_hits.items():
            if hit:
                hit_writers[req_id] = (
                    hit,
                    {
                        before.get(step.new_block_ids[req_id][g][hit // BLOCK_SIZE - 1])
                        for g in h.groups
                    },
                )
    follow_ups = {r: v for r, v in hit_writers.items() if "-f" in r}
    if retention == 0:
        assert all(kinds == {"prefill"} for _, kinds in follow_ups.values()), follow_ups
        for name in turn_1:
            # The first follow-up caches the junction; the other three hit it.
            junction = len(h.tokens[name]) // BLOCK_SIZE * BLOCK_SIZE
            later = [follow_ups.get(f"{name}-f{i}", (0,))[0] for i in range(1, 4)]
            assert later == [junction] * 3, follow_ups
    else:
        assert any("decode" in kinds for _, kinds in follow_ups.values()), follow_ups


def _continue_text(h: _Harness, text: list[int]) -> None:
    """Make a request whose tokens are a prefix of ``text`` continue it: its
    drafts and its sampled tokens are the next tokens of ``text``, as greedy
    decoding of a passage the model reproduces."""
    scheduler = h.scheduler
    update_from_output = scheduler.update_from_output
    update_draft_token_ids = scheduler.update_draft_token_ids

    def next_tokens(req_id: str, tokens: list[int]) -> list[int]:
        request = scheduler.requests[req_id]
        if list(request.all_token_ids) != text[: request.num_tokens]:
            return list(tokens)
        return text[request.num_tokens : request.num_tokens + len(tokens)]

    def update(out, model_runner_output):
        sampled = [
            next_tokens(r, t) if t else t
            for r, t in zip(
                model_runner_output.req_ids, model_runner_output.sampled_token_ids
            )
        ]
        return update_from_output(
            out, dataclasses.replace(model_runner_output, sampled_token_ids=sampled)
        )

    def drafts(draft_token_ids):
        return update_draft_token_ids(
            DraftTokenIds(
                draft_token_ids.req_ids,
                [
                    next_tokens(r, d)
                    for r, d in zip(
                        draft_token_ids.req_ids, draft_token_ids.draft_token_ids
                    )
                ],
            )
        )

    scheduler.update_from_output = update
    scheduler.update_draft_token_ids = drafts


@pytest.mark.parametrize("same_text", [True, False])
def test_follow_ups_of_one_text_hit_the_first_follow_ups_junction(same_text):
    """A device check at retention 0 and block 2048 saw follow-ups hit the
    state at 2048 for turn-1 prompts of 2043, 2045 and 2047 tokens of one
    article, but not for 2040, all on one server. Turn 1's decode crosses
    2048; each follow-up is turn 1's first 2088 tokens plus a question.

    Here at block 1024: when the model continues the article word for word,
    every turn 1 decodes the same tokens, so every follow-up starts with the
    same 1024 tokens. The first follow-up (prompt 1016) has no GDN hit, so it
    recomputes and caches the state at its junction (1024) from its prefill;
    the later follow-ups hit that state. No hit reads a state turn 1's decode
    wrote, and every read is exact although the plugin runs no boundary copy
    at retention 0. With a different text per turn, no follow-up hits.
    """
    h = _Harness("align", num_spec=K, retention=0)
    text = _prompt(7, 4000)
    _continue_text(h, text)
    prompt_lens: dict[str, int] = {}
    hits = {}
    for i, length in enumerate((1016, 1019, 1021, 1023)):
        turn = text if same_text else _prompt(100 + i, 4000)
        first, follow_up = f"a{length}", f"b{length}"
        prompt_lens[first] = length
        h.add(first, turn[:length], max_tokens=1088 - length)
        h.run(400, _accept_pattern, until=lambda r=first: h.finished(r))
        prompt = h.tokens[first][:1064] + _prompt(99, 7)
        prompt_lens[follow_up] = len(prompt)
        h.add(follow_up, prompt, max_tokens=4)
        h.run(400, _accept_pattern, until=lambda r=follow_up: h.finished(r))
        hits[length] = next(
            (s.new_hits[follow_up] for s in h.steps if follow_up in s.new_hits), None
        )
    if not same_text:
        assert hits == {1016: 0, 1019: 0, 1021: 0, 1023: 0}, hits
        return
    assert hits == {1016: 0, 1019: 1024, 1021: 1024, 1023: 1024}, hits
    for step, before in zip(h.steps, _last_writers(h, prompt_lens)):
        for req_id, hit in step.new_hits.items():
            if hit:
                writers = {
                    before.get(step.new_block_ids[req_id][g][hit // BLOCK_SIZE - 1])
                    for g in h.groups
                }
                assert writers == {"prefill"}, (req_id, hit, writers)


def _checkpoint_scenario(
    retention, checkpoint: bool, writes_checkpoints=True, block_size=BLOCK_SIZE
):
    """Prompts below, at and past block boundaries, each served alone, then
    follow-ups that repeat each prompt and add text, so they hit its cached
    GDN state. 3100 tokens follow a 1024-token request at once, so their
    first chunk is the rest of a step's budget and their second chunk runs
    past a block boundary on a running request."""
    h = _Harness(
        "align",
        num_spec=K,
        retention=retention,
        max_num_batched_tokens=4096,
        checkpoint=checkpoint,
        block_size=block_size,
    )
    h.writes_checkpoints = writes_checkpoints
    prompts = {f"a{n}": _prompt(n, n) for n in (900, 1500, 2048, 2049, 3000)}
    for req_id, prompt in prompts.items():
        h.add(req_id, prompt, max_tokens=4)
        h.run(accept=_accept_pattern)
    prompts["c1024"], prompts["a3100"] = _prompt(7, 3072), _prompt(3100, 3100)
    h.add("c1024", prompts["c1024"], max_tokens=4)
    h.add("a3100", prompts["a3100"], max_tokens=4)
    h.run(accept=_accept_pattern)
    for req_id, prompt in list(prompts.items()):
        prompts["b" + req_id] = prompt + _prompt(len(prompt) + 1, 300)
        h.add("b" + req_id, prompts["b" + req_id], max_tokens=4)
        h.run(accept=_accept_pattern)
    chunks: dict[str, int] = {}
    for step in h.steps:
        for row in step.rows:
            if row.num_computed < len(prompts[row.req_id]):
                chunks[row.req_id] = chunks.get(row.req_id, 0) + 1
    hits = {r: n for step in h.steps for r, n in step.new_hits.items()}
    pool = h.scheduler.kv_cache_manager.block_pool
    cached = {
        (get_group_id(block.block_hash), block.block_hash_num_tokens)
        for block in pool.blocks
        if block.block_hash is not None and get_group_id(block.block_hash) in h.groups
    }
    return h, chunks, hits, cached, prompts


@pytest.mark.parametrize("block_size", [896, BLOCK_SIZE])
@pytest.mark.parametrize("retention", [0, None])
def test_prefill_checkpoints_keep_the_split_prefix_cache(retention, block_size):
    """``MambaSpec.num_prefill_checkpoint_blocks``: vLLM no longer ends a
    prefill chunk at the prompt's last cacheable boundary; it reserves a block
    for the state there, and the TT model writes it from
    ``MambaCacheGroups.checkpoint_inputs``.

    Against the same requests without checkpoint blocks: fewer prefill
    chunks, a hit at least as long for every follow-up, every cached GDN
    state of the split version cached too (group and tokens), and every read
    exact (the harness ledgers), with upstream's kernel and the TT model each
    writing the states. At retention 0 vLLM keeps the prompt's own
    checkpoint (``is_prompt_checkpoint`` in
    ``MambaManager._cache_partial_tail_block``), so follow-ups hit it."""
    _, split_chunks, split_hits, split_cached, _ = _checkpoint_scenario(
        retention, False, block_size=block_size
    )
    h, chunks, hits, cached, prompts = _checkpoint_scenario(
        retention, True, block_size=block_size
    )
    assert any(step.checkpoints for step in h.steps)
    # A first prompt that fits one chunk takes one. A follow-up whose GDN hit
    # is shorter than its attention hit still stops a chunk at the junction
    # (vLLM's ``junction_stop``), as without checkpoint blocks.
    firsts = [r for r in chunks if not r.startswith("b") and r != "a3100"]
    assert all(chunks[r] == 1 for r in firsts), chunks
    assert all(chunks[r] <= split_chunks[r] for r in chunks), (chunks, split_chunks)
    assert sum(chunks.values()) < sum(split_chunks.values())
    # At least the split version's hits and cached states. A prompt of whole
    # blocks runs as one chunk either way, and only the checkpoint leaves the
    # state at its replay boundary; with a checkpoint on a running request
    # vLLM also keeps the previous chunk's draft blocks, which dense
    # retention caches.
    assert all(hits[r] >= split_hits[r] for r in hits), (hits, split_hits)
    whole_blocks = {"b" + r for r, p in prompts.items() if len(p) % block_size == 0}
    assert {r for r in hits if hits[r] > split_hits[r]} <= whole_blocks
    assert split_cached <= cached


def test_a_model_that_ignores_the_checkpoint_inputs_serves_stale_hits():
    """Negative control: a follow-up that hits a checkpoint block reads the
    state its prefill was to write; without the write the ledger finds the
    block empty."""
    with pytest.raises(AssertionError, match="tt"):
        _checkpoint_scenario(0, True, writes_checkpoints=False)
    with pytest.raises(AssertionError, match="tt"):
        _checkpoint_scenario(0, True, writes_checkpoints=False, block_size=896)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_dp_engines_keep_their_own_gdn_state(checkpoint):
    """Standard multi-process DP, as the Galaxy runs ``data_parallel_size`` 4:
    each engine has its own ``TTScheduler``, ``KVCacheManager`` and
    ``MambaCacheGroups``.

    Four engines here, in one process, step in lockstep with different
    prompts, and every read stays exact (each engine's ledgers). Then one
    follow-up of engine 2's prompt runs on every engine, and only engine 2
    holds the GDN state it hits, which is why the DP router sends such a
    follow-up to the engine that was sent its prefix
    (``prefix_routing.route_dp_request``)."""
    engines = [
        _Harness(
            "align",
            num_spec=K,
            retention=0,
            max_num_batched_tokens=4096,
            checkpoint=checkpoint,
        )
        for _ in range(4)
    ]
    prompts = [_prompt(500 + i, 1500 + 700 * i) for i in range(4)]
    for i, (h, prompt) in enumerate(zip(engines, prompts)):
        h.add(f"e{i}", prompt, max_tokens=8)
    for _ in range(400):
        if not any([h.step(_accept_pattern) for h in engines]):
            break
    assert all(h.finished(f"e{i}") for i, h in enumerate(engines))

    follow_up = prompts[2] + _prompt(9, 300)
    for h in engines:
        h.add("f", follow_up, max_tokens=2)
        h.run(accept=_accept_pattern)
    hits = [
        next(s.new_hits["f"] for s in h.steps if "f" in s.new_hits) for h in engines
    ]
    assert hits[2] > 0 and hits[:2] + hits[3:] == [0, 0, 0]


# The Galaxy deployment's prefill knobs (tt-inference-server
# atupe/qwen38-glx-mtp) that the plugin accepts: chunked prefill with a
# 16384-token chunk limit, and one decode step after each prefill step.
DEPLOYMENT_TT_CONFIG = {
    "decode_interleave_prefill_steps": 1,
}


# 896: the block of a 7-bank Blackhole Galaxy (UBB_BLACKHOLE_CF), which does
# not divide the 16384-token chunk limit.
@pytest.mark.parametrize("block_size", [896, 1024, 2048])
def test_deployment_prefill_knobs_keep_chunks_on_the_gdn_block_grid(block_size):
    """Mode align needs every prefill chunk that does not finish its prompt to
    end on a GDN block boundary: block-table column ``p`` holds the state after
    exactly ``(p + 1) * block_size`` tokens, and prefix caching hashes it as
    such. ``TTScheduler`` applies its 128-token grid after vLLM's
    block-aligned split, so the grid must keep the chunk on the block grid
    too.

    The real ``TTScheduler`` and ``KVCacheManager`` serve, with the
    deployment's knobs and K=3: a 140000-token prompt, a 20000-token prompt, a
    short one, and a follow-up that shares the long prompt's first 70000
    tokens (its GDN hit is shorter than its attention hit, so vLLM also stops a
    chunk at the junction). Every partial chunk ends on the block grid, and a
    prefill step follows a prefill step only when no request is decoding.
    vLLM 0.31 drops ``long_prefill_token_threshold`` for a step with one
    eligible request, and ``TTScheduler`` hides the decoding requests from a
    prefill step, so a lone prompt's chunk can be longer than 16384 tokens.
    """
    max_model_len = 145_000
    planner = _planner_config("align", 0)
    planner.model_config.max_model_len = max_model_len
    planner.model_config.original_max_model_len = max_model_len
    kv_cache_config = _kv_cache_config(
        planner, _qwen38_specs("align", K, block_size=block_size), 640
    )
    scheduler = _scheduler(
        kv_cache_config,
        "align",
        K,
        0,
        525312,
        max_model_len=max_model_len,
        block_size=block_size,
        long_prefill_token_threshold=16384,
        tt_config=DEPLOYMENT_TT_CONFIG,
    )
    init_none_hash(sha256)
    long_prompt = _prompt(1, 140_037)
    prompts = {
        "long": long_prompt,
        "mid": _prompt(2, 20_005),
        "short": _prompt(3, 3000),
    }

    def add(req_id: str, prompt: list[int]) -> None:
        params = SamplingParams(max_tokens=3, ignore_eos=True)
        params.update_from_generation_config({}, eos_token_id=2)
        scheduler.add_request(
            Request(
                request_id=req_id,
                prompt_token_ids=prompt,
                sampling_params=params,
                pooling_params=None,
                block_hasher=get_request_block_hasher(
                    scheduler.hash_block_size, sha256
                ),
            )
        )

    for req_id, prompt in prompts.items():
        add(req_id, prompt)
    chunks: list[tuple[str, int, int]] = []
    previous_was_prefill = False
    follow_up_added = False
    for step in range(400):
        if not follow_up_added and "long" not in scheduler.requests:
            prompts["hit"] = long_prompt[:70_000] + _prompt(4, 5000)
            add("hit", prompts["hit"])
            follow_up_added = True
        decoding = [
            r.request_id
            for r in scheduler.running
            if r.num_computed_tokens >= r.num_prompt_tokens
        ]
        starts = {
            r.request_id: r.num_computed_tokens
            for r in (*scheduler.running, *scheduler.waiting)
        }
        out = scheduler.schedule()
        if not out.num_scheduled_tokens:
            if not scheduler.requests:
                break
            continue
        prefill = {
            r: (starts.get(r, 0), n)
            for r, n in out.num_scheduled_tokens.items()
            if starts.get(r, 0) < len(prompts[r])
        }
        for new in out.scheduled_new_reqs:
            if new.req_id in prefill:
                prefill[new.req_id] = (
                    new.num_computed_tokens,
                    out.num_scheduled_tokens[new.req_id],
                )
        if prefill:
            assert not (previous_was_prefill and decoding), (step, decoding)
        previous_was_prefill = bool(prefill)
        for req_id, (start, n) in prefill.items():
            chunks.append((req_id, start, n))
            if start + n == len(prompts[req_id]):
                continue
            assert (start + n) % block_size == 0, (req_id, start, n)
        sampled = []
        for req_id in out.num_scheduled_tokens:
            request = scheduler.requests[req_id]
            if request.num_computed_tokens < request.num_prompt_tokens:
                sampled.append([])
                continue
            drafts = out.scheduled_spec_decode_tokens.get(req_id, [])
            sampled.append(list(drafts[: len(drafts) // 2]) + [7])
        req_ids = list(out.num_scheduled_tokens)
        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: i for i, r in enumerate(req_ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        running = [r for r, s in zip(req_ids, sampled) if s and r in scheduler.requests]
        if running:
            scheduler.update_draft_token_ids(
                DraftTokenIds(running, [[11, 12, 13] for _ in running])
            )
    else:
        raise AssertionError("scenario did not finish")
    assert follow_up_added and not scheduler.requests
    long_chunks = [(start, n) for r, start, n in chunks if r == "long"]
    assert len(long_chunks) > 1, long_chunks
    # At retention 0 no GDN state of the long prompt lies below 70000, so the
    # follow-up starts at 0 and stops a chunk at its junction, the attention hit.
    junction = 70_000 // block_size * block_size
    hit_chunks = [(start, n) for r, start, n in chunks if r == "hit"]
    assert hit_chunks[0][0] == 0, hit_chunks
    assert any(start + n == junction for start, n in hit_chunks), hit_chunks


def test_mode_none_needs_a_gdn_block_size_of_max_model_len():
    """In mode none ``MambaSpec.block_size`` must cover the whole sequence.

    Upstream resolves ``mamba_block_size`` to max_model_len in mode none; the
    TT model sets ``MambaSpec.block_size`` itself. With 1024 instead, vLLM's
    ``MambaManager`` allocates one block per 1024 tokens in every GDN group,
    like attention, while the TT pool sizing counts one state block per
    request and group.
    """
    cfg = _planner_config("none")
    kv_cache_config = _kv_cache_config(
        cfg, _qwen38_specs("none", num_spec=0, gdn_block_size=BLOCK_SIZE), NUM_BLOCKS
    )
    scheduler = _scheduler(kv_cache_config, "none", 0, 0, 4096)
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=2, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    scheduler.add_request(
        Request(
            request_id="a",
            prompt_token_ids=_prompt(1, 3500),
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        )
    )
    scheduler.schedule()
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers[:3]:
        assert len(manager.req_to_blocks["a"]) == 4


# --- The TT runner's moves with K=3 drafts ---------------------------------


@pytest.mark.parametrize(("mode", "row_width"), [("none", 1 + K), ("align", 8 + K)])
def test_tt_runner_accepts_speculative_state_blocks(mode, row_width):
    kv_cache_config = _kv_cache_config(
        _planner_config(mode), _qwen38_specs(mode, num_spec=K), NUM_BLOCKS
    )
    groups = mamba_cache_groups(
        kv_cache_config.kv_cache_groups,
        SimpleNamespace(mamba_cache_mode=mode),
        MAX_MODEL_LEN,
    )
    assert groups.num_speculative_blocks == K
    # Upstream's ``MambaSpec.max_num_blocks_per_req``: 1 + K blocks in mode
    # none, cdiv(8192, 1024) + K block-table columns in mode align.
    assert groups.max_num_blocks(MAX_MODEL_LEN) == row_width


def test_tt_runner_state_window_has_1_plus_k_columns():
    spec = _qwen38_specs("align", num_spec=K)["model.layers.0.linear_attn"]
    groups = MambaCacheGroups(
        group_indices=(0,),
        layer_names=(("model.layers.0.linear_attn",),),
        mamba_cache_mode="align",
        block_size=BLOCK_SIZE,
        attention_group_index=3,
        num_speculative_blocks=K,
    )
    block_table = torch.arange(1, 4 * 11 + 1, dtype=torch.int32).reshape(4, 11)
    seq_lens = torch.tensor([1, 1024, 1028, 8190])
    expected = mamba_get_block_table_tensor(block_table, seq_lens, spec, "align")
    assert expected.shape == (4, 1 + K)
    assert torch.equal(groups.state_windows(block_table, seq_lens), expected)
    assert torch.equal(groups.state_block_ids(block_table, seq_lens), expected[:, :1])


@pytest.mark.parametrize("mode", ["none", "align"])
def test_tt_runner_hands_the_model_the_upstream_window_and_start_state(mode):
    """Each row's window is upstream's, and window column ``count - 1`` is the
    block upstream's kernel reads the start state from. The harness also
    checks every read of the TT state layout against the row's tokens."""
    h = _Harness(mode, num_spec=K, retention=None)
    h.add("a", _prompt(1, 1010), max_tokens=40)
    h.add("bb", _prompt(2, 700), max_tokens=40)
    h.run(accept=_accept_pattern)
    assert any(c > 1 for s in h.steps for c in s.groups_counts.values())
    for step in h.steps:
        for row in step.rows:
            assert step.groups_windows[row.req_id] == row.windows
            count = step.groups_counts[row.req_id]
            assert {
                g: window[count - 1] for g, window in row.windows.items()
            } == step.initial_blocks[row.req_id]


def test_tt_runner_preprocess_copies_shift_by_accepted_drafts():
    """``begin_step`` copies from upstream's temporal source, column
    ``prev + num_accepted - 1``, to the new column, as ``preprocess_mamba``."""
    h = _Harness("align", num_spec=K, retention=None)
    h.add("a", _prompt(1, 1010), max_tokens=60)
    h.add("bb", _prompt(2, 2030), max_tokens=60)
    h.run(accept=_accept_pattern)
    assert any(c.conv_row_offset for s in h.steps for c in s.pre_copies)
    for step in h.steps:
        assert step.groups_pre == [(c.temporal_src, c.dst) for c in step.pre_copies]


def test_tt_runner_copies_the_boundary_state_after_acceptance():
    """``after_acceptance`` makes upstream ``postprocess_mamba_align_gpu``'s
    copies, and a later request's prefix hit reads the state they moved."""
    h = _Harness("align", num_spec=K, retention=None)
    h.add("a", _prompt(1, 1900), max_tokens=250)
    h.run(max_steps=400, accept=_accept_pattern, until=lambda: h.finished("a"))
    h.add("b", h.tokens["a"][:2100], max_tokens=2)
    h.run(accept=_accept_pattern)
    assert any(s.post_copies for s in h.steps)
    for step in h.steps:
        assert step.groups_post == [(c.temporal_src, c.dst) for c in step.post_copies]
    hits = {r: n for s in h.steps for r, n in s.new_hits.items()}
    assert hits["b"] == 2048


@pytest.mark.parametrize("mode", ["none", "align"])
def test_tt_runner_moves_with_drafts_match_upstream(mode):
    """``TTModelRunner`` itself, speculating with K=3, makes upstream's moves.

    ``build_model_input`` hands the model upstream's window and the count
    whose column holds upstream's start state, and copies as
    ``preprocess_mamba``; ``commit_spec_acceptance`` copies as
    ``postprocess_mamba_align_gpu``.
    """
    h = _Harness(mode, num_spec=K, retention=None, plugin=True)
    h.add("a", _prompt(1, 1010), max_tokens=60)
    h.add("bb", _prompt(2, 2030), max_tokens=60)
    h.run(accept=_accept_pattern)
    assert h.runner._mamba_groups.num_speculative_blocks == K
    verifies = [s for s in h.steps if s.runner_post is not None]
    assert verifies
    for step in h.steps:
        assert step.runner_copies == [(c.temporal_src, c.dst) for c in step.pre_copies]
        assert (step.runner_post or []) == [
            (c.temporal_src, c.dst) for c in step.post_copies
        ]
        for row in step.rows:
            assert step.runner_windows[row.req_id] == row.windows
            count = step.runner_counts[row.req_id]
            assert {
                g: window[count - 1] for g, window in row.windows.items()
            } == step.initial_blocks[row.req_id]
    if mode == "align":
        assert any(s.runner_post for s in verifies)


def test_state_windows_and_counts_reach_the_model():
    """Prefill and decode get ``mamba_state`` with ``1 + K`` window columns per
    group and the start-state counts."""
    h = _Harness("align", num_spec=K, retention=None, plugin=True)
    h.add("a", _prompt(1, 1010), max_tokens=8)
    h.run(accept=_accept_pattern)
    prefill = h.steps[0].runner_input
    decode = next(s.runner_input for s in h.steps if s.runner_input.spec_mode)
    groups = prefill.mamba_state.groups
    assert len(groups) == 3
    assert {g.windows.shape for g in groups} == {(MAX_BATCH, 1 + K)}
    assert torch.equal(
        prefill.mamba_state.accepted_counts, torch.ones(MAX_BATCH, dtype=torch.int32)
    )

    h.model.calls.clear()
    h.runner.submit_prefill(prefill, [1])
    with contextlib.suppress(Exception):
        # The fake model returns no VerifyOutput, which the runner refuses
        # after the call.
        h.runner.async_decode.submit_decode(decode, read_from_device=True)
    forwards = {name: kwargs for name, kwargs in h.model.calls}
    for name, model_input in (("prefill_forward", prefill), ("decode_forward", decode)):
        assert forwards[name]["mamba_state"] is model_input.mamba_state


def _batch_rows(rows: list[list[int]], *shape: int) -> torch.Tensor:
    """``rows`` as the first rows of an int32 ``[MAX_BATCH, *shape]`` tensor;
    the padding rows hold block 0."""
    out = torch.zeros((MAX_BATCH, *shape), dtype=torch.int32)
    for i, row in enumerate(rows):
        out[i] = torch.tensor(row, dtype=torch.int32).reshape(shape)
    return out


def _assert_packed(state, kv_cache_config, expected: dict[int, dict]) -> None:
    """``state`` holds one ``MambaGroupState`` per GDN group, in KV cache group
    order, and each GDN layer finds in the group that names it exactly the
    tensors of its own KV cache group: ``expected[g]`` maps a
    ``MambaGroupState`` field to its value for group ``g``."""
    groups = kv_cache_config.kv_cache_groups
    gdn = [i for i, g in enumerate(groups) if isinstance(g.kv_cache_spec, MambaSpec)]
    assert [g.layer_names for g in state.groups] == [
        tuple(groups[i].layer_names) for i in gdn
    ]
    owner = {name: i for i in gdn for name in groups[i].layer_names}
    by_layer = {name: g for g in state.groups for name in g.layer_names}
    assert by_layer.keys() == owner.keys()
    assert sum(len(g.layer_names) for g in state.groups) == len(owner)
    for name, group_state in by_layer.items():
        want = expected[owner[name]]
        for field_name in ("windows", "checkpoint_blocks"):
            got, value = getattr(group_state, field_name), want.get(field_name)
            if value is None:
                assert got is None, (name, field_name)
                continue
            assert got.dtype == torch.int32 and got.device.type == "cpu"
            assert torch.equal(got, value), (name, field_name, got, value)
        if "chain" not in want:
            assert group_state.chain is None, name
            continue
        for field_name, value in want["chain"].items():
            got = getattr(group_state.chain, field_name)
            assert got.dtype == torch.int32 and got.device.type == "cpu"
            assert torch.equal(got, value), (name, field_name, got, value)


def test_mamba_state_gives_each_gdn_layer_its_own_groups_tensors():
    """``mamba_state`` packs each GDN group's state inputs once, with the
    group's ``layer_names``: three GDN groups, K=3, mode align, a prefill with
    a checkpoint block, a verify, and a chained verify built while that verify
    is pending.

    Every GDN group's row holds different blocks, so a tensor packed for the
    wrong group, or a group with another group's layer names, fails. The
    expected values come from the block ids alone: the windows from upstream
    ``mamba_get_block_table_tensor`` (``state_window``), the checkpoint from
    the scheduler's ``(position, column)`` entry, and the chained verify's
    choices from the block-table columns worked out in the comments.
    """
    cfg = _planner_config("align", retention=None)
    kv_cache_config = _kv_cache_config(
        cfg, _qwen38_specs("align", K, checkpoint=True), NUM_BLOCKS
    )
    model = _RecordingModel()
    runner = _tt_runner(cfg, kv_cache_config, model, num_spec=K)
    groups = kv_cache_config.kv_cache_groups
    gdn = [i for i, g in enumerate(groups) if isinstance(g.kv_cache_spec, MambaSpec)]
    assert len(gdn) == 3
    spec = groups[gdn[0]].kv_cache_spec
    width = runner._mamba_groups.max_num_blocks(MAX_MODEL_LEN)
    # A full block-table row per GDN group and request, distinct block ids.
    blocks = iter(range(1, NUM_BLOCKS))
    rows = {
        r: {g: [next(blocks) for _ in range(width)] for g in gdn} for r in ("a", "b")
    }
    attention = {"a": [next(blocks) for _ in range(3)], "b": [next(blocks)]}

    def block_ids(r):
        return tuple(
            rows[r][g] if g in gdn else attention[r] for g in range(len(groups))
        )

    def windows(seq_lens):
        return {
            g: _batch_rows(
                [
                    state_window(rows[r][g], seq_len, spec, width)
                    for r, seq_len in seq_lens.items()
                ],
                1 + K,
            )
            for g in gdn
        }

    # The prefill: a (2045 tokens) leaves the state after 1024 tokens in
    # block-table column 0; b (700 tokens) leaves none.
    step = _step(
        {"a": 2045, "b": 700},
        new=[
            _new_request("a", 2045, block_ids("a")),
            _new_request("b", 700, block_ids("b")),
        ],
    )
    step._tt_mamba_checkpoints = {"a": ((1024, 0),)}
    prefill = runner.build_model_input(step, None)
    assert prefill.row_req_ids[:2] == ["a", "b"]
    state = prefill.mamba_state
    assert torch.equal(state.accepted_counts, torch.ones(MAX_BATCH, dtype=torch.int32))
    assert torch.equal(state.checkpoint_positions, _batch_rows([[1024], [0]], 1))
    expected_windows = windows({"a": 2045, "b": 700})
    _assert_packed(
        state,
        kv_cache_config,
        {
            g: {
                "windows": expected_windows[g],
                "checkpoint_blocks": _batch_rows([[rows["a"][g][0]], [0]], 1),
            }
            for g in gdn
        },
    )
    runner.submit_prefill(prefill, [2])
    assert model.calls[-1][0] == "prefill_forward"
    assert model.calls[-1][1]["mamba_state"] is state

    # The verify: one sampled token each, then 1 + K tokens from 2045 and 700.
    runner._apply_sampled_tokens_to_state(
        torch.tensor([[11], [12]], dtype=torch.int32), req_ids=["a", "b"]
    )
    step = _step({"a": 1 + K, "b": 1 + K}, cached=[("a", 2045, None), ("b", 700, None)])
    step.scheduled_spec_decode_tokens = {"a": [21, 22, 23], "b": [31, 32, 33]}
    verify = runner.build_model_input(step, None)
    assert verify.spec_mode is not None
    assert verify.row_req_ids[:2] == ["a", "b"]
    state = verify.mamba_state
    assert torch.equal(state.accepted_counts, torch.ones(MAX_BATCH, dtype=torch.int32))
    assert state.checkpoint_positions is None
    expected_windows = windows({"a": 2045 + 1 + K, "b": 700 + 1 + K})
    _assert_packed(
        state, kv_cache_config, {g: {"windows": expected_windows[g]} for g in gdn}
    )

    # The chained verify, built while the verify is pending: entry c - 1 is
    # for a verify that accepted c tokens. a's verify runs in column 2 from
    # 2045; with c >= 3 it reaches 2048, so the state after 2048 tokens, in its
    # window column 2 (block-table column 4), goes to column 1. a's chained
    # step stays in column 2 and starts from window column c - 1. b stays in
    # column 0 and copies nothing.
    step = _step(
        {"a": 1 + K, "b": 1 + K},
        cached=[("a", 2045 + 1 + K, None), ("b", 700 + 1 + K, None)],
    )
    chained = runner._attach_mamba_state(
        step, dataclasses.replace(verify, chain_rows=torch.ones(2, dtype=torch.bool))
    )
    state = chained.mamba_state
    counts = torch.ones(MAX_BATCH, dtype=torch.int32)
    counts[:2] = -1
    assert torch.equal(state.accepted_counts, counts)
    assert state.checkpoint_positions is None
    counts_range = range(1, 2 + K)
    expected = {}
    for g in gdn:
        a, b = rows["a"][g], rows["b"][g]
        chain_windows = _batch_rows(
            [
                [state_window(a, 2045 + c + 1 + K, spec, width) for c in counts_range],
                [state_window(b, 700 + c + 1 + K, spec, width) for c in counts_range],
            ],
            1 + K,
            1 + K,
        )
        expected[g] = {
            "windows": chain_windows[:, -1],
            "chain": {
                "copies": _batch_rows(
                    [
                        [(0, 0) if c < 3 else (a[4], a[1]) for c in counts_range],
                        [(0, 0)] * (1 + K),
                    ],
                    1 + K,
                    2,
                ),
                "start_blocks": _batch_rows(
                    [
                        [a[1 + c] for c in counts_range],
                        [b[c - 1] for c in counts_range],
                    ],
                    1 + K,
                ),
                "windows": chain_windows,
            },
        }
    _assert_packed(state, kv_cache_config, expected)


def test_speculative_rows_are_sized_like_upstream():
    """GDN block-table rows hold ``MambaSpec.max_num_blocks_per_req`` blocks,
    so the scheduler's ``1 + K`` (none) or position + K (align) blocks fit."""
    widths = {}
    for mode in ("none", "align"):
        cfg = _planner_config(mode)
        kv_cache_config = _kv_cache_config(cfg, _qwen38_specs(mode, K), NUM_BLOCKS)
        runner = _tt_runner(cfg, kv_cache_config, _RecordingModel(), num_spec=K)
        widths[mode] = (
            [
                t.max_num_blocks_per_req
                for t in runner.input_batch.block_table.block_tables
            ],
            runner.max_num_blocks_per_req,
        )
    assert widths == {"none": ([4, 4, 4, 8], 8), "align": ([11, 11, 11, 8], 16)}


def test_mamba_draft_length_must_match_the_launch():
    cfg = _planner_config("align")
    kv_cache_config = _kv_cache_config(cfg, _qwen38_specs("align", K), NUM_BLOCKS)
    with pytest.raises(ValueError, match="differs from the launch's 0"):
        _tt_runner(cfg, kv_cache_config, _RecordingModel())


@pytest.mark.parametrize(
    ("mode", "gdn_block_size", "match"),
    [
        ("align", 2 * BLOCK_SIZE, "all groups need one block_size"),
        ("none", BLOCK_SIZE, "below max_model_len"),
    ],
)
def test_mamba_block_size_rules_are_checked(mode, gdn_block_size, match):
    """Mode align needs one block size for GDN and attention: the scheduler
    ends prefill chunks on multiples of the smallest group block size. Mode
    none needs a GDN block that covers max_model_len."""
    kv_cache_config = _kv_cache_config(
        _planner_config(mode),
        _qwen38_specs(mode, num_spec=0, gdn_block_size=gdn_block_size),
        NUM_BLOCKS,
    )
    with pytest.raises(ValueError, match=match):
        mamba_cache_groups(
            kv_cache_config.kv_cache_groups,
            SimpleNamespace(mamba_cache_mode=mode),
            MAX_MODEL_LEN,
        )
