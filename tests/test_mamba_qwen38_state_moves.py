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
    _RecordingModel,
    _vllm_config,
)
from vllm_tt_plugin import scheduler as scheduler_module
from vllm_tt_plugin.config import store_tt_spec_plan
from vllm_tt_plugin.mamba_state import MambaCacheGroups, mamba_cache_groups
from vllm_tt_plugin.model_input import TTModelInput
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.scheduler import TTScheduler
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
) -> dict:
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
) -> VllmConfig:
    """A ``VllmConfig`` resolved by the TT platform hook for ``model_class``.

    The cache mode a hybrid model gets from vLLM's config resolution
    (``MambaModelConfig``) is set before the hook runs: mode align comes with
    prefix caching.
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
        block_size=BLOCK_SIZE,
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
        )


def _scheduler(
    kv_cache_config,
    mode: str,
    num_spec: int,
    retention: int | None,
    max_num_batched_tokens: int,
    prefix_match_unit: int | None = None,
) -> TTScheduler:
    """A real ``TTScheduler`` over a real ``KVCacheManager``, started as
    ``EngineCore`` starts it: ``block_size`` is set from the groups and
    validated. The model declares chunked prefill and prefix caching.
    """
    config = _engine_config(mode, retention, max_num_batched_tokens, prefix_match_unit)
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
    ):
        self.mode = mode
        self.num_spec = num_spec
        self.cfg = _planner_config(mode, retention)
        self.kv_cache_config = _kv_cache_config(
            self.cfg, _qwen38_specs(mode, num_spec), NUM_BLOCKS
        )
        self.scheduler = _scheduler(
            self.kv_cache_config,
            mode,
            num_spec,
            retention,
            max_num_batched_tokens,
            prefix_match_unit,
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
        """``MambaCacheGroups.step_inputs`` over this step's rows, one "layer"
        per KV cache group."""
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
        inputs = self.plugin_groups.step_inputs(
            tables,
            req_ids,
            {r: self.req_states[r].num_computed_tokens for r in req_ids},
            out.num_scheduled_tokens,
            list(range(num_groups)),
            len(req_ids),
        )
        windows = (
            inputs.state_block_windows_per_layer or inputs.state_block_ids_per_layer
        )
        for i, r in enumerate(req_ids):
            record.groups_windows[r] = {g: windows[g][i].tolist() for g in self.groups}
            record.groups_counts[r] = (
                1
                if inputs.state_accepted_counts is None
                else int(inputs.state_accepted_counts[i])
            )
            assert [
                int(inputs.state_block_ids_per_layer[g][i, 0]) for g in self.groups
            ] == [record.groups_windows[r][g][0] for g in self.groups]

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
            layer_groups = self.runner._layer_to_group_idx
            windows = model_input.state_block_windows_per_layer
            for row, req_id in enumerate(model_input.row_req_ids or []):
                record.runner_state_ids[req_id] = {
                    layer_groups[layer]: int(ids[row, 0])
                    for layer, ids in enumerate(model_input.state_block_ids_per_layer)
                    if ids is not None
                }
                if windows is not None:
                    record.runner_windows[req_id] = {
                        layer_groups[layer]: ids[row].tolist()
                        for layer, ids in enumerate(windows)
                        if ids is not None
                    }
                    record.runner_counts[req_id] = int(
                        model_input.state_accepted_counts[row]
                    )

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
                        assert live <= 2 + self.num_spec, (req_id, g, live)
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
    boundary. In every step the TT runner's ``state_block_ids_per_layer``
    equals the upstream window column, and in mode align its
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
    assert not h.plugin_groups.retains_decode_boundaries
    h.add("a", _prompt(1, 1010), max_tokens=60)
    h.add("bb", _prompt(2, 2030), max_tokens=60)
    h.run(accept=_accept_pattern)
    assert h.finished("a") and h.finished("bb")
    assert any(s.post_copies for s in h.steps), "upstream copies at boundaries"
    assert not any(s.groups_post for s in h.steps)


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
    """Prefill and decode get ``state_block_windows_per_layer`` and
    ``state_accepted_counts`` next to ``state_block_ids_per_layer``."""
    h = _Harness("align", num_spec=K, retention=None, plugin=True)
    h.add("a", _prompt(1, 1010), max_tokens=8)
    h.run(accept=_accept_pattern)
    prefill = h.steps[0].runner_input
    decode = next(s.runner_input for s in h.steps if s.runner_input.spec_mode)
    windows = prefill.state_block_windows_per_layer
    assert len(windows) == NUM_LAYERS
    assert all((w is None) == _is_attention(i) for i, w in enumerate(windows))
    assert windows[0].shape == (MAX_BATCH, 1 + K)
    assert torch.equal(
        prefill.state_accepted_counts, torch.ones(MAX_BATCH, dtype=torch.int32)
    )

    h.model.calls.clear()
    h.runner.submit_prefill(prefill, [1])
    with contextlib.suppress(Exception):
        # The fake model returns no VerifyOutput, which the runner refuses
        # after the call.
        h.runner.async_decode.submit_decode(decode, read_from_device=True)
    forwards = {name: kwargs for name, kwargs in h.model.calls}
    for name, model_input in (("prefill_forward", prefill), ("decode_forward", decode)):
        kwargs = forwards[name]
        assert kwargs["state_block_windows_per_layer"] is (
            model_input.state_block_windows_per_layer
        )
        assert kwargs["state_accepted_counts"] is model_input.state_accepted_counts


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
