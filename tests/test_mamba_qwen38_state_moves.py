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
ids. With K=3 the TT runner has no implementation yet; the ``xfail`` tests
name each missing move.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
    SpeculativeConfig,
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
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.mamba_runner_reference import (
    StateCopy,
    StateLedger,
    UpstreamMambaRunner,
    label,
    state_window,
)
from tests.test_mamba_kv_cache import (
    _kv_cache_config,
    _RecordingModel,
    _vllm_config,
)
from vllm_tt_plugin.config import store_tt_spec_plan
from vllm_tt_plugin.mamba_state import MambaCacheGroups, mamba_cache_groups
from vllm_tt_plugin.model_runner import TTModelRunner
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


def _scheduler(
    kv_cache_config,
    mode: str,
    num_spec: int,
    retention: int | None,
    max_num_batched_tokens: int,
) -> TTScheduler:
    """A real ``TTScheduler`` over a real ``KVCacheManager``.

    The cache fields a hybrid model would get from vLLM's config resolution
    (``MambaModelConfig``) are set after construction, as the engine core has
    them when it builds the scheduler: ``block_size`` is the groups' block
    size, and mode align comes with prefix caching.
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
    cache_config.prefix_cache_retention_interval = retention
    cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    # The stub declares no chunked-prefill support, so the platform hook turned
    # chunked prefill off; mode align needs it (``MambaModelConfig`` asserts).
    config.scheduler_config.enable_chunked_prefill = True
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
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
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        hash_block_size=hash_block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _tt_runner(cfg, kv_cache_config, model) -> TTModelRunner:
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
    plugin_copies: list[tuple[int, int]] | None = None
    plugin_state_ids: dict[str, dict[int, int]] = field(default_factory=dict)
    # The upstream block a row's recurrent kernel reads its initial temporal
    # state from, per group.
    initial_blocks: dict[str, dict[int, int]] = field(default_factory=dict)


class _Harness:
    """Drives ``TTScheduler`` and the upstream reference through whole steps.

    Each step: the scheduler schedules; the upstream runner's view of every
    request (``req_states``: computed tokens and append-only block-table rows)
    is updated as the GPU runner's ``_update_states`` does; the reference runs
    its preprocess copies and selects each row's state window; the ledger
    checks the initial-state reads and records the writes; acceptance is
    simulated; the reference runs its postprocess copies; the scheduler gets
    the sampled tokens and the next drafts. With ``plugin=True`` the TT
    runner builds its model input from the same scheduler output; with
    ``plugin_groups=True`` only the TT runner's ``MambaCacheGroups`` runs.
    """

    def __init__(
        self,
        mode: str,
        num_spec: int,
        retention: int | None = 0,
        max_num_batched_tokens: int = 2048,
        plugin: bool = False,
        plugin_groups: bool = False,
    ):
        self.mode = mode
        self.num_spec = num_spec
        self.cfg = _planner_config(mode, retention)
        self.kv_cache_config = _kv_cache_config(
            self.cfg, _qwen38_specs(mode, num_spec), NUM_BLOCKS
        )
        self.scheduler = _scheduler(
            self.kv_cache_config, mode, num_spec, retention, max_num_batched_tokens
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
            _tt_runner(self.cfg, self.kv_cache_config, self.model) if plugin else None
        )
        # The TT runner's MambaCacheGroups built directly, skipping
        # ``mamba_cache_groups``'s rejection of speculative blocks, so its moves
        # can be compared with K > 0.
        self.plugin_groups = (
            MambaCacheGroups(
                group_indices=tuple(self.groups),
                mamba_cache_mode=mode,
                block_size=BLOCK_SIZE,
                attention_group_index=3,
            )
            if plugin_groups
            else None
        )
        self.steps: list[_Step] = []
        # Each request's tokens (prompt and output) after its last step.
        self.tokens: dict[str, list[int]] = {}
        self.draft_base = 70_000
        self.bonus_base = 50_000

    def add(self, req_id: str, prompt: list[int], max_tokens: int = 64) -> None:
        init_none_hash(sha256)
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, eos_token_id=2)
        self.scheduler.add_request(
            Request(
                request_id=req_id,
                prompt_token_ids=list(prompt),
                sampling_params=params,
                pooling_params=None,
                block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
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

        self.ledger.zero(out.new_block_ids_to_zero or ())
        self.ledger.copy(
            [
                StateCopy(g, c.src_block_id, c.src_block_id, 0, c.dst_block_id)
                for c in out.kv_cache_block_copies or ()
                for g in self.groups
            ]
        )
        req_ids = list(out.num_scheduled_tokens)
        plugin_copies = (
            self.plugin_groups.align_state_copies(out, self.req_states)
            if self.plugin_groups is not None
            else None
        )
        pre_copies = (
            self.reference.preprocess(out, req_ids, self.req_states)
            if self.mode == "align"
            else []
        )
        self.ledger.copy(pre_copies)

        record = _Step(
            dict(out.num_scheduled_tokens), new_hits, new_block_ids, [], pre_copies, []
        )
        if plugin_copies is not None:
            record.plugin_copies = plugin_copies
            seq_lens = torch.tensor(
                [
                    self.req_states[r].num_computed_tokens + out.num_scheduled_tokens[r]
                    for r in req_ids
                ]
            )
            for g in self.groups:
                table = torch.zeros((len(req_ids), self.width), dtype=torch.int32)
                for i, r in enumerate(req_ids):
                    row = self.req_states[r].block_ids[g]
                    table[i, : len(row)] = torch.tensor(row, dtype=torch.int32)
                ids = self.plugin_groups.state_block_ids(table, seq_lens)
                for i, r in enumerate(req_ids):
                    record.plugin_state_ids.setdefault(r, {})[g] = int(ids[i, 0])
        if self.runner is not None:
            self.model.calls.clear()
            model_input = self.runner.build_model_input(out, None)
            copies = [
                pair
                for name, kwargs in self.model.calls
                if name == "copy_kv_cache_blocks"
                for pair in kwargs["copies"]
            ]
            vllm_copies = [
                (c.src_block_id, c.dst_block_id)
                for c in out.kv_cache_block_copies or ()
            ]
            record.plugin_copies = copies[len(vllm_copies) :]
            for row, req_id in enumerate(model_input.row_req_ids or []):
                record.plugin_state_ids[req_id] = {
                    self.runner._layer_to_group_idx[layer]: int(ids[row, 0])
                    for layer, ids in enumerate(model_input.state_block_ids_per_layer)
                    if ids is not None
                }

        accepted: dict[str, int] = {}
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
                    expect = label(tokens[:nc])
                    got = self.ledger.initial_state(
                        g, window, num_accepted, speculating
                    )
                    assert got == (expect, expect), (req_id, g, nc, window)
                real_tokens = tokens[nc : nc + ns - len(drafts)]
                query = real_tokens + drafts
                if speculating:
                    labels = [
                        label(tokens[:nc] + query[: t + 1]) for t in range(len(query))
                    ]
                else:
                    labels = [label(tokens[:nc] + query)]
                self.ledger.write(g, window, labels, speculating)
            record.rows.append(_Row(req_id, nc, ns, drafts, windows))

            emits_token = nc + ns - len(drafts) >= request.num_tokens
            if emits_token:
                a = max(0, min(accept(req_id, len(self.steps), drafts), len(drafts)))
                accepted[req_id] = a + 1
                sampled.append(
                    drafts[:a] + [self.bonus_base + 10 * len(self.steps) + row_idx]
                )
                if speculating:
                    for g in self.groups:
                        self.ledger.reject(g, windows[g], a + 1)
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

    tt_groups = mamba_cache_groups(groups, SimpleNamespace(mamba_cache_mode=mode))
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
    h = _Harness(mode, num_spec=0, max_num_batched_tokens=1500)
    h.add("a", _prompt(1, 3500), max_tokens=1)
    h.run()
    assert [s.num_scheduled["a"] for s in h.steps] == chunks


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
        assert step.plugin_copies == [(c.temporal_src, c.dst) for c in step.pre_copies]
        assert all(c.conv_src == c.temporal_src for c in step.pre_copies)
        copies_seen += len(step.pre_copies)
        for row in step.rows:
            assert step.plugin_state_ids[row.req_id] == {
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


# --- The TT runner with K=3 drafts: moves it does not implement yet ---------


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="mamba_cache_groups rejects MambaSpec.num_speculative_blocks > 0",
)
@pytest.mark.parametrize("mode", ["none", "align"])
def test_tt_runner_accepts_speculative_state_blocks(mode):
    kv_cache_config = _kv_cache_config(
        _planner_config(mode), _qwen38_specs(mode, num_spec=K), NUM_BLOCKS
    )
    mamba_cache_groups(
        kv_cache_config.kv_cache_groups, SimpleNamespace(mamba_cache_mode=mode)
    )


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="state_block_ids_per_layer carries one block per row; a speculating "
    "row needs the 1 + K window of mamba_get_block_table_tensor",
)
def test_tt_runner_state_window_has_1_plus_k_columns():
    spec = _qwen38_specs("align", num_spec=K)["model.layers.0.linear_attn"]
    groups = MambaCacheGroups(
        group_indices=(0,),
        mamba_cache_mode="align",
        block_size=BLOCK_SIZE,
        attention_group_index=3,
    )
    block_table = torch.arange(1, 4 * 11 + 1, dtype=torch.int32).reshape(4, 11)
    seq_lens = torch.tensor([1, 1024, 1028, 8190])
    expected = mamba_get_block_table_tensor(block_table, seq_lens, spec, "align")
    assert expected.shape == (4, 1 + K)
    assert torch.equal(groups.state_block_ids(block_table, seq_lens), expected)


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="TT passes no accepted-token count: the initial state of a "
    "speculating row is at window column num_accepted - 1, not column 0",
)
@pytest.mark.parametrize("mode", ["none", "align"])
def test_tt_runner_hands_the_model_the_initial_state_block(mode):
    h = _Harness(mode, num_spec=K, retention=None, plugin_groups=True)
    h.add("a", _prompt(1, 1010), max_tokens=40)
    h.add("bb", _prompt(2, 700), max_tokens=40)
    h.run(accept=_accept_pattern)
    for step in h.steps:
        assert step.plugin_state_ids == step.initial_blocks


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="align_state_copies copies block_ids[prev] -> block_ids[curr]; "
    "upstream shifts the source by the accepted drafts (temporal column "
    "prev + num_accepted - 1, conv rows num_accepted - 1)",
)
def test_tt_runner_preprocess_copies_shift_by_accepted_drafts():
    h = _Harness("align", num_spec=K, retention=None, plugin_groups=True)
    h.add("a", _prompt(1, 1010), max_tokens=60)
    h.add("bb", _prompt(2, 2030), max_tokens=60)
    h.run(accept=_accept_pattern)
    assert any(c.conv_row_offset for s in h.steps for c in s.pre_copies)
    for step in h.steps:
        assert step.plugin_copies == [(c.temporal_src, c.dst) for c in step.pre_copies]


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="the TT runner has no copy after the accept walk "
    "(postprocess_mamba_align_gpu) that puts a reached block-boundary state "
    "into its column; prefix hits on decode-time boundaries need it when "
    "prefix_cache_retention_interval is None or positive",
)
def test_tt_runner_copies_the_boundary_state_after_acceptance():
    h = _Harness("align", num_spec=K, retention=None, plugin_groups=True)
    h.add("a", _prompt(1, 1900), max_tokens=250)
    h.run(max_steps=400, accept=_accept_pattern)
    post = {(c.temporal_src, c.dst) for s in h.steps for c in s.post_copies}
    assert post
    plugin = {pair for s in h.steps for pair in s.plugin_copies}
    assert post <= plugin


@pytest.mark.xfail(
    raises=pytest.fail.Exception,
    strict=True,
    reason="mamba_cache_groups does not check that mode align uses one block "
    "size for GDN and attention; the scheduler aligns chunk ends to the "
    "smallest group block size, so states land mid-column",
)
def test_mode_align_requires_one_block_size_for_gdn_and_attention():
    kv_cache_config = _kv_cache_config(
        _planner_config("align"),
        _qwen38_specs("align", num_spec=0, gdn_block_size=2 * BLOCK_SIZE),
        NUM_BLOCKS,
    )
    with pytest.raises(ValueError):
        mamba_cache_groups(
            kv_cache_config.kv_cache_groups,
            SimpleNamespace(mamba_cache_mode="align"),
        )
