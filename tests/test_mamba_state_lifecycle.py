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
  contract asks of the recurrent kernels (zeroing first, then read the
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
    SpeculativeConfig,
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
from vllm_tt_plugin.config import store_tt_spec_plan
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler
from vllm_tt_plugin.spec_admission import (
    MODEL_OWNED_DRAFT_METHOD,
    MODEL_OWNED_DRAFT_SENTINEL,
)
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    DRAFTER_STATE_INTERNAL,
    DraftOutput,
    SpecPlan,
)

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
EOS = 2
NGRAM_LOOKUP = {"prompt_lookup_min": 2, "prompt_lookup_max": 4}


def _is_attention(layer: int) -> bool:
    return layer % 4 == 3


def _qwen38_specs(
    mode: str = "none",
    checkpoint_alignment: int | None = None,
    num_speculative_blocks: int = 0,
):
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
        num_speculative_blocks=num_speculative_blocks,
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
    spec_plan: SpecPlan | None = None,
    spec_method: str = MODEL_OWNED_DRAFT_METHOD,
) -> TTScheduler:
    """A real ``TTScheduler`` over a real ``KVCacheManager``.

    Prefix caching is on exactly in mode ``align``. With ``spec_plan`` the
    scheduler schedules the drafts of the model-owned drafter.
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
    if spec_plan is not None:
        # Attached after construction, as the platform would resolve a plan
        # from the stub model class, which declares none.
        config.speculative_config = SpeculativeConfig(
            method=spec_method,
            num_speculative_tokens=spec_plan.effective_k,
            target_model_config=config.model_config,
            target_parallel_config=config.parallel_config,
            **(
                {"model": MODEL_OWNED_DRAFT_SENTINEL}
                if spec_method == MODEL_OWNED_DRAFT_METHOD
                else NGRAM_LOOKUP
            ),
        )
        store_tt_spec_plan(config, spec_plan)
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


class UpstreamAlignRunner:
    """The Mamba bookkeeping of upstream's GPU model runner in mode ``align``.

    Calls upstream ``preprocess_mamba`` and records, instead of copying, the
    block-table columns of each copy it makes before the forward: the column
    of the request's state after its last step (the running column plus the
    accepted-token bias), and the column of the step's window column 0.
    ``mamba_state_idx`` is the map upstream keeps across steps.

    Each step is handed the runner's accepted counts. Upstream's post-step
    ``postprocess_mamba_align_gpu`` would also copy a reached block-boundary
    state into the block of its own column and then reset the count to 1 when
    that column is the running one; the TT runner reads the same state from
    window column ``count - 1`` instead, and vLLM keeps no decode-reached
    boundary at ``prefix_cache_retention_interval`` 0.
    """

    def __init__(self, kv_cache_config, group_indices) -> None:
        self.kv_cache_config = kv_cache_config
        self.mamba_state_idx: dict[str, int] = {}
        self.copy_bufs = SimpleNamespace(
            mamba_group_ids=list(group_indices),
            mamba_spec=kv_cache_config.kv_cache_groups[group_indices[0]].kv_cache_spec,
            offset=0,
        )

    def columns(self, scheduler_output, row_req_ids, requests, counts):
        """Request id -> (source column, window column 0) of the step.

        ``counts`` maps each row's request to its previous step's committed
        tokens. Without a copy the step reads window column ``count - 1``, and
        a request at position 0 reads nothing (source -1).
        """
        copied: dict[str, tuple[int, int]] = {}

        def record(_bufs, _config, _funcs, _groups, src, dst, bias, req_state, _ctx):
            copied[req_state.req_id] = (src + bias, dst)

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
                    num_accepted_tokens_cpu=np.array(
                        [counts[r] for r in row_req_ids], np.int32
                    ),
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
                result[r] = (running + counts[r] - 1 if started else -1, running)
        return result


class _DraftingModel(_RecordingModel):
    """A ``_RecordingModel`` whose drafter offers what ``_Harness`` scripts."""

    def __init__(self, harness: _Harness):
        super().__init__()
        self.harness = harness

    def propose_draft_tokens(
        self, num_drafts, committed, positions, counts, hidden=None
    ) -> DraftOutput:
        return self.harness.propose(num_drafts, positions)


class _Harness:
    """Drives ``TTScheduler`` and ``TTModelRunner`` through whole steps.

    With ``num_speculative_tokens`` K > 0 the launch speculates with the
    model-owned drafter, for which ``propose`` stands in, or with the
    runner's n-gram proposer (``spec_method`` ``ngram``). Each verify accepts
    a scripted prefix of the drafts through the runner's own accept walk and
    commit, so the scheduler and the runner see the counts a real verify
    produces.
    """

    def __init__(
        self,
        num_blocks: int,
        max_num_batched_tokens: int = 2048,
        *,
        mode: str = "none",
        checkpoint_alignment: int | None = None,
        prefix_match_unit: int | None = None,
        num_speculative_tokens: int = 0,
        narrow_decode: bool = True,
        offers: tuple[int, ...] | None = None,
        spec_method: str = MODEL_OWNED_DRAFT_METHOD,
    ):
        cfg = _vllm_config(mode)
        cfg.model_config.max_model_len = MAX_MODEL_LEN
        cfg.model_config.original_max_model_len = MAX_MODEL_LEN
        cfg.scheduler_config.max_num_seqs = MAX_NUM_SEQS
        cfg.cache_config.prefix_match_unit = prefix_match_unit
        self.mode = mode
        self.num_spec = num_speculative_tokens
        spec_plan = None
        if num_speculative_tokens:
            spec_plan = SpecPlan(
                effective_k=num_speculative_tokens,
                lanes_per_request=1,
                extra_bytes_per_seq=0,
                extra_bytes_per_token=0,
                accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
                drafter_state=DRAFTER_STATE_INTERNAL,
                supports_narrow_decode=narrow_decode,
            )
            cfg.speculative_config = SimpleNamespace(
                method=spec_method,
                num_speculative_tokens=num_speculative_tokens,
                use_eagle=lambda: False,
                use_eagle_block_drop=lambda: False,
                **NGRAM_LOOKUP,
            )
            store_tt_spec_plan(cfg, spec_plan)
        self.kv_cache_config = _kv_cache_config(
            cfg,
            _qwen38_specs(mode, checkpoint_alignment, num_speculative_tokens),
            num_blocks,
        )
        self.scheduler = _scheduler(
            self.kv_cache_config,
            max_num_batched_tokens,
            mode,
            prefix_match_unit,
            spec_plan=spec_plan,
            spec_method=spec_method,
        )
        self.hash_block_size = self.scheduler.hash_block_size
        self.model = (
            _DraftingModel(self)
            if num_speculative_tokens and spec_method == MODEL_OWNED_DRAFT_METHOD
            else _RecordingModel()
        )
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
        # Request id -> start position of each step that began from a block
        # vLLM kept for the prefix cache (new and resumed requests), and the
        # first GDN group's block it read.
        self.hits: dict[str, list[int]] = {}
        self.hit_blocks: dict[str, list[int]] = {}
        # (request id, position) of every prefill checkpoint written, and the
        # first GDN group's block it went to.
        self.checkpoints: list[tuple[str, int]] = []
        self.checkpoint_blocks: list[int] = []
        # Steps in which a row read a block outside its own window.
        self.crossings = 0
        # Request id -> its tokens and its (start, end) positions in each
        # prefill step, and its first step.
        self.prefill_chunks: dict[str, list[int]] = {}
        self.prefill_spans: dict[str, list[tuple[int, int]]] = {}
        self.first_step: dict[str, int] = {}
        self.cached_states_checked = 0
        self.output_base = 50_000
        self.num_steps = 0
        self.num_added = 0
        # Speculative decoding. ``offers`` cycles the number of drafts the
        # drafter offers a row; ``accepted[(v, a)]`` counts verify rows that
        # carried ``v`` valid drafts and committed ``a`` tokens.
        self.offers = offers or tuple(range(num_speculative_tokens, -1, -1))
        self.num_offers = 0
        self.accept_cycle: dict[int, int] = {}
        self.accepted: dict[tuple[int, int], int] = {}
        self.step_kinds: dict[str, int] = {}
        # (step kind, spec_mode, perform_device_sampling) of every step, and
        # per verify (spec_mode, perform_device_sampling, request id -> valid
        # drafts).
        self.step_modes: set[tuple] = set()
        self.verify_steps: list[tuple[str, bool, dict[str, int]]] = []
        # Verify rows whose tokens span two block-table columns, and steps
        # whose window began in a lower column than the request's previous
        # one.
        self.verify_crossings = 0
        self.windows_moved_back = 0
        self.last_window: dict[str, int] = {}
        # Request id -> its output length at which the drafter offers EOS as
        # its second draft, which the verify then accepts with the drafts
        # after it.
        self.eos_after: dict[str, int] = {}
        self.proposing_req_ids: list[str] = []
        self.draft_rng = torch.Generator().manual_seed(7)

    def add(
        self,
        req_id: str,
        prompt_len: int = 0,
        max_tokens: int = 8,
        prompt: list[int] | None = None,
        ignore_eos: bool = True,
        sampling: dict | None = None,
    ) -> None:
        init_none_hash(sha256)
        # Greedy unless ``sampling`` says otherwise, so the argmax_ids walk can
        # certify a speculating request.
        params = SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            **{"temperature": 0.0, **(sampling or {})},
        )
        params.update_from_generation_config({}, eos_token_id=EOS)
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

    def kept_gdn_blocks(self) -> set[int]:
        """The GDN blocks vLLM keeps under a prefix hash."""
        return {
            block.block_id
            for block in self.scheduler.kv_cache_manager.block_pool.blocks
            if block.block_hash is not None
            and get_group_id(block.block_hash) in self.groups
        }

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

    def propose(self, num_drafts: int, positions: torch.Tensor) -> DraftOutput:
        """The drafter: each live row offers the next count of ``offers``."""
        rows = int(positions.shape[0])
        drafts = torch.randint(
            3, 32, (rows, num_drafts), generator=self.draft_rng, dtype=torch.int32
        )
        num_valid = torch.zeros(rows, dtype=torch.int32)
        for row, req_id in enumerate(self.proposing_req_ids):
            if int(positions[row, 0]) < 0:
                continue
            offered = self.offers[self.num_offers % len(self.offers)]
            self.num_offers += 1
            request = self.requests[req_id]
            if (
                req_id in self.eos_after
                and request.num_output_tokens >= self.eos_after[req_id]
            ):
                offered = num_drafts
                drafts[row, 1] = EOS
            num_valid[row] = offered
        return DraftOutput(draft_token_ids=drafts, num_valid=num_valid)

    def verify_ids(self, model_input) -> torch.Tensor:
        """``argmax_ids`` that accept a scripted prefix of each row's drafts.

        A row with ``v`` valid drafts commits ``a`` tokens, ``a`` cycling
        through ``1..v+1`` per ``v``: the first ``a - 1`` drafts match, and
        draft ``a - 1`` is rejected unless ``a`` is ``v + 1`` (all drafts and
        the bonus). A row whose drafts carry EOS accepts all of them.
        """
        tokens = model_input.input_tokens
        rows, width = (int(n) for n in tokens.shape)
        drafts = model_input.draft_token_ids
        ids = torch.full((rows, width), 5, dtype=torch.int32)
        for row, req_id in enumerate(model_input.row_req_ids):
            v = int(model_input.num_valid_drafts[row])
            if v and bool((drafts[row, :v] == EOS).any()):
                a = v + 1
            else:
                cycle = self.accept_cycle.get(v, 0)
                self.accept_cycle[v] = cycle + 1
                a = 1 + cycle % (v + 1)
            self.accepted[(v, a)] = self.accepted.get((v, a), 0) + 1
            ids[row, : a - 1] = drafts[row, : a - 1]
            if a - 1 < v:
                # Any other token rejects draft a - 1.
                ids[row, a - 1] = 3 + (int(drafts[row, a - 1]) - 2) % 29
            else:
                ids[row, a - 1] = 3 + (self.num_steps + row) % 29
        return ids

    def step(self) -> bool:
        out = self.scheduler.schedule()
        self.preempted.update(out.preempted_req_ids or ())
        if not out.total_num_scheduled_tokens:
            self.runner.build_model_input(out, None)
            return bool(self.scheduler.requests)
        self.model.calls.clear()
        model_input = self.runner.build_model_input(out, None)
        for name, kwargs in self.model.calls:
            if name == "zero_kv_cache_blocks":
                self.ledger.zero(self.groups, kwargs["block_ids"])
                self.zeroed += kwargs["block_ids"]
        row_req_ids = model_input.row_req_ids
        is_prefill = model_input.prompt_lens is not None
        verify = model_input.spec_mode is not None
        kind = "prefill" if is_prefill else "verify" if verify else "decode"
        self.step_kinds[kind] = self.step_kinds.get(kind, 0) + 1
        device_sampled = bool(model_input.perform_device_sampling)
        self.step_modes.add((kind, model_input.spec_mode, device_sampled))
        if verify:
            valid = model_input.num_valid_drafts[: len(row_req_ids)].tolist()
            self.verify_steps.append(
                (model_input.spec_mode, device_sampled, dict(zip(row_req_ids, valid)))
            )
        positions = torch.as_tensor(model_input.input_positions)
        if positions.dim() == 2:
            positions = positions[:, 0]
        live = len(row_req_ids)
        counts = (
            [1] * live
            if model_input.accepted_counts is None
            else model_input.accepted_counts[:live].tolist()
        )
        counts_by_req = dict(zip(row_req_ids, counts))
        state = model_input.mamba_state
        assert [g.layer_names for g in state.groups] == [
            tuple(self.kv_cache_config.kv_cache_groups[g].layer_names)
            for g in self.groups
        ]
        width = 1 + self.num_spec
        assert {tuple(g.windows.shape) for g in state.groups} == {
            (self.runner.tt_max_batch_size, width)
        }
        if is_prefill:
            mask = state.intermediate_prefill_mask
            assert mask.dtype == torch.bool
            assert mask[live:].all()
        else:
            assert state.intermediate_prefill_mask is None
        upstream = (
            self.upstream.columns(out, row_req_ids, self.runner.requests, counts_by_req)
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
        attention = self.runner._mamba_groups.attention_group_index
        rows = []
        for row, req_id in enumerate(row_req_ids):
            request = self.scheduler.requests[req_id]
            start = int(positions[row])
            end = start + out.num_scheduled_tokens[req_id]
            assert start == computed[req_id]
            self.first_step.setdefault(req_id, self.num_steps)
            if is_prefill:
                self.prefill_chunks.setdefault(req_id, []).append(end - start)
                self.prefill_spans.setdefault(req_id, []).append((start, end))
                # The drafter starts a request only after its last chunk.
                assert bool(state.intermediate_prefill_mask[row]) == (
                    end < request.num_tokens
                )
            if verify and (start // BLOCK_SIZE) != ((end - 1) // BLOCK_SIZE):
                self.verify_crossings += 1
            # The attention page table is the scheduler's attention blocks.
            held = self.scheduler_blocks(req_id, attention)
            page_table = model_input.block_tables_per_group[attention][row]
            assert page_table[: len(held)].tolist() == held, (req_id, held)
            self.rows_seen.setdefault(req_id, set()).add(row)
            if req_id in self.preempted and start == 0:
                self.resumed_from_zero.add(req_id)
            if req_id in fresh and start > 0:
                self.hits.setdefault(req_id, []).append(start)
                self.hit_blocks.setdefault(req_id, []).append(
                    int(state.groups[0].start_blocks[row])
                )
            position = (
                0
                if state.checkpoint_positions is None
                else int(state.checkpoint_positions[row, 0])
            )
            if position:
                self.checkpoints.append((req_id, position))
                self.checkpoint_blocks.append(
                    int(state.groups[0].checkpoint_blocks[row, 0])
                )
            blocks = []
            for i, g in enumerate(self.groups):
                group = state.groups[i]
                src = int(group.start_blocks[row])
                window = group.windows[row].tolist()
                checkpoint = (
                    (position, int(group.checkpoint_blocks[row, 0]))
                    if position
                    else None
                )
                self.check_blocks(
                    req_id, g, start, end, src, window, checkpoint, counts[row]
                )
                if upstream is not None:
                    row_ids = self.runner.requests[req_id].block_ids[g]
                    src_column, window_column = upstream[req_id]
                    assert src == (row_ids[src_column] if src_column >= 0 else 0)
                    assert window[0] == row_ids[window_column]
                    if i == 0:
                        last = self.last_window.get(req_id)
                        if req_id not in fresh and last is not None:
                            self.windows_moved_back += window_column < last
                        self.last_window[req_id] = window_column
                blocks.append((g, src, window, checkpoint))
            rows.append((request, start, end, blocks))
        # The kernels read every start state before any row writes. No two
        # rows write one block, and a start block outside the row's own window
        # is written by no row: it can be a block vLLM keeps for the prefix
        # cache.
        written: dict[tuple[int, int], int] = {}
        for row, (request, start, _end, blocks) in enumerate(rows):
            for g, src, window, checkpoint in blocks:
                self.ledger.read(g, src, list(request.all_token_ids), start)
                own = [*window, *([checkpoint[1]] if checkpoint else [])]
                assert len(set(own)) == len(own), (request.request_id, own)
                for block in own:
                    assert written.setdefault((g, block), row) == row, (
                        f"rows {written[(g, block)]} and {row} write block {block}"
                    )
        crossed = False
        for row, (_request, _start, _end, blocks) in enumerate(rows):
            for g, src, window, _checkpoint in blocks:
                if src and src not in window:
                    crossed = True
                    assert (g, src) not in written, (
                        f"block {src} is row {row}'s start state and row "
                        f"{written[(g, src)]} writes it"
                    )
        self.crossings += crossed
        for row, (request, start, end, blocks) in enumerate(rows):
            tokens = list(request.all_token_ids)
            for g, _src, window, checkpoint in blocks:
                if checkpoint:
                    self.ledger.write(g, checkpoint[1], tokens, checkpoint[0])
                if verify:
                    # Window column t takes the state after query token t;
                    # the columns past the row's valid drafts take anything.
                    block = model_input.input_tokens[row].tolist()
                    assert block[0] == tokens[start]
                    valid = int(model_input.num_valid_drafts[row])
                    for t, dst in enumerate(window):
                        if t <= valid:
                            self.ledger.states[(g, dst)] = tuple(
                                tokens[:start] + block[: t + 1]
                            )
                            self.ledger.gdn_blocks.add(dst)
                        else:
                            self.ledger.states[(g, dst)] = GARBAGE
                    continue
                self.ledger.write(g, window[0], tokens, end)
                for dst in window[1:]:
                    self.ledger.states[(g, dst)] = GARBAGE
        for i, g in enumerate(self.groups):
            group = state.groups[i]
            assert set(group.windows[live:].reshape(-1).tolist()) <= {0}
            assert set(group.start_blocks[live:].tolist()) <= {0}
            self.ledger.run_padding_row(g)
        if self.mode == "align":
            self.check_cached_states()
        output = self.finish_step(model_input, rows, is_prefill, verify)
        self.scheduler.update_from_output(out, output)
        drafts = self.runner.take_draft_token_ids()
        if drafts is not None:
            self.scheduler.update_draft_token_ids(drafts)
        self.num_steps += 1
        return True

    def finish_step(self, model_input, rows, is_prefill, verify) -> ModelRunnerOutput:
        """Commit the step's tokens through the runner and build its output."""
        row_req_ids = model_input.row_req_ids
        self.proposing_req_ids = list(row_req_ids)
        if verify:
            answer = self.verify_ids(model_input)
            if model_input.spec_mode == ACCEPT_MODE_LOGITS:
                # The target puts all its mass on the scripted ids, so the
                # host rejection sampling commits the scripted counts.
                vocab = int(self.runner.vocab_size)
                answer = torch.nn.functional.one_hot(answer.to(torch.int64), vocab)
                answer = torch.log(answer.to(torch.float32))
            walk = self.runner.walk_spec_acceptance(model_input, answer)
            prefixes = self.runner.spec_committed_prefixes(
                model_input, walk.committed, walk.counts
            )
            self.runner.commit_spec_acceptance(
                model_input, walk.committed, walk.counts, None
            )
            return self.runner.build_spec_runner_output(row_req_ids, prefixes)
        sampled: list[list[int]] = []
        for row, (request, _start, end, _blocks) in enumerate(rows):
            emits = end >= request.num_tokens
            token = (
                3 + (self.num_steps + row) % 29
                if self.num_spec
                else self.output_base + self.num_steps
            )
            sampled.append([token] if emits else [])
        # The runner keeps its own copy of each request's tokens, as its
        # sampling path does after a forward.
        emitted = [(r, t) for r, t in zip(row_req_ids, sampled) if t]
        if emitted:
            self.runner._apply_sampled_tokens_to_state(
                torch.tensor([t for _, t in emitted], dtype=torch.int32),
                req_ids=[r for r, _ in emitted],
            )
        if self.num_spec and not is_prefill:
            # As the runner's ordinary-decode tail does: the drafter is asked
            # after every decode step.
            self.runner.propose_after_plain_step(
                model_input,
                torch.tensor([t for _, t in emitted], dtype=torch.int32),
                list(row_req_ids),
            )
        return ModelRunnerOutput(
            req_ids=list(row_req_ids),
            req_id_to_index={r: i for i, r in enumerate(row_req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def check_blocks(
        self, req_id, group, start, end, src, window, checkpoint, count
    ) -> None:
        """A row's blocks are blocks vLLM holds for the request, at the
        columns upstream ``mamba_get_block_table_tensor`` selects."""
        held = self.scheduler_blocks(req_id, group)
        width = 1 + self.num_spec
        table = torch.zeros((1, max(len(held), width)), dtype=torch.int32)
        table[0, : len(held)] = torch.tensor(held, dtype=torch.int32)
        upstream = mamba_get_block_table_tensor(
            table,
            torch.tensor([end]),
            self.kv_cache_config.kv_cache_groups[group].kv_cache_spec,
            self.mode,
        )
        assert window == upstream[0, :width].tolist(), (req_id, group, held)
        if self.mode == "none":
            assert held == window, (req_id, group, held)
            assert src == (held[count - 1] if start else 0), (req_id, held, count)
        else:
            column = (end - 1) // BLOCK_SIZE
            assert window == held[column : column + width], (req_id, group, held)
            if not start:
                assert src == 0
            elif self.num_spec:
                # vLLM still holds the start block for the request, wherever
                # it now sits in the row.
                assert src in held, (req_id, group, src, held)
            else:
                assert src == held[(start - 1) // BLOCK_SIZE], (req_id, group, held)
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


@pytest.mark.parametrize("mode", ["none", "align"])
def test_requests_finish_and_their_blocks_serve_later_requests(mode):
    """Completion: freed GDN blocks come back to new requests.

    Two waves of requests share a pool that holds one wave. A block that held
    a finished request's GDN state can go to another request's GDN group,
    which starts from zero state, or to an attention group, which vLLM lists
    for zeroing. Chunked prefill splits the 3000-token prompt.
    """
    h = _Harness(num_blocks=24, mode=mode)
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


@pytest.mark.parametrize("mode", ["none", "align"])
def test_abort_preemption_and_row_movement_keep_each_state_with_its_request(mode):
    """Abort, preemption and row movement in one run.

    The pool holds 14 blocks: three 1000-token requests with one attention
    block and three GDN blocks each, so the fourth waits until request r2 is
    aborted mid-decode. Decoding past the 1024-token boundary needs a second
    attention block per request, and in mode align new GDN blocks too, and the
    pool cannot give every request one: the scheduler preempts. A preempted
    request loses its blocks and resumes as a prefill from position 0: a
    1000-token prompt reaches no block boundary whose state vLLM could keep.
    The abort, the preemption and finishing requests move the remaining rows.
    """
    h = _Harness(num_blocks=15, mode=mode)
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


@pytest.mark.parametrize("max_num_batched_tokens", [1024, 2048, 4096])
def test_prefill_checkpoints_match_the_blocks_mambamanager_reserves(
    max_num_batched_tokens,
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


@pytest.mark.parametrize("mode", ["none", "align"])
def test_prefill_chunks_resume_at_kv_block_boundaries(mode):
    """The TT model writes a prefill chunk's attention keys and values only
    from the start of a KV block, so with MambaSpec groups every chunk that
    does not finish its prompt ends on a multiple of the 1024-token block, not
    on the default 128-token grid.

    Request a's 1300-token prompt leaves 748 tokens of the 2048-token step for
    request b. On the 128-token grid b would prefill 640 tokens and its next
    chunk would start inside block 0; on the block grid b waits for the next
    step. In mode align vLLM's own split also ends chunks on block boundaries.
    """
    h = _Harness(num_blocks=40, max_num_batched_tokens=2048, mode=mode)
    assert h.scheduler._prefill_chunk_alignment == BLOCK_SIZE
    for req_id, prompt_len in (("a", 1300), ("b", 3000), ("c", 2600), ("d", 900)):
        h.add(req_id, prompt_len, max_tokens=4)
    h.run()

    chunked = {r: spans for r, spans in h.prefill_spans.items() if len(spans) > 1}
    assert chunked, "no prompt was chunked"
    for req_id, spans in h.prefill_spans.items():
        for start, end in spans[:-1]:
            assert start % BLOCK_SIZE == end % BLOCK_SIZE == 0, (req_id, spans)


def test_an_operator_chunk_alignment_must_be_a_multiple_of_the_kv_block():
    """``additional_config['tt']['prefill_chunk_alignment']`` defaults to the
    KV block size for a model with MambaSpec groups and must be a positive
    multiple of it; a model without them keeps the 128-token default."""
    kv_cache_config = _Harness(num_blocks=32).kv_cache_config

    for alignment in (640, 0):
        with pytest.raises(
            ValueError,
            match=rf"prefill_chunk_alignment'\]={alignment} is not a positive "
            "multiple of the KV block size 1024",
        ):
            _scheduler(kv_cache_config, 2048, prefill_chunk_alignment=alignment)
    scheduler = _scheduler(kv_cache_config, 4096, prefill_chunk_alignment=2048)
    assert scheduler._prefill_chunk_alignment == 2048


def test_prefix_match_unit_below_the_kv_block_fails_at_startup():
    """A ``prefix_match_unit`` below the block size makes vLLM end prefill
    chunks, and resume prefix hits, inside a block, off the KV block grid
    that prefill chunks of a MambaSpec model must start on; the scheduler
    refuses to start rather than stall."""
    kv_cache_config = _Harness(num_blocks=32, mode="align").kv_cache_config

    with pytest.raises(ValueError, match="prefix_match_unit=128"):
        _scheduler(kv_cache_config, 2048, "align", prefix_match_unit=128)
    _scheduler(kv_cache_config, 2048, "align", prefix_match_unit=BLOCK_SIZE)


@pytest.mark.parametrize("mode", ["none", "align"])
def test_forced_prefix_cache_reset_resumes_running_requests_on_new_blocks(mode):
    """``reset_prefix_cache(reset_running_requests=True)`` preempts the running
    requests and resumes them in the same step, on new blocks. The runner drops
    their persistent-batch rows, which hold the freed blocks, and adds them
    back as resumed requests: a row that kept its old row would append the new
    attention blocks to the freed ones and read its GDN state from a freed
    block."""
    h = _Harness(num_blocks=32, max_num_batched_tokens=4096, mode=mode)
    h.add("a", 2500, max_tokens=40)
    h.add("b", 1500, max_tokens=40)
    for _ in range(5):
        h.step()
    assert {"a", "b"} <= set(h.runner.input_batch.req_id_to_index)

    assert h.scheduler.reset_prefix_cache(reset_running_requests=True)
    h.run()

    assert not h.scheduler.requests
    assert h.resumed_from_zero == {"a", "b"}


# region Prefix reuse in cache mode align


@pytest.mark.parametrize(
    ("checkpoint_alignment", "max_num_batched_tokens", "prefill_chunks"),
    [
        # One prefill of 2500 tokens writes the state after 2048 tokens into
        # the block vLLM reserved for it.
        (64, 4096, [2500]),
        # Without checkpoint blocks vLLM ends a chunk at 2048 instead.
        (None, 4096, [2048, 452]),
        # 1024-token steps: no chunk holds a checkpoint.
        (64, 1024, [1024, 1024, 452]),
        (None, 1024, [1024, 1024, 452]),
    ],
)
def test_prompt_boundary_state_is_the_same_unchunked_and_chunked(
    checkpoint_alignment, max_num_batched_tokens, prefill_chunks
):
    """Request a prefills a 2500-token prompt, then request b sends the same
    prompt: an exact hit on the state after 2048 tokens, the prompt's last
    block boundary, from which b prefills the remaining 452 tokens. Whether a
    wrote that state as a prefill checkpoint or as the end of a chunk, b reads
    the state of exactly its first 2048 tokens, as a cold prefill computes it.
    """
    h = _Harness(
        num_blocks=32,
        max_num_batched_tokens=max_num_batched_tokens,
        mode="align",
        checkpoint_alignment=checkpoint_alignment,
    )
    prompt = _prompt(1, 2500)
    h.add("a", prompt=prompt, max_tokens=8)
    h.run()
    h.add("b", prompt=prompt, max_tokens=8)
    h.run()

    assert h.prefill_chunks["a"] == prefill_chunks
    assert h.prefill_chunks["b"] == [452]
    assert h.hits == {"b": [2048]}
    if prefill_chunks == [2500]:
        assert h.checkpoints == [("a", 2048)]
        assert h.hit_blocks["b"] == h.checkpoint_blocks
    else:
        assert h.checkpoints == []


def test_partial_hits_restore_the_longest_kept_state_and_write_new_checkpoints():
    """A partial hit covers less than the prompt, and a prefill that continues
    from it can write a checkpoint of its own.

    Request a's 2500 tokens leave the state after 2048 tokens. Request b sends
    3600 tokens with the same first 2500: it hits 2048, reads column 1's block,
    writes the state after 3072 tokens into the column 2 block vLLM reserves,
    and its end state into column 3. Request c shares 3500 tokens with b and
    hits b's checkpoint.
    """
    h = _Harness(
        num_blocks=40,
        max_num_batched_tokens=4096,
        mode="align",
        checkpoint_alignment=64,
    )
    shared = _prompt(1, 4000)
    h.add("a", prompt=shared[:2500], max_tokens=4)
    h.run()
    h.add("b", prompt=shared[:3600], max_tokens=4)
    h.run()
    h.add("c", prompt=shared[:3500] + _prompt(2, 300), max_tokens=4)
    h.run()

    assert h.hits == {"b": [2048], "c": [3072]}
    assert h.checkpoints == [("a", 2048), ("b", 3072)]
    assert h.hit_blocks == {
        "b": [h.checkpoint_blocks[0]],
        "c": [h.checkpoint_blocks[1]],
    }


def test_diverging_continuations_read_one_kept_prefix_and_leave_it_intact():
    """Requests b, c and d share request a's first 2048 tokens and then
    diverge. All three hit the same kept block in one step, each writes its
    own blocks, and they decode across the 3072-token boundary. The kept block
    keeps the state of the shared 2048 tokens throughout:
    ``_Harness.check_cached_states`` compares it after every step, and no
    block a row reads as its start state is written in the same step."""
    h = _Harness(num_blocks=48, max_num_batched_tokens=4096, mode="align")
    prefix = _prompt(1, 2048)
    h.add("a", prompt=prefix + _prompt(2, 452), max_tokens=4)
    h.run()
    for i, name in enumerate("bcd"):
        h.add(name, prompt=prefix + _prompt(10 + i, 700 + 100 * i), max_tokens=400)
    h.run(max_steps=800)

    assert h.hits == {"b": [2048], "c": [2048], "d": [2048]}
    assert len({block for blocks in h.hit_blocks.values() for block in blocks}) == 1
    assert len({h.first_step[r] for r in "bcd"}) == 1, "the hits ran in two steps"
    assert h.crossings >= 4, "not every request crossed into a new column"


def test_evicted_prefix_states_are_recomputed_and_their_blocks_reused():
    """A small pool: three 3000-token requests after request a evict a's kept
    state, its blocks go to other requests' attention and GDN groups, and
    request b with a's prompt prefills from position 0."""
    h = _Harness(num_blocks=16, max_num_batched_tokens=4096, mode="align")
    prompt = _prompt(1, 2500)
    h.add("a", prompt=prompt, max_tokens=8)
    h.run()
    kept = h.kept_gdn_blocks()
    assert kept, "request a left no kept GDN state"
    for i in range(3):
        h.add(f"filler-{i}", 3000, max_tokens=8)
    h.run()
    h.add("b", prompt=prompt, max_tokens=8)
    h.run()

    assert "b" not in h.hits
    assert kept & set(h.zeroed), "no kept GDN block went back to attention"


def test_preempted_requests_resume_from_zero_or_from_a_kept_prefix():
    """Four requests decode past block boundaries in a 16-block pool. vLLM
    preempts some; a resumed request replays its tokens from position 0, or
    from the state at a block boundary of its own prompt that vLLM kept."""
    h = _Harness(num_blocks=16, max_num_batched_tokens=4096, mode="align")
    for i in range(MAX_NUM_SEQS):
        h.add(f"r{i}", 1000 + 300 * i, max_tokens=200)
    h.run(max_steps=2000)

    assert not h.scheduler.requests
    assert h.resumed_from_zero, "no preempted request resumed from position 0"
    assert h.preempted & set(h.hits), "no preempted request resumed from its prefix"


@pytest.mark.parametrize("prompt_len", [1000, 1023, 1024, 1025])
def test_align_running_state_moves_into_each_new_column_once(prompt_len):
    """In mode align the request's state moves to a new block when its tokens
    cross 1024: in the prefill chunk or the decode that crosses, the row reads
    column 0's block and writes column 1's, and from then on it updates
    column 1's block in place."""
    h = _Harness(num_blocks=16, mode="align")
    h.add("a", prompt_len, max_tokens=60)
    h.run()

    assert h.crossings == 1


# endregion
