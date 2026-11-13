# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A request admitted with exactly one uncomputed token under speculation.

With speculative decoding on, vLLM's ``Scheduler.schedule`` pads a request it
admits from a waiting queue with exactly one uncomputed token to
``1 + num_spec_tokens`` scheduled tokens, with ``[-1] * num_spec_tokens`` in
``scheduled_spec_decode_tokens``, for its GPU runner's uniform decode shape.
Two ways in: a new request whose prefix-cache hit covers all but its last
prompt token, and a preempted request that resumes with such a hit.
``TTModelRunner`` runs both as a prefill, so ``TTScheduler`` must schedule the
one token alone.

``_Harness`` drives a real ``TTScheduler`` with its ``KVCacheManager`` (prefix
caching on) and a real ``TTModelRunner`` through the engine's synchronous step
order. The model is ``_Target``: a deterministic greedy target with a
model-owned drafter whose drafts agree with the target for a varying number of
leading drafts. ``_Target`` records every prefill row that ends past its
request's last token, and the outputs must equal the target's greedy
continuation of each prompt.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.config import store_tt_spec_plan
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler
from vllm_tt_plugin.spec_admission import (
    MODEL_OWNED_DRAFT_METHOD,
    MODEL_OWNED_DRAFT_SENTINEL,
)
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    DRAFTER_STATE_INTERNAL,
    PLACEHOLDER_TOKEN_ID,
    DraftOutput,
    SpecPlan,
    VerifyOutput,
)

K = 3
T = K + 1
BLOCK_SIZE = 16
MAX_MODEL_LEN = 256
MAX_NUM_SEQS = 2
VOCAB = 32
FIRST_ID = 3
EOS = 2
LOCAL_MODEL_CONFIG = Path(__file__).parent.parent / "model_configs" / "qwen2"


def _target(prefix: list[int]) -> int:
    """The target model's greedy token after ``prefix``."""
    tail = prefix[-3:]
    return FIRST_ID + (
        sum((i + 1) * t for i, t in enumerate(tail)) + 7 * len(prefix)
    ) % (VOCAB - FIRST_ID)


def _greedy(prompt: list[int], count: int) -> list[int]:
    """The target's first ``count`` tokens after ``prompt``."""
    tokens = list(prompt)
    for _ in range(count):
        tokens.append(_target(tokens))
    return tokens[len(prompt) :]


def _drafts(prefix: list[int]) -> list[int]:
    """K drafts after ``prefix``: the first ``j`` are the target's own chain,
    the rest differ from it; ``j`` follows the last token and the position."""
    agree = (prefix[-1] + len(prefix)) % T
    chain = list(prefix)
    drafts = []
    for k in range(K):
        token = _target(chain)
        if k >= agree:
            token = FIRST_ID + (token - FIRST_ID + 1) % (VOCAB - FIRST_ID)
        drafts.append(token)
        chain.append(token)
    return drafts


def _prompt(seed: int, length: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(FIRST_ID, VOCAB, (length,), generator=g).tolist()


class _Target:
    """The model: a deterministic target and its own drafter.

    A prefill row's next token follows the tokens the runner hands it, so a
    row that runs past the request's last token samples after stale ids. A
    verify row's answer at block column ``t`` follows the request's committed
    tokens before the row's first position and the block's columns 0 to ``t``.
    """

    def __init__(self, harness: _Harness):
        self.h = harness
        # (request id, first position, end position, request tokens) of every
        # prefill row that ran past its request's last token.
        self.overruns: list[tuple[str, int, int, int]] = []

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return ["kv"]

    def _live(self) -> list[str]:
        batch = self.h.runner.input_batch
        return list(batch.req_ids[: batch.num_reqs])

    def prefill_forward(self, tokens, prompt_lens, start_pos, **kwargs):
        live = self._live()
        logits = torch.zeros((len(live), 1, VOCAB))
        for row, req_id in enumerate(live):
            start, end = int(start_pos[row]), int(prompt_lens[row])
            request = self.h.requests[req_id]
            if end > request.num_tokens:
                self.overruns.append((req_id, start, end, request.num_tokens))
            logits[row, 0, _target(tokens[row, :end].tolist())] = 1.0
        return logits

    def decode_forward(self, *, tokens, start_pos, **kwargs):
        if kwargs.get("spec_mode") != ACCEPT_MODE_ARGMAX_IDS:
            raise AssertionError(
                f"a decode without a verify: {kwargs.get('spec_mode')}"
            )
        ids = torch.full(tuple(tokens.shape), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        for row, req_id in enumerate(self._live()):
            p0 = int(start_pos[row, 0])
            history = list(self.h.requests[req_id].all_token_ids[:p0])
            block = tokens[row].tolist()
            for t in range(int(kwargs["num_valid_drafts"][row]) + 1):
                ids[row, t] = _target(history + block[: t + 1])
        return VerifyOutput(spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids)

    def propose_draft_tokens(
        self, num_drafts, committed, positions, counts, hidden=None
    ):
        rows = int(committed.shape[0])
        drafts = torch.zeros((rows, num_drafts), dtype=torch.int32)
        num_valid = torch.zeros(rows, dtype=torch.int32)
        for row, req_id in enumerate(self._live()):
            if int(positions[row, 0]) < 0:
                continue
            count = int(counts[row])
            history = list(self.h.requests[req_id].all_token_ids)
            history += committed[row, :count].tolist()
            drafts[row] = torch.tensor(_drafts(history)[:num_drafts])
            num_valid[row] = num_drafts
        return DraftOutput(draft_token_ids=drafts, num_valid=num_valid)


class _StubModel:
    """Stands in for the TT model class while the platform hook runs."""


@contextmanager
def _stub_model_resolution():
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
        yield


def _plan() -> SpecPlan:
    return SpecPlan(
        effective_k=K,
        lanes_per_request=1,
        extra_bytes_per_seq=0,
        extra_bytes_per_token=0,
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
        drafter_state=DRAFTER_STATE_INTERNAL,
        supports_narrow_decode=False,
    )


def _speculative_config(config) -> SpeculativeConfig:
    return SpeculativeConfig(
        method=MODEL_OWNED_DRAFT_METHOD,
        model=MODEL_OWNED_DRAFT_SENTINEL,
        num_speculative_tokens=K,
        target_model_config=config.model_config,
        target_parallel_config=config.parallel_config,
    )


def _scheduler(kv_cache_config: KVCacheConfig) -> TTScheduler:
    """A real ``TTScheduler`` over a real ``KVCacheManager``, prefix caching on."""
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        async_scheduling=False,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE, gpu_memory_utilization=0.9, cache_dtype="auto"
    )
    with _stub_model_resolution():
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )
    # The platform hook turns both off for the stub, which declares neither.
    cache_config.enable_prefix_caching = True
    config.scheduler_config.enable_chunked_prefill = True
    cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    # Attached after construction: the platform would resolve a plan from the
    # model class, and the stub declares none.
    config.speculative_config = _speculative_config(config)
    store_tt_spec_plan(config, _plan())
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


class _CacheConfig(SimpleNamespace):
    def get_resolved_kv_cache_layout(self):
        return KVCacheLayout[self.kv_cache_layout]


def _runner_config() -> SimpleNamespace:
    """The fields ``TTModelRunner`` reads."""
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            model="spec-test",
            max_model_len=MAX_MODEL_LEN,
            original_max_model_len=MAX_MODEL_LEN,
            get_sliding_window=lambda: None,
            is_encoder_decoder=False,
            is_multimodal_model=False,
            uses_mrope=False,
            hf_config=SimpleNamespace(model_type="spec-test"),
            get_vocab_size=lambda: VOCAB,
            get_num_layers_by_block_type=lambda *_args: 1,
            logits_processors=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_MODEL_LEN,
            disable_hybrid_kv_cache_manager=False,
            async_scheduling=False,
            enable_chunked_prefill=True,
        ),
        lora_config=None,
        load_config=None,
        observability_config=None,
        device_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        cache_config=_CacheConfig(
            block_size=BLOCK_SIZE,
            num_gpu_blocks_override=None,
            kv_cache_layout=KVCacheLayout.LBNHC.name,
            enable_prefix_caching=True,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        additional_config={},
        speculative_config=SimpleNamespace(
            method=MODEL_OWNED_DRAFT_METHOD,
            num_speculative_tokens=K,
            use_eagle=lambda: False,
        ),
        kv_transfer_config=None,
    )
    store_tt_spec_plan(cfg, _plan())
    return cfg


class _Harness:
    """Runs requests through ``EngineCore.step``'s synchronous order."""

    def __init__(self, num_blocks: int):
        self.kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["model.layers.0.self_attn"],
                    FullAttentionSpec(
                        block_size=BLOCK_SIZE,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        )
        self.scheduler = _scheduler(self.kv_cache_config)
        self.device = _Target(self)
        self.runner = TTModelRunner(
            vllm_config=_runner_config(),
            mesh_device=None,
            trace_mode="none",
            enable_model_warmup=False,
            num_devices=1,
        )
        self.runner.model = self.device
        self.runner.initialize_kv_cache(self.kv_cache_config)
        self.requests: dict[str, Request] = {}
        self.prompts: dict[str, list[int]] = {}
        self.max_tokens: dict[str, int] = {}
        self.preempted: set[str] = set()
        # (request id, how it was admitted) of every admission with exactly
        # one uncomputed token after a prefix-cache hit.
        self.one_token_admissions: list[tuple[str, str]] = []

    def add(self, req_id: str, prompt: list[int], max_tokens: int) -> None:
        init_none_hash(sha256)
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0.0)
        params.update_from_generation_config({}, eos_token_id=EOS)
        request = Request(
            request_id=req_id,
            prompt_token_ids=list(prompt),
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        )
        self.requests[req_id] = request
        self.prompts[req_id] = list(prompt)
        self.max_tokens[req_id] = max_tokens
        self.scheduler.add_request(request)

    def _record_admissions(self, out) -> None:
        cached = out.scheduled_cached_reqs
        admitted = [
            ("new", new.req_id, new.num_computed_tokens)
            for new in out.scheduled_new_reqs
        ]
        admitted += [
            ("resumed", req_id, computed)
            for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens)
            if req_id in cached.resumed_req_ids
        ]
        for kind, req_id, computed in admitted:
            if computed and computed + 1 == self.requests[req_id].num_tokens:
                self.one_token_admissions.append((req_id, kind))

    def run(self, max_steps: int = 2000) -> None:
        for _ in range(max_steps):
            if not self.scheduler.has_requests():
                return
            out = self.scheduler.schedule()
            self.preempted |= set(out.preempted_req_ids or ())
            self._record_admissions(out)
            output = self.runner.execute_model(out)
            if output is None:
                output = self.runner.sample_tokens(None)
            assert isinstance(output, ModelRunnerOutput), type(output)
            self.scheduler.update_from_output(out, output)
            drafts = self.runner.take_draft_token_ids()
            if drafts is not None:
                self.scheduler.update_draft_token_ids(drafts)
        raise AssertionError("scenario did not finish")

    def check_outputs(self) -> None:
        for req_id, request in self.requests.items():
            expected = _greedy(self.prompts[req_id], self.max_tokens[req_id])
            assert list(request.output_token_ids) == expected, req_id


def test_a_prefix_hit_that_leaves_one_token_prefills_that_token():
    """Request b repeats request a's prompt of four blocks and one token, so
    its prefix-cache hit leaves only the last token. It prefills that token
    and takes its first token from it."""
    h = _Harness(num_blocks=64)
    prompt = _prompt(1, 4 * BLOCK_SIZE + 1)
    h.add("a", prompt, max_tokens=12)
    h.run()
    h.add("b", prompt, max_tokens=12)
    h.run()

    assert ("b", "new") in h.one_token_admissions
    assert h.device.overruns == []
    h.check_outputs()


def test_a_resume_that_leaves_one_token_prefills_that_token():
    """Two requests decode in a pool too small for both, so vLLM preempts
    request b. Its prompt and output end one token past a block boundary when
    it resumes, so its prefix-cache hit on its own blocks leaves one token."""
    h = _Harness(num_blocks=12)
    h.add("a", _prompt(1, 41), max_tokens=40)
    h.add("b", _prompt(2, 47), max_tokens=40)
    h.run()

    assert h.preempted == {"b"}
    assert ("b", "resumed") in h.one_token_admissions
    assert h.device.overruns == []
    h.check_outputs()
