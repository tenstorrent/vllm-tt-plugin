# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The KV allocation covers the rows a model-owned drafter writes next.

A model-owned drafter is selected with vLLM's ``custom_class`` method and the
``vllm_tt_plugin.model_owned_drafter`` sentinel, and it proposes inside the
step that verifies: once the accept walk commits, the model's fused body runs
over the anchor plus K drafts and writes K/V for those K+1 positions. All K+1
sit past the last position the step itself committed, so the step's own
allocation does not cover them. ``Scheduler.__init__`` raises
``num_lookahead_tokens`` only for the drafter methods upstream recognises, and
``custom_class`` is not among them, which leaves a row that crosses into the
next block with no block to land in.

``TTScheduler.__init__`` therefore raises ``num_lookahead_tokens`` to
``spec_lookahead_tokens(plan, K, method)``, and ``KVCacheManager.allocate_slots``
receives it on every call the scheduler makes.

``tests/spec/test_spec_scheduler.py`` pins the number that helper returns.
These tests build a real ``TTScheduler`` over a real ``KVCacheManager`` and
read the blocks the manager holds for the request, so a reservation the helper
computes but the allocator never receives fails here.
"""

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
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.config import store_tt_spec_plan
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

BLOCK_SIZE = 64
MAX_MODEL_LEN = 256
NUM_SPEC_TOKENS = 5
# What one proposal writes past the last committed position: the anchor plus
# every draft.
WRITE_EXTENT = NUM_SPEC_TOKENS + 1
# Ids far from the prompt's filler token, so a draft cannot be mistaken for a
# prompt position in a failure.
DRAFT_IDS = [900 + offset for offset in range(NUM_SPEC_TOKENS)]
LOCAL_MODEL_CONFIG = Path(__file__).parent.parent / "model_configs" / "qwen2"


class _StubModel:
    """Stands in for a resolved TT model class while the platform hook runs.

    Speculative admission is not the subject here, so the plan reaches the
    config through ``store_tt_spec_plan`` and this class declares nothing."""


@contextmanager
def _stub_model_resolution():
    """Resolve the stub instead of tt-metal's TTQwen2ForCausalLM while the
    platform hook runs inside VllmConfig.__post_init__ (mirrors
    test_block_scheduler._stub_model_resolution)."""
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


def _plan() -> SpecPlan:
    return SpecPlan(
        effective_k=NUM_SPEC_TOKENS,
        lanes_per_request=1,
        extra_bytes_per_seq=0,
        extra_bytes_per_token=0,
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
        drafter_state=DRAFTER_STATE_INTERNAL,
        supports_narrow_decode=False,
    )


def _scheduler(*, admitted: bool) -> TTScheduler:
    """A real TTScheduler over a real KVCacheManager.

    ``admitted`` selects only whether a plan is published. The speculative
    flags are identical either way, so a difference in what the allocator
    reserves is attributable to the plan and to nothing else.
    """
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=False,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        # Prefix caching would let a second request reuse blocks and make the
        # block count depend on cache state rather than on the reservation.
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
    # Attached after construction: the platform hook resolves a plan of its own
    # from the model class, and this stub declares no speculative capability.
    config.speculative_config = SpeculativeConfig(
        method=MODEL_OWNED_DRAFT_METHOD,
        model=MODEL_OWNED_DRAFT_SENTINEL,
        num_speculative_tokens=NUM_SPEC_TOKENS,
        target_model_config=config.model_config,
        target_parallel_config=config.parallel_config,
    )
    if admitted:
        store_tt_spec_plan(config, _plan())
    num_blocks = MAX_MODEL_LEN // BLOCK_SIZE + 2
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


def _request(prompt_len: int) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=64, ignore_eos=True)
    sampling_params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id="req-0",
        prompt_token_ids=[1] * prompt_len,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _prefilled(scheduler: TTScheduler, prompt_len: int) -> Request:
    """Drive one prefill step so the request reaches decode with
    ``num_computed_tokens == prompt_len`` and one sampled token appended."""
    request = _request(prompt_len)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {request.request_id: prompt_len}
    scheduler.update_from_output(
        prefill,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[7]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == prompt_len
    return request


def _allocated_blocks(scheduler: TTScheduler, request: Request) -> list[int]:
    """The blocks the manager holds for the request, in its single group."""
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    return block_ids


def _blocks_for(num_positions: int) -> int:
    return -(-num_positions // BLOCK_SIZE)


def test_the_reserved_lookahead_is_the_anchor_and_every_draft():
    assert _scheduler(admitted=True).num_lookahead_tokens == WRITE_EXTENT
    # Upstream leaves custom_class at zero, so this is what the allocator gets
    # for the same speculative flags with no plan published.
    assert _scheduler(admitted=False).num_lookahead_tokens == 0


def test_a_draftless_step_reserves_the_block_its_first_proposal_crosses_into():
    """The step after a prefill carries no drafts, and the proposal it makes
    still writes K+1 rows past the token it commits."""
    scheduler = _scheduler(admitted=True)
    # c = 63: the one token this step commits takes position 63, the last
    # position of block 0. The proposal that follows writes positions 64
    # through 69, so block 1 has to be allocated before the model runs.
    computed = 63
    request = _prefilled(scheduler, computed)
    # The reservation reaches every allocate_slots call, the prefill admission
    # included, so block 1 is already held here.
    assert len(_allocated_blocks(scheduler, request)) == _blocks_for(
        computed + WRITE_EXTENT
    )

    decode = scheduler.schedule()

    # One token of its own, no drafts.
    assert decode.num_scheduled_tokens == {request.request_id: 1}
    assert not decode.scheduled_spec_decode_tokens
    # 63 committed + 1 sampled + 6 written by the proposal = 70 positions.
    assert len(_allocated_blocks(scheduler, request)) == _blocks_for(
        computed + 1 + WRITE_EXTENT
    )
    assert _blocks_for(computed + 1 + WRITE_EXTENT) == 2


def test_a_fully_accepted_block_reserves_past_the_boundary_it_lands_on():
    """Every draft committing puts the last committed position on a block
    boundary, and the next proposal writes K+1 rows past that."""
    scheduler = _scheduler(admitted=True)
    # c = 58: this step computes its own sampled token at position 58 and the
    # K=5 drafts at 59 through 63, so full acceptance commits position 63, the
    # last position of block 0. The proposal that follows writes 64 through
    # 69, which is block 1.
    computed = 58
    request = _prefilled(scheduler, computed)
    assert _allocated_blocks(scheduler, request) == [1]
    request.spec_token_ids = list(DRAFT_IDS)

    decode = scheduler.schedule()

    assert decode.num_scheduled_tokens == {request.request_id: 1 + NUM_SPEC_TOKENS}
    assert decode.scheduled_spec_decode_tokens[request.request_id] == DRAFT_IDS
    blocks = _allocated_blocks(scheduler, request)
    # 58 computed + 1 sampled + 5 drafts + 6 written by the proposal = 70.
    assert len(blocks) == _blocks_for(computed + 1 + NUM_SPEC_TOKENS + WRITE_EXTENT)
    assert len(blocks) == 2
    # The last row the proposal writes is position c + K + WRITE_EXTENT = 69,
    # and the highest position the allocation holds has to reach it.
    assert len(blocks) * BLOCK_SIZE - 1 >= computed + 2 * NUM_SPEC_TOKENS + 1


def test_without_an_admitted_plan_the_crossing_block_is_absent():
    """The reservation comes from the published plan, not from the speculative
    flags: the same flags with no plan allocate only the step's own tokens."""
    scheduler = _scheduler(admitted=False)
    computed = 58
    request = _prefilled(scheduler, computed)
    request.spec_token_ids = list(DRAFT_IDS)

    decode = scheduler.schedule()

    assert decode.num_scheduled_tokens == {request.request_id: 1 + NUM_SPEC_TOKENS}
    # 58 computed + 1 sampled + 5 drafts = 64 positions, exactly block 0.
    assert len(_allocated_blocks(scheduler, request)) == _blocks_for(
        computed + 1 + NUM_SPEC_TOKENS
    )
    assert _allocated_blocks(scheduler, request) == [1]
