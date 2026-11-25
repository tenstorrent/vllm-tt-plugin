# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefix-cache hit confirmation: the model trims vLLM's hit before it is committed.

``TTWorker.load_model`` publishes the loaded model's ``confirm_prefix_cache_hit``;
a ``TTScheduler`` constructed afterwards routes its KV cache manager's hit lookup
through it, floors the answer to the scheduler's block size and cuts the computed
blocks to it; the platform validates the declaration at config time.
"""

import gc
import weakref
from contextlib import contextmanager
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
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

import vllm_tt_plugin.scheduler as tt_scheduler
from vllm_tt_plugin.config import (
    store_tt_lane_count,
    store_tt_spec_plan,
    validate_tt_prefix_hit_confirmation,
)
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
from vllm_tt_plugin.worker import TTWorker

BLOCK_SIZE = 32
MAX_MODEL_LEN = 512
NUM_SPEC_TOKENS = 3
LOCAL_MODEL_CONFIG = Path(__file__).parent.parent / "model_configs" / "qwen2"


class _StubModel:
    """Stands in for the TT model class while the platform hook runs."""


@contextmanager
def _stub_model_resolution():
    """Resolve ``_StubModel`` instead of a tt-metal class in ``VllmConfig``."""
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


def _manager(num_blocks=16):
    config = KVCacheConfig(
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
    return KVCacheManager(
        config,
        max_model_len=1024,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
    )


def _scheduler(num_new, confirm, hit_blocks=4):
    """A scheduler whose manager's original lookup answers a ``num_new``-token
    hit over the pool's first ``hit_blocks`` blocks and a 64-token boundary."""
    manager = _manager()
    blocks = manager.create_kv_cache_blocks(
        (manager.block_pool.blocks[1 : 1 + hit_blocks],)
    )
    s = TTScheduler.__new__(TTScheduler)
    s.kv_cache_manager = manager
    s.block_size = BLOCK_SIZE
    s._prefix_hit_confirmer_ref = lambda: confirm
    s.lookup_bounds = []

    def lookup(request, max_cache_hit_length=None):
        s.lookup_bounds.append(max_cache_hit_length)
        return blocks, num_new, 64

    s._orig_get_computed_blocks = lookup
    return s, blocks


def _request(n_tokens=200, mm=(), output=()):
    prompt = list(range(n_tokens))
    return SimpleNamespace(
        request_id="req",
        prompt_token_ids=prompt,
        all_token_ids=prompt + list(output),
        mm_features=list(mm),
    )


def test_scheduler_trims_the_hit_to_the_confirmed_block_multiple():
    seen = {}

    def confirm(ids, mm_items, num_new):
        seen["args"] = (len(ids), mm_items, num_new)
        return 70  # the model resumes at 70: floored to 64 = 2 blocks

    item = SimpleNamespace(
        identifier="img-1",
        mm_position=SimpleNamespace(offset=5, length=8),
        modality="image",
    )
    s, blocks = _scheduler(128, confirm)
    out_blocks, honoured, boundary = s._confirmed_computed_blocks(_request(mm=[item]))
    assert seen["args"] == (200, [("img-1", 5, 8, "image")], 128)
    assert honoured == 64 and boundary == 0
    assert out_blocks.blocks == (blocks.blocks[0][:2],)


def test_the_confirmer_sees_every_token_the_hit_was_hashed_over():
    # vLLM's hit reaches num_tokens - 1, so for a request resumed from
    # preemption it covers output tokens too: the model is asked about the
    # whole sequence, handed over as the request's own view, not a copy.
    seen = {}

    def confirm(ids, mm_items, num_new):
        seen["ids"] = ids
        return num_new

    s, blocks = _scheduler(224, confirm, hit_blocks=7)
    request = _request(200, output=range(1000, 1030))
    assert s._confirmed_computed_blocks(request) == (blocks, 224, 64)
    assert seen["ids"] is request.all_token_ids
    assert len(seen["ids"]) == 230


def test_scheduler_keeps_the_hit_the_model_honours_in_full():
    s, blocks = _scheduler(128, lambda ids, mm, n: n)
    assert s._confirmed_computed_blocks(_request()) == (blocks, 128, 64)
    # vLLM's bound on the hit (max_cache_hit_length) goes to the lookup unchanged
    assert s._confirmed_computed_blocks(_request(), 192) == (blocks, 128, 64)
    assert s.lookup_bounds == [None, 192]
    # an answer above the proposal serves the proposal: vLLM's hit is the most
    # the request can start from
    s, blocks = _scheduler(128, lambda ids, mm, n: 10_000)
    assert s._confirmed_computed_blocks(_request()) == (blocks, 128, 64)


def test_scheduler_passes_a_miss_through_without_asking_the_model():
    calls = []
    s, blocks = _scheduler(0, lambda ids, mm, n: calls.append(n) or 0)
    assert s._confirmed_computed_blocks(_request()) == (blocks, 0, 64)
    assert calls == []


def test_scheduler_drops_a_hit_the_model_refuses_and_rejects_a_negative_answer():
    s, blocks = _scheduler(128, lambda ids, mm, n: 0)
    out_blocks, honoured, boundary = s._confirmed_computed_blocks(_request())
    assert (honoured, boundary) == (0, 0)
    assert out_blocks is s.kv_cache_manager.empty_kv_cache_blocks
    s, blocks = _scheduler(128, lambda ids, mm, n: -5)
    with pytest.raises(ValueError, match="answered -5 cached tokens for the 128-token"):
        s._confirmed_computed_blocks(_request())


def test_a_scheduler_takes_the_published_confirmer_at_construction(monkeypatch):
    # the engine core builds its scheduler after the executor loaded the model,
    # so the worker's publication is there first; without one the manager's
    # lookup is left alone
    manager = _manager()
    original = manager.get_computed_blocks
    s = TTScheduler.__new__(TTScheduler)
    s.kv_cache_manager = manager
    s._take_published_prefix_hit_confirmer()
    assert s._prefix_hit_confirmer_ref is None
    assert manager.get_computed_blocks == original

    confirm = lambda ids, mm, n: n  # noqa: E731
    monkeypatch.setattr(tt_scheduler, "_published_confirmer", confirm)
    s = TTScheduler.__new__(TTScheduler)
    s.kv_cache_manager = manager
    s._take_published_prefix_hit_confirmer()
    assert s._prefix_hit_confirmer_ref() is confirm
    assert manager.get_computed_blocks == s._confirmed_computed_blocks
    assert s._orig_get_computed_blocks == original


def test_the_confirmer_does_not_keep_the_model_alive():
    # EngineCore shuts the executor down before the scheduler, so neither the
    # scheduler nor the published global may hold the model past
    # TTWorker.shutdown, which closes the mesh the model's tensors live on.
    # Assigned directly rather than monkeypatched: monkeypatch would keep a
    # reference to the replaced value (conftest restores the global).
    model = _Confirming()
    tt_scheduler._published_confirmer = model.confirm_prefix_cache_hit
    s, _ = _scheduler(128, None)
    s._take_published_prefix_hit_confirmer()
    assert s._prefix_hit_confirmer_ref() == model.confirm_prefix_cache_hit

    TTWorker.shutdown(SimpleNamespace(model_runner=None, mesh_device=None))
    assert tt_scheduler._published_confirmer is None
    alive = weakref.ref(model)
    del model
    gc.collect()
    assert alive() is None
    assert s._prefix_hit_confirmer_ref() is None
    s._orig_get_computed_blocks = lambda request, max_cache_hit_length=None: (
        None,
        128,
        64,
    )
    with pytest.raises(RuntimeError, match="was released"):
        s._confirmed_computed_blocks(_request())


class _Confirming:
    model_capabilities = {
        "supports_prefix_caching": True,
        "confirms_prefix_cache_hits": True,
    }

    def confirm_prefix_cache_hit(self, ids, mm_items, n):
        return n


class _Plain:
    model_capabilities = {"supports_prefix_caching": True}

    def confirm_prefix_cache_hit(self, ids, mm_items, n):
        return n


def _worker(model, loaded):
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(model=model, load_model=lambda: loaded.append(1))
    )
    worker._prefix_hit_confirmer = lambda: TTWorker._prefix_hit_confirmer(worker)
    return worker


def test_worker_publishes_the_confirmer_at_load_model_only_when_declared(
    monkeypatch,
):
    monkeypatch.setattr(tt_scheduler, "_published_confirmer", None)
    loaded = []
    model = _Confirming()
    TTWorker.load_model(_worker(model, loaded))
    assert loaded == [1]
    assert tt_scheduler._published_confirmer == model.confirm_prefix_cache_hit
    assert tt_scheduler._published_confirmer([1, 2], [], 32) == 32

    TTWorker.load_model(_worker(_Plain(), loaded))
    assert tt_scheduler._published_confirmer is None
    TTWorker.load_model(_worker(object(), loaded))  # no capabilities at all
    assert tt_scheduler._published_confirmer is None
    assert loaded == [1, 1, 1]

    # the loaded model's declaration is checked as the platform checks the class
    class NotABool(_Confirming):
        model_capabilities = {
            **_Confirming.model_capabilities,
            "confirms_prefix_cache_hits": "yes",
        }

    with pytest.raises(ValueError, match="as a boolean, got 'yes'"):
        TTWorker.load_model(_worker(NotABool(), loaded))
    assert tt_scheduler._published_confirmer is None


def _config(backend=None):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            distributed_executor_backend=backend, data_parallel_size=1
        ),
        additional_config={},
        kv_transfer_config=None,
    )


def test_platform_validates_the_declaration():
    validate_tt_prefix_hit_confirmation(
        _config(), _Confirming.model_capabilities, _Confirming
    )
    validate_tt_prefix_hit_confirmation(
        _config("uni"), _Confirming.model_capabilities, _Confirming
    )
    # not declared: nothing is checked
    validate_tt_prefix_hit_confirmation(_config("mp"), {}, object)
    validate_tt_prefix_hit_confirmation(_config("mp"), None, object)

    # a non-boolean declaration is refused, as prefill_next_token_ids' is
    with pytest.raises(
        ValueError, match=r"confirms_prefix_cache_hits'\] as a boolean, got 1"
    ):
        validate_tt_prefix_hit_confirmation(
            _config(),
            {**_Confirming.model_capabilities, "confirms_prefix_cache_hits": 1},
            _Confirming,
        )
    with pytest.raises(ValueError, match="without supports_prefix_caching=True"):
        validate_tt_prefix_hit_confirmation(
            _config(), {"confirms_prefix_cache_hits": True}, _Confirming
        )

    class NoMethod:
        model_capabilities = _Confirming.model_capabilities

    with pytest.raises(ValueError, match="without a confirm_prefix_cache_hit"):
        validate_tt_prefix_hit_confirmation(
            _config(), NoMethod.model_capabilities, NoMethod
        )
    # the worker that loads the model must share the scheduler's process
    with pytest.raises(ValueError, match="distributed_executor_backend='mp'"):
        validate_tt_prefix_hit_confirmation(
            _config("mp"), _Confirming.model_capabilities, _Confirming
        )
    # a KV connector adds its external tokens on top of the confirmed hit
    connector = _config("uni")
    connector.kv_transfer_config = object()
    with pytest.raises(ValueError, match="a KV connector"):
        validate_tt_prefix_hit_confirmation(
            connector, _Confirming.model_capabilities, _Confirming
        )
    # a model that does not confirm hits keeps its connector
    validate_tt_prefix_hit_confirmation(connector, {}, object)
    # lane-DP: every lane's scheduler would ask the one lane-blind confirmer
    lanes = _config("uni")
    store_tt_lane_count(lanes, 4)
    with pytest.raises(ValueError, match="single-process lane-DP runs 4 TT lanes"):
        validate_tt_prefix_hit_confirmation(
            lanes, _Confirming.model_capabilities, _Confirming
        )
    # a lane-DP launch of a model that does not confirm hits is not affected
    validate_tt_prefix_hit_confirmation(lanes, {}, object)


class _AnswersWith:
    """A model whose confirmer answers a fixed count and records the call."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def confirm_prefix_cache_hit(self, ids, mm_items, num_new):
        self.calls.append((list(ids), list(mm_items), num_new))
        return num_new if self.answer is None else self.answer


def _real_scheduler(speculative: bool = False) -> TTScheduler:
    """A real TTScheduler over a real prefix-caching KVCacheManager, built
    after the confirmer was published, as the engine core builds it after
    TTWorker.load_model. ``speculative`` adds a model-owned drafter plan
    with ``NUM_SPEC_TOKENS`` drafts."""
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=2,
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
        enable_prefix_caching=True,
    )
    with _stub_model_resolution():
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )
    # The stub model declares no supports_prefix_caching, so the platform hook
    # turned it off; the scheduler under test is a prefix-caching one.
    config.cache_config.enable_prefix_caching = True
    num_blocks = 2 * MAX_MODEL_LEN // BLOCK_SIZE + 1
    config.cache_config.num_gpu_blocks = num_blocks
    if speculative:
        # Attached after construction: the stub model declares no plan.
        config.speculative_config = SpeculativeConfig(
            method=MODEL_OWNED_DRAFT_METHOD,
            model=MODEL_OWNED_DRAFT_SENTINEL,
            num_speculative_tokens=NUM_SPEC_TOKENS,
            target_model_config=config.model_config,
            target_parallel_config=config.parallel_config,
        )
        store_tt_spec_plan(
            config,
            SpecPlan(
                effective_k=NUM_SPEC_TOKENS,
                lanes_per_request=1,
                extra_bytes_per_seq=0,
                extra_bytes_per_token=0,
                accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
                drafter_state=DRAFTER_STATE_INTERNAL,
                supports_narrow_decode=False,
            ),
        )
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
        log_stats=False,
        structured_output_manager=StructuredOutputManager(config),
    )


def _real_request(request_id: str, n_tokens: int, max_tokens: int = 1) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    sampling_params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(n_tokens)),
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


@pytest.mark.parametrize(
    "answer, served",
    [
        (100, 96),  # floored to 3 blocks: the request starts at 96
        (None, 192),  # the whole proposal: vLLM's 6-block hit stands
        (0, 0),  # refused: the request prefills from scratch
    ],
)
def test_schedule_starts_a_request_at_the_hit_the_model_confirms(answer, served):
    model = _AnswersWith(answer)
    tt_scheduler._published_confirmer = model.confirm_prefix_cache_hit
    scheduler = _real_scheduler()

    # A 200-token prompt computed once leaves 6 full blocks in the cache.
    first = _real_request("first", 200)
    scheduler.add_request(first)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"first": 200}
    assert model.calls == []  # nothing cached yet: nothing to confirm
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=["first"],
            req_id_to_index={"first": 0},
            sampled_token_ids=[[7]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert first.status == RequestStatus.FINISHED_LENGTH_CAPPED

    # The same prompt again: vLLM proposes 192 cached tokens (its hit stops
    # short of the last token), and the model's answer is what is scheduled.
    second = _real_request("second", 200)
    scheduler.add_request(second)
    step = scheduler.schedule()
    assert model.calls == [(list(range(200)), [], 192)]
    (new,) = step.scheduled_new_reqs
    assert new.req_id == "second"
    assert new.num_computed_tokens == served
    assert step.num_scheduled_tokens == {"second": 200 - served}


def _commit(scheduler: TTScheduler, step, token: int) -> None:
    """Commit one sampled ``token`` for every request ``step`` scheduled."""
    req_ids = list(step.num_scheduled_tokens)
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[token] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


@pytest.mark.parametrize(
    "answer, served, scheduled",
    [
        # Honoured in full, the hit leaves one token: vLLM pads the admission
        # with NUM_SPEC_TOKENS placeholder drafts, which TTScheduler drops.
        pytest.param(None, 192, 1, id="one-token-left"),
        # Trimmed to 3 blocks: the admission is a plain prefill, never padded.
        pytest.param(100, 96, 97, id="trimmed"),
    ],
)
def test_a_confirmed_hit_under_speculation_admits_only_uncomputed_tokens(
    answer, served, scheduled
):
    # vLLM decides the one-token padding after the hit lookup, so it pads on
    # the hit the model confirmed, and the commit of the admission step then
    # advances the request by its one sampled token, with no drafts to settle.
    model = _AnswersWith(answer)
    tt_scheduler._published_confirmer = model.confirm_prefix_cache_hit
    scheduler = _real_scheduler(speculative=True)

    first = _real_request("first", 192)
    scheduler.add_request(first)
    _commit(scheduler, scheduler.schedule(), 7)

    second = _real_request("second", 193, max_tokens=2)
    scheduler.add_request(second)
    step = scheduler.schedule()
    assert model.calls[-1][2] == 192
    assert step.scheduled_new_reqs[0].num_computed_tokens == served
    assert step.num_scheduled_tokens == {"second": scheduled}
    assert step.scheduled_spec_decode_tokens == {}
    _commit(scheduler, step, 9)
    assert second.num_computed_tokens == 193
    assert second.num_output_placeholders == 0
    assert list(second.output_token_ids) == [9]
