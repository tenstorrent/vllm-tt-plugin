# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefix-cache hit confirmation: the model trims vLLM's hit before it is committed.

``TTWorker.load_model`` publishes the loaded model's ``confirm_prefix_cache_hit``;
a ``TTScheduler`` constructed afterwards routes its KV cache manager's hit lookup
through it, floors the answer to the scheduler's block size and cuts the computed
blocks to it; the platform validates the declaration at config time.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

import vllm_tt_plugin.scheduler as tt_scheduler
from vllm_tt_plugin.config import validate_tt_prefix_hit_confirmation
from vllm_tt_plugin.scheduler import TTScheduler
from vllm_tt_plugin.worker import TTWorker

BLOCK_SIZE = 32


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
    s._prefix_hit_confirmer = confirm
    s._orig_get_computed_blocks = lambda request: (blocks, num_new, 64)
    return s, blocks


def _request(n_tokens=200, mm=()):
    return SimpleNamespace(
        request_id="req", prompt_token_ids=list(range(n_tokens)), mm_features=list(mm)
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


def test_scheduler_keeps_the_hit_the_model_honours_in_full():
    s, blocks = _scheduler(128, lambda ids, mm, n: n)
    assert s._confirmed_computed_blocks(_request()) == (blocks, 128, 64)
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
    assert s._prefix_hit_confirmer is None
    assert manager.get_computed_blocks == original

    confirm = lambda ids, mm, n: n  # noqa: E731
    monkeypatch.setattr(tt_scheduler, "_published_confirmer", confirm)
    s = TTScheduler.__new__(TTScheduler)
    s.kv_cache_manager = manager
    s._take_published_prefix_hit_confirmer()
    assert s._prefix_hit_confirmer is confirm
    assert manager.get_computed_blocks == s._confirmed_computed_blocks
    assert s._orig_get_computed_blocks == original


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


def _config(backend=None):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(distributed_executor_backend=backend)
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
