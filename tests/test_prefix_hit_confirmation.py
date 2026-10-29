"""Prefix-cache hit confirmation: the model trims vLLM's hit before it is committed.

``mm_identifiers`` ride next to the placeholder fields;
``TTWorker.prefix_cache_hit_confirmer`` is the ``collective_rpc`` target;
``TTScheduler._confirmed_computed_blocks`` floors the model's answer to a block
multiple and cuts the computed blocks to it.
"""

from types import SimpleNamespace

import torch

from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler
from vllm_tt_plugin.worker import TTWorker


def _field(t):
    return SimpleNamespace(data=t)


def _runner(requests_by_id, req_ids):
    return SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=req_ids, num_reqs=len(req_ids)),
        requests=requests_by_id,
        _validate_mm_feature=lambda mm_feature: TTModelRunner._validate_mm_feature(
            None, mm_feature
        ),
        _MM_BASE_FIELDS=TTModelRunner._MM_BASE_FIELDS,
        _MM_PLACEHOLDER_FIELDS=TTModelRunner._MM_PLACEHOLDER_FIELDS,
    )


def test_gather_forwards_the_item_identifier_even_without_a_payload():
    pv = torch.zeros(4, 3)
    cached = SimpleNamespace(  # payload stripped by vLLM: prefix-cached item
        modality="image",
        data=None,
        identifier="img-aaa",
        mm_position=SimpleNamespace(offset=3, length=4),
    )
    fresh = SimpleNamespace(
        modality="image",
        data={"pixel_values": _field(pv)},
        identifier="img-bbb",
        mm_position=SimpleNamespace(offset=10, length=4),
    )
    runner = _runner(
        {
            "text": SimpleNamespace(mm_features=[]),
            "mm": SimpleNamespace(mm_features=[cached, fresh]),
        },
        ["text", "mm"],
    )
    out = TTModelRunner._gather_multi_modal_inputs(runner)
    assert list(out)[-3:] == [
        "mm_placeholder_offsets",
        "mm_placeholder_lengths",
        "mm_identifiers",
    ]
    assert out["mm_identifiers"] == [None, ["img-aaa", "img-bbb"]]
    assert out["mm_placeholder_offsets"] == [None, [3, 10]]
    assert out["pixel_values"] == [None, [None, pv]]


class _Blocks:
    def __init__(self, blocks):
        self.blocks = blocks


class _Manager:
    block_size = 32

    def __init__(self, groups):
        self._groups = groups
        self.created = []

    def get_computed_blocks(
        self, request
    ):  # replaced by the hook in __init__; kept for the fake
        raise AssertionError("the hook calls the saved original, not the manager")

    def create_kv_cache_blocks(self, blocks):
        self.created.append(blocks)
        return _Blocks(blocks)


def _scheduler(num_new, confirm, groups=2, blocks_per_group=4):
    mgr = _Manager(groups)
    blocks = _Blocks(
        tuple([f"g{g}b{b}" for b in range(blocks_per_group)] for g in range(groups))
    )
    fake = SimpleNamespace(
        kv_cache_manager=mgr,
        _prefix_hit_confirmer=confirm,
        _orig_get_computed_blocks=lambda request: (blocks, num_new, 64),
    )
    return fake, mgr, blocks


def _request(n_tokens=200, mm=()):
    return SimpleNamespace(prompt_token_ids=list(range(n_tokens)), mm_features=list(mm))


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
    fake, mgr, blocks = _scheduler(128, confirm)
    out_blocks, honoured, boundary = TTScheduler._confirmed_computed_blocks(
        fake, _request(mm=[item])
    )
    assert seen["args"] == (200, [("img-1", 5, 8, "image")], 128)
    assert honoured == 64 and boundary == 0
    assert out_blocks.blocks == (["g0b0", "g0b1"], ["g1b0", "g1b1"])
    assert mgr.created == [(["g0b0", "g0b1"], ["g1b0", "g1b1"])]


def test_scheduler_keeps_the_hit_the_model_honours_in_full():
    fake, mgr, blocks = _scheduler(128, lambda ids, mm, n: n)
    out = TTScheduler._confirmed_computed_blocks(fake, _request())
    assert out == (blocks, 128, 64) and mgr.created == []
    # a scheduler without an installed confirmer takes the one the in-process
    # worker published at load_model (the engine core builds its scheduler
    # after the model loaded, so the published one is there first)
    from vllm_tt_plugin.scheduler import publish_prefix_hit_confirmer

    publish_prefix_hit_confirmer(lambda ids, mm, n: n // 2)
    try:
        fake, mgr, blocks = _scheduler(128, None)
        out = TTScheduler._confirmed_computed_blocks(fake, _request())
        assert out[1] == 64 and fake._prefix_hit_confirmer is not None
    finally:
        publish_prefix_hit_confirmer(None)
    fake, mgr, blocks = _scheduler(128, None)
    assert TTScheduler._confirmed_computed_blocks(fake, _request()) == (blocks, 128, 64)


def test_scheduler_passes_through_without_a_confirmer_or_a_hit():
    fake, mgr, blocks = _scheduler(128, None)
    assert TTScheduler._confirmed_computed_blocks(fake, _request()) == (blocks, 128, 64)
    calls = []
    fake, mgr, blocks = _scheduler(0, lambda ids, mm, n: calls.append(n) or 0)
    assert TTScheduler._confirmed_computed_blocks(fake, _request()) == (blocks, 0, 64)
    assert calls == []


def test_scheduler_clamps_an_over_eager_or_negative_answer():
    fake, mgr, blocks = _scheduler(128, lambda ids, mm, n: 10_000)
    assert TTScheduler._confirmed_computed_blocks(fake, _request())[1] == 128
    fake, mgr, blocks = _scheduler(128, lambda ids, mm, n: -5)
    out_blocks, honoured, boundary = TTScheduler._confirmed_computed_blocks(
        fake, _request()
    )
    assert (honoured, boundary) == (0, 0) and out_blocks.blocks == ([], [])


def test_worker_exposes_the_confirmer_only_when_declared():
    class Confirming:
        model_capabilities = {
            "supports_prefix_caching": True,
            "confirms_prefix_cache_hits": True,
        }

        def confirm_prefix_cache_hit(self, ids, mm_items, n):
            return n

    class Plain:
        model_capabilities = {"supports_prefix_caching": True}

        def confirm_prefix_cache_hit(self, ids, mm_items, n):
            return n

    worker = SimpleNamespace(model_runner=SimpleNamespace(model=Confirming()))
    fn = TTWorker.prefix_cache_hit_confirmer(worker)
    assert fn is not None and fn([1, 2], [], 32) == 32
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=Plain()))
    assert TTWorker.prefix_cache_hit_confirmer(worker) is None
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=None))
    assert TTWorker.prefix_cache_hit_confirmer(worker) is None
    # the engine-core wrapper (installed from the platform's load hooks): after
    # the original __init__ it hands the worker's confirmer to the scheduler,
    # only when the engine's prefix cache is on
    import vllm.v1.engine.core as engine_core

    from vllm_tt_plugin.platform import _install_prefix_hit_confirmer_patch

    # the wrapper closes over the constructor it found at install time: a
    # stand-in replaces upstream's here (the conftest fixture restores the real
    # one and the install marker after the test)
    engine_core.__dict__.pop("_tt_original_engine_core_init", None)
    engine_core.EngineCore.__init__ = lambda self, *a, **k: None
    _install_prefix_hit_confirmer_patch()
    _install_prefix_hit_confirmer_patch()  # idempotent
    assert (
        engine_core.EngineCore.__init__ is not engine_core._tt_original_engine_core_init
    )

    class FakeEngine:
        def __init__(self, enable_prefix_caching, confirmer):
            self.vllm_config = SimpleNamespace(
                cache_config=SimpleNamespace(
                    enable_prefix_caching=enable_prefix_caching
                )
            )
            self.scheduler = SimpleNamespace(installed=[])
            self.scheduler.set_prefix_hit_confirmer = self.scheduler.installed.append
            self.model_executor = SimpleNamespace(
                collective_rpc=lambda name: [confirmer]
            )
            engine_core.EngineCore.__init__(self)

    fn = lambda ids, mm_items, n: n  # noqa: E731
    assert FakeEngine(True, fn).scheduler.installed == [fn]
    assert FakeEngine(True, None).scheduler.installed == []
    assert FakeEngine(False, fn).scheduler.installed == []
