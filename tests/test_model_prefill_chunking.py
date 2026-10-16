# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Model-declared prefill window alignment (``prefill_chunk_alignment``) and the cap on prompts part-way
through prefill (``max_partial_prefills``): the scheduler ends every non-final piece of a split prompt on a
multiple of the window and keeps at most ``max_partial_prefills`` prompts split at a time, so a model that can
resume a prompt only at a window boundary and keeps one resumable state per prefill slot never recomputes."""

import types
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch
import vllm  # noqa: F401  (finish vLLM's own bootstrap before the plugin module)
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
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.config import (
    get_tt_max_partial_prefills,
    get_tt_prefill_chunk_alignment,
)
from vllm_tt_plugin.platform import _store_prefill_chunking
from vllm_tt_plugin.scheduler import TTScheduler

from .test_block_scheduler import LOCAL_MODEL_CONFIG

BLOCK = 128
MAX_LEN = 16384


@contextmanager
def _stub_model_resolution(model_class):
    """Resolve ``model_class`` instead of a tt-metal model while the platform hook runs inside
    VllmConfig.__post_init__ (as test_block_scheduler._stub_model_resolution)."""
    with (
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch("vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups", return_value=None),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch("vllm.model_executor.model_loader.utils.get_model_architecture", return_value=(model_class, None)),
    ):
        yield


def _scheduler(*, budget, align=0, cap=0, max_num_seqs=8, hit_policy=""):
    class _Model:
        model_capabilities = {
            "supports_chunked_prefill": True,
            "prefill_chunk_alignment": align,
            "max_partial_prefills": cap,
            **({"supports_prefix_caching": True, "prefix_cache_hit_policy": hit_policy} if hit_policy else {}),
        }

    model_config = ModelConfig(model=str(LOCAL_MODEL_CONFIG), dtype="float16", seed=42, skip_tokenizer_init=True)
    model_config.max_model_len = MAX_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=budget,
        max_model_len=MAX_LEN,
        enable_chunked_prefill=True,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto", enable_prefix_caching=bool(hit_policy)
    )
    with _stub_model_resolution(_Model):
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            # The plugin's own default alignment (128) applies to every model; the
            # "not declared" cases of these tests switch it off explicitly.
            additional_config={} if align else {"tt": {"prefill_chunk_alignment": 0}},
        )
    assert config.scheduler_config.enable_chunked_prefill
    assert get_tt_prefill_chunk_alignment(config) == align
    num_blocks = 4 * MAX_LEN // BLOCK
    cache_config.num_gpu_blocks = num_blocks
    kv = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)
            )
        ],
    )
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv,
        block_size=BLOCK,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(rid, n):
    init_none_hash(sha256)
    sp = SamplingParams(max_tokens=1, ignore_eos=True)
    sp.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=rid,
        prompt_token_ids=list(range(3, 3 + n)),
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _drain(scheduler, requests, max_steps=200):
    """Schedule until every request finished its prefill (max_tokens=1: it then finishes at once). -> per
    request [(start, end)] pieces in order, and the largest number of prompts that were part-way through
    prefill after any step."""
    for r in requests:
        scheduler.add_request(r)
    pieces = {r.request_id: [] for r in requests}
    done, worst_partial = set(), 0
    for _ in range(max_steps):
        if len(done) == len(requests):
            break
        before = {r.request_id: r.num_computed_tokens for r in requests}
        out = scheduler.schedule()
        ids = list(out.num_scheduled_tokens)
        if not ids:  # (the decode-interleave policy may spend one step on decodes: none are running here)
            out = scheduler.schedule()
            ids = list(out.num_scheduled_tokens)
        assert ids, "scheduler stalled with prefill work pending"
        sampled = []
        for rid in ids:
            r = next(x for x in requests if x.request_id == rid)
            n = out.num_scheduled_tokens[rid]
            start = r.num_computed_tokens - n  # (scheduling advances num_computed_tokens; a prefix-cache hit moves the start of the first piece)
            pieces[rid].append((start, start + n))
            fin = r.num_computed_tokens >= r.num_prompt_tokens
            sampled.append([7] if fin else [])
            if fin:
                done.add(rid)
        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={rid: i for i, rid in enumerate(ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        worst_partial = max(worst_partial, sum(1 for r in scheduler.running if r.is_prefill_chunk))
    else:
        pytest.fail("prefill did not finish")
    return pieces, worst_partial


def test_pieces_end_on_window_multiples_and_cover_the_prompt():
    s = _scheduler(budget=2000, align=512)
    pieces, _ = _drain(s, [_request("a", 5000)])
    p = pieces["a"]
    assert p == [(0, 1536), (1536, 3072), (3072, 5000)]
    assert all(end % 512 == 0 for _, end in p[:-1])


def test_without_the_capability_pieces_follow_the_budget():
    s = _scheduler(budget=2000, align=0)
    pieces, _ = _drain(s, [_request("a", 5000)])
    assert pieces["a"] == [(0, 2000), (2000, 4000), (4000, 5000)]


def test_prompt_that_fits_the_budget_is_one_piece_even_if_unaligned():
    s = _scheduler(budget=2000, align=512)
    pieces, _ = _drain(s, [_request("a", 1900)])
    assert pieces["a"] == [(0, 1900)]


def test_leftover_budget_below_a_window_is_not_spent_on_an_unresumable_piece():
    s = _scheduler(budget=2000, align=512)
    pieces, _ = _drain(s, [_request("a", 3600), _request("b", 3600)])
    for rid in "ab":
        assert all(end % 512 == 0 for _, end in pieces[rid][:-1]), pieces[rid]
        # contiguous: every piece starts where the previous one ended (no restart from 0)
        assert all(p[1][0] == p[0][1] for p in zip(pieces[rid], pieces[rid][1:])), pieces[rid]
        assert pieces[rid][-1][1] == 3600


def test_partial_prefills_are_capped_and_everything_completes_aligned():
    s = _scheduler(budget=2048, align=512, cap=3)
    reqs = [_request(f"r{i}", 6000) for i in range(6)]
    pieces, worst = _drain(s, reqs)
    assert worst <= 3
    for rid, p in pieces.items():
        assert all(end % 512 == 0 for _, end in p[:-1]), (rid, p)
        assert all(x[1][0] == x[0][1] for x in zip(p, p[1:])), (rid, p)
        assert p[0][0] == 0 and p[-1][1] == 6000


def test_mixed_sizes_many_prompts_every_split_piece_is_aligned():
    sizes = [700, 5100, 130, 8300, 2049, 512, 513, 1024, 9999, 64, 3000, 4096, 4097, 1, 7777, 2500]
    for budget in (1024, 2048, 3000, 4608):
        s = _scheduler(budget=budget, align=512, cap=3)
        reqs = [_request(f"r{i}", n) for i, n in enumerate(sizes)]
        pieces, worst = _drain(s, reqs, max_steps=500)
        assert worst <= 3
        for r, n in zip(reqs, sizes):
            p = pieces[r.request_id]
            assert p[0][0] == 0 and p[-1][1] == n
            assert all(end % 512 == 0 for _, end in p[:-1]), (budget, n, p)
            assert all(x[1][0] == x[0][1] for x in zip(p, p[1:])), (budget, n, p)


def test_short_prompts_behind_split_long_ones_take_no_slot():
    s = _scheduler(budget=4096, align=512, cap=1)
    pieces, worst = _drain(s, [_request("long", 9000), _request("s1", 300)])
    assert worst <= 1
    assert pieces["long"][0][0] == 0 and pieces["long"][-1][1] == 9000
    assert pieces["s1"] == [(0, 300)]


# ---- _hold_back_beyond_partial_cap on stubs (no scheduler) -------------------------------------------------


def _stub(budget, waiting, partial=(), align=512, cap=3):
    s = types.SimpleNamespace(
        policy=SchedulingPolicy.FCFS,
        max_num_scheduled_tokens=budget,
        _prefill_chunk_alignment=align,
        _max_partial_prefills=cap,
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=131072)),
    )
    s.waiting = create_request_queue(s.policy)
    for r in waiting:
        s.waiting.add_request(r)
    s._hold_back_beyond_partial_cap = lambda p: TTScheduler._hold_back_beyond_partial_cap(s, p)
    held = TTScheduler._hold_back_partial_prefills(s, list(partial))
    return [r.name for r in s.waiting], [r.name for r in held]


def _r(n, computed=0, name=None):
    return types.SimpleNamespace(num_tokens=n, num_computed_tokens=computed, name=name or n)


def test_cap_leaves_cap_minus_running_slots():
    # two prompts already part-way: one more may start
    assert _stub(2048, [_r(9000, name="a"), _r(9000, name="b")], partial=[_r(9000, 4096), _r(9000, 4096)]) == (
        ["a"],
        ["b"],
    )


def test_cap_counts_only_prompts_that_do_not_fit():
    assert _stub(8192, [_r(100, name="s"), _r(30000, name="L1"), _r(30000, name="L2")], cap=1) == (
        ["s", "L1"],
        ["L2"],
    )


def test_cap_keeps_fifo_order_once_one_prompt_is_held():
    assert _stub(2048, [_r(9000, name="L1"), _r(100, name="s")], partial=[_r(9000, 1024)], cap=1) == (
        [],
        ["L1", "s"],
    )


def test_no_cap_declared_leaves_the_queue_alone():
    assert _stub(2048, [_r(9000), _r(9000)], cap=0) == ([9000, 9000], [])
    assert _stub(2048, [_r(9000), _r(9000)], align=0) == ([9000, 9000], [])


# ---- platform handoff --------------------------------------------------------------------------------------


def _cfg(chunked=True, budget=2048):
    return types.SimpleNamespace(
        additional_config={},
        scheduler_config=types.SimpleNamespace(enable_chunked_prefill=chunked, max_num_batched_tokens=budget),
    )


class _M:
    pass


def test_platform_stores_the_capabilities():
    c = _cfg()
    _store_prefill_chunking(c, {"prefill_chunk_alignment": 512, "max_partial_prefills": 3}, _M)
    assert (get_tt_prefill_chunk_alignment(c), get_tt_max_partial_prefills(c)) == (512, 3)


def test_platform_stores_zero_when_chunked_prefill_is_off_or_undeclared():
    c = _cfg(chunked=False)
    _store_prefill_chunking(c, {"prefill_chunk_alignment": 512, "max_partial_prefills": 3}, _M)
    # The scheduler ignores alignment when chunked prefill is off; the plugin default stays.
    assert (get_tt_prefill_chunk_alignment(c), get_tt_max_partial_prefills(c)) == (128, 0)
    c = _cfg()
    _store_prefill_chunking(c, {}, _M)
    assert (get_tt_prefill_chunk_alignment(c), get_tt_max_partial_prefills(c)) == (128, 0)


def test_platform_rejects_a_budget_below_one_window():
    with pytest.raises(ValueError, match="never advance"):
        _store_prefill_chunking(_cfg(budget=256), {"prefill_chunk_alignment": 512}, _M)


@pytest.mark.parametrize("bad", [-1, True, 1.5, "512"])
def test_platform_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        _store_prefill_chunking(_cfg(), {"prefill_chunk_alignment": bad}, _M)


# ---- reset_prefix_cache reaches the model through the next SchedulerOutput ---------------------------------------------------------


def test_prefix_cache_reset_flag_rides_on_the_next_scheduler_output():
    from vllm_tt_plugin.scheduler import get_tt_prefix_cache_reset

    s = _scheduler(budget=2048, align=512)
    s.add_request(_request("a", 700))
    assert not get_tt_prefix_cache_reset(s.schedule())
    assert s.reset_prefix_cache() in (True, False)
    s._pending_prefix_cache_reset = True  # (a successful reset sets it; the base reset may refuse while a request runs)
    s.add_request(_request("b", 700))
    out = s.schedule()
    assert get_tt_prefix_cache_reset(out)
    s.add_request(_request("c", 700))
    assert not get_tt_prefix_cache_reset(s.schedule())  # one-shot


# ---- prefix-cache hits are rounded to the model's snapshot grid -------------------------------------------------------------------


def test_round_tt_prefix_hit():
    from vllm_tt_plugin.config import round_tt_prefix_hit as r

    assert r(640, 727, 512, "pow2") == 512
    assert r(4992, 5000, 512, "pow2") == 4096  # 9 chunks -> 8
    assert r(4992, 5000, 512, "all") == 4608  # 9 chunks, but 5000 - 128 = 4872 -> 9 * 512 = 4608
    assert r(4992, 5000, 512, "3") == 1536
    assert r(640, 700, 512, "pow2") == 512 and r(640, 600, 512, "pow2") == 0  # the last 128 positions are replayed: 600 - 128 = 472 < 512
    assert r(300, 5000, 512, "pow2") == 0
    assert r(640, 727, 512, "") == 640  # no policy: untouched


def test_scheduler_rounds_the_prefix_hit_and_keeps_the_prompt_chunked():
    s = _scheduler(budget=2048, align=512, hit_policy="pow2")
    a = _request("a", 6000)
    pieces_a, _ = _drain(s, [a])
    assert pieces_a["a"][0] == (0, 2048)
    b = _request("b", 6000)  # identical prompt: vLLM reports a 5888-token hit (46 blocks)
    pieces_b, _ = _drain(s, [b])
    # rounded to 4096 (8 chunks), not 5888: the rest is scheduled in aligned pieces of the budget
    assert pieces_b["b"][0][0] == 4096, pieces_b["b"]
    assert all(end % 512 == 0 for _, end in pieces_b["b"][:-1]) and pieces_b["b"][-1][1] == 6000


def test_scheduler_without_the_policy_keeps_vllm_hits_untouched():
    s = _scheduler(budget=2048, align=512, hit_policy="")
    assert s.kv_cache_manager.enable_caching is False  # (no capability: the platform disabled the cache)
