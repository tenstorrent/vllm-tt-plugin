# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Block-output models with ``MambaSpec`` state in mamba cache mode align.

The TT scheduler drives vLLM's real hybrid coordinator here. A block-output
model commits several tokens per decode step, so these tests pin what the
scheduler and the runner do differently from a token model: the block-aligned
prompt split, caching of blocks filled by decode steps, and state columns that
never move back.
"""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.test_block_scheduler import BLOCK_SIZE, CANVAS, _scheduler
from vllm_tt_plugin import worker as worker_module
from vllm_tt_plugin.scheduler import TTScheduler

ATTN_LAYERS = [3, 7]
MAMBA_LAYERS = [0, 1, 2, 4, 5, 6]
ATTN_SPEC = FullAttentionSpec(
    block_size=BLOCK_SIZE, num_kv_heads=1, head_size=4, dtype=torch.float32
)


def _align_scheduler(
    output_width: int, retention_interval: int | None = 0, **scheduler_kwargs
) -> TTScheduler:
    """A TT scheduler over vLLM's real hybrid coordinator with prefix caching on
    and the Mamba groups in mode align (state block = BLOCK_SIZE tokens).
    ``retention_interval`` is ``prefix_cache_retention_interval``; vLLM's
    default 0 keeps Mamba states only at replay boundaries and shared-prefix
    junctions, ``None`` keeps every block."""
    base = _scheduler(output_width=output_width, max_num_seqs=2, **scheduler_kwargs)
    config = base.vllm_config
    config.cache_config.prefix_cache_retention_interval = retention_interval
    config.cache_config.enable_prefix_caching = True
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.mamba_block_size = BLOCK_SIZE
    align = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((3, 8), (2, 4, 4)),
        dtypes=(torch.float32, torch.float32),
        page_size_padded=ATTN_SPEC.page_size_bytes,
        mamba_cache_mode="align",
    )
    specs = {f"model.layers.{i}.self_attn": ATTN_SPEC for i in ATTN_LAYERS}
    specs |= {f"model.layers.{i}.linear_attn": align for i in MAMBA_LAYERS}
    sizing_config, groups = worker_module._kv_cache_groups_for_sizing(config, specs)
    group_size = max(len(g.layer_names) for g in groups)
    kv_cache_config = get_kv_cache_config_from_groups(
        sizing_config,
        groups,
        available_memory=32 * ATTN_SPEC.page_size_bytes * group_size,
    )
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, config
    )
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(req_id: str, prompt: list[int], max_tokens: int) -> Request:
    init_none_hash(sha256)
    return Request(
        request_id=req_id,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _finish_step(scheduler, out, tokens_per_req):
    req_ids = list(out.num_scheduled_tokens)
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[tokens_per_req.get(r, []) for r in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _drive(scheduler, request, outputs, async_scheduling):
    """Run ``outputs`` (one commit per scheduled step) through ``scheduler``;
    under async scheduling each step is scheduled before the previous one
    commits, as the engine core does. Returns the scheduler outputs."""
    pending = None
    seen = []
    for tokens in outputs:
        out = scheduler.schedule()
        if not out.num_scheduled_tokens:
            break
        seen.append(out)
        if async_scheduling:
            if pending is not None:
                _finish_step(scheduler, *pending)
            pending = (out, {request.request_id: tokens})
        else:
            _finish_step(scheduler, out, {request.request_id: tokens})
    if pending is not None:
        _finish_step(scheduler, *pending)
    return seen


def _blocks(n: int, first: int = 1) -> list[list[int]]:
    return [list(range(100 * k, 100 * k + CANVAS)) for k in range(first, first + n)]


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "prompt_len", [BLOCK_SIZE + 44, BLOCK_SIZE - 28], ids=["split", "decode-crosses"]
)
def test_align_adaptive_block_model_prompt_and_decode(prompt_len, async_scheduling):
    """A prompt longer than a block is cut at the boundary: the intermediate
    chunk commits nothing, the final chunk its one anchor token. Solo decodes
    then commit whole blocks, and a decode that crosses a block boundary stays
    a decode: vLLM's align split must not cut it as a resumed replay."""
    scheduler = _align_scheduler(
        output_width=CANVAS, adaptive=True, async_scheduling=async_scheduling
    )
    request = _request("a", range(prompt_len), max_tokens=1 + 5 * CANVAS)
    scheduler.add_request(request)
    prefill = [[], [7]] if prompt_len > BLOCK_SIZE else [[7]]
    _drive(scheduler, request, prefill + _blocks(3), async_scheduling)
    assert list(request.output_token_ids) == [7] + [t for b in _blocks(3) for t in b]


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("retention_interval", "expected_hit"),
    [(None, BLOCK_SIZE), (0, 0)],
    ids=["dense-retention", "default-retention"],
)
def test_blocks_filled_by_block_output_decodes_are_cached_as_vllm_caches_them(
    async_scheduling, retention_interval, expected_hit
):
    """A prompt shorter than a block, then block decodes that fill it. The
    scheduler offers the block to vLLM's cache once its last fed token commits,
    as AsyncScheduler does for a token model. With dense retention a resend of
    prompt plus answer then hits the whole block. With vLLM's default retention
    the Mamba groups keep no state for it (only replay boundaries and
    shared-prefix junctions), so the hit is 0, as for a token model."""
    scheduler = _align_scheduler(
        output_width=CANVAS,
        retention_interval=retention_interval,
        adaptive=True,
        async_scheduling=async_scheduling,
    )
    first = _request("a", range(BLOCK_SIZE - 28), max_tokens=1 + 3 * CANVAS)
    scheduler.add_request(first)
    _drive(scheduler, first, [[7]] + _blocks(3), async_scheduling)
    assert first.num_tokens > BLOCK_SIZE
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)

    resend = _request("b", first.all_token_ids[: BLOCK_SIZE + 4], max_tokens=1)
    scheduler.add_request(resend)
    (new,) = scheduler.schedule().scheduled_new_reqs
    assert new.num_computed_tokens == expected_hit


def test_a_replay_past_the_prompt_is_still_cut_at_the_block_boundary():
    """A resumed request replays its output as prefill. Its chunks must still
    end on a block boundary, so the boundary state is a chunk's final state;
    only a block-output decode, which starts at most one output block below the
    end of the request, is left uncut."""
    scheduler = _align_scheduler(output_width=CANVAS, adaptive=True)
    request = _request("a", range(100), max_tokens=200)
    request.append_output_token_ids(list(range(60)))  # 160 tokens
    request.num_computed_tokens = 110
    assert scheduler._mamba_block_aligned_split(request, 50) == BLOCK_SIZE - 110

    request.num_computed_tokens = 160 - CANVAS
    assert scheduler._mamba_block_aligned_split(request, CANVAS) == CANVAS


def test_align_state_column_never_moves_back():
    """A block-output step that reserves a canvas past a boundary moves the
    state to the next column. A later step with fewer scheduled tokens must
    not copy it back over the earlier column's block, which may hold the
    boundary state by then."""
    from types import SimpleNamespace

    from vllm_tt_plugin.mamba_state import MambaCacheGroups

    groups = MambaCacheGroups(
        group_indices=(0,),
        mamba_cache_mode="align",
        block_size=BLOCK_SIZE,
        attention_group_index=1,
    )
    req = SimpleNamespace(num_computed_tokens=120, block_ids=[[11, 12, 13], [0]])

    def step(num_scheduled):
        out = SimpleNamespace(
            finished_req_ids=(),
            preempted_req_ids=(),
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=()),
            num_scheduled_tokens={"a": num_scheduled},
        )
        copies = groups.align_state_copies(out, {"a": req})
        table = torch.tensor([req.block_ids[0]])
        ids = groups.state_block_ids_per_layer(
            [table, torch.zeros_like(table)],
            ["a"],
            {"a": req},
            {"a": num_scheduled},
            [0, 1],
            1,
        )
        return copies, int(ids[0][0, 0])

    assert step(CANVAS) == ([(11, 12)], 12)  # canvas reaches column 1
    req.num_computed_tokens = 124
    assert step(1) == ([], 12)  # stays in column 1, no copy back
    req.num_computed_tokens = 2 * BLOCK_SIZE - 4
    assert step(CANVAS) == ([(12, 13)], 13)


@pytest.mark.parametrize(
    ("output_tokens_per_step", "output_blocks_per_request"),
    [(16, 1), (2048 + 16, 3)],
    ids=["canvas-in-one-block", "canvas-spans-blocks"],
)
def test_block_output_mamba_pool_reserves_one_output_canvas_per_request(
    output_tokens_per_step, output_blocks_per_request
):
    """A block-output model with MambaSpec state is sized like a token model,
    except that each running request's output reservation covers a whole output
    canvas: one block when the canvas fits in one, more when it does not."""
    from unittest.mock import patch

    from tests.test_mamba_kv_cache import (
        MAX_BATCH,
        _layer_specs,
        _model_class,
        _vllm_config,
    )
    from vllm_tt_plugin.config import store_tt_output_tokens_per_step
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    cfg = _vllm_config()
    cfg.model_config.max_model_len = 8192
    store_tt_output_tokens_per_step(cfg, output_tokens_per_step)
    specs = _layer_specs(1024)
    with patch(
        "vllm_tt_plugin.worker.get_model_architecture",
        return_value=(_model_class(specs, 131_072), "hybrid-mamba-test"),
    ):
        blocks = get_num_available_blocks_tt(cfg)
    # 128 attention blocks for the token budget and 3 GDN state blocks per
    # request, as for a token model (test_mamba_kv_cache), plus the canvas.
    assert blocks == 128 + MAX_BATCH * output_blocks_per_request + 3 * MAX_BATCH
