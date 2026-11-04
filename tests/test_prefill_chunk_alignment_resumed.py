# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Every intermediate prefill chunk ends on the alignment grid, also for
preempted / resumed requests and prefix-cache hits on 64-token blocks.

Production 2026-11-03 20:15:28Z (Qwen3.8, block_size 64, cap table
[[0, 16384], [131072, 8192]]): a request preempted at 96 % KV usage came back
with 67439 prompt + 7587 output tokens (num_tokens 75026) and a 54080-token
prefix hit. Its chunk [54080, 70464) ended past the prompt, so the scheduler
skipped the alignment, but below num_tokens, so the model runner flagged it
intermediate: the model's 128-grid assert took the engine down.
"""

import random
from types import SimpleNamespace

import vllm  # noqa: F401  (resolve the platform plugin before importing ours)

from vllm_tt_plugin.config import parse_prefill_chunk_cap_table
from vllm_tt_plugin.scheduler import TTScheduler

ALIGN = 128
PROD_CAPS = parse_prefill_chunk_cap_table([[0, 16384], [131072, 8192]], ALIGN)


def _scheduler(caps=PROD_CAPS, step=525312, threshold=16384):
    s = TTScheduler.__new__(TTScheduler)
    s._prefill_chunk_alignment = ALIGN
    s.need_mamba_block_aligned_split = False
    s._prefill_chunk_cap_table = tuple(caps)
    s.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=step,
        max_num_scheduled_tokens=None,
        long_prefill_token_threshold=threshold,
    )
    return s


def _request(prompt, output=0, computed=0):
    return SimpleNamespace(
        num_prompt_tokens=prompt,
        num_tokens=prompt + output,
        num_computed_tokens=computed,
        mm_features=[],
        has_encoder_inputs=False,
    )


def _intermediate(request, start, grant):
    # model_runner: intermediate_prefill_mask = prompt_lens < num_tokens
    return start + grant < request.num_tokens


def test_prod_crash_resumed_request_chunk_is_aligned():
    s = _scheduler()
    req = _request(67439, output=7587)  # waiting (preempted): num_computed_tokens 0
    grant = s._mamba_block_aligned_split(req, 16384, 54080, 0)
    end = 54080 + grant
    assert _intermediate(req, 54080, grant)
    assert end % ALIGN == 0, end  # was 70464 (= 64 mod 128) before the fix
    assert end == 70400


def test_resumed_request_final_chunk_is_left_alone():
    s = _scheduler()
    req = _request(67439, output=7587, computed=70400)
    assert s._mamba_block_aligned_split(req, 75026 - 70400) == 75026 - 70400
    # the chunk that ends exactly at num_tokens - 1 is still intermediate: aligned
    grant = s._mamba_block_aligned_split(req, 75025 - 70400)
    assert (70400 + grant) % ALIGN == 0


def test_resumed_chunk_inside_the_prompt_is_aligned_as_before():
    s = _scheduler()
    req = _request(67439, output=7587)
    grant = s._mamba_block_aligned_split(req, 16384, 17344, 0)  # hit = 64 mod 128
    assert 17344 + grant == 33664  # the aligned chunk seen in the prod log


def test_prefix_hit_off_grid_with_short_leftover_budget_waits():
    s = _scheduler()
    req = _request(90000)  # admitting: num_computed_tokens 0, 64-aligned hit
    assert s._mamba_block_aligned_split(req, 50, 70400, 0) == 0  # was 50 -> end 70450
    assert s._mamba_block_aligned_split(req, 60, 70336, 0) == 0  # was 60 -> end 70396
    # enough budget: rounded down to the grid
    assert 70336 + s._mamba_block_aligned_split(req, 1000, 70336, 0) == 71296


def test_running_request_on_grid_with_short_leftover_budget_waits():
    s = _scheduler()
    req = _request(90000, computed=70400)
    assert s._mamba_block_aligned_split(req, 64) == 0  # was 64 -> end 70464
    assert s._mamba_block_aligned_split(req, 4096) == 4096


def test_decode_and_spec_rows_untouched():
    s = _scheduler()
    dec = _request(300, output=10, computed=309)
    assert s._mamba_block_aligned_split(dec, 1) == 1
    assert s._mamba_block_aligned_split(dec, 4) == 4
    resumed_dec = _request(300, output=10, computed=309)
    assert s._mamba_block_aligned_split(resumed_dec, 1) == 1


def _walk(s, prompt, output, hit, rng):
    """Drive one (possibly resumed) request's prefill to completion with random
    leftover budgets; returns the list of intermediate chunk ends."""
    req = _request(prompt, output)
    ends, steps = [], 0
    start, admitting_hit = hit, hit
    while start < req.num_tokens:
        steps += 1
        assert steps < 10000, "no progress"
        budget = rng.choice([rng.randint(1, 200), rng.randint(1, 40000), 525312])
        n = min(req.num_tokens - start, 16384, budget)
        if req.num_computed_tokens == 0:
            g = s._mamba_block_aligned_split(req, n, admitting_hit, 0)
        else:
            g = s._mamba_block_aligned_split(req, n)
        if g == 0:
            continue
        assert 0 < g <= n
        start += g
        req.num_computed_tokens = start
        if start < req.num_tokens:
            ends.append(start)
    return ends


def test_random_walks_never_produce_an_off_grid_intermediate_end():
    s = _scheduler()
    rng = random.Random(20261103)
    for _ in range(3000):
        prompt = rng.randint(2, 200000)
        output = rng.choice([0, 0, rng.randint(1, 20000)])
        total = prompt + output
        hit = rng.randint(0, max(0, (total - 1) // 64)) * 64 if rng.random() < 0.7 else 0
        ends = _walk(s, prompt, output, hit, rng)
        bad = [e for e in ends if e % ALIGN]
        assert not bad, (prompt, output, hit, bad)
