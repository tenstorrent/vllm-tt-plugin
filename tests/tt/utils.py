# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
import asyncio
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class RequestConfig:
    """Configuration for a single request in a batch."""

    prompt: str
    max_tokens: int = 10
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logprobs: int | None = None
    min_p: float = 0.0
    bad_words: list[str] | None = None
    logit_bias: dict[int, float] | None = None
    allowed_token_ids: list[int] | None = None
    min_tokens: int = 0
    return_tokens_as_token_ids: bool = False


async def send_request(
    async_client, model: str, config: RequestConfig, return_full_response: bool = False
):
    """Send a single async legacy completion request (old API)."""
    extra_body: dict[str, Any] = {}
    if config.top_k is not None:
        extra_body["top_k"] = config.top_k
    if config.repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = config.repetition_penalty
    if config.min_p != 0.0:
        extra_body["min_p"] = config.min_p
    if config.min_tokens != 0:
        extra_body["min_tokens"] = config.min_tokens
    if config.logit_bias is not None:
        extra_body["logit_bias"] = config.logit_bias
    if config.allowed_token_ids is not None:
        extra_body["allowed_token_ids"] = config.allowed_token_ids
    if config.return_tokens_as_token_ids:
        extra_body["return_tokens_as_token_ids"] = True
    if config.bad_words is not None:
        raise ValueError("bad_words is not supported in legacy completions API")

    response = await async_client.completions.create(
        model=model,
        prompt=config.prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        logprobs=config.logprobs,
        extra_body=extra_body if extra_body else None,
    )
    if return_full_response:
        return response
    return response.choices[0].text


async def send_chat_request(
    async_client, model: str, config: RequestConfig, return_full_response: bool = False
):
    """Send a single async chat completion request (new API)."""
    extra_body: dict[str, Any] = {}
    if config.top_k is not None:
        extra_body["top_k"] = config.top_k
    if config.repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = config.repetition_penalty
    if config.min_p != 0.0:
        extra_body["min_p"] = config.min_p
    if config.min_tokens != 0:
        extra_body["min_tokens"] = config.min_tokens
    if config.bad_words is not None:
        extra_body["bad_words"] = config.bad_words
    if config.logit_bias is not None:
        extra_body["logit_bias"] = config.logit_bias
    if config.allowed_token_ids is not None:
        extra_body["allowed_token_ids"] = config.allowed_token_ids

    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": config.prompt}],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        logprobs=config.logprobs is not None,
        top_logprobs=config.logprobs,
        extra_body=extra_body if extra_body else None,
    )
    if return_full_response:
        return response
    return response.choices[0].message.content


async def send_batch_concurrent(
    async_client,
    model: str,
    configs: list[RequestConfig],
    use_chat: bool = False,
    return_full_response: bool = False,
):
    """
    Send multiple requests concurrently with different per-request parameters.
    The vLLM server will batch these together internally.

    Args:
        use_chat: use chat completions API instead of legacy completions API.
        return_full_response: return full response objects vs just text.
    """
    send_fn = send_chat_request if use_chat else send_request
    tasks = [
        send_fn(async_client, model, cfg, return_full_response=return_full_response)
        for cfg in configs
    ]
    return await asyncio.gather(*tasks)


def run_concurrent_batch(
    tt_server,
    tt_model_name,
    configs: list[RequestConfig],
    use_chat: bool = False,
    return_full_response: bool = False,
):
    """
    Synchronous wrapper to run concurrent requests.
    Returns list of output texts (or full responses) in same order as configs.

    Args:
        use_chat: use chat completions API instead of legacy completions API.
        return_full_response: return full response objects vs just text.
    """

    async def _run():
        async_client = tt_server.get_async_client()
        try:
            return await send_batch_concurrent(
                async_client,
                tt_model_name,
                configs,
                use_chat=use_chat,
                return_full_response=return_full_response,
            )
        finally:
            await async_client.close()

    return asyncio.run(_run())


def assert_varied(results, min_varied, explanation):
    unique_results = set(results)
    assert len(unique_results) >= min_varied, (
        f"{explanation}\n"
        f"Expected at least {min_varied} unique results.\n"
        f"Only {len(unique_results)}/{len(results)} were varied.\n"
        f"Results: {results}"
    )


def assert_pairwise_varied(results1, results2, min_varied, explanation):
    different = [x != y for x, y in zip(results1, results2)]
    assert sum(different) >= min_varied, (
        f"{explanation}\n"
        f"Expected difference on re-run.\n"
        f"Results: {results1} + {results2}"
    )


def assert_deterministic(results, explanation):
    unique_results = set(results)
    assert len(unique_results) == 1, (
        f"{explanation}\n"
        f"Expected reproducible outputs.\n"
        f"Got {len(unique_results)} unique results out of {len(results)}.\n"
        f"Results: {results}"
    )


# --- Near-tie tolerance -------------------------------------------------------
#
# TT decode accumulates ~0.15-0.25 relative error in the residual stream versus a
# precision-matched reference, because cross-device partial sums are rounded to
# bfloat8_b *before* being summed (ccl_dtype). That lands as a few tenths of a nat
# on the logits. It is deterministic for a fixed computation, but the computation
# is not fixed: slot position, batch composition and CCL reduce ordering all change
# the arithmetic. So when two candidate tokens sit closer together than that margin,
# which one wins depends on where in the batch the request landed.
#
# The practical signature is ONE request out of a batch disagreeing with the rest.
# A real determinism bug does not look like that -- it corrupts many slots, or the
# same slot every time. We therefore warn on a single outlier and fail on more.
NEAR_TIE_NOTE = (
    "This is the signature of a near-tie: two candidate tokens closer together than "
    "TT's decode error margin, so batch position decides the winner. It is a known "
    "precision property, not a sampling bug. If this warning becomes frequent, or "
    "the count rises above one, treat it as a real regression."
)


def assert_deterministic_allow_near_tie(
    results, explanation, max_warn_outliers: int = 1
):
    """Like assert_deterministic, but tolerate a single odd-one-out with a warning.

    Fails when more than ``max_warn_outliers`` results disagree with the majority,
    which is what an actual determinism bug looks like. Use for assertions over a
    *batch*, where near-tie flips are possible; keep plain assert_deterministic for
    repeated single requests, which share one batch position and must not vary.
    """
    if len(set(results)) == 1:
        return

    counts = Counter(results)
    majority, majority_n = counts.most_common(1)[0]
    outliers = [r for r in results if r != majority]

    if len(outliers) <= max_warn_outliers:
        warnings.warn(
            f"{explanation}\n"
            f"{len(outliers)}/{len(results)} result(s) differed from the majority.\n"
            f"{NEAR_TIE_NOTE}\n"
            f"majority ({majority_n}x): {majority!r}\n"
            f"outlier(s): {outliers!r}",
            stacklevel=2,
        )
        return

    raise AssertionError(
        f"{explanation}\n"
        f"Expected reproducible outputs across the batch.\n"
        f"Got {len(counts)} unique results out of {len(results)}; "
        f"{len(outliers)} differed from the majority "
        f"(more than the {max_warn_outliers} tolerated as a near-tie).\n"
        f"Results: {results}"
    )


# --- Penalty measurement ------------------------------------------------------
#
# NOTE: asserting on logprobs does NOT work here. The logprobs this server returns
# are computed from PRE-penalty logits: with identical sampled text, presence,
# frequency and repetition penalties all leave every reported logprob byte-identical,
# even though the penalty demonstrably reaches the sampler. So a penalty is only
# observable through the sampled TEXT.
#
# That makes a single-prompt text assertion fragile, because a penalty of P only
# changes the text when P exceeds the model's top-2 gap at a repeat -- a property of
# the model, not the penalty. The fix is to assert over a SET of prompts: a working
# penalty moves most of them, a broken one moves none, and no single prompt's
# confidence can decide the result.


def count_prompts_changed_by(
    tt_server, tt_model_name, prompts, max_tokens=24, **penalty_kwargs
):
    """Return (n_changed, detail_lines) for greedy output with vs without a penalty."""
    changed = 0
    detail = []
    for prompt in prompts:
        base, pen = run_concurrent_batch(
            tt_server,
            tt_model_name,
            [
                RequestConfig(prompt=prompt, max_tokens=max_tokens, temperature=0),
                RequestConfig(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0,
                    **penalty_kwargs,
                ),
            ],
        )
        if base != pen:
            changed += 1
            detail.append(f"  CHANGED {prompt!r}")
        else:
            detail.append(f"  same    {prompt!r}\n    -> {base!r}")
    return changed, detail
