# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
import asyncio
import re
import warnings
from collections import Counter
from dataclasses import dataclass, replace
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


def extract_token_ids(response):
    """Token ids for one legacy-completion response.

    Requires the request to have been sent with ``return_tokens_as_token_ids=True``
    and ``logprobs`` set: vLLM then reports each token as the string "token_id:NNN"
    in ``choices[0].logprobs.tokens``. Falls back to the raw token strings if the
    server did not use the token-id form, so a determinism assertion still has
    something exact to compare.
    """
    logprobs = getattr(response.choices[0], "logprobs", None)
    tokens = getattr(logprobs, "tokens", None) if logprobs is not None else None
    if not tokens:
        return response.choices[0].text
    return [
        int(t.split(":", 1)[1])
        if isinstance(t, str) and t.startswith("token_id:")
        else t
        for t in tokens
    ]


def run_concurrent_batch_tokens(tt_server, tt_model_name, configs):
    """Like run_concurrent_batch, but returns per-request TOKEN ID sequences.

    Determinism assertions compare token-by-token, so they can report *where* two
    outputs part company rather than only that they differ. Text comparison cannot:
    one flipped token makes every later token unrelated.

    The configs are sent with return_tokens_as_token_ids and logprobs=0 so the
    server reports the sampled token ids; sampling parameters are untouched.
    """
    token_configs = [
        replace(cfg, return_tokens_as_token_ids=True, logprobs=cfg.logprobs or 0)
        for cfg in configs
    ]
    responses = run_concurrent_batch(
        tt_server, tt_model_name, token_configs, return_full_response=True
    )
    return [extract_token_ids(r) for r in responses]


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
# Comparing whole outputs for equality cannot express that. One flipped token at
# position 2 makes the remaining 18 tokens unrelated, so `len(set(results))` reports
# a single near-tie and a totally corrupt batch identically. What discriminates them
# is WHERE the outputs part company and HOW MANY slots part company together:
#
#   one slot, deep mid-stream, identical prefix -> near-tie, or seed handling
#   several slots, first tokens, same index      -> shared upstream cause
#
# So we diverge-index instead of set-count: find the first differing unit against
# the majority, report the histogram, and gate on the number of deviating SLOTS.
NEAR_TIE_NOTE = (
    "This is the signature of a near-tie: two candidate tokens closer together than "
    "TT's decode error margin, so batch position decides the winner. It is a known "
    "precision property, not a sampling bug. If this warning becomes frequent, or "
    "more than one slot deviates, treat it as a real regression."
)

# Word-ish units used when a result is plain text. Real token ids are exact and are
# used instead whenever the caller passes them (see run_concurrent_batch_tokens);
# this split only has to be stable and monotonic to make a divergence index useful.
_UNIT_RE = re.compile(r"\s*\S+|\s+")


def _key(result):
    """Hashable identity for a result (token-id sequences are lists)."""
    return result if isinstance(result, str) else tuple(result)


def _as_units(result):
    """Decompose one result into comparable units.

    Sequences (token id lists from ``return_tokens_as_token_ids``) are used verbatim.
    Strings are split into word-ish units, which approximates token position closely
    enough to tell "diverged immediately" from "diverged deep mid-stream".
    """
    if isinstance(result, str):
        return _UNIT_RE.findall(result)
    return list(result)


def first_divergence(a, b):
    """Index of the first unit where ``a`` and ``b`` differ, or None if one is a
    prefix of the other and they are otherwise equal."""
    ua, ub = _as_units(a), _as_units(b)
    for i, (x, y) in enumerate(zip(ua, ub)):
        if x != y:
            return i
    if len(ua) != len(ub):
        return min(len(ua), len(ub))
    return None


def _describe_divergences(results, majority):
    """(outlier_slots, histogram, shared_prefix_units) for results vs majority."""
    outlier_slots = [i for i, r in enumerate(results) if r != majority]
    histogram = Counter()
    shared = None
    for i in outlier_slots:
        idx = first_divergence(majority, results[i])
        histogram[idx] += 1
        shared = idx if shared is None else min(shared, idx)
    return outlier_slots, histogram, shared


def assert_deterministic_allow_near_tie(
    results,
    explanation,
    max_outlier_slots: int = 1,
    min_common_prefix: int = 1,
):
    """Like assert_deterministic, but tolerate a single deviating slot with a warning.

    A near-tie is defined RELATIVE TO A MAJORITY, so three conditions must all hold
    before a deviation is excused:

    1. A strict majority exists (``majority_n > len(outliers)``). With two results
       that disagree there is no majority and therefore no outlier -- neither is more
       authoritative than the other, so the disagreement is reported, not excused.
       Without this the failure branch is unreachable at n == 2, which silently
       disabled every 2-element assertion.
    2. Each deviation shares at least ``min_common_prefix`` leading units with the
       majority. A near-tie parts company at ONE point after a shared prefix; an
       output that differs from its first token is a different phenomenon and is
       never excused, however few slots show it.
    3. At most ``max_outlier_slots`` slots deviate.

    ``results`` may be output texts or token-id sequences. Token ids give exact
    divergence indices; text falls back to word-ish units.
    """
    if len(set(map(_key, results))) == 1:
        return

    counts = Counter(map(_key, results))
    majority_key, majority_n = counts.most_common(1)[0]
    majority = next(r for r in results if _key(r) == majority_key)
    outlier_slots, histogram, earliest = _describe_divergences(results, majority)
    where = ", ".join(
        f"index {idx} x{n}"
        for idx, n in sorted(histogram.items(), key=lambda kv: (kv[0] is None, kv[0]))
    )
    alternatives = sorted({_key(results[i]) for i in outlier_slots})

    detail = (
        f"{len(outlier_slots)}/{len(results)} slot(s) diverged from the majority "
        f"({majority_n}x).\n"
        f"Divergence position: {where} (earliest: {earliest}).\n"
        f"Slots: {outlier_slots}\n"
        f"Majority: {majority!r}\n"
        f"Distinct alternatives ({len(alternatives)}): {alternatives!r}"
    )

    reasons = []
    if majority_n <= len(outlier_slots):
        reasons.append(
            f"no strict majority ({majority_n} vs {len(outlier_slots)} deviating): "
            f"nothing here is authoritative enough for the others to be near-ties"
        )
    if earliest is not None and earliest < min_common_prefix:
        reasons.append(
            f"diverges at index {earliest}, before the {min_common_prefix}-unit "
            f"shared prefix a near-tie requires"
        )
    if len(outlier_slots) > max_outlier_slots:
        reasons.append(
            f"{len(outlier_slots)} slots deviate, more than the "
            f"{max_outlier_slots} tolerated"
        )

    if not reasons:
        warnings.warn(f"{explanation}\n{detail}\n{NEAR_TIE_NOTE}", stacklevel=2)
        return

    raise AssertionError(
        f"{explanation}\n"
        f"Expected reproducible outputs across the batch.\n"
        f"{detail}\n"
        "Not excusable as a near-tie because: " + "; ".join(reasons)
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
    """Return (n_changed, n_slot_noise, detail_lines) for greedy output ± a penalty.

    Each prompt is run as a TWO-REQUEST batch, so the baseline and the penalised
    request occupy different slots. Slot position alone can flip a near-tie on TT
    (that is the whole premise of the near-tie tolerance above), so a raw "changed"
    count cannot distinguish "the penalty did something" from "the two slots
    disagreed anyway".

    So every prompt is also run as a CONTROL: the same two-request batch, same
    width, same slots, both requests UNPENALISED. Any disagreement there is slot
    noise by construction, and ``n_slot_noise`` is the floor that ``n_changed``
    has to clear before it means anything.
    """
    changed = 0
    slot_noise = 0
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
        control_a, control_b = run_concurrent_batch(
            tt_server,
            tt_model_name,
            [
                RequestConfig(prompt=prompt, max_tokens=max_tokens, temperature=0),
                RequestConfig(prompt=prompt, max_tokens=max_tokens, temperature=0),
            ],
        )
        noisy = control_a != control_b
        if noisy:
            slot_noise += 1
        if base != pen:
            changed += 1
            detail.append(
                f"  CHANGED {prompt!r}"
                + ("   [control also disagreed]" if noisy else "")
            )
        else:
            detail.append(f"  same    {prompt!r}\n    -> {base!r}")
    return changed, slot_noise, detail
