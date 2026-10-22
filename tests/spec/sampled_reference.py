# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""An independent small-vocabulary reference for the sampled accept walk.

The target's joint distribution comes from a fixed causal table, with the
ordinary sampler's controls applied in plain Python, and the chi-square check
compares committed sequences against it.
"""

from __future__ import annotations

import math

import torch


def _chi_square_critical(df: int, z: float = 3.72) -> float:
    """Upper critical value of chi-square, by the Wilson-Hilferty approximation.

    ``z = 3.72`` is the one-sided normal quantile for a tail of 1e-4. With the
    global generator seeded these tests are deterministic, so the threshold
    decides how far a wrong distribution must be before the test sees it,
    not how often a right one fails.
    """
    term = 2.0 / (9.0 * df)
    return df * (1.0 - term + z * math.sqrt(term)) ** 3


def _chi_square(observed: dict, expected: dict, total: int) -> tuple[float, int]:
    """Pearson's statistic with bins under five expected counts pooled."""
    statistic = 0.0
    bins = 0
    pooled_observed = 0.0
    pooled_expected = 0.0
    for outcome, probability in expected.items():
        count = probability * total
        if count < 5:
            pooled_observed += observed.get(outcome, 0)
            pooled_expected += count
            continue
        statistic += (observed.get(outcome, 0) - count) ** 2 / count
        bins += 1
    if pooled_expected > 0:
        statistic += (pooled_observed - pooled_expected) ** 2 / max(
            pooled_expected, 1e-9
        )
        bins += 1
    unexpected = sum(n for outcome, n in observed.items() if outcome not in expected)
    assert unexpected == 0, (
        f"{unexpected} committed sequences have probability zero under the target"
    )
    return statistic, bins - 1


SMALL_VOCAB = 6
PROMPT = [2, 4, 1]


def _causal_table(seed: int) -> torch.Tensor:
    """``[V, positions, V]`` logits: the target after a token at a position.

    Some entries are ruled out entirely, so a reference that ignored zero
    probabilities, or a walk that committed one, cannot agree with the other.
    """
    generator = torch.Generator().manual_seed(seed)
    table = torch.randn(SMALL_VOCAB, 16, SMALL_VOCAB, generator=generator) * 1.5
    ruled_out = torch.rand(SMALL_VOCAB, 16, SMALL_VOCAB, generator=generator) < 0.2
    # Never rule out every token after a context.
    ruled_out[..., 0] = False
    return table.masked_fill(ruled_out, float("-inf"))


def _reference_distribution(
    raw: list[float],
    history: list[int],
    prompt: list[int],
    controls: dict,
) -> list[float]:
    """The ordinary sampler's distribution, computed in plain Python.

    Repetition first, over prompt and output, then frequency and presence over
    the output; then temperature, top-k and top-p, in that order.
    """
    logits = list(raw)
    repetition = controls.get("repetition", 1.0)
    frequency = controls.get("frequency", 0.0)
    presence = controls.get("presence", 0.0)
    for token in range(SMALL_VOCAB):
        if token in history or token in prompt:
            if logits[token] > 0:
                logits[token] /= repetition
            else:
                logits[token] *= repetition
        count = history.count(token)
        logits[token] -= frequency * count
        logits[token] -= presence * (1.0 if count else 0.0)

    temperature = controls.get("temperature", 1.0)
    logits = [value / temperature for value in logits]

    def softmax(values):
        finite = [v for v in values if v != float("-inf")]
        top = max(finite)
        weights = [math.exp(v - top) if v != float("-inf") else 0.0 for v in values]
        total = sum(weights)
        return [w / total for w in weights]

    top_k = controls.get("top_k")
    if top_k:
        kth = sorted(logits, reverse=True)[top_k - 1]
        logits = [v if v >= kth else float("-inf") for v in logits]
    top_p = controls.get("top_p")
    if top_p:
        probs = softmax(logits)
        order = sorted(range(SMALL_VOCAB), key=lambda t: probs[t])
        cumulative = 0.0
        removed = set()
        for token in order[:-1]:
            cumulative += probs[token]
            if cumulative <= 1.0 - top_p:
                removed.add(token)
        logits = [float("-inf") if t in removed else v for t, v in enumerate(logits)]
    return softmax(logits)


def _argmax_drafter(table):
    """Proposes the target's own unpenalized argmax, which is often accepted."""

    def drafter(token, position):
        return table[token, position.clamp(max=table.shape[1] - 1)].argmax(dim=-1)

    return drafter


def _wrong_drafter(token, position):
    """Proposes a token that ignores the target, which is often rejected."""
    return (token * 5 + position * 3 + 1) % SMALL_VOCAB


def _sampling_drafter(table):
    """Samples each draft from a distribution unlike the target, with its own RNG.

    The walk treats every draft as a point mass, which is lossless for any
    draft its own random draws did not choose. This drafter is the case that
    claim has to survive: random drafts from a ``q`` far from ``p``.
    """
    generator = torch.Generator().manual_seed(123)

    def drafter(token, position):
        logits = table[token, position.clamp(max=table.shape[1] - 1)]
        q = (-0.7 * logits.nan_to_num(neginf=-30.0)).softmax(dim=-1)
        return torch.multinomial(q, 1, generator=generator).squeeze(1)

    return drafter


def _pack(allowed: torch.Tensor) -> torch.Tensor:
    """xgrammar's packed layout for a ``[..., V]`` bool allowlist."""
    words = (allowed.shape[-1] + 31) // 32
    padded = torch.zeros((*allowed.shape[:-1], words * 32), dtype=torch.bool)
    padded[..., : allowed.shape[-1]] = allowed
    bits = padded.reshape(*allowed.shape[:-1], words, 32).to(torch.int64)
    weights = torch.tensor([1 << b for b in range(32)], dtype=torch.int64)
    packed = (bits * weights).sum(-1)
    # Two's complement, which is how an int32 word holds bit 31.
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
