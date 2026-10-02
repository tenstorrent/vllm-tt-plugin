# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the ``depth`` target emits, step by step, in closed form.

`DummySpecDecodeModel` under `TT_SPEC_TARGET=depth` is arithmetic, so the whole
response is predictable from the accept depth alone, and predictable per step
rather than only in total. That matters for the termination tests: a token that
ends a request has to be chosen from strictly inside a step's committed prefix,
and which tokens those are depends on the depth.

The sequence, for a draft length of ``k`` and an accept depth of ``d``:

- The prefill commits token 0, because `DummyNoOpModel.prefill_forward` returns
  zero logits and the argmax of that is 0.
- The first decode step has no drafts in flight, because the drafter runs after
  a commit, so it commits ``last + 1``.
- Every later step is offered ``k`` drafts, which are ``last + 1 .. last + k``.
  It accepts the first ``d`` of them unchanged. If ``d == k`` there is nothing
  to reject and the bonus follows, so the step commits ``k + 1`` consecutive
  tokens. If ``d < k`` the step commits ``d`` consecutive tokens and then the
  target's own choice at the position that rejected, which is one past the
  draft offered there: the sequence skips a value.

So an accept-all run counts up by one forever, and a partially accepting run
counts up by one inside each accepted prefix and by two across each rejection.
"""

from __future__ import annotations

import math


def depth_target_blocks(k: int, depth: int | None, steps: int) -> list[list[int]]:
    """The tokens each step commits, starting with the prefill's own.

    Returns one list per step: the prefill's single token, then the first
    decode step's single token, then one block per speculative step.
    """
    accepted = k if depth is None else min(depth, k)
    blocks: list[list[int]] = [[0], [1]]
    last = 1
    while len(blocks) < steps:
        block: list[int] = []
        for _ in range(accepted):
            last += 1
            block.append(last)
        if accepted == k:
            # Nothing rejected, so the bonus follows the last accepted draft.
            last += 1
        else:
            # The position that rejected commits the target's own choice there,
            # which is one past the draft it was offered.
            last += 2
        block.append(last)
        blocks.append(block)
    return blocks


def depth_target_ids(k: int, depth: int | None, count: int) -> list[int]:
    """The first ``count`` committed ids, flattened."""
    blocks = depth_target_blocks(k, depth, steps=count + 2)
    flat = [token for block in blocks for token in block]
    return flat[:count]


def a_token_strictly_inside_a_prefix(k: int, depth: int | None) -> int | None:
    """A token a step commits neither first nor last, or ``None``.

    This is what a termination test needs: ending a request on the first token
    of a block proves nothing about discarding the rest of it, and ending on
    the last token leaves nothing to discard. ``None`` means this
    configuration commits blocks too narrow to have an inside, which is the
    case at an accept depth of 0.
    """
    for block in depth_target_blocks(k, depth, steps=6)[2:]:
        if len(block) >= 3:
            return block[1]
    return None


def ids_up_to_and_including(k: int, depth: int | None, token: int) -> list[int]:
    """Every committed id up to ``token``, which is where a stop lands.

    vLLM carries the stopping token as the last id of the response and leaves
    it out of the text, so this is the whole expected ``token_ids`` for a
    request that stopped on ``token``.
    """
    ids = depth_target_ids(k, depth, count=512)
    return ids[: ids.index(token) + 1]


def discarded_after(k: int, depth: int | None, token: int) -> int:
    """How many tokens of ``token``'s own step follow it.

    A request that ends on ``token`` has to discard exactly these, which is
    the property a termination test is after. Zero means this configuration
    puts ``token`` at the end of its step, so ending there discards nothing
    and proves nothing.
    """
    for block in depth_target_blocks(k, depth, steps=64):
        if token in block:
            return len(block) - block.index(token) - 1
    return 0


# The ``fixed`` target's rule, which is a different instrument from the
# ``depth`` target above. The depth target returns each draft unchanged up to
# its accept depth, so its output is a function of what was drafted and the
# depth is a knob for acceptance accounting. The fixed target chooses by one
# rule for an ordinary decode and for a verify alike, reading only the token
# and its position, so its output is a property of the prompt tail and of
# nothing else: not of the accept depth, not of how much any step committed,
# not of whether a step overlapped another, and not of whether the launch
# schedules synchronously. That is what makes it the target for an equality
# claim, and the accept depth then only decides how much of each step's block
# the drafter gets right.

FIXED_TARGET_VOCAB = 128256


def fixed_target_choice(token: int, position: int) -> int:
    """The one rule: what this target chooses after ``token`` at ``position``."""
    return (token * 31 + position * 7 + 11) % FIXED_TARGET_VOCAB


def fixed_target_ids(prompt: list[int], count: int) -> list[int]:
    """The first ``count`` ids a ``fixed`` target emits for ``prompt``.

    One rule, from the prompt's last token at that token's position, and then
    from each emitted token at its own. The prefill follows it too, which is
    what lets this sequence describe a request that was preempted, reset or
    replayed: each of those replays the history through a prefill, and a
    prefill outside the rule would put a token in the middle of the response
    that the rule cannot explain.

    A replay emits no token of its own here. It recomputes the same choice from
    the same tail, which is the token the request already had.
    """
    ids: list[int] = []
    token, position = prompt[-1], len(prompt) - 1
    while len(ids) < count:
        token = fixed_target_choice(token, position)
        position += 1
        ids.append(token)
    return ids


# The ``fixed`` target's distribution, which a launch declaring the ``logits``
# accept mode samples from. Its argmax is ``fixed_target_choice``, so a greedy
# request emits the rule's sequence. The support is the rule's choice and three
# tokens shared by every context, and nothing else: a committed token outside
# the support is a wrong commit, not sampling noise.
#
# Mirrors FIXED_SHARED_ALTERNATIVES and FIXED_PROBABILITIES in tt-metal's
# models/vllm_test_utils/spec_test/test_model.py, which places the shared
# tokens modulo the vocabulary; the dummy's 128256 needs no wrap.

FIXED_SHARED_ALTERNATIVES = (17, 4099, 65537)
FIXED_PROBABILITIES = (0.4, 0.3, 0.2, 0.1)


def fixed_target_support(token: int, position: int) -> list[int]:
    """The tokens that can follow ``token`` at ``position``, rule's choice first."""
    return [fixed_target_choice(token, position), *FIXED_SHARED_ALTERNATIVES]


def fixed_target_distribution(
    token: int,
    position: int,
    *,
    history: list[int] = (),
    prompt: list[int] = (),
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> dict[int, float]:
    """What vLLM's ordinary sampler draws from after ``token`` at ``position``.

    Computed here, in plain Python, rather than through the plugin or vLLM, so
    a test compares the server against an independent statement of the
    sampling semantics: repetition over prompt and output first, then
    frequency and presence over the output, then temperature, top-k and top-p.
    ``temperature`` 0 returns the greedy point mass. Tokens outside the
    support stay at probability 0 under every control.
    """
    weights: dict[int, float] = {}
    for candidate, probability in zip(
        fixed_target_support(token, position), FIXED_PROBABILITIES
    ):
        weights[candidate] = weights.get(candidate, 0.0) + probability
    logits = {}
    for candidate, probability in weights.items():
        value = math.log(probability)
        if candidate in history or candidate in prompt:
            value = (
                value / repetition_penalty if value > 0 else value * repetition_penalty
            )
        count = list(history).count(candidate)
        value -= frequency_penalty * count
        value -= presence_penalty * (1.0 if count else 0.0)
        logits[candidate] = value
    if temperature == 0:
        best = max(logits, key=lambda candidate: logits[candidate])
        return {best: 1.0}
    logits = {candidate: value / temperature for candidate, value in logits.items()}
    if top_k and top_k < len(logits):
        kth = sorted(logits.values(), reverse=True)[top_k - 1]
        logits = {c: v for c, v in logits.items() if v >= kth}
    probs = _softmax(logits)
    if top_p < 1.0:
        cumulative = 0.0
        ascending = sorted(probs, key=lambda candidate: probs[candidate])
        for candidate in ascending[:-1]:
            cumulative += probs[candidate]
            if cumulative <= 1.0 - top_p:
                logits.pop(candidate)
        probs = _softmax(logits)
    return probs


def _softmax(logits: dict[int, float]) -> dict[int, float]:
    top = max(logits.values())
    weights = {c: math.exp(v - top) for c, v in logits.items()}
    total = sum(weights.values())
    return {c: w / total for c, w in weights.items()}


def fixed_target_greedy_ids(prompt: list[int], count: int, **controls) -> list[int]:
    """A greedy request's whole output under ``controls``, penalties included."""
    ids: list[int] = []
    token, position = prompt[-1], len(prompt) - 1
    while len(ids) < count:
        distribution = fixed_target_distribution(
            token, position, history=ids, prompt=prompt, temperature=0, **controls
        )
        token = next(iter(distribution))
        position += 1
        ids.append(token)
    return ids
