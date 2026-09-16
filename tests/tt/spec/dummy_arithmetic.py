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
