# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""The cohort's first tokens from one decode step: ``TT_SCHED_FINAL_STEP=1``.

Under ``TT_SCHED_DEFER_FINAL`` (``sched_defer_final``) a cohort's final chunks run one a
step, each a padded prefill call yielding its request's first token. Every decode
window but the last holds the finals after its own. With this flag a multi-chunk
request's bulk prefill runs to ``n - 1`` tokens
instead of the last chunk
boundary (its last chunk partial, one token short) and is presented to the generator as
a complete prompt of
``n - 1`` tokens, so the generator scores it, joins it into its decode slot as the stock
path does (no park, no
restore: the generator's ``TT_V41_DEFER_FINAL`` must be unset) and the runner discards
the chunk's logits (an
intermediate chunk to vLLM, one prompt token left). When the cohort closes, its parked
requests are scheduled
together in one step of one token each (their last prompt token at position ``n - 1``),
which the runner routes
to ``decode_forward``: one 32-row round yields the 32 first tokens, and the cohort's
decode continues from it.
The serial finals, their restores and the host between them are gone; the partial
chunk's device time moves into
the bulk phase. A one-chunk prompt keeps the stock path (its prefill is its final).
Unset, nothing changes.

The rule the scheduler and the runner share: a request whose prompt is longer than the
chunk (the token budget)
is a *stepped* request; its bulk ends at ``num_prompt_tokens - 1`` and the generator
sees that length.
"""

from __future__ import annotations

import os

ENV = "TT_SCHED_FINAL_STEP"
MODEL_ENV = "TT_V41_DEFER_FINAL"  # the generator's parking: must be off under this mode


def enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    v = env.get(ENV, "0").strip()
    if v not in ("0", "1", ""):
        raise ValueError(f"{ENV}={v!r}: 1 or unset")
    return v == "1"


def check_agreement(environ=None, defer_enabled: bool | None = None) -> None:
    """Refuse at start-up a configuration the generator would serve wrong: this mode
    needs the deferred-final
    scheduling (the cohorts) and the generator's parking off (a parked request is never
    restored here)."""
    env = os.environ if environ is None else environ
    if not enabled(env):
        return
    if defer_enabled is False:
        raise ValueError(f"{ENV} needs TT_SCHED_DEFER_FINAL (the cohorts it steps)")
    if env.get(MODEL_ENV, "0").strip() == "1":
        raise ValueError(
            f"{ENV}=1 with {MODEL_ENV}=1: the generator must not park "
            f"(unset {MODEL_ENV})"
        )


def stepped(num_prompt_tokens: int, chunk: int) -> bool:
    """Whether a prompt of that length takes the stepped path (its bulk to n - 1, its
    first token from a decode
    step): every multi-chunk prompt; a one-chunk prompt's prefill is its final."""
    return num_prompt_tokens > chunk


def bulk_end(num_prompt_tokens: int, chunk: int) -> int:
    """The last position the bulk prefill computes (exclusive): n - 1 for a stepped
    prompt."""
    return (
        num_prompt_tokens - 1
        if stepped(num_prompt_tokens, chunk)
        else num_prompt_tokens
    )


def is_final_step(rows: list[tuple[int, int, int]], chunk: int) -> bool:
    """Whether a prefill-looking step is a cohort's final step: every row ``(computed,
    prompt, scheduled)`` still
    prefilling is a stepped prompt with exactly its last token scheduled. A step that
    mixes such rows with
    others is a scheduler defect and is refused: the generator would assert on an
    unaligned chunk or serve a
    wrong prefix."""
    final = other = 0
    for computed, prompt, scheduled in rows:
        if (
            computed < prompt
            and scheduled == 1
            and computed == prompt - 1
            and stepped(prompt, chunk)
        ):
            final += 1
        else:
            other += 1
    if final and other:
        raise AssertionError(
            f"{ENV}: {final} final-step row(s) with {other} other row(s)"
        )
    return final > 0
