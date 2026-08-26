# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Scheduler-driven chunked prefill against a live TT server.

Requires a server started with ``--enable-chunked-prefill`` and run with
``--tt-chunked-prefill-budget=<max_num_batched_tokens>``; every test here skips
without it.

The failure mode is silent. A resume offset the model cannot honour makes the
request attend the wrong prefix, and nothing raises: throughput checks pass and
the response is fluent. So each prompt carries a passphrase in its own body and
is asked to repeat it, which fails loudly when part of that prompt's K/V is
wrong.

Concurrency is the point, not prompt length. A single prefill in flight gets the
whole token budget every step, so its boundaries are the well-behaved multiples.
Boundaries only go ragged when several prefills share a step and one is admitted
into what is left of the budget, which is when a misaligned resume offset reaches
the model. A block-size-aligned-but-not-q_chunk-aligned offset scored 10/20 here
against 20/20 with chunked prefill off.
"""

import random

import pytest

from tests.tt.utils import RequestConfig, run_concurrent_batch

# The passphrase sits this far into the filler, so a prompt split into several
# chunks plants it past a boundary and recalling it needs K/V written at a
# nonzero chunk_start_idx to be correct.
NEEDLE_POSITION = 0.75

# Four prompts of roughly one budget each overflow every step, so each step
# admits someone into a ragged remainder. Three rounds is twelve samples, and a
# per-request failure rate of ~50% (what the misaligned-offset bug produced)
# survives that with probability 2**-12.
CONCURRENCY = 4
ROUNDS = 3

# Roughly one token per filler word for every tokenizer in scope. Prompts are
# sized in words rather than measured, so the test needs no tokenizer.
FILLER_WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliett",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
]


def _needle_prompt(num_words: int, passphrase: str, seed: int, nonce: str) -> str:
    """Filler with ``passphrase`` planted inside it and a question about it.

    ``nonce`` keeps every prompt distinct so prefix caching never serves one
    request from another's blocks, which would mask exactly what is under test.
    """
    rng = random.Random(seed)
    words = [rng.choice(FILLER_WORDS) for _ in range(num_words)]
    at = int(num_words * NEEDLE_POSITION)
    words[at:at] = f"The secret passphrase is {passphrase}.".split()
    return (
        f"{nonce}Read the following text and answer the question at the end.\n\n"
        + " ".join(words)
        + "\n\nQuestion: what is the secret passphrase?"
        + "\nAnswer: the secret passphrase is"
    )


def _recalled(output: str | None, passphrase: str) -> bool:
    return passphrase.lower() in (output or "").lower()


@pytest.fixture(scope="module")
def budget(chunked_prefill_budget):
    """The served ``max_num_batched_tokens``.

    This is a claim by whoever launched the test, not a fact read back from the
    server, and the tests cannot check it themselves. Nothing client-visible
    distinguishes a split prefill from an unsplit one: the scheduler config is
    absent from ``/metrics``, and an intermediate chunk emits no token, so it
    raises no iteration stats either. Passphrase recall also succeeds perfectly
    well on an unsplit prefill, so against a server that silently disabled
    chunked prefill every test here would pass while testing nothing.

    The workflow closes that hole instead, by requiring the server log to report
    chunked prefill enabled at this budget before pytest runs. Anyone running
    this by hand should check the same line.
    """
    if chunked_prefill_budget <= 0:
        pytest.skip(
            "Needs a server with chunked prefill enabled; pass "
            "--tt-chunked-prefill-budget=<max_num_batched_tokens>"
        )
    return chunked_prefill_budget


def test_a_solo_split_prefill_recalls_its_needle(tt_server, tt_model_name, budget):
    """One prompt, several engine steps, nothing else in flight.

    Every boundary is the whole budget here, so this is the well-behaved case. A
    failure means multi-step prefill is broken outright, before any question of
    which offsets the model can honour.
    """
    passphrase = "cobalt-heron-42"
    prompt = _needle_prompt(budget, passphrase, seed=0xC401, nonce="Solo.\n")

    (output,) = run_concurrent_batch(
        tt_server,
        tt_model_name,
        [RequestConfig(prompt=prompt, max_tokens=24, temperature=0)],
    )

    assert _recalled(output, passphrase), (
        f"a solo split prefill lost the passphrase planted {NEEDLE_POSITION:.0%} "
        f"into its prompt: want {passphrase!r}, got {output!r}"
    )


def test_prefills_sharing_a_step_each_recall_their_own_needle(
    tt_server, tt_model_name, budget
):
    """Several prompts that together overflow one step's token budget.

    A request admitted into the remainder of a step gets whatever tokens were
    left, which is a multiple of nothing, and resumes from there on the next
    step. Every prompt carries its own passphrase, so a request reading the
    wrong prefix shows up as that request answering wrongly rather than as a
    change in aggregate throughput.
    """
    misses: list[str] = []
    total = 0

    for rnd in range(ROUNDS):
        passphrases = [f"cobalt-heron-{rnd}-{i}" for i in range(CONCURRENCY)]
        configs = [
            RequestConfig(
                prompt=_needle_prompt(
                    budget, passphrase, seed=0xC401 + i, nonce=f"Round {rnd} doc {i}.\n"
                ),
                max_tokens=24,
                temperature=0,
            )
            for i, passphrase in enumerate(passphrases)
        ]

        outputs = run_concurrent_batch(tt_server, tt_model_name, configs)
        total += len(outputs)
        for i, (passphrase, output) in enumerate(zip(passphrases, outputs)):
            if not _recalled(output, passphrase):
                misses.append(
                    f"round {rnd} request {i}: want {passphrase!r}, got {output!r}"
                )

    assert not misses, (
        f"{len(misses)} of {total} concurrent split prefills answered with a "
        "passphrase that is not the one planted in their own prompt, so they "
        "attended the wrong prefix:\n  " + "\n  ".join(misses)
    )
