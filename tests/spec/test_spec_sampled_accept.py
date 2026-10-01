# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The sampled accept walk preserves the ordinary sampler's distribution.

``accept_sampled_drafts`` is what a ``logits`` verify runs: the penalties per
candidate column against that column's own history, then temperature, min_p
and top-k/top-p, then rejection sampling. These tests hold it to two
references that do not share its code.

vLLM's own ``Sampler``, for a row that carries no drafts. Such a row commits
exactly one token, the bonus, and it must be the token an ordinary sampled step
draws: the same distribution, the same arithmetic, and the same consumption of
the request's generator. A seeded request then reads the same.

An independent small-vocabulary reference, for rows that carry drafts. The
reference enumerates the target's joint distribution over the first committed
tokens from a fixed causal table, applying the controls in the ordinary
sampler's order with its own arithmetic. A simulation that drafts, verifies and
commits step by step must reproduce that joint distribution, whatever the
drafter proposed.
"""

from __future__ import annotations

import math
from itertools import product

import pytest
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.sampler import Sampler

from vllm_tt_plugin.spec_accept import (
    SpecPenalties,
    SpecSamplingInputs,
    accept_sampled_drafts,
    accept_speculated_tokens,
    apply_speculative_penalties,
)
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

# region Helpers


def _min_p_processor(min_p: torch.Tensor) -> MinPLogitsProcessor:
    """vLLM's min_p processor holding ``min_p`` for a batch of that many rows."""
    processor = object.__new__(MinPLogitsProcessor)
    processor.min_p_count = int((min_p > 0).sum())
    processor.min_p = min_p.to(torch.float32).unsqueeze(1)
    return processor


def _ordinary_sample(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    generators: dict[int, torch.Generator],
    min_p: torch.Tensor | None = None,
    prompt_token_ids: torch.Tensor | None = None,
    output_token_ids: list[list[int]] | None = None,
    presence: torch.Tensor | None = None,
    frequency: torch.Tensor | None = None,
    repetition: torch.Tensor | None = None,
) -> torch.Tensor:
    """One ordinary sampled step through vLLM's ``Sampler``, as the runner builds it."""
    rows = int(logits.shape[0])
    no_penalties = presence is None
    zeros = torch.zeros(rows)
    return Sampler()(
        logits=logits.clone(),
        sampling_metadata=SamplingMetadata(
            temperature=temperature,
            all_greedy=bool((temperature == 0).all()),
            all_random=bool((temperature != 0).all()),
            top_p=top_p,
            top_k=top_k,
            generators=generators,
            max_num_logprobs=None,
            no_penalties=no_penalties,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=zeros if frequency is None else frequency,
            presence_penalties=zeros if presence is None else presence,
            repetition_penalties=torch.ones(rows) if repetition is None else repetition,
            output_token_ids=output_token_ids or [[] for _ in range(rows)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(
                [] if min_p is None else [_min_p_processor(min_p)]
            ),
        ),
    ).sampled_token_ids.view(-1)


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


# endregion Helpers

# region A row without drafts is an ordinary sampled step

VOCAB = 32


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize(
    "controls",
    [
        {"temperature": 1.0},
        {"temperature": 0.6, "top_k": 5},
        {"temperature": 1.3, "top_p": 0.8},
        {"temperature": 0.9, "top_k": 7, "top_p": 0.7},
        {"temperature": 1.0, "min_p": 0.2},
        {"temperature": 0.7, "top_k": 7, "min_p": 0.1},
    ],
    ids=["temperature", "top-k", "top-p", "top-k-and-top-p", "min-p", "min-p-order"],
)
def test_a_draftless_seeded_row_draws_the_ordinary_sampler_s_token(seed, controls):
    """Same token and same generator state as one ordinary sampled step.

    The bonus is drawn first and from the target at the row's own count, so a
    row with no drafts reduces to an ordinary sample. Exact equality, not a
    distribution: a seeded request must read the same whether or not it
    travelled in a verify that other rows drafted for.
    """
    torch.manual_seed(1000 + seed)
    rows, width = 3, 4
    logits = torch.randn(rows, width, VOCAB) * 2.0
    temperature = torch.full((rows,), controls["temperature"])
    top_k = torch.full((rows,), controls.get("top_k", VOCAB), dtype=torch.int32)
    top_p = torch.full((rows,), controls.get("top_p", 1.0))
    min_p = torch.full((rows,), controls["min_p"]) if "min_p" in controls else None

    walked_generators = {
        row: torch.Generator().manual_seed(seed * 10 + row) for row in range(rows)
    }
    walked = accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=torch.zeros(rows, width - 1, dtype=torch.int32),
        num_valid_drafts=torch.zeros(rows, dtype=torch.int32),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        generators=walked_generators,
    )

    ordinary_generators = {
        row: torch.Generator().manual_seed(seed * 10 + row) for row in range(rows)
    }
    ordinary = _ordinary_sample(
        logits[:, 0, :],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        generators=ordinary_generators,
    )

    assert walked.accepted_counts.tolist() == [1] * rows
    assert walked.committed_token_ids[:, 0].tolist() == ordinary.tolist()
    for row in range(rows):
        assert torch.equal(
            walked_generators[row].get_state(), ordinary_generators[row].get_state()
        ), f"row {row} advanced its generator differently from an ordinary step"


@pytest.mark.parametrize("seed", range(8))
def test_a_draftless_penalized_row_draws_the_ordinary_sampler_s_token(seed):
    """Column 0 is penalized against the committed history, as an ordinary step is."""
    torch.manual_seed(2000 + seed)
    rows, width = 2, 3
    logits = torch.randn(rows, width, VOCAB)
    history = [[1, 2, 2, 5], [7, 7, 7]]
    prompt = torch.tensor([[3, 4, VOCAB], [9, 1, 2]], dtype=torch.int64)
    presence = torch.tensor([0.5, 0.0])
    frequency = torch.tensor([0.3, 0.8])
    repetition = torch.tensor([1.4, 1.1])
    temperature = torch.tensor([1.0, 0.7])
    top_k = torch.full((rows,), VOCAB, dtype=torch.int32)
    top_p = torch.ones(rows)

    walked_generators = {
        row: torch.Generator().manual_seed(seed + row) for row in range(rows)
    }
    walked = accept_sampled_drafts(
        logits,
        torch.zeros(rows, width - 1, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        SpecSamplingInputs(
            vocab_size=VOCAB,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generators=walked_generators,
            penalties=SpecPenalties(
                prompt_token_ids=prompt,
                output_token_ids=history,
                presence=presence,
                frequency=frequency,
                repetition=repetition,
            ),
        ),
    )
    ordinary = _ordinary_sample(
        logits[:, 0, :],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generators={
            row: torch.Generator().manual_seed(seed + row) for row in range(rows)
        },
        prompt_token_ids=prompt,
        output_token_ids=history,
        presence=presence,
        frequency=frequency,
        repetition=repetition,
    )

    assert walked.committed_token_ids[:, 0].tolist() == ordinary.tolist()


def test_a_greedy_penalized_row_commits_the_penalized_argmax():
    """Temperature 0 with a penalty is greedy over the penalized logits.

    The greedy walk reads whatever logits it is handed, so the penalties have
    to be applied before it; the raw argmax here is a token the history
    penalizes below another.
    """
    logits = torch.full((1, 2, VOCAB), -5.0)
    logits[0, :, 4] = 2.0
    logits[0, :, 9] = 1.5
    result = accept_sampled_drafts(
        logits,
        torch.tensor([[9]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        SpecSamplingInputs(
            vocab_size=VOCAB,
            temperature=torch.zeros(1),
            penalties=SpecPenalties(
                prompt_token_ids=torch.tensor([[VOCAB]]),
                output_token_ids=[[4]],
                presence=torch.tensor([1.0]),
                frequency=torch.zeros(1),
                repetition=torch.ones(1),
            ),
        ),
    )
    # Token 4 is in the output, so 2.0 - 1.0 < 1.5: the draft 9 is the
    # penalized argmax and is accepted, and 9 then joins column 1's history.
    assert result.committed_token_ids.tolist() == [[9, 4]]


# endregion A row without drafts is an ordinary sampled step

# region The penalties read each column's own history


def test_each_column_is_penalized_after_the_drafts_before_it():
    """Column ``j`` reads the committed output plus drafts ``0..j-1``.

    The bonus column reads every valid draft and a column past the row's count
    reads no padding, so a placeholder never enters a history.
    """
    rows, width = 2, 4
    logits = torch.randn(rows, width, VOCAB)
    drafts = torch.tensor(
        [[5, 6, 7], [8, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]], dtype=torch.int32
    )
    num_valid = torch.tensor([3, 1], dtype=torch.int32)
    penalties = SpecPenalties(
        prompt_token_ids=torch.tensor([[1, 2], [3, VOCAB]]),
        output_token_ids=[[5], []],
        presence=torch.tensor([0.4, 0.9]),
        frequency=torch.tensor([0.2, 0.1]),
        repetition=torch.tensor([1.3, 1.2]),
    )

    penalized = apply_speculative_penalties(logits, drafts, num_valid, penalties)

    expected_histories = [
        [[5], [5, 5], [5, 5, 6], [5, 5, 6, 7]],
        [[], [8], [8], [8]],
    ]
    for row in range(rows):
        for column in range(width):
            expected = apply_all_penalties(
                logits[row : row + 1, column].clone(),
                penalties.prompt_token_ids[row : row + 1],
                penalties.presence[row : row + 1],
                penalties.frequency[row : row + 1],
                penalties.repetition[row : row + 1],
                [expected_histories[row][column]],
            )
            assert torch.equal(penalized[row, column], expected[0]), (row, column)


def test_the_penalties_do_not_modify_the_caller_s_logits():
    logits = torch.randn(1, 2, VOCAB)
    before = logits.clone()
    apply_speculative_penalties(
        logits,
        torch.tensor([[3]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        SpecPenalties(
            prompt_token_ids=torch.tensor([[3]]),
            output_token_ids=[[3]],
            presence=torch.ones(1),
            frequency=torch.ones(1),
            repetition=torch.tensor([2.0]),
        ),
    )
    assert torch.equal(logits, before)


# endregion The penalties read each column's own history

# region Structure


def test_a_draft_the_target_rules_out_is_always_rejected():
    """A zero-probability draft rejects, and the correction is a supported token.

    The accept test is ``p / q >= u`` with ``q = 1`` for a deterministic
    drafter, so ``p = 0`` must reject whatever ``u`` is drawn, and the residual
    is the target with the draft removed, which holds no ruled-out token.
    """
    torch.manual_seed(7)
    rows = 4096
    logits = torch.full((rows, 2, 8), float("-inf"))
    logits[:, :, 1] = 0.0
    logits[:, :, 2] = -0.5
    result = accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=torch.full((rows, 1), 5, dtype=torch.int32),
        num_valid_drafts=torch.ones(rows, dtype=torch.int32),
        temperature=torch.ones(rows),
    )
    assert result.accepted_counts.tolist() == [1] * rows
    assert set(result.committed_token_ids[:, 0].tolist()) == {1, 2}


def test_a_draft_outside_the_vocabulary_is_refused_before_the_penalties():
    """The named refusal, not an index error from inside vLLM's penalty op."""
    with pytest.raises(ValueError, match="Offending"):
        accept_sampled_drafts(
            torch.zeros(1, 2, VOCAB),
            torch.tensor([[VOCAB + 5]], dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            SpecSamplingInputs(
                vocab_size=VOCAB,
                temperature=torch.ones(1),
                penalties=SpecPenalties(
                    prompt_token_ids=torch.tensor([[1]]),
                    output_token_ids=[[1]],
                    presence=torch.ones(1),
                    frequency=torch.zeros(1),
                    repetition=torch.ones(1),
                ),
            ),
        )


def test_a_logits_tensor_of_the_wrong_vocabulary_width_is_refused():
    """A row-folded readback is refused before it is sampled as a vocabulary."""
    with pytest.raises(ValueError, match="whole vocabulary"):
        accept_sampled_drafts(
            torch.zeros(1, 2, 16),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            SpecSamplingInputs(vocab_size=32, temperature=torch.ones(1)),
        )


@pytest.mark.parametrize(
    "broken",
    [
        {"top_k": torch.full((2,), 4, dtype=torch.int32)},
        {"top_p": torch.ones(1)},
        {"prompt_token_ids": torch.tensor([[1, 2]])},
        {"output_token_ids": [[1]]},
        {"presence": torch.zeros(2)},
    ],
    ids=["top-k", "top-p", "prompt", "history", "presence"],
)
def test_a_control_with_the_wrong_row_count_is_refused_by_name(broken):
    """A short penalty prompt is not refused further down: it penalizes row 0."""
    rows = 3
    penalties = {
        "prompt_token_ids": torch.tensor([[1, 2]] * rows),
        "output_token_ids": [[1]] * rows,
        "presence": torch.zeros(rows),
        "frequency": torch.zeros(rows),
        "repetition": torch.full((rows,), 1.2),
    }
    controls = {"top_k": None, "top_p": None}
    for name, value in broken.items():
        (penalties if name in penalties else controls)[name] = value

    with pytest.raises(ValueError, match=next(iter(broken))):
        accept_sampled_drafts(
            torch.zeros(rows, 2, VOCAB),
            torch.zeros(rows, 1, dtype=torch.int32),
            torch.ones(rows, dtype=torch.int32),
            SpecSamplingInputs(
                vocab_size=VOCAB,
                temperature=torch.ones(rows),
                penalties=SpecPenalties(**penalties),
                **controls,
            ),
        )


# endregion Structure

# region The committed distribution, against an independent reference

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
    the output; then temperature, min_p, top-k and top-p, in that order.
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

    probs = softmax(logits)
    min_p = controls.get("min_p", 0.0)
    if min_p:
        threshold = max(probs) * min_p
        logits = [v if p >= threshold else float("-inf") for v, p in zip(logits, probs)]
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


def _reference_joint(table, controls, length):
    """Exact joint probability of every ``length``-token continuation."""
    joint = {}
    for sequence in product(range(SMALL_VOCAB), repeat=length):
        probability = 1.0
        token, position = PROMPT[-1], len(PROMPT) - 1
        history: list[int] = []
        for emitted in sequence:
            distribution = _reference_distribution(
                table[token, position].tolist(), history, PROMPT, controls
            )
            probability *= distribution[emitted]
            if probability == 0.0:
                break
            history.append(emitted)
            token, position = emitted, position + 1
        if probability > 0.0:
            joint[sequence] = probability
    return joint


def _simulate(table, controls, drafter, length, rows, k, seed):
    """Draft, verify and commit until every row has ``length`` tokens.

    Each step proposes up to ``k`` drafts per row with a deterministic drafter,
    builds the target's logits for every candidate column from the column's
    own input token and position, and walks acceptance. The draft count varies
    by row and by step so short prefixes, zero-draft rows and full blocks all
    occur in one batch.
    """
    torch.manual_seed(seed)
    outputs = [[] for _ in range(rows)]
    step = 0
    while min(len(out) for out in outputs) < length:
        last = torch.tensor(
            [out[-1] if out else PROMPT[-1] for out in outputs], dtype=torch.int64
        )
        start = torch.tensor(
            [len(PROMPT) - 1 + len(out) for out in outputs], dtype=torch.int64
        )
        num_valid = torch.tensor(
            [(row + step) % (k + 1) for row in range(rows)], dtype=torch.int32
        )
        drafts = torch.full((rows, k), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        token = last.clone()
        for column in range(k):
            token = drafter(token, start + column)
            drafts[:, column] = torch.where(
                column < num_valid,
                token.to(torch.int32),
                torch.tensor(PLACEHOLDER_TOKEN_ID),
            )
        inputs = torch.cat(
            [last.unsqueeze(1), drafts.to(torch.int64).clamp(min=0)], dim=1
        )
        positions = start.unsqueeze(1) + torch.arange(k + 1).unsqueeze(0)
        logits = table[inputs, positions.clamp(max=table.shape[1] - 1)]

        penalties = None
        if any(name in controls for name in ("presence", "frequency", "repetition")):
            penalties = SpecPenalties(
                prompt_token_ids=torch.tensor([PROMPT] * rows, dtype=torch.int64),
                output_token_ids=[list(out) for out in outputs],
                presence=torch.full((rows,), controls.get("presence", 0.0)),
                frequency=torch.full((rows,), controls.get("frequency", 0.0)),
                repetition=torch.full((rows,), controls.get("repetition", 1.0)),
            )
        result = accept_sampled_drafts(
            logits,
            drafts,
            num_valid,
            SpecSamplingInputs(
                vocab_size=SMALL_VOCAB,
                temperature=torch.full((rows,), controls.get("temperature", 1.0)),
                top_k=torch.full(
                    (rows,), controls.get("top_k", SMALL_VOCAB), dtype=torch.int32
                ),
                top_p=torch.full((rows,), controls.get("top_p", 1.0)),
                min_p=(
                    torch.full((rows,), controls["min_p"])
                    if "min_p" in controls
                    else None
                ),
                penalties=penalties,
            ),
        )
        for row in range(rows):
            count = int(result.accepted_counts[row])
            outputs[row].extend(result.committed_token_ids[row, :count].tolist())
        step += 1
    return [tuple(out[:length]) for out in outputs]


def _argmax_drafter(table):
    """Proposes the target's own unpenalized argmax, which is often accepted."""

    def drafter(token, position):
        return table[token, position.clamp(max=table.shape[1] - 1)].argmax(dim=-1)

    return drafter


def _wrong_drafter(token, position):
    """Proposes a token that ignores the target, which is often rejected."""
    return (token * 5 + position * 3 + 1) % SMALL_VOCAB


@pytest.mark.parametrize(
    "controls",
    [
        {"temperature": 1.0},
        {"temperature": 0.7, "top_k": 3},
        {"temperature": 1.2, "top_p": 0.8},
        {"temperature": 1.0, "min_p": 0.15},
        {"temperature": 1.0, "presence": 0.6, "frequency": 0.4},
        {"temperature": 0.8, "repetition": 1.6},
        {"temperature": 0.7, "top_k": 4, "min_p": 0.1},
    ],
    ids=[
        "temperature",
        "top-k",
        "top-p",
        "min-p",
        "presence-frequency",
        "repetition",
        "min-p-order",
    ],
)
@pytest.mark.parametrize("drafter_name", ["argmax", "wrong"])
def test_the_committed_sequences_follow_the_target_joint_distribution(
    controls, drafter_name
):
    """Next-token frequencies and the conditional continuation, jointly.

    Three committed tokens per row cover the first token, the token after it
    and the one after that, so a walk that is right about marginals but wrong
    about what follows an accepted or a corrected token fails too. The rows
    mix zero, partial and full draft counts in every step.
    """
    table = _causal_table(seed=11)
    drafter = _argmax_drafter(table) if drafter_name == "argmax" else _wrong_drafter
    rows = 24000
    sequences = _simulate(table, controls, drafter, length=3, rows=rows, k=3, seed=5)
    observed: dict = {}
    for sequence in sequences:
        observed[sequence] = observed.get(sequence, 0) + 1

    expected = _reference_joint(table, controls, length=3)
    statistic, df = _chi_square(observed, expected, rows)

    assert statistic < _chi_square_critical(df), (
        f"chi-square {statistic:.1f} over {df} degrees of freedom: the committed "
        f"sequences under {controls} with the {drafter_name} drafter are not "
        "distributed as the target"
    )


def _joint_statistic(sequences, controls) -> tuple[float, float]:
    table = _causal_table(seed=11)
    observed: dict = {}
    for sequence in sequences:
        observed[sequence] = observed.get(sequence, 0) + 1
    statistic, df = _chi_square(
        observed, _reference_joint(table, controls, 3), len(sequences)
    )
    return statistic, _chi_square_critical(df)


def test_the_joint_check_catches_an_argmax_commit_one_time_in_twenty():
    """Its power, against a bias much smaller than committing the argmax.

    One sequence in twenty is replaced by the greedy chain, which is what a
    walk that committed the argmax on some rarely taken path would produce.
    """
    table = _causal_table(seed=11)
    controls = {"temperature": 1.0}
    sequences = _simulate(table, controls, _argmax_drafter(table), 3, 24000, 3, 5)
    greedy = []
    token, position = PROMPT[-1], len(PROMPT) - 1
    for _ in range(3):
        token = int(table[token, position].argmax())
        greedy.append(token)
        position += 1
    biased = [tuple(greedy) if i % 20 == 0 else seq for i, seq in enumerate(sequences)]

    statistic, critical = _joint_statistic(biased, controls)
    assert statistic > critical, (statistic, critical)


def test_the_joint_check_catches_temperature_missing_from_drafted_columns(
    monkeypatch,
):
    """Its power, against a walk that tempers column 0 and no other.

    Column 0 alone is what a row without drafts reads, so the exact
    comparison with the ordinary sampler cannot see this; only the
    continuation after an accepted draft does.
    """
    import vllm_tt_plugin.spec_accept as spec_accept

    original = spec_accept._constrained_probs

    def column_zero_only(target_logits, temperature, top_k, top_p, min_p=None):
        tempered = original(target_logits, temperature, top_k, top_p, min_p)
        untempered = original(
            target_logits,
            torch.where(temperature > 0, torch.ones_like(temperature), temperature),
            top_k,
            top_p,
            min_p,
        )
        return torch.cat([tempered[:, :1], untempered[:, 1:]], dim=1)

    monkeypatch.setattr(spec_accept, "_constrained_probs", column_zero_only)
    table = _causal_table(seed=11)
    controls = {"temperature": 0.7}
    sequences = _simulate(table, controls, _argmax_drafter(table), 3, 24000, 3, 5)

    statistic, critical = _joint_statistic(sequences, controls)
    assert statistic > critical, (statistic, critical)


# endregion The committed distribution, against an independent reference
