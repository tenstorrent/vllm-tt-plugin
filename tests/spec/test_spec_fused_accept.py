# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ``fused_sample`` accept mode's host half: the runner's draws and its checks.

A model that accepts on the device walks with uniforms the runner drew, so the
runner's draws must be exactly the host walk's: ``predraw_uniforms`` is pinned
here to ``_point_mass_walk`` by walking the same block both ways, from copies
of the same generators and the same global stream. The walk the runner runs
over a fused verify only checks what came back, and those checks are pinned
too.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_tt_plugin.model_input import TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner, fused_spec_sampling
from vllm_tt_plugin.spec_accept import accept_speculated_tokens, predraw_uniforms
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_FUSED_SAMPLE,
    PLACEHOLDER_TOKEN_ID,
    VerifyOutput,
)

VOCAB = 50
K = 3


def _clone(generators: dict[int, torch.Generator]) -> dict[int, torch.Generator]:
    out = {}
    for row, generator in generators.items():
        out[row] = torch.Generator(device="cpu")
        out[row].set_state(generator.get_state())
    return out


def _case(seed: int, rows: int):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, K + 1, VOCAB, generator=g) * 3
    drafts = torch.randint(0, VOCAB, (rows, K), generator=g, dtype=torch.int32)
    # Drafts the target likes, so the walk accepts some.
    best = logits[:, :K].argmax(dim=-1).to(torch.int32)
    drafts = torch.where(torch.rand(rows, K, generator=g) < 0.6, best, drafts)
    num_valid = torch.randint(1, K + 1, (rows,), generator=g, dtype=torch.int32)
    return logits, drafts, num_valid


@pytest.mark.parametrize(
    "top_k, top_p",
    [(None, None), (None, 0.9), (5, None), (8, 0.8)],
    ids=["temperature", "nucleus", "top-k", "top-k-nucleus"],
)
@pytest.mark.parametrize("seed", range(6))
def test_predrawn_uniforms_commit_what_the_drawing_walk_commits(seed, top_k, top_p):
    rows = 5
    logits, drafts, num_valid = _case(seed, rows)
    temperature = torch.tensor([1.0, 0.0, 0.7, 1.3, 1.0])  # row 1 greedy
    # Rows 0 and 3 seeded; row 4 unseeded and draftless (its bonus is a uniform
    # from the global stream either way).
    num_valid[4] = 0
    generators = {0: torch.Generator().manual_seed(100 + seed)}
    generators[3] = torch.Generator().manual_seed(200 + seed)
    controls = {
        "top_k": None
        if top_k is None
        else torch.full((rows,), top_k, dtype=torch.int32),
        "top_p": None if top_p is None else torch.full((rows,), top_p),
    }

    torch.manual_seed(seed)
    copies = _clone(generators)
    drawn = accept_speculated_tokens(
        logits, drafts, num_valid, temperature, generators=generators, **controls
    )
    after_drawn = torch.get_rng_state()

    torch.manual_seed(seed)
    uniforms = predraw_uniforms(copies, rows, K, num_valid)
    assert torch.get_rng_state().equal(after_drawn), (
        "the global stream moved differently"
    )
    for row, generator in generators.items():
        assert generator.get_state().equal(copies[row].get_state()), row
    given = accept_speculated_tokens(
        logits, drafts, num_valid, temperature, uniforms=uniforms, **controls
    )
    assert torch.get_rng_state().equal(after_drawn), "a walk on given uniforms drew"

    assert given.accepted_counts.tolist() == drawn.accepted_counts.tolist()
    assert given.committed_token_ids.tolist() == drawn.committed_token_ids.tolist()


def test_the_table_layout_is_bonus_accepts_correction():
    generator = torch.Generator().manual_seed(5)
    reference = torch.Generator().manual_seed(5)
    torch.manual_seed(0)
    table = predraw_uniforms({0: generator}, 1, K, torch.tensor([2], dtype=torch.int32))
    own = torch.rand(4, dtype=torch.float64, generator=reference)
    # A seeded row draws its bonus, its two accept tests, then its correction.
    assert table[0, 0] == own[0] and table[0, 1:3].tolist() == own[1:3].tolist()
    assert table[0, K + 1] == own[3]
    assert table.shape == (1, K + 2) and table.dtype == torch.float64


def test_a_seeded_row_without_drafts_cannot_be_predrawn():
    with pytest.raises(ValueError, match="seeded rows \\[1\\]"):
        predraw_uniforms(
            {1: torch.Generator().manual_seed(0)},
            2,
            K,
            torch.tensor([2, 0], dtype=torch.int32),
        )


def test_given_uniforms_are_point_mass_only_and_shaped_per_row():
    logits, drafts, num_valid = _case(0, 2)
    temperature = torch.ones(2)
    with pytest.raises(ValueError, match="draft_probs must be None"):
        accept_speculated_tokens(
            logits,
            drafts,
            num_valid,
            temperature,
            draft_probs=torch.full((2, K, VOCAB), 1.0 / VOCAB),
            uniforms=torch.zeros(2, K + 2),
        )
    with pytest.raises(ValueError, match="uniforms must be"):
        accept_speculated_tokens(
            logits, drafts, num_valid, temperature, uniforms=torch.zeros(2, K + 1)
        )


# region The runner's side of a fused verify


def _model_input(row_req_ids, num_valid, drafts, temperature=None, generators=None):
    rows = len(row_req_ids)
    temperature = torch.zeros(rows) if temperature is None else temperature
    return SimpleNamespace(
        row_req_ids=row_req_ids,
        input_tokens=torch.zeros(rows, K + 1, dtype=torch.int32),
        num_valid_drafts=torch.tensor(num_valid, dtype=torch.int32),
        draft_token_ids=torch.tensor(drafts, dtype=torch.int32),
        max_num_logprobs=[None],
        tt_sampling_params=TTSamplingParams(
            temperature=temperature,
            top_k=torch.full((rows,), VOCAB, dtype=torch.int32),
            top_p=torch.full((rows,), 0.9),
        ),
        generators_list=[generators or {}],
    )


def _fused(committed, counts):
    return VerifyOutput(
        spec_mode=ACCEPT_MODE_FUSED_SAMPLE,
        accepted_token_ids=torch.tensor(committed, dtype=torch.int32),
        accepted_counts=torch.tensor(counts, dtype=torch.int32),
    )


P = PLACEHOLDER_TOKEN_ID
DRAFTS = [[4, 5, 6], [P, P, P]]


def test_a_fused_walk_passes_the_committed_block_on():
    walk = TTModelRunner._walk_fused_acceptance(
        _model_input(["a", None], [3, 0], DRAFTS),
        _fused([[4, 5, 9, P], [0, P, P, P]], [3, 1]),
    )
    assert walk.counts.tolist() == [3, 1] and walk.logprobs is None
    assert walk.committed[0].tolist() == [4, 5, 9, P]


@pytest.mark.parametrize(
    "committed, counts, match",
    [
        ([[4, 7, 9, P], [0, P, P, P]], [3, 1], "other than the draft"),
        ([[4, 5, 6, 8], [0, 1, P, P]], [4, 2], "more than 1 \\+ num_valid_drafts"),
        ([[4, 5, P, P], [0, P, P, P]], [3, 1], "PLACEHOLDER_TOKEN_ID inside"),
        ([[4, 5, 9], [0, P, P]], [3, 1], "every submitted row"),
    ],
    ids=["not-the-draft", "past-the-drafts", "count-and-ids-disagree", "narrow"],
)
def test_a_fused_walk_refuses_a_block_the_drafts_cannot_explain(
    committed, counts, match
):
    with pytest.raises(ValueError, match=match):
        TTModelRunner._walk_fused_acceptance(
            _model_input(["a", None], [3, 0], DRAFTS), _fused(committed, counts)
        )


def test_a_fused_walk_refuses_another_mode_and_logprobs():
    model_input = _model_input(["a", None], [3, 0], DRAFTS)
    with pytest.raises(TypeError, match="VerifyOutput"):
        TTModelRunner._walk_fused_acceptance(model_input, torch.zeros(2, K + 1))
    model_input.max_num_logprobs = [1]
    with pytest.raises(RuntimeError, match="logprobs"):
        TTModelRunner._walk_fused_acceptance(
            model_input, _fused([[4, 5, 9, P], [0, P, P, P]], [3, 1])
        )


def test_an_all_greedy_step_sends_and_draws_nothing():
    rng = torch.get_rng_state()
    assert fused_spec_sampling(_model_input(["a", None], [3, 0], DRAFTS)) is None
    assert torch.get_rng_state().equal(rng)


def test_spec_sampling_covers_every_row_and_draws_for_the_live_ones():
    generator = torch.Generator().manual_seed(1)
    copy = torch.Generator().manual_seed(1)
    model_input = _model_input(
        [None, "s", "g"],
        [0, 2, 1],
        [[P, P, P], [3, 4, P], [8, P, P]],
        temperature=torch.tensor([0.0, 0.8, 0.0]),
        generators={1: generator},
    )
    torch.manual_seed(3)
    sampling = fused_spec_sampling(model_input)
    torch.manual_seed(3)
    # The live rows, in row order: what the logits walk would be handed.
    expected = predraw_uniforms(
        {0: copy}, 2, K, torch.tensor([2, 1], dtype=torch.int32)
    )

    assert sampling["greedy"].tolist() == [True, False, True]
    assert sampling["temperature"].tolist() == pytest.approx([0.0, 0.8, 0.0])
    assert sampling["top_k"].tolist() == [1, VOCAB, 1]
    assert sampling["top_p"].tolist() == pytest.approx([1.0, 0.9, 1.0])
    assert sampling["uniforms"].dtype == torch.float32
    assert sampling["uniforms"][1:].tolist() == expected.to(torch.float32).tolist()
    assert not sampling["uniforms"][0].any()
    assert generator.get_state().equal(copy.get_state())


def test_a_front_packed_step_is_accepted_on_device_as_the_logits_walk_would():
    """The front-packed synchronous path chooses and commits as the lane path.

    A seeded nucleus request and an unseeded top-k one: each fused step commits
    what the logits walk commits on the same block from the same generator and
    global stream, and leaves both where that walk leaves them. The first step
    carries no drafts, and the seeded row then needs its own race: ``logits``.
    """
    from vllm.sampling_params import SamplingParams

    from . import test_spec_lane_e2e as lane
    from . import test_spec_sampled_runner as harness
    from .sampled_target import SAMPLED_VOCAB_SIZE, rule_logits

    model = lane._fused_target()
    runner = harness._runner(model)
    nucleus = SamplingParams(temperature=1.0, top_p=0.95, seed=11, ignore_eos=True)
    harness._add_request(runner, "s", nucleus)
    harness._add_request(runner, "k", SamplingParams(temperature=0.7, top_k=3), 9)
    harness._step(runner, "s", "k")
    for _ in range(8):
        drafts = {r: harness._argmax_drafts(runner, r) for r in ("s", "k")}
        tails = [harness._tail(runner, r) for r in ("s", "k")]
        copy = _clone({0: runner.requests["s"].generator})
        rng = torch.get_rng_state()
        committed = harness._step(runner, "s", "k", drafts=drafts)
        after = torch.get_rng_state()

        tokens = torch.tensor([[t, *drafts[r]] for (t, _), r in zip(tails, "sk")])
        positions = torch.tensor([[p + j for j in range(K + 1)] for _, p in tails])
        torch.set_rng_state(rng)
        reference = accept_speculated_tokens(
            rule_logits(tokens, positions),
            tokens[:, 1:].to(torch.int32),
            torch.full((2,), K, dtype=torch.int32),
            torch.tensor([1.0, 0.7]),
            top_k=torch.tensor([SAMPLED_VOCAB_SIZE, 3], dtype=torch.int32),
            top_p=torch.tensor([0.95, 1.0]),
            generators=copy,
        )
        assert torch.get_rng_state().equal(after)
        assert copy[0].get_state().equal(runner.requests["s"].generator.get_state())
        assert committed == {
            r: reference.committed_token_ids[
                i, : int(reference.accepted_counts[i])
            ].tolist()
            for i, r in enumerate(("s", "k"))
        }
    assert model.verify_modes[0] == "logits"
    assert set(model.verify_modes[1:]) == {ACCEPT_MODE_FUSED_SAMPLE}


# endregion The runner's side of a fused verify
