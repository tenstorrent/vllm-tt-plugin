# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ``topk`` accept mode samples on the exact top-M support as ``logits`` does.

A ``topk`` verify returns, per candidate column, the M largest raw logits
and their token ids in the support table's tie order (value descending, token
id ascending) plus the verify argmax and a lazy full-logits fetch. These tests
hold the support walk to the same walk over the table built from the full
logits (``_constrained_table``): the same probability table for every
position the support can decide, the same committed tokens and the same
consumption of every generator, bit for bit; one full-logits fetch for the
positions it cannot decide; and refusal of every row the runner must not
route here. The runner tests hold ``_spec_mode_for_step`` to the eligibility
rules and the flags, and drive a seeded sampled request through the topk mode
twice to the same output.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from vllm_tt_plugin.async_decode import _verify_output_tensor
from vllm_tt_plugin.spec_accept import (
    SpecPenalties,
    SpecSamplingInputs,
    SpecTokenFilters,
    _commit_walks,
    _constrained_table,
    _FetchOnce,
    _greedy_walk,
    _support_table,
    _walk_table,
    accept_sampled_support,
    device_topk_enabled,
    params_support_servable,
)
from vllm_tt_plugin.spec_admission import resolve_speculative_plan
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    ACCEPT_MODE_TOPK,
    DRAFTER_STATE_INTERNAL,
    MODE_REQUIRED_FIELDS,
    PLACEHOLDER_TOKEN_ID,
    SpecPlan,
    VerifyOutput,
)

from .sampled_target import SampledTarget
from .test_spec_sampled_runner import (
    DRAFT_LEN,
    GREEDY,
    MAX_NUM_REQS,
    _add_request,
    _argmax_drafts,
    _build,
    _output,
    _runner,
    _step,
)

V = 4096
M = 64
K = 3

# region Helpers


def exact_support(logits: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference support: a full stable sort, (value desc, token id asc)."""
    rows, width, vocab = logits.shape
    values, ids = torch.sort(
        logits.reshape(-1, vocab), dim=1, descending=True, stable=True
    )
    return (
        values[:, :m].reshape(rows, width, m).contiguous(),
        ids[:, :m].reshape(rows, width, m).to(torch.int32),
    )


def tied_logits(seed: int, rows: int, width: int, vocab: int = V) -> torch.Tensor:
    """bf16-valued logits with ties inside the top-M and at its boundary."""
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(rows, width, vocab, generator=g) * 3).to(torch.bfloat16)
    flat = logits.reshape(-1, vocab)
    for row in range(flat.shape[0]):
        # A tie group of 3 near the top, and a group of 6 straddling the
        # support boundary of a top_k of 20 (these land inside the 64).
        top = torch.topk(flat[row].float(), 30).indices
        flat[row, top[2:5]] = flat[row, top[2]].clone()
        flat[row, top[18:24]] = flat[row, top[18]].clone()
    return logits.to(torch.float32)


def controls(rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row controls: a greedy row (2), top_k from 1 to 50, top_p on and off.

    top_k stays well below M: on bf16 logits two neighbours of the 64th
    candidate tie about half the time, so a top_k of M-1 would often be
    undecidable and take the reference path, which draws the same
    distribution through a different inverse CDF (see the M-1 test below).
    """
    if rows > 8:
        raise ValueError("at most 8 rows of controls")
    temperature = torch.tensor([0.6, 1.0, 0.0, 0.8, 1.3, 0.7, 0.5, 1.0])[:rows]
    top_k = torch.tensor([20, 1, 20, 50, 40, 5, 2, 30], dtype=torch.int32)[:rows]
    top_p = torch.tensor([0.95, 0.95, 1.0, 0.9, 1.0, 0.5, 0.99, 0.8])[:rows]
    return temperature, top_k, top_p


def drafts_for(logits: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Drafts from each column's top-3, so acceptances and rejections both happen."""
    rows, width, _ = logits.shape
    g = torch.Generator().manual_seed(seed + 2000)
    top3 = torch.topk(logits[:, : width - 1], 3, dim=-1).indices
    pick = torch.randint(0, 3, (rows, width - 1), generator=g)
    drafts = top3.gather(2, pick.unsqueeze(2)).squeeze(2).to(torch.int32)
    num_valid = torch.randint(0, width, (rows,), generator=g).to(torch.int32)
    return drafts, num_valid


def sampling_inputs(rows, temperature, top_k, top_p, seeds) -> SpecSamplingInputs:
    generators = {}
    for row, seed in seeds.items():
        generators[row] = torch.Generator().manual_seed(seed)
    return SpecSamplingInputs(
        vocab_size=V,
        temperature=temperature.clone(),
        top_k=top_k.clone(),
        top_p=None if top_p is None else top_p.clone(),
        generators=generators,
    )


def full_logits_walk(
    logits: torch.Tensor,
    drafts: torch.Tensor,
    num_valid: torch.Tensor,
    sampling: SpecSamplingInputs,
):
    """The support modes' walk over the table built from the full logits."""
    rows, width, vocab = logits.shape
    is_greedy = sampling.temperature <= 0.0
    greedy = _greedy_walk(logits, drafts, width - 1) if bool(is_greedy.any()) else None
    random = None
    if bool((~is_greedy).any()):
        table = _constrained_table(
            logits, sampling.temperature, sampling.top_k, sampling.top_p
        )
        random = _walk_table(
            table, drafts, num_valid, sampling.generators, rows, width - 1, vocab
        )
    return _commit_walks(is_greedy, greedy, random, num_valid, rows, width)


def generator_states(sampling: SpecSamplingInputs) -> dict[int, torch.Tensor]:
    return {row: g.get_state().clone() for row, g in sampling.generators.items()}


# endregion Helpers

# region The support table and the support walk against the full-logits walk


@pytest.mark.parametrize("seed", range(12))
def test_the_support_table_is_the_full_logits_table_bit_for_bit(seed):
    """Every sampled position's probabilities are torch.equal across the two tables."""
    rows, width = 8, K + 1
    logits = tied_logits(seed, rows, width)
    temperature, top_k, top_p = controls(rows)
    values, ids = exact_support(logits, M)

    reference = _constrained_table(logits, temperature, top_k, top_p)
    fetch = _FetchOnce(lambda: logits, rows, width, V)
    table = _support_table(values, ids, temperature, top_k, top_p, V, fetch)

    want, got = reference.to_dense(), table.to_dense()
    sampled = (temperature > 0).unsqueeze(1).expand(rows, width).reshape(-1)
    assert torch.equal(want[sampled], got[sampled])
    # The injected ties stay inside the 64 candidates, so nothing fell back.
    assert fetch.calls == 0


@pytest.mark.parametrize("seed", range(30))
def test_the_support_walk_commits_what_the_full_logits_walk_commits(seed):
    """Same tokens, same counts, same generator and global stream consumption."""
    rows, width = 8, K + 1
    logits = tied_logits(seed, rows, width)
    temperature, top_k, top_p = controls(rows)
    drafts, num_valid = drafts_for(logits, seed)
    seeds = {
        0: 11 + seed,
        3: 22 + seed,
        5: 33 + seed,
    }  # rows 2 is greedy; 5 may be draftless
    argmax_ids = logits.argmax(-1).to(torch.int32)
    values, ids = exact_support(logits, M)

    full = sampling_inputs(rows, temperature, top_k, top_p, seeds)
    torch.manual_seed(seed)
    reference = full_logits_walk(logits, drafts, num_valid, full)
    reference_rng = torch.get_rng_state()

    support = sampling_inputs(rows, temperature, top_k, top_p, seeds)
    torch.manual_seed(seed)
    fetch = _FetchOnce(lambda: logits, rows, width, V)
    result = accept_sampled_support(
        argmax_ids, values, ids, drafts, num_valid, support, fetch
    )

    assert fetch.calls == 0, "every position was decided from the support"
    assert torch.equal(result.committed_token_ids, reference.committed_token_ids)
    assert torch.equal(result.accepted_counts, reference.accepted_counts)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    for row, state in generator_states(full).items():
        assert torch.equal(generator_states(support)[row], state), row


def test_a_top_k_of_m_minus_one_is_served_when_the_boundary_does_not_tie():
    """Distinct values: the 64th candidate cannot tie with the 63rd; M-1 is decided."""
    rows, width = 2, K + 1
    g = torch.Generator().manual_seed(21)
    logits = torch.stack(
        [torch.randperm(V, generator=g).float() * 1e-2 for _ in range(rows * width)]
    ).reshape(rows, width, V)
    temperature = torch.tensor([1.0, 0.7])
    top_k = torch.tensor([M - 1, M - 1], dtype=torch.int32)
    top_p = torch.tensor([1.0, 0.9])
    drafts, num_valid = drafts_for(logits, 21)
    values, ids = exact_support(logits, M)
    seeds = {0: 1, 1: 2}
    torch.manual_seed(2)
    reference = full_logits_walk(
        logits,
        drafts,
        num_valid,
        sampling_inputs(rows, temperature, top_k, top_p, seeds),
    )
    torch.manual_seed(2)
    fetch = _FetchOnce(lambda: logits, rows, width, V)
    result = accept_sampled_support(
        logits.argmax(-1).to(torch.int32),
        values,
        ids,
        drafts,
        num_valid,
        sampling_inputs(rows, temperature, top_k, top_p, seeds),
        fetch,
    )
    assert fetch.calls == 0
    assert torch.equal(result.committed_token_ids, reference.committed_token_ids)
    assert torch.equal(result.accepted_counts, reference.accepted_counts)


def test_an_undecidable_position_fetches_the_full_logits_once_and_matches():
    """500 logits tie at row 0's k-th value: the support cannot hold them all."""
    rows, width = 3, K + 1
    logits = tied_logits(7, rows, width)
    logits[0, :, :500] = 10.0
    temperature = torch.tensor([1.0, 0.8, 0.6])
    top_k = torch.tensor([20, 20, 5], dtype=torch.int32)
    top_p = torch.tensor([1.0, 0.95, 0.9])
    drafts, num_valid = drafts_for(logits, 7)
    values, ids = exact_support(logits, M)
    seeds = {0: 5, 1: 6}

    torch.manual_seed(1)
    reference = full_logits_walk(
        logits,
        drafts,
        num_valid,
        sampling_inputs(rows, temperature, top_k, top_p, seeds),
    )
    torch.manual_seed(1)
    fetch = _FetchOnce(lambda: logits, rows, width, V)
    result = accept_sampled_support(
        logits.argmax(-1).to(torch.int32),
        values,
        ids,
        drafts,
        num_valid,
        sampling_inputs(rows, temperature, top_k, top_p, seeds),
        fetch,
    )

    assert fetch.calls == 1
    assert torch.equal(result.committed_token_ids, reference.committed_token_ids)
    assert torch.equal(result.accepted_counts, reference.accepted_counts)

    with pytest.raises(ValueError, match="need the full logits"):
        accept_sampled_support(
            logits.argmax(-1).to(torch.int32),
            values,
            ids,
            drafts,
            num_valid,
            sampling_inputs(rows, temperature, top_k, top_p, {}),
            None,
        )


def test_a_tie_group_ending_exactly_at_the_support_boundary_is_undecidable():
    """The M-th candidate tying with the k-th value is the overflow rule's edge."""
    rows, width = 1, 1
    logits = torch.full((rows, width, V), -5.0)
    logits[0, 0, :10] = 3.0  # above the k-th
    logits[0, 0, 10 : 10 + (M - 10) + 1] = (
        1.0  # the k-th value (k=15) through the (M+1)-th candidate
    )
    temperature = torch.tensor([1.0])
    top_k = torch.tensor([15], dtype=torch.int32)
    values, ids = exact_support(logits, M)
    fetch = _FetchOnce(lambda: logits, rows, width, V)
    table = _support_table(values, ids, temperature, top_k, None, V, fetch)
    assert fetch.calls == 1, "a tie reaching the last candidate must fall back"
    assert torch.allclose(
        table.to_dense(),
        _constrained_table(logits, temperature, top_k, None).to_dense(),
    )
    # Two tie members fewer: the group now ends at the 63rd candidate, the 64th
    # is strictly below the k-th value, and the column is decided from the
    # support. (One fewer is not enough: the rule needs the last candidate
    # below the k-th, since it cannot see past it.)
    logits[0, 0, M - 1 : M + 1] = -5.0
    values, ids = exact_support(logits, M)
    fetch = _FetchOnce(lambda: logits, rows, width, V)
    table = _support_table(values, ids, temperature, top_k, None, V, fetch)
    assert fetch.calls == 0
    assert torch.equal(
        table.to_dense(),
        _constrained_table(logits, temperature, top_k, None).to_dense(),
    )


def test_a_fetch_of_the_wrong_shape_is_refused_by_name():
    rows, width = 2, K + 1
    logits = tied_logits(3, rows, width)
    logits[0, :, :500] = 10.0
    values, ids = exact_support(logits, M)
    sampling = sampling_inputs(
        rows,
        torch.tensor([1.0, 1.0]),
        torch.tensor([20, 20], dtype=torch.int32),
        None,
        {},
    )
    drafts, num_valid = drafts_for(logits, 3)
    with pytest.raises(ValueError, match=r"fetch_logits must return .*\[B, 1\+K, V\]"):
        accept_sampled_support(
            logits.argmax(-1).to(torch.int32),
            values,
            ids,
            drafts,
            num_valid,
            sampling,
            lambda: logits[:, :, : V // 2],
        )


@pytest.mark.parametrize(
    "top_k, match",
    [
        (torch.tensor([20, 0, 20]), r"rows \[1\] sample with top_k \[4096\]"),
        (torch.tensor([20, M, 20]), r"rows \[1\] sample with top_k \[64\]"),
        (None, r"rows \[0, 1, 2\] sample with top_k None"),
    ],
)
def test_a_sampled_row_the_support_cannot_serve_is_refused(top_k, match):
    rows, width = 3, K + 1
    logits = tied_logits(5, rows, width)
    values, ids = exact_support(logits, M)
    drafts, num_valid = drafts_for(logits, 5)
    if top_k is not None:
        top_k = top_k.to(torch.int32)
        top_k[top_k == 0] = V  # how the batch stores a disabled top_k
    sampling = SpecSamplingInputs(
        vocab_size=V, temperature=torch.ones(rows), top_k=top_k, top_p=None
    )
    with pytest.raises(ValueError, match=match):
        accept_sampled_support(
            logits.argmax(-1).to(torch.int32), values, ids, drafts, num_valid, sampling
        )


def test_greedy_rows_with_any_top_k_commit_their_argmax_ids():
    """A greedy row's top_k is not read: it commits by id equality."""
    rows, width = 2, K + 1
    logits = tied_logits(9, rows, width)
    values, ids = exact_support(logits, M)
    argmax_ids = logits.argmax(-1).to(torch.int32)
    drafts = argmax_ids[:, :K].clone()
    drafts[1, 1] = (drafts[1, 1] + 1) % V
    num_valid = torch.tensor([K, K], dtype=torch.int32)
    sampling = SpecSamplingInputs(
        vocab_size=V,
        temperature=torch.zeros(rows),
        top_k=torch.tensor([V, 0], dtype=torch.int32),
        top_p=None,
    )
    result = accept_sampled_support(
        argmax_ids, values, ids, drafts, num_valid, sampling
    )
    assert result.accepted_counts.tolist() == [K + 1, 2]
    assert torch.equal(result.committed_token_ids[0], argmax_ids[0])
    assert result.committed_token_ids[1, :2].tolist() == argmax_ids[1, :2].tolist()


@pytest.mark.parametrize(
    "field, value",
    [
        (
            "penalties",
            SpecPenalties(
                prompt_token_ids=torch.zeros(2, 1, dtype=torch.int64),
                output_token_ids=[[], []],
                presence=torch.zeros(2),
                frequency=torch.zeros(2),
                repetition=torch.ones(2),
            ),
        ),
        ("filters", SpecTokenFilters(min_tokens={0: (3, (1,))})),
        ("num_logprobs", 0),
    ],
)
def test_controls_that_need_the_vocabulary_are_refused(field, value):
    rows, width = 2, K + 1
    logits = tied_logits(2, rows, width)
    values, ids = exact_support(logits, M)
    drafts, num_valid = drafts_for(logits, 2)
    sampling = SpecSamplingInputs(
        vocab_size=V,
        temperature=torch.ones(rows),
        top_k=torch.tensor([20, 20], dtype=torch.int32),
        top_p=None,
        **{field: value},
    )
    with pytest.raises(ValueError, match=f"handed \\['{field}'\\]"):
        accept_sampled_support(
            logits.argmax(-1).to(torch.int32), values, ids, drafts, num_valid, sampling
        )


@pytest.mark.parametrize(
    "env", [("VLLM_TT_SPEC_FAST_TOPK", "0"), ("VLLM_TT_SPEC_TIE_EXACT", "1")]
)
def test_the_support_walk_refuses_a_launch_without_the_fast_path(monkeypatch, env):
    monkeypatch.setenv(*env)
    rows, width = 1, K + 1
    logits = tied_logits(4, rows, width)
    values, ids = exact_support(logits, M)
    drafts, num_valid = drafts_for(logits, 4)
    sampling = SpecSamplingInputs(
        vocab_size=V,
        temperature=torch.ones(1),
        top_k=torch.tensor([20], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="must not ask for the topk mode"):
        accept_sampled_support(
            logits.argmax(-1).to(torch.int32), values, ids, drafts, num_valid, sampling
        )


def test_malformed_support_tensors_are_refused_by_name():
    rows, width = 2, K + 1
    logits = tied_logits(6, rows, width)
    values, ids = exact_support(logits, M)
    drafts, num_valid = drafts_for(logits, 6)
    sampling = SpecSamplingInputs(
        vocab_size=V,
        temperature=torch.ones(rows),
        top_k=torch.tensor([20, 20], dtype=torch.int32),
    )
    argmax_ids = logits.argmax(-1).to(torch.int32)
    with pytest.raises(ValueError, match="topk_values and topk_ids must both be"):
        accept_sampled_support(
            argmax_ids, values[:, :, :10], ids, drafts, num_valid, sampling
        )
    with pytest.raises(TypeError, match="topk_ids integer"):
        accept_sampled_support(
            argmax_ids, values, ids.float(), drafts, num_valid, sampling
        )
    bad = ids.clone()
    bad[0, 0, 0] = V
    with pytest.raises(ValueError, match="topk_ids must lie in"):
        accept_sampled_support(argmax_ids, values, bad, drafts, num_valid, sampling)


def test_the_audit_passes_on_a_matching_support_and_catches_a_wrong_one(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_TOPK_AUDIT", "1")
    rows, width = 2, K + 1
    logits = tied_logits(8, rows, width)
    values, ids = exact_support(logits, M)
    drafts, num_valid = drafts_for(logits, 8)
    temperature, top_k, top_p = controls(rows)
    argmax_ids = logits.argmax(-1).to(torch.int32)
    accept_sampled_support(
        argmax_ids,
        values,
        ids,
        drafts,
        num_valid,
        sampling_inputs(rows, temperature, top_k, top_p, {}),
        lambda: logits,
    )
    wrong = values.clone()
    wrong[0, 0, 0] += 1.0  # a value the full logits do not hold
    with pytest.raises(AssertionError, match="topk audit: row 0 column 0"):
        accept_sampled_support(
            argmax_ids,
            wrong,
            ids,
            drafts,
            num_valid,
            sampling_inputs(rows, temperature, top_k, top_p, {}),
            lambda: logits,
        )


# endregion

# region Eligibility and flags


def _params(**kw) -> SamplingParams:
    bad_words = kw.pop("bad_words_token_ids", None)
    params = SamplingParams(ignore_eos=True, max_tokens=64, **kw)
    if bad_words is not None:
        params._bad_words_token_ids = bad_words
    return params


@pytest.mark.parametrize(
    "params, output_len, servable",
    [
        (_params(temperature=1.0, top_k=20), 0, True),
        (_params(temperature=1.0, top_k=1), 0, True),
        (_params(temperature=1.0, top_k=M - 1, top_p=0.5), 0, True),
        (_params(temperature=0.0, top_k=20), 0, True),
        (_params(temperature=1.0, top_k=0), 0, False),
        (_params(temperature=1.0, top_k=-1), 0, False),
        (_params(temperature=1.0, top_k=M), 0, False),
        (_params(temperature=1.0, top_k=20, logprobs=0), 0, False),
        (_params(temperature=1.0, top_k=20, allowed_token_ids=[1, 2]), 0, False),
        (_params(temperature=1.0, top_k=20, bad_words_token_ids=[[3]]), 0, False),
        (
            _params(
                temperature=1.0,
                top_k=20,
                structured_outputs=StructuredOutputsParams(regex="a+"),
            ),
            0,
            False,
        ),
        (
            _params(temperature=1.0, top_k=20, min_tokens=5, stop_token_ids=[7]),
            2,
            False,
        ),
        (_params(temperature=1.0, top_k=20, min_tokens=5, stop_token_ids=[7]), 5, True),
    ],
)
def test_params_support_servable(params, output_len, servable):
    assert params_support_servable(params, output_len, M) is servable


@pytest.mark.parametrize(
    "env, enabled",
    [
        ({}, True),
        ({"VLLM_TT_SPEC_DEVICE_TOPK": "1"}, True),
        ({"VLLM_TT_SPEC_DEVICE_TOPK": "0"}, False),
        ({"VLLM_TT_SPEC_DEVICE_TOPK": "1", "VLLM_TT_SPEC_FAST_TOPK": "0"}, False),
        ({"VLLM_TT_SPEC_DEVICE_TOPK": "1", "VLLM_TT_SPEC_TIE_EXACT": "1"}, False),
    ],
)
def test_device_topk_enabled_reads_the_three_flags(monkeypatch, env, enabled):
    for name in (
        "VLLM_TT_SPEC_DEVICE_TOPK",
        "VLLM_TT_SPEC_FAST_TOPK",
        "VLLM_TT_SPEC_TIE_EXACT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert device_topk_enabled() is enabled


# endregion

# region Contract plumbing


def test_the_topk_mode_declares_its_fields_and_the_output_requires_them():
    assert MODE_REQUIRED_FIELDS[ACCEPT_MODE_TOPK] == (
        "argmax_ids",
        "topk_values",
        "topk_ids",
    )
    ids = torch.zeros(2, 4, dtype=torch.int32)
    values = torch.zeros(2, 4, 8)
    with pytest.raises(ValueError, match="missing \\['topk_values', 'topk_ids'\\]"):
        VerifyOutput(spec_mode=ACCEPT_MODE_TOPK, argmax_ids=ids)
    out = VerifyOutput(
        spec_mode=ACCEPT_MODE_TOPK,
        argmax_ids=ids,
        topk_values=values,
        topk_ids=ids.unsqueeze(2).expand(2, 4, 8),
    )
    assert out.fetch_logits is None
    assert _verify_output_tensor(out, "m", ACCEPT_MODE_TOPK) is out
    with pytest.raises(NotImplementedError, match="answered in 'topk'"):
        _verify_output_tensor(out, "m", ACCEPT_MODE_LOGITS)


def _plan(**overrides) -> SpecPlan:
    fields = dict(
        effective_k=3,
        lanes_per_request=1,
        extra_bytes_per_seq=0,
        extra_bytes_per_token=0,
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS),
        drafter_state=DRAFTER_STATE_INTERNAL,
    )
    fields.update(overrides)
    return SpecPlan(**fields)


def test_a_plan_offering_topk_declares_its_candidate_count():
    plan = _plan(
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS, ACCEPT_MODE_TOPK),
        topk_candidates=64,
    )
    assert plan.topk_candidates == 64
    with pytest.raises(
        ValueError, match="topk_candidates must be an integer of at least 2"
    ):
        _plan(accept_modes=(ACCEPT_MODE_LOGITS, ACCEPT_MODE_TOPK))
    with pytest.raises(
        ValueError, match="topk_candidates must be an integer of at least 2"
    ):
        _plan(accept_modes=(ACCEPT_MODE_LOGITS, ACCEPT_MODE_TOPK), topk_candidates=1)
    with pytest.raises(ValueError, match="meaningful only with 'topk'"):
        _plan(topk_candidates=64)
    assert _plan().topk_candidates is None


class _TopkTarget(SampledTarget):
    """``SampledTarget`` that also serves ``topk`` from the same rule logits."""

    accept_modes = (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS, ACCEPT_MODE_TOPK)
    topk_candidates = 8

    def __init__(self) -> None:
        super().__init__()
        self.fetches = 0

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs: int, requested_k: int):
        del vllm_config, max_num_seqs
        return SpecPlan(
            effective_k=requested_k,
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=cls.accept_modes,
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=cls.supports_narrow_decode,
            topk_candidates=cls.topk_candidates
            if ACCEPT_MODE_TOPK in cls.accept_modes
            else None,
        )

    def answer(self, spec_mode: str, logits: torch.Tensor) -> VerifyOutput:
        if spec_mode != ACCEPT_MODE_TOPK:
            return super().answer(spec_mode, logits)
        values, ids = exact_support(logits, self.topk_candidates)

        def fetch():
            self.fetches += 1
            return logits

        return VerifyOutput(
            spec_mode=spec_mode,
            argmax_ids=logits.argmax(dim=-1).to(torch.int32),
            topk_values=values,
            topk_ids=ids,
            fetch_logits=fetch,
        )


def _topk_runner(model=None):
    model = model or _TopkTarget()
    runner = _runner(model)
    runner._spec_plan = type(model).spec_plan(None, MAX_NUM_REQS, DRAFT_LEN)
    return model, runner


def test_admission_refuses_topk_without_logits():
    class _TopkOnly(_TopkTarget):
        accept_modes = (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_TOPK)

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="ngram", num_speculative_tokens=3, model=None
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        additional_config={},
    )
    with pytest.raises(ValueError, match="'topk' without 'logits'"):
        resolve_speculative_plan(config, _TopkOnly, _TopkOnly.model_capabilities, 1)
    plan = resolve_speculative_plan(
        config, _TopkTarget, _TopkTarget.model_capabilities, 1
    )
    assert plan.topk_candidates == 8


# endregion

# region The runner: which mode a verify asks for, and what it commits

SAMPLED = SamplingParams(temperature=1.0, top_k=5, seed=3, ignore_eos=True)


@pytest.fixture
def topk_on(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_DEVICE_TOPK", "1")
    monkeypatch.delenv("VLLM_TT_SPEC_FAST_TOPK", raising=False)
    monkeypatch.delenv("VLLM_TT_SPEC_TIE_EXACT", raising=False)


def _mode_for(runner, *req_ids) -> str:
    _step(runner, *req_ids)
    return _build(runner, *req_ids).spec_mode


def test_a_servable_sampled_row_makes_the_verify_ask_for_topk(topk_on):
    model, runner = _topk_runner()
    _add_request(runner, "s", SAMPLED)
    assert _mode_for(runner, "s") == ACCEPT_MODE_TOPK


def test_the_flag_off_keeps_the_verify_in_logits_mode(monkeypatch):
    monkeypatch.setenv("VLLM_TT_SPEC_DEVICE_TOPK", "0")
    model, runner = _topk_runner()
    _add_request(runner, "s", SAMPLED)
    assert _mode_for(runner, "s") == ACCEPT_MODE_LOGITS


def test_a_model_without_topk_is_asked_for_logits(topk_on):
    model, runner = _topk_runner(SampledTarget())
    _add_request(runner, "s", SAMPLED)
    assert _mode_for(runner, "s") == ACCEPT_MODE_LOGITS


def test_a_greedy_batch_still_asks_for_argmax_ids(topk_on):
    model, runner = _topk_runner()
    _add_request(runner, "g", GREEDY)
    assert _mode_for(runner, "g") == ACCEPT_MODE_ARGMAX_IDS


def test_a_greedy_row_beside_a_servable_sampled_row_keeps_topk(topk_on):
    model, runner = _topk_runner()
    _add_request(runner, "g", GREEDY, first_token=1)
    _add_request(runner, "s", SAMPLED, first_token=9)
    assert _mode_for(runner, "g", "s") == ACCEPT_MODE_TOPK


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=1.0, top_k=0, seed=3, ignore_eos=True),
        SamplingParams(temperature=1.0, top_k=8, seed=3, ignore_eos=True),
        SamplingParams(temperature=1.0, top_k=5, logprobs=1, ignore_eos=True),
        SamplingParams(temperature=1.0, top_k=5, presence_penalty=0.5, ignore_eos=True),
        SamplingParams(
            temperature=1.0, top_k=5, repetition_penalty=1.2, ignore_eos=True
        ),
        SamplingParams(
            temperature=1.0, top_k=5, allowed_token_ids=[1, 2, 3], ignore_eos=True
        ),
        SamplingParams(
            temperature=1.0,
            top_k=5,
            min_tokens=30,
            max_tokens=64,
            stop_token_ids=[2],
            ignore_eos=True,
        ),
    ],
    ids=[
        "top_k off",
        "top_k is M",
        "logprobs",
        "presence",
        "repetition",
        "allowlist",
        "min_tokens",
    ],
)
def test_one_row_the_support_cannot_serve_makes_the_whole_step_logits(topk_on, params):
    model, runner = _topk_runner()
    _add_request(runner, "s", SAMPLED, first_token=1)
    _add_request(runner, "x", params, first_token=9)
    assert _mode_for(runner, "s", "x") == ACCEPT_MODE_LOGITS


def _speculate(runner, req_id: str, count: int) -> tuple[list[int], list[int]]:
    widths = []
    while len(_output(runner, req_id)) < count:
        committed = _step(
            runner, req_id, drafts={req_id: _argmax_drafts(runner, req_id)}
        )
        widths.append(len(committed[req_id]))
    return _output(runner, req_id)[:count], widths


def test_topk_speculation_repeats_a_seeded_stream(monkeypatch):
    """The same seeded request, drafted the same way, twice in topk mode."""
    monkeypatch.setenv("VLLM_TT_SPEC_DEVICE_TOPK", "1")
    streams = []
    for _ in range(2):
        model, runner = _topk_runner()
        _add_request(
            runner,
            "s",
            SamplingParams(temperature=1.0, top_k=5, seed=77, ignore_eos=True),
        )
        stream, widths = _speculate(runner, "s", 40)
        assert set(model.verify_modes) == {ACCEPT_MODE_TOPK}
        assert model.fetches == 0, (
            "the rule's supports fit inside M; nothing should fall back"
        )
        assert max(widths) > 1 and min(widths) < DRAFT_LEN + 1
        streams.append(stream)
    assert streams[0] == streams[1]


def test_padding_rows_commit_the_verify_argmax_in_topk_mode(topk_on):
    model, runner = _topk_runner()
    _add_request(runner, "s", SAMPLED)
    _step(runner, "s")
    model_input = _build(runner, "s", drafts={"s": _argmax_drafts(runner, "s")})
    assert model_input.spec_mode == ACCEPT_MODE_TOPK
    rows, width = model_input.input_tokens.shape
    assert rows == MAX_NUM_REQS
    verify = model.decode_forward(
        model_input.input_tokens,
        model_input.input_positions,
        spec_mode=model_input.spec_mode,
        num_valid_drafts=model_input.num_valid_drafts,
        accepted_counts=model_input.accepted_counts,
    )
    committed, counts, logprobs = runner.walk_spec_acceptance(model_input, verify)
    assert logprobs is None
    assert counts[1:].tolist() == [1] * (rows - 1)
    assert committed[1:, 0].tolist() == verify.argmax_ids[1:, 0].tolist()
    assert bool((committed[1:, 1:] == PLACEHOLDER_TOKEN_ID).all())
    assert 1 <= int(counts[0]) <= width


def test_a_support_of_the_wrong_width_is_refused_by_name(topk_on):
    model, runner = _topk_runner()
    _add_request(runner, "s", SAMPLED)
    _step(runner, "s")
    model_input = _build(runner, "s", drafts={"s": _argmax_drafts(runner, "s")})
    verify = model.decode_forward(
        model_input.input_tokens,
        model_input.input_positions,
        spec_mode=model_input.spec_mode,
        num_valid_drafts=model_input.num_valid_drafts,
        accepted_counts=model_input.accepted_counts,
    )
    narrow = VerifyOutput(
        spec_mode=ACCEPT_MODE_TOPK,
        argmax_ids=verify.argmax_ids,
        topk_values=verify.topk_values[:, :, :4],
        topk_ids=verify.topk_ids[:, :, :4],
    )
    with pytest.raises(ValueError, match="topk_values of shape"):
        runner.walk_spec_acceptance(model_input, narrow)
    with pytest.raises(TypeError, match="must reach the walk as its VerifyOutput"):
        runner.walk_spec_acceptance(model_input, verify.topk_values)


def test_a_topk_verify_disables_device_sampling(topk_on):
    model, runner = _topk_runner()
    _add_request(runner, "s", SAMPLED)
    _step(runner, "s")
    model_input = _build(runner, "s", drafts={"s": _argmax_drafts(runner, "s")})
    assert model_input.spec_mode == ACCEPT_MODE_TOPK
    assert model_input.perform_device_sampling is False


# endregion
