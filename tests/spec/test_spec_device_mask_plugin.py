# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device token masks of the fused verify, on the host.

A ``fused_sample`` model that declares ``supports_spec_device_mask`` applies one
packed token mask per row and candidate column inside its step. The plugin
composes that mask (``spec_masks``) from the grammar rows the scheduler filled,
``allowed_token_ids``, ``bad_words`` and ``min_tokens``, and submits such a step
in two halves: ``decode_forward_begin`` at ``execute_model`` and
``decode_forward_finish(mask)`` at ``sample_tokens``. These tests pin the plugin
side:

- the composed mask equals what the plugin's own ``logits`` walk applies
  (``apply_speculative_token_filters`` and ``_apply_grammar_bitmask``), on random
  logits, including multi-token bad words, ``min_tokens`` and a narrow grammar
  bitmask;
- the real xgrammar rows of vLLM's ``StructuredOutputManager`` (offline, the
  Qwen3.8 tokenizer) give the same commit through the mask as through the
  ``logits`` walk, for valid drafts, a first or a middle invalid draft, padding
  and the reasoning boundary;
- the predicate and the mode the runner asks for, with the capability on and off;
- the step flow: begin at ``execute_model``, finish at ``sample_tokens``, no mask
  no change, never chained, and what the fake model saw.

The capability is one runner attribute (``_spec_device_mask``), which
``load_model`` sets from ``model_capabilities``; the default is False and then
nothing changes (the rest of ``tests/spec`` is that regression).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner, fused_spec_sampling
from vllm_tt_plugin.spec_accept import (
    SpecSamplingInputs,
    SpecTokenFilters,
    _apply_grammar_bitmask,
    accept_sampled_drafts,
    accept_sampled_support,
    apply_speculative_token_filters,
    has_mask_control,
    mask_row_refusal,
    params_fused_servable,
)
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_FUSED_SAMPLE,
    ACCEPT_MODE_LOGITS,
    SpecCandidates,
    VerifyOutput,
)
from vllm_tt_plugin.spec_device_logprobs import device_spec_logprobs
from vllm_tt_plugin.spec_device_penalties import SpecPenaltyTracker
from vllm_tt_plugin.spec_masks import (
    MASK_VALUE_FLOOR,
    MaskRowSpec,
    SpecMaskCache,
    SpecMaskPlan,
    StaticMasks,
    all_ones,
    bad_word_last_token,
    clear_ids_,
    compose_step_mask,
    count_mask_fallback,
    count_mask_step,
    device_mask_stats,
    empty_rows,
    is_unconstrained,
    mask_plan_for_step,
    mask_words,
    normalise_width,
    pack_allowed_ids,
    reset_device_mask_stats,
    static_masks,
    step_mask,
)
from vllm_tt_plugin.structured_output import spec_grammar_bitmask_for_tt_batch

from . import test_spec_sampled_controls_runner as ctl
from . import test_spec_sampled_runner as sync_harness
from .sampled_target import SAMPLED_VOCAB_SIZE, rule_logits
from .test_spec_async import BLOCK_SIZE, MAX_MODEL_LEN, MAX_NUM_REQS
from .test_spec_chain_async import Engine, _make_runner
from .test_spec_chain_async import K as CHAIN_K
from .test_spec_device_logprobs_plugin import LogprobsTarget, _candidates, _load
from .test_spec_device_topk import exact_support
from .test_spec_sampled_controls import _pack

V = 200  # not a multiple of 32: the last word has padding bits
W = mask_words(V)
WIDTH = 4  # 1 + K, K = 3


def unpack_allowed(mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """``[..., vocab_size]`` bool, True where the packed ``[..., W]`` mask allows it.

    A mask narrower than the vocabulary leaves the missing ids disallowed."""
    bits = torch.arange(32, dtype=torch.int32)
    allowed = ((mask.to(torch.int32).unsqueeze(-1) >> bits) & 1).bool()
    allowed = allowed.reshape(*mask.shape[:-1], -1)
    have = int(allowed.shape[-1])
    if have < vocab_size:
        pad = torch.zeros((*allowed.shape[:-1], vocab_size - have), dtype=torch.bool)
        allowed = torch.cat([allowed, pad], dim=-1)
    return allowed[..., :vocab_size]


@pytest.fixture(autouse=True)
def _stats():
    reset_device_mask_stats()


# region H1/H2: the composed mask equals the plugin's own filters


def _random_case(seed: int):
    """Random controls for 4 rows; returns the pieces both arms read."""
    gen = torch.Generator().manual_seed(seed)
    rows = 4
    logits = torch.randn(rows, WIDTH, V, generator=gen)
    drafts = torch.randint(0, V, (rows, WIDTH - 1), generator=gen).to(torch.int32)
    valid = torch.randint(0, WIDTH, (rows,), generator=gen).to(torch.int32)
    histories = [
        torch.randint(0, V, (int(n),), generator=gen).tolist()
        for n in torch.randint(0, 7, (rows,), generator=gen)
    ]
    specs, allowed_mask, bad_words, min_tokens = (
        [],
        torch.zeros(rows, V, dtype=torch.bool),
        {},
        {},
    )
    for row in range(rows):
        draft = [int(t) for t in drafts[row, : int(valid[row])]]
        allowed = None
        if int(torch.randint(0, 2, (1,), generator=gen)):
            allowed = torch.randperm(V, generator=gen)[
                : int(torch.randint(30, 120, (1,), generator=gen))
            ].tolist()
            allowed_mask[row] = True
            allowed_mask[row, allowed] = False
        words: list[list[int]] = []
        if int(torch.randint(0, 2, (1,), generator=gen)):
            column = int(torch.randint(0, WIDTH, (1,), generator=gen))
            history = histories[row] + draft[:column]
            ban = int(torch.randint(0, V, (1,), generator=gen))
            words.append(
                history[-2:] + [ban] if len(history) >= 2 else [ban]
            )  # bites at that column
            words.append([int(torch.randint(0, V, (1,), generator=gen))])  # one token
            words.append([5, 6, 7])  # a word that mostly does not match
            bad_words[row] = words
        remaining, stops = 0, []
        if int(torch.randint(0, 2, (1,), generator=gen)):
            remaining = int(torch.randint(0, 7, (1,), generator=gen))
            stops = torch.randperm(V, generator=gen)[:3].tolist()
            min_tokens[row] = (remaining, tuple(stops))
        specs.append(
            MaskRowSpec(
                req_id=f"r{row}",
                allowed_ids=allowed,
                bad_words=words,
                committed=histories[row],
                min_remaining=remaining if stops else 0,
                stop_ids=stops,
                drafts=draft,
            )
        )
    return logits, drafts, valid, histories, specs, allowed_mask, bad_words, min_tokens


def _random_grammar(seed: int, narrow: bool):
    """Scheduler rows: rows 0 and 2 constrained at some columns, the others all ones."""
    gen = torch.Generator().manual_seed(1000 + seed)
    allowed = torch.rand(4, WIDTH, V, generator=gen) < 0.6
    packed = _pack(allowed)
    packed[1] = -1
    packed[3] = -1
    packed[2, 1:] = -1  # only column 0 constrained
    if narrow:
        packed = packed[
            ..., : W - 1
        ].contiguous()  # ids of the last word are disallowed
    return packed


@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_h1_the_composed_mask_equals_the_reference_filters(seed, narrow):
    logits, drafts, valid, histories, specs, allowed_mask, bad_words, min_tokens = (
        _random_case(seed)
    )
    grammar = _random_grammar(seed, narrow)
    grammar_rows = (0, 2)

    # The logits walk: grammar first, then allowed ids, bad words, min_tokens.
    filters = SpecTokenFilters(
        allowed_token_ids_mask=allowed_mask,
        bad_words_token_ids=bad_words,
        output_token_ids=histories,
        min_tokens=min_tokens,
    )
    reference = apply_speculative_token_filters(
        _apply_grammar_bitmask(logits, grammar), drafts, valid, filters
    )

    cache = SpecMaskCache()
    static = static_masks(specs, cache, width=WIDTH, vocab_size=V)
    mask = compose_step_mask(
        static, grammar, grammar_rows, rows=4, width=WIDTH, vocab_size=V
    )
    ours = logits.masked_fill(~unpack_allowed(mask, V), float("-inf"))

    assert mask.dtype == torch.int32 and tuple(mask.shape) == (4, WIDTH, W)
    assert torch.equal(ours, reference)
    # empty_rows agrees with the logits: a row with nothing allowed is all -inf
    assert torch.equal(empty_rows(mask), torch.isneginf(reference).all(dim=-1))


def test_h1_an_unconstrained_row_stays_all_ones_and_padding_rows_too():
    specs = [MaskRowSpec(req_id="a"), MaskRowSpec(req_id="b", allowed_ids=[3, 4])]
    static = static_masks(specs, SpecMaskCache(), width=WIDTH, vocab_size=V)
    assert [spec.constrained for spec in specs] == [False, True]
    mask = compose_step_mask(static, None, (), rows=6, width=WIDTH, vocab_size=V)
    assert tuple(mask.shape) == (6, WIDTH, W)
    assert bool(is_unconstrained(mask[0]).all())
    assert bool(is_unconstrained(mask[2:]).all())
    assert unpack_allowed(mask[1, 0], V).nonzero().reshape(-1).tolist() == [3, 4]


@pytest.mark.parametrize("token", [0, 31, 32, 33, V - 1])
def test_h2_the_bit_of_a_token_is_bit_v_mod_32_of_word_v_div_32(token):
    row = pack_allowed_ids([token], V)
    expected = torch.zeros(W, dtype=torch.int64)
    expected[token // 32] = 1 << (token % 32)
    # int32 holds bit 31 as the sign bit
    expected = torch.where(expected >= 2**31, expected - 2**32, expected).to(
        torch.int32
    )
    assert torch.equal(row, expected)
    other = all_ones(words=W)
    clear_ids_(other, [token], V)
    assert int(unpack_allowed(other, V).sum()) == W * 32 - 1 - (W * 32 - V)


def test_h2_ids_outside_the_vocabulary_are_dropped_and_the_tail_bits_are_cleared():
    assert int(unpack_allowed(pack_allowed_ids([1, V + 5, -2], V), V).sum()) == 1
    static = static_masks(
        [MaskRowSpec(req_id="a", bad_words=[[7]])],
        SpecMaskCache(),
        width=1,
        vocab_size=V,
    )
    mask = compose_step_mask(static, None, (), rows=1, width=1, vocab_size=V)
    # the last word keeps its V % 32 low bits; the bits past the vocabulary are 0
    assert int(mask[0, 0, -1]) == (1 << (V % 32)) - 1
    assert int(unpack_allowed(mask, V).sum()) == V - 1


def test_h2_a_grammar_bitmask_narrower_than_the_vocabulary_disallows_the_missing_ids():
    grammar = torch.full((1, WIDTH, W - 2), -1, dtype=torch.int32)
    grammar[0, 0, 0] = 0b101
    mask = compose_step_mask(None, grammar, (0,), rows=1, width=WIDTH, vocab_size=V)
    allowed = unpack_allowed(mask[0, 0], V)
    assert allowed.nonzero().reshape(-1).tolist() == [0, 2] + list(
        range(32, (W - 2) * 32)
    )
    with pytest.raises(ValueError, match="wider"):
        normalise_width(torch.zeros(1, W + 1, dtype=torch.int32), W)


def test_the_static_row_of_a_request_is_cached_until_it_finishes():
    cache = SpecMaskCache()
    spec = MaskRowSpec(req_id="a", allowed_ids=[1, 2, 3], bad_words=[[2], [8, 9]])
    first = cache.static_row(spec, V)
    assert cache.static_row(spec, V) is first
    # the one-token bad word is in the row, the longer one is not
    assert unpack_allowed(first, V).nonzero().reshape(-1).tolist() == [1, 3]
    cache.drop("a")
    assert cache.static_row(spec, V) is not first
    # a row with neither control has no static row
    assert cache.static_row(MaskRowSpec(req_id="b", bad_words=[[8, 9]]), V) is None


def test_a_multi_token_bad_word_reads_the_output_and_the_drafts_before_the_column():
    word = [10, 11, 12]
    assert bad_word_last_token(word, [3, 10, 11]) == 12
    assert bad_word_last_token(word, [10, 11, 3]) is None
    assert bad_word_last_token([12], []) == 12  # one token: always
    assert bad_word_last_token(word, [11]) is None  # history shorter than the prefix
    spec = MaskRowSpec(
        req_id="a", bad_words=[word], committed=[4, 10], drafts=[11, 12, 13]
    )
    static = static_masks([spec], SpecMaskCache(), width=WIDTH, vocab_size=V)
    banned = [
        (~unpack_allowed(static.mask[0, column], V)).nonzero().reshape(-1).tolist()
        for column in range(WIDTH)
    ]
    # column 0: history [4, 10]; column 1: [.., 10, 11] bans 12; later: no
    assert banned == [[], [12], [], []]


# endregion

# region H11: min_tokens


@pytest.mark.parametrize("remaining, columns", [(0, 0), (1, 1), (3, 3), (4, 4), (9, 4)])
def test_h11_the_stop_ids_are_masked_while_the_column_is_below_the_remaining_count(
    remaining, columns
):
    spec = MaskRowSpec(
        req_id="a", min_remaining=remaining, stop_ids=[7, 9], drafts=[1, 2, 3]
    )
    static = static_masks([spec], SpecMaskCache(), width=WIDTH, vocab_size=V)
    for column in range(WIDTH):
        allowed = unpack_allowed(static.mask[0, column], V)
        assert bool(allowed[7]) is (column >= columns)
        assert bool(allowed[9]) is (column >= columns)
    assert spec.constrained is (remaining > 0)


def test_h11_a_row_with_no_stop_ids_or_no_drafts_is_handled_like_the_reference():
    plain = MaskRowSpec(req_id="a", min_remaining=5, stop_ids=[])
    assert not plain.constrained
    spec = MaskRowSpec(req_id="b", min_remaining=2, stop_ids=[4])  # no drafts
    static = static_masks([spec], SpecMaskCache(), width=WIDTH, vocab_size=V)
    assert [bool(unpack_allowed(static.mask[0, c], V)[4]) for c in range(WIDTH)] == [
        False,
        False,
        True,
        True,
    ]


def _state(params, output=()):
    return SimpleNamespace(sampling_params=params, output_token_ids=list(output))


def _params(**kw):
    kw.setdefault("ignore_eos", True)
    kw.setdefault("max_tokens", 64)
    kw.setdefault("temperature", 0.0)
    return SamplingParams(**kw)


def test_h11_the_plan_reads_the_remaining_count_off_the_committed_output():
    requests = {
        "a": _state(_params(min_tokens=5, stop_token_ids=[7]), output=[1, 2]),
        "b": _state(_params(min_tokens=5, stop_token_ids=[7]), output=[1, 2, 3, 4, 5]),
    }
    drafts = torch.zeros((2, 3), dtype=torch.int32)
    valid = torch.zeros(2, dtype=torch.int32)
    plan, refusal = mask_plan_for_step(
        requests,
        ["a", "b"],
        drafts,
        valid,
        (),
        SpecMaskCache(),
        width=WIDTH,
        vocab_size=V,
    )
    assert refusal is None and not plan.static.empty
    # remaining 3: columns 0..2 mask the stop id
    assert [
        bool(unpack_allowed(plan.static.mask[0, c], V)[7]) for c in range(WIDTH)
    ] == [
        False,
        False,
        False,
        True,
    ]


def test_h11_a_composed_row_with_no_token_left_is_refused_at_the_plan():
    """An allowlist inside the held-back stop ids leaves nothing: logits mode."""
    requests = {
        "a": _state(
            _params(min_tokens=9, stop_token_ids=[7, 8], allowed_token_ids=[7, 8])
        )
    }
    plan, refusal = mask_plan_for_step(
        requests,
        ["a"],
        torch.zeros((1, 3), dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        (),
        SpecMaskCache(),
        width=WIDTH,
        vocab_size=V,
    )
    assert plan is None and refusal == "empty_row"


# endregion

# region step_mask: the sample-time half


def _plan(rows=2, grammar=(0,), reasoning=(), **specs):
    static = static_masks(
        [MaskRowSpec(req_id=f"r{i}", **specs.get(f"r{i}", {})) for i in range(rows)],
        SpecMaskCache(),
        width=WIDTH,
        vocab_size=V,
    )
    return SpecMaskPlan(static, tuple(grammar), tuple(reasoning))


def test_step_mask_ands_the_scheduler_rows_into_the_static_part_and_counts_rows():
    plan = _plan(grammar=(0,), r1={"allowed_ids": [1, 2, 3]})
    grammar = torch.full((2, WIDTH, W), -1, dtype=torch.int32)
    grammar[0] = _pack(torch.arange(V) % 2 == 0).expand(WIDTH, W)
    mask, constrained = step_mask(
        plan, grammar, ["r0", "r1"], rows=3, width=WIDTH, vocab_size=V
    )
    assert constrained == 2 and tuple(mask.shape) == (3, WIDTH, W)
    assert int(unpack_allowed(mask[0, 0], V).sum()) == V // 2
    assert unpack_allowed(mask[1, 2], V).nonzero().reshape(-1).tolist() == [1, 2, 3]
    assert bool(is_unconstrained(mask[2]).all())  # the padding row


def test_step_mask_is_none_when_no_row_ends_up_constrained():
    plan = _plan(grammar=(0, 1))
    grammar = torch.full((2, WIDTH, W), -1, dtype=torch.int32)
    assert step_mask(
        plan, grammar, ["r0", "r1"], rows=2, width=WIDTH, vocab_size=V
    ) == (
        None,
        0,
    )


def test_step_mask_refuses_what_the_worker_and_the_scheduler_disagree_about():
    plan = _plan(grammar=(0,))
    # no grammar output: the grammar rows are unconstrained (an abort race)
    assert step_mask(plan, None, ["r0", "r1"], rows=2, width=WIDTH, vocab_size=V) == (
        None,
        0,
    )
    # a reasoning row the scheduler constrains
    plan = _plan(grammar=(), reasoning=(1,), r0={"allowed_ids": [1]})
    grammar = torch.full((2, WIDTH, W), -1, dtype=torch.int32)
    grammar[1, 2, 0] = 0
    with pytest.raises(RuntimeError, match="reasoning phase open"):
        step_mask(plan, grammar, ["r0", "r1"], rows=2, width=WIDTH, vocab_size=V)
    # a grammar row that leaves nothing
    plan = _plan(grammar=(0,))
    grammar = torch.full((2, WIDTH, W), -1, dtype=torch.int32)
    grammar[0, 1] = 0
    with pytest.raises(RuntimeError, match=r"allows no token.*r0"):
        step_mask(plan, grammar, ["r0", "r1"], rows=2, width=WIDTH, vocab_size=V)


def test_the_counters_report_steps_rows_and_fallbacks_by_reason():
    count_mask_step(3)
    count_mask_step(1)
    count_mask_fallback("empty_row")
    count_mask_fallback("empty_row")
    count_mask_fallback("grammar_with_controls")
    assert device_mask_stats() == {
        "steps": 2,
        "rows": 4,
        "fallbacks": {"empty_row": 2, "grammar_with_controls": 1},
        "chained": 0,
        "late": 0,
        "waits": 0,
    }


# endregion

# region H3: scheduler rows to the plugin's rows


def test_h3_each_request_gets_its_own_scheduler_rows_in_the_runner_slot_order():
    """Three structured requests, scheduler order a, b, c, runner rows c, pad, a, b.

    ``b`` has fewer scheduled drafts (2 rows), ``c`` was declined (absent from
    the grammar output, as a request the preceding step finished is), the pad row
    owns no request. Column ``j`` of a request is its row ``j``, all ones past
    its row count."""
    rows_per_request = {"a": 4, "b": 2, "c": 4}
    grammar_ids = ["a", "b"]  # c is declined
    scheduler = torch.full((8, W), -1, dtype=torch.int32)
    patterns = {}
    for index in range(8):
        allowed = torch.zeros(V, dtype=torch.bool)
        allowed[index * 10 : index * 10 + 5] = True
        scheduler[index] = _pack(allowed)
        patterns[index] = allowed
    # rows 0..3 are a's, 4..5 are b's
    grammar = spec_grammar_bitmask_for_tt_batch(
        bitmask=scheduler[:6],
        structured_output_request_ids=grammar_ids,
        rows_per_request=rows_per_request,
        row_req_ids=["c", None, "a", "b"],
        batch_length=4,
        width=WIDTH,
    )
    mask = compose_step_mask(
        None, grammar, (0, 2, 3), rows=4, width=WIDTH, vocab_size=V
    )
    allowed = unpack_allowed(mask, V)
    assert bool(is_unconstrained(mask[0]).all())  # declined
    assert bool(is_unconstrained(mask[1]).all())  # padding
    for column in range(WIDTH):
        assert torch.equal(allowed[2, column], patterns[column])  # a: rows 0..3
    assert torch.equal(allowed[3, 0], patterns[4])
    assert torch.equal(allowed[3, 1], patterns[5])
    assert bool(is_unconstrained(mask[3, 2:]).all())  # b: only 2 rows


# endregion

# region H7: the predicate


def _servable(
    params, *, mask, output_len=0, grammar_ok=False, penalized=False, **flags
):
    return params_fused_servable(
        params,
        output_len,
        8,
        V,
        grammar_ok=grammar_ok,
        penalized=penalized,
        device_mask=mask,
        **flags,
    )


STRUCTURED = StructuredOutputsParams(json_object=True)


@pytest.mark.parametrize(
    "kwargs, mask, grammar_ok, expected",
    [
        (dict(), True, False, True),
        (dict(structured_outputs=STRUCTURED), False, False, False),
        (dict(structured_outputs=STRUCTURED), True, False, True),
        (
            dict(structured_outputs=STRUCTURED),
            False,
            True,
            True,
        ),  # open phase: as before
        (dict(allowed_token_ids=[1, 2]), False, False, False),
        (dict(allowed_token_ids=[1, 2]), True, False, True),
        (dict(min_tokens=4, stop_token_ids=[7]), False, False, False),
        (dict(min_tokens=4, stop_token_ids=[7]), True, False, True),
        # a live grammar row with another control: its composed row may be empty
        (
            dict(structured_outputs=STRUCTURED, allowed_token_ids=[1, 2]),
            True,
            False,
            False,
        ),
        (
            dict(structured_outputs=STRUCTURED, min_tokens=4, stop_token_ids=[7]),
            True,
            False,
            False,
        ),
        # ... but a row in its open reasoning phase has no grammar mask
        (
            dict(structured_outputs=STRUCTURED, allowed_token_ids=[1, 2]),
            True,
            True,
            True,
        ),
        # sampled top_k inside the support, outside it
        (dict(temperature=1.0, top_k=5, allowed_token_ids=[1, 2]), True, False, True),
        (dict(temperature=1.0, top_k=8, allowed_token_ids=[1, 2]), True, False, False),
        (dict(temperature=1.0, top_k=0, allowed_token_ids=[1, 2]), True, False, False),
    ],
)
def test_h7_the_predicate_matrix(kwargs, mask, grammar_ok, expected):
    params = _params(**kwargs)
    assert _servable(params, mask=mask, grammar_ok=grammar_ok) is expected


def test_h7_bad_words_are_a_control_too():
    params = _params(temperature=0.0)
    params._bad_words_token_ids = [[4, 5]]
    assert has_mask_control(params, 0, False)
    assert _servable(params, mask=False) is False
    assert _servable(params, mask=True) is True
    params2 = _params(structured_outputs=STRUCTURED)
    params2._bad_words_token_ids = [[4]]
    assert mask_row_refusal(params2, 0, False) == "grammar_with_controls"


@pytest.mark.parametrize(
    "kwargs, grammar_only_ok",
    [
        (dict(structured_outputs=STRUCTURED), True),
        (dict(allowed_token_ids=[1, 2]), False),
        (dict(min_tokens=4, stop_token_ids=[7]), False),
        (dict(), True),
    ],
)
def test_h7_a_logprobs_row_is_served_only_with_a_grammar_only_mask(
    kwargs, grammar_only_ok
):
    params = _params(logprobs=3, **kwargs)
    served = _servable(params, mask=True, device_logprobs=True)
    assert served is grammar_only_ok
    # without the logprobs capability the old rule: not served
    assert _servable(params, mask=True, device_logprobs=False) is False
    if not grammar_only_ok:
        assert mask_row_refusal(params, 0, False) == "logprobs_with_controls"


def test_h7_a_logprobs_row_inside_its_open_reasoning_phase_is_served_as_before():
    params = _params(logprobs=3, structured_outputs=STRUCTURED)
    assert _servable(params, mask=True, grammar_ok=True, device_logprobs=True) is True
    assert _servable(params, mask=False, grammar_ok=True, device_logprobs=True) is True


@pytest.mark.parametrize(
    "kwargs, penalties_cap, expected",
    [
        (
            dict(allowed_token_ids=[1, 2]),
            False,
            False,
        ),  # a penalty needs its own capability
        (dict(allowed_token_ids=[1, 2]), True, True),
        (dict(logprobs=2, allowed_token_ids=[1, 2]), True, False),
        (dict(structured_outputs=STRUCTURED), True, True),
    ],
)
def test_h7_penalties_next_to_the_mask(kwargs, penalties_cap, expected):
    params = _params(presence_penalty=0.5, **kwargs)
    assert (
        _servable(
            params,
            mask=True,
            penalized=True,
            device_penalties=penalties_cap,
            device_logprobs=True,
        )
        is expected
    )


def test_h7_a_certificate_row_with_a_mask_is_servable():
    params = _params(temperature=1.0, top_p=0.9, allowed_token_ids=[1, 2])
    assert _servable(params, mask=True, device_top_p=True) is True
    assert _servable(params, mask=True, device_top_p=False) is False


def test_h7_with_the_capability_off_the_predicate_is_the_old_one_over_a_grid():
    for temperature in (0.0, 0.7):
        for top_k in (0, 1, 7, 8):
            for logprobs in (None, 0, 5):
                for kwargs in (
                    dict(),
                    dict(structured_outputs=STRUCTURED),
                    dict(allowed_token_ids=[1]),
                    dict(min_tokens=3, stop_token_ids=[7]),
                ):
                    params = _params(
                        temperature=temperature,
                        top_k=top_k,
                        logprobs=logprobs,
                        **kwargs,
                    )
                    for grammar_ok in (False, True):
                        old = params_fused_servable(
                            params, 0, 8, V, device_logprobs=True, grammar_ok=grammar_ok
                        )
                        off = params_fused_servable(
                            params,
                            0,
                            8,
                            V,
                            device_logprobs=True,
                            grammar_ok=grammar_ok,
                            device_mask=False,
                        )
                        assert old is off


# endregion

# region Output: masked top-N entries


def test_a_masked_top_n_entry_is_minus_inf_like_the_host_path():
    """The device writes -2**99 at a masked token; the host path has -inf."""
    ids = torch.tensor([[[5, 9, 11, 12]]], dtype=torch.int32)
    values = torch.tensor([[[3.0, 1.0, -(2.0**99), -(2.0**99)]]], dtype=torch.float32)
    lse = torch.tensor([[3.5]], dtype=torch.float32)
    tensors = device_spec_logprobs(
        SpecCandidates(ids=ids, values=values, lse=lse),
        torch.tensor([[5]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        3,
    )
    logprobs = tensors.logprobs
    assert logprobs[0, 0].item() == pytest.approx(3.0 - 3.5)
    assert logprobs[0, 1].item() == pytest.approx(3.0 - 3.5)
    assert logprobs[0, 2].item() == pytest.approx(1.0 - 3.5)
    assert bool(torch.isneginf(logprobs[0, 3]))
    assert MASK_VALUE_FLOOR > -(2.0**99)


# endregion

# region H4/H5: the real xgrammar rows of vLLM's StructuredOutputManager (offline)

REAL_VOCAB = 248320
THINK_END = 248069  # </think>
SCHEMA = '{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}'


@pytest.fixture(scope="module")
def xg():
    """vLLM's manager: xgrammar, the qwen3 reasoning parser, the Qwen3.8 tokenizer.

    Skipped where xgrammar or the cached tokenizer is not available offline."""
    pytest.importorskip("xgrammar")
    try:
        from vllm.config import ModelConfig, StructuredOutputsConfig, VllmConfig
        from vllm.v1.structured_output import StructuredOutputManager

        model_config = ModelConfig(
            model="Qwen/Qwen3.8-27B",
            tokenizer="Qwen/Qwen3.8-27B",
            dtype="bfloat16",
            max_model_len=2048,
        )
        manager = StructuredOutputManager(
            VllmConfig(
                model_config=model_config,
                structured_outputs_config=StructuredOutputsConfig(
                    backend="xgrammar", reasoning_parser="qwen3"
                ),
            )
        )
        assert manager.tokenizer is not None
    except Exception as exc:  # no network and no cached tokenizer
        pytest.skip(
            f"xgrammar manager or the Qwen3.8 tokenizer is unavailable offline: {exc}"
        )
    return manager


def _grammar_request(manager, history=(), *, reasoning_ended=True):
    from vllm.v1.request import Request

    params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        structured_outputs=StructuredOutputsParams(json=SCHEMA),
    )
    params.structured_outputs._backend = "xgrammar"
    request = Request(
        request_id="r",
        prompt_token_ids=manager.tokenizer.encode("hello"),
        sampling_params=params,
        pooling_params=None,
    )
    manager.grammar_init(request)
    compiling = request.structured_output_request._grammar
    if hasattr(compiling, "result"):  # compiled on the manager's executor
        compiling.result(timeout=120)
    # The grammar governs from the first token once reasoning is over (thinking
    # off, or an earlier step committed the end id).
    request.structured_output_request.reasoning_ended = reasoning_ended
    if history:
        assert manager.accept_tokens(request, list(history))
    return request


def _scheduler_step(manager, request, drafts):
    """What ``EngineCore`` does for a deferred structured step: validate the
    drafts (the invalid tail becomes -1 padding), then fill the bitmask rows."""
    validated = manager.validate_tokens(request, list(drafts))
    scheduled = validated + [-1] * (len(drafts) - len(validated))
    bitmask = manager.grammar_bitmask({"r": request}, ["r"], {"r": scheduled})
    return torch.from_numpy(bitmask), scheduled


def _plugin_rows(bitmask, scheduled):
    return spec_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["r"],
        rows_per_request={"r": len(scheduled) + 1},
        row_req_ids=["r"],
        batch_length=1,
        width=WIDTH,
    )


def _logits(seed, drafts):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, WIDTH, REAL_VOCAB, generator=gen) * 2
    for column, token in enumerate(drafts):
        logits[0, column, token] += (
            9.0  # likely to be accepted unless the mask rules it out
        )
    return logits


def _walk(logits, drafts, grammar, mask, temperature, seed):
    """The ``logits`` walk under the scheduler rows, and on the masked logits."""
    drafts_t = torch.tensor(
        [drafts + [-1] * (WIDTH - 1 - len(drafts))], dtype=torch.int32
    )
    valid = torch.tensor([len(drafts)], dtype=torch.int32)

    def run(lg, filters):
        sampling = SpecSamplingInputs(
            vocab_size=REAL_VOCAB,
            temperature=torch.tensor([temperature]),
            top_k=None if temperature == 0 else torch.tensor([20], dtype=torch.int32),
            generators={}
            if temperature == 0
            else {0: torch.Generator().manual_seed(seed)},
            filters=filters,
        )
        return accept_sampled_drafts(lg, drafts_t, valid, sampling)

    reference = run(logits, SpecTokenFilters(grammar_bitmask=grammar))
    masked = run(
        logits.masked_fill(~unpack_allowed(mask, REAL_VOCAB), float("-inf")), None
    )
    return reference, masked


def _grammar_plan():
    static = StaticMasks(all_ones(1, WIDTH, words=mask_words(REAL_VOCAB)), ())
    return SpecMaskPlan(static, (0,))


def _tokens(manager, text):
    return manager.tokenizer.encode(text)


@pytest.mark.parametrize("temperature", [0.0, 0.8])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "case",
    ["valid", "first_invalid", "middle_invalid", "padding", "short", "after_prefix"],
)
def test_h4_the_masked_commit_equals_the_logits_walk_on_real_xgrammar_rows(
    xg, case, seed, temperature
):
    ids = _tokens(xg, '{"a": 12}')  # '{"', 'a', '":', ' ', '1', '2', '}'
    wrong = _tokens(xg, "zebra")[0]
    history = ids[:2] if case == "after_prefix" else ()
    rest = ids[len(history) :]
    drafts = {
        "valid": rest[:3],
        "first_invalid": [wrong, rest[1], rest[2]],
        "middle_invalid": [rest[0], wrong, rest[2]],
        "padding": [rest[0], rest[1], wrong],
        "short": rest[:1],  # fewer drafts than columns: the rest is placeholders
        "after_prefix": rest[:3],
    }[case]
    request = _grammar_request(xg, history)
    bitmask, scheduled = _scheduler_step(xg, request, drafts)
    grammar = _plugin_rows(bitmask, scheduled)
    mask, constrained = step_mask(
        _grammar_plan(), grammar, ["r"], rows=1, width=WIDTH, vocab_size=REAL_VOCAB
    )
    assert constrained == 1 and mask.dtype == torch.int32
    assert tuple(mask.shape) == (1, WIDTH, mask_words(REAL_VOCAB))
    # rows after the first invalid draft hold the grammar of nothing: all ones
    cut = len(xg.validate_tokens(request, list(drafts)))
    assert bool(is_unconstrained(mask[0, cut + 1 :]).all())
    assert not bool(is_unconstrained(mask[0, : cut + 1]).any())

    reference, masked = _walk(
        _logits(seed, drafts), drafts, grammar, mask, temperature, seed
    )
    assert torch.equal(masked.committed_token_ids, reference.committed_token_ids)
    assert torch.equal(masked.accepted_counts, reference.accepted_counts)
    # and what was committed is valid for the grammar
    committed = reference.committed_token_ids[
        0, : int(reference.accepted_counts[0])
    ].tolist()
    assert xg.validate_tokens(request, committed) == committed
    if case in ("first_invalid",):
        assert committed[0] != wrong  # the invalid draft is never committed


@pytest.mark.parametrize("temperature", [0.0, 0.8])
@pytest.mark.parametrize("end_at", [0, 1, 2, None])
def test_h5_the_boundary_step_commits_what_the_logits_walk_does(
    xg, end_at, temperature
):
    """A reasoning-open request: the scheduler rows are all ones up to and including
    the end id and the grammar's start after it."""
    start = _tokens(xg, '{"a": 1')[:3]
    thought = _tokens(xg, "think")[0]
    drafts = [thought, thought, thought]
    if end_at is not None:
        drafts[end_at] = THINK_END
        for column in range(end_at + 1, 3):
            drafts[column] = start[column - end_at - 1]
    request = _grammar_request(xg, reasoning_ended=None)
    bitmask, scheduled = _scheduler_step(xg, request, drafts)
    grammar = _plugin_rows(bitmask, scheduled)
    constrained_rows = (~is_unconstrained(grammar[0])).nonzero().reshape(-1).tolist()
    assert constrained_rows == (
        [] if end_at is None else list(range(end_at + 1, WIDTH))
    )

    requests = {"r": _state(_params(structured_outputs=STRUCTURED))}
    drafts_t = torch.tensor([drafts], dtype=torch.int32)
    valid = torch.tensor([3], dtype=torch.int32)
    # No end id in the drafts: the row stays in its reasoning phase and needs no mask.
    reasoning_ok = ("r",) if end_at is None else ()
    plan, refusal = mask_plan_for_step(
        requests,
        ["r"],
        drafts_t,
        valid,
        reasoning_ok,
        SpecMaskCache(),
        width=WIDTH,
        vocab_size=REAL_VOCAB,
    )
    assert refusal is None
    if end_at is None:
        assert plan is None
        return
    assert plan.grammar_rows == (0,) and plan.reasoning_rows == ()
    mask, constrained = step_mask(
        plan, grammar, ["r"], rows=1, width=WIDTH, vocab_size=REAL_VOCAB
    )
    assert constrained == 1
    reference, masked = _walk(_logits(7, drafts), drafts, grammar, mask, temperature, 7)
    assert torch.equal(masked.committed_token_ids, reference.committed_token_ids)
    assert torch.equal(masked.accepted_counts, reference.accepted_counts)
    committed = reference.committed_token_ids[
        0, : int(reference.accepted_counts[0])
    ].tolist()
    assert xg.validate_tokens(request, committed) == committed


# endregion

# region The fake model with the two-call API

MASK_NEG = -(2.0**99)  # what the model writes at a disallowed token
SV = SAMPLED_VOCAB_SIZE  # the fake target's vocabulary: 64 ids, 2 mask words


class _PendingHandle:
    """What ``decode_forward_begin`` returns: queued, nothing is readable."""

    pending = True

    def __init__(self, tip_id=None):
        self.tip_id = tip_id

    def wait(self, *args, **kwargs):
        raise RuntimeError("a pending split step cannot be read before its finish")


class MaskTarget(LogprobsTarget):
    """The logprobs chain target with the mask capability and the two-call API.

    ``decode_forward_begin`` only records the call and returns a pending handle;
    the whole step runs in ``decode_forward_finish(mask)``, where the logits are
    masked before the support, the argmax and the logprob payload are taken
    (what the device does between its main trace and its top-k trace). It raises
    on every input the contract forbids: a second begin, any other call while a
    step is pending, a chained begin, a begin without ``spec_sampling`` or with a
    mask in it, a mask of the wrong dtype or shape, an empty mask row."""

    model_capabilities = {
        **LogprobsTarget.model_capabilities,
        "supports_spec_device_mask": True,
    }

    with_drafts = True

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []  # ("one_call"|"begin"|"finish", ...)
        self.masks: list[torch.Tensor | None] = []
        self.begin_inputs: list[dict] = []
        self._split = None
        self._mask = None
        self._cand = None

    # the device edit: the plugin's mask unpacked, disallowed ids to the mask value
    def _masked(self, logits):
        if self._mask is None:
            return logits
        allowed = unpack_allowed(self._mask, int(logits.shape[-1]))
        rows = int(allowed.shape[0])
        logits = logits.clone()
        logits[:rows] = logits[:rows].masked_fill(~allowed, MASK_NEG)
        return logits

    def _accept(self, tokens, positions, num_valid, ss):
        logits = self._masked(rule_logits(tokens, positions))
        rows = int(tokens.shape[0])
        values, ids = exact_support(logits, self.topk_candidates)
        # the device accept body ignores a k-th value that is the mask value (its
        # tie rule has a guard for it): the reference sees -inf there
        values = torch.where(
            values < MASK_VALUE_FLOOR, torch.full_like(values, float("-inf")), values
        )
        argmax = logits.argmax(dim=-1).to(torch.int32)
        if ss is None:
            temperature = torch.zeros(rows)
            top_k = torch.ones(rows, dtype=torch.int32)
            top_p = torch.ones(rows)
            uniforms = None
        else:
            ss = self._as_support_sampling(ss)
            temperature, top_k, top_p, uniforms = (
                ss["temperature"],
                ss["top_k"],
                ss["top_p"],
                ss["uniforms"],
            )
        result = accept_sampled_support(
            argmax,
            values,
            ids,
            tokens[:, 1:],
            num_valid,
            SpecSamplingInputs(
                vocab_size=int(logits.shape[-1]),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generators={},
            ),
            None,
            uniforms=uniforms,
        )
        self._cand = _candidates(logits) if self._wants(ss) else None
        return result.committed_token_ids, result.accepted_counts, argmax

    def _chain_forward(self, *args, **kwargs):
        """The chain forward, whose readout also carries ``FusedReadout.drafts``:
        per row, what the next chained step verifies (``with_drafts=False``: a
        model that reports none)."""
        out = super()._chain_forward(*args, **kwargs)
        if self.with_drafts:
            handle = out.handle
            readout = handle._readout
            drafts = torch.zeros(
                (int(readout.accepted_counts.shape[0]), self._num_drafts),
                dtype=torch.int32,
            )
            for row, (last, next_p0) in self._dev_state.items():
                drafts[row] = torch.tensor(
                    self._chain_drafts(last, next_p0), dtype=torch.int32
                )
            handle._readout = dataclasses.replace(readout, drafts=drafts)
        return out

    def _chain_accept(self, tokens, p0, num_valid, spec_sampling, kwargs):
        columns = torch.arange(tokens.shape[1], dtype=torch.int32).unsqueeze(0)
        positions = p0.unsqueeze(1) + columns
        positions = torch.where(
            (columns > num_valid.unsqueeze(1)) | (p0 < 0).unsqueeze(1),
            torch.full_like(positions, -1),
            positions,
        )
        committed, counts, _ = self._accept(tokens, positions, num_valid, spec_sampling)
        self._pending = self._cand
        return committed, counts

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        if self._split is not None:
            raise RuntimeError("a split step is pending: finish it first")
        if spec_mode == ACCEPT_MODE_FUSED_SAMPLE:
            self.calls.append(("one_call", bool(kwargs.get("defer_read"))))
            if not kwargs.get("defer_read"):
                return self._fused_sync(tokens, start_pos, kwargs)
        return super().decode_forward(tokens, start_pos, spec_mode, **kwargs)

    def _fused_sync(self, tokens, start_pos, kwargs):
        ss = kwargs.get("spec_sampling")
        self._chain_invalidate()  # a synchronous step leaves nothing to chain onto
        self.verify_modes.append(ACCEPT_MODE_FUSED_SAMPLE)
        self.spec_sampling_calls.append(ss)
        committed, counts, argmax = self._accept(
            tokens, start_pos, kwargs["num_valid_drafts"], ss
        )
        return VerifyOutput(
            spec_mode=ACCEPT_MODE_FUSED_SAMPLE,
            argmax_ids=argmax,
            accepted_token_ids=committed,
            accepted_counts=counts,
            candidates=self._cand,
        )

    def decode_forward_begin(self, tokens, start_pos, spec_mode=None, **kwargs):
        if self._split is not None:
            raise RuntimeError(
                "the previous split step has no decode_forward_finish yet"
            )
        if spec_mode != ACCEPT_MODE_FUSED_SAMPLE:
            raise ValueError("decode_forward_begin needs spec_mode 'fused_sample'")
        chained = kwargs.get("chain_rows") is not None
        if chained and not kwargs.get("defer_read"):
            raise ValueError("a chained begin needs defer_read")
        if (kwargs.get("chain_tip") is not None) != chained:
            raise ValueError("chain_rows and chain_tip come together")
        ss = kwargs.get("spec_sampling")
        if ss is None:
            raise ValueError("decode_forward_begin needs spec_sampling")
        if "mask" in ss:
            raise ValueError("decode_forward_begin takes no spec_sampling['mask']")
        self.calls.append(("begin", bool(kwargs.get("defer_read"))))
        self.begin_inputs.append(
            {
                "tokens": tokens.clone(),
                "num_valid_drafts": kwargs["num_valid_drafts"].clone(),
                "spec_sampling": ss,
                "defer_read": bool(kwargs.get("defer_read")),
                "chain_rows": kwargs.get("chain_rows"),
                "chain_tip": kwargs.get("chain_tip"),
            }
        )
        self._split = (tokens, start_pos, kwargs)
        # a deferred begin knows its tip (the forward is queued), as the engine's
        tip = (self._tip_id or 0) + 1 if kwargs.get("defer_read") else None
        return VerifyOutput(spec_mode=spec_mode, handle=_PendingHandle(tip))

    def decode_forward_finish(self, tt_out, mask=None):
        if self._split is None or not isinstance(tt_out.handle, _PendingHandle):
            raise RuntimeError("decode_forward_finish: no pending split step")
        tokens, start_pos, kwargs = self._split
        if mask is not None:
            rows, width = tokens.shape
            ok = mask.dtype == torch.int32 and mask.dim() == 3
            if not (ok and mask.shape[1] == width and 1 <= mask.shape[0] <= rows):
                raise ValueError(f"bad mask {mask.dtype} {tuple(mask.shape)}")
            if bool((mask == 0).all(dim=-1).any()):
                raise ValueError("the token mask has a row that allows no token")
        self.calls.append(("finish", mask is not None))
        self.masks.append(mask)
        self._split = None
        self._mask = mask
        try:
            return self._fused_sync_or_deferred(tokens, start_pos, kwargs)
        finally:
            self._mask = None

    def _fused_sync_or_deferred(self, tokens, start_pos, kwargs):
        if kwargs.get("defer_read"):
            self.verify_modes.append(ACCEPT_MODE_FUSED_SAMPLE)
            self.spec_sampling_calls.append(kwargs.get("spec_sampling"))
            self._num_drafts = int(tokens.shape[1]) - 1
            return self._chain_forward(
                tokens,
                start_pos,
                kwargs["num_valid_drafts"],
                kwargs["accepted_counts"],
                kwargs.get("spec_sampling"),
                kwargs,
            )
        return self._fused_sync(tokens, start_pos, kwargs)

    def read_decode_output(self, tt_out, async_read=False, **kwargs):
        if isinstance(getattr(tt_out, "handle", tt_out), _PendingHandle):
            raise RuntimeError("a pending split step cannot be read")
        return super().read_decode_output(tt_out, async_read=async_read, **kwargs)

    def propose_draft_tokens(self, *args, **kwargs):
        if self._split is not None:
            raise RuntimeError("a split step is pending: finish it first")
        return super().propose_draft_tokens(*args, **kwargs)

    @property
    def finishes(self):
        return [call for call in self.calls if call[0] == "finish"]


# endregion

# region The synchronous step path (begin at the forward, finish at the sample)

DRAFT_LEN = sync_harness.DRAFT_LEN
GREEDY = sync_harness.GREEDY


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (
        "VLLM_TT_SPEC_FUSED",
        "VLLM_TT_SPEC_FUSED_SAMPLED",
        "VLLM_TT_SPEC_CHAIN",
        "VLLM_TT_SPEC_CHAIN_SAMPLED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_TT_SPEC_DEVICE_TOPK", "1")


def _mask_runner(*, mask=True, logprobs=False, top_p=False, penalties=False):
    """A synchronous runner over the controls harness, on the mask-capable fake."""
    model = MaskTarget()
    runner = ctl._runner(model)
    runner._spec_plan = type(model).spec_plan(
        None, sync_harness.MAX_NUM_REQS, DRAFT_LEN
    )
    runner._spec_device_mask = mask
    runner._spec_device_logprobs = logprobs
    runner._spec_device_top_p = top_p
    runner._spec_device_penalties = penalties
    runner._spec_penalty_tracker = SpecPenaltyTracker()
    runner.model_config.logprobs_mode = "raw_logprobs"
    runner.async_decode = TTAsyncDecodeController(runner)
    return model, runner


def _step(runner, *req_ids, drafts=None, grammar="auto"):
    """One synchronous decode step as ``_forward_with_model_input`` and
    ``sample_tokens`` run it; returns the model input and the committed ids."""
    drafts = dict(drafts or {})
    for req_id in list(drafts):
        if runner.requests[req_id].sampling_params.structured_outputs is not None:
            drafts[req_id] = ctl._validated(
                list(runner.requests[req_id].output_token_ids), drafts[req_id]
            )
    runner.input_batch.refresh_logitsprocs()
    model_input = sync_harness._build(runner, *req_ids, drafts=drafts)
    grammar_output = (
        ctl._grammar_output(runner, drafts) if grammar == "auto" else grammar
    )
    fwd = runner._forward_with_model_input(model_input)
    output = runner._finish_front_packed_sync(grammar_output, fwd=fwd)
    committed = {
        req_id: list(output.sampled_token_ids[output.req_id_to_index[req_id]])
        for req_id in req_ids
    }
    runner.last_output = output
    return model_input, committed


def _run(runner, req_id, count, drafts_for=None, neighbor=None):
    req_ids = (req_id,) if neighbor is None else (neighbor, req_id)
    while len(sync_harness._output(runner, req_id)) < count:
        drafts = {}
        if neighbor is not None:
            drafts[neighbor] = sync_harness._argmax_drafts(runner, neighbor)
        if drafts_for is not None:
            drafts[req_id] = drafts_for(runner)
        _step(runner, *req_ids, drafts=drafts)
    return sync_harness._output(runner, req_id)[:count]


def _pairs(model):
    """The calls of the fake, with each begin followed at once by its finish."""
    calls = [call[0] for call in model.calls]
    for index, name in enumerate(calls):
        if name == "begin":
            assert calls[index + 1 : index + 2] == ["finish"], calls
    return calls


@pytest.mark.parametrize("drafts", ["right", "wrong"])
@pytest.mark.parametrize("case", ["allowed", "bad-words", "min-tokens", "structured"])
def test_a_masked_greedy_row_commits_what_the_logits_walk_commits(case, drafts):
    """The reference arm is ordinary decoding on the host sampler; the masked arm
    speculates through the two-call step with right and with bent drafts."""
    params = ctl._params(case, temperature=0.0)
    reference, _ = ctl._run_plain(params, 40)
    assert reference[:30] != ctl._run_plain(GREEDY, 30)[0], (
        "the control changes nothing"
    )

    def right(runner):
        done = len(sync_harness._output(runner, "c"))
        return reference[done : done + DRAFT_LEN]

    def wrong(runner):
        bent = right(runner)
        if len(bent) > 1:
            bent[1] = (bent[1] + 1) % SV
        return bent

    model, runner = _mask_runner()
    sync_harness._add_request(runner, "c", params)
    emitted = _run(runner, "c", 30, drafts_for=right if drafts == "right" else wrong)

    assert emitted == reference[:30]
    calls = _pairs(model)
    assert "begin" in calls and calls.count("begin") == calls.count("finish")
    # the model never saw the mask key in a one-call step
    assert all("mask" not in (inp["spec_sampling"] or {}) for inp in model.begin_inputs)
    ctl._assert_follows_its_controls(emitted, params, first_token=1)
    if case != "min-tokens":
        assert "one_call" not in calls  # every verify of the row carried a control


def test_an_allowed_ids_row_is_finished_at_once_and_the_mask_has_the_contract_shape():
    model, runner = _mask_runner()
    sync_harness._add_request(runner, "g", GREEDY, first_token=21)
    sync_harness._add_request(
        runner, "a", ctl._params("allowed", temperature=0.0), first_token=1
    )
    _step(runner, "g", "a")  # the first step verifies nothing
    model_input, _ = _step(
        runner,
        "g",
        "a",
        drafts={r: sync_harness._argmax_drafts(runner, r) for r in ("g", "a")},
    )
    assert model_input.spec_mask is not None
    assert model.calls[-2:] == [("begin", False), ("finish", True)]
    mask = model.masks[-1]
    rows, width = (int(s) for s in model_input.input_tokens.shape)
    assert mask.dtype == torch.int32 and tuple(mask.shape) == (
        rows,
        DRAFT_LEN + 1,
        mask_words(SV),
    )
    row_of = {r: i for i, r in enumerate(model_input.row_req_ids)}
    # the allowed row allows the even ids at every column; the others are all ones
    evens = unpack_allowed(mask[row_of["a"]], SV)
    assert bool(evens[:, 0::2].all()) and not bool(evens[:, 1::2].any())
    assert bool(is_unconstrained(mask[row_of["g"]]).all())
    assert bool(
        is_unconstrained(mask[len(model_input.row_req_ids) :]).all()
    )  # padding rows
    # the begin carried the sampling controls of a sampled step, a greedy row greedy
    ss = model.begin_inputs[-1]["spec_sampling"]
    assert bool(ss["greedy"][: len(row_of)].all()) and "mask" not in ss
    assert tuple(ss["uniforms"].shape) == (rows, width + 1) and not bool(
        ss["uniforms"].any()
    )


def test_a_step_with_no_mask_row_keeps_the_one_call_path_exactly():
    on_model, on = _mask_runner()
    off_model, off = _mask_runner(mask=False)
    for runner in (on, off):
        sync_harness._add_request(runner, "g", GREEDY, first_token=21)
        sync_harness._add_request(runner, "h", GREEDY, first_token=7)
    for _ in range(4):
        for runner in (on, off):
            drafts = {r: sync_harness._argmax_drafts(runner, r) for r in ("g", "h")}
            _step(runner, "g", "h", drafts=drafts)
    assert on_model.calls == off_model.calls
    assert {name for name, *_ in on_model.calls} == {"one_call"}
    assert sync_harness._output(on, "g") == sync_harness._output(off, "g")
    assert [c for c in on_model.spec_sampling_calls] == [None] * len(
        on_model.spec_sampling_calls
    )


def test_with_the_capability_off_a_control_row_asks_for_logits_as_before():
    model, runner = _mask_runner(mask=False)
    sync_harness._add_request(runner, "a", ctl._params("allowed", temperature=0.0))
    _step(runner, "a")
    mi, _ = _step(runner, "a", drafts={"a": sync_harness._argmax_drafts(runner, "a")})
    assert mi.spec_mode == ACCEPT_MODE_LOGITS and mi.spec_mask is None
    assert model.verify_modes[-1] == ACCEPT_MODE_LOGITS
    assert not any(name in ("begin", "finish") for name, *_ in model.calls)


def test_a_logits_fallback_is_counted_by_reason():
    model, runner = _mask_runner(logprobs=True)
    cases = {
        "logprobs_with_controls": ctl._params("allowed", temperature=0.0, logprobs=2),
        "grammar_with_controls": SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            structured_outputs=STRUCTURED,
            min_tokens=9,
            stop_token_ids=[5],
        ),
        "empty_row": SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            min_tokens=9,
            stop_token_ids=[4, 6],
            allowed_token_ids=[4, 6],
        ),
        "other": SamplingParams(  # a mask-served control on a row the mask cannot serve
            temperature=1.0, top_k=40, ignore_eos=True, allowed_token_ids=[4, 6, 8]
        ),
    }
    for index, (reason, params) in enumerate(cases.items()):
        req = f"r{index}"
        sync_harness._add_request(runner, req, params, first_token=1 + index)
        before = device_mask_stats()["fallbacks"].get(reason, 0)
        mi = sync_harness._build(
            runner, req, drafts={req: sync_harness._argmax_drafts(runner, req)}
        )
        assert mi.spec_mode == ACCEPT_MODE_LOGITS and mi.spec_mask is None, reason
        assert device_mask_stats()["fallbacks"].get(reason, 0) == before + 1, reason
        runner.requests.pop(req)
        runner.input_batch.remove_request(req)
    assert set(device_mask_stats()["fallbacks"]) == set(cases)


def test_a_stale_pending_step_is_refused_not_read():
    model, runner = _mask_runner()
    sync_harness._add_request(runner, "a", ctl._params("allowed", temperature=0.0))
    _step(runner, "a")
    runner.input_batch.refresh_logitsprocs()
    mi = sync_harness._build(
        runner, "a", drafts={"a": sync_harness._argmax_drafts(runner, "a")}
    )
    controller = runner.async_decode
    submission = controller.submit_decode(mi, read_from_device=False, async_read=False)
    assert submission.mask_pending and model.calls[-1] == ("begin", False)
    with pytest.raises(RuntimeError, match="before its mask finish"):
        controller.finalize_decode(submission)
    done = controller.finish_masked_decode(submission, None, async_read=False)
    assert not done.mask_pending and model.calls[-1] == ("finish", False)
    with pytest.raises(RuntimeError, match="not a pending masked step"):
        controller.finish_masked_decode(done, None, async_read=False)
    assert controller.finalize_decode(done) is not None


def test_the_sync_path_refuses_a_grammar_output_that_lacks_a_structured_row():
    from vllm.v1.core.sched.output import GrammarOutput

    model, runner = _mask_runner()
    sync_harness._add_request(runner, "s", ctl._params("structured", temperature=0.0))
    _step(runner, "s")
    empty = GrammarOutput(
        [], torch.zeros((0, mask_words(SV)), dtype=torch.int32).numpy()
    )
    with pytest.raises(RuntimeError, match="missing TT batch request IDs"):
        _step(
            runner,
            "s",
            drafts={"s": sync_harness._argmax_drafts(runner, "s")},
            grammar=empty,
        )


END_IDS = (60, 61)  # the reasoning-end ids of the fake vocabulary
OPEN = (False, END_IDS)


def _reasoning_runner(*, mask=True):
    model, runner = _mask_runner(mask=mask)
    runner._reasoning_starts = {"t": OPEN}
    sync_harness._add_request(
        runner, "t", ctl._params("structured", temperature=0.0), first_token=1
    )
    sync_harness._add_request(
        runner, "a", ctl._params("allowed", temperature=0.0), first_token=9
    )
    _step(runner, "t", "a", grammar=_ones_grammar(runner, "t", rows=1))
    return model, runner


def _ones_grammar(runner, req_id, rows=DRAFT_LEN + 1):
    from vllm.v1.core.sched.output import GrammarOutput

    ones = torch.full((rows, mask_words(SV)), -1, dtype=torch.int32)
    return GrammarOutput([req_id], ones.numpy())


def test_a_reasoning_row_next_to_a_mask_row_needs_all_ones_rows_from_the_scheduler():
    """The row stays on its open phase (no grammar mask, no draft handoff); the
    scheduler holds all ones for it, and a cleared bit is a disagreement."""
    model, runner = _reasoning_runner()
    thought = [3, 4, 5]
    drafts = {"t": thought, "a": sync_harness._argmax_drafts(runner, "a")}
    mi, committed = _step(
        runner, "t", "a", drafts=drafts, grammar=_ones_grammar(runner, "t")
    )
    row_of = {r: i for i, r in enumerate(mi.row_req_ids)}
    assert mi.spec_mask.grammar_rows == () and mi.spec_mask.reasoning_rows == (
        row_of["t"],
    )
    assert runner._grammar_draft_handoff == {}
    assert model.calls[-1] == ("finish", True)
    with pytest.raises(RuntimeError, match="reasoning phase open"):
        _step(
            runner, "t", "a", drafts=drafts, grammar=ctl._grammar_output(runner, drafts)
        )


def test_h5_a_boundary_step_takes_the_mask_with_the_capability_and_logits_without():
    for capability in (True, False):
        model, runner = _reasoning_runner(mask=capability)
        runner.requests.pop("a")
        runner.input_batch.remove_request("a")
        end_at = {"t": [3, END_IDS[0], 5]}  # the end id is a draft
        runner.input_batch.refresh_logitsprocs()
        mi = sync_harness._build(runner, "t", drafts=end_at)
        if capability:
            row = mi.row_req_ids.index("t")
            assert mi.spec_mode == ACCEPT_MODE_FUSED_SAMPLE
            assert mi.spec_mask.grammar_rows == (row,)
            assert mi.spec_mask.reasoning_rows == ()
        else:
            assert mi.spec_mode == ACCEPT_MODE_LOGITS and mi.spec_mask is None


def test_a_grammar_row_with_no_grammar_output_at_all_fails_loudly():
    model, runner = _mask_runner()
    sync_harness._add_request(runner, "s", ctl._params("structured", temperature=0.0))
    _step(runner, "s")
    with pytest.raises(RuntimeError, match="grammar output is absent"):
        _step(
            runner,
            "s",
            drafts={"s": sync_harness._argmax_drafts(runner, "s")},
            grammar=None,
        )


# endregion

# region The asynchronous step path, chained run

EVENS = list(range(0, SV, 2))


class MaskEngine(Engine):
    """``EngineCore.step_with_batch_queue`` for steps with a structured row.

    After ``execute_model`` a step with a structured row defers its sampling:
    the core waits for the oldest step's output (the scheduler advances each
    grammar by what that step committed), takes the runner's drafts, validates
    each against its grammar (the invalid tail becomes -1 padding), builds the
    bitmask from what survived and only then calls ``sample_tokens``. Any other
    step is sampled at once with no grammar output. ``after_execute`` records
    the fake's calls right after each ``execute_model``; ``handed`` the drafts
    the runner handed over."""

    def __init__(self, runner, **kw):
        super().__init__(runner, **kw)
        self.after_execute: list[list[tuple]] = []
        self.handed: list[dict[str, list[int]]] = []

    def _structured_rows(self):
        runner = self.runner
        return [
            req_id
            for req_id in runner.input_batch.req_ids[: runner.input_batch.num_reqs]
            if runner.requests[req_id].sampling_params.structured_outputs is not None
        ]

    def _grammar(self):
        runner = self.runner
        handed = runner.take_draft_token_ids()
        by_request = dict(zip(handed.req_ids, handed.draft_token_ids)) if handed else {}
        self.handed.append({r: list(d) for r, d in by_request.items()})
        structured = self._structured_rows()
        if not structured:
            return None
        rows = []
        for req_id in structured:
            # the scheduler's grammar state: every output it has been given
            history = list(self.outputs.get(req_id, ()))
            valid = ctl._validated(history, by_request.get(req_id, []))
            rows.append(
                ctl._scheduler_rows(history, valid + [-1] * (CHAIN_K - len(valid)))
            )
        return GrammarOutput(structured, torch.cat(rows).numpy())

    def iterate(self):
        so = self.schedule()
        k = self.step
        self.step += 1
        self.step_rows[k] = list(so.num_scheduled_tokens)
        assert self.runner.execute_model(so) is None
        self.after_execute.append(list(self.runner.model.calls))
        if self.inflight and self._structured_rows():
            # the structured step waits for the output of the one before it
            self.resolve_oldest()
        wrapper = self.runner.sample_tokens(self._grammar())
        self.inflight.append((wrapper, k))
        if len(self.inflight) >= 2:
            self.resolve_oldest()


def _mask_params(case: str) -> SamplingParams:
    if case == "allowed":
        return SamplingParams(temperature=0.0, ignore_eos=True, allowed_token_ids=EVENS)
    if case == "structured":
        return SamplingParams(
            temperature=0.0, ignore_eos=True, structured_outputs=ctl._structured()
        )
    return SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        min_tokens=7,
        stop_token_ids=[3, 5, 9],
    )


def _async_run(monkeypatch, *, mask, case="allowed", steps=16, stop_after=None):
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN_SAMPLED", "1")
    torch.manual_seed(5)
    model = MaskTarget()
    runner = _make_runner(model)
    # The target's vocabulary and vLLM's min_tokens processor, as a launch builds.
    runner.input_batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=SV,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
        logitsprocs=LogitsProcessors(
            [MinTokensLogitsProcessor(None, torch.device("cpu"), False)]
        ),
    )
    runner._spec_device_mask = mask
    runner.model_config.logprobs_mode = "raw_logprobs"
    engine = MaskEngine(runner)
    engine.add("g1", 1)
    engine.add("g2", 9)
    engine.add("x", 17, _mask_params(case), stop_after=stop_after)
    for _ in range(steps):
        engine.iterate()
    engine.finish()
    return model, runner, engine


@pytest.mark.parametrize("case", ["allowed", "structured", "min_tokens"])
def test_an_asynchronous_masked_run_has_the_logits_run_s_streams(monkeypatch, case):
    on_model, on, on_engine = _async_run(monkeypatch, mask=True, case=case)
    off_model, off, off_engine = _async_run(monkeypatch, mask=False, case=case)
    for req_id in ("g1", "g2", "x"):
        n = min(len(on_engine.outputs[req_id]), len(off_engine.outputs[req_id]))
        assert n >= 16
        assert on_engine.outputs[req_id][:n] == off_engine.outputs[req_id][:n], req_id
    follows = on_engine.outputs["x"]
    if case == "allowed":
        assert all(token in EVENS for token in follows)
    if case == "structured":
        history: list[int] = []
        for token in follows:
            assert ctl._grammar_allows(history, token), (history, token)
            history.append(token)
    if case == "min_tokens":
        assert not {3, 5, 9} & set(follows[:7])
    assert off_model.calls == [] or not any(
        c[0] in ("begin", "finish") for c in off_model.calls
    )
    assert any(c[0] == "begin" for c in on_model.calls)


def test_a_masked_step_begins_at_execute_model_and_finishes_at_sample_tokens(
    monkeypatch,
):
    model, runner, engine = _async_run(monkeypatch, mask=True)
    calls = [c[0] for c in model.calls]
    assert calls.count("begin") == calls.count("finish") > 8
    # right after execute_model the step was begun and not finished
    for snapshot in engine.after_execute:
        assert snapshot[-1][0] == "begin"
        assert [c[0] for c in snapshot].count("begin") == [
            c[0] for c in snapshot
        ].count("finish") + 1
    # the begin is the deferred form the reader thread reads after the finish
    assert all(call == ("begin", True) for call in model.calls if call[0] == "begin")
    assert all(inp["defer_read"] for inp in model.begin_inputs)
    # every step carried a begin: the allowed row is live all the time
    assert len(model.begin_inputs) == len(engine.after_execute)


def test_a_grammar_step_hands_the_drafts_it_verifies_to_the_scheduler(monkeypatch):
    """The scheduler makes grammar rows 1..K from the drafts the runner hands over,
    so they must be the drafts the step verifies: the begin call's, for a step that
    is not chained, and the drafts the step before it reported, for one that is
    (the tip's readout; its own block holds placeholders)."""
    model, runner, engine = _async_run(monkeypatch, mask=True, case="structured")
    plain = chained = 0
    for index, handed in enumerate(engine.handed):
        if "x" not in handed:
            continue
        verified = model.begin_inputs[index]
        row = engine.step_rows[index].index("x")
        valid = int(verified["num_valid_drafts"][row])
        if verified["chain_rows"] is None:
            assert handed["x"] == verified["tokens"][row, 1 : 1 + valid].tolist()
            plain += 1
        else:
            tip = next(h for h in model.handles if h.tip_id == verified["chain_tip"])
            assert handed["x"] == tip._readout.drafts[row, :valid].tolist()
            chained += 1
    assert plain >= 1 and chained > 4
    # a step with no grammar row hands nothing over
    model, runner, engine = _async_run(monkeypatch, mask=True, case="allowed")
    assert all(not handed for handed in engine.handed)


def test_an_async_grammar_row_with_no_grammar_output_is_left_unconstrained(monkeypatch):
    """A request an abort removed is absent from the grammar output: no raise."""
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = MaskTarget()
    runner = _make_runner(model)
    runner._spec_device_mask = True
    engine = MaskEngine(runner)
    engine.add("x", 17, _mask_params("structured"))
    engine.iterate()
    engine._grammar = lambda: None
    engine.iterate()
    engine.finish()
    assert model.calls[-1] == ("finish", False)


def test_a_second_begin_before_the_finish_is_the_models_error(monkeypatch):
    """One split step at a time: the runner finishes each step in ``sample_tokens``."""
    monkeypatch.setenv("VLLM_TT_SPEC_CHAIN", "1")
    model = MaskTarget()
    runner = _make_runner(model)
    runner._spec_device_mask = True
    engine = MaskEngine(runner)
    engine.add("x", 17, _mask_params("allowed"))
    engine.iterate()  # step 0: finished
    so = engine.schedule()
    assert runner.execute_model(so) is None
    assert model.calls[-1][0] == "begin"
    # the runner drains the pending step first, and refuses to read it unfinished
    with pytest.raises(RuntimeError, match="before its mask finish"):
        runner.execute_model(engine.schedule())


# endregion

# region H8/H7 on the runner: the sampling controls and the modes


def test_h8_a_masked_all_greedy_step_sends_the_dict_and_draws_nothing():
    model, runner = _mask_runner()
    sync_harness._add_request(runner, "a", ctl._params("allowed", temperature=0.0))
    sync_harness._add_request(runner, "g", GREEDY, first_token=21)
    _step(runner, "a", "g")
    runner.input_batch.refresh_logitsprocs()
    mi = sync_harness._build(
        runner,
        "a",
        "g",
        drafts={r: sync_harness._argmax_drafts(runner, r) for r in ("a", "g")},
    )
    assert mi.spec_mask is not None
    torch.manual_seed(9)
    state = torch.get_rng_state()
    ss = fused_spec_sampling(mi, runner._spec_plan, vocab_size=SV)
    assert torch.equal(state, torch.get_rng_state())
    rows = int(mi.input_tokens.shape[0])
    assert ss["greedy"].dtype == torch.bool and bool(ss["greedy"].all())
    assert ss["top_k"].dtype == torch.int32 and tuple(ss["top_k"].shape) == (rows,)
    assert not bool(ss["uniforms"].any()) and "mask" not in ss
    # without a mask row the same step sends nothing, as before
    other_model, other = _mask_runner()
    sync_harness._add_request(other, "g", GREEDY, first_token=21)
    _step(other, "g")
    plain = sync_harness._build(
        other, "g", drafts={"g": sync_harness._argmax_drafts(other, "g")}
    )
    assert plain.spec_mask is None
    assert fused_spec_sampling(plain, other._spec_plan, vocab_size=SV) is None


def test_h8_a_sampled_masked_row_draws_its_uniforms_and_a_greedy_one_stays_greedy():
    model, runner = _mask_runner()
    sync_harness._add_request(
        runner,
        "s",
        SamplingParams(
            temperature=1.0, top_k=4, ignore_eos=True, allowed_token_ids=EVENS, seed=4
        ),
        first_token=3,
    )
    sync_harness._add_request(
        runner, "a", ctl._params("allowed", temperature=0.0), first_token=1
    )
    _step(runner, "s", "a")
    mi, _ = _step(
        runner,
        "s",
        "a",
        drafts={r: sync_harness._argmax_drafts(runner, r) for r in ("s", "a")},
    )
    ss = model.begin_inputs[-1]["spec_sampling"]
    row_of = {r: i for i, r in enumerate(mi.row_req_ids)}
    assert bool(ss["greedy"][row_of["a"]]) and not bool(ss["greedy"][row_of["s"]])
    assert bool(ss["uniforms"][row_of["s"]].any())
    assert int(ss["top_k"][row_of["s"]]) == 4


@pytest.mark.parametrize(
    "params_kwargs, caps, expected",
    [
        # allowed / bad words / min tokens / grammar: masked with the capability
        (dict(allowed_token_ids=EVENS), dict(), "masked"),
        (dict(structured_outputs=STRUCTURED), dict(), "masked"),
        (dict(min_tokens=6, stop_token_ids=[3]), dict(), "masked"),
        (dict(allowed_token_ids=EVENS), dict(mask=False), ACCEPT_MODE_LOGITS),
        # logprobs: only with a grammar-only mask and the logprobs capability
        (
            dict(logprobs=2, structured_outputs=STRUCTURED),
            dict(logprobs=True),
            "masked",
        ),
        (dict(logprobs=2, structured_outputs=STRUCTURED), dict(), ACCEPT_MODE_LOGITS),
        (
            dict(logprobs=2, allowed_token_ids=EVENS),
            dict(logprobs=True),
            ACCEPT_MODE_LOGITS,
        ),
        (
            dict(logprobs=2, min_tokens=6, stop_token_ids=[3]),
            dict(logprobs=True),
            ACCEPT_MODE_LOGITS,
        ),
        # penalties: the model's own capability decides
        (
            dict(presence_penalty=0.5, allowed_token_ids=EVENS),
            dict(penalties=True),
            "masked",
        ),
        (
            dict(presence_penalty=0.5, allowed_token_ids=EVENS),
            dict(),
            ACCEPT_MODE_LOGITS,
        ),
        (
            dict(presence_penalty=0.5, logprobs=2, structured_outputs=STRUCTURED),
            dict(penalties=True, logprobs=True),
            ACCEPT_MODE_LOGITS,
        ),
        # a certificate row with a mask is served (a synchronous step)
        (
            dict(temperature=1.0, top_p=0.9, allowed_token_ids=EVENS),
            dict(top_p=True),
            "masked",
        ),
        (
            dict(temperature=1.0, top_p=0.9, allowed_token_ids=EVENS),
            dict(),
            ACCEPT_MODE_LOGITS,
        ),
        # a sampled top_k outside the support never was
        (
            dict(temperature=1.0, top_k=40, allowed_token_ids=EVENS),
            dict(),
            ACCEPT_MODE_LOGITS,
        ),
        # a grammar row with another control: logits, decided at the build
        (
            dict(structured_outputs=STRUCTURED, allowed_token_ids=EVENS),
            dict(),
            ACCEPT_MODE_LOGITS,
        ),
    ],
)
def test_h7_the_mode_and_the_plan_follow_the_row_and_the_capabilities(
    params_kwargs, caps, expected
):
    kwargs = {"temperature": 0.0, "ignore_eos": True, **params_kwargs}
    caps = dict(caps)
    mask = caps.pop("mask", True)
    model, runner = _mask_runner(mask=mask, **caps)
    sync_harness._add_request(runner, "x", SamplingParams(**kwargs), first_token=3)
    sync_harness._add_request(runner, "g", GREEDY, first_token=21)
    mi = sync_harness._build(
        runner,
        "x",
        "g",
        drafts={r: sync_harness._argmax_drafts(runner, r) for r in ("x", "g")},
    )
    if expected == "masked":
        assert mi.spec_mode == ACCEPT_MODE_FUSED_SAMPLE and mi.spec_mask is not None
    else:
        assert mi.spec_mode == expected and mi.spec_mask is None


def test_a_certificate_row_and_a_mask_row_run_one_synchronous_step():
    model, runner = _mask_runner(top_p=True)
    sync_harness._add_request(
        runner,
        "c",
        SamplingParams(temperature=1.0, top_p=0.9, ignore_eos=True, seed=2),
        first_token=1,
    )
    sync_harness._add_request(
        runner, "a", ctl._params("allowed", temperature=0.0), first_token=9
    )
    _step(runner, "c", "a")
    mi, committed = _step(
        runner,
        "c",
        "a",
        drafts={r: sync_harness._argmax_drafts(runner, r) for r in ("c", "a")},
    )
    assert mi.spec_mask is not None
    assert model.calls[-2:] == [("begin", False), ("finish", True)]
    assert all(token in EVENS for token in committed["a"])
    ss = model.begin_inputs[-1]["spec_sampling"]
    assert bool(((~ss["greedy"]) & (ss["top_k"] <= 0)).any())  # the certificate row


def test_h6_a_logprobs_row_with_a_grammar_only_mask_gets_its_payload_after_the_mask():
    """The device payload is taken after the mask: the host path's logprobs, and a
    masked top-N entry (the device writes -2**99 there) is -inf."""
    model, runner = _mask_runner(logprobs=True)
    params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        logprobs=6,
        structured_outputs=ctl._structured(),
    )
    sync_harness._add_request(runner, "l", params, first_token=1)
    sync_harness._add_request(runner, "g", GREEDY, first_token=21)
    _step(runner, "l", "g")
    seen = 0
    for _ in range(4):
        before = list(runner.requests["l"].output_token_ids)
        drafts = {r: sync_harness._argmax_drafts(runner, r) for r in ("l", "g")}
        mi, committed = _step(runner, "l", "g", drafts=drafts)
        assert mi.spec_mode == ACCEPT_MODE_FUSED_SAMPLE and mi.spec_mask is not None
        assert model.calls[-1] == ("finish", True)
        output = runner.last_output
        sliced = output.logprobs.slice_request(
            output.req_id_to_index["l"], len(committed["l"])
        )
        for position, token in enumerate(committed["l"]):
            history = before + committed["l"][:position]
            assert ctl._grammar_allows(history, token)
            ids = sliced.logprob_token_ids[position].tolist()
            values = torch.as_tensor(sliced.logprobs[position])
            assert ids[0] == token
            assert float(values[0]) > -1e3  # the committed token has real mass
            for id_, value in zip(ids[1:], values[1:]):
                if not ctl._grammar_allows(history, id_):
                    assert float(value) == float("-inf"), (history, id_)
            # a masked token that the candidates hold is -inf, never the mask value
            assert not bool(((values < MASK_VALUE_FLOOR) & ~torch.isinf(values)).any())
            seen += 1
    assert seen >= 4


# endregion

# region The capability is read from the model, and the counters are reported


def test_the_capability_is_read_at_load_and_defaults_to_false():
    assert _load({"supports_spec_device_mask": True})._spec_device_mask is True
    assert _load({"supports_spec_device_mask": False})._spec_device_mask is False
    assert _load({})._spec_device_mask is False


def test_the_capability_is_read_again_after_warmup():
    """The model withdraws it at warmup when it built no mask traces."""
    calls = []
    model = SimpleNamespace(
        model_capabilities={"supports_spec_device_mask": True},
        warmup_model_prefill=lambda **kw: None,
        warmup_model_decode=lambda **kw: calls.append(kw),
    )
    runner = SimpleNamespace(
        model=model,
        kv_caches=None,
        tt_max_batch_size=1,
        max_num_blocks_per_req=1,
        supports_device_grammar=False,
        trace_mode="decode_only",
        _spec_device_mask=True,
    )
    model.model_capabilities["supports_spec_device_mask"] = False  # as the model does
    TTModelRunner.warmup_model(runner)
    assert runner._spec_device_mask is False and calls


def test_the_submission_line_reports_the_mask_counters():
    assert TTAsyncDecodeController._device_masks_report() == ""
    count_mask_step(2)
    count_mask_fallback("empty_row")
    line = TTAsyncDecodeController._device_masks_report()
    assert (
        "device masks: 1 steps, 2 masked rows, 1 logits fallbacks (empty_row=1)" in line
    )


def test_the_mask_adds_no_environment_variable_and_no_config_key():
    import inspect

    from vllm_tt_plugin import spec_masks

    source = inspect.getsource(spec_masks)
    assert "os.environ" not in source and "additional_config" not in source


# endregion
