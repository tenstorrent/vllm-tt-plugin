# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The token mask of a deferred verify (``vllm_tt_plugin.spec_token_mask``).

A model declaring ``supports_spec_device_mask`` draws each verify column from
the target masked by one packed row. The mask must rule out exactly what the
runner's own ``logits`` walk rules out (``_apply_grammar_bitmask`` and
``apply_speculative_token_filters``), so these tests compare the two on random
controls, then pin the packing, the per-request cache, the per-column bad-word
and ``min_tokens`` rules, and the worst-case refusal.
"""

from __future__ import annotations

import pytest
import torch

from vllm_tt_plugin.spec_accept import (
    SpecTokenFilters,
    _apply_grammar_bitmask,
    apply_speculative_token_filters,
)
from vllm_tt_plugin.spec_token_mask import (
    MaskRowControls,
    SpecMaskCache,
    clear_ids_,
    compose_token_mask,
    mask_may_allow_nothing,
    mask_words,
    pack_allowed_ids,
    static_token_mask,
    unconstrained_mask,
)

V = 200  # not a multiple of 32: the last word has padding bits
W = mask_words(V)
WIDTH = 4  # 1 + K, K = 3


def _unpack(mask: torch.Tensor, vocab_size: int = V) -> torch.Tensor:
    """``[..., vocab_size]`` bool, True where the packed ``[..., W]`` mask allows."""
    bits = torch.arange(32, dtype=torch.int32)
    allowed = ((mask.to(torch.int32).unsqueeze(-1) >> bits) & 1).bool()
    allowed = allowed.reshape(*mask.shape[:-1], -1)
    have = int(allowed.shape[-1])
    if have < vocab_size:
        pad = torch.zeros((*allowed.shape[:-1], vocab_size - have), dtype=torch.bool)
        allowed = torch.cat([allowed, pad], dim=-1)
    return allowed[..., :vocab_size]


def _pack(allowed: torch.Tensor) -> torch.Tensor:
    """xgrammar's packing of a ``[..., V]`` bool, ``[..., ceil(V / 32)]`` int32."""
    words = mask_words(allowed.shape[-1])
    bits = torch.zeros((*allowed.shape[:-1], words * 32), dtype=torch.int64)
    bits[..., : allowed.shape[-1]] = allowed.to(torch.int64)
    packed = (bits.reshape(*allowed.shape[:-1], words, 32) << torch.arange(32)).sum(-1)
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)


def _random_case(seed: int):
    """Random controls for 4 rows, as both the mask and the logits walk read them."""
    gen = torch.Generator().manual_seed(seed)
    rows = 4
    logits = torch.randn(rows, WIDTH, V, generator=gen)
    drafts = torch.randint(0, V, (rows, WIDTH - 1), generator=gen).to(torch.int32)
    valid = torch.randint(0, WIDTH, (rows,), generator=gen).to(torch.int32)
    histories = [
        torch.randint(0, V, (int(n),), generator=gen).tolist()
        for n in torch.randint(0, 7, (rows,), generator=gen)
    ]
    controls, disallowed, bad_words, min_tokens = (
        [],
        torch.zeros(rows, V, dtype=torch.bool),
        {},
        {},
    )
    for row in range(rows):
        draft = [int(t) for t in drafts[row, : int(valid[row])]]
        allowed = None
        if int(torch.randint(0, 2, (1,), generator=gen)):
            size = int(torch.randint(30, 120, (1,), generator=gen))
            allowed = torch.randperm(V, generator=gen)[:size].tolist()
            disallowed[row] = True
            disallowed[row, allowed] = False
        words: list[list[int]] = []
        if int(torch.randint(0, 2, (1,), generator=gen)):
            column = int(torch.randint(0, WIDTH, (1,), generator=gen))
            history = histories[row] + draft[:column]
            ban = int(torch.randint(0, V, (1,), generator=gen))
            # bites at that column
            words.append(history[-2:] + [ban] if len(history) >= 2 else [ban])
            words.append([int(torch.randint(0, V, (1,), generator=gen))])
            words.append([5, 6, 7])  # a word that mostly does not match
            bad_words[row] = words
        remaining, stops = 0, []
        if int(torch.randint(0, 2, (1,), generator=gen)):
            remaining = int(torch.randint(0, 7, (1,), generator=gen))
            stops = torch.randperm(V, generator=gen)[:3].tolist()
            min_tokens[row] = (remaining, tuple(stops))
        controls.append(
            MaskRowControls(
                req_id=f"r{row}",
                allowed_ids=allowed,
                bad_words=words,
                committed=histories[row],
                drafts=draft,
                min_remaining=remaining if stops else 0,
                stop_ids=stops,
            )
        )
    return logits, drafts, valid, histories, controls, disallowed, bad_words, min_tokens


def _random_grammar(seed: int, narrow: bool) -> torch.Tensor:
    """Scheduler rows: rows 0 and 2 constrained at some columns, the others all ones."""
    gen = torch.Generator().manual_seed(1000 + seed)
    packed = _pack(torch.rand(4, WIDTH, V, generator=gen) < 0.6)
    packed[1] = -1
    packed[3] = -1
    packed[2, 1:] = -1  # only column 0 constrained
    if narrow:
        packed = packed[..., : W - 1].contiguous()  # the last word's ids are out
    return packed


@pytest.mark.parametrize("narrow", [False, True], ids=["full", "narrow"])
@pytest.mark.parametrize("seed", range(10))
def test_the_composed_mask_rules_out_what_the_logits_walk_rules_out(seed, narrow):
    logits, drafts, valid, histories, controls, disallowed, bad_words, min_tokens = (
        _random_case(seed)
    )
    grammar = _random_grammar(seed, narrow)
    filters = SpecTokenFilters(
        allowed_token_ids_mask=disallowed,
        bad_words_token_ids=bad_words,
        output_token_ids=histories,
        min_tokens=min_tokens,
    )
    reference = apply_speculative_token_filters(
        _apply_grammar_bitmask(logits, grammar), drafts, valid, filters
    )

    static = static_token_mask(controls, SpecMaskCache(), width=WIDTH, vocab_size=V)
    mask = compose_token_mask(static, grammar)
    ours = logits.masked_fill(~_unpack(mask), float("-inf"))

    assert mask.dtype == torch.int32 and tuple(mask.shape) == (4, WIDTH, W)
    assert torch.equal(ours, reference)


def test_rows_without_a_control_stay_unconstrained():
    controls = [
        MaskRowControls(req_id="a"),
        MaskRowControls(req_id="b", allowed_ids=[3, 4]),
        None,
    ]
    mask = static_token_mask(controls, SpecMaskCache(), width=WIDTH, vocab_size=V)
    assert tuple(mask.shape) == (3, WIDTH, W)
    assert bool((mask[0] == -1).all()) and bool((mask[2] == -1).all())
    assert _unpack(mask[1, 0]).nonzero().reshape(-1).tolist() == [3, 4]
    assert (
        static_token_mask([None, controls[0]], SpecMaskCache(), width=2, vocab_size=V)
        is None
    )
    assert compose_token_mask(None, None) is None
    assert compose_token_mask(mask, None) is mask


@pytest.mark.parametrize("token", [0, 31, 32, 33, V - 1])
def test_the_bit_of_a_token_is_bit_v_mod_32_of_word_v_div_32(token):
    row = pack_allowed_ids([token], V)
    expected = torch.zeros(W, dtype=torch.int64)
    expected[token // 32] = 1 << (token % 32)
    expected = torch.where(expected >= 2**31, expected - 2**32, expected).to(
        torch.int32
    )
    assert torch.equal(row, expected)
    other = unconstrained_mask(words=W)
    clear_ids_(other, [token], V)
    assert int(_unpack(other).sum()) == V - 1
    assert int(_unpack(pack_allowed_ids([1, V + 5, -2], V)).sum()) == 1


def test_a_narrow_grammar_rules_out_the_ids_it_cannot_name_in_its_rows_only():
    grammar = torch.full((2, WIDTH, W - 2), -1, dtype=torch.int32)
    grammar[0, 0, 0] = 0b101
    static = unconstrained_mask(2, WIDTH, words=W)
    mask = compose_token_mask(static, grammar)
    tail = list(range(32, (W - 2) * 32))
    assert _unpack(mask[0, 0]).nonzero().reshape(-1).tolist() == [0, 2, *tail]
    assert _unpack(mask[0, 1]).nonzero().reshape(-1).tolist() == list(
        range((W - 2) * 32)
    )
    assert bool((mask[1] == -1).all()), "an unconstrained row stays unconstrained"
    with pytest.raises(ValueError, match="wider"):
        compose_token_mask(static[..., :2], grammar)


def test_the_static_row_of_a_request_is_cached_until_it_leaves():
    cache = SpecMaskCache()
    controls = MaskRowControls(
        req_id="a", allowed_ids=[1, 2, 3], bad_words=[[2], [8, 9]]
    )
    first = cache.static_row(controls, V)
    assert cache.static_row(controls, V) is first
    # the one-token bad word is in the row, the longer one is not
    assert _unpack(first).nonzero().reshape(-1).tolist() == [1, 3]
    cache.prune(["b"])
    assert len(cache) == 0 and cache.static_row(controls, V) is not first
    assert cache.static_row(MaskRowControls(req_id="b", bad_words=[[8, 9]]), V) is None


def test_a_longer_bad_word_reads_the_output_and_the_drafts_before_the_column():
    controls = MaskRowControls(
        req_id="a", bad_words=[[10, 11, 12]], committed=[4, 10], drafts=[11, 12, 13]
    )
    mask = static_token_mask([controls], SpecMaskCache(), width=WIDTH, vocab_size=V)
    banned = [
        (~_unpack(mask[0, column])).nonzero().reshape(-1).tolist()
        for column in range(WIDTH)
    ]
    # column 0: history [4, 10]; column 1: [.., 10, 11] bans 12; later: no
    assert banned == [[], [12], [], []]


@pytest.mark.parametrize("remaining, columns", [(1, 1), (3, 3), (4, 4), (9, 4)])
def test_the_stop_ids_are_masked_while_the_column_is_below_the_remaining_count(
    remaining, columns
):
    controls = MaskRowControls(
        req_id="a", min_remaining=remaining, stop_ids=[7, 9], drafts=[1, 2, 3]
    )
    mask = static_token_mask([controls], SpecMaskCache(), width=WIDTH, vocab_size=V)
    for column in range(WIDTH):
        allowed = _unpack(mask[0, column])
        assert bool(allowed[7]) is (column >= columns)
        assert bool(allowed[9]) is (column >= columns)
    assert not MaskRowControls(req_id="b", min_remaining=5, stop_ids=()).constrained


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        pytest.param({"structured": True}, False, id="grammar-alone"),
        pytest.param(
            {"structured": True, "bad_words": [[3]]}, True, id="grammar-and-bad-word"
        ),
        pytest.param(
            {"structured": True, "stop_ids": [5], "min_tokens_pending": True},
            True,
            id="grammar-and-min-tokens",
        ),
        pytest.param({"allowed_ids": [3, 4], "bad_words": [[3]]}, False, id="one-left"),
        pytest.param(
            {
                "allowed_ids": [3, 4],
                "bad_words": [[1, 3]],
                "stop_ids": [4],
                "min_tokens_pending": True,
            },
            True,
            id="none-left",
        ),
        pytest.param(
            {"allowed_ids": [3, 4], "stop_ids": [4], "min_tokens_pending": False},
            False,
            id="min-tokens-reached",
        ),
        pytest.param(
            {"bad_words": [[3]], "stop_ids": [4], "min_tokens_pending": True},
            False,
            id="no-allowlist",
        ),
    ],
)
def test_a_row_whose_mask_can_allow_nothing_is_refused_at_the_worst_case(
    kwargs, expected
):
    args = {
        "structured": False,
        "allowed_ids": None,
        "bad_words": (),
        "stop_ids": (),
        "min_tokens_pending": False,
        **kwargs,
    }
    assert mask_may_allow_nothing(**args, vocab_size=V) is expected
