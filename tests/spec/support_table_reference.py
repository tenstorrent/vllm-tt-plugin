# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The support walk's probability table and walk, built from full logits.

The ``topk`` and ``fused_sample`` modes build their table from a verify's
top-M support (``spec_accept._support_table``). The tests hold that table, and
the walk over it, to the same construction over the full logits here.
"""

from __future__ import annotations

import torch

from vllm_tt_plugin.spec_accept import (
    AcceptResult,
    SpecSamplingInputs,
    _commit_walks,
    _expand_to_columns,
    _greedy_walk,
    _ProbTable,
    _reference_rows,
    _safe_temperature,
    _support_cut,
    _walk_table,
)

# Largest top_k served from candidates, and the extra candidates kept so that
# ties at the k-th value are seen. A row whose ties overflow the margin takes
# the reference path.
TOPK_MAX = 1024
TOPK_MARGIN = 64


def constrained_table(
    target_logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
) -> _ProbTable:
    """The support modes' probability table, built from full logits.

    Same masks as ``apply_top_k_top_p_pytorch`` followed by a softmax: top-k
    keeps every logit ``>=`` the k-th largest (ties kept), then top-p drops
    the ascending-cumsum prefix ``<= 1 - p`` but always keeps the largest. Both
    are evaluated on the top ``k_max + margin`` candidates only, which holds
    every survivor of top-k. Temperature is applied to the candidates, which is
    the same division of the same values; it is monotone, so the candidates of
    the raw logits are the candidates of the tempered ones. Rows the candidates
    cannot decide (top-k off or above the cap with top-p active, tie overflow)
    use the reference path. A top-p cut inside a group of equal logits keeps
    the group's lowest ids, as the support walk does.
    """
    rows, width, vocab = target_logits.shape
    n = rows * width
    flat = target_logits.reshape(n, vocab)
    temp = _expand_to_columns(_safe_temperature(temperature), width).to(torch.float32)
    k = None if top_k is None else _expand_to_columns(top_k, width)
    p = None if top_p is None else _expand_to_columns(top_p, width)

    off = torch.zeros(n, dtype=torch.bool)
    k_on = off if k is None else (k > 0) & (k < vocab)
    p_on = off if p is None else p < 1.0
    active = k_on | p_on
    fast = off if k is None else k_on & (k <= TOPK_MAX)
    fallback = active & ~fast

    table = _ProbTable(n, vocab)
    dense_ids = (~active).nonzero().squeeze(1)
    dense_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    if dense_ids.numel():
        sub = flat[dense_ids].to(torch.float32) / temp[dense_ids].unsqueeze(1)
        dense_rows.append((dense_ids, sub.softmax(dim=-1)))

    if bool(fast.any()):
        ids = fast.nonzero().squeeze(1)
        sub = flat if ids.numel() == n else flat[ids]
        k_f = k[ids].to(torch.long)
        cand = min(int(k_f.max()) + TOPK_MARGIN, vocab)
        vals, idx = torch.topk(sub, cand, dim=-1)
        vals, idx, overflow = _support_cut(
            vals, idx, temp[ids], k_f, None if p is None else p[ids], cand < vocab
        )
        ok = ~overflow
        if bool(ok.any()):
            table.set_sparse(
                ids[ok], idx[ok], vals[ok].softmax(dim=-1, dtype=torch.float32)
            )
        if bool(overflow.any()):
            fallback = fallback.clone()
            fallback[ids[overflow]] = True

    if bool(fallback.any()):
        ids = fallback.nonzero().squeeze(1)
        k_ref = None if k is None else torch.where(k_on, k, vocab)[ids]
        p_ref = None if p is None else p[ids]
        dense_rows.append((ids, _reference_rows(flat[ids], temp[ids], k_ref, p_ref)))

    if dense_rows:
        table.set_dense(
            torch.cat([i for i, _ in dense_rows]), torch.cat([d for _, d in dense_rows])
        )
    return table


def full_logits_walk(
    logits: torch.Tensor,
    drafts: torch.Tensor,
    num_valid: torch.Tensor,
    sampling: SpecSamplingInputs,
) -> AcceptResult:
    """The support modes' walk over the table built from the full logits."""
    rows, width, vocab = logits.shape
    is_greedy = sampling.temperature <= 0.0
    greedy = _greedy_walk(logits, drafts, width - 1) if bool(is_greedy.any()) else None
    random = None
    if bool((~is_greedy).any()):
        table = constrained_table(
            logits, sampling.temperature, sampling.top_k, sampling.top_p
        )
        random = _walk_table(
            table, drafts, num_valid, sampling.generators, rows, width - 1, vocab
        )
    return _commit_walks(is_greedy, greedy, random, num_valid, rows, width)
