# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import torch
from vllm.v1.outputs import LogprobsTensors


def build_logprobs_from_topk(
    top_k_logprobs: torch.Tensor,
    top_k_indices: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    max_num_logprobs: int,
) -> LogprobsTensors:
    """Build LogprobsTensors from device top-K logprobs.

    Device always computes top-32 logprobs sorted descending.
    This function trims to max_num_logprobs which is in range (0-20) to
    match the OpenAI API limit, then packs into LogprobsTensors format
    expected by the downstream vLLM pipeline.
    """
    sz = top_k_logprobs.shape[0]
    n = max_num_logprobs

    # Cast both tensors to int64 to avoid uint32/int32 promotion errors.
    if sampled_token_ids.dim() == 1:
        sampled_token_ids = sampled_token_ids.unsqueeze(-1)
    sampled_expanded = sampled_token_ids.to(torch.int64)
    # ttnn.sampling selects from the same fixed top-32 candidate set returned
    # here. If that device invariant changes, the sampled logprob can be wrong.
    match_mask = top_k_indices.to(torch.int64) == sampled_expanded
    positions = match_mask.int().argmax(dim=-1)

    if positions.dim() < top_k_logprobs.dim():
        positions = positions.unsqueeze(-1)
    sampled_logprob = top_k_logprobs.gather(1, positions.long())
    # vLLM's rank is the number of vocabulary scores at or above the sampled
    # token's, ties included, so it is 1-based. Every token scoring above the
    # lowest returned score is in the returned set, which makes the count exact
    # unless the sampled token ties that lowest score: tokens outside the set can
    # tie too, and the count is then the number of returned ties, a lower bound.
    ranks = (top_k_logprobs >= sampled_logprob).sum(dim=-1)

    logprob_token_ids = torch.zeros(sz, n + 1, dtype=torch.int32)
    logprobs_values = torch.zeros(sz, n + 1, dtype=torch.float32)
    logprob_token_ids[:, 0] = sampled_token_ids.squeeze(-1)
    logprobs_values[:, 0] = sampled_logprob.squeeze(-1)
    logprob_token_ids[:, 1 : n + 1] = top_k_indices[:, :n].to(torch.int32)
    logprobs_values[:, 1 : n + 1] = top_k_logprobs[:, :n].to(torch.float32)

    selected_token_ranks = ranks.to(torch.int32)
    return LogprobsTensors(
        logprob_token_ids,
        logprobs_values,
        selected_token_ranks,
    )


def build_device_logprobs(
    tt_log_probs: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    sampled_token_ids: torch.Tensor,
    rows: torch.Tensor,
    max_num_logprobs: int,
) -> LogprobsTensors:
    """Pack logprobs for device-sampled tokens at ``rows``.

    ``sampled_token_ids`` is the already-selected ``[n]`` sampled tokens (one
    per row); ``rows`` indexes the device output's batch dimension. Covers the
    top-K device path (gpt-oss returns a sorted top-32 set) and the
    single-sampled-logprob path (one logprob per row).

    A single logprob says nothing about where the token ranks in the
    vocabulary, so only the model can report that rank. A payload with
    ``sampled_logprobs`` and ``selected_token_ranks`` attributes (a named tuple
    works) carries it: the rank is the number of vocabulary scores at or above
    the sampled token's, ties included. A bare logprob tensor still reports
    ``-1``, because no correct rank can be derived from it.
    """
    n = sampled_token_ids.shape[0]
    reported_ranks = getattr(tt_log_probs, "selected_token_ranks", None)
    if reported_ranks is not None:
        return LogprobsTensors(
            logprob_token_ids=sampled_token_ids.reshape(n, 1).to(torch.int32),
            logprobs=tt_log_probs.sampled_logprobs[rows]
            .reshape(n, 1)
            .to(torch.float32),
            selected_token_ranks=reported_ranks[rows].reshape(n).to(torch.int32),
        )
    if isinstance(tt_log_probs, tuple):
        top_k_logprobs, top_k_indices = tt_log_probs
        return build_logprobs_from_topk(
            top_k_logprobs=top_k_logprobs[rows],
            top_k_indices=top_k_indices[rows],
            sampled_token_ids=sampled_token_ids,
            max_num_logprobs=max_num_logprobs,
        )
    sampled_log_probs = tt_log_probs[rows].reshape(n)
    return LogprobsTensors(
        logprob_token_ids=sampled_token_ids.reshape(n, 1).to(torch.int32),
        logprobs=sampled_log_probs.reshape(n, 1).to(torch.float32),
        selected_token_ranks=torch.full((n,), -1, dtype=torch.int32),
    )
