# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Packs host logprob payloads from device sampling for vLLM.

``build_device_logprobs`` selects model-output rows and dispatches the three
supported payload forms. ``build_logprobs_from_topk`` packs an already-selected
candidate set. These functions neither sample tokens nor read from a TT device;
callers supply host tensors and retain ownership of their unmodified inputs.
"""

from __future__ import annotations

__all__ = (
    "build_device_logprobs",
    "build_logprobs_from_topk",
)

from typing import Protocol, cast

import torch
from vllm.v1.outputs import LogprobsTensors


class RankedLogprobs(Protocol):
    """Describes a structural payload carrying sampled logprobs and their ranks.

    Payloads supply the tensor attributes without inheriting from this protocol.
    Read-only properties allow immutable payloads, including named tuples.
    """

    @property
    def sampled_logprobs(self) -> torch.Tensor:
        """One floating-point sampled logprob per model-output row."""
        ...

    @property
    def selected_token_ranks(self) -> torch.Tensor:
        """One integer vocabulary rank per model-output row, including ties."""
        ...


def build_device_logprobs(
    logprobs: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | RankedLogprobs,
    sampled_token_ids: torch.Tensor,
    rows: torch.Tensor,
    max_num_logprobs: int,
) -> LogprobsTensors:
    """Selects payload rows and packs device-sampled logprobs for vLLM.

    Parameters
    ----------
    logprobs : torch.Tensor, tuple of torch.Tensor, or RankedLogprobs
        Host payload covering the model output's batch rows. A bare floating
        tensor contains one sampled logprob per row. A pair
        ``(top_k_logprobs, top_k_indices)`` contains sorted candidate tensors
        described by ``build_logprobs_from_topk``. An object with tensor
        attributes ``sampled_logprobs`` and ``selected_token_ranks`` contains
        one sampled logprob and its model-reported rank per row. Rank is the
        number of vocabulary scores at or above the sampled score, ties
        included. The attribute form takes precedence over tuple handling,
        so a named tuple can carry reported ranks. When that object also has
        ``top_logprobs`` and ``top_token_ids`` tensors of shape ``[B, N]``,
        each row's ``N`` largest logprobs in descending order and their ids
        (``spec_decode.SpecLogprobs``, flattened to one row per token), the
        first ``max_num_logprobs`` of them follow the sampled token.
    sampled_token_ids : torch.Tensor
        Host integer IDs of shape ``[R]`` or ``[R, 1]``, already selected and
        aligned with ``rows``. These IDs are not indexed again.
    rows : torch.Tensor
        Host integer row indices of shape ``[R]`` into the payload's batch
        dimension. Their order determines the output row order.
    max_num_logprobs : int
        Number of top candidates to include for a top-K payload; see
        ``build_logprobs_from_topk``, or for a ranked payload that carries
        top candidates. Ignored for the other sampled-logprob payloads.

    Returns
    -------
    LogprobsTensors
        Top-K packing follows ``build_logprobs_from_topk``. Sampled-logprob
        payloads produce int32 token IDs ``[R, 1]``, float32 natural-log
        probabilities ``[R, 1]``, and int32 ranks ``[R]``; a ranked payload
        with top candidates widens the first two to ``[R, 1 + N]`` for
        ``N = max_num_logprobs``. A bare tensor has unknown rank ``-1``;
        reported ranks are preserved. The token-ID tensor can share storage
        with ``sampled_token_ids`` when it is already int32.

    Raises
    ------
    ValueError
        A ranked payload with top candidates carries fewer than
        ``max_num_logprobs`` of them.

    Examples
    --------
    Select payload rows while supplying sampled IDs in that same order.
    This example uses only CPU tensors and needs no TT hardware.

    >>> from types import SimpleNamespace
    >>> import torch
    >>> from vllm_tt_plugin.logprobs import build_device_logprobs
    >>> payload = SimpleNamespace(
    ...     sampled_logprobs=torch.tensor([-0.25, -1.5, -0.75]),
    ...     selected_token_ranks=torch.tensor([1, 7, 3]),
    ... )
    >>> result = build_device_logprobs(
    ...     payload,
    ...     torch.tensor([43, 42], dtype=torch.int32),
    ...     torch.tensor([2, 1]),
    ...     max_num_logprobs=0,
    ... )
    >>> result.logprob_token_ids.tolist()
    [[43], [42]]
    >>> result.logprobs.tolist()
    [[-0.75], [-1.5]]
    >>> result.selected_token_ranks.tolist()
    [3, 7]
    """
    num_rows = sampled_token_ids.shape[0]

    reported_ranks: torch.Tensor | None = getattr(
        logprobs, "selected_token_ranks", None
    )
    if reported_ranks is not None:
        token_ids = sampled_token_ids.reshape(num_rows, 1).to(torch.int32)
        values = (
            cast(RankedLogprobs, logprobs)
            .sampled_logprobs[rows]
            .reshape(num_rows, 1)
            .to(torch.float32)
        )
        top_logprobs = getattr(logprobs, "top_logprobs", None)
        if top_logprobs is not None and max_num_logprobs > 0:
            if max_num_logprobs > top_logprobs.shape[-1]:
                raise ValueError(
                    f"max_num_logprobs={max_num_logprobs} exceeds the "
                    f"{top_logprobs.shape[-1]} top logprobs the payload carries"
                )
            top_ids = logprobs.top_token_ids[rows, :max_num_logprobs]
            token_ids = torch.cat([token_ids, top_ids.to(torch.int32)], dim=1)
            values = torch.cat(
                [values, top_logprobs[rows, :max_num_logprobs].to(torch.float32)],
                dim=1,
            )
        return LogprobsTensors(
            logprob_token_ids=token_ids,
            logprobs=values,
            selected_token_ranks=reported_ranks[rows].reshape(num_rows).to(torch.int32),
        )

    if isinstance(logprobs, tuple):
        top_k_logprobs, top_k_indices = logprobs
        return build_logprobs_from_topk(
            logprobs=top_k_logprobs[rows],
            indices=top_k_indices[rows],
            sampled_token_ids=sampled_token_ids,
            max_num_logprobs=max_num_logprobs,
        )

    sampled_logprobs = cast(torch.Tensor, logprobs)[rows].reshape(num_rows)
    return LogprobsTensors(
        logprob_token_ids=sampled_token_ids.reshape(num_rows, 1).to(torch.int32),
        logprobs=sampled_logprobs.reshape(num_rows, 1).to(torch.float32),
        selected_token_ranks=torch.full((num_rows,), -1, dtype=torch.int32),
    )


def build_logprobs_from_topk(
    logprobs: torch.Tensor,
    indices: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    max_num_logprobs: int,
) -> LogprobsTensors:
    """Packs sampled tokens and a prefix of their sorted top-K candidates.

    Parameters
    ----------
    logprobs : torch.Tensor
        The ``K > 0`` highest host floating-point natural-log probabilities
        per row, of shape ``[B, K]`` and sorted in descending order along
        the candidate dimension.
    indices : torch.Tensor
        Host integer token IDs of shape ``[B, K]``, matching the logprobs.
        Each row has unique IDs and includes that row's sampled token.
    sampled_token_ids : torch.Tensor
        Host integer sampled token IDs of shape ``[B]`` or ``[B, 1]``, in
        candidate-row order. Unsigned device token IDs are accepted.
    max_num_logprobs : int
        Number ``N`` of leading candidates to return, with ``0 <= N <= K``.
        Zero returns only the sampled token. The caller chooses the request's
        limit; this function does not impose an API-specific maximum.

    Returns
    -------
    LogprobsTensors
        New int32 token IDs and float32 logprobs of shape ``[B, N + 1]``.
        Column zero holds the sampled token and its logprob; the remaining
        columns preserve the first ``N`` candidates, including any duplicate
        of the sampled token. Int32 ranks have shape ``[B]``. Input row order
        is preserved and no input tensor is modified.

    Notes
    -----
    Rank counts all ``K`` supplied scores at or above the sampled score,
    including ties, independently of ``N``. This is the exact vocabulary rank
    unless the sampled score ties the lowest supplied score: omitted vocabulary
    scores can tie it too, so the returned count is then a lower bound.

    Examples
    --------
    The sampled token can also appear among the requested candidates. Its
    rank includes tied candidates beyond that prefix. No TT hardware is needed.

    >>> import torch
    >>> from vllm_tt_plugin.logprobs import build_logprobs_from_topk
    >>> result = build_logprobs_from_topk(
    ...     torch.tensor([[-1.0, -1.0, -2.0]]),
    ...     torch.tensor([[11, 12, 13]], dtype=torch.int32),
    ...     torch.tensor([11], dtype=torch.uint32),
    ...     max_num_logprobs=1,
    ... )
    >>> result.logprob_token_ids.tolist(), result.logprobs.tolist()
    ([[11, 11]], [[-1.0, -1.0]])
    >>> result.selected_token_ranks.tolist()
    [2]
    """
    num_rows = logprobs.shape[0]

    if sampled_token_ids.dim() == 1:
        sampled_token_ids = sampled_token_ids.unsqueeze(-1)

    # Normalize integer dtypes before comparing signed and unsigned IDs.
    sampled_ids_int64 = sampled_token_ids.to(torch.int64)
    candidate_matches = indices.to(torch.int64) == sampled_ids_int64

    sampled_positions = candidate_matches.int().argmax(dim=-1)
    if sampled_positions.dim() < logprobs.dim():
        sampled_positions = sampled_positions.unsqueeze(-1)

    sampled_logprobs = logprobs.gather(1, sampled_positions.long())
    selected_token_ranks = (logprobs >= sampled_logprobs).sum(dim=-1)

    logprob_token_ids = torch.zeros(num_rows, max_num_logprobs + 1, dtype=torch.int32)
    logprobs_values = torch.zeros(num_rows, max_num_logprobs + 1, dtype=torch.float32)
    logprob_token_ids[:, 0] = sampled_token_ids.squeeze(-1)
    logprobs_values[:, 0] = sampled_logprobs.squeeze(-1)
    logprob_token_ids[:, 1 : max_num_logprobs + 1] = indices[:, :max_num_logprobs].to(
        torch.int32
    )
    logprobs_values[:, 1 : max_num_logprobs + 1] = logprobs[:, :max_num_logprobs].to(
        torch.float32
    )

    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs_values,
        selected_token_ranks=selected_token_ranks.to(torch.int32),
    )
