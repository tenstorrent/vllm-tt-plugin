# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM logprobs from a device verify's ``SpecCandidates``.

A ``fused_sample`` model that declares ``supports_spec_device_logprobs`` gives,
per verify position, its ``M`` largest raw logits with their ids and the raw
``logsumexp`` of the whole vocabulary (``spec_decode.SpecCandidates``). This
module turns that into the ``LogprobsTensors`` the host walk
(``spec_accept.committed_logprobs``) builds from full logits, in the same
layout, for the committed columns only:

- column 0 of a row is the committed token: its raw logprob ``value - lse``
  and its rank (the count of candidate values at or above its own, which is
  what vLLM's ``gather_logprobs`` counts);
- columns ``1..N`` are the top-``N`` ids and logprobs.

The values are the logits' own numbers minus the ``lse``, so they match the
host path to the precision of the ``lse``. Ties inside the top-``N`` list
resolve to the lower id (the order of the candidates); vLLM's ``topk`` picks
an arbitrary id among equal values.

Two cases a token cannot be placed exactly are counted (``device_logprob_stats``),
not raised, because the committed token is still correct and only its logprob
metadata is affected:

- ``lower_bound``: the chosen value equals the last candidate's, so tokens
  past the ``M`` candidates may tie with it. The reported rank is a lower
  bound; the logprob is exact.
- ``missing``: the chosen id is not among the candidates. A model
  declaring the capability never commits such a token (a committed token is
  inside the top ``M`` by construction). The logprob reported is the last
  candidate's value, an upper bound, and the rank is ``M + 1``. It is logged
  as a warning.

The counters are module-level, like ``spec_accept.support_fallback_stats``,
because the walk runs on the reader thread and touches no runner state.
"""

import threading
from typing import TYPE_CHECKING

import torch

from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.spec_accept import _within_count
from vllm_tt_plugin.spec_decode import SpecCandidates

if TYPE_CHECKING:
    from vllm.v1.outputs import LogprobsTensors

logger = init_tt_logger(__name__)


class _DeviceLogprobStats:
    """Counts the steps built from candidates and the tokens not placed exactly."""

    _FIELDS = ("steps", "rank_lower_bound", "missing_id")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            for name in self._FIELDS:
                setattr(self, name, 0)

    def add(self, **counts: int) -> None:
        with self._lock:
            for name, value in counts.items():
                setattr(self, name, getattr(self, name) + int(value))

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {name: getattr(self, name) for name in self._FIELDS}


_STATS = _DeviceLogprobStats()


def device_logprob_stats() -> dict[str, int]:
    """A copy of the counters: ``steps``, ``rank_lower_bound``, ``missing_id``."""
    return _STATS.snapshot()


def device_spec_logprobs(
    candidates: SpecCandidates,
    committed_token_ids: torch.Tensor,
    accepted_counts: torch.Tensor,
    num_logprobs: int,
    asked_rows: torch.Tensor | None = None,
) -> "LogprobsTensors":
    """The committed tokens' logprobs, ranks and top-N ids from ``candidates``.

    ``candidates`` ``[R, W, M]`` and ``committed_token_ids`` ``[R, W]`` cover
    the same rows (the step's live rows) and columns. ``num_logprobs`` is the
    step's largest request, as ``committed_logprobs`` takes it. ``asked_rows``
    is a ``[R]`` bool mask of the rows that asked for logprobs, which is
    the set the counters cover (every row's tensors are built, as on the host
    path); None counts every row.

    Returns the ``LogprobsTensors``. Raises ``ValueError`` for shapes that do
    not agree, and when ``M`` is below ``num_logprobs``.
    """
    # Deferred for the same import cycle as ``committed_logprobs``.
    from vllm.v1.outputs import LogprobsTensors

    rows, width = (int(size) for size in committed_token_ids.shape)
    expected = (rows, width)
    if tuple(candidates.ids.shape[:2]) != expected:
        raise ValueError(
            f"candidates must cover the committed block {expected}: every row and "
            f"every column; got {tuple(candidates.ids.shape)}"
        )
    num_candidates = int(candidates.ids.shape[2])
    if num_candidates < max(int(num_logprobs), 1):
        raise ValueError(
            f"candidates hold {num_candidates} entries per position, too few for "
            f"num_logprobs={num_logprobs} (and the sampled token)"
        )
    if tuple(accepted_counts.shape) != (rows,):
        raise ValueError(
            f"accepted_counts must be ({rows},), got {tuple(accepted_counts.shape)}"
        )
    # Boolean indexing reads row-major, so row 0's committed columns come first,
    # as in ``committed_logprobs``.
    within = _within_count(rows, width, accepted_counts)
    ids = candidates.ids[within].to(torch.int64)
    values = candidates.values[within]
    lse = candidates.lse[within]
    chosen = committed_token_ids[within].to(torch.int64)

    hit = ids == chosen.unsqueeze(1)
    found = hit.any(dim=1)
    position = hit.to(torch.int8).argmax(dim=1, keepdim=True)
    last = values[:, -1:]
    chosen_value = torch.where(found.unsqueeze(1), values.gather(1, position), last)
    rank = torch.where(
        found,
        (values >= chosen_value).sum(dim=1),
        torch.full_like(chosen, num_candidates + 1),
    )
    lower_bound = found & (last.squeeze(1) >= chosen_value.squeeze(1))
    missing = ~found

    if asked_rows is not None:
        row_of = within.nonzero()[:, 0]
        asked = asked_rows.to(torch.bool)[row_of]
        lower_bound, missing = lower_bound & asked, missing & asked

    top = int(num_logprobs)
    tensors = LogprobsTensors(
        torch.cat([chosen.unsqueeze(1), ids[:, :top]], dim=1).to(torch.int32),
        torch.cat(
            [chosen_value - lse.unsqueeze(1), values[:, :top] - lse.unsqueeze(1)],
            dim=1,
        ),
        rank,
        cu_num_generated_tokens=[
            0,
            *accepted_counts.to(torch.int64).cumsum(0).tolist(),
        ],
    )
    _STATS.add(
        steps=1, rank_lower_bound=int(lower_bound.sum()), missing_id=int(missing.sum())
    )
    if bool(missing.any()):
        logger.warning(
            "device logprobs: %d committed token(s) were not among the model's "
            "%d candidates; their logprob is an upper bound",
            int(missing.sum()),
            num_candidates,
        )
    return tensors
