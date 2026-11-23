# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Provides ranked host payloads shared by packing and decode integration tests."""

from typing import NamedTuple

import torch


class ReportedLogprobs(NamedTuple):
    """Carries one sampled logprob and its reported vocabulary rank per batch row.

    Both tensors have shape ``[B]`` for the same B model-output rows. Keeping
    this payload a tuple exercises attribute dispatch ahead of top-K unpacking.
    """

    sampled_logprobs: torch.Tensor
    selected_token_ranks: torch.Tensor
