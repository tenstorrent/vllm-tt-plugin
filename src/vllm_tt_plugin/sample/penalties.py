# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Presence, frequency and repetition penalties on host logits.

Adapted from vLLM v0.31.0 ``vllm/v1/sample/ops/penalties.py``.
"""

import numpy as np
import torch
from vllm.model_executor.layers.utils import apply_penalties


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """Applies presence, frequency and repetition penalties to the logits."""
    _, vocab_size = logits.shape
    output_tokens_t = _pad_output_token_ids(output_token_ids, vocab_size).to(
        logits.device
    )
    # apply_penalties scatters by token id, and -1 is not one: map -1 entries
    # to the vocab_size pad.
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)
    return apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


def _pad_output_token_ids(
    output_token_ids: list[list[int]], vocab_size: int
) -> torch.Tensor:
    """Right-pad the histories with ``vocab_size``, which is not a token id."""
    max_len = max(map(len, output_token_ids), default=0)
    padded = np.full((len(output_token_ids), max_len), vocab_size, dtype=np.int64)
    for row, tokens in enumerate(output_token_ids):
        padded[row, : len(tokens)] = tokens
    return torch.from_numpy(padded)
