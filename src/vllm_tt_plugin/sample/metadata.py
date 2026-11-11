# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Per-step sampling inputs of the host sampler.

Adapted from vLLM v0.31.0 ``vllm/v1/sample/metadata.py``, without the fields
the TT runner never sets (logprob token ids, speculative token ids, thinking
budget).
"""

from dataclasses import dataclass

import torch

from vllm_tt_plugin.sample.logits_processor import LogitsProcessors


@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor | None
    top_k: torch.Tensor | None

    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: list[list[int]]

    # Shape (batch size, vocab size). True marks a token that is not allowed,
    # so the mask can be applied with ``masked_fill_``.
    allowed_token_ids_mask: torch.Tensor | None

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]

    logitsprocs: LogitsProcessors
