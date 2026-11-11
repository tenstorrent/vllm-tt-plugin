# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Per-request state that the TT model runner keeps across scheduler steps.

Adapted from ``CachedRequestState`` in vLLM v0.31.0
``vllm/v1/worker/gpu_input_batch.py``, reduced to the fields the TT runner
reads.
"""

from dataclasses import dataclass

import torch
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds


@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    generator: torch.Generator | None

    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]

    mrope_position_delta: int | None = None

    lora_request: LoRARequest | None = None
    prompt_embeds: torch.Tensor | None = None

    pooling_params: PoolingParams | None = None

    def __post_init__(self):
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)
