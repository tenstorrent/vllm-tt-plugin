# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Top-k and top-p filtering followed by random sampling, in PyTorch.

Adapted from vLLM v0.31.0 ``vllm/v1/sample/ops/topk_topp_sampler.py``, keeping
only the PyTorch-native sampler: the TT host sampler runs on CPU tensors.
"""

import torch
import torch.nn as nn
from vllm.config.model import LogprobsMode
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p


class TopKTopPSampler(nn.Module):
    """Optional top-k and top-p filtering followed by weighted random sampling.

    The logits tensor may be updated in place.
    """

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ) -> None:
        super().__init__()
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = use_fp64_gumbel

    def forward(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(logits, generators, k, p)

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return (
            random_sample(probs, generators, self.use_fp64_gumbel),
            logits_to_return,
        )


def sample_with_exponential_noise(probs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if q.dtype == probs.dtype:
        scores = probs.div_(q)
    else:
        scores = q.reciprocal_()
        scores.mul_(probs)
    return scores.argmax(dim=-1).view(-1)


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Sample one token per row: argmax of ``probs / q``, ``q`` exponential noise.

    Rows without their own generator share one batched draw; a row with a
    generator then redraws its noise from that generator.
    """
    dtype = torch.float64 if use_fp64_gumbel else probs.dtype
    q = torch.empty(probs.shape, dtype=dtype, device=probs.device)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    for i, generator in generators.items():
        q[i].exponential_(generator=generator)
    return sample_with_exponential_noise(probs, q)
