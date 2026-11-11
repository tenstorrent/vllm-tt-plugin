# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The host sampler.

Adapted from vLLM v0.31.0 ``vllm/v1/sample/sampler.py``, without the inputs the
TT runner never sets (logprob token ids, speculative tokens, thinking budget),
and with the custom logits processors applied after the penalties, where
vLLM's V2 interface places them.
"""

import torch
import torch.nn as nn
from vllm.config.model import LogprobsMode
from vllm.v1.outputs import LogprobsTensors, SamplerOutput

from vllm_tt_plugin.sample.bad_words import apply_bad_words
from vllm_tt_plugin.sample.metadata import SamplingMetadata
from vllm_tt_plugin.sample.penalties import apply_all_penalties
from vllm_tt_plugin.sample.topk_topp_sampler import TopKTopPSampler

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):
    """Samples one token per row of host logits, in this order:

    1. If logprobs are requested, take them from the raw logits: their
       log-softmax for ``raw_logprobs``, a copy for ``raw_logits``.
    2. Convert logits to float32.
    3. Apply the allowed token ids mask.
    4. Apply bad words exclusion.
    5. Apply the built-in processors that can change greedy sampling
       (min_tokens, logit_bias).
    6. Apply repetition, frequency and presence penalties.
    7. Apply the custom logits processors.
    8. Sample. Greedy rows take the argmax. Other rows get temperature, the
       built-in processors that cannot change greedy sampling (min_p), top-k
       and top-p, then a random draw.
    9. Gather the logprobs of the top ``max_num_logprobs`` tokens and of the
       sampled token. The sampled token can be among the top tokens, so a
       row carries ``max_num_logprobs`` or ``max_num_logprobs + 1`` entries.
    """

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode, use_fp64_gumbel)
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = use_fp64_gumbel

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # Logprobs come from the logits before penalties and temperature.
        num_logprobs = sampling_metadata.max_num_logprobs
        raw_logprobs: torch.Tensor | None = None
        if num_logprobs is not None:
            if self.logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif self.logprobs_mode == "raw_logits":
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        logits = logits.to(torch.float32)
        logits = self.apply_logits_processors(logits, sampling_metadata)
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        # Token ids index tensors later on, which needs int64.
        sampled = sampled.long()

        logprobs_tensors = None
        if num_logprobs is not None:
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )

        return SamplerOutput(
            # One sampled token per row, shape (batch size, 1).
            sampled_token_ids=sampled.to(torch.int32).unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # Greedy rows carry temperature 0; divide them by 1 instead.
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """
        if sampling_metadata.all_greedy and sampling_metadata.all_random:
            raise ValueError(
                "SamplingMetadata.all_greedy and SamplingMetadata.all_random "
                "are both True"
            )
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if self.logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif self.logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        if sampling_metadata.temperature is None:
            raise ValueError(
                "SamplingMetadata.temperature is None but not every row is greedy"
            )

        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,
        )
        return sampled, processed_logprobs

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: maximum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)

        """
        if token_ids.dtype != torch.int64:
            raise ValueError(f"token_ids must be int64, got {token_ids.dtype}")
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Rank of the token: how many entries are at least as likely.
        token_ranks = (logprobs >= token_logprobs).sum(-1)

        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        output_token_ids = sampling_metadata.output_token_ids

        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)

        custom = sampling_metadata.logitsprocs.custom
        if custom is not None:
            logits = custom.apply(logits)
        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        if sampling_metadata.prompt_token_ids is None:
            raise ValueError(
                "SamplingMetadata.prompt_token_ids is None but penalties are set"
            )
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )
