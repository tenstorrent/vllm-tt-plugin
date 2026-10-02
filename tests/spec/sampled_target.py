# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A target with a real distribution, the same one for every kind of step.

``DeterministicTarget`` answers with an argmax, which is all a greedy walk can
use. A sampled walk needs the target's whole distribution, and it needs that
distribution to be independent of what was drafted, or a speculated run and an
ordinary sampled run would be drawn from different models.

So every answer here reads one causal rule. After ``token`` at ``position``
the target puts its mass on four tokens, unequally:

    base = (token * 31 + position * 7 + 11) % vocab
    support = base, base + 3, base + 7, base + 11   (mod vocab)
    probabilities 0.4, 0.3, 0.2, 0.1, and 0 everywhere else

An ordinary decode returns ``[B, 1, V]`` logits for each row's single input
token. A verify returns ``[B, 1+K, V]`` logits, column ``j`` from input column
``j``, in ``logits`` mode, or the per-column argmax, which is ``base``, in
``argmax_ids`` mode. Nothing reads ``num_valid_drafts`` or the drafts as
anything but inputs.

The zero-probability tokens matter: a walk that commits a draft the target
rules out produces a token outside every support, which no amount of sampling
noise explains.
"""

from __future__ import annotations

import math

import torch

from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    DRAFTER_STATE_INTERNAL,
    SpecPlan,
    VerifyOutput,
    check_spec_side_tensors,
)

SAMPLED_VOCAB_SIZE = 64
SUPPORT_OFFSETS = (0, 3, 7, 11)
SUPPORT_PROBABILITIES = (0.4, 0.3, 0.2, 0.1)


def base_token(token: int, position: int) -> int:
    return (int(token) * 31 + int(position) * 7 + 11) % SAMPLED_VOCAB_SIZE


def support(token: int, position: int) -> list[int]:
    """The tokens the target can follow ``token`` at ``position`` with, by rank."""
    base = base_token(token, position)
    return [(base + offset) % SAMPLED_VOCAB_SIZE for offset in SUPPORT_OFFSETS]


def rule_logits(tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``[..., V]`` log-probabilities after each token at its position."""
    base = (tokens.to(torch.int64) * 31 + positions.to(torch.int64) * 7 + 11) % (
        SAMPLED_VOCAB_SIZE
    )
    logits = torch.full((*base.shape, SAMPLED_VOCAB_SIZE), float("-inf"))
    for offset, probability in zip(SUPPORT_OFFSETS, SUPPORT_PROBABILITIES):
        index = ((base + offset) % SAMPLED_VOCAB_SIZE).unsqueeze(-1)
        logits.scatter_(-1, index, math.log(probability))
    return logits


class SampledTarget:
    """Serves an ordinary decode and a verify in either mode, by one rule."""

    vocab_size = SAMPLED_VOCAB_SIZE
    accept_modes: tuple[str, ...] = (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS)
    supports_narrow_decode = False
    model_capabilities = {"supports_spec_decode": True, "spec_requirements": []}

    def __init__(self) -> None:
        self.verify_modes: list[str] = []
        self.plain_calls = 0

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs: int, requested_k: int):
        del vllm_config, max_num_seqs
        return SpecPlan(
            effective_k=requested_k,
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=cls.accept_modes,
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=cls.supports_narrow_decode,
        )

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        rows = int(tokens.shape[0])
        if spec_mode is None:
            self.plain_calls += 1
            positions = start_pos.reshape(rows, -1)[:, :1]
            return rule_logits(tokens.reshape(rows, -1)[:, :1], positions)
        if spec_mode not in self.accept_modes:
            raise ValueError(f"SampledTarget does not serve {spec_mode!r}")
        check_spec_side_tensors(
            kwargs["num_valid_drafts"],
            kwargs["accepted_counts"],
            rows,
            int(tokens.shape[1]) - 1,
        )
        self.verify_modes.append(spec_mode)
        logits = rule_logits(tokens, start_pos)
        return self.answer(spec_mode, logits)

    def answer(self, spec_mode: str, logits: torch.Tensor) -> VerifyOutput:
        if spec_mode == ACCEPT_MODE_LOGITS:
            return VerifyOutput(spec_mode=spec_mode, logits=logits)
        return VerifyOutput(
            spec_mode=spec_mode, argmax_ids=logits.argmax(dim=-1).to(torch.int32)
        )


def sampled_target(**knobs) -> type[SampledTarget]:
    """A configured subclass, so one test cannot configure the next."""
    return type("ConfiguredSampledTarget", (SampledTarget,), dict(knobs))
