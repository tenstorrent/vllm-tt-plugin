# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Contract types for speculative decoding between the runner and a TT model.

The contract these types encode is specified in
https://github.com/tenstorrent/vllm-tt-plugin/issues/110. This module holds the
wire surface and the validation of the values that cross it, so a model class
and the runner can agree on shapes and modes before either side implements a
step of the loop. It reads no ``model_capabilities`` key and admits no
configuration: ``normalize_declared_values`` validates a declaration a caller
has already read, and lives here next to the constant sets it validates.

Per the contract, one step is verify then propose: the runner calls
``decode_forward`` over the ``[B, 1+K]`` candidate block, walks acceptance, and
calls ``propose_draft_tokens`` with that same step's hidden state.

Two properties are enforced by these types rather than left to each caller,
because leaving either to a caller invites an off-by-one or a shape error that
only surfaces inside an accept walk: the inclusive range a valid
``accepted_counts`` entry lies in, and the pairing between a ``spec_mode`` and
the fields a verify return must carry for it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

# Accept modes. The first two are named for what the verify call returns, which
# is what the runner branches on; naming the performer instead cannot describe a
# device argmax followed by a host walk, which is what both existing tt-metal
# implementations do.
#
#   "logits"       returns logits [B, 1+K, V]. The host walks acceptance, so
#                  this is the only mode that can serve a request needing host
#                  arbitration: structured output, host logits processors,
#                  min_p, logit_bias, bad_words, allowed_token_ids, min_tokens
#                  or logprobs. It pays a [B, 1+K, V] readback.
#   "argmax_ids"   returns the verify argmax ids [B, 1+K]. The host walks
#                  acceptance greedily, comparing ids, so no logits cross. This
#                  mode is greedy only.
#   "fused_sample" returns accepted ids and counts. The device accepts and
#                  samples: rejection sampling, the residual correction and the
#                  bonus token all happen there, which is the fusion the name
#                  describes. It is the only mode that may be followed by a
#                  verify with accepted_counts=None, because it is the only one
#                  that leaves an authoritative count on the device.
ACCEPT_MODE_LOGITS = "logits"
ACCEPT_MODE_ARGMAX_IDS = "argmax_ids"
ACCEPT_MODE_FUSED_SAMPLE = "fused_sample"
ACCEPT_MODES = frozenset(
    {ACCEPT_MODE_LOGITS, ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_FUSED_SAMPLE}
)

# Where a drafter's own state lives. "internal" means the model allocates it
# and reports its size through the two byte fields of SpecPlan. "paged" means
# the drafter needs a scheduler-owned growing cache, declared through the model
# class's get_kv_cache_spec hook.
DRAFTER_STATE_INTERNAL = "internal"
DRAFTER_STATE_PAGED = "paged"
DRAFTER_STATES = frozenset({DRAFTER_STATE_INTERNAL, DRAFTER_STATE_PAGED})

# What a drafting method needs from the model. The runner maps a vLLM
# SpeculativeConfig method name onto these; a model never enumerates method
# names, so a new upstream method does not require a model change.
SPEC_REQUIREMENT_DEVICE_PROPOSE = "device_propose"
SPEC_REQUIREMENT_HIDDEN_FEED = "hidden_feed"
SPEC_REQUIREMENT_DRAFTER_SCORES = "drafter_scores"
SPEC_REQUIREMENT_PAGED_DRAFTER_CACHE = "paged_drafter_cache"
SPEC_REQUIREMENTS = frozenset(
    {
        SPEC_REQUIREMENT_DEVICE_PROPOSE,
        SPEC_REQUIREMENT_HIDDEN_FEED,
        SPEC_REQUIREMENT_DRAFTER_SCORES,
        SPEC_REQUIREMENT_PAGED_DRAFTER_CACHE,
    }
)

# How the target hidden state reaches a device drafter. Both are live modes: a
# drafter on another mesh, or one that is a separate model instance, needs the
# host round trip even though the first TT implementation retains it on device.
HIDDEN_HANDOFF_ON_DEVICE = "on_device"
HIDDEN_HANDOFF_ROUNDTRIP = "roundtrip"
HIDDEN_HANDOFFS = frozenset({HIDDEN_HANDOFF_ON_DEVICE, HIDDEN_HANDOFF_ROUNDTRIP})

# The contract's name for the target hidden state a device drafter consumes.
# Deliberately unconstrained: the runner holds it and hands it back without
# interpreting its dtype, layout or tensor-parallel fracturing. Named so the
# term is greppable on both sides of the boundary.
HiddenHandle = Any

# Which verify-return fields each mode must carry. Contract information, so a
# caller can check a return it did not build.
MODE_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    ACCEPT_MODE_LOGITS: ("logits",),
    ACCEPT_MODE_ARGMAX_IDS: ("argmax_ids",),
    ACCEPT_MODE_FUSED_SAMPLE: ("accepted_token_ids", "accepted_counts"),
}


@dataclass(frozen=True)
class SpecPlan:
    """What one model can serve at one ``(max_num_seqs, requested_k)`` point.

    Returned by a model class's ``spec_plan`` classmethod at config time. The
    runner budgets with these numbers and never inspects the physical verify
    layout: a model's lane arithmetic, L1 fit and state budget stay private,
    and only their consequences cross.

    ``accept_modes`` is stored as a tuple because the instance is frozen and a
    list field would be shared mutable state on a value object.
    """

    effective_k: int
    # Decode rows one speculating request occupies while verifying its
    # [B, 1+K] block. Unrelated to a lane-DP lane: this counts rows of the
    # model's decode batch, not TT lanes in an engine. The runner checks it
    # against its own row budget and never asks how the rows are arranged.
    lanes_per_request: int
    # Fixed device bytes per speculating request, independent of sequence
    # length: candidate state slots, a conv stash, retained hidden rows.
    extra_bytes_per_seq: int
    # Device bytes per KV token beyond the target KV, which is what a drafter
    # with its own paged KV pair costs, since that cache grows with the
    # sequence. A model reporting only the fixed part under-reserves by the
    # whole drafter cache.
    extra_bytes_per_token: int
    accept_modes: tuple[str, ...]
    drafter_state: str
    # Whether the model also serves a narrow [B, 1] decode alongside the wide
    # [B, 1+K] one. Speculative decode calls are uniformly 1+K wide even on a
    # step where no request carries drafts, so that a model needs one verify
    # shape rather than two; a model that sets this offers a second, narrower
    # shape and the runner prefers it on those steps.
    supports_narrow_decode: bool = False

    def __post_init__(self) -> None:
        if self.effective_k < 1:
            raise ValueError(
                "SpecPlan.effective_k must be at least 1; a model that cannot "
                f"speculate returns SpecReject instead, got {self.effective_k}"
            )
        if self.lanes_per_request < 1:
            raise ValueError(
                "SpecPlan.lanes_per_request must be at least 1, got "
                f"{self.lanes_per_request}"
            )
        for name in ("extra_bytes_per_seq", "extra_bytes_per_token"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"SpecPlan.{name} must not be negative, got {value}")

        modes = normalize_declared_values(
            self.accept_modes, ACCEPT_MODES, "SpecPlan.accept_modes"
        )
        object.__setattr__(self, "accept_modes", modes)
        if not modes:
            raise ValueError("SpecPlan.accept_modes must name at least one mode")
        if self.drafter_state not in DRAFTER_STATES:
            raise ValueError(
                f"SpecPlan.drafter_state {self.drafter_state!r} is not one of "
                f"{sorted(DRAFTER_STATES)}"
            )

    @property
    def accepted_counts_range(self) -> tuple[int, int]:
        """Inclusive range a valid ``accepted_counts`` entry lies in.

        A count, not an index: 1 means only the input token stood and every
        draft was rejected. It is never 0, and it is 1 after a prefill and
        after a non-speculating step, so the runner never special-cases the
        first speculative step. A model selecting a per-candidate state slot
        selects slot ``accepted_counts - 1``.
        """
        return (1, 1 + self.effective_k)

    @property
    def block_width(self) -> int:
        """Row width of every speculative decode call, ``1 + effective_k``.

        Uniform once speculation is on, including on a step where no request
        carries drafts, so a model needs one verify shape and not two.
        """
        return 1 + self.effective_k


@dataclass(frozen=True)
class SpecReject:
    """Why a model cannot serve a ``(max_num_seqs, requested_k)`` point.

    ``supported_k`` carries what the same ``max_num_seqs`` would accept, so the
    config-time error can tell an operator what to ask for instead. It is empty
    when the model can serve no draft length at that concurrency.
    """

    reason: str
    supported_k: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("SpecReject.reason must explain the refusal")
        supported = tuple(self.supported_k)
        object.__setattr__(self, "supported_k", supported)
        bad = [k for k in supported if k < 1]
        if bad:
            raise ValueError(
                "SpecReject.supported_k must hold draft lengths of 1 or more, "
                f"got {bad}"
            )
        if len(set(supported)) != len(supported):
            raise ValueError(
                f"SpecReject.supported_k repeats a value: {list(supported)}"
            )


@dataclass(frozen=True)
class DraftOutput:
    """What a device drafter returns for one step.

    ``draft_token_ids`` is ``[B, K]`` and padded. How many of row ``i``'s
    drafts are real is the runner's own bookkeeping, carried into the verify
    call as ``num_valid_drafts``, not encoded in this tensor.
    ``draft_scores`` is ``[B, K, q]``, the drafter's top ``q`` scores per
    drafted position, for a drafter that produces them and ``None`` otherwise.
    An accept rule that needs the drafter distribution reads them; a runner
    that does not must not require them.
    """

    draft_token_ids: "torch.Tensor"
    draft_scores: "torch.Tensor | None" = None


@dataclass(frozen=True)
class VerifyOutput:
    """What one verify call returns, and the mode that says which fields apply.

    The mode-to-field pairing is enforced rather than documented because a
    return that carries the wrong field for its mode fails later, inside the
    accept walk, where the cause is no longer visible.
    """

    spec_mode: str
    # [B, 1+K] verify argmax ids, for spec_mode "argmax_ids".
    argmax_ids: "torch.Tensor | None" = None
    # [B, 1+K, V] over the whole vocabulary, for spec_mode "logits".
    logits: "torch.Tensor | None" = None
    # [B, 1+K] committed prefix plus correction or bonus, for "fused_sample".
    accepted_token_ids: "torch.Tensor | None" = None
    # [B] committed token count per row, for "fused_sample". Same domain as the
    # accepted_counts a verify takes as input: see SpecPlan.
    accepted_counts: "torch.Tensor | None" = None
    # The target hidden state a device drafter consumes, held by the runner and
    # handed back to propose_draft_tokens uninterpreted. None when the model
    # retains it on device.
    hidden: HiddenHandle = None

    def __post_init__(self) -> None:
        required = MODE_REQUIRED_FIELDS.get(self.spec_mode)
        if required is None:
            raise ValueError(
                f"VerifyOutput.spec_mode {self.spec_mode!r} is not one of "
                f"{sorted(ACCEPT_MODES)}"
            )
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"VerifyOutput for spec_mode {self.spec_mode!r} must carry "
                f"{list(required)}, missing {missing}"
            )


def normalize_declared_values(
    values: Sequence[str] | None, known: frozenset[str], label: str
) -> tuple[str, ...]:
    """Validate a declared capability list against its known value set.

    An absent declaration is empty, per the default-if-absent capability
    convention. A present value that is not known raises, because a typo in a
    capability list would otherwise silently disable the feature it names.
    """
    if values is None:
        return ()
    declared = tuple(values)
    unknown = [value for value in declared if value not in known]
    if unknown:
        raise ValueError(
            f"{label} has unknown values {unknown}; known values are {sorted(known)}"
        )
    if len(set(declared)) != len(declared):
        raise ValueError(f"{label} repeats a value: {list(declared)}")
    return declared


__all__ = [
    "ACCEPT_MODES",
    "ACCEPT_MODE_ARGMAX_IDS",
    "ACCEPT_MODE_FUSED_SAMPLE",
    "ACCEPT_MODE_LOGITS",
    "DRAFTER_STATES",
    "DRAFTER_STATE_INTERNAL",
    "DRAFTER_STATE_PAGED",
    "HIDDEN_HANDOFFS",
    "HIDDEN_HANDOFF_ON_DEVICE",
    "HIDDEN_HANDOFF_ROUNDTRIP",
    "SPEC_REQUIREMENTS",
    "SPEC_REQUIREMENT_DEVICE_PROPOSE",
    "SPEC_REQUIREMENT_DRAFTER_SCORES",
    "SPEC_REQUIREMENT_HIDDEN_FEED",
    "SPEC_REQUIREMENT_PAGED_DRAFTER_CACHE",
    "DraftOutput",
    "MODE_REQUIRED_FIELDS",
    "HiddenHandle",
    "SpecPlan",
    "SpecReject",
    "VerifyOutput",
    "normalize_declared_values",
]
