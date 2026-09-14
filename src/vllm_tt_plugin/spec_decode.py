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
    from vllm.config import VllmConfig

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

# Where a drafter's own state lives. The first two describe a cost the runner
# must budget through the two byte fields of SpecPlan: "internal" means the
# model allocates the state itself, "paged" means the drafter needs a
# scheduler-owned growing cache declared through the model class's
# get_kv_cache_spec hook. The third describes a constraint instead of a cost,
# which is why a byte field alone cannot carry it: a drafter that cross-attends
# into the TARGET's caches reserves nothing but requires things to stay true of
# how those caches are organised, carried in
# SpecPlan.drafter_target_cache_requires.
DRAFTER_STATE_INTERNAL = "internal"
DRAFTER_STATE_PAGED = "paged"
DRAFTER_STATE_SHARED_WITH_TARGET = "shared_with_target"
DRAFTER_STATES = frozenset(
    {
        DRAFTER_STATE_INTERNAL,
        DRAFTER_STATE_PAGED,
        DRAFTER_STATE_SHARED_WITH_TARGET,
    }
)

# What a "shared_with_target" drafter requires of the target's caches. These are
# requirements rather than costs because violating either produces reads from
# the wrong cache positions, not an allocation failure.
#
#   "named_layer_caches"   the drafter reads specific target layers, normally
#                          the last of each attention kind, so those caches
#                          must stay allocated and addressable for the life of
#                          the request.
#   "absolute_positions"   the drafter addresses target cache slots by absolute
#                          position, so a target whose sliding layers are a
#                          bounded ring must apply the same wrap modulo in the
#                          drafter's attention, and the drafter's own window
#                          must match the target's.
DRAFTER_TARGET_CACHE_NAMED_LAYER_CACHES = "named_layer_caches"
DRAFTER_TARGET_CACHE_ABSOLUTE_POSITIONS = "absolute_positions"
DRAFTER_TARGET_CACHE_REQUIREMENTS = frozenset(
    {
        DRAFTER_TARGET_CACHE_NAMED_LAYER_CACHES,
        DRAFTER_TARGET_CACHE_ABSOLUTE_POSITIONS,
    }
)

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

# Tail marker for a fixed-width block of token ids that committed fewer tokens
# than the width. The runner truncates at the first occurrence. A real token id
# must never be used, including the end-of-sequence id: it is indistinguishable
# from a committed token, so it silently turns a short commit into a stop.
# Matches upstream's PLACEHOLDER_TOKEN_ID.
PLACEHOLDER_TOKEN_ID = -1

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
    # Only meaningful for drafter_state "shared_with_target", where both byte
    # fields are zero and what crosses instead is what must stay true of the
    # target's caches. Empty for the two states that carry a cost.
    drafter_target_cache_requires: tuple[str, ...] = ()
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

        requires = normalize_declared_values(
            self.drafter_target_cache_requires,
            DRAFTER_TARGET_CACHE_REQUIREMENTS,
            "SpecPlan.drafter_target_cache_requires",
        )
        object.__setattr__(self, "drafter_target_cache_requires", requires)
        if requires and self.drafter_state != DRAFTER_STATE_SHARED_WITH_TARGET:
            raise ValueError(
                "SpecPlan.drafter_target_cache_requires applies only to "
                f"drafter_state {DRAFTER_STATE_SHARED_WITH_TARGET!r}, but "
                f"drafter_state is {self.drafter_state!r} with requirements "
                f"{list(requires)}"
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
    # Rows that committed fewer than 1+K tokens pad the tail with
    # PLACEHOLDER_TOKEN_ID; the runner truncates at the first one.
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
    "DRAFTER_STATE_SHARED_WITH_TARGET",
    "DRAFTER_STATE_INTERNAL",
    "DRAFTER_STATE_PAGED",
    "DRAFTER_TARGET_CACHE_ABSOLUTE_POSITIONS",
    "DRAFTER_TARGET_CACHE_NAMED_LAYER_CACHES",
    "DRAFTER_TARGET_CACHE_REQUIREMENTS",
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
    "PLACEHOLDER_TOKEN_ID",
    "HiddenHandle",
    "SpecPlan",
    "SpecReject",
    "VerifyOutput",
    "admit_speculative_config",
    "method_requirements",
    "normalize_declared_values",
]


# vLLM speculative methods that draft entirely on the host and ask nothing of
# the model. A method absent from every table below is refused by name at
# config time rather than assumed serviceable.
_HOST_DRAFTER_METHODS = frozenset({"ngram", "suffix"})

# Methods whose drafter runs on device and reads the target's hidden state,
# beyond the EAGLE and MTP family that vLLM groups under EagleModelTypes.
_HIDDEN_FEED_METHODS = frozenset({"medusa", "mlp_speculator"})


def method_requirements(method: str) -> tuple[str, ...]:
    """What a vLLM speculative method requires of the model.

    The plugin owns this mapping, so a new upstream method name is a plugin
    change rather than a model release, and a model declares what it can serve
    without ever enumerating method names.
    """
    # Deferred: this module is imported while vLLM resolves its platform
    # plugin, so importing vLLM's config package at module scope inverts that
    # bootstrap.
    from typing import get_args

    from vllm.config.speculative import EagleModelTypes, SpeculativeMethod

    if method in _HOST_DRAFTER_METHODS:
        return ()
    if method in get_args(EagleModelTypes) or method in _HIDDEN_FEED_METHODS:
        return (SPEC_REQUIREMENT_DEVICE_PROPOSE, SPEC_REQUIREMENT_HIDDEN_FEED)
    if method == "draft_model":
        return (
            SPEC_REQUIREMENT_DEVICE_PROPOSE,
            SPEC_REQUIREMENT_PAGED_DRAFTER_CACHE,
        )
    known = sorted(get_args(SpeculativeMethod))
    raise ValueError(
        f"speculative method {method!r} is not supported by the TT backend. "
        f"Supported: host drafters {sorted(_HOST_DRAFTER_METHODS)}, the EAGLE "
        f"and MTP family, {sorted(_HIDDEN_FEED_METHODS)}, and 'draft_model'. "
        f"vLLM knows {known}"
    )


def admit_speculative_config(
    vllm_config: "VllmConfig",
    model_class: type,
    model_capabilities: dict | None,
    *,
    declared_output_tokens_per_step: int,
) -> SpecPlan | None:
    """Resolve a speculative configuration at config time, or refuse it.

    Returns the model's plan when speculation is admitted, and ``None`` when
    the configuration asks for no speculation. Every refusal raises with the
    offending values; speculation is never disabled silently, because a server
    that quietly serves without it reports speedups it did not achieve.

    ``declared_output_tokens_per_step`` is the width the MODEL declared, read
    before anything stores a speculative width, so the block-output rail can be
    detected by what the model asked for rather than by what speculation would
    later set.
    """
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if not speculative_config:
        return None

    capabilities = model_capabilities or {}
    if not capabilities.get("supports_spec_decode", False):
        raise ValueError(
            f"{model_class.__name__} does not declare "
            "model_capabilities['supports_spec_decode'], so it cannot serve a "
            "speculative_config"
        )

    # Both own output_tokens_per_step, and the block-output rail's machinery
    # (placeholder accounting, neutralized sampling controls, disabled
    # logprobs) is wrong for speculation, which honours sampling through its
    # logits mode instead.
    if declared_output_tokens_per_step > 1:
        raise ValueError(
            f"{model_class.__name__} declares output_tokens_per_step="
            f"{declared_output_tokens_per_step}, which selects the block-output "
            "rail, and a speculative_config was also requested. The two own the "
            "same output width and cannot be combined; pick one"
        )

    method = getattr(speculative_config, "method", None)
    if not method:
        raise ValueError(
            "speculative_config carries no method; the TT backend resolves a "
            "model's requirements from the method name"
        )
    required = method_requirements(str(method))
    declared = normalize_declared_values(
        capabilities.get("spec_requirements"),
        SPEC_REQUIREMENTS,
        f"{model_class.__name__} model_capabilities['spec_requirements']",
    )
    missing = [name for name in required if name not in declared]
    if missing:
        raise ValueError(
            f"speculative method {method!r} requires {list(required)}, but "
            f"{model_class.__name__} declares spec_requirements "
            f"{list(declared)}; missing {missing}"
        )

    handoff = normalize_declared_values(
        capabilities.get("spec_hidden_handoff"),
        HIDDEN_HANDOFFS,
        f"{model_class.__name__} model_capabilities['spec_hidden_handoff']",
    )
    if SPEC_REQUIREMENT_HIDDEN_FEED in required and not handoff:
        raise ValueError(
            f"speculative method {method!r} feeds the target hidden state to "
            f"its drafter, but {model_class.__name__} declares no "
            f"spec_hidden_handoff; expected one of {sorted(HIDDEN_HANDOFFS)}"
        )

    spec_plan = getattr(model_class, "spec_plan", None)
    if spec_plan is None:
        raise ValueError(
            f"{model_class.__name__} declares supports_spec_decode but has no "
            "spec_plan classmethod, so its feasible (max_num_seqs, K) points "
            "cannot be resolved"
        )

    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    requested_k = int(speculative_config.num_speculative_tokens)
    outcome = spec_plan(vllm_config, max_num_seqs, requested_k)

    if isinstance(outcome, SpecReject):
        supported = (
            f"; supported draft lengths at max_num_seqs={max_num_seqs}: "
            f"{list(outcome.supported_k)}"
            if outcome.supported_k
            else f"; no draft length is supported at max_num_seqs={max_num_seqs}"
        )
        raise ValueError(
            f"{model_class.__name__} refused speculation at "
            f"max_num_seqs={max_num_seqs}, num_speculative_tokens="
            f"{requested_k}: {outcome.reason}{supported}"
        )
    if not isinstance(outcome, SpecPlan):
        raise ValueError(
            f"{model_class.__name__}.spec_plan must return SpecPlan or "
            f"SpecReject, got {type(outcome).__name__}"
        )
    if outcome.effective_k > requested_k:
        raise ValueError(
            f"{model_class.__name__}.spec_plan returned effective_k="
            f"{outcome.effective_k}, above the requested "
            f"num_speculative_tokens={requested_k}"
        )

    # Refused until a model implements them, so the first server to reach
    # either path is not the first test of it.
    if ACCEPT_MODE_FUSED_SAMPLE in outcome.accept_modes:
        raise ValueError(
            f"{model_class.__name__}.spec_plan declares accept mode "
            f"{ACCEPT_MODE_FUSED_SAMPLE!r}, which no model implements and the "
            "runner cannot yet drive"
        )
    if outcome.drafter_state == DRAFTER_STATE_PAGED:
        raise ValueError(
            f"{model_class.__name__}.spec_plan declares drafter_state "
            f"{DRAFTER_STATE_PAGED!r}, which needs a scheduler-owned drafter "
            "cache the runner does not yet allocate"
        )
    return outcome
