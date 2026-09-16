# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Contract types for speculative decoding between the runner and a TT model.

The contract these types encode is specified in
https://github.com/tenstorrent/vllm-tt-plugin/issues/110. This module holds the
wire surface and the validation of the values that cross it, so a model class
and the runner can agree on shapes and modes before either side implements a
step of the loop. It reads no ``model_capabilities`` key, admits no
configuration and imports nothing from vLLM at run time:
``normalize_declared_values`` validates a declaration a caller has already
read, and lives here next to the constant sets it validates. The plugin's own
admission policy lives in ``spec_admission``, which imports from here.

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

    Returned by a model class's ``spec_plan`` classmethod at config time.
    Resource fields describe the model's requirements, but the plugin does not
    yet enforce their row or byte budgets. The physical verify layout stays
    private to the model.

    ``accept_modes`` is stored as a tuple because the instance is frozen and a
    list field would be shared mutable state on a value object.
    """

    effective_k: int
    # Decode rows one speculating request occupies while verifying its
    # [B, 1+K] block. Unrelated to a lane-DP lane: this counts rows of the
    # model's decode batch, not TT lanes in an engine. The plugin validates
    # this declaration but does not yet check it against a row budget.
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
        """Row width of a **wide** speculative decode call, ``1 + effective_k``.

        Every speculative step is this wide, including a step where no request
        carries drafts, so a model needs one verify shape and not two. The one
        exception is a model that sets ``supports_narrow_decode``: it also
        receives the plain decode's own shapes on a draftless step, and this
        property does not describe that call.
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
    if isinstance(values, str):
        raise ValueError(
            f"{label} must be a list of values, not the single string "
            f"{values!r}; wrap it in a list"
        )
    declared = tuple(values)
    unknown = [value for value in declared if value not in known]
    if unknown:
        raise ValueError(
            f"{label} has unknown values {unknown}; known values are {sorted(known)}"
        )
    if len(set(declared)) != len(declared):
        raise ValueError(f"{label} repeats a value: {list(declared)}")
    return declared


def check_spec_side_tensors(
    num_valid_drafts: "torch.Tensor | None",
    accepted_counts: "torch.Tensor | None",
    rows: int,
    num_drafts: int,
    call: str = "verify",
) -> None:
    """Validate the two ``[B]`` tensors a verify receives beside its block.

    Lives here, in the contract module both sides import, because every model
    implementing the contract has to check the same domain and a model that
    checks a weaker one is a poor witness for it. Two independent copies of
    this drifted apart once already.

    ``num_valid_drafts`` says how many of a row's draft columns are real, in
    ``[0, num_drafts]``. ``accepted_counts`` says how many tokens that row's
    previous step committed, in ``[1, 1 + num_drafts]``: a count and not an
    index, so 0 is never valid and a model reading ``accepted_counts - 1`` to
    select a candidate state never indexes -1.

    Both are int32, because the runner builds them that way and a float or a
    wider integer here means the caller built something else.
    """
    import torch

    for name, tensor, low, high in (
        ("num_valid_drafts", num_valid_drafts, 0, num_drafts),
        ("accepted_counts", accepted_counts, 1, 1 + num_drafts),
    ):
        if tensor is None:
            raise ValueError(
                f"{call} {name} must be present; only accepted_counts may be "
                "None, and only after a fused_sample step left an "
                "authoritative count on the device"
            )
        if tensor.shape != (rows,):
            raise ValueError(
                f"{call} {name} must be [{rows}], got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.int32:
            raise ValueError(f"{call} {name} must be int32, got {tensor.dtype}")
        out_of_range = tensor[(tensor < low) | (tensor > high)]
        if out_of_range.numel():
            raise ValueError(
                f"{call} {name} entries must lie in [{low}, {high}], got "
                f"{out_of_range.tolist()}"
            )


def accept_greedy_drafts(
    argmax_ids: "torch.Tensor",
    draft_token_ids: "torch.Tensor",
    num_valid_drafts: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Walk greedy acceptance over one verify's ``argmax_ids``.

    The accept mode ``"argmax_ids"`` returns what the target model would have
    chosen at each of the ``1+K`` candidate positions, and greedy acceptance
    takes each draft that matches and stops at the first that does not. The
    target's own choice at the position that rejected commits in the rejected
    draft's place, so a row that rejects its very first draft still commits one
    token and the count is never 0.

    A row that accepts all of its drafts commits one more, the bonus, which is
    the argmax at the column past its last draft. The uniform ``1+K`` width is
    what makes that column already present.

    Args:
        argmax_ids: ``[B, 1+K]``.
        draft_token_ids: ``[B, K]``. Entries past a row's count are padding and
            are not read as candidates.
        num_valid_drafts: ``[B]`` in ``[0, K]``.

    Returns:
        ``(committed_token_ids, accepted_counts)``. The ids are ``[B, 1+K]``
        int32, each row holding its committed prefix followed by
        ``PLACEHOLDER_TOKEN_ID``; the counts are ``[B]`` int32 in ``[1, 1+K]``.
    """
    import torch

    if argmax_ids.dim() != 2:
        raise ValueError(
            f"accept_greedy_drafts argmax_ids must be 2-D [B, 1+K], got "
            f"{tuple(argmax_ids.shape)}"
        )
    rows, width = argmax_ids.shape
    num_drafts = width - 1
    if draft_token_ids.shape != (rows, num_drafts):
        raise ValueError(
            f"accept_greedy_drafts draft_token_ids must be [{rows}, "
            f"{num_drafts}] to match argmax_ids {tuple(argmax_ids.shape)}, got "
            f"{tuple(draft_token_ids.shape)}"
        )
    if num_valid_drafts.shape != (rows,):
        raise ValueError(
            f"accept_greedy_drafts num_valid_drafts must be [{rows}], got "
            f"{tuple(num_valid_drafts.shape)}"
        )

    ids = argmax_ids.to(torch.int32)
    if num_drafts == 0:
        # A narrow step: the model was handed one column because no row carried
        # a draft, so every row commits that column and nothing else. Handled
        # before the walk because an argmax over a zero-width reduction raises.
        return ids, torch.ones(rows, dtype=torch.int32)
    matched = ids[:, :num_drafts] == draft_token_ids.to(torch.int32)
    # A column past a row's own count holds padding, not a candidate, so it can
    # neither be accepted nor reject the row.
    valid = torch.arange(num_drafts).unsqueeze(0) < num_valid_drafts.unsqueeze(1)
    if bool((draft_token_ids.eq(PLACEHOLDER_TOKEN_ID) & valid).any()):
        # The padding marker inside a row's own count is not a token the model
        # can have chosen, so nothing downstream can catch it: the model is
        # handed the same column, returns it unchanged, the comparison above
        # matches, and the marker commits as an output token. The caller
        # counted more drafts than it delivered.
        raise ValueError(
            "accept_greedy_drafts was given PLACEHOLDER_TOKEN_ID as a draft "
            "inside a row's num_valid_drafts prefix: num_valid_drafts "
            f"{num_valid_drafts.tolist()} against draft_token_ids "
            f"{draft_token_ids.tolist()}"
        )
    rejected = valid & ~matched
    # argmax over an all-False row returns 0, so the index is only meaningful
    # once a rejection is known to exist.
    first_rejection = rejected.to(torch.int8).argmax(dim=1)
    counts = torch.where(
        rejected.any(dim=1), first_rejection + 1, num_valid_drafts.to(torch.int64) + 1
    ).to(torch.int32)

    columns = torch.arange(width).unsqueeze(0).expand(rows, width)
    committed = torch.where(
        columns < counts.to(torch.int64).unsqueeze(1),
        ids,
        torch.full_like(ids, PLACEHOLDER_TOKEN_ID),
    )
    return committed, counts


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
    "accept_greedy_drafts",
    "check_spec_side_tensors",
    "normalize_declared_values",
]
