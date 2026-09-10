# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A host model class that implements the speculative-decoding contract.

The contract is specified in
https://github.com/tenstorrent/vllm-tt-plugin/issues/110. No tt-metal model
implements it yet, so the runner side has nothing to be driven against. This
stand-in implements it in host torch with deterministic arithmetic, which makes
it two things at once: a driver for the runner-side work, and a check on the
contract itself, because it raises on every input the contract forbids rather
than tolerating it.

One step, per the contract, is verify then propose: ``decode_forward`` runs the
``[B, 1+K]`` candidate block, the caller walks acceptance, and
``propose_draft_tokens`` then drafts from that same step's hidden state.

``argmax_ids`` and ``logits`` are both served because the mode dispatch is the
runner code most likely to be wrong, and a second mode costs nothing here. A
test pins that the two return the same ids, so a test of one is evidence about
the other.

``fused_sample`` is refused because no implementation serves it today, on
either side. The contract keeps the mode for a model that will; a stand-in that
served it would give a passing test for a path nothing can run.
"""

import types

import torch

from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    DRAFTER_STATE_INTERNAL,
    HIDDEN_HANDOFF_ON_DEVICE,
    SPEC_REQUIREMENT_DEVICE_PROPOSE,
    SPEC_REQUIREMENT_HIDDEN_FEED,
    DraftOutput,
    SpecPlan,
    SpecReject,
    VerifyOutput,
)

# Small enough to build dense logits for in a test, wide enough that a drafted
# id and a verified id do not collide by accident.
FAKE_VOCAB_SIZE = 512


class FakeSpecModel:
    """Contract-conformant stand-in for a resolved TT model class.

    Knobs are class attributes because ``spec_plan`` is a classmethod that the
    runner calls at config time, before any instance exists. Build a configured
    variant with :func:`make_fake_spec_model` rather than mutating these, so
    one test cannot configure the next.
    """

    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": [
            SPEC_REQUIREMENT_DEVICE_PROPOSE,
            SPEC_REQUIREMENT_HIDDEN_FEED,
        ],
        "spec_hidden_handoff": [HIDDEN_HANDOFF_ON_DEVICE],
    }

    # Defaults mirror the Qwen3.6 calibration the contract records, so a test
    # that exercises a realistic refusal does not have to invent numbers: a
    # discrete supported set, single-user concurrency, and a state cost of the
    # right order. The byte costs are flat rather than the calibration's
    # K x 36 MiB, because nothing here budgets against them.
    supported_k: tuple[int, ...] = (3, 7, 11)
    max_supported_num_seqs: int = 1
    accept_modes: tuple[str, ...] = (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS)
    drafter_state: str = DRAFTER_STATE_INTERNAL
    extra_bytes_per_seq: int = 36 << 20
    extra_bytes_per_token: int = 1 << 10
    vocab_size: int = FAKE_VOCAB_SIZE
    # How many of the K drafts the verify agrees with. The rest diverge, so a
    # test can predict the accepted count exactly. None accepts every draft.
    accept_depth: int | None = None

    def __init__(self) -> None:
        self.propose_calls: list[dict] = []
        self.verify_calls: list[dict] = []

    # ---- config time -----------------------------------------------------

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs: int, requested_k: int):
        del vllm_config  # the stand-in's feasibility does not depend on it
        if max_num_seqs > cls.max_supported_num_seqs:
            return SpecReject(
                reason=(
                    f"speculation needs max_num_seqs <= {cls.max_supported_num_seqs}, "
                    f"got {max_num_seqs}"
                ),
                supported_k=(),
            )
        usable = tuple(k for k in sorted(cls.supported_k) if k <= requested_k)
        if not usable:
            return SpecReject(
                reason=(
                    f"requested_k {requested_k} is below every supported draft "
                    f"length {sorted(cls.supported_k)}"
                ),
                supported_k=tuple(sorted(cls.supported_k)),
            )
        effective_k = usable[-1]
        return SpecPlan(
            effective_k=effective_k,
            lanes_per_request=effective_k + 1,
            extra_bytes_per_seq=cls.extra_bytes_per_seq,
            extra_bytes_per_token=cls.extra_bytes_per_token,
            accept_modes=cls.accept_modes,
            drafter_state=cls.drafter_state,
            supports_narrow_decode=False,
        )

    # ---- the primitives --------------------------------------------------

    def propose_draft_tokens(
        self,
        num_drafts: int,
        committed_tokens,
        committed_positions,
        accepted_counts,
        hidden=None,
    ) -> DraftOutput:
        rows, _ = self._check_block(
            "propose", committed_tokens, committed_positions, 1 + num_drafts
        )
        self._check_accepted_counts(accepted_counts, rows, num_drafts)
        self.propose_calls.append(
            {
                "num_drafts": num_drafts,
                "rows": rows,
                "accepted_counts": accepted_counts.clone(),
                "hidden_was_none": hidden is None,
            }
        )
        # Deterministic and derived only from the last committed token per row,
        # so a test can predict both the drafts and the accepted count.
        last = self._last_committed(committed_tokens, accepted_counts)
        offsets = torch.arange(1, num_drafts + 1, dtype=torch.int32)
        drafts = (last.unsqueeze(1) + offsets.unsqueeze(0)) % self.vocab_size
        return DraftOutput(draft_token_ids=drafts.to(torch.int32))

    def decode_forward(
        self,
        tokens,
        positions,
        num_valid_drafts,
        accepted_counts,
        spec_mode: str,
        page_table=None,
        slot_mapping=None,
        sampling_params=None,
    ) -> VerifyOutput:
        """The verify primitive: one forward over the [B, 1+K] candidate block.

        ``spec_mode`` has no default, so a caller that forgets it fails rather
        than silently receiving greedy ids.
        """
        del page_table, slot_mapping, sampling_params
        if spec_mode not in self.accept_modes:
            raise ValueError(
                f"FakeSpecModel serves {list(self.accept_modes)}, "
                f"asked for {spec_mode!r}"
            )
        rows, block_width = self._check_block("verify", tokens, positions)
        num_drafts = block_width - 1
        self._check_accepted_counts(accepted_counts, rows, num_drafts)
        if num_valid_drafts.shape != (rows,):
            raise ValueError(
                f"verify num_valid_drafts must be [{rows}], got "
                f"{tuple(num_valid_drafts.shape)}"
            )
        out_of_range = num_valid_drafts[
            (num_valid_drafts < 0) | (num_valid_drafts > num_drafts)
        ]
        if out_of_range.numel():
            raise ValueError(
                f"verify num_valid_drafts entries must lie in [0, {num_drafts}], "
                f"got {out_of_range.tolist()}"
            )
        self.verify_calls.append(
            {
                "rows": rows,
                "block_width": block_width,
                "spec_mode": spec_mode,
                "num_valid_drafts": num_valid_drafts.clone(),
                "accepted_counts": accepted_counts.clone(),
            }
        )

        verified = self._verified_ids(tokens, num_valid_drafts)
        if spec_mode == ACCEPT_MODE_ARGMAX_IDS:
            return VerifyOutput(spec_mode=spec_mode, argmax_ids=verified, hidden=None)
        # The two modes must agree, or a test of one proves nothing about the
        # other: the logits argmax is the ids the other mode returns.
        logits = torch.zeros(rows, block_width, self.vocab_size)
        logits.scatter_(2, verified.to(torch.int64).unsqueeze(2), 1.0)
        return VerifyOutput(spec_mode=spec_mode, logits=logits, hidden=None)

    # ---- contract checks -------------------------------------------------

    def _check_block(
        self, call: str, tokens, positions, block_width: int | None = None
    ) -> tuple[int, int]:
        """Validate a candidate block and return its ``(rows, width)``.

        The width is derived here rather than unpacked by the caller, so a
        mis-ranked tensor produces the shape error naming the offender instead
        of a bare unpacking failure.
        """
        if tokens.dim() != 2 or positions.dim() != 2:
            raise ValueError(
                f"{call} tokens and positions must both be 2-D [B, 1+K], got "
                f"{tuple(tokens.shape)} and {tuple(positions.shape)}"
            )
        if tokens.shape != positions.shape:
            raise ValueError(
                f"{call} tokens {tuple(tokens.shape)} and positions "
                f"{tuple(positions.shape)} must have the same shape"
            )
        rows, width = int(tokens.shape[0]), int(tokens.shape[1])
        if rows < 1:
            raise ValueError(
                f"{call} needs at least one row: the runner does not issue a "
                f"step with no requests, got {tuple(tokens.shape)}"
            )
        if block_width is not None and width != block_width:
            raise ValueError(
                f"{call} expects the uniform width 1+K = {block_width}, got {width}"
            )
        return rows, width

    def _check_accepted_counts(self, accepted_counts, rows: int, num_drafts: int):
        if accepted_counts is None:
            raise ValueError(
                "accepted_counts may be None only after a fused_sample step, "
                "which FakeSpecModel does not serve"
            )
        if accepted_counts.shape != (rows,):
            raise ValueError(
                f"accepted_counts must be [{rows}], got {tuple(accepted_counts.shape)}"
            )
        if accepted_counts.dtype != torch.int32:
            raise ValueError(
                f"accepted_counts must be int32, got {accepted_counts.dtype}"
            )
        low, high = 1, 1 + num_drafts
        out_of_range = accepted_counts[
            (accepted_counts < low) | (accepted_counts > high)
        ]
        if out_of_range.numel():
            raise ValueError(
                f"accepted_counts entries must lie in [{low}, {high}]; a count "
                f"of 0 is never valid, got {out_of_range.tolist()}"
            )

    # ---- deterministic arithmetic ---------------------------------------

    @staticmethod
    def _last_committed(committed_tokens, accepted_counts):
        index = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        return committed_tokens.to(torch.int64).gather(1, index).squeeze(1)

    def _verified_ids(self, tokens, num_valid_drafts):
        """Ids the verify claims, per row, agreeing up to that row's cap.

        Column 0 always repeats the input token, which is already committed.
        Row ``i`` agrees with its drafted column ``1+j`` while ``j`` is below
        ``min(accept_depth, num_valid_drafts[i])``, and otherwise returns an id
        that differs from the draft at that column, so the accept walk for that
        row stops there.

        The cap is per row and never reduced across the batch. A batch-wide cap
        would let one grammar-truncated request destroy every other request's
        speculation, which the contract forbids, and no runner driven by this
        stand-in would ever exercise the mixed case.
        """
        num_drafts = int(tokens.shape[1]) - 1
        depth = num_drafts if self.accept_depth is None else self.accept_depth
        cap = torch.clamp(num_valid_drafts.to(torch.int64), max=depth)
        drafted = tokens[:, 1:].to(torch.int64)
        columns = torch.arange(num_drafts, dtype=torch.int64)
        diverge = columns.unsqueeze(0) >= cap.unsqueeze(1)
        verified = tokens.clone().to(torch.int64)
        verified[:, 1:] = torch.where(diverge, (drafted + 1) % self.vocab_size, drafted)
        return verified.to(torch.int32)


def make_fake_spec_model(**knobs) -> type[FakeSpecModel]:
    """Return a fresh :class:`FakeSpecModel` subclass with ``knobs`` applied.

    A subclass rather than a mutated class, because ``spec_plan`` is a
    classmethod and a mutated attribute would leak into every later test.
    """
    settable = _settable_knobs()
    unknown = sorted(set(knobs) - set(settable))
    if unknown:
        raise ValueError(
            f"make_fake_spec_model got unknown knobs {unknown}; "
            f"settable knobs are {settable}"
        )
    return type("ConfiguredFakeSpecModel", (FakeSpecModel,), dict(knobs))


# A classmethod object is not callable on every supported interpreter, so the
# descriptor types are named rather than filtered with callable().
_NON_KNOB_TYPES = (types.FunctionType, classmethod, staticmethod, property)


def _settable_knobs() -> list[str]:
    """Class attributes a caller may override, excluding the contract methods.

    Overriding a method name would replace a primitive with a plain value and
    fail at the call site instead of here.
    """
    return sorted(
        name
        for name, value in vars(FakeSpecModel).items()
        if not name.startswith("_") and not isinstance(value, _NON_KNOB_TYPES)
    )
