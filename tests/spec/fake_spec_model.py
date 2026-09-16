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
    ACCEPT_MODE_FUSED_SAMPLE,
    ACCEPT_MODE_LOGITS,
    DRAFTER_STATE_INTERNAL,
    HIDDEN_HANDOFF_ON_DEVICE,
    SPEC_REQUIREMENT_DEVICE_PROPOSE,
    SPEC_REQUIREMENT_HIDDEN_FEED,
    DraftOutput,
    SpecPlan,
    SpecReject,
    VerifyOutput,
    check_spec_side_tensors,
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
        # The handle the last verify returned. A fresh object per verify, and
        # deliberately of no useful type: the contract is that the runner hands
        # it back without interpreting it, so a test can only check identity,
        # and anything the runner did to it would show up as a different
        # object rather than as a wrong value.
        self.verify_hidden: object | None = None

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
        check_spec_side_tensors(
            torch.zeros(rows, dtype=torch.int32),
            accepted_counts,
            rows,
            num_drafts,
            call="propose",
        )
        self.propose_calls.append(
            {
                "num_drafts": num_drafts,
                "rows": rows,
                "accepted_counts": accepted_counts.clone(),
                "hidden_was_none": hidden is None,
                "hidden": hidden,
                "committed_tokens": committed_tokens.clone(),
                "committed_positions": committed_positions.clone(),
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
        start_pos,
        num_valid_drafts,
        accepted_counts,
        spec_mode: str,
        **kwargs,
    ) -> VerifyOutput:
        """The verify primitive: one forward over the [B, 1+K] candidate block.

        The signature is the plugin's ordinary decode call, ``tokens`` and
        ``start_pos``, plus the three speculative arguments. That is what makes
        the contract an extension of the existing call rather than a second one
        every model would have to grow: a speculative block travels in the same
        two tensors, only wider.

        ``spec_mode`` has no default, so a caller that forgets it fails rather
        than silently receiving greedy ids. Everything else the runner passes a
        decode, the page tables, the kv cache and the reload commands, this
        stand-in has no use for.
        """
        del kwargs
        positions = start_pos
        if spec_mode not in self.accept_modes:
            raise ValueError(
                f"FakeSpecModel serves {list(self.accept_modes)}, "
                f"asked for {spec_mode!r}"
            )
        if spec_mode == ACCEPT_MODE_FUSED_SAMPLE:
            # Reachable only through a configured variant that declares the
            # mode. Refused by name rather than falling through to the logits
            # return, which VerifyOutput would reject for this mode.
            raise NotImplementedError(
                "FakeSpecModel does not implement on-device rejection "
                "sampling; no model does"
            )
        rows, block_width = self._check_block("verify", tokens, positions)
        num_drafts = block_width - 1
        check_spec_side_tensors(num_valid_drafts, accepted_counts, rows, num_drafts)
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
        self.verify_hidden = object()
        if spec_mode == ACCEPT_MODE_ARGMAX_IDS:
            return VerifyOutput(
                spec_mode=spec_mode, argmax_ids=verified, hidden=self.verify_hidden
            )
        # The two modes must agree, or a test of one proves nothing about the
        # other: the logits argmax is the ids the other mode returns.
        logits = torch.zeros(rows, block_width, self.vocab_size)
        logits.scatter_(2, verified.to(torch.int64).unsqueeze(2), 1.0)
        return VerifyOutput(
            spec_mode=spec_mode, logits=logits, hidden=self.verify_hidden
        )

    # ---- contract checks -------------------------------------------------

    def _check_block(
        self, call: str, tokens, positions, block_width: int | None = None
    ) -> tuple[int, int]:
        """Validate a candidate block and return its ``(rows, width)``.

        The width is derived here rather than unpacked by the caller, so a
        mis-ranked tensor produces the shape error naming the offender instead
        of a bare unpacking failure.

        A narrow verify is the exception the contract carves out: a model
        declaring ``supports_narrow_decode`` receives the plain decode's own
        shapes on a step where no row carries a draft, which are ``[B, 1]``
        tokens and a 1-D ``[B]`` positions. Requiring both to be 2-D here would
        refuse the very call that declaration asks for.
        """
        narrow = tokens.dim() == 2 and tokens.shape[1] == 1 and positions.dim() == 1
        if narrow:
            if tokens.shape[0] != positions.shape[0]:
                raise ValueError(
                    f"{call} narrow tokens {tuple(tokens.shape)} and positions "
                    f"{tuple(positions.shape)} must agree on the row count"
                )
        elif tokens.dim() != 2 or positions.dim() != 2:
            raise ValueError(
                f"{call} tokens and positions must both be 2-D [B, 1+K], or "
                f"[B, 1] against a 1-D [B] for a narrow step, got "
                f"{tuple(tokens.shape)} and {tuple(positions.shape)}"
            )
        elif tokens.shape != positions.shape:
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

    # ---- deterministic arithmetic ---------------------------------------

    @staticmethod
    def _last_committed(committed_tokens, accepted_counts):
        index = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        return committed_tokens.to(torch.int64).gather(1, index).squeeze(1)

    def _verified_ids(self, tokens, num_valid_drafts):
        """Ids the verify claims, per row, agreeing up to that row's cap.

        Column ``j`` is what this model would choose at candidate position
        ``j``, which is the token draft ``j`` has to match, and the last column
        is the bonus that follows a fully accepted row. That is upstream's
        layout: its greedy kernel compares ``target_argmax[pos]`` against
        ``draft_token_ids[pos]`` and writes the committed token at ``pos``, so
        no column of the block is spent echoing an input the runner already
        holds.

        Row ``i`` agrees with its draft ``j`` while ``j`` is below
        ``min(accept_depth, num_valid_drafts[i])``, and otherwise returns an id
        that differs from that draft, so the accept walk for that row stops
        there.

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
        verified = torch.empty_like(tokens, dtype=torch.int64)
        verified[:, :num_drafts] = torch.where(
            diverge, (drafted + 1) % self.vocab_size, drafted
        )
        # The bonus sits at each row's own valid draft count, which is where the
        # contract says a row finds it, and not at a column fixed for the batch:
        # a row carrying no drafts has its bonus at column 0.
        #
        # Its value continues this stand-in's own drafting arithmetic one step
        # past that row's last draft. Its drafter proposes ``last + 1 + j``, so
        # accepting n of them leaves ``last + n + 1`` next, which makes a
        # speculated run and an unspeculated one walk the same token sequence.
        valid = num_valid_drafts.to(torch.int64)
        bonus = (tokens[:, 0].to(torch.int64) + valid + 1) % self.vocab_size
        verified.scatter_(1, valid.unsqueeze(1), bonus.unsqueeze(1))
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
