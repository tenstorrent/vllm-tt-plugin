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

import threading
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
    FusedReadout,
    SpecPlan,
    SpecReject,
    VerifyOutput,
    accept_greedy_drafts,
    check_spec_side_tensors,
)

# Small enough to build dense logits for in a test, wide enough that a drafted
# id and a verified id do not collide by accident.
FAKE_VOCAB_SIZE = 512


class FakeFusedHandle:
    """The opaque handle a deferred fused verify returns (chain mode).

    Holds the step's result the way a device holds it: nothing about it is
    readable from the host until it is ``wait``-ed on, which is what the reader
    thread does. ``wait`` blocks until the test releases the step (a held read,
    ``hold_reads``) and records which thread waited. ``finished`` is set by the
    lazy engine-thread bookkeeping (``propose_draft_tokens``), which must run
    after the wait and on the thread that submitted the step.
    """

    def __init__(
        self, tip_id, readout, nv_next, *, ready: bool = True, engine_thread=None
    ):
        self.tip_id = tip_id
        self._engine_thread = engine_thread
        self._readout = readout
        self.nv_next = nv_next
        self._ready = threading.Event()
        if ready:
            self._ready.set()
        self.waits = 0
        self.wait_thread = None
        self.finished = False

    def release(self) -> None:
        self._ready.set()

    @property
    def waited(self) -> bool:
        return self.waits > 0

    def wait(self, timeout: float = 30.0):
        if threading.get_ident() == self._engine_thread:
            # The engine thread waits only to drain: the device finishes on its
            # own, so a held read is a reader-thread concept.
            self._ready.set()
        assert self._ready.wait(timeout), "a held fused read was never released"
        self.waits += 1
        self.wait_thread = threading.get_ident()
        return self._readout


class ChainReadMixin:
    """``read_decode_output`` for a chain-capable fake (see ``make_fake_spec_model``).

    A separate class because the plugin's submit path reads
    ``hasattr(model, "read_decode_output")`` to decide whether a step reads
    back through events, so only a chain-capable model may have it."""

    def read_decode_output(self, tt_out, async_read: bool = False, **kwargs):
        del kwargs
        if isinstance(tt_out, FakeFusedHandle):
            assert not async_read, (
                "a fused handle is read with read_decode_output(handle)"
            )
            return tt_out.wait()
        # Anything else is a verify that already carries host tensors (a launch
        # with chaining off): nothing to wait for.
        assert async_read, "only a fused handle is read synchronously"
        return tt_out, []


class ChainFusedMixin:
    """The chain-mode contract on a deterministic target (stage S4, host fake).

    ``decode_forward(..., defer_read=True)`` on a fused verify enqueues and
    returns ``VerifyOutput(handle=...)`` with no host tensors. The "device"
    keeps, per state slot, the row's last committed token and the position its
    next column 0 will have; a chained step (``chain_rows``) builds its inputs
    from that, ignoring the host's placeholders, and clamps its draft count by
    the host's hint. It raises on every input the contract forbids: a tip that
    is not the last enqueued fused step, a chain row with real tokens or a
    non-placeholder count, a provisional position below the real one, a row with
    no state.

    A subclass supplies ``_chain_drafts``, ``_chain_accept`` and sets
    ``vocab_size``.
    """

    max_model_len: int | None = None

    def _chain_init(self) -> None:
        self._tip_id: int | None = 0
        # slot -> (last committed token, next column-0 position)
        self._dev_state: dict[int, tuple[int, int]] = {}
        self.engine_thread: int | None = None
        self.hold_reads = False
        self.handles: list[FakeFusedHandle] = []
        self.chain_log: list[dict] = []
        # Slots the runner sent as finishing, per call; freed like a finish.
        self.finishing_log: list[list[int]] = []

    def _chain_room(self, num_drafts: int, next_p0: int) -> int:
        if self.max_model_len is None:
            return num_drafts
        return max(0, min(num_drafts, self.max_model_len - (next_p0 + 1)))

    def _chain_drafts(self, last: int, position: int) -> list[int]:
        raise NotImplementedError

    def _chain_accept(self, tokens, p0, num_valid, spec_sampling, kwargs):
        raise NotImplementedError

    def _chain_invalidate(self) -> None:
        """A non-fused step ran: nothing may chain onto the state or the tip."""
        self._dev_state = {}
        self._tip_id = None

    def _chain_forward(
        self,
        tokens,
        start_pos,
        num_valid_drafts,
        accepted_counts,
        spec_sampling,
        kwargs,
    ) -> VerifyOutput:
        rows, width = int(tokens.shape[0]), int(tokens.shape[1])
        num_drafts = width - 1
        self._num_drafts = num_drafts
        thread = threading.get_ident()
        if self.engine_thread is None:
            self.engine_thread = thread
        elif self.engine_thread != thread:
            raise AssertionError(
                "decode_forward must be called on one (the engine) thread"
            )
        chain_rows = kwargs.get("chain_rows")
        chain_tip = kwargs.get("chain_tip")
        remap = kwargs.get("slot_remap")
        finishing = kwargs.get("finishing_slots")
        if finishing:
            if not self.model_capabilities.get("supports_finishing_slots", False):
                raise AssertionError("finishing_slots sent without the capability")
            self.finishing_log.append(list(finishing))
            for slot in finishing:
                self._dev_state.pop(int(slot), None)
        else:
            self.finishing_log.append([])
        if chain_rows is None:
            if chain_tip is not None:
                raise ValueError("chain_tip without chain_rows")
            chain_rows = torch.zeros(rows, dtype=torch.bool)
        else:
            if chain_rows.dtype != torch.bool or tuple(chain_rows.shape) != (rows,):
                raise ValueError(f"chain_rows must be bool [{rows}], got {chain_rows}")
            if chain_tip is None or chain_tip != self._tip_id:
                raise ValueError(
                    f"chain_tip {chain_tip} is not the last enqueued fused step "
                    f"{self._tip_id}"
                )
        eff_tokens = tokens.clone()
        eff_nv = num_valid_drafts.clone()
        p0 = start_pos[:, 0].clone().to(torch.int32)
        count_check = accepted_counts.clone()
        for row in range(rows):
            if not bool(chain_rows[row]):
                continue
            slot = int(remap[row]) if remap is not None else row
            state = self._dev_state.get(slot)
            if state is None:
                raise ValueError(
                    f"chain row {row} reads slot {slot}, which has no state"
                )
            if not bool((tokens[row] == -1).all()):
                raise ValueError(f"chain row {row} must carry placeholder tokens")
            if int(accepted_counts[row]) != -1:
                raise ValueError(f"chain row {row} must carry accepted_counts -1")
            last, next_p0 = state
            if int(start_pos[row, 0]) < next_p0:
                raise ValueError(
                    f"chain row {row}: provisional position {int(start_pos[row, 0])} "
                    f"is below the real one {next_p0}"
                )
            drafts = self._chain_drafts(last, next_p0)
            nv = min(int(num_valid_drafts[row]), self._chain_room(num_drafts, next_p0))
            eff_tokens[row, 0] = last
            eff_tokens[row, 1:] = torch.tensor(drafts, dtype=torch.int32)
            eff_tokens[row, 1 + nv :] = -1
            eff_nv[row] = nv
            p0[row] = next_p0
            count_check[row] = 1
        check_spec_side_tensors(eff_nv, count_check, rows, num_drafts)
        committed, counts = self._chain_accept(
            eff_tokens, p0, eff_nv, spec_sampling, kwargs
        )
        new_state: dict[int, tuple[int, int]] = {}
        nv_next = torch.zeros(rows, dtype=torch.int32)
        for row in range(rows):
            if int(p0[row]) < 0:
                continue
            count = int(counts[row])
            last = int(committed[row, count - 1])
            next_p0 = int(p0[row]) + count
            new_state[row] = (last, next_p0)
            nv_next[row] = self._chain_room(num_drafts, next_p0)
        self._dev_state = new_state
        self._tip_id = (self._tip_id or 0) + 1
        handle = FakeFusedHandle(
            self._tip_id,
            FusedReadout(committed.to(torch.int32), counts.to(torch.int32), p0),
            nv_next,
            ready=not self.hold_reads,
            engine_thread=self.engine_thread,
        )
        self.handles.append(handle)
        self.verify_hidden = handle
        self.chain_log.append(
            {
                "tip": handle.tip_id,
                "chain_rows": chain_rows.clone(),
                "tokens": tokens.clone(),
                "positions": start_pos.clone(),
                "accepted_counts": accepted_counts.clone(),
                "num_valid_drafts": num_valid_drafts.clone(),
                "eff_nv": eff_nv.clone(),
                "p0": p0.clone(),
            }
        )
        return VerifyOutput(spec_mode=ACCEPT_MODE_FUSED_SAMPLE, handle=handle)

    def _chain_propose(
        self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden
    ) -> DraftOutput:
        """Propose from a deferred step's handle, on the engine thread.

        Runs after the handle was read."""
        if threading.get_ident() != self.engine_thread:
            raise AssertionError(
                "the lazy finish of a fused step runs on the engine thread"
            )
        if not hidden.waited:
            raise AssertionError("propose_draft_tokens ran before the step was read")
        if hidden.finished:
            raise AssertionError("a fused step is finished once")
        hidden.finished = True
        self._num_drafts = num_drafts
        p0 = hidden._readout.p0
        expected = torch.where(p0 >= 0, p0 + 1, torch.full_like(p0, -1))
        if not torch.equal(committed_positions[:, 0].to(torch.int32), expected):
            raise AssertionError(
                "propose was given positions that are not the device's p0 + 1: "
                f"{committed_positions[:, 0].tolist()} vs {expected.tolist()}"
            )
        rows = int(committed_tokens.shape[0])
        index = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        last = committed_tokens.to(torch.int64).gather(1, index).squeeze(1)
        position = committed_positions.to(torch.int64).gather(1, index).squeeze(1)
        drafts = torch.zeros((rows, num_drafts), dtype=torch.int32)
        for row in range(rows):
            if int(p0[row]) >= 0:
                drafts[row] = torch.tensor(
                    self._chain_drafts(int(last[row]), int(position[row])),
                    dtype=torch.int32,
                )
        return DraftOutput(draft_token_ids=drafts, num_valid=hidden.nv_next.clone())


class FakeSpecModel(ChainFusedMixin):
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
    # Published in the plan, which is what per-request admission reads: a
    # launch without it verifies every step and refuses a sampled request.
    supports_narrow_decode: bool = False
    # Opt in to serving ``fused_sample`` (stage S1, greedy rows): the verify
    # returns accepted ids and counts computed by the greedy accept rule from
    # the same deterministic target the other modes use. Off by default, where
    # the mode is refused as before.
    serve_fused: bool = False
    # Opt in to the chain-mode contract (stage S4): implies ``serve_fused``,
    # declares ``supports_async_spec_decode``, and serves ``defer_read`` /
    # ``chain_rows`` / ``chain_tip`` with a ``FakeFusedHandle``. Build it with
    # ``make_fake_spec_model(serve_chain=True)``.
    serve_chain: bool = False
    # Per-position acceptance, for a stream whose acceptance varies: the depth
    # at a row whose column 0 sits at position p is ``accept_cycle[p % len]``.
    accept_cycle: tuple[int, ...] | None = None
    # The device's own cap on a row's draft count (the room under it), the
    # chain twin of the runner's ``max_model_len`` trim. None: no cap.
    max_model_len: int | None = None

    def __init__(self) -> None:
        self._chain_init()
        self.propose_calls: list[dict] = []
        self.verify_calls: list[dict] = []
        # Recorded apart from the verifies, because "this step ran as an
        # ordinary decode" is the property a speculating launch's overlap rests
        # on, and a test that counted only verifies could not see it.
        self.plain_calls: list[dict] = []
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
            supports_narrow_decode=cls.supports_narrow_decode,
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
        if isinstance(hidden, FakeFusedHandle):
            self.propose_calls.append(
                {
                    "num_drafts": num_drafts,
                    "rows": rows,
                    "accepted_counts": accepted_counts.clone(),
                    "hidden_was_none": False,
                    "hidden": hidden,
                    "committed_tokens": committed_tokens.clone(),
                    "committed_positions": committed_positions.clone(),
                }
            )
            return self._chain_propose(
                num_drafts,
                committed_tokens,
                committed_positions,
                accepted_counts,
                hidden,
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
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode: str | None = None,
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
        spec_sampling = kwargs.get("spec_sampling")
        chain_kwargs = kwargs
        del kwargs
        positions = start_pos
        if spec_mode is None:
            # The ordinary decode call, which a model declaring
            # ``supports_narrow_decode`` also serves: a speculating launch
            # sends it on a step that verifies nothing, and that step is what
            # can overlap. ``num_valid_drafts`` belongs to a verify, so its
            # presence without ``spec_mode`` is the plain call made wrong.
            if num_valid_drafts is not None:
                # A ``TypeError``, which is what the signature raised when
                # ``spec_mode`` had no default: a caller that sends the side
                # tensors and forgets the mode called this wrong, and must not
                # receive greedy ids for it.
                raise TypeError(
                    "FakeSpecModel was sent num_valid_drafts with no "
                    "spec_mode; the runner sends it only with a verify"
                )
            if accepted_counts is not None:
                # An unresolved multi-token commit, carried into an ordinary
                # decode because a row that is not speculable shares the step.
                # The upper bound is the verify's, which this call cannot see.
                self._check_plain_accepted_counts(tokens, accepted_counts)
            if self.serve_chain:
                self._chain_invalidate()
            return self._plain_decode(tokens, positions, accepted_counts)
        if spec_mode not in self.accept_modes:
            raise ValueError(
                f"FakeSpecModel serves {list(self.accept_modes)}, "
                f"asked for {spec_mode!r}"
            )
        if spec_mode == ACCEPT_MODE_FUSED_SAMPLE and not self.serve_fused:
            # Reachable only through a configured variant that declares the
            # mode. Refused by name rather than falling through to the logits
            # return, which VerifyOutput would reject for this mode.
            raise NotImplementedError(
                "FakeSpecModel does not implement on-device rejection "
                "sampling; no model does"
            )
        rows, block_width = self._check_block("verify", tokens, positions)
        num_drafts = block_width - 1
        deferred = bool(chain_kwargs.get("defer_read"))
        if deferred and not (
            self.serve_chain and spec_mode == ACCEPT_MODE_FUSED_SAMPLE
        ):
            raise ValueError(
                "defer_read is a fused_sample verify on a chain-capable model"
            )
        if not deferred:
            check_spec_side_tensors(num_valid_drafts, accepted_counts, rows, num_drafts)
        self.verify_calls.append(
            {
                "rows": rows,
                "block_width": block_width,
                "spec_mode": spec_mode,
                "num_valid_drafts": num_valid_drafts.clone(),
                "accepted_counts": accepted_counts.clone(),
                # Stage S2: the sampled-row controls and uniforms, when sent.
                "spec_sampling": spec_sampling,
                "defer_read": deferred,
                "chain_rows": chain_kwargs.get("chain_rows"),
                "chain_tip": chain_kwargs.get("chain_tip"),
            }
        )
        if deferred:
            return self._chain_forward(
                tokens,
                start_pos,
                num_valid_drafts,
                accepted_counts,
                spec_sampling,
                chain_kwargs,
            )
        if self.serve_chain:
            self._chain_invalidate()

        verified = self._verified_ids(
            tokens, num_valid_drafts, self._column0(positions)
        )
        self.verify_hidden = object()
        if spec_mode == ACCEPT_MODE_ARGMAX_IDS:
            return VerifyOutput(
                spec_mode=spec_mode, argmax_ids=verified, hidden=self.verify_hidden
            )
        if spec_mode == ACCEPT_MODE_FUSED_SAMPLE:
            committed, counts = accept_greedy_drafts(
                verified, tokens[:, 1:], num_valid_drafts
            )
            return VerifyOutput(
                spec_mode=spec_mode,
                argmax_ids=verified,
                accepted_token_ids=committed,
                accepted_counts=counts,
                hidden=self.verify_hidden,
            )
        # The two modes must agree, or a test of one proves nothing about the
        # other: the logits argmax is the ids the other mode returns.
        logits = torch.zeros(rows, block_width, self.vocab_size)
        logits.scatter_(2, verified.to(torch.int64).unsqueeze(2), 1.0)
        return VerifyOutput(
            spec_mode=spec_mode, logits=logits, hidden=self.verify_hidden
        )

    # ---- contract checks -------------------------------------------------

    @staticmethod
    def _check_plain_accepted_counts(tokens, accepted_counts) -> None:
        rows = int(tokens.shape[0])
        if accepted_counts.shape != (rows,) or accepted_counts.dtype != torch.int32:
            raise ValueError(
                f"ordinary decode accepted_counts must be [{rows}] int32, got "
                f"{tuple(accepted_counts.shape)} {accepted_counts.dtype}"
            )
        if bool((accepted_counts < 1).any()):
            raise ValueError(
                "ordinary decode accepted_counts entries are counts and never "
                f"0, got {accepted_counts.tolist()}"
            )

    def _plain_decode(self, tokens, positions, accepted_counts=None):
        """``[B, 1, V]`` logits whose argmax is this row's next token.

        The same rule the verify uses for the token that follows a row's last
        committed one, so a plain step and a verify agree about what this model
        would choose: the host sampling tail then commits ``last + 1``.
        """
        rows = int(tokens.shape[0])
        self.plain_calls.append(
            {
                "rows": rows,
                "width": int(tokens.shape[1]),
                "accepted_counts": (
                    None if accepted_counts is None else accepted_counts.clone()
                ),
            }
        )
        last = tokens.reshape(rows, -1)[:, 0].to(torch.int64)
        choice = (last + 1) % self.vocab_size
        logits = torch.zeros(rows, 1, self.vocab_size)
        logits.scatter_(2, choice.unsqueeze(1).unsqueeze(2), 1.0)
        return logits

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

    @staticmethod
    def _column0(positions):
        return positions[:, 0] if positions.dim() == 2 else positions

    def _chain_drafts(self, last: int, position: int) -> list[int]:
        del position
        return [(last + 1 + j) % self.vocab_size for j in range(self._num_drafts)]

    def _chain_accept(self, tokens, p0, num_valid, spec_sampling, kwargs):
        del spec_sampling, kwargs
        self._num_drafts = int(tokens.shape[1]) - 1
        verified = self._verified_ids(tokens, num_valid, p0)
        return accept_greedy_drafts(verified, tokens[:, 1:], num_valid)

    def _verified_ids(self, tokens, num_valid_drafts, p0=None):
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
        if self.accept_cycle is not None and p0 is not None:
            cycle = self.accept_cycle
            depth = torch.tensor(
                [cycle[max(int(p), 0) % len(cycle)] for p in p0], dtype=torch.int64
            )
            cap = torch.minimum(num_valid_drafts.to(torch.int64), depth)
        else:
            cap = torch.clamp(num_valid_drafts.to(torch.int64), max=depth)
        drafted = tokens[:, 1:].to(torch.int64)
        columns = torch.arange(num_drafts, dtype=torch.int64)
        diverge = columns.unsqueeze(0) >= cap.unsqueeze(1)
        verified = torch.zeros_like(tokens, dtype=torch.int64)
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
    bases: tuple[type, ...] = (FakeSpecModel,)
    if knobs.get("serve_chain"):
        # Chain mode runs fused, and declares what the platform gate reads.
        knobs.setdefault("serve_fused", True)
        knobs.setdefault(
            "accept_modes",
            (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS, ACCEPT_MODE_FUSED_SAMPLE),
        )
        knobs.setdefault(
            "model_capabilities",
            {
                **FakeSpecModel.model_capabilities,
                "supports_async_decode": True,
                "supports_async_spec_decode": True,
            },
        )
        bases = (ChainReadMixin, FakeSpecModel)
    return type("ConfiguredFakeSpecModel", bases, dict(knobs))


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
