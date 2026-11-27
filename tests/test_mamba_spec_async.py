# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculative decoding over Qwen3.8 GDN state in vLLM ``MambaSpec`` blocks
under asynchronous scheduling.

Under asynchronous scheduling vLLM's engine schedules step n + 1 before it
reads step n's output (``EngineCore.step_with_batch_queue``), so
``AsyncScheduler`` counts the drafts of the verify in flight as accepted, and
``MambaManager`` allocates, moves and frees each request's state blocks by
that count. The runner drains every verify and commits it before it builds the
request's next step, so the start block it names follows the count the
verify really committed.

``_AsyncHarness`` drives a real ``TTScheduler`` with its ``KVCacheManager``
and a real ``TTModelRunner`` through their engine entry points
(``schedule``, ``execute_model``, ``sample_tokens``, ``update_from_output``)
in the engine's order, synchronous or with a batch queue of two steps. The
model is ``_LedgerDevice``: a deterministic target (the next token is a
function of the token prefix) with a drafter whose drafts agree with the
target for a prefix length that cycles with the position, so verifies commit
every count, also near block boundaries. The device keeps the state each
block holds (``StateLedger``) in the order steps are submitted, as a device
command queue runs them, and checks every step:

- every start state a row reads is the state of exactly the tokens before the
  row's first query token, whoever wrote or zeroed the block before;
- no two rows of a step write one block, and no row writes a block another
  row reads as its start state;
- in mode ``align`` every block vLLM keeps in its prefix cache holds the
  state of exactly the tokens its hash covers.

The pool is small, so finished requests' blocks serve later requests, and
greedy outputs must equal those of a synchronous run.

With ``sampled_ids`` the device also declares ``spec_ids_are_sampled``: on a
step the runner samples on the device, a sampled row's ids are draws
(``_draw``), whose seed a seeded row derives from the request seed and the
input position (``_position_seed``) and an unseeded row takes from fresh
entropy, so a seeded request's token at a position does not depend on how
steps group the positions.
"""

from __future__ import annotations

import dataclasses
import random
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from tests.test_mamba_kv_cache import _kv_cache_config, _vllm_config
from tests.test_mamba_state_lifecycle import (
    BLOCK_SIZE,
    EOS,
    GARBAGE,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    StateLedger,
    _prompt,
    _qwen38_specs,
    _scheduler,
)
from vllm_tt_plugin.config import store_tt_spec_plan
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.spec_admission import MODEL_OWNED_DRAFT_METHOD
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    DRAFTER_STATE_INTERNAL,
    MAX_SPEC_DEVICE_LOGPROBS,
    PLACEHOLDER_TOKEN_ID,
    DraftOutput,
    SpecLogprobs,
    SpecPlan,
    VerifyOutput,
)

K = 3
T = K + 1
# _vllm_config's vocabulary.
VOCAB = 32
FIRST_ID = 3
# The device sampler's top-k bound with ``sampled_ids``. Below the vocabulary,
# as the Qwen3.8 class's 32 is below its 248320: the input batch stores a
# top-k at or above the vocabulary as the vocabulary.
MAX_DEVICE_TOP_K = 8


def _target(prefix: list[int] | tuple[int, ...]) -> int:
    """The target model's greedy token after ``prefix``."""
    tail = tuple(prefix[-3:])
    return FIRST_ID + (
        sum((i + 1) * t for i, t in enumerate(tail)) + 7 * len(prefix)
    ) % (VOCAB - FIRST_ID)


def _agreeing_drafts(prefix: list[int], agree: int) -> list[int]:
    """K drafts after ``prefix``: the first ``agree`` are the target's own
    chain, the rest differ from it."""
    chain = list(prefix)
    drafts = []
    for k in range(K):
        token = _target(chain)
        if k >= agree:
            token = FIRST_ID + (token - FIRST_ID + 1) % (VOCAB - FIRST_ID)
        drafts.append(token)
        chain.append(token)
    return drafts


def _drafts(prefix: list[int]) -> list[int]:
    """K drafts after ``prefix`` whose agreeing prefix cycles with the
    position."""
    return _agreeing_drafts(prefix, len(prefix) % T)


def _boundary_drafts(prefix: list[int]) -> list[int]:
    """K drafts after ``prefix`` that all agree away from a block boundary, so
    a greedy request moves K + 1 positions per verify, and within 9 positions
    of one agree for 0 to K - 1 drafts, varying with the request and the
    boundary, so a greedy verify there commits every count 1 to K."""
    n = len(prefix)
    if n % BLOCK_SIZE < BLOCK_SIZE - 9:
        return _agreeing_drafts(prefix, K)
    return _agreeing_drafts(prefix, (prefix[0] + n // BLOCK_SIZE + n) % K)


def _position_seed(seed: int, position: int) -> int:
    """The device seed of a seeded row's draw at input ``position``."""
    return (seed * 1_000_003 + (position + 1) * 7919) % (1 << 20)


def _draw(prefix: list[int], device_seed: int) -> int:
    """The device sampler's draw after ``prefix``: the target's token for
    three seeds in four, so drafts are often accepted and every count occurs,
    another token otherwise."""
    token = _target(prefix)
    if device_seed % 4:
        return token
    return FIRST_ID + (token - FIRST_ID + 1 + device_seed % 7) % (VOCAB - FIRST_ID)


def _device_logprob(prefix: list[int], token: int) -> float:
    """The raw logprob the device reports for ``token`` after ``prefix``
    (``supports_spec_device_logprobs``): a function of both, so a published
    logprob is checkable against the token history alone. Never 0, which
    is what the host sampler reports on this harness's one-hot logits."""
    return -float(1 + (7 * sum(prefix[-4:]) + 3 * len(prefix) + token) % 61) / 8.0


def _device_rank(prefix: list[int], token: int) -> int:
    return 1 + (len(prefix) + token) % 5


def _device_top(prefix: list[int]) -> tuple[list[int], list[float]]:
    """The top ``MAX_SPEC_DEVICE_LOGPROBS`` ids and logprobs after ``prefix``."""
    first = (sum(prefix[-3:]) + len(prefix)) % VOCAB
    ids = [(first + i) % VOCAB for i in range(MAX_SPEC_DEVICE_LOGPROBS)]
    return ids, [-0.25 * i - (len(prefix) % 3) for i in range(MAX_SPEC_DEVICE_LOGPROBS)]


class _Answer:
    """A verify's deferred ``argmax_ids``: the ids, and how often the host read
    them; with ``payload`` the ``SpecLogprobs`` the read returns with them."""

    def __init__(
        self,
        ids: torch.Tensor,
        record: SimpleNamespace,
        payload: SpecLogprobs | None = None,
    ):
        self.ids = ids
        self.record = record
        self.payload = payload
        self.reads = 0


class _LedgerDevice:
    """A deterministic target and drafter that keep GDN state in ``StateLedger``.

    ``truth[req]`` is the request's token history as the device computed it.
    Each verify walks its own ids the way the runner does and keeps the
    drafts of the commit for the propose that follows (the drafter runs with
    the verify, as on a one-enqueue step), and each propose checks that the
    runner committed what the device accepted. A chained verify derives each
    row's positions, and so a seeded row's seeds, from that count.
    """

    decode_input_update_contract = 1

    def __init__(self, harness: _AsyncHarness, chained: bool):
        self.h = harness
        self.model_capabilities = {
            "supports_async_decode": True,
            "supports_async_spec_decode": True,
            "supports_chained_spec_verify": chained,
        }
        self.calls: list[tuple[str, dict]] = []
        self.truth: dict[str, list[int]] = {}
        self.next_drafts: dict[str, list[int]] = {}
        # Each submitted decode's row -> request map, popped by its propose.
        self.steps: deque[SimpleNamespace] = deque()
        self.verifies = 0
        self.chained = 0
        # Chained verifies whose ``slot_remap`` moves a live row.
        self.moved_rows = 0
        self.prefills = 0
        # Prefills submitted while a verify was unread, and the requests of
        # that verify whose rows' state slots they took.
        self.prefills_behind = 0
        self.slots_reused_behind: list[str] = []
        self.ordinary_decodes = 0
        # Ordinary decodes submitted right after another ordinary decode.
        self.plain_after_plain = 0
        self.last_kind = None
        # ``(verify rows so far, request, record) -> bool``: the first verify
        # row it selects gets a device count one draft longer than the walk of
        # its ids, for the guard tests.
        self.miscount = None
        self.miscounted_successors = 0
        # Every verify's record, and the fresh entropy of unseeded draws.
        self.records: list[SimpleNamespace] = []
        self.entropy = random.Random(0)
        # ``(request, token history) -> bool``: the drafter declines the row
        # (offers no drafts) after that history.
        self.decline = None
        self.accepted: dict[tuple[int, int], int] = {}
        self.boundary_counts: set[int] = set()
        # The drafter's rule: token history -> its K drafts.
        self.drafts = _drafts
        # Every row the device ran, in order: (row, start block per group,
        # window blocks per group).
        self.inputs: list[tuple] = []
        # The token mask of the verify being sampled (``supports_spec_device_
        # mask``), and the token-masked verifies sampled, chained or not.
        self.mask: torch.Tensor | None = None
        self.masked_verifies = 0
        self.chained_masked = 0

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return ["kv"]

    def zero_kv_cache_blocks(self, kv_cache=None, block_ids=()):
        self.h.ledger.zero(self.h.groups, list(block_ids))

    def _live(self) -> list[str]:
        batch = self.h.runner.input_batch
        return list(batch.req_ids[: batch.num_reqs])

    def _run_rows(self, mamba_state, rows) -> None:
        """Read every row's start state, then write its window, as one kernel
        launch does: ``rows`` is ``[(row, prefix, start_blocks, writes)]``,
        ``writes`` ``[(window column, state)]``, per group."""
        written: dict[tuple[int, int], int] = {}
        for row, _prefix, starts, _writes in rows:
            self.inputs.append(
                (
                    row,
                    tuple(starts),
                    tuple(tuple(g.windows[row].tolist()) for g in mamba_state.groups),
                )
            )
        for row, prefix, starts, writes in rows:
            for i, g in enumerate(self.h.groups):
                self.h.ledger.read(g, starts[i], prefix, len(prefix))
                window = mamba_state.groups[i].windows[row].tolist()
                for t, _state in writes:
                    block = window[t]
                    assert written.setdefault((g, block), row) == row, (
                        f"rows {written[(g, block)]} and {row} write block {block}"
                    )
        for row, _prefix, starts, _writes in rows:
            for i, g in enumerate(self.h.groups):
                window = mamba_state.groups[i].windows[row].tolist()
                if starts[i] and starts[i] not in window:
                    assert (g, starts[i]) not in written, (
                        f"block {starts[i]} is row {row}'s start state and row "
                        f"{written[(g, starts[i])]} writes it"
                    )
        for row, _prefix, _starts, writes in rows:
            for i, g in enumerate(self.h.groups):
                window = mamba_state.groups[i].windows[row].tolist()
                for t, block in enumerate(window):
                    state = dict(writes).get(t, GARBAGE)
                    self.h.ledger.states[(g, block)] = state
                    if state is not GARBAGE:
                        self.h.ledger.gdn_blocks.add(block)
        for g in self.h.groups:
            self.h.ledger.run_padding_row(g)

    def prefill_forward(self, tokens, prompt_lens, start_pos, mamba_state, **kwargs):
        self.last_kind = "prefill"
        self.prefills += 1
        runner = self.h.runner
        unread = [
            step
            for step in runner._pending_async_steps
            if isinstance(getattr(step, "verify_output", None), VerifyOutput)
            and not step.is_resolved()
        ]
        if unread:
            # The model's side of supports_prefill_over_pending_spec: one
            # unread verify, whose propose comes after this prefill.
            assert len(unread) == 1, unread
            assert self.model_capabilities.get("supports_prefill_over_pending_spec")
            self.prefills_behind += 1
            verify_rows = unread[0].model_input.row_req_ids
            for slot in kwargs.get("empty_slots") or ():
                if slot < len(verify_rows):
                    self.slots_reused_behind.append(verify_rows[slot])
        live = self._live()
        logits = torch.zeros((len(live), 1, VOCAB))
        rows = []
        for row, req_id in enumerate(live):
            start, end = int(start_pos[row]), int(prompt_lens[row])
            prefix = tokens[row, :end].tolist()
            writes = [(0, tuple(prefix))]
            position = (
                0
                if mamba_state.checkpoint_positions is None
                else int(mamba_state.checkpoint_positions[row, 0])
            )
            starts = [int(g.start_blocks[row]) for g in mamba_state.groups]
            rows.append((row, prefix[:start], starts, writes))
            if position:
                for i, g in enumerate(self.h.groups):
                    block = int(mamba_state.groups[i].checkpoint_blocks[row, 0])
                    self.h.ledger.write(g, block, prefix, position)
            if not bool(mamba_state.intermediate_prefill_mask[row]):
                token = _target(prefix)
                allowed = self.h.requests[req_id].sampling_params.allowed_token_ids
                if allowed:
                    # The host sampler applies the allowlist to these logits.
                    token = self._masked(token, allowed.__contains__)
                self.truth[req_id] = [*prefix, token]
                logits[row, 0, token] = 1e4
        self._run_rows(mamba_state, rows)
        return logits

    def decode_forward(self, *, tokens, start_pos, mamba_state, **kwargs):
        if kwargs.get("reload_inputs") is not True:
            raise AssertionError(
                f"reload_inputs={kwargs.get('reload_inputs')}: every decode "
                "of this model reloads, its ordinary decode being host-answered"
            )
        live = self._live()
        if kwargs.pop("defer_device_sampling", False):
            # A token-masked verify: it runs when sample_deferred_decode
            # brings its mask, over the rows it was submitted with.
            self.last_kind = "verify"
            return SimpleNamespace(
                live=live,
                tokens=tokens.clone(),
                start_pos=start_pos.clone(),
                mamba_state=mamba_state,
                kwargs=kwargs,
            )
        if kwargs.get("spec_mode") is not None:
            self.last_kind = "verify"
        if kwargs.get("spec_mode") is None:
            self.ordinary_decodes += 1
            self.plain_after_plain += self.last_kind == "plain"
            self.last_kind = "plain"
            logits = self._narrow(
                live, tokens, start_pos, mamba_state, kwargs.get("sampling_params")
            )
            if kwargs.get("sampling_params") is None:
                return logits
            # A device-sampled ordinary decode: host ids, read inside the call.
            return logits[:, 0].argmax(dim=-1).to(torch.int32).reshape(-1, 1)
        return self._verify(live, tokens, start_pos, mamba_state, kwargs)

    def sample_deferred_decode(
        self, payload, sampling_params=None, grammar_bitmask=None, num_valid_drafts=None
    ):
        kwargs = {
            **payload.kwargs,
            "sampling_params": sampling_params,
            "num_valid_drafts": num_valid_drafts,
        }
        self.mask = grammar_bitmask
        try:
            out = self._verify(
                payload.live,
                payload.tokens,
                payload.start_pos,
                payload.mamba_state,
                kwargs,
            )
        finally:
            self.mask = None
        self.masked_verifies += 1
        self.chained_masked += kwargs.get("spec_predecessor") is not None
        return out

    def _masked(self, token: int, allowed) -> int:
        """The first token from ``token`` upward (mod VOCAB) that ``allowed``
        holds: the masked choice of a row whose unmasked choice is ``token``."""
        for step in range(VOCAB):
            candidate = (token + step) % VOCAB
            if allowed(candidate):
                return candidate
        raise AssertionError("a token mask allowed nothing")

    def _draws(self, row: int, sampling) -> bool:
        """Whether ``row``'s ids are draws: a sampled row of a device-sampled
        step (``sampling``) on a device declaring ``spec_ids_are_sampled``."""
        return (
            sampling is not None
            and bool(self.model_capabilities.get("spec_ids_are_sampled"))
            and float(sampling.temperature[row]) > 0
            and int(sampling.top_k[row]) != 1
        )

    def _token(self, prefix, row: int, position: int, sampling, column=None) -> int:
        """The device's token after ``prefix`` at input ``position``: the
        target's, or a drawing row's draw, seeded by the request seed and the
        position or by fresh entropy; under a token mask, the first token from
        there upward that the mask of ``row`` and ``column`` allows."""
        if not self._draws(row, sampling):
            token = _target(prefix)
        elif sampling.seed[row] is None:
            token = _draw(prefix, self.entropy.randrange(1 << 20))
        else:
            token = _draw(prefix, _position_seed(sampling.seed[row], position))
        if self.mask is None or column is None:
            return token
        words = self.mask[row, column]
        return self._masked(
            token, lambda t: bool((int(words[t // 32]) >> (t % 32)) & 1)
        )

    def _offer(self, req_id: str) -> int:
        """The drafts the drafter offers after the request's history."""
        declined = self.decline is not None and self.decline(req_id, self.truth[req_id])
        return 0 if declined else K

    def _narrow(self, live, tokens, start_pos, mamba_state, sampling=None):
        logits = torch.zeros((int(tokens.shape[0]), 1, VOCAB))
        rows = []
        for row, req_id in enumerate(live):
            truth = self.truth[req_id]
            position = int(start_pos[row])
            assert position == len(truth) - 1 and int(tokens[row, 0]) == truth[-1], (
                req_id,
                position,
                len(truth),
            )
            starts = [int(g.start_blocks[row]) for g in mamba_state.groups]
            rows.append((row, truth[:position], starts, [(0, tuple(truth))]))
            token = self._token(truth, row, position, sampling)
            allowed = self.h.requests[req_id].sampling_params.allowed_token_ids
            if sampling is None and allowed:
                # The host sampler applies the allowlist to these logits.
                token = self._masked(token, allowed.__contains__)
            truth.append(token)
            logits[row, 0, truth[-1]] = 1e4
            self.next_drafts[req_id] = self.drafts(truth)
        self._run_rows(mamba_state, rows)
        self.steps.append(
            SimpleNamespace(
                rows=list(live),
                counts=[1] * len(live),
                next_drafts=[self.next_drafts[r] for r in live],
                offered=[self._offer(r) for r in live],
            )
        )
        return logits

    def _verify(self, live, tokens, start_pos, mamba_state, kwargs):
        num_valid = kwargs["num_valid_drafts"]
        rows_out = int(tokens.shape[0])
        ids = torch.full((rows_out, T), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        sampling = kwargs.get("sampling_params")
        predecessor = kwargs.get("spec_predecessor")
        record = SimpleNamespace(
            rows=list(live),
            counts=[],
            ids=ids,
            p0=[],
            next_drafts=[],
            offered=[],
            miscounted=False,
            sampled=[self._draws(row, sampling) for row in range(len(live))],
            chained=predecessor is not None,
            followed=False,
            # row -> the token history before each verified column's token.
            prefixes={},
        )
        self.records.append(record)
        for row, req_id in enumerate(live):
            params = self.h.requests[req_id].sampling_params
            if params.temperature > 0 and not record.sampled[row]:
                raise AssertionError(
                    f"{req_id} samples and its argmax_ids verify answers the argmax"
                )
        remap = kwargs.get("slot_remap")
        sources = list(range(len(live))) if remap is None else remap.tolist()
        if predecessor is not None:
            self.chained += 1
            self.moved_rows += sources[: len(live)] != list(range(len(live)))
            predecessor = predecessor.argmax_ids.record
            predecessor.followed = True
        rows = []
        for row, req_id in enumerate(live):
            truth = self.truth[req_id]
            if predecessor is None:
                p0 = int(start_pos[row, 0])
                nv = int(num_valid[row])
                block = tokens[row, : nv + 1].tolist()
                starts = [int(g.start_blocks[row]) for g in mamba_state.groups]
                drafts = self.next_drafts.get(req_id, [])
                assert block[1:] == drafts[:nv], (req_id, block)
            else:
                # The row continues the unread predecessor's row ``source``
                # (``slot_remap``) from the count the device accepted; the
                # runner sent placeholders.
                source = sources[row]
                assert predecessor.rows[source] == req_id, (
                    predecessor.rows,
                    live,
                    sources,
                )
                count = predecessor.counts[source]
                previous = predecessor.p0[source]
                assert tokens[row].tolist() == [PLACEHOLDER_TOKEN_ID] * T
                assert start_pos[row].tolist() == [previous + T + j for j in range(T)]
                assert int(kwargs["accepted_counts"][row]) == 1
                p0 = previous + count
                nv = min(int(num_valid[row]), predecessor.offered[source])
                drafts = predecessor.next_drafts[source]
                block = [int(predecessor.ids[source, count - 1]), *drafts[:nv]]
                starts = [
                    int(g.start_blocks_by_count[row, count - 1])
                    for g in mamba_state.groups
                ]
                if predecessor.miscounted:
                    # The device built the row from its wrong count; the
                    # predecessor's propose is what reports it.
                    record.miscounted = True
                    self.miscounted_successors += 1
                    continue
            assert p0 == len(truth) - 1 and block[0] == truth[-1], (req_id, p0)
            self._verify_row(
                row, req_id, p0, block, starts, ids, record, rows, sampling
            )
        self._run_rows(mamba_state, rows)
        self.steps.append(record)
        return VerifyOutput(
            spec_mode=ACCEPT_MODE_ARGMAX_IDS,
            argmax_ids=_Answer(ids, record, self._payload(live, record, kwargs)),
        )

    def _payload(self, live, record, kwargs) -> SpecLogprobs | None:
        """The ``SpecLogprobs`` of a verify sent ``spec_num_logprobs``, which
        the runner sends exactly when this device declares
        ``supports_spec_device_logprobs`` and a live row asks for logprobs."""
        asked = [self.h.requests[r].sampling_params.logprobs for r in live]
        sent = kwargs.get("spec_num_logprobs")
        declared = self.model_capabilities.get("supports_spec_device_logprobs")
        if not declared or all(n is None for n in asked):
            assert sent is None, sent
            return None
        rows = int(record.ids.shape[0])
        expected = [-2 if n is None else n for n in asked] + [-2] * (rows - len(live))
        assert sent.dtype == torch.int32 and sent.tolist() == expected, (
            sent,
            expected,
        )
        n = MAX_SPEC_DEVICE_LOGPROBS
        payload = SpecLogprobs(
            torch.zeros(rows, T),
            torch.zeros(rows, T, dtype=torch.int32),
            torch.zeros(rows, T, n),
            torch.zeros(rows, T, n, dtype=torch.int32),
        )
        for row, prefixes in record.prefixes.items():
            for t, prefix in enumerate(prefixes):
                token = int(record.ids[row, t])
                payload.sampled_logprobs[row, t] = _device_logprob(prefix, token)
                payload.selected_token_ranks[row, t] = _device_rank(prefix, token)
                top_ids, top_logprobs = _device_top(prefix)
                payload.top_token_ids[row, t] = torch.tensor(top_ids)
                payload.top_logprobs[row, t] = torch.tensor(top_logprobs)
        return payload

    def _verify_row(self, row, req_id, p0, block, starts, ids, record, rows, sampling):
        truth = self.truth[req_id]
        prefix = truth[:p0]
        nv = len(block) - 1
        for t in range(T):
            ids[row, t] = (
                self._token(prefix + block[: t + 1], row, p0 + t, sampling, t)
                if t <= nv
                else PLACEHOLDER_TOKEN_ID
            )
        record.prefixes[row] = [prefix + block[: t + 1] for t in range(nv + 1)]
        count = 1
        while count <= nv and block[count] == int(ids[row, count - 1]):
            count += 1
        writes = [(t, tuple(prefix + block[: t + 1])) for t in range(nv + 1)]
        rows.append((row, prefix, starts, writes))
        self.verifies += 1
        self.accepted[(nv, count)] = self.accepted.get((nv, count), 0) + 1
        if min(p0 % BLOCK_SIZE, BLOCK_SIZE - p0 % BLOCK_SIZE) <= 2 * T:
            self.boundary_counts.add(count)
        committed = [int(t) for t in ids[row, :count]]
        if (
            self.miscount is not None
            and count <= nv
            and self.miscount(self.verifies, req_id, record)
        ):
            # A device acceptance one draft longer than the walk of its ids.
            count += 1
            self.miscount = None
            record.miscounted = True
        record.counts.append(count)
        record.p0.append(p0)
        truth[p0 + 1 :] = committed
        self.next_drafts[req_id] = self.drafts(truth)
        record.next_drafts.append(self.next_drafts[req_id])
        record.offered.append(self._offer(req_id))

    def read_decode_output(self, tt_out, async_read: bool = False, **kwargs):
        return (tt_out, []) if async_read else tt_out

    def process_decode_output_host(self, tt_out, is_tokens=False, **kwargs):
        if not isinstance(tt_out, _Answer):
            raise AssertionError(f"nothing deferred to read: {type(tt_out).__name__}")
        tt_out.reads += 1
        if tt_out.payload is not None:
            return tt_out.ids, tt_out.payload
        return tt_out.ids

    def propose_draft_tokens(
        self, num_drafts, committed, positions, counts, hidden=None
    ):
        step = self.steps.popleft()
        drafts = torch.full((int(committed.shape[0]), num_drafts), 0, dtype=torch.int32)
        num_valid = torch.zeros(int(committed.shape[0]), dtype=torch.int32)
        for row, req_id in enumerate(step.rows):
            if int(positions[row, 0]) < 0:
                continue
            count = int(counts[row])
            if count != step.counts[row]:
                # The model's guard: a chained step continued the device's count.
                raise ValueError(
                    f"request {req_id}: the runner committed {count} tokens, "
                    f"the device accepted {step.counts[row]}"
                )
            truth = self.truth[req_id]
            first = int(positions[row, 0])
            assert committed[row, :count].tolist() == truth[first : first + count], (
                req_id
            )
            drafts[row] = torch.tensor(step.next_drafts[row], dtype=torch.int32)
            num_valid[row] = step.offered[row]
        return DraftOutput(draft_token_ids=drafts, num_valid=num_valid)


class _AsyncHarness:
    """Runs requests through the engine's step order, synchronous or with a
    batch queue of two steps (``asynchronous``)."""

    def __init__(
        self,
        num_blocks: int,
        *,
        mode: str = "align",
        asynchronous: bool = True,
        chained: bool = False,
        max_num_batched_tokens: int = 4096,
        trace_mode: str = "none",
        device_sampling: bool = False,
        sampled_ids: bool = False,
        prefill_over_pending: bool = False,
        device_logprobs: bool = False,
        token_masks: bool = False,
        device_top_p: bool = False,
    ):
        cfg = _vllm_config(mode)
        cfg.model_config.max_model_len = MAX_MODEL_LEN
        cfg.model_config.original_max_model_len = MAX_MODEL_LEN
        cfg.scheduler_config.max_num_seqs = MAX_NUM_SEQS
        cfg.scheduler_config.async_scheduling = asynchronous
        cfg.cache_config.prefix_match_unit = None
        self.mode = mode
        self.asynchronous = asynchronous
        spec_plan = SpecPlan(
            effective_k=K,
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=True,
        )
        cfg.speculative_config = SimpleNamespace(
            method=MODEL_OWNED_DRAFT_METHOD,
            num_speculative_tokens=K,
            use_eagle=lambda: False,
            use_eagle_block_drop=lambda: False,
        )
        store_tt_spec_plan(cfg, spec_plan)
        self.kv_cache_config = _kv_cache_config(
            cfg, _qwen38_specs(mode, num_speculative_blocks=K), num_blocks
        )
        self.scheduler = _scheduler(
            self.kv_cache_config,
            max_num_batched_tokens,
            mode,
            spec_plan=spec_plan,
        )
        self.scheduler.scheduler_config.async_scheduling = asynchronous
        self.hash_block_size = self.scheduler.hash_block_size
        self.device = _LedgerDevice(self, chained)
        self.runner = TTModelRunner(
            vllm_config=cfg,
            mesh_device=None,
            trace_mode=trace_mode,
            enable_model_warmup=False,
            num_devices=1,
        )
        self.runner.model = self.device
        if device_sampling or sampled_ids:
            self.device.model_capabilities["supports_sample_on_device"] = True
            self.runner.sample_on_device_mode = "decode_only"
        if sampled_ids:
            # The Qwen3.8 MTP class's sampling declarations.
            self.device.model_capabilities.update(
                spec_ids_are_sampled=True,
                max_device_top_k=MAX_DEVICE_TOP_K,
                supports_device_penalties=False,
            )
        if device_logprobs:
            self.device.model_capabilities["supports_spec_device_logprobs"] = True
        if token_masks:
            # A model drawing its verifies under the plugin's token mask.
            self.device.model_capabilities["supports_spec_device_mask"] = True
        if device_top_p:
            self.device.model_capabilities["supports_spec_device_top_p"] = True
        if prefill_over_pending:
            # As TTPlatform stores the validated declaration for the scheduler.
            self.device.model_capabilities["supports_prefill_over_pending_spec"] = True
            self.scheduler._prefill_over_pending_spec = True
        self.runner.initialize_kv_cache(self.kv_cache_config)
        self.groups = list(self.runner._mamba_groups.group_indices)
        self.ledger = StateLedger()
        self.requests: dict[str, Request] = {}
        self.hash_tokens: dict[bytes, tuple[int, ...]] = {}
        self.outputs: dict[str, list[int]] = {}
        self.cached_states_checked = 0
        self.num_added = 0
        # Steps scheduled while the previous one's output was unread.
        self.scheduled_ahead = 0
        self.preempted: set[str] = set()

    def add(
        self,
        req_id: str,
        prompt_len: int,
        max_tokens: int,
        temperature: float = 0.0,
        **sampling,
    ) -> None:
        init_none_hash(sha256)
        params = SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=True,
            temperature=temperature,
            **sampling,
        )
        params.update_from_generation_config({}, eos_token_id=EOS)
        self.num_added += 1
        request = Request(
            request_id=req_id,
            prompt_token_ids=_prompt(self.num_added, prompt_len),
            sampling_params=params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(self.hash_block_size, sha256),
        )
        self.requests[req_id] = request
        self.scheduler.add_request(request)

    def check_cached_states(self) -> None:
        for request in self.requests.values():
            tokens = request.all_token_ids
            for j, block_hash in enumerate(request.block_hashes):
                self.hash_tokens.setdefault(
                    block_hash, tuple(tokens[: (j + 1) * self.hash_block_size])
                )
        # vLLM caches a prefill's blocks when it schedules the prefill. A
        # prefill TTScheduler keeps for its next schedule() call runs after
        # this commit, and no step reads its blocks before it runs.
        manager = self.scheduler.kv_cache_manager
        deferred = self.scheduler._deferred_prefill
        unwritten = {
            block_id
            for req_id in (deferred.num_scheduled_tokens if deferred else ())
            for ids in manager.get_block_ids(req_id)
            for block_id in ids
        }
        for block in manager.block_pool.blocks:
            key = block.block_hash
            if key is None or get_group_id(key) not in self.groups:
                continue
            if block.block_id in unwritten:
                continue
            tokens = self.hash_tokens[get_block_hash(key)]
            held = self.ledger.states.get((get_group_id(key), block.block_id))
            assert held == tokens, (
                f"cached GDN block {block.block_id} of group {get_group_id(key)} "
                f"is keyed on {len(tokens)} tokens"
            )
            self.cached_states_checked += 1

    def _update(self, scheduler_output, output) -> None:
        if hasattr(output, "get_output"):
            output = output.get_output()
        assert isinstance(output, ModelRunnerOutput), type(output)
        self.scheduler.update_from_output(scheduler_output, output)
        if self.mode == "align":
            self.check_cached_states()

    def run(self, max_steps: int = 2000) -> None:
        """``EngineCore.step`` (synchronous) or ``step_with_batch_queue``."""
        queue: deque = deque()
        depth = 2 if self.asynchronous else 1
        for _ in range(max_steps):
            if not self.scheduler.has_requests() and not queue:
                break
            scheduled = False
            if self.scheduler.has_requests() and len(queue) < depth:
                self.scheduled_ahead += bool(queue)
                out = self.scheduler.schedule()
                result = self.runner.execute_model(out)
                if result is None:
                    result = self.runner.sample_tokens(None)
                queue.appendleft((out, result))
                scheduled = out.total_num_scheduled_tokens > 0
                if len(queue) < depth and (scheduled or self.scheduler.has_requests()):
                    continue
            out, result = queue.pop()
            self.preempted |= set(out.preempted_req_ids or ())
            self._update(out, result)
            if not self.asynchronous:
                drafts = self.runner.take_draft_token_ids()
                if drafts is not None:
                    self.scheduler.update_draft_token_ids(drafts)
        else:
            raise AssertionError("scenario did not finish")
        for req_id, request in self.requests.items():
            self.outputs[req_id] = list(request.output_token_ids)


def _scenario(h: _AsyncHarness) -> None:
    """Prompts that end a few tokens short of a block boundary, so most
    verifies run near one, and more requests than rows, so finished
    requests' blocks serve the later ones."""
    for i, prompt_len in enumerate((1019, 1022, 2045, 3066, 1020, 2047, 700, 4093)):
        h.add(f"r{i}", prompt_len, max_tokens=24)


@pytest.mark.parametrize("num_blocks", [48, 28])
@pytest.mark.parametrize("mode", ["none", "align"])
def test_async_verifies_resume_from_the_state_of_their_committed_count(
    mode, num_blocks
):
    """Under asynchronous scheduling every verify drains before the next build,
    every start state read is the request's own, every count is committed
    next to block boundaries, and the outputs equal a synchronous run's. With
    28 blocks in mode ``align`` requests are also preempted and resume."""
    reference = _AsyncHarness(num_blocks, mode=mode, asynchronous=False)
    _scenario(reference)
    reference.run()

    h = _AsyncHarness(num_blocks, mode=mode, asynchronous=True)
    _scenario(h)
    h.run()

    assert h.outputs == reference.outputs
    assert all(len(o) == 24 for o in h.outputs.values())
    assert set(range(1, T + 1)) <= h.device.boundary_counts
    assert h.device.verifies > 40
    assert h.scheduled_ahead > 20, "steps were scheduled while a verify was in flight"
    if mode == "align":
        assert h.cached_states_checked
        if num_blocks == 28:
            assert h.preempted


@pytest.mark.parametrize("num_blocks", [48, 28])
@pytest.mark.parametrize("mode", ["none", "align"])
def test_chained_verifies_continue_the_unread_verify(mode, num_blocks):
    """With ``supports_chained_spec_verify`` most verifies are submitted over
    the unread verify before them. The device builds each chained row from
    the count it accepted, and reads the start block of that count
    (``start_blocks_by_count``); the predecessor's commit fills the chained
    step's rows, whose walk then commits what the device computed. Every
    start state read is the request's own and the outputs equal a
    synchronous run's. Joins and prefills (at the empty step the scheduler
    returns ahead of them) and preemptions drain instead."""
    reference = _AsyncHarness(num_blocks, mode=mode, asynchronous=False)
    _scenario(reference)
    reference.run()

    h = _AsyncHarness(num_blocks, mode=mode, asynchronous=True, chained=True)
    _scenario(h)
    h.run()

    assert h.outputs == reference.outputs
    assert set(range(1, T + 1)) <= h.device.boundary_counts
    controller = h.runner.async_decode
    assert h.device.chained == controller._chained_verify_submissions > 10
    assert controller._overlapped_unsafe_submissions == 0
    # TTScheduler returns an empty step ahead of each prefill while a verify
    # is unread; that step schedules no rows, so it drains for "rows".
    assert controller._chain_drains["rows"]
    if mode == "align":
        assert h.cached_states_checked


@pytest.mark.parametrize(
    ("num_blocks", "sampled"), [(24, False), (28, False), (80, False), (80, True)]
)
def test_a_chained_verify_next_to_a_block_boundary_is_the_drained_verify(
    num_blocks, sampled
):
    """In mode ``align`` vLLM schedules a verify over the unread one counting
    that one's drafts as accepted, and allocates the verify's window by that
    count. Next to a block boundary the window then begins one column past
    the column that holds, by position, the row's last query token at a low
    count; a drained verify writes that window for every count, and reads
    the start block of the count committed. A chained verify after every
    such count (1 to K) hands the device the same start blocks and window
    blocks as a drained run, row for row and step for step, and leaves the
    same GDN state in every block; the outputs equal a synchronous run's.
    With 28 blocks greedy requests are also preempted and resume."""
    prompts = [BLOCK_SIZE * (1 + i % 3) - 8 - 4 * (i // 3) - i % 2 for i in range(16)]

    def harness(**kwargs) -> _AsyncHarness:
        h = _AsyncHarness(num_blocks, mode="align", sampled_ids=sampled, **kwargs)
        h.device.drafts = _boundary_drafts
        for i, prompt_len in enumerate(prompts):
            sampling = dict(temperature=0.8, top_k=4, seed=7 + i) if sampled else {}
            h.add(f"r{i}", prompt_len, max_tokens=40, **sampling)
        return h

    reference = harness(asynchronous=False)
    reference.run()
    drained = harness(asynchronous=True)
    drained.run()
    h = harness(asynchronous=True, chained=True)
    # The predecessor's count of each chained verify whose window begins past
    # the column of its last query token by position.
    crossed: list[int] = []
    previous: dict[str, int] = {}
    verify_row = h.device._verify_row

    def note_crossing(row, req_id, p0, block, starts, ids, record, rows, sampling):
        window = h.runner._mamba_groups.windows[req_id].column
        if record.chained and window != (p0 + K) // BLOCK_SIZE:
            crossed.append(p0 - previous[req_id])
        previous[req_id] = p0
        verify_row(row, req_id, p0, block, starts, ids, record, rows, sampling)

    h.device._verify_row = note_crossing
    h.run()

    assert h.outputs == reference.outputs == drained.outputs
    assert all(len(o) == 40 for o in h.outputs.values())
    assert set(crossed) == set(range(1, T))
    assert h.device.inputs == drained.device.inputs
    assert h.ledger.states == drained.ledger.states
    assert h.cached_states_checked
    controller = h.runner.async_decode
    assert h.device.chained == controller._chained_verify_submissions > 30
    assert controller._overlapped_unsafe_submissions == 0
    assert set(controller._chain_drains) <= {"pending", "rows", "lifecycle"}
    if num_blocks == 28:
        assert h.preempted


LEAVING = (
    ("r0", 1019, 28),
    ("r1", 700, 16),
    ("r2", 2045, 32),
    ("r3", 900, 24),
    ("r4", 1500, 40),
    ("r5", 1022, 36),
)
# Requests stopped by a token of their own output, and the one aborted.
LEAVING_STOPPED = ("r2", "r5")
LEAVING_ABORTED = "r4"


def _leaving_scenario(
    h: _AsyncHarness, *, sampled: bool, stops: dict[str, int] | None = None
) -> None:
    """Two more requests than rows, of different lengths, so that once the
    last one has joined requests finish in every row with nothing waiting:
    by max_tokens, by a stop token (``stops``) and, on an asynchronous run,
    one by an abort. With ``sampled`` every other request is a seeded sampled
    one."""
    for i, (req_id, prompt_len, max_tokens) in enumerate(LEAVING):
        sampling: dict = {}
        if sampled and i % 2:
            sampling.update(temperature=0.8, top_k=5, seed=21 + i)
        if stops and req_id in stops:
            sampling.update(stop_token_ids=[stops[req_id]])
        h.add(req_id, prompt_len, max_tokens=max_tokens, **sampling)
    if not h.asynchronous:
        return
    update = h._update

    def abort(scheduler_output, output):
        update(scheduler_output, output)
        request = h.scheduler.requests.get(LEAVING_ABORTED)
        if request is not None and len(request.output_token_ids) >= 25:
            h.scheduler.finish_requests(LEAVING_ABORTED, RequestStatus.FINISHED_ABORTED)

    h._update = abort


def _leaving_stops(num_blocks: int, sampled: bool) -> dict[str, int]:
    """A stop token for each of ``LEAVING_STOPPED``: the first token of its
    output that it had not produced before its eleventh."""
    h = _AsyncHarness(num_blocks, mode="align", sampled_ids=sampled, asynchronous=False)
    _leaving_scenario(h, sampled=sampled)
    h.run()
    stops = {}
    for req_id in LEAVING_STOPPED:
        out = h.outputs[req_id]
        stops[req_id] = next(
            t for i, t in enumerate(out) if i >= 10 and t not in out[:i]
        )
    return stops


class _LeavingRows:
    """Per ``execute_model`` call over an unread verify: the predecessor's
    rows that leave (finished, or not scheduled), whether one of them was not
    its last row, and whether the step chained."""

    def __init__(self, h: _AsyncHarness):
        self.steps: list[SimpleNamespace] = []
        # Steps over an unread verify that schedule nothing.
        self.empty_drains = 0
        runner, execute = h.runner, h.runner.execute_model
        controller = runner.async_decode

        def record(scheduler_output):
            with runner._steady_decode_lock:
                pending = list(runner._pending_async_steps)
            previous = (
                list(pending[0].model_input.row_req_ids or ())
                if len(pending) == 1 and hasattr(pending[0], "chain_successor")
                else []
            )
            scheduled = set(scheduler_output.num_scheduled_tokens)
            finished = set(scheduler_output.finished_req_ids)
            before = controller._chained_verify_submissions
            rows_drains = controller._chain_drains["rows"]
            result = execute(scheduler_output)
            leaving = [r for r in previous if r not in scheduled]
            if previous and not scheduled:
                self.empty_drains += 1
            self.steps.append(
                SimpleNamespace(
                    finished=[r for r in leaving if r in finished],
                    skipped=[r for r in leaving if r not in finished],
                    moved=bool(leaving)
                    and previous[: len(previous) - len(leaving)]
                    != [r for r in previous if r in scheduled],
                    prefill=bool(scheduler_output.scheduled_new_reqs),
                    chained=controller._chained_verify_submissions > before,
                    rows_drain=controller._chain_drains["rows"] > rows_drains,
                    empty=not scheduled,
                )
            )
            return result

        runner.execute_model = record


@pytest.mark.parametrize(
    ("num_blocks", "sampled"), [(48, False), (80, False), (80, True)]
)
def test_a_chained_verify_continues_rows_that_moved(num_blocks, sampled):
    """A request that finishes (max_tokens, a stop token, an abort) or that
    vLLM leaves out because its in-flight step reaches max_tokens leaves the
    persistent batch, and ``InputBatch.condense`` moves the last row into its
    place. The next verify still chains: ``slot_remap`` names each row's
    predecessor row, and the device builds the row from that row's count. The
    device gets the same start blocks and window blocks as a drained run, row
    for row and step for step, and leaves the same GDN state in every block;
    the outputs equal a synchronous run's. Only steps that schedule nothing
    drain for "rows". With 48 blocks a request is also preempted and
    resumes."""

    stops = _leaving_stops(num_blocks, sampled)

    def harness(**kwargs) -> _AsyncHarness:
        h = _AsyncHarness(num_blocks, mode="align", sampled_ids=sampled, **kwargs)
        _leaving_scenario(h, sampled=sampled, stops=stops)
        return h

    reference = harness(asynchronous=False)
    reference.run()
    drained = harness(asynchronous=True)
    drained.run()
    h = harness(asynchronous=True, chained=True)
    leaving = _LeavingRows(h)
    h.run()

    aborted = h.outputs.pop(LEAVING_ABORTED)
    assert aborted == drained.outputs.pop(LEAVING_ABORTED)
    assert 25 <= len(aborted) < 40
    assert aborted == reference.outputs.pop(LEAVING_ABORTED)[: len(aborted)]
    assert h.outputs == reference.outputs == drained.outputs
    assert h.requests[LEAVING_ABORTED].status == RequestStatus.FINISHED_ABORTED
    assert {h.requests[r].status for r in LEAVING_STOPPED} == {
        RequestStatus.FINISHED_STOPPED
    }
    assert h.device.inputs == drained.device.inputs
    assert h.ledger.states == drained.ledger.states
    assert h.cached_states_checked
    controller = h.runner.async_decode
    assert controller._overlapped_unsafe_submissions == 0
    assert h.device.chained == controller._chained_verify_submissions
    # A finished row that was not the last one, and a row vLLM left out, were
    # chained over.
    assert h.device.moved_rows
    assert any(s.chained and s.moved and s.finished for s in leaving.steps)
    assert any(s.chained and s.skipped for s in leaving.steps)
    # Every step over an unread verify that still schedules a predecessor
    # row chains: what drains for "rows" schedules nothing (the empty step
    # ``TTScheduler`` returns ahead of a prefill, or every row left).
    assert controller._chain_drains["rows"] == leaving.empty_drains
    if num_blocks == 48:
        assert h.preempted


@pytest.mark.parametrize(
    ("num_blocks", "sampled", "scenario"),
    [(80, False, "joins"), (28, False, "joins"), (80, True, "leaving")],
)
def test_a_prefill_runs_behind_the_unread_verify(num_blocks, sampled, scenario):
    """With ``supports_prefill_over_pending_spec`` ``TTScheduler`` returns a
    join's prefill while a verify is held, without the empty release step
    before it, and ``TTModelRunner`` submits the prefill behind the unread
    verify: the joining request often takes the state slot of a request that
    finished in that verify's rows. The verify is read, committed and proposed
    after the prefill, and the step after the prefill does not chain on it.
    The device gets the same rows, start blocks and window blocks, step for
    step, as the chained run that drains at the release step, and leaves the
    same GDN state; the outputs equal a synchronous run's. With 28 blocks
    requests are also preempted and resume: their prefill drains."""
    if scenario == "joins":
        stops = None

        def setup(h):
            _scenario(h)

    else:
        stops = _leaving_stops(num_blocks, sampled)

        def setup(h):
            _leaving_scenario(h, sampled=sampled, stops=stops)

    def harness(**kwargs) -> _AsyncHarness:
        h = _AsyncHarness(num_blocks, mode="align", sampled_ids=sampled, **kwargs)
        setup(h)
        return h

    reference = harness(asynchronous=False)
    reference.run()
    base = harness(asynchronous=True, chained=True)
    base.run()
    h = harness(asynchronous=True, chained=True, prefill_over_pending=True)
    leaving = _LeavingRows(h)
    h.run()

    if scenario == "leaving":
        aborted = h.outputs.pop(LEAVING_ABORTED)
        assert aborted == base.outputs.pop(LEAVING_ABORTED)
        assert aborted == reference.outputs.pop(LEAVING_ABORTED)[: len(aborted)]
    assert h.outputs == reference.outputs == base.outputs
    assert h.device.inputs == base.device.inputs
    assert h.ledger.states == base.ledger.states
    assert h.cached_states_checked
    controller = h.runner.async_decode
    assert controller._overlapped_unsafe_submissions == 0
    # Every prefill behind an unread verify is one the runner counted, and
    # took the place of a release step that drained in the base run.
    assert h.device.prefills_behind == controller._prefills_over_pending > 0
    assert base.device.prefills_behind == 0
    if num_blocks == 80:
        assert h.device.slots_reused_behind, "no prefill took a finished row's slot"
    # What drains for "rows" schedules nothing; the release steps ahead of
    # the prefills that went behind a verify are gone.
    assert all(step.empty for step in leaving.steps if step.rows_drain)
    assert (
        controller._chain_drains["rows"]
        < base.runner.async_decode._chain_drains["rows"]
    )
    if num_blocks == 28:
        assert h.preempted


@pytest.mark.parametrize("chained", [False, True])
def test_a_decode_row_keeps_its_proposal_across_a_prefill_step(chained):
    """Under asynchronous scheduling a verify's commit, and the proposal it
    publishes, run at the next step's build, after ``_update_states``. When a
    request arrives, that step is its prefill, and a prefill step takes every
    decode row out of the persistent batch. The decode rows' proposals still
    reach their next verify, which checks all K drafts, as on a synchronous
    launch; only the new request's first decode has none."""
    decoding, arriving = [("a", 700, 40), ("b", 900, 40)], ("c", 700, 8)
    h = _AsyncHarness(48, mode="align", asynchronous=True, chained=chained)
    for request in decoding:
        h.add(*request)
    steps: list[tuple[str, dict[str, int]]] = []
    device = h.device
    prefill, verify, narrow = device.prefill_forward, device._verify, device._narrow

    def record_prefill(tokens, prompt_lens, start_pos, mamba_state, **kwargs):
        steps.append(("prefill", dict.fromkeys(device._live(), 0)))
        return prefill(tokens, prompt_lens, start_pos, mamba_state, **kwargs)

    def record_verify(live, tokens, start_pos, mamba_state, kwargs):
        num_valid = kwargs["num_valid_drafts"].tolist()
        steps.append(("verify", dict(zip(live, num_valid))))
        return verify(live, tokens, start_pos, mamba_state, kwargs)

    def record_narrow(live, tokens, start_pos, mamba_state, sampling=None):
        steps.append(("ordinary", dict.fromkeys(live, 0)))
        return narrow(live, tokens, start_pos, mamba_state, sampling)

    device.prefill_forward = record_prefill
    device._verify = record_verify
    device._narrow = record_narrow
    update, updates = h._update, []

    def arrive_mid_decode(scheduler_output, output):
        update(scheduler_output, output)
        updates.append(scheduler_output)
        if len(updates) == 6:
            h.add(*arriving)

    h._update = arrive_mid_decode
    h.run()

    joined = next(i for i, (_, rows) in enumerate(steps) if "c" in rows)
    assert steps[joined] == ("prefill", {"c": 0})
    assert steps[joined - 1][0] == "verify", "a and b were decoding when c arrived"
    assert steps[joined + 1] == ("verify", {"a": K, "b": K, "c": 0})
    reference = _AsyncHarness(48, mode="align", asynchronous=False)
    for request in (*decoding, arriving):
        reference.add(*request)
    reference.run()
    assert h.outputs == reference.outputs


def test_a_row_near_max_model_len_drains_its_verify():
    """A row within two steps of ``max_model_len`` is not chained: the
    predecessor's commit could be cut there. vLLM then schedules fewer than
    ``1 + K`` tokens for it too."""
    h = _AsyncHarness(48, mode="none", asynchronous=True, chained=True)
    h.add("long", MAX_MODEL_LEN - 3 * T, max_tokens=2 * T)
    h.run()
    reference = _AsyncHarness(48, mode="none", asynchronous=False)
    reference.add("long", MAX_MODEL_LEN - 3 * T, max_tokens=2 * T)
    reference.run()
    assert h.outputs == reference.outputs
    drains = h.runner.async_decode._chain_drains
    assert {"width", "max_model_len"} & {r for r, n in drains.items() if n}


def test_a_device_count_unequal_to_the_walk_raises_before_its_step_is_served():
    """A chained step continues the device's count. When that count differs
    from the runner's walk of the same ids, the predecessor's propose raises
    (as the model's does), during the chained submission, before the
    scheduler is handed the chained step's tokens."""
    h = _AsyncHarness(48, mode="align", asynchronous=True, chained=True)
    _scenario(h)
    # Verify 29 is r4's row of a chained verify that another chained verify
    # of the same rows follows, with no prefill or join between them, so the
    # device builds that successor from the wrong count. A miscount on a
    # verify that a prefill or join follows drains instead, and no chained
    # step would run on it.
    h.device.miscount = lambda verifies, req_id, record: verifies >= 29
    served = []
    update = h.scheduler.update_from_output

    def record(scheduler_output, output):
        served.append(dict(zip(output.req_ids, output.sampled_token_ids)))
        return update(scheduler_output, output)

    h.scheduler.update_from_output = record
    with pytest.raises(ValueError, match="the device accepted"):
        h.run()
    assert h.device.miscounted_successors, "a chained step ran on the wrong count"
    reference = _AsyncHarness(48, mode="align", asynchronous=False)
    _scenario(reference)
    reference.run()
    for req_id, request in h.requests.items():
        tokens = list(request.output_token_ids)
        assert tokens == reference.outputs[req_id][: len(tokens)], req_id


@pytest.mark.parametrize("chained", [False, True])
def test_a_host_answering_model_never_gets_resident_decode_inputs(chained):
    """The Qwen3.8 adapter declares ``supports_async_decode`` while its
    ordinary decode returns host ids read inside the call. With traced,
    device-sampled decoding (the deployment's), ``submit_decode`` records
    every such step as host-answered, so ``plan_decode_reload`` reloads the
    next ordinary decode (``host_answered_to_plain``), as it does after a
    verify (``verify_to_plain``) and a prefill (``note_prefill_submitted``):
    no ordinary decode is ever planned with resident inputs or registered as
    overlap-safe. The device model raises on ``reload_inputs=False``."""
    reference = _AsyncHarness(
        48, mode="align", asynchronous=False, trace_mode="all", device_sampling=True
    )
    h = _AsyncHarness(
        48,
        mode="align",
        asynchronous=True,
        chained=chained,
        trace_mode="all",
        device_sampling=True,
    )
    for harness in (reference, h):
        _scenario(harness)
        # A sampled request is not speculable here, so the steps it shares
        # are ordinary decodes, one after another.
        harness.add("sampled", 900, max_tokens=24, temperature=1.0)
    reference.run()
    controller = h.runner.async_decode
    plans, registered = [], []
    plan, register = (
        controller.plan_decode_reload,
        controller.register_pending_async_step,
    )

    def planned(model_input):
        result = plan(model_input)
        plans.append(
            (model_input.spec_mode, model_input.perform_device_sampling, result)
        )
        return result

    def registered_step(step, *, steady, overlap_ok):
        registered.append((type(step).__name__, steady, overlap_ok))
        return register(step, steady=steady, overlap_ok=overlap_ok)

    controller.plan_decode_reload = planned
    controller.register_pending_async_step = registered_step
    h.run()

    assert h.outputs == reference.outputs
    ordinary = [p for mode, sampled, p in plans if mode is None]
    assert h.device.plain_after_plain > 5
    assert all(sampled for mode, sampled, _ in plans if mode is None)
    assert ordinary and all(p.reload_inputs and not p.overlap_safe for p in ordinary)
    assert not [r for r in registered if r[0] == "AsyncTTModelRunnerOutput" and r[2]]
    assert (h.device.chained > 0) == chained


# Chained verification of device-sampled rows (``spec_ids_are_sampled`` with
# ``supports_chained_spec_verify``).

SEEDED = ("s0", "s1", "s2")
# A pool in which ``_sampled_scenario`` preempts no request in either mode.
SAMPLED_BLOCKS = 80


def _sampled_scenario(h: _AsyncHarness, *, unseeded: bool = True) -> None:
    """Greedy and seeded sampled requests (top-k within the device bound),
    with ``unseeded`` one sampled request without a seed, more of them than
    rows; the drafter declines a row after every seventh position, so some
    chained rows verify no draft."""
    for i, prompt_len in enumerate((1019, 2045, 700)):
        h.add(f"g{i}", prompt_len, max_tokens=24)
    for req_id, prompt_len, seed in zip(SEEDED, (1022, 3066, 900), (11, 12, 13)):
        h.add(req_id, prompt_len, max_tokens=24, temperature=0.8, top_k=5, seed=seed)
    if unseeded:
        h.add("u0", 1500, max_tokens=24, temperature=1.0, top_k=MAX_DEVICE_TOP_K)
    h.device.decline = lambda req_id, truth: len(truth) % 7 == 3


def _sampled_rows(records, select) -> list[tuple[str, int]]:
    """``(request, count)`` of the sampled rows of the verify records
    ``select`` keeps."""
    return [
        (req_id, record.counts[row])
        for record in records
        if select(record)
        for row, req_id in enumerate(record.rows)
        if record.sampled[row] and row < len(record.counts)
    ]


@pytest.mark.parametrize("mode", ["none", "align"])
def test_device_sampled_rows_chain_like_greedy_rows(mode):
    """On a device-sampled step a sampled row's ids are draws, so it chains
    like a greedy row: a chained verify continues the device's count over the
    predecessor's draws, and its rows draw at the positions that count gives,
    a seeded row with the seed of that position. No sampled row drains
    (``sampling``); predecessors of chained steps commit every count, rows
    the drafter declined included; seeded and greedy outputs equal a
    synchronous run's."""
    reference = _AsyncHarness(
        SAMPLED_BLOCKS, mode=mode, asynchronous=False, sampled_ids=True
    )
    _sampled_scenario(reference)
    reference.run()

    h = _AsyncHarness(
        SAMPLED_BLOCKS, mode=mode, asynchronous=True, chained=True, sampled_ids=True
    )
    _sampled_scenario(h)
    h.run()

    for req_id, output in reference.outputs.items():
        if req_id != "u0":
            assert h.outputs[req_id] == output, req_id
    assert len(h.outputs["u0"]) == 24
    controller = h.runner.async_decode
    assert controller._chain_drains["sampling"] == 0
    assert controller._overlapped_unsafe_submissions == 0
    chained = _sampled_rows(h.device.records, lambda r: r.chained)
    assert {req_id for req_id, _ in chained} == {*SEEDED, "u0"}
    followed = _sampled_rows(h.device.records, lambda r: r.followed)
    assert {count for _, count in followed} == set(range(1, T + 1))
    assert any(
        not record.offered[row]
        for record in h.device.records
        if record.followed
        for row in range(len(record.offered))
        if record.rows[row] in SEEDED
    ), "a chained step continued a seeded row the drafter declined"


@pytest.mark.parametrize(
    "ineligible",
    [
        # More candidates than the device sampler keeps (max_device_top_k).
        {"temperature": 1.0, "top_k": 20},
        # Logprobs: host-sampled on this one-device launch
        # (check_perform_device_sampling), and never certifiable by the walk.
        {"temperature": 0.0, "logprobs": 1},
    ],
)
def test_a_step_the_device_cannot_sample_drains_its_sampled_rows(ineligible):
    """While a request the device cannot sample is in the batch, every step
    samples on the host: a sampled row is then not certifiable by the id
    walk, so its steps are ordinary decodes and no verify chains over them.
    Before the request arrives and after it finishes, sampled rows chain."""
    h = _AsyncHarness(
        SAMPLED_BLOCKS, mode="align", asynchronous=True, chained=True, sampled_ids=True
    )
    for req_id, prompt_len, seed in (("g0", 1019, None), ("s0", 1022, 5)):
        sampling = {} if seed is None else {"temperature": 0.8, "top_k": 5}
        h.add(req_id, prompt_len, max_tokens=60, seed=seed, **sampling)
    kinds: list[tuple[str, bool, list[str]]] = []
    device = h.device
    verify, narrow = device._verify, device._narrow

    def record_verify(live, tokens, start_pos, mamba_state, kwargs):
        kinds.append(("verify", "spec_predecessor" in kwargs, list(live)))
        return verify(live, tokens, start_pos, mamba_state, kwargs)

    def record_narrow(live, tokens, start_pos, mamba_state, sampling=None):
        kinds.append(("ordinary", sampling is not None, list(live)))
        return narrow(live, tokens, start_pos, mamba_state, sampling)

    device._verify, device._narrow = record_verify, record_narrow
    update, updates = h._update, []

    def arrive(scheduler_output, output):
        update(scheduler_output, output)
        updates.append(scheduler_output)
        if len(updates) == 12:
            h.add("x", 800, max_tokens=12, **ineligible)

    h._update = arrive
    h.run()

    with_x = [i for i, (_, _, rows) in enumerate(kinds) if "x" in rows]
    assert with_x
    first, last = with_x[0], with_x[-1]
    assert all(kinds[i][:2] == ("ordinary", False) for i in with_x), (
        "every step x shares samples on the host"
    )
    chained_s0 = [
        i
        for i, (kind, chained, rows) in enumerate(kinds)
        if kind == "verify" and chained and "s0" in rows
    ]
    assert any(i < first for i in chained_s0) and any(i > last for i in chained_s0)
    reference = _AsyncHarness(
        SAMPLED_BLOCKS, mode="align", asynchronous=False, sampled_ids=True
    )
    reference.add("g0", 1019, max_tokens=60)
    reference.run()
    assert h.outputs["g0"] == reference.outputs["g0"]
    assert len(h.outputs["s0"]) == 60 and len(h.outputs["x"]) == 12


def test_a_build_that_samples_on_the_host_refuses_a_chained_sampled_row():
    """``plan_chain`` reads the device-sampling decision before
    ``_update_states``; the build decides again, and a chained sampled row
    its own decision does not certify raises instead of being verified
    against the argmax. Here the build's decision turns to host sampling
    once verifies chain, while ``plan_chain``'s does not."""
    h = _AsyncHarness(
        SAMPLED_BLOCKS, mode="none", asynchronous=True, chained=True, sampled_ids=True
    )
    _sampled_scenario(h, unseeded=False)
    runner = h.runner
    decide, build = runner._decode_device_sampled, runner.build_model_input
    building = []

    def build_model_input(*args, **kwargs):
        building.append(True)
        try:
            return build(*args, **kwargs)
        finally:
            building.pop()

    def decide_in_build(*args, **kwargs):
        if building and h.device.chained >= 5:
            return False
        return decide(*args, **kwargs)

    runner.build_model_input = build_model_input
    runner._decode_device_sampled = decide_in_build
    with pytest.raises(RuntimeError, match="not certifiable by the id walk"):
        h.run()


# Penalties on the verify (``supports_spec_device_penalties`` with
# ``spec_ids_are_sampled`` and ``supports_chained_spec_verify``).

PENALISED = {
    "p0": {"repetition_penalty": 1.3},
    "p1": {"temperature": 0.8, "top_k": 5, "seed": 21, "presence_penalty": 0.5},
}


def _penalised_scenario(h: _AsyncHarness) -> None:
    """Two greedy requests next to a greedy penalised and a sampled penalised
    one; the drafter declines a row after every seventh position."""
    for i, prompt_len in enumerate((1019, 700)):
        h.add(f"g{i}", prompt_len, max_tokens=24)
    for (req_id, sampling), prompt_len in zip(PENALISED.items(), (1100, 900)):
        h.add(req_id, prompt_len, max_tokens=24, **sampling)
    h.device.decline = lambda req_id, truth: len(truth) % 7 == 3


def _history(tokens, row: int) -> list[int]:
    return [t for t in tokens[row].tolist() if t != -1]


def _vocab_prompt(seed: int, length: int) -> list[int]:
    """A prompt of ids the host sampler's penalties can index (below VOCAB)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(FIRST_ID, VOCAB, (length,), generator=g).tolist()


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "not-declared"])
def test_penalised_rows_verify_and_chain_with_their_histories(monkeypatch, declared):
    """A model declaring ``supports_spec_device_penalties`` applies the
    penalties itself, so a decode step with a penalised row samples on the
    device: the penalised rows verify in ``argmax_ids`` and chain (no
    ``sampling`` drain), and every such step hands the model each row's prompt
    and committed output. On a chained verify that output lacks exactly the
    predecessor's commit, which the model adds itself. Without the declaration
    the penalised rows never reach a verify and their steps sample on the
    host. Greedy outputs equal a synchronous run's."""
    monkeypatch.setitem(globals(), "_prompt", _vocab_prompt)
    h = _AsyncHarness(
        SAMPLED_BLOCKS, mode="align", asynchronous=True, chained=True, sampled_ids=True
    )
    h.device.model_capabilities["supports_spec_device_penalties"] = declared
    _penalised_scenario(h)
    device, steps = h.device, []
    decode = device.decode_forward

    def record(*, tokens, start_pos, mamba_state, **kwargs):
        live = device._live()
        rows = [row for row, req_id in enumerate(live) if req_id in PENALISED]
        if rows:
            verify = kwargs.get("spec_mode") is not None
            predecessor = kwargs.get("spec_predecessor")
            sampled = kwargs.get("sampling_params") is not None
            steps.append((verify, predecessor is not None, sampled))
            if sampled:
                remap = kwargs.get("slot_remap")
                for row in rows:
                    req_id = live[row]
                    prompt = _history(kwargs["prompt_tokens"], row)
                    output = _history(kwargs["output_tokens"], row)
                    assert prompt == list(h.requests[req_id].prompt_token_ids)
                    truth = device.truth[req_id]
                    if predecessor is not None:
                        source = row if remap is None else int(remap[row])
                        record = predecessor.argmax_ids.record
                        truth = truth[: len(truth) - record.counts[source]]
                    assert prompt + output == truth, (req_id, len(output))
        return decode(
            tokens=tokens, start_pos=start_pos, mamba_state=mamba_state, **kwargs
        )

    device.decode_forward = record
    h.run()

    reference = _AsyncHarness(
        SAMPLED_BLOCKS, mode="align", asynchronous=False, sampled_ids=True
    )
    _penalised_scenario(reference)
    reference.run()
    for req_id in ("g0", "g1", "p0"):
        assert h.outputs[req_id] == reference.outputs[req_id], req_id
    assert len(h.outputs["p1"]) == 24
    verifies = [s for s in steps if s[0]]
    if declared:
        assert all(sampled for _, _, sampled in steps), (
            "a penalised step sampled on the host"
        )
        assert any(chained for _, chained, _ in verifies), "no penalised row chained"
        assert h.runner.async_decode._chain_drains["sampling"] == 0
    else:
        assert not verifies, "a penalised row reached a verify"
        assert not any(sampled for _, _, sampled in steps)


def test_a_chained_step_built_for_another_path_raises():
    """Only the asynchronous decode path completes a chained verify
    (``complete_chain``). ``plan_chain`` drains each step that would build for
    another path, such as a structured-output step, which defers device
    sampling. Here the build defers device sampling once verifies chain, while
    ``plan_chain`` chained the step: it raises instead of dropping the
    predecessor's commit."""
    h = _AsyncHarness(48, mode="none", asynchronous=True, chained=True)
    _scenario(h)
    runner = h.runner
    build = runner.build_model_input

    def build_model_input(*args, **kwargs):
        model_input = build(*args, **kwargs)
        if kwargs.get("chain") is not None and h.device.chained >= 3:
            return dataclasses.replace(
                model_input, perform_device_sampling=True, defer_device_sampling=True
            )
        return model_input

    runner.build_model_input = build_model_input
    with pytest.raises(RuntimeError, match="complete_chain"):
        h.run()


def test_a_sampled_device_count_unequal_to_the_walk_raises_before_its_step_is_served():
    """The guard covers sampled rows: a device count over the draws one draft
    longer than the runner's walk of the same draws, on a sampled row of a
    chained verify, raises at that verify's propose, before vLLM receives the
    tokens of the chained step built on it."""
    h = _AsyncHarness(
        SAMPLED_BLOCKS, mode="align", asynchronous=True, chained=True, sampled_ids=True
    )
    _sampled_scenario(h, unseeded=False)
    h.device.miscount = lambda verifies, req_id, record: (
        verifies > 20 and record.chained and req_id in SEEDED
    )
    with pytest.raises(ValueError, match="the device accepted"):
        h.run()
    assert h.device.miscounted_successors, "a chained step ran on the wrong count"
    reference = _AsyncHarness(
        SAMPLED_BLOCKS, mode="align", asynchronous=False, sampled_ids=True
    )
    _sampled_scenario(reference, unseeded=False)
    reference.run()
    for req_id, request in h.requests.items():
        tokens = list(request.output_token_ids)
        assert tokens == reference.outputs[req_id][: len(tokens)], req_id


def _collect_logprobs(h: _AsyncHarness) -> dict[str, list[tuple]]:
    """Wrap the harness's update to collect, per request, each published
    token with its logprob row: ``(prefix, token, ids, logprobs, rank)``,
    ``prefix`` the request's token history before the token."""
    published: dict[str, list[tuple]] = {}
    update = h._update

    def collect(scheduler_output, output):
        resolved = output.get_output() if hasattr(output, "get_output") else output
        lists = resolved.logprobs
        for index, req_id in enumerate(resolved.req_ids):
            tokens = resolved.sampled_token_ids[index]
            request = h.requests[req_id]
            if lists is None or request.sampling_params.logprobs is None:
                continue
            rows = lists.slice_request(index, len(tokens))
            history = list(request.all_token_ids)
            for j, token in enumerate(tokens):
                published.setdefault(req_id, []).append(
                    (
                        history + list(tokens[:j]),
                        int(token),
                        [int(i) for i in rows.logprob_token_ids[j]],
                        [float(v) for v in rows.logprobs[j]],
                        int(rows.sampled_token_ranks[j]),
                    )
                )
        update(scheduler_output, resolved)

    h._update = collect
    return published


@pytest.mark.parametrize("mode", ["none", "align"])
def test_a_verify_payload_serves_logprobs_rows_and_they_chain(mode):
    """On a device declaring ``supports_spec_device_logprobs`` a request with
    logprobs 0 to 20 is certifiable by the id walk: greedy, and sampled on a
    device-sampled step, which a verify stays on although this one-device
    launch keeps the device sampler's own logprobs on the host. Its steps are
    ``argmax_ids`` verifies sent ``spec_num_logprobs`` (the device checks the
    rows it was sent), they chain like any other row, and every token a verify
    commits is published with the payload's logprob, rank and top ids at its
    column. Ordinary decodes (prompt ends, declined drafts) sample those rows
    on the host, whose logits give logprob 0."""
    h = _AsyncHarness(
        SAMPLED_BLOCKS,
        mode=mode,
        asynchronous=True,
        chained=True,
        sampled_ids=True,
        device_logprobs=True,
    )
    h.add("g0", 1019, max_tokens=24)
    h.add("l0", 1022, max_tokens=24, logprobs=0)
    h.add("l5", 2045, max_tokens=24, logprobs=5)
    h.add("s3", 900, max_tokens=24, temperature=0.8, top_k=5, seed=7, logprobs=3)
    h.add("s20", 1500, max_tokens=24, temperature=1.0, top_k=5, logprobs=20)
    h.device.decline = lambda req_id, truth: len(truth) % 7 == 3
    published = _collect_logprobs(h)
    h.run()

    controller = h.runner.async_decode
    assert controller._chain_drains["sampling"] == 0
    chained = {
        req_id
        for record in h.device.records
        if record.chained
        for req_id in record.rows
    }
    assert {"l0", "l5", "s3", "s20"} <= chained
    for req_id, asked in (("l0", 0), ("l5", 5), ("s3", 3), ("s20", 20)):
        # The runner publishes a verify's whole commit; vLLM cuts it at
        # max_tokens afterwards.
        entries = published[req_id]
        assert len(entries) >= 24, req_id
        from_device = 0
        for prefix, token, ids, logprobs, rank in entries:
            assert ids[0] == token, (req_id, ids)
            assert len(ids) >= asked + 1, (req_id, ids)
            if logprobs[0] == 0.0:
                continue  # the host sampler's, on an ordinary decode
            from_device += 1
            assert logprobs[0] == pytest.approx(_device_logprob(prefix, token))
            assert rank == _device_rank(prefix, token)
            top_ids, top_logprobs = _device_top(prefix)
            n = len(ids) - 1
            assert ids[1:] == top_ids[:n], req_id
            assert logprobs[1:] == pytest.approx(top_logprobs[:n]), req_id
        assert from_device >= 16, (req_id, from_device)
    reference = _AsyncHarness(
        SAMPLED_BLOCKS, mode=mode, asynchronous=False, sampled_ids=True
    )
    reference.add("g0", 1019, max_tokens=24)
    reference.add("l0", 1022, max_tokens=24)
    reference.add("l5", 2045, max_tokens=24)
    reference.run()
    for req_id in ("g0", "l0", "l5"):
        assert h.outputs[req_id] == reference.outputs[req_id], req_id


@pytest.mark.parametrize(
    "logprobs, extra",
    [
        # More than the payload holds.
        (21, {}),
        # Every token of the vocabulary.
        (-1, {}),
    ],
)
def test_a_logprobs_request_the_payload_cannot_serve_stays_uncertifiable(
    logprobs, extra
):
    """A request whose logprobs the payload does not hold is not certifiable
    by the id walk even on a device declaring the capability: on this launch,
    which offers ``argmax_ids`` only, it is offered no drafts and its steps
    are ordinary decodes sampled on the host."""
    h = _AsyncHarness(
        SAMPLED_BLOCKS,
        mode="none",
        asynchronous=True,
        chained=True,
        sampled_ids=True,
        device_logprobs=True,
    )
    h.add("g0", 1019, max_tokens=16)
    h.add("x", 800, max_tokens=12, logprobs=logprobs, **extra)
    verified: list[list[str]] = []
    verify = h.device._verify

    def record_verify(live, tokens, start_pos, mamba_state, kwargs):
        verified.append(list(live))
        return verify(live, tokens, start_pos, mamba_state, kwargs)

    h.device._verify = record_verify
    h.run()
    assert not any("x" in rows for rows in verified)
    assert len(h.outputs["x"]) == 12


def test_a_host_sampled_verify_returns_the_payload_of_its_greedy_rows():
    """On a launch that samples nothing on the device a greedy logprobs row
    is still certifiable: its verify samples on the host and its read
    returns ``(ids, payload)``, from which the walk publishes the logprobs."""
    h = _AsyncHarness(48, mode="none", asynchronous=True, chained=True)
    h.device.model_capabilities["supports_spec_device_logprobs"] = True
    h.add("g0", 1019, max_tokens=16)
    h.add("l2", 1022, max_tokens=16, logprobs=2)
    published = _collect_logprobs(h)
    h.run()

    assert h.runner.sample_on_device_mode is None
    assert any("l2" in r.rows for r in h.device.records if r.chained)
    from_device = [
        (prefix, token, logprobs[0])
        for prefix, token, _ids, logprobs, _rank in published["l2"]
        if logprobs[0] != 0.0
    ]
    assert len(from_device) >= 10
    for prefix, token, value in from_device:
        assert value == pytest.approx(_device_logprob(prefix, token))


ALLOWED = tuple(range(0, VOCAB, 2))


def _masked_scenario(h: _AsyncHarness) -> None:
    """``_sampled_scenario`` with an allowlist on one greedy and one seeded
    sampled request, which a verify carries in its token mask."""
    _sampled_scenario(h, unseeded=False)
    h.add("ga", 1300, max_tokens=24, allowed_token_ids=list(ALLOWED))
    h.add(
        "sa",
        2500,
        max_tokens=24,
        temperature=0.8,
        top_k=5,
        seed=21,
        allowed_token_ids=list(ALLOWED),
    )


@pytest.mark.parametrize("mode", ["none", "align"])
def test_token_masked_rows_chain_to_their_unchained_output(mode):
    """On a model chaining token-masked verifies, a row with an allowlist no
    longer drains the chain: its chained verify defers its draws to
    sample_tokens, under the mask its predecessor's commit gives it. Greedy
    outputs equal those of unchained token-masked verifies, and every
    allowlisted token is allowed. A seeded row is compared only for its
    allowlist: an ordinary decode that holds an allowlisted row samples on
    the host, and which steps are ordinary decodes depends on chaining."""
    reference = _AsyncHarness(
        SAMPLED_BLOCKS, mode=mode, asynchronous=True, sampled_ids=True, token_masks=True
    )
    _masked_scenario(reference)
    reference.run()

    h = _AsyncHarness(
        SAMPLED_BLOCKS,
        mode=mode,
        asynchronous=True,
        chained=True,
        sampled_ids=True,
        token_masks=True,
    )
    _masked_scenario(h)
    h.run()

    for req_id in ("g0", "g1", "g2", "ga"):
        assert h.outputs[req_id] == reference.outputs[req_id], req_id
    for req_id in ("ga", "sa"):
        assert len(h.outputs[req_id]) == 24
        assert set(h.outputs[req_id]) <= set(ALLOWED), req_id
    controller = h.runner.async_decode
    assert controller._chain_drains["sampling"] == 0
    assert reference.device.masked_verifies and not reference.device.chained_masked
    assert h.device.chained_masked > 0
    assert controller._token_masked_verify_submissions == h.device.masked_verifies


def test_without_chained_masks_a_token_masked_row_drains_the_chain():
    """The capability pair is required: a model that chains but draws no
    token mask keeps an allowlisted row out of every verify."""
    h = _AsyncHarness(
        SAMPLED_BLOCKS, mode="none", asynchronous=True, chained=True, sampled_ids=True
    )
    _masked_scenario(h)
    h.run()
    assert set(h.outputs["ga"]) <= set(ALLOWED)
    assert h.device.masked_verifies == 0
    assert not any("ga" in record.rows for record in h.device.records)
    assert h.device.chained > 0


@pytest.mark.parametrize("mode", ["none", "align"])
def test_a_token_masked_verify_serves_logprobs_rows_and_not_a_masked_one(mode):
    """On a model declaring ``supports_spec_device_mask``,
    ``supports_spec_device_logprobs`` and ``supports_chained_spec_verify``, a
    token-masked verify carries a logprobs row next to an allowlisted one:
    both chain, the verify is sent ``spec_num_logprobs`` (the device checks
    the rows) and the logprobs row's tokens are published with the payload's
    logprobs. A request with an allowlist and logprobs is served by neither
    payload (it would describe the masked logits, where vLLM's raw logprobs
    precede the allowlist): it never reaches a verify, and its tokens stay
    allowlisted."""
    h = _AsyncHarness(
        SAMPLED_BLOCKS,
        mode=mode,
        asynchronous=True,
        chained=True,
        sampled_ids=True,
        device_logprobs=True,
        token_masks=True,
    )
    h.add("ga", 1300, max_tokens=24, allowed_token_ids=list(ALLOWED))
    h.add("l5", 2045, max_tokens=24, logprobs=5)
    h.add("x", 900, max_tokens=6, logprobs=2, allowed_token_ids=list(ALLOWED))
    published = _collect_logprobs(h)
    h.run()

    assert not any("x" in record.rows for record in h.device.records)
    assert set(h.outputs["x"]) <= set(ALLOWED) and len(h.outputs["x"]) == 6
    assert set(h.outputs["ga"]) <= set(ALLOWED)
    together = [r for r in h.device.records if {"ga", "l5"} <= set(r.rows)]
    assert together and any(r.chained for r in together)
    assert h.device.masked_verifies >= len(together)
    from_device = [e for e in published["l5"] if e[3][0] != 0.0]
    assert len(from_device) >= 12
    for prefix, token, ids, logprobs, rank in from_device:
        assert logprobs[0] == pytest.approx(_device_logprob(prefix, token))
        assert ids[1:] == _device_top(prefix)[0][:5]


def test_a_token_masked_verify_carries_penalised_rows_and_not_a_penalised_logprobs_row(
    monkeypatch,
):
    """With ``supports_spec_device_mask``, ``supports_spec_device_penalties``,
    ``supports_spec_device_logprobs`` and chaining declared: an allowlisted
    greedy row with a repetition penalty verifies under the token mask and
    chains, and its deferred verify brings each penalised row's prompt and
    output at submission. A penalised request that asks for logprobs is served
    by no payload (it would describe the penalized logits, where vLLM's raw
    logprobs precede the penalties): it never reaches a verify."""
    monkeypatch.setitem(globals(), "_prompt", _vocab_prompt)
    h = _AsyncHarness(
        SAMPLED_BLOCKS,
        mode="align",
        asynchronous=True,
        chained=True,
        sampled_ids=True,
        device_logprobs=True,
        token_masks=True,
    )
    h.device.model_capabilities["supports_spec_device_penalties"] = True
    h.add("g0", 1019, max_tokens=24)
    h.add(
        "pa",
        1100,
        max_tokens=24,
        repetition_penalty=1.3,
        allowed_token_ids=list(ALLOWED),
    )
    h.add("l5", 900, max_tokens=24, logprobs=5)
    h.add("pl", 700, max_tokens=6, repetition_penalty=1.2, logprobs=3)
    device, deferred = h.device, []
    decode = device.decode_forward

    def record(*, tokens, start_pos, mamba_state, **kwargs):
        live = device._live()
        if kwargs.get("defer_device_sampling") and "pa" in live:
            row = live.index("pa")
            prompt = _history(kwargs["prompt_tokens"], row)
            deferred.append(prompt == list(h.requests["pa"].prompt_token_ids))
        return decode(
            tokens=tokens, start_pos=start_pos, mamba_state=mamba_state, **kwargs
        )

    device.decode_forward = record
    h.run()

    assert not any("pl" in record.rows for record in device.records)
    assert len(h.outputs["pl"]) == 6
    assert set(h.outputs["pa"]) <= set(ALLOWED) and len(h.outputs["pa"]) == 24
    masked = [r for r in device.records if "pa" in r.rows]
    assert masked and any(r.chained for r in masked)
    assert deferred and all(deferred), (
        "a deferred verify lacked a penalised row's prompt"
    )
    assert device.masked_verifies >= len(masked)


@pytest.mark.parametrize("declared", [True, False])
def test_a_top_p_row_with_top_k_off_stays_on_device_verifies_that_never_chain(
    declared,
):
    """On a device declaring ``supports_spec_device_top_p`` a sampled request
    with top-k off and top_p 0.9 is certifiable on a device-sampled step: its
    steps are device-sampled verifies (its ids are draws), although its top-k
    exceeds ``max_device_top_k``. The device may draw its rows again when the
    verify is read, so no step chains on such a verify (reason ``top_p``), no
    prefill runs behind it, and an ordinary decode carrying it samples on the
    host. Without the declaration it samples on the host: no verify, no draw."""
    h = _AsyncHarness(
        SAMPLED_BLOCKS,
        mode="align",
        asynchronous=True,
        chained=True,
        sampled_ids=True,
        prefill_over_pending=True,
        device_top_p=declared,
    )
    h.add("g0", 1019, max_tokens=40)
    h.add("t0", 1022, max_tokens=40, temperature=0.9, top_p=0.9, seed=3)
    device = h.device
    verify, narrow, prefill = device._verify, device._narrow, device.prefill_forward
    verified: list[tuple[bool, bool]] = []
    narrow_sampling: list[bool] = []
    behind: list[list[str]] = []

    def record_verify(live, tokens, start_pos, mamba_state, kwargs):
        if "t0" in live:
            verified.append(("spec_predecessor" in kwargs, "sampling_params" in kwargs))
        return verify(live, tokens, start_pos, mamba_state, kwargs)

    def record_narrow(live, tokens, start_pos, mamba_state, sampling=None):
        if "t0" in live:
            narrow_sampling.append(sampling is not None)
        return narrow(live, tokens, start_pos, mamba_state, sampling)

    def record_prefill(*args, **kwargs):
        for step in h.runner._pending_async_steps:
            unread = isinstance(getattr(step, "verify_output", None), VerifyOutput)
            if unread and not step.is_resolved():
                behind.append(list(step.model_input.row_req_ids))
        return prefill(*args, **kwargs)

    device._verify, device._narrow = record_verify, record_narrow
    device.prefill_forward = record_prefill
    update, updates = h._update, []

    def arrive(scheduler_output, output):
        update(scheduler_output, output)
        updates.append(scheduler_output)
        if len(updates) in (10, 20):
            h.add(f"j{len(updates)}", 700, max_tokens=6)

    h._update = arrive
    h.run()

    assert len(h.outputs["t0"]) == 40
    assert not any(sampling for sampling in narrow_sampling)
    if not declared:
        assert not verified
        return
    assert len(verified) > 5 and all(sampled for _, sampled in verified)
    assert not any(chained for chained, _ in verified)
    assert h.runner.async_decode._chain_drains["top_p"] > 0
    assert not any("t0" in rows for rows in behind)


def test_top_p_rows_with_a_mask_or_a_penalty_verify_on_the_device_and_never_chain(
    monkeypatch,
):
    """A row with top-k off and top_p 0.9 that also carries an allowlist, and
    one that carries a presence penalty, on a model declaring
    ``supports_spec_device_top_p`` with ``supports_spec_device_mask`` and
    ``supports_spec_device_penalties``: their steps are device-sampled
    verifies, token-masked where the allowlist asks for it, and no step chains
    on a verify holding them (reason ``top_p``). The allowlisted row's tokens
    stay allowlisted."""
    monkeypatch.setitem(globals(), "_prompt", _vocab_prompt)
    h = _AsyncHarness(
        SAMPLED_BLOCKS,
        mode="align",
        asynchronous=True,
        chained=True,
        sampled_ids=True,
        token_masks=True,
        device_top_p=True,
    )
    h.device.model_capabilities["supports_spec_device_penalties"] = True
    h.add("g0", 1019, max_tokens=32)
    h.add(
        "tm",
        1100,
        max_tokens=32,
        temperature=0.9,
        top_p=0.9,
        seed=5,
        allowed_token_ids=list(ALLOWED),
    )
    h.add("tp", 900, max_tokens=32, temperature=0.9, top_p=0.9, presence_penalty=0.5)
    device = h.device
    verify = device._verify
    verified: list[tuple[list[str], bool, bool, bool]] = []

    def record_verify(live, tokens, start_pos, mamba_state, kwargs):
        verified.append(
            (
                list(live),
                "spec_predecessor" in kwargs,
                kwargs.get("sampling_params") is not None,
                device.mask is not None,
            )
        )
        return verify(live, tokens, start_pos, mamba_state, kwargs)

    device._verify = record_verify
    h.run()

    holding = [v for v in verified if {"tm", "tp"} & set(v[0])]
    assert len(holding) > 5 and all(sampled for _, _, sampled, _ in holding)
    assert any(masked for rows, _, _, masked in holding if "tm" in rows)
    assert not any(chained for _, chained, _, _ in holding)
    assert h.runner.async_decode._chain_drains["top_p"] > 0
    assert set(h.outputs["tm"]) <= set(ALLOWED) and len(h.outputs["tm"]) == 32
    assert len(h.outputs["tp"]) == 32
