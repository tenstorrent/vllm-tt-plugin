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
from vllm.v1.request import Request

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
    PLACEHOLDER_TOKEN_ID,
    DraftOutput,
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


class _Answer:
    """A verify's deferred ``argmax_ids``: the ids, and how often the host read them."""

    def __init__(self, ids: torch.Tensor, record: SimpleNamespace):
        self.ids = ids
        self.record = record
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
                self.truth[req_id] = [*prefix, _target(prefix)]
                logits[row, 0, _target(prefix)] = 1e4
        self._run_rows(mamba_state, rows)
        return logits

    def decode_forward(self, *, tokens, start_pos, mamba_state, **kwargs):
        if kwargs.get("reload_inputs") is not True:
            raise AssertionError(
                f"reload_inputs={kwargs.get('reload_inputs')}: every decode "
                "of this model reloads, its ordinary decode being host-answered"
            )
        live = self._live()
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

    def _draws(self, row: int, sampling) -> bool:
        """Whether ``row``'s ids are draws: a sampled row of a device-sampled
        step (``sampling``) on a device declaring ``spec_ids_are_sampled``."""
        return (
            sampling is not None
            and bool(self.model_capabilities.get("spec_ids_are_sampled"))
            and float(sampling.temperature[row]) > 0
            and int(sampling.top_k[row]) != 1
        )

    def _token(self, prefix, row: int, position: int, sampling) -> int:
        """The device's token after ``prefix`` at input ``position``: the
        target's, or a drawing row's draw, seeded by the request seed and the
        position or by fresh entropy."""
        if not self._draws(row, sampling):
            return _target(prefix)
        seed = sampling.seed[row]
        if seed is None:
            return _draw(prefix, self.entropy.randrange(1 << 20))
        return _draw(prefix, _position_seed(seed, position))

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
            truth.append(self._token(truth, row, position, sampling))
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
        )
        self.records.append(record)
        for row, req_id in enumerate(live):
            params = self.h.requests[req_id].sampling_params
            if params.temperature > 0 and not record.sampled[row]:
                raise AssertionError(
                    f"{req_id} samples and its argmax_ids verify answers the argmax"
                )
        if predecessor is not None:
            self.chained += 1
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
                # The row continues the unread predecessor from the count the
                # device accepted; the runner sent placeholders.
                assert predecessor.rows[row] == req_id, (predecessor.rows, live)
                count = predecessor.counts[row]
                previous = predecessor.p0[row]
                assert tokens[row].tolist() == [PLACEHOLDER_TOKEN_ID] * T
                assert start_pos[row].tolist() == [previous + T + j for j in range(T)]
                assert int(kwargs["accepted_counts"][row]) == 1
                p0 = previous + count
                nv = min(int(num_valid[row]), predecessor.offered[row])
                drafts = predecessor.next_drafts[row]
                block = [int(predecessor.ids[row, count - 1]), *drafts[:nv]]
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
            spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=_Answer(ids, record)
        )

    def _verify_row(self, row, req_id, p0, block, starts, ids, record, rows, sampling):
        truth = self.truth[req_id]
        prefix = truth[:p0]
        nv = len(block) - 1
        for t in range(T):
            ids[row, t] = (
                self._token(prefix + block[: t + 1], row, p0 + t, sampling)
                if t <= nv
                else PLACEHOLDER_TOKEN_ID
            )
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
    returns ahead of them), finished rows that are not the last and
    preemptions drain instead."""
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
