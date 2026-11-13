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
"""

from __future__ import annotations

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


def _target(prefix: list[int] | tuple[int, ...]) -> int:
    """The target model's greedy token after ``prefix``."""
    tail = tuple(prefix[-3:])
    return FIRST_ID + (
        sum((i + 1) * t for i, t in enumerate(tail)) + 7 * len(prefix)
    ) % (VOCAB - FIRST_ID)


def _drafts(prefix: list[int]) -> list[int]:
    """K drafts after ``prefix``: the first ``j`` are the target's own chain,
    the rest differ from it; ``j`` cycles with the position."""
    agree = len(prefix) % T
    chain = list(prefix)
    drafts = []
    for k in range(K):
        token = _target(chain)
        if k >= agree:
            token = FIRST_ID + (token - FIRST_ID + 1) % (VOCAB - FIRST_ID)
        drafts.append(token)
        chain.append(token)
    return drafts


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
    runner committed what the device accepted.
    """

    decode_input_update_contract = 1

    def __init__(self, harness: _AsyncHarness, chained: bool):
        self.h = harness
        self.model_capabilities = {
            "supports_async_decode": True,
            "supports_async_spec_decode": True,
        }
        self.calls: list[tuple[str, dict]] = []
        self.truth: dict[str, list[int]] = {}
        self.next_drafts: dict[str, list[int]] = {}
        # Each submitted decode's row -> request map, popped by its propose.
        self.steps: deque[SimpleNamespace] = deque()
        self.verifies = 0
        self.accepted: dict[tuple[int, int], int] = {}
        self.boundary_counts: set[int] = set()

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
                logits[row, 0, _target(prefix)] = 1.0
        self._run_rows(mamba_state, rows)
        return logits

    def decode_forward(self, *, tokens, start_pos, mamba_state, **kwargs):
        if kwargs.get("reload_inputs") is not True:
            raise AssertionError(
                f"reload_inputs={kwargs.get('reload_inputs')}: every decode "
                "of this model reloads, its ordinary decode being host-answered"
            )
        live = self._live()
        if kwargs.get("spec_mode") is None:
            return self._narrow(live, tokens, start_pos, mamba_state)
        return self._verify(live, tokens, start_pos, mamba_state, kwargs)

    def _narrow(self, live, tokens, start_pos, mamba_state):
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
            truth.append(_target(truth))
            logits[row, 0, truth[-1]] = 1.0
            self.next_drafts[req_id] = _drafts(truth)
        self._run_rows(mamba_state, rows)
        self.steps.append(SimpleNamespace(rows=list(live), counts=[1] * len(live)))
        return logits

    def _verify(self, live, tokens, start_pos, mamba_state, kwargs):
        num_valid = kwargs["num_valid_drafts"]
        rows_out = int(tokens.shape[0])
        ids = torch.full((rows_out, T), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        record = SimpleNamespace(rows=list(live), counts=[], ids=ids, p0=[])
        rows = []
        for row, req_id in enumerate(live):
            truth = self.truth[req_id]
            p0 = int(start_pos[row, 0])
            nv = int(num_valid[row])
            block = tokens[row, : nv + 1].tolist()
            starts = [int(g.start_blocks[row]) for g in mamba_state.groups]
            assert p0 == len(truth) - 1 and block[0] == truth[-1], (req_id, p0)
            assert block[1:] == self.next_drafts.get(req_id, [])[:nv], (req_id, block)
            self._verify_row(row, req_id, p0, block, starts, ids, record, rows)
        self._run_rows(mamba_state, rows)
        self.steps.append(record)
        return VerifyOutput(
            spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=_Answer(ids, record)
        )

    def _verify_row(self, row, req_id, p0, block, starts, ids, record, rows):
        truth = self.truth[req_id]
        prefix = truth[:p0]
        nv = len(block) - 1
        for t in range(T):
            ids[row, t] = (
                _target(prefix + block[: t + 1]) if t <= nv else PLACEHOLDER_TOKEN_ID
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
        record.counts.append(count)
        record.p0.append(p0)
        truth[p0 + 1 :] = committed
        self.next_drafts[req_id] = _drafts(truth)

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
            assert count == step.counts[row], (req_id, count, step.counts[row])
            truth = self.truth[req_id]
            first = int(positions[row, 0])
            assert committed[row, :count].tolist() == truth[first : first + count], (
                req_id
            )
            drafts[row] = torch.tensor(self.next_drafts[req_id], dtype=torch.int32)
            num_valid[row] = num_drafts
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
            trace_mode="none",
            enable_model_warmup=False,
            num_devices=1,
        )
        self.runner.model = self.device
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

    def add(self, req_id: str, prompt_len: int, max_tokens: int) -> None:
        init_none_hash(sha256)
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0.0)
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
        for block in self.scheduler.kv_cache_manager.block_pool.blocks:
            key = block.block_hash
            if key is None or get_group_id(key) not in self.groups:
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
