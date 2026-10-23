# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A lane-DP target that keeps device-resident state, for the async lane path.

``SampledTarget``'s causal rule (``rule_logits``) is the model: its argmax is
what a greedy request emits and its distribution is what a sampled one draws
from, whatever was drafted. On top of it this stand-in behaves like a traced
lane decode on a device, which is what the asynchronous lane path relies on:

- An ordinary decode samples on the device and, when the runner sends
  ``reload_inputs=False``, reads its own last token and position per slot
  instead of the host's inputs, which may be a step behind. A sampled row draws
  from a stream keyed by its seed (or slot) and absolute position, so the token
  does not depend on when the host read it.
- A verify answers ``argmax_ids``, ``logits`` or ``fused_sample``
  (``accept_speculated_tokens`` with the runner's uniforms, as a device accept
  computes it) over the whole slot grid. After a ``fused_sample`` verify the
  device drafts for the next step itself, from the tokens it just accepted.
- The chain contract (SPEC_DECODE_CONTRACT.md section 4j): a fused verify sent
  ``defer_read=True`` returns a ``LaneFusedHandle`` and no host tensors; a
  verify sent ``chain_rows`` builds those rows from the device's own last
  token, position and drafts, and raises on every input the contract forbids.
- The drafter proposes the rule's argmax chain from each row's last committed
  token, except at a position where ``position % miss_every == miss_at``,
  where it drafts the wrong token, so acceptance varies along a request.
  ``draft_policy="solo"`` drafts only for a lane holding one request, as
  MiMo-V2.6-Flash's drafter did before it drafted for two; ``"lane_load"``
  drafts by the lane's load as MiMo's does now: the full K for a request
  alone, one draft each for two, none for three or more.
- ``lanes_per_request`` and ``k_by_rows`` go into the plan. A verify records
  ``lane_rows``, each lane's ``sum(1 + num_valid_drafts)`` over its live
  rows, which is how many decode rows a model laying one candidate token per
  row (MiMo, ``lanes_per_request == 1 + K``) fills, and so how many passes of
  its ``per_lane`` rows the verify takes (``passes``).

``read_decode_output`` hands back host output with no events, so a step
completes only when the runner resolves its deferred output: exactly where the
engine's batch queue would. A handle can be held (``hold_reads``) until the
test releases it.
"""

from __future__ import annotations

import threading

import torch

from vllm_tt_plugin.spec_accept import accept_speculated_tokens
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_FUSED_SAMPLE,
    ACCEPT_MODE_LOGITS,
    DRAFTER_STATE_INTERNAL,
    SPEC_REQUIREMENT_DEVICE_PROPOSE,
    DraftOutput,
    FusedReadout,
    SpecPlan,
    VerifyOutput,
    check_spec_side_tensors,
)

from .sampled_target import SAMPLED_VOCAB_SIZE, base_token, rule_logits


def drafter_token(token: int, position: int, miss_every: int, miss_at: int) -> int:
    """What the drafter proposes after ``token`` at ``position``."""
    choice = base_token(token, position)
    if miss_every and position % miss_every == miss_at:
        choice = (choice + 1) % SAMPLED_VOCAB_SIZE
    return choice


def greedy_chain(token: int, position: int, count: int) -> list[int]:
    """The rule's argmax continuation: what a greedy request emits."""
    chain = []
    for _ in range(count):
        token = base_token(token, position)
        position += 1
        chain.append(token)
    return chain


class LaneFusedHandle:
    """What a deferred fused verify returns: readable only once waited on."""

    def __init__(self, tip_id: int, readout: FusedReadout, drafts, offered, *, hold):
        self.tip_id = tip_id
        self.readout = readout
        self.drafts = drafts
        self.offered = offered
        self._ready = threading.Event()
        if not hold:
            self._ready.set()
        self.read_threads: list[int] = []
        self.proposed = False

    def release(self) -> None:
        self._ready.set()

    def wait(self) -> FusedReadout:
        assert self._ready.wait(timeout=30), "a held fused verify was never released"
        self.read_threads.append(threading.get_ident())
        return self.readout


class LaneDeviceTarget:
    vocab_size = SAMPLED_VOCAB_SIZE
    accept_modes: tuple[str, ...] = (
        ACCEPT_MODE_ARGMAX_IDS,
        ACCEPT_MODE_LOGITS,
        ACCEPT_MODE_FUSED_SAMPLE,
    )
    supports_narrow_decode = True
    decode_input_update_contract = 1
    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": [SPEC_REQUIREMENT_DEVICE_PROPOSE],
        "supports_async_decode": True,
        "supports_async_spec_decode": True,
        "max_device_top_k": 8,
        "supports_device_top_p_without_top_k": True,
    }
    # Lanes are ``per_lane`` consecutive rows; set by the harness.
    per_lane = 2
    draft_policy = "always"
    lanes_per_request = 1
    k_by_rows: tuple[tuple[int, int], ...] = ()
    miss_every = 5
    miss_at = 2
    # Hold every deferred read until the test releases its handle.
    hold_reads = False
    # Report this offset on every chained row's p0, to test the runner's check.
    misreport_p0 = 0
    # False: answer defer_read with host tensors, as a model that does not
    # chain yet may.
    honor_defer_read = True

    def __init__(self) -> None:
        # slot -> (last committed token, its position), what a traced decode
        # keeps on the device between steps.
        self.resident: dict[int, tuple[int, int]] = {}
        # slot -> (drafts, count) the device drafted after the last fused verify.
        self.device_drafts: dict[int, tuple[list[int], int]] = {}
        self.tip: int | None = None
        self.next_tip = 0
        self.handles: list[LaneFusedHandle] = []
        self.calls: list[dict] = []
        self.propose_calls: list[dict] = []
        self.engine_thread: int | None = None
        self.freed: list[int] = []
        self.unresolved_at_prefill: list[int] = []
        self.prefill_spans: list[tuple[int, int, int]] = []

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs: int, requested_k: int):
        del vllm_config, max_num_seqs
        return SpecPlan(
            effective_k=requested_k,
            lanes_per_request=cls.lanes_per_request,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=cls.accept_modes,
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=cls.supports_narrow_decode,
            k_by_rows=cls.k_by_rows,
        )

    # ---- helpers ---------------------------------------------------------

    def _on_engine_thread(self) -> None:
        thread = threading.get_ident()
        if self.engine_thread is None:
            self.engine_thread = thread
        elif self.engine_thread != thread:
            raise AssertionError("the device is driven from one (the engine) thread")

    @property
    def verify_modes(self) -> list[str]:
        return [call["spec_mode"] for call in self.calls if call["spec_mode"]]

    @property
    def plain_steps(self) -> int:
        return sum(1 for call in self.calls if call["spec_mode"] is None)

    @property
    def verify_calls(self) -> list[dict]:
        return [call for call in self.calls if call["spec_mode"] is not None]

    def passes(self, call: dict) -> dict[int, int]:
        """Per lane, the passes of ``per_lane`` rows a packing verify takes."""
        return {
            lane: -(-rows // self.per_lane) for lane, rows in call["lane_rows"].items()
        }

    def _lane_rows(self, start_pos, num_valid) -> dict[int, int]:
        rows: dict[int, int] = {}
        for row in range(int(start_pos.shape[0])):
            if int(start_pos[row, 0]) >= 0:
                lane = row // self.per_lane
                rows[lane] = rows.get(lane, 0) + 1 + int(num_valid[row])
        return rows

    @property
    def chained_calls(self) -> list[dict]:
        return [call for call in self.calls if call.get("chain_rows") is not None]

    def _device_sample(self, params, row: int, token: int, position: int) -> int:
        """One device-sampled token: argmax for a greedy row, else a draw from
        a stream keyed by the row's seed (or slot) and the absolute position."""
        temperature = float(params.temperature[row])
        if temperature <= 0.0:
            return base_token(token, position)
        seed = params.seed[row] if params.seed is not None else None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(((seed if seed is not None else row) << 20) + position)
        logits = rule_logits(torch.tensor([token]), torch.tensor([position]))[0]
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator))

    def _drafts_for(self, live: list[bool], last, position, num_drafts: int):
        rows = len(live)
        lanes: dict[int, int] = {}
        for row, alive in enumerate(live):
            if alive:
                lanes[row // self.per_lane] = lanes.get(row // self.per_lane, 0) + 1
        drafts = torch.full((rows, num_drafts), -1, dtype=torch.int32)
        offered = torch.zeros(rows, dtype=torch.int32)
        for row in range(rows):
            if not live[row]:
                continue
            load = lanes[row // self.per_lane]
            if self.draft_policy == "solo" and load != 1:
                continue
            count = num_drafts
            if self.draft_policy == "lane_load":
                count = {1: num_drafts, 2: 1}.get(load, 0)
            if not count:
                continue
            token, pos = int(last[row]), int(position[row])
            for j in range(count):
                token = drafter_token(token, pos, self.miss_every, self.miss_at)
                pos += 1
                drafts[row, j] = token
            offered[row] = count
        return drafts, offered

    def _invalidate_chain(self) -> None:
        """A step that is not a deferred fused verify: nothing may chain on."""
        self.tip = None
        self.device_drafts = {}

    # ---- the contract ----------------------------------------------------

    def prefill_forward(
        self, tokens, page_table, kv_cache, prompt_lens, start_pos, **kwargs
    ):
        """One prefill per row into its slot (``empty_slots``): the first
        token is sampled on the device, or its logits returned for the host to
        sample, and the slot's state starts over.

        A row may start past 0, where the prefix cache already holds the prompt
        head. ``prefill_spans`` records each row's ``(slot, start, end)``, and
        ``unresolved_at_prefill`` how many deferred verifies were still unread
        when the prefill was submitted.
        """
        self._on_engine_thread()
        del page_table, kv_cache
        self.unresolved_at_prefill.append(
            sum(1 for handle in self.handles if not handle.read_threads)
        )
        params = kwargs.get("sampling_params")
        rows = int(tokens.shape[0])
        if params is None:
            out = torch.zeros(rows, self.vocab_size)
        else:
            out = torch.zeros(rows, dtype=torch.int32)
        for i, slot in enumerate(kwargs["empty_slots"]):
            start, length = int(start_pos[i]), int(prompt_lens[i])
            if not 0 <= start < length:
                raise AssertionError(f"row {i} prefills [{start}, {length})")
            self.prefill_spans.append((slot, start, length))
            last = int(tokens[i, length - 1])
            self.device_drafts.pop(slot, None)
            if params is None:
                position = torch.tensor([length - 1])
                out[i] = rule_logits(torch.tensor([last]), position)[0]
                # The host samples the first token; the device never sees it.
                self.resident.pop(slot, None)
                continue
            first = self._device_sample(params, i, last, length - 1)
            out[i] = first
            self.resident[slot] = (first, length)
        return out

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        self._on_engine_thread()
        rows = int(tokens.shape[0])
        call = {
            "spec_mode": spec_mode,
            "rows": rows,
            "tokens": tokens.clone(),
            "positions": start_pos.clone(),
            "reload_inputs": kwargs.get("reload_inputs"),
            "num_valid_drafts": kwargs.get("num_valid_drafts"),
            "accepted_counts": kwargs.get("accepted_counts"),
            "spec_sampling": kwargs.get("spec_sampling"),
            "defer_read": kwargs.get("defer_read", False),
            "chain_rows": kwargs.get("chain_rows"),
            "chain_tip": kwargs.get("chain_tip"),
            "finishing_slots": kwargs.get("finishing_slots"),
        }
        self.calls.append(call)
        if call["finishing_slots"] is not None:
            if not self.model_capabilities.get("supports_finishing_slots", False):
                raise ValueError("finishing_slots without supports_finishing_slots")
            for slot in call["finishing_slots"]:
                # Freed like a finished request: nothing chains on it.
                self.resident.pop(slot, None)
                self.device_drafts.pop(slot, None)
                self.freed.append(slot)
        if spec_mode is None:
            self._invalidate_chain()
            return self._plain(tokens, start_pos, kwargs)
        call["lane_rows"] = self._lane_rows(start_pos, kwargs["num_valid_drafts"])
        if spec_mode not in self.accept_modes:
            raise ValueError(f"LaneDeviceTarget does not serve {spec_mode!r}")
        if spec_mode != ACCEPT_MODE_FUSED_SAMPLE:
            if call["defer_read"] or call["chain_rows"] is not None:
                raise ValueError("defer_read and chain_rows come with fused_sample")
            check_spec_side_tensors(
                kwargs["num_valid_drafts"],
                kwargs["accepted_counts"],
                rows,
                int(tokens.shape[1]) - 1,
            )
            self._invalidate_chain()
            self.resident = {}
            logits = rule_logits(tokens, start_pos)
            if spec_mode == ACCEPT_MODE_LOGITS:
                return VerifyOutput(spec_mode=spec_mode, logits=logits)
            return VerifyOutput(
                spec_mode=spec_mode,
                argmax_ids=logits.argmax(dim=-1).to(torch.int32),
            )
        return self._fused(tokens, start_pos, kwargs, call)

    def _plain(self, tokens, start_pos, kwargs):
        rows = int(tokens.shape[0])
        if "sampling_params" not in kwargs:
            # Host sampling: the rule's logits for the host's inputs, which a
            # host-sampled step always reloads.
            self.resident = {}
            positions = start_pos.reshape(rows, -1)[:, :1]
            return rule_logits(tokens.reshape(rows, -1)[:, :1], positions)
        params = kwargs["sampling_params"]
        if kwargs.get("reload_inputs", True):
            inputs = {
                row: (int(tokens[row, 0]), int(start_pos[row]))
                for row in range(rows)
                if int(start_pos[row]) >= 0
            }
        else:
            inputs = dict(self.resident)
            # The host may be a step behind, but never ahead, and never names
            # a row the device holds nothing for.
            for row in range(rows):
                host = int(start_pos[row])
                if host >= 0 and (row not in inputs or host > inputs[row][1]):
                    raise AssertionError(
                        f"row {row}: resident decode with host position {host} "
                        f"ahead of the device's {inputs.get(row)}"
                    )
        out = torch.zeros(rows, dtype=torch.int32)
        resident = {}
        for row, (token, position) in inputs.items():
            sampled = self._device_sample(params, row, token, position)
            out[row] = sampled
            resident[row] = (sampled, position + 1)
        self.resident = resident
        return out

    def _chain_inputs(self, tokens, start_pos, num_valid, counts, call):
        """The block the device verifies: chained rows from its own state."""
        chain_rows = call["chain_rows"]
        rows, width = (int(size) for size in tokens.shape)
        tokens, positions = tokens.clone(), start_pos.clone()
        counts = counts.clone()
        if chain_rows is None:
            if call["chain_tip"] is not None:
                raise ValueError("chain_tip without chain_rows")
            return tokens, positions, counts
        if not call["defer_read"]:
            raise ValueError("chain_rows needs defer_read=True")
        if chain_rows.dtype != torch.bool or tuple(chain_rows.shape) != (rows,):
            raise ValueError(f"chain_rows must be bool [{rows}], got {chain_rows}")
        if self.tip is None or call["chain_tip"] != self.tip:
            raise ValueError(
                f"chain_tip {call['chain_tip']} is not the last deferred fused "
                f"verify {self.tip}"
            )
        for row in range(rows):
            if not bool(chain_rows[row]):
                continue
            if row not in self.resident or row not in self.device_drafts:
                raise ValueError(f"chain row {row} has no device state")
            if not bool((tokens[row] == -1).all()):
                raise ValueError(f"chain row {row} carries host tokens")
            if int(counts[row]) != -1:
                raise ValueError(f"chain row {row} must carry accepted_counts -1")
            last, position = self.resident[row]
            if int(start_pos[row, 0]) < position:
                raise ValueError(
                    f"chain row {row}: provisional position {int(start_pos[row, 0])} "
                    f"below the device's {position}"
                )
            drafts, offered = self.device_drafts[row]
            hint = int(num_valid[row])
            if hint > offered:
                raise ValueError(
                    f"chain row {row} asks to verify {hint} drafts and the "
                    f"device drafted {offered}"
                )
            tokens[row] = -1
            tokens[row, 0] = last
            tokens[row, 1 : 1 + hint] = torch.tensor(drafts[:hint], dtype=torch.int32)
            positions[row] = -1
            positions[row, : 1 + hint] = torch.arange(position, position + 1 + hint)
            counts[row] = 1
        return tokens, positions, counts

    def _fused(self, tokens, start_pos, kwargs, call):
        rows, width = (int(size) for size in tokens.shape)
        num_valid = kwargs["num_valid_drafts"]
        tokens, positions, counts = self._chain_inputs(
            tokens, start_pos, num_valid, kwargs["accepted_counts"], call
        )
        check_spec_side_tensors(num_valid, counts, rows, width - 1)
        call["device_tokens"] = tokens.clone()
        call["device_positions"] = positions.clone()
        committed, accepted = self._accept(
            tokens, positions, num_valid, kwargs.get("spec_sampling")
        )
        call["committed"], call["accepted"] = committed, accepted
        first = positions[:, 0].to(torch.int32)
        self._settle(committed, accepted, first)
        live = [int(first[row]) >= 0 for row in range(rows)]
        last = torch.tensor(
            [int(committed[row, int(accepted[row]) - 1]) for row in range(rows)]
        )
        drafts, offered = self._drafts_for(live, last, first + accepted, width - 1)
        self.device_drafts = {
            row: (drafts[row].tolist(), int(offered[row]))
            for row in range(rows)
            if live[row]
        }
        if not call["defer_read"] or not self.honor_defer_read:
            self.tip = None
            return VerifyOutput(
                spec_mode=ACCEPT_MODE_FUSED_SAMPLE,
                accepted_token_ids=committed,
                accepted_counts=accepted,
            )
        reported = first.clone()
        if call["chain_rows"] is not None and self.misreport_p0:
            reported[call["chain_rows"]] += self.misreport_p0
        self.next_tip += 1
        self.tip = self.next_tip
        handle = LaneFusedHandle(
            self.tip,
            FusedReadout(committed.to(torch.int32), accepted.to(torch.int32), reported),
            drafts,
            offered,
            hold=self.hold_reads,
        )
        self.handles.append(handle)
        return VerifyOutput(spec_mode=ACCEPT_MODE_FUSED_SAMPLE, handle=handle)

    def _accept(self, tokens, start_pos, num_valid, sampling):
        rows, width = (int(size) for size in tokens.shape)
        if sampling is None:
            temperature, top_k, top_p = torch.zeros(rows), None, None
            uniforms = torch.zeros(rows, width + 1, dtype=torch.float64)
        else:
            temperature = sampling["temperature"]
            top_k = sampling["top_k"].masked_fill(
                sampling["top_k"] < 1, SAMPLED_VOCAB_SIZE
            )
            top_p = sampling["top_p"]
            uniforms = sampling["uniforms"].to(torch.float64)
        result = accept_speculated_tokens(
            target_logits=rule_logits(tokens, start_pos),
            draft_token_ids=tokens[:, 1:],
            num_valid_drafts=num_valid,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            uniforms=uniforms,
        )
        return result.committed_token_ids, result.accepted_counts

    def _settle(self, committed, counts, first) -> None:
        """Keep each live row's last committed token and its position."""
        self.resident = {
            row: (
                int(committed[row, int(counts[row]) - 1]),
                int(first[row]) + int(counts[row]),
            )
            for row in range(int(committed.shape[0]))
            if int(first[row]) >= 0
        }

    def read_decode_output(self, tt_out, async_read: bool = False, **kwargs):
        del kwargs
        if isinstance(tt_out, LaneFusedHandle):
            if async_read:
                raise AssertionError("a handle is read with read_decode_output(handle)")
            return tt_out.wait()
        if not async_read:
            raise AssertionError("only the asynchronous read is expected here")
        return tt_out, []

    def propose_draft_tokens(
        self,
        num_drafts,
        committed_tokens,
        committed_positions,
        accepted_counts,
        hidden=None,
    ):
        self._on_engine_thread()
        rows = int(committed_tokens.shape[0])
        check_spec_side_tensors(
            torch.zeros(rows, dtype=torch.int32),
            accepted_counts,
            rows,
            num_drafts,
            call="propose",
        )
        self.propose_calls.append(
            {
                "rows": rows,
                "hidden": hidden,
                "positions": committed_positions.clone(),
            }
        )
        if isinstance(hidden, LaneFusedHandle):
            # The device drafted already; this reads the drafts back once, on
            # the engine thread, after the step was read.
            if not hidden.read_threads:
                raise AssertionError("a deferred verify was proposed before its read")
            if hidden.proposed:
                raise AssertionError("a deferred verify is proposed once")
            hidden.proposed = True
            p0 = hidden.readout.p0
            sent = committed_positions[:, 0]
            for row in range(rows):
                if int(sent[row]) >= 0 and int(sent[row]) != int(p0[row]) + 1:
                    raise AssertionError(
                        f"row {row} proposed from position {int(sent[row])}, "
                        f"the device committed from {int(p0[row]) + 1}"
                    )
            return DraftOutput(draft_token_ids=hidden.drafts, num_valid=hidden.offered)
        index = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        last = committed_tokens.to(torch.int64).gather(1, index).squeeze(1)
        position = committed_positions.to(torch.int64).gather(1, index).squeeze(1)
        live = [int(position[row]) >= 0 for row in range(rows)]
        drafts, offered = self._drafts_for(live, last, position, num_drafts)
        return DraftOutput(draft_token_ids=drafts, num_valid=offered)


def lane_device_target(**knobs) -> type[LaneDeviceTarget]:
    """A configured subclass, so one test cannot configure the next."""
    return type("ConfiguredLaneDeviceTarget", (LaneDeviceTarget,), dict(knobs))
