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
  computes it) over the whole slot grid.
- The drafter proposes the rule's argmax chain from each row's last committed
  token, except at a position where ``position % miss_every == miss_at``,
  where it drafts the wrong token, so acceptance varies along a request.
  ``draft_policy="solo"`` drafts only for a lane holding one request, as
  MiMo-V2.6-Flash's drafter does.

``read_decode_output`` hands back host output with no events, so a step
completes only when the runner resolves its deferred output: exactly where the
engine's batch queue would.
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
    miss_every = 5
    miss_at = 2

    def __init__(self) -> None:
        # slot -> (last committed token, its position), what a traced decode
        # keeps on the device between steps.
        self.resident: dict[int, tuple[int, int]] = {}
        self.calls: list[dict] = []
        self.propose_calls: list[dict] = []
        self.engine_thread: int | None = None

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs: int, requested_k: int):
        del vllm_config, max_num_seqs
        return SpecPlan(
            effective_k=requested_k,
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=cls.accept_modes,
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=cls.supports_narrow_decode,
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

    def _lane_rows(self, live: list[bool]) -> dict[int, int]:
        """Live rows per lane."""
        counts: dict[int, int] = {}
        for row, alive in enumerate(live):
            if alive:
                lane = row // self.per_lane
                counts[lane] = counts.get(lane, 0) + 1
        return counts

    def _drafts_for(self, live: list[bool], last, position, num_drafts: int):
        rows = len(live)
        lanes = self._lane_rows(live)
        drafts = torch.full((rows, num_drafts), -1, dtype=torch.int32)
        offered = torch.zeros(rows, dtype=torch.int32)
        for row in range(rows):
            if not live[row]:
                continue
            if self.draft_policy == "solo" and lanes[row // self.per_lane] != 1:
                continue
            token, pos = int(last[row]), int(position[row])
            for j in range(num_drafts):
                token = drafter_token(token, pos, self.miss_every, self.miss_at)
                pos += 1
                drafts[row, j] = token
            offered[row] = num_drafts
        return drafts, offered

    # ---- the contract ----------------------------------------------------

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
        }
        self.calls.append(call)
        if spec_mode is None:
            return self._plain(tokens, start_pos, kwargs)
        width = int(tokens.shape[1])
        check_spec_side_tensors(
            kwargs["num_valid_drafts"], kwargs["accepted_counts"], rows, width - 1
        )
        logits = rule_logits(tokens, start_pos)
        if spec_mode == ACCEPT_MODE_LOGITS:
            self.resident = {}
            return VerifyOutput(spec_mode=spec_mode, logits=logits)
        if spec_mode == ACCEPT_MODE_ARGMAX_IDS:
            self.resident = {}
            return VerifyOutput(
                spec_mode=spec_mode,
                argmax_ids=logits.argmax(dim=-1).to(torch.int32),
            )
        if spec_mode != ACCEPT_MODE_FUSED_SAMPLE:
            raise ValueError(f"LaneDeviceTarget does not serve {spec_mode!r}")
        committed, counts = self._accept(
            tokens, start_pos, kwargs["num_valid_drafts"], kwargs.get("spec_sampling")
        )
        self._settle(committed, counts, start_pos[:, 0])
        return VerifyOutput(
            spec_mode=spec_mode, accepted_token_ids=committed, accepted_counts=counts
        )

    def _plain(self, tokens, start_pos, kwargs):
        if "sampling_params" not in kwargs:
            raise AssertionError("an ordinary lane decode samples on the device here")
        rows = int(tokens.shape[0])
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
        index = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        last = committed_tokens.to(torch.int64).gather(1, index).squeeze(1)
        position = committed_positions.to(torch.int64).gather(1, index).squeeze(1)
        live = [int(position[row]) >= 0 for row in range(rows)]
        self.propose_calls.append(
            {
                "rows": rows,
                "hidden": hidden,
                "positions": committed_positions.clone(),
                "live": live,
            }
        )
        drafts, offered = self._drafts_for(live, last, position, num_drafts)
        return DraftOutput(draft_token_ids=drafts, num_valid=offered)


def lane_device_target(**knobs) -> type[LaneDeviceTarget]:
    """A configured subclass, so one test cannot configure the next."""
    return type("ConfiguredLaneDeviceTarget", (LaneDeviceTarget,), dict(knobs))
