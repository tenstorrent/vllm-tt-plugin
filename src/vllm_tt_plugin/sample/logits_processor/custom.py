# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Custom logits processors on the host sampler, through vLLM's V2 interface.

``--logits-processors`` classes and installed ``vllm.logits_processors`` entry
points must implement ``vllm.v1.worker.gpu.sample.logits_processor``'s
``LogitsProcessor``. That interface keys a processor's per-request state by a
request slot that stays fixed for the request's lifetime, and shares a
slot-indexed token history, ``LogitsProcRequestState``. Persistent-batch rows
are not fixed: condense and lane moves relocate requests. So
``CustomLogitsProcessors`` gives every request a slot of its own, follows the
request's row through each ``BatchUpdate``, and keeps the slot-indexed history.
"""

from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    build_custom_logits_processors,
)
from vllm.v1.worker.gpu.sample.logits_processor.loader import LOGITSPROCS_GROUP

from vllm_tt_plugin.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.sample.logits_processor import (
        LogitsProcessor as V2LogitsProcessor,
    )


class _HostTensor:
    """One host tensor under each name V2's buffer types expose.

    Processors read ``LogitsProcRequestState`` fields as ``.gpu``, ``.cpu`` or
    ``.np``. The host sampler runs on CPU, so all three share one storage.
    """

    def __init__(self, *size: int, dtype: torch.dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.gpu = self.cpu
        self.np = self.cpu.numpy()


class _HostRequestStates:
    """The fields ``LogitsProcRequestState.from_request_state`` reads."""

    def __init__(self, max_num_reqs: int, max_model_len: int, vocab_size: int):
        self.device = torch.device("cpu")
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size
        self.all_token_ids = _HostTensor(max_num_reqs, max_model_len, dtype=torch.int32)
        self.prompt_len = _HostTensor(max_num_reqs, dtype=torch.int32)
        self.prefill_len = _HostTensor(max_num_reqs, dtype=torch.int32)
        self.total_len = _HostTensor(max_num_reqs, dtype=torch.int32)


class CustomLogitsProcessors(LogitsProcessor):
    """Runs V2-interface processors on logits whose row ``i`` is batch row ``i``.

    A row that holds no request (a lane slot gap) is not shown to the
    processors. A row that is mid-prompt this step is shown at the request's
    last committed position, not at the end of its scheduled chunk; its token
    is discarded.
    """

    def __init__(
        self,
        processors: Sequence["V2LogitsProcessor"],
        req_states: _HostRequestStates,
    ):
        self.processors = list(processors)
        self.req_states = req_states
        max_num_reqs = req_states.max_num_reqs
        # pop() hands out the lowest free slot first.
        self._free_slots = list(range(max_num_reqs - 1, -1, -1))
        self._row_slot: dict[int, int] = {}
        # slot -> the request's live output list, and how much of it is
        # already copied into all_token_ids.
        self._output_token_ids: dict[int, list[int]] = {}
        self._num_synced = np.zeros(max_num_reqs, dtype=np.int64)
        self._needs_processing = np.zeros(max_num_reqs, dtype=bool)

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for row in batch_update.removed:
            self._release_row(row)
        for row, params, prompt_token_ids, output_token_ids in batch_update.added:
            # The persistent batch drops a row from ``removed`` when a new
            # request reuses it in the same step, so the previous occupant's
            # slot is released here.
            self._release_row(row)
            self._add(row, params, prompt_token_ids or [], output_token_ids)
        for src, dst, direction in batch_update.moved:
            src_slot = self._row_slot.pop(src, None)
            if direction == MoveDirectionality.SWAP:
                dst_slot = self._row_slot.pop(dst, None)
                if dst_slot is not None:
                    self._row_slot[src] = dst_slot
            else:
                self._release_row(dst)
            if src_slot is not None:
                self._row_slot[dst] = src_slot
        if batch_update.added:
            for processor in self.processors:
                processor.apply_staged_writes()

    def _add(
        self,
        row: int,
        params: SamplingParams,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
    ) -> None:
        if not self._free_slots:
            raise RuntimeError(
                f"No free custom logits processor slot for batch row {row}: "
                f"all {self.req_states.max_num_reqs} slots are held by rows "
                f"{sorted(self._row_slot)}"
            )
        slot = self._free_slots.pop()
        num_prompt = len(prompt_token_ids)
        num_output = len(output_token_ids)
        states = self.req_states
        states.all_token_ids.np[slot, :num_prompt] = prompt_token_ids
        states.all_token_ids.np[slot, num_prompt : num_prompt + num_output] = (
            output_token_ids
        )
        states.prompt_len.np[slot] = num_prompt
        states.prefill_len.np[slot] = num_prompt + num_output
        states.total_len.np[slot] = num_prompt + num_output
        self._output_token_ids[slot] = output_token_ids
        self._num_synced[slot] = num_output
        needs_processing = False
        for processor in self.processors:
            needs_processing |= bool(processor.add_request(slot, params))
        self._needs_processing[slot] = needs_processing
        self._row_slot[row] = slot

    def _release_row(self, row: int) -> None:
        slot = self._row_slot.pop(row, None)
        if slot is None:
            return
        del self._output_token_ids[slot]
        self._needs_processing[slot] = False
        self._free_slots.append(slot)

    def _sync_history(self, slots: np.ndarray) -> None:
        """Copy output tokens committed since the last sync into all_token_ids."""
        states = self.req_states
        for slot in slots.tolist():
            output = self._output_token_ids[slot]
            start = int(self._num_synced[slot])
            if len(output) == start:
                continue
            offset = int(states.prompt_len.np[slot])
            states.all_token_ids.np[slot, offset + start : offset + len(output)] = (
                output[start:]
            )
            self._num_synced[slot] = len(output)
            states.total_len.np[slot] = offset + len(output)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        rows = [row for row in range(logits.shape[0]) if row in self._row_slot]
        if not rows:
            return logits
        slots = np.array([self._row_slot[row] for row in rows], dtype=np.int32)
        if not self._needs_processing[slots].any():
            return logits
        self._sync_history(slots)
        total_len = self.req_states.total_len.np[slots]
        pos = total_len.astype(np.int64) - 1
        slot_index = torch.from_numpy(slots)
        ctx = LogitsContext(
            expanded_idx_mapping=slot_index,
            idx_mapping=slot_index,
            idx_mapping_np=slots,
            expanded_local_pos=torch.zeros(len(rows), dtype=torch.int32),
            input_ids=torch.from_numpy(self.req_states.all_token_ids.np[slots, pos]),
            pos=torch.from_numpy(pos),
            seq_lens_upper_bound_np=total_len,
        )
        if len(rows) == logits.shape[0]:
            for processor in self.processors:
                logits = processor.apply(logits, ctx)
            return logits
        row_index = torch.tensor(rows, dtype=torch.int64)
        selected = logits[row_index]
        for processor in self.processors:
            selected = processor.apply(selected, ctx)
        logits[row_index] = selected
        return logits


def build_custom_logitsprocs(
    vllm_config: "VllmConfig", custom_logitsprocs: Sequence[str | type]
) -> CustomLogitsProcessors | None:
    """Load the operator's processors, or return None when there are none.

    vLLM's V2 loader reads ``custom_logitsprocs`` and the installed
    ``vllm.logits_processors`` entry points, and raises for a class that does
    not implement the V2 interface.
    """
    if not custom_logitsprocs and not entry_points(group=LOGITSPROCS_GROUP):
        return None
    model_config = vllm_config.model_config
    req_states = _HostRequestStates(
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        max_model_len=model_config.max_model_len,
        vocab_size=model_config.get_vocab_size(),
    )
    processors = build_custom_logits_processors(
        vllm_config,
        req_states,
        is_pooling_model=False,
        custom_logitsprocs=custom_logitsprocs,
    )
    if not processors:
        return None
    return CustomLogitsProcessors(processors, req_states)
