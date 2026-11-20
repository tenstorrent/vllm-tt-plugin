# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Custom logits processors on the host sampler, through vLLM's V2 interface.

``--logits-processors`` classes and installed ``vllm.logits_processors`` entry
points must implement ``vllm.v1.worker.gpu.sample.logits_processor``'s
``LogitsProcessor``. That interface keys a processor's per-request state by a
slot, and shares a slot-indexed token history, ``LogitsProcRequestState``. In
vLLM's V2 runner a request takes a slot, and the processors see
``add_request``, when a ``scheduled_new_reqs`` entry admits it; the request
keeps the slot until it finishes, is preempted, or a later
``scheduled_new_reqs`` entry rebuilds it. A step that does not schedule a
running request leaves its slot alone.

A persistent-batch row does not follow that lifetime. In front-packed mode,
``TTModelRunner._update_states`` removes a running request from the batch on
every step that does not schedule it (every prefill step, for a decoding
request) and adds it back later, and ``InputBatch.condense`` moves requests
between rows. So ``CustomLogitsProcessors`` keys slots by request id, binds
each request's current row to its slot through each ``BatchUpdate``, and frees
a slot only through ``release_requests``.
"""

from collections.abc import Iterable, Sequence
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


class CustomLogitsProcessors:
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
        self._req_slot: dict[str, int] = {}
        self._row_slot: dict[int, int] = {}
        # slot -> the request's live output list, and how much of it is
        # already copied into all_token_ids. The list is the one the request's
        # CachedRequestState holds, so it also identifies the lifetime that
        # took the slot.
        self._output_token_ids: dict[int, list[int]] = {}
        self._num_synced = np.zeros(max_num_reqs, dtype=np.int64)
        self._needs_processing = np.zeros(max_num_reqs, dtype=bool)

    def release_requests(self, req_ids: Iterable[str]) -> None:
        """Free the slots of requests whose lifetime ends: finished, preempted,
        or rebuilt from a ``scheduled_new_reqs`` entry. The rebuilt request
        takes a new slot, and the processors see ``add_request`` again, when
        the batch adds it back."""
        for req_id in req_ids:
            slot = self._req_slot.pop(req_id, None)
            if slot is None:
                continue
            for row in [row for row, held in self._row_slot.items() if held == slot]:
                del self._row_slot[row]
            del self._output_token_ids[slot]
            self._needs_processing[slot] = False
            self._free_slots.append(slot)

    def update_state(
        self, batch_update: BatchUpdate | None, req_ids: Sequence[str | None]
    ) -> None:
        """Bind batch rows to slots after one step's batch changes.

        ``req_ids`` is the batch's row-to-request list after those changes. A
        removed row only loses its binding: the request keeps its slot, and the
        processors their per-request state, until ``release_requests``.
        """
        if batch_update is None:
            return
        for row in batch_update.removed:
            self._row_slot.pop(row, None)
        added: dict[int, tuple[SamplingParams, list[int], list[int]]] = {}
        for row, params, prompt_token_ids, output_token_ids in batch_update.added:
            # The persistent batch drops a row from ``removed`` when a request
            # reuses it in the same step, so the previous occupant's binding
            # is dropped here.
            self._row_slot.pop(row, None)
            added[row] = (params, prompt_token_ids or [], output_token_ids)
        for src, dst, direction in batch_update.moved:
            if direction != MoveDirectionality.UNIDIRECTIONAL:
                raise RuntimeError(
                    f"Custom logits processors follow one-way row moves only; "
                    f"got {direction} for rows {src} -> {dst}"
                )
            slot = self._row_slot.pop(src, None)
            self._row_slot.pop(dst, None)
            if slot is not None:
                self._row_slot[dst] = slot
            if src in added:
                added[dst] = added.pop(src)
        admitted = False
        for row, (params, prompt_token_ids, output_token_ids) in added.items():
            req_id = req_ids[row] if row < len(req_ids) else None
            if req_id is None:
                raise RuntimeError(
                    f"Batch row {row} was added this step but holds no request "
                    f"after the step's moves; row requests: {list(req_ids)}"
                )
            slot = self._req_slot.get(req_id)
            if slot is None:
                slot = self._admit(req_id, params, prompt_token_ids, output_token_ids)
                admitted = True
            elif self._output_token_ids[slot] is not output_token_ids:
                raise RuntimeError(
                    f"Request {req_id} came back on batch row {row} with rebuilt "
                    f"state while it still holds custom logits processor slot "
                    f"{slot}; release_requests was not called for its previous "
                    "lifetime"
                )
            self._row_slot[row] = slot
        if admitted:
            for processor in self.processors:
                processor.apply_staged_writes()

    def _admit(
        self,
        req_id: str,
        params: SamplingParams,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
    ) -> int:
        if not self._free_slots:
            raise RuntimeError(
                f"No free custom logits processor slot for request {req_id}: "
                f"all {self.req_states.max_num_reqs} slots are held by requests "
                f"{sorted(self._req_slot)}"
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
        # Output tokens present at admission come from a resume after
        # preemption, which V2 counts as prefill.
        states.prefill_len.np[slot] = num_prompt + num_output
        states.total_len.np[slot] = num_prompt + num_output
        self._output_token_ids[slot] = output_token_ids
        self._num_synced[slot] = num_output
        needs_processing = False
        for processor in self.processors:
            needs_processing |= bool(processor.add_request(slot, params))
        self._needs_processing[slot] = needs_processing
        self._req_slot[req_id] = slot
        return slot

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


def custom_logitsproc_names(custom_logitsprocs: Sequence[str | type]) -> list[str]:
    """The custom processors vLLM's V2 loader loads: the ``--logits-processors``
    entries and the installed ``vllm.logits_processors`` entry points. vLLM's
    ``InputProcessor`` validates requests against both sources."""
    names = [
        entry if isinstance(entry, str) else entry.__qualname__
        for entry in custom_logitsprocs
    ]
    names += [
        entry_point.value for entry_point in entry_points(group=LOGITSPROCS_GROUP)
    ]
    return names


def build_custom_logitsprocs(
    vllm_config: "VllmConfig", custom_logitsprocs: Sequence[str | type]
) -> CustomLogitsProcessors | None:
    """Load the operator's processors, or return None when there are none.

    vLLM's V2 loader reads ``custom_logitsprocs`` and the installed
    ``vllm.logits_processors`` entry points, and raises for a class that does
    not implement the V2 interface.
    """
    if not custom_logitsproc_names(custom_logitsprocs):
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
