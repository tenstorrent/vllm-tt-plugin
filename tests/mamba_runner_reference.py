# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Reference for the Mamba state moves of vLLM's V1 GPU model runner.

With ``MambaSpec`` KV cache groups the scheduler only allocates block ids. The
upstream GPU model runner decides, every step, which blocks a recurrent layer
reads and writes and which state copies run:

- ``mamba_get_block_table_tensor`` (``vllm/v1/attention/backends/utils.py``)
  selects each row's state window: ``1 + num_speculative_blocks`` block ids,
  starting at column 0 in mode ``none`` and at column
  ``(seq_len - 1) // block_size`` in mode ``align``.
- ``preprocess_mamba`` (``vllm/v1/worker/mamba_utils.py``, mode ``align``)
  copies the running state into the block of the new column when a row's
  tokens cross into a new column. The source is the accepted state of the
  previous step: temporal state from column ``prev + num_accepted - 1``, conv
  state from column ``prev`` shifted by ``num_accepted - 1`` rows.
- ``postprocess_mamba_align_gpu`` (same file, mode ``align`` with speculative
  decoding) copies the state at a block boundary that the accepted tokens
  reached into the block of that boundary's column, so prefix caching finds
  it there.

``UpstreamMambaRunner`` calls upstream's own ``preprocess_mamba`` and
``mamba_get_block_table_tensor`` on CPU tensors and reads the copies back from
the source and destination pointers ``preprocess_mamba`` records. The
postprocess has only a Triton implementation upstream, so
``UpstreamMambaRunner.postprocess`` is a line-by-line port of the decision part
of ``postprocess_mamba_fused_kernel``; the byte copy follows the same
``get_conv_copy_spec`` / ``get_temporal_copy_spec`` rules as the preprocess.

``StateLedger`` replays the reads and writes of the recurrent kernels on
symbolic states ("the state after these tokens"), so a test can check that
every read finds the state of exactly the tokens the row has computed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    is_conv_state_dim_first,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker import mamba_utils


@dataclass(frozen=True)
class StateCopy:
    """One block copy of one Mamba group, as block ids."""

    group: int
    temporal_src: int
    conv_src: int
    # First conv-state row copied: the conv window shifts by the accepted
    # draft tokens.
    conv_row_offset: int
    dst: int


def mamba_spec_of(group: KVCacheGroupSpec) -> MambaSpec:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        spec = spec.first_spec
    if not isinstance(spec, MambaSpec):
        raise TypeError(f"group {group.layer_names[:2]} is not a Mamba group")
    return spec


def mamba_group_indices(kv_cache_config: KVCacheConfig) -> list[int]:
    return [
        i
        for i, group in enumerate(kv_cache_config.kv_cache_groups)
        if isinstance(
            group.kv_cache_spec.first_spec
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else group.kv_cache_spec,
            MambaSpec,
        )
    ]


def state_window(
    block_ids: Sequence[int], seq_len: int, spec: MambaSpec, width: int
) -> list[int]:
    """One row of upstream ``mamba_get_block_table_tensor``.

    ``width`` is the block-table row width the GPU runner allocates; mode
    ``none`` passes the whole row through, so its window is the row itself.
    """
    row = torch.zeros((1, width), dtype=torch.int32)
    row[0, : len(block_ids)] = torch.tensor(block_ids, dtype=torch.int32)
    window = mamba_get_block_table_tensor(
        row, torch.tensor([seq_len]), spec, spec.mamba_cache_mode
    )
    return window[0, : 1 + spec.num_speculative_blocks].tolist()


class UpstreamMambaRunner:
    """The Mamba bookkeeping of the upstream V1 GPU runner, without kernels.

    Holds what the GPU runner keeps across steps: ``mamba_state_idx`` (the
    column of each request's running state after its last step) and
    ``num_accepted`` (``InputBatch.num_accepted_tokens_cpu``, 1 when the last
    step accepted no draft or a copy reset it).
    """

    def __init__(self, kv_cache_config: KVCacheConfig, max_num_reqs: int = 8):
        if is_conv_state_dim_first():
            raise RuntimeError("the reference decodes only the SD conv layout")
        self.group_indices = mamba_group_indices(kv_cache_config)
        groups = kv_cache_config.kv_cache_groups
        self.spec = mamba_spec_of(groups[self.group_indices[0]])
        self.mamba_state_idx: dict[str, int] = {}
        self.num_accepted: dict[str, int] = {}
        # Upstream walks every layer of a group with the same block ids, so one
        # layer per group decides the copies. The states are tiny stand-ins
        # with upstream's GDN conv layout: kernel_size - 1 + K rows.
        conv_rows = 3 + self.spec.num_speculative_blocks
        num_blocks = kv_cache_config.num_blocks
        self._config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec([g.layer_names[0]], g.kv_cache_spec)
                if i in self.group_indices
                else g
                for i, g in enumerate(groups)
            ],
        )
        self._conv = {
            g: torch.zeros((num_blocks, conv_rows, 2), dtype=torch.int32)
            for g in self.group_indices
        }
        self._temporal = {
            g: torch.zeros((num_blocks, 4), dtype=torch.int32)
            for g in self.group_indices
        }
        self._forward_context = {
            self._config.kv_cache_groups[g].layer_names[0]: SimpleNamespace(
                kv_cache=[self._conv[g], self._temporal[g]]
            )
            for g in self.group_indices
        }
        self._copy_funcs = {
            self.spec.mamba_type: (
                MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
            )
        }
        self._copy_bufs = mamba_utils.MambaCopyBuffers.create(
            max_num_reqs,
            self._config,
            self._copy_funcs,
            lambda n, dtype: CpuGpuBuffer(n, dtype=dtype, device=torch.device("cpu")),
        )

    @property
    def block_size(self) -> int:
        return self.spec.block_size

    def preprocess(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: Sequence[str],
        req_states: Mapping[str, Any],
    ) -> list[StateCopy]:
        """Upstream ``preprocess_mamba`` for the rows ``req_ids``."""
        for req_id in req_ids:
            self.num_accepted.setdefault(req_id, 1)
        input_batch = SimpleNamespace(
            req_ids=list(req_ids),
            num_accepted_tokens_cpu=np.array(
                [self.num_accepted[r] for r in req_ids], dtype=np.int32
            ),
        )
        captured: list[tuple[int, int, int]] = []

        def capture(copy_bufs):
            n = copy_bufs.offset
            captured.extend(
                zip(
                    copy_bufs.src_ptrs.np[:n].tolist(),
                    copy_bufs.dst_ptrs.np[:n].tolist(),
                    copy_bufs.sizes.np[:n].tolist(),
                )
            )

        with patch.object(mamba_utils, "do_mamba_copy_block", capture):
            mamba_utils.preprocess_mamba(
                scheduler_output,
                self._config,
                SimpleNamespace(enable_prefix_caching=True),
                self.mamba_state_idx,
                input_batch,
                dict(req_states),
                self._forward_context,
                self._copy_funcs,
                self._copy_bufs,
            )
        for i, req_id in enumerate(req_ids):
            self.num_accepted[req_id] = int(input_batch.num_accepted_tokens_cpu[i])
        return self._decode(captured)

    def postprocess(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: Sequence[str],
        req_states: Mapping[str, Any],
        num_accepted: Mapping[str, int],
    ) -> list[StateCopy]:
        """Port of ``postprocess_mamba_fused_kernel``'s decisions (V1 path).

        ``req_states`` hold the computed-token counts from before this step.
        ``num_accepted`` counts each row's accepted tokens, the sampled token
        included.
        """
        block_size = self.block_size
        copies: list[StateCopy] = []
        for req_id in req_ids:
            accepted = num_accepted[req_id]
            self.num_accepted[req_id] = accepted
            src_col = self.mamba_state_idx[req_id]
            num_draft = len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            )
            running = (
                req_states[req_id].num_computed_tokens
                + scheduler_output.num_scheduled_tokens[req_id]
                - num_draft
            )
            new_computed = running + accepted - 1
            aligned = new_computed // block_size * block_size
            if aligned < running:
                continue
            bias = aligned - running
            dst_col = aligned // block_size - 1
            if src_col == dst_col:
                self.num_accepted[req_id] = 1
                if bias == 0:
                    continue
            for g in self.group_indices:
                block_ids = req_states[req_id].block_ids[g]
                copies.append(
                    StateCopy(
                        group=g,
                        temporal_src=block_ids[src_col + bias],
                        conv_src=block_ids[src_col],
                        conv_row_offset=bias,
                        dst=block_ids[dst_col],
                    )
                )
        return copies

    def forget(self, req_id: str) -> None:
        self.mamba_state_idx.pop(req_id, None)
        self.num_accepted.pop(req_id, None)

    def _locate(self, ptr: int) -> tuple[int, str, int, int]:
        """(group, "conv" or "temporal", block id, row) of a state pointer."""
        for g in self.group_indices:
            for kind, state in (
                ("conv", self._conv[g]),
                ("temporal", self._temporal[g]),
            ):
                base = state.data_ptr()
                stride = state.stride(0) * state.element_size()
                if base <= ptr < base + state.shape[0] * stride:
                    block, rest = divmod(ptr - base, stride)
                    row_bytes = state.stride(1) * state.element_size()
                    return g, kind, block, rest // row_bytes
        raise ValueError(f"pointer {ptr:#x} is in no Mamba state tensor")

    def _decode(self, captured: list[tuple[int, int, int]]) -> list[StateCopy]:
        # ``collect_mamba_copy_meta`` records, per request and group, the conv
        # state and then the temporal state of each layer.
        copies: list[StateCopy] = []
        for (conv_src, conv_dst, _), (temp_src, temp_dst, _) in zip(
            captured[0::2], captured[1::2]
        ):
            g, kind, conv_block, conv_row = self._locate(conv_src)
            g_dst, kind_dst, dst, _ = self._locate(conv_dst)
            g_t, kind_t, temporal_block, _ = self._locate(temp_src)
            g_td, kind_td, dst_t, _ = self._locate(temp_dst)
            if (kind, kind_dst, kind_t, kind_td) != (
                "conv",
                "conv",
                "temporal",
                "temporal",
            ):
                raise ValueError(f"unexpected copy order {captured}")
            if not g == g_dst == g_t == g_td or dst != dst_t:
                raise ValueError(f"conv and temporal copies disagree: {captured}")
            copies.append(StateCopy(g, temporal_block, conv_block, conv_row, dst))
        return copies


# A symbolic state: the number of tokens it has consumed and a hash of them.
Label = tuple[int, int]


def label(tokens: Sequence[int]) -> Label:
    return (len(tokens), hash(tuple(tokens)))


@dataclass
class _BlockState:
    temporal: Label | None
    # Conv state of upstream's GDN layout: entry ``o`` is the conv window
    # shifted by ``o`` rows, i.e. after ``o`` more accepted tokens.
    conv: list[Label | None] = field(default_factory=list)


class StateLedger:
    """Symbolic contents of each Mamba block, per group.

    Replays what the recurrent kernels read and write for one row per step,
    following upstream's GDN kernel contract: a speculating row reads its
    initial temporal state from window column ``num_accepted - 1`` and its
    conv state from column 0 shifted by ``num_accepted - 1`` rows, then writes
    the state after each of its ``1 + K`` query tokens into window columns
    ``0..K`` (conv: column 0 keeps all ``1 + K`` shifts). A row without drafts
    reads and writes column 0 only.
    """

    def __init__(self, group_indices: Sequence[int]):
        self.blocks: dict[int, dict[int, _BlockState]] = {g: {} for g in group_indices}

    def zero(self, block_ids: Sequence[int]) -> None:
        for blocks in self.blocks.values():
            for b in block_ids:
                blocks[b] = _BlockState(label(()), [label(())])

    def initial_state(
        self, group: int, window: Sequence[int], num_accepted: int, speculating: bool
    ) -> tuple[Label | None, Label | None]:
        blocks = self.blocks[group]
        shift = num_accepted - 1 if speculating else 0
        temporal = blocks.get(window[shift])
        conv = blocks.get(window[0])
        return (
            None if temporal is None else temporal.temporal,
            None if conv is None or len(conv.conv) <= shift else conv.conv[shift],
        )

    def write(
        self,
        group: int,
        window: Sequence[int],
        query_labels: Sequence[Label],
        speculating: bool,
    ) -> None:
        """``query_labels[t]`` is the state after query token ``t``."""
        blocks = self.blocks[group]
        if not speculating:
            blocks[window[0]] = _BlockState(query_labels[-1], [query_labels[-1]])
            return
        for t, lab in enumerate(query_labels):
            blocks.setdefault(window[t], _BlockState(None))
            blocks[window[t]] = _BlockState(lab, blocks[window[t]].conv)
        blocks[window[0]].conv = list(query_labels)

    def reject(self, group: int, window: Sequence[int], num_accepted: int) -> None:
        """Drafts past ``num_accepted`` were rejected: their states are junk."""
        blocks = self.blocks[group]
        for t in range(num_accepted, len(window)):
            blocks[window[t]].temporal = None
        conv = blocks[window[0]].conv
        for t in range(num_accepted, len(conv)):
            conv[t] = None

    def copy(self, copies: Sequence[StateCopy]) -> None:
        """Apply one batch of copies; every source is read before any write."""
        staged = []
        for c in copies:
            blocks = self.blocks[c.group]
            src_t = blocks.get(c.temporal_src)
            src_c = blocks.get(c.conv_src)
            conv = (
                None
                if src_c is None or len(src_c.conv) <= c.conv_row_offset
                else src_c.conv[c.conv_row_offset]
            )
            staged.append(
                (c, _BlockState(None if src_t is None else src_t.temporal, [conv]))
            )
        for c, state in staged:
            self.blocks[c.group][c.dst] = state

    def state(self, group: int, block_id: int) -> tuple[Label | None, Label | None]:
        s = self.blocks[group].get(block_id)
        if s is None:
            return None, None
        return s.temporal, (s.conv[0] if s.conv else None)
