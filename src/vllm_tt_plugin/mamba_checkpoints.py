# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""vLLM's prefill checkpoint rule for ``MambaSpec`` groups.

In ``mamba_cache_mode`` ``align`` with ``MambaSpec.num_prefill_checkpoint_blocks``
1, vLLM's scheduler lets one prefill chunk run past the prompt's last cacheable
position instead of ending the chunk there. ``MambaManager`` then reserves a
block in the chunk's next-to-last block-table column and, after the step,
caches it as the state after that position. The model must write that state
into the block while it continues the prompt.

The upstream GPU runner finds the position and the column with
``compute_mamba_prefill_checkpoints`` (``vllm/model_executor/layers/mamba/
checkpoint.py``) from each row's token counts, by the rule ``MambaManager``
reserves with, and reads the block from the request's block-table row. The TT
runner does the same through ``prefill_checkpoints``, the one place it applies
that rule, so it reads no private scheduler state. ``MambaManager`` declines
some checkpoints the rule allows, for example in a chunk that does not reach
the end of the prefill; the block-table column then holds the null block 0,
and ``MambaCacheGroups.step_state`` drops the checkpoint. A reserved block in
a column the rule does not name fails there instead of going unwritten.
"""

from __future__ import annotations

from collections.abc import Sequence

from vllm.model_executor.layers.mamba.checkpoint import (
    compute_mamba_prefill_checkpoints,
)


def prefill_checkpoints(
    num_computed_tokens: Sequence[int],
    num_scheduled_tokens: Sequence[int],
    *,
    mamba_block_size: int,
    prefix_match_unit: int,
    checkpoint_alignment: int,
) -> list[tuple[int, int] | None]:
    """Each row's ``(position, block-table column)`` checkpoint, or ``None``.

    ``position`` is the token count whose state the row must write, and
    ``column`` the block-table column of the block vLLM reserves for it.
    ``prefix_match_unit`` is ``cache_config.prefix_match_unit``, or the Mamba
    block size when unset, as upstream's ``MambaPrefillCheckpointBuilder``
    passes it. TT runs without speculative decoding, so vLLM drops no EAGLE
    block.
    """
    seq_lens = [c + s for c, s in zip(num_computed_tokens, num_scheduled_tokens)]
    offsets, columns = compute_mamba_prefill_checkpoints(
        seq_lens,
        list(num_scheduled_tokens),
        hash_block_size=prefix_match_unit,
        mamba_block_size=mamba_block_size,
        checkpoint_alignment=checkpoint_alignment,
        drop_eagle_block=False,
    )
    return [
        (computed + offset, column) if column >= 0 else None
        for computed, offset, column in zip(num_computed_tokens, offsets, columns)
    ]


__all__ = ["prefill_checkpoints"]
