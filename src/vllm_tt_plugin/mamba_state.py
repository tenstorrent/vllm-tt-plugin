# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Runner-side handling of vLLM ``MambaSpec`` KV cache groups.

A model whose ``get_kv_cache_spec`` returns ``MambaSpec`` for its recurrent
layers gets one or more Mamba groups next to its attention groups. vLLM's
hybrid KV cache manager allocates the block ids of every group from one pool
and overlays the groups on the same memory: block id ``b`` names the same
bytes in every group, and a block id belongs to one group at a time. The TT
model keeps that overlay. It allocates one buffer per layer slot (the
``tensor_idx`` of ``allocate_kv_cache_per_layer``) and keeps each Mamba
layer's state inside the buffer of its slot, at the block ids that layer's
group owns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec, MambaSpec

# ``CacheConfig.mamba_cache_mode`` values the TT runner implements.
SUPPORTED_MAMBA_CACHE_MODES = ("none",)


@dataclass(frozen=True)
class MambaCacheGroups:
    """The ``MambaSpec`` groups of one ``KVCacheConfig``."""

    # Indices into ``KVCacheConfig.kv_cache_groups``, in group order.
    group_indices: tuple[int, ...]
    # ``CacheConfig.mamba_cache_mode``; every Mamba group's spec carries it.
    mamba_cache_mode: str


def mamba_cache_groups(
    kv_cache_groups: Sequence[KVCacheGroupSpec], cache_config: Any
) -> MambaCacheGroups | None:
    """The Mamba groups of ``kv_cache_groups``, or ``None`` when there are none.

    Raises for a Mamba configuration the TT runner does not implement.
    """
    group_indices = tuple(
        i
        for i, group in enumerate(kv_cache_groups)
        if isinstance(group.kv_cache_spec, MambaSpec)
    )
    if not group_indices:
        return None
    if not any(isinstance(g.kv_cache_spec, AttentionSpec) for g in kv_cache_groups):
        raise ValueError(
            "The KV cache config has MambaSpec groups but no attention group. "
            "A TT model keeps Mamba state in the attention layer-slot buffers, "
            "so it needs at least one attention layer; groups: "
            f"{[type(g.kv_cache_spec).__name__ for g in kv_cache_groups]}"
        )
    mode = cache_config.mamba_cache_mode
    for i in group_indices:
        spec = kv_cache_groups[i].kv_cache_spec
        if spec.mamba_cache_mode != mode:
            raise ValueError(
                f"MambaSpec of KV cache group {i} has mamba_cache_mode="
                f"{spec.mamba_cache_mode!r}, but cache_config.mamba_cache_mode="
                f"{mode!r}; the model's get_kv_cache_spec must use the resolved "
                "cache mode"
            )
        if spec.num_speculative_blocks:
            raise ValueError(
                f"MambaSpec of KV cache group {i} has num_speculative_blocks="
                f"{spec.num_speculative_blocks}; speculative decoding with "
                "MambaSpec groups is not supported on TT yet"
            )
    if mode not in SUPPORTED_MAMBA_CACHE_MODES:
        raise ValueError(
            f"mamba_cache_mode={mode!r} is not supported on TT; supported: "
            f"{list(SUPPORTED_MAMBA_CACHE_MODES)}"
        )
    return MambaCacheGroups(group_indices=group_indices, mamba_cache_mode=mode)
