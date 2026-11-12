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

The TT runner implements ``mamba_cache_mode`` ``none`` without speculative
decoding: a request holds one state block per Mamba group for its whole life.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec, MambaSpec

from vllm_tt_plugin.config import is_tt_block_output_model, uses_tt_lane_coordinator

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# ``CacheConfig.mamba_cache_mode`` values the TT runner implements.
SUPPORTED_MAMBA_CACHE_MODES = ("none",)


@dataclass(frozen=True)
class MambaCacheGroups:
    """The ``MambaSpec`` groups of one ``KVCacheConfig``."""

    # Indices into ``KVCacheConfig.kv_cache_groups``, in group order.
    group_indices: tuple[int, ...]
    # Each Mamba group's ``KVCacheGroupSpec.layer_names``, in the order of
    # ``group_indices``.
    layer_names: tuple[tuple[str, ...], ...]


def check_mamba_launch(vllm_config: VllmConfig, specs: Iterable[MambaSpec]) -> None:
    """Raise for a launch with ``MambaSpec`` layers that TT does not implement.

    ``specs`` are the model's ``MambaSpec`` layer specs, or the specs of the
    Mamba groups. The worker calls this before it loads the model, and the
    runner again for the KV cache groups.
    """
    if uses_tt_lane_coordinator(vllm_config):
        raise ValueError(
            "MambaSpec KV cache groups are not supported with single-process "
            "lane-DP; launch with standard multi-process data parallelism"
        )
    if is_tt_block_output_model(vllm_config):
        raise ValueError(
            "Block-output models (output_tokens_per_step > 1) do not support "
            "MambaSpec KV cache groups"
        )
    if vllm_config.speculative_config is not None:
        raise ValueError(
            "Speculative decoding is not supported with MambaSpec KV cache "
            "groups on TT yet; launch without --speculative-config"
        )
    mode = vllm_config.cache_config.mamba_cache_mode
    if mode not in SUPPORTED_MAMBA_CACHE_MODES:
        # vLLM selects mode "align" (or "all") only with prefix caching on.
        raise ValueError(
            f"mamba_cache_mode={mode!r} is not supported with MambaSpec KV "
            f"cache groups on TT (supported: {list(SUPPORTED_MAMBA_CACHE_MODES)}); "
            "TT does not reuse cached recurrent state yet, so launch with "
            "--no-enable-prefix-caching"
        )
    max_model_len = vllm_config.model_config.max_model_len
    for spec in specs:
        if spec.mamba_cache_mode != mode:
            raise ValueError(
                f"MambaSpec has mamba_cache_mode={spec.mamba_cache_mode!r}, "
                f"but cache_config.mamba_cache_mode={mode!r}; the model's "
                "get_kv_cache_spec must use the resolved cache mode"
            )
        if spec.num_speculative_blocks or spec.num_prefill_checkpoint_blocks:
            raise ValueError(
                "MambaSpec has num_speculative_blocks="
                f"{spec.num_speculative_blocks} and num_prefill_checkpoint_blocks="
                f"{spec.num_prefill_checkpoint_blocks}; TT supports neither "
                "speculative nor prefill-checkpoint state blocks yet, so both "
                "must be 0"
            )
        if spec.block_size < max_model_len:
            # MambaManager allocates one block per block_size tokens in mode
            # none, so a smaller block size gives a request several state
            # blocks per group, which neither the pool sizing nor the state
            # inputs count.
            raise ValueError(
                f"MambaSpec block_size {spec.block_size} is below max_model_len "
                f"{max_model_len} in mamba_cache_mode='none'; declare block_size "
                "as cache_config.mamba_block_size, which vLLM resolves to "
                "max_model_len"
            )


def mamba_cache_groups(
    kv_cache_groups: Sequence[KVCacheGroupSpec], vllm_config: VllmConfig
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
    check_mamba_launch(
        vllm_config, (kv_cache_groups[i].kv_cache_spec for i in group_indices)
    )
    return MambaCacheGroups(
        group_indices=group_indices,
        layer_names=tuple(tuple(kv_cache_groups[i].layer_names) for i in group_indices),
    )


__all__ = [
    "SUPPORTED_MAMBA_CACHE_MODES",
    "MambaCacheGroups",
    "check_mamba_launch",
    "mamba_cache_groups",
]
