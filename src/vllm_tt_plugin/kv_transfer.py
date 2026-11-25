# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""What ``TTModelRunner`` gives a worker-side KV connector.

A connector written against torch sizes its CPU buffer from one torch tensor
per vLLM layer name. That tensor comes from the model's ``KVLayout``: a buffer
block of a layer is the layer's parts' blocks in order, each part shard after
shard, exactly as the chips hold them.

Principal entry point: ``kv_cache_descriptions``.
"""

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_tt_plugin.kv_layout import KVLayout


def kv_cache_descriptions(
    kv_cache_config: KVCacheConfig, layout: KVLayout
) -> dict[str, torch.Tensor]:
    """A torch tensor per layer name that describes the layer's KV cache.

    No KV is behind one: it is one element expanded to the shape. A layer is
    described as bytes, ``(num_blocks, num_kv_heads, block_size, n)`` uint8,
    sized so that a block is exactly one transferred block of the layer.

    Raises
    ------
    ValueError
        When a layer's transferred block does not divide into its spec's
        heads and block size.
    """
    descriptions = {}
    for group in kv_cache_config.kv_cache_groups:
        # ``UniformTypeKVCacheSpecs`` wraps a spec per layer.
        per_layer_specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        for name in group.layer_names:
            spec = per_layer_specs[name] if per_layer_specs else group.kv_cache_spec
            block_bytes = layout.block_bytes(name)
            last, remainder = divmod(block_bytes, spec.num_kv_heads * spec.block_size)
            if remainder:
                raise ValueError(
                    f"Layer {name}: a transferred block of {block_bytes} bytes "
                    f"does not divide into {spec.num_kv_heads} heads of "
                    f"{spec.block_size} tokens."
                )
            shape = (
                kv_cache_config.num_blocks,
                spec.num_kv_heads,
                spec.block_size,
                last,
            )
            descriptions[name] = torch.empty(1, dtype=torch.uint8).expand(shape)
    return descriptions
