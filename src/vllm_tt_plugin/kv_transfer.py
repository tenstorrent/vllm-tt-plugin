# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""What ``TTModelRunner`` gives a worker-side KV connector."""

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig


def kv_cache_descriptions(kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """A torch tensor per layer name that describes the layer's KV cache.

    Each has vLLM's logical KV shape ``(num_blocks, num_kv_heads, block_size,
    2 * head_size)``, K and V side by side in the last dimension, and the
    spec's dtype. No KV is behind it: it is one element expanded to that
    shape. The KV itself is in the model's ttnn tensors, which a connector
    written against torch cannot read.
    """
    descriptions = {}
    for group in kv_cache_config.kv_cache_groups:
        # ``UniformTypeKVCacheSpecs`` wraps a spec per layer.
        per_layer_specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        for name in group.layer_names:
            spec = per_layer_specs[name] if per_layer_specs else group.kv_cache_spec
            shape = (
                kv_cache_config.num_blocks,
                spec.num_kv_heads,
                spec.block_size,
                2 * spec.head_size,
            )
            descriptions[name] = torch.empty(1, dtype=spec.dtype).expand(shape)
    return descriptions
