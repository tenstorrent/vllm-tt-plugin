# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""What ``TTModelRunner`` gives a worker-side KV connector.

A connector written against torch sizes its CPU buffer from one torch tensor
per vLLM layer name and moves KV between that buffer and the device with a
``CopyBlocksOp``. Both come from the model's ``KVLayout``: a buffer block of a
layer is the layer's parts' blocks in order, each part shard after shard,
exactly as the chips hold them.

Principal entry points: ``kv_cache_descriptions`` and ``make_copy_blocks``.
"""

import numpy as np
import torch
import ttnn
from vllm.distributed.kv_transfer.kv_connector.v1.base import CopyBlocksOp
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_tt_plugin.kv_layout import KVLayout, KVPart


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


def make_copy_blocks(layout: KVLayout) -> CopyBlocksOp:
    """The copy between the model's ttnn KV cache and a connector's CPU buffer.

    The buffer is the connector's own: per layer name, a CPU tensor shaped
    like ``kv_cache_descriptions``. The connector's "device" dict holds those
    descriptions and is ignored; ``layout`` is the real device side.

    Every part's transfers are queued without waiting and the last one waits,
    so the copy returns once all of them have landed: the command queue runs
    them in order.
    """

    def copy_blocks(src, dst, src_block_ids, dst_block_ids, direction) -> None:
        to_host = direction == "d2h"
        buffers = dst if to_host else src
        device_ids, host_ids = (
            (src_block_ids, dst_block_ids)
            if to_host
            else (dst_block_ids, src_block_ids)
        )
        if not device_ids:
            return
        parts = [
            (name, offset, part)
            for name, layer_parts in layout.layers.items()
            for offset, part in _with_offsets(layer_parts)
        ]
        for i, (name, offset, part) in enumerate(parts):
            host = _buffer_bytes(buffers[name])
            regions = _part_regions(
                part,
                device_ids,
                host_ids,
                host.ctypes.data + offset,
                layout.block_bytes(name),
                to_host,
            )
            blocking = i == len(parts) - 1
            if to_host:
                ttnn.copy_device_to_host_regions(
                    part.tensor, regions, blocking=blocking
                )
            else:
                ttnn.copy_host_to_device_regions(
                    part.tensor, regions, blocking=blocking
                )

    return copy_blocks


def _with_offsets(parts: tuple[KVPart, ...]):
    """Each part with its byte offset inside a transferred block."""
    offset = 0
    for part in parts:
        yield offset, part
        offset += part.transfer_bytes


def _buffer_bytes(buffer: torch.Tensor) -> np.ndarray:
    """A connector's uint8 buffer for one layer as flat bytes, block after block.

    NixlConnector allocates ``(blocks, tokens, heads, n)`` and hands out the
    view ``(blocks, heads, tokens, n)``. ``view`` raises if neither is the
    buffer's memory order.
    """
    if not buffer.is_contiguous():
        buffer = buffer.permute(0, 2, 1, 3)
    return buffer.view(-1).numpy()


def _part_regions(
    part: KVPart,
    device_ids: list[int],
    host_ids: list[int],
    host_base: int,
    host_block_bytes: int,
    to_host: bool,
) -> np.ndarray:
    """Rows of ``(chip, offset on the chip, bytes, host address)`` for one part.

    Shard ``i`` of device block ``d`` is bytes ``d * block_bytes`` of each chip
    holding it, and sits at ``host_base + h * host_block_bytes + i *
    block_bytes`` for host block ``h``. A copy to the host reads each shard
    from its first chip; a copy to the device writes it to every chip.

    Returns
    -------
    numpy.ndarray
        ``(chips used * len(device_ids), 4)`` uint64, chip by chip, in the
        order of ``device_ids``.
    """
    targets = [
        (chip, i * part.block_bytes)
        for i, shard in enumerate(part.shards)
        for chip in (shard[:1] if to_host else shard)
    ]
    chips = np.array([chip for chip, _ in targets], dtype=np.uint64)
    offsets = np.array([offset for _, offset in targets], dtype=np.uint64)
    device = np.asarray(device_ids, dtype=np.uint64)
    host = np.asarray(host_ids, dtype=np.uint64)
    regions = np.empty((len(targets) * len(device), 4), dtype=np.uint64)
    regions[:, 0] = np.repeat(chips, len(device))
    regions[:, 1] = np.tile(device, len(targets)) * part.block_bytes
    regions[:, 2] = part.block_bytes
    regions[:, 3] = (
        host_base
        + np.tile(host, len(targets)) * host_block_bytes
        + np.repeat(offsets, len(device))
    )
    return regions
