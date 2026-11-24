# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""What ``TTModelRunner`` gives a worker-side KV connector."""

import numpy as np
import torch
import ttnn
from vllm.distributed.kv_transfer.kv_connector.v1.base import CopyBlocksOp
from vllm.v1.kv_cache_interface import KVCacheConfig

# A 32x32 tile of bfloat8_b: one byte per value and a shared exponent per row
# of 16 values.
_BFLOAT8_TILE_BYTES = 1088


def _bfloat8_block_bytes(cache) -> int:
    """Bytes one block takes in each chip's buffer of a bfloat8_b KV cache.

    A paged KV cache is tiled block by block, so a block is one byte range.
    """
    _, heads, block_size, head_dim = tuple(cache.shape)
    return heads * (block_size // 32) * (head_dim // 32) * _BFLOAT8_TILE_BYTES


def kv_cache_descriptions(
    kv_cache_config: KVCacheConfig, kv_tensors: dict[str, tuple]
) -> dict[str, torch.Tensor]:
    """A torch tensor per layer name that describes the layer's KV cache.

    A connector sizes its CPU buffer from these. No KV is behind one: it is
    one element expanded to the shape. The KV itself is in the model's ttnn
    tensors, which a connector written against torch cannot read.

    A layer is described as bytes: ``(num_blocks, num_kv_heads, block_size,
    n)`` uint8, where a block's bytes are exactly the packed tiles of its K
    and V on every chip. ``make_copy_blocks`` moves those tiles unchanged, so
    the cache must be bfloat8_b, and a buffer block is only meaningful to a
    server whose cache has the same KV heads in total, head size and block
    size. The number of chips may differ: each chip holds an equal share of
    the heads in chip order, so the chips' tiles joined in that order are the
    tiles one chip would hold.
    """
    descriptions = {}
    for group in kv_cache_config.kv_cache_groups:
        # ``UniformTypeKVCacheSpecs`` wraps a spec per layer.
        per_layer_specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        for name in group.layer_names:
            spec = per_layer_specs[name] if per_layer_specs else group.kv_cache_spec
            k, _ = kv_tensors[name]
            if k.dtype != ttnn.bfloat8_b:
                raise ValueError(
                    f"A KV connector needs a bfloat8_b KV cache; layer {name} "
                    f"has {k.dtype}."
                )
            block_bytes = 2 * k.device().get_num_devices() * _bfloat8_block_bytes(k)
            last, remainder = divmod(block_bytes, spec.num_kv_heads * spec.block_size)
            if remainder:
                raise ValueError(
                    f"Layer {name}: a packed KV block of {block_bytes} bytes "
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


def kv_tensors_by_layer_name(
    kv_caches, layer_index_by_name: dict[str, int]
) -> dict[str, tuple]:
    """The model's ``(k, v)`` ttnn tensors for each ``kv_cache_config`` layer name.

    ``kv_caches`` is what the model's allocator returned: ``[k, v]`` per layer,
    or that list inside a list with one entry per submesh. A KV connector runs
    with one submesh per engine (no lanes), so there is one submesh.
    """
    per_layer = kv_caches
    if (
        isinstance(per_layer, (list, tuple))
        and len(per_layer) == 1
        and isinstance(per_layer[0], (list, tuple))
        and isinstance(per_layer[0][0], (list, tuple))
    ):
        (per_layer,) = per_layer
    if not isinstance(per_layer, (list, tuple)) or len(per_layer) != len(
        layer_index_by_name
    ):
        raise ValueError(
            "A KV connector needs one [k, v] KV cache entry per kv_cache_config "
            f"layer name; got layer names {list(layer_index_by_name)} and "
            f"kv_caches={kv_caches!r}."
        )
    kv_tensors = {}
    for name, index in layer_index_by_name.items():
        k, v = per_layer[index]
        kv_tensors[name] = (k, v)
    return kv_tensors


def make_copy_blocks(kv_tensors: dict[str, tuple]) -> CopyBlocksOp:
    """The copy between the model's ttnn KV cache and a connector's CPU buffer.

    The buffer is the connector's own: per layer name, a CPU tensor shaped
    like ``kv_cache_descriptions``. The connector's "device" dict holds those
    descriptions and is ignored; ``kv_tensors`` is the real device side.

    Every layer's transfers are queued without waiting and the last one waits,
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
        names = list(kv_tensors)
        for name in names:
            k, v = kv_tensors[name]
            _copy_packed_blocks(
                k,
                v,
                _buffer_bytes(buffers[name]),
                device_ids,
                host_ids,
                to_host,
                wait=name == names[-1],
            )

    return copy_blocks


def _buffer_bytes(buffer: torch.Tensor) -> np.ndarray:
    """A connector's uint8 buffer for one layer as flat bytes, block after block.

    NixlConnector allocates ``(blocks, tokens, heads, n)`` and hands out the
    view ``(blocks, heads, tokens, n)``. ``view`` raises if neither is the
    buffer's memory order.
    """
    if not buffer.is_contiguous():
        buffer = buffer.permute(0, 2, 1, 3)
    return buffer.view(-1).numpy()


def _copy_packed_blocks(
    k,
    v,
    host: np.ndarray,
    device_ids: list[int],
    host_ids: list[int],
    to_host: bool,
    wait: bool,
) -> None:
    """Move blocks of one layer's bfloat8_b K and V between chips and host bytes.

    A buffer block holds K's tiles from each chip in chip order, then V's. No
    program runs on the device and nothing is converted on the host: each
    (chip, block) is one region copied straight to or from its place in
    ``host``.

    Parameters
    ----------
    k, v
        The layer's ttnn K and V caches, each paged ``[blocks, heads, tokens,
        head_dim]`` with the heads split evenly over the chips.
    host
        The layer's buffer as flat bytes; it must stay alive until the copy
        has landed.
    device_ids, host_ids
        Block IDs on the chips and in ``host``, pairwise.
    to_host
        Copy chips to ``host`` when true, ``host`` to chips otherwise.
    wait
        Return only once V's transfers, and so every earlier one on the queue,
        have landed.
    """
    chip_bytes = _bfloat8_block_bytes(k)
    chips = k.device().get_num_devices()
    cache_bytes = chips * chip_bytes
    block_bytes = 2 * cache_bytes
    for i, cache in enumerate((k, v)):
        regions = _block_regions(
            device_ids,
            host_ids,
            chips,
            chip_bytes,
            block_bytes,
            host.ctypes.data + i * cache_bytes,
        )
        blocking = wait and i == 1
        if to_host:
            ttnn.copy_device_to_host_regions(cache, regions, blocking=blocking)
        else:
            ttnn.copy_host_to_device_regions(cache, regions, blocking=blocking)


def _block_regions(
    device_ids: list[int],
    host_ids: list[int],
    chips: int,
    chip_bytes: int,
    block_bytes: int,
    host_base: int,
) -> np.ndarray:
    """Rows of ``(chip, offset on the chip, bytes, host address)`` for a block copy.

    Chip ``c``'s share of device block ``d`` is bytes ``d * chip_bytes`` of its
    buffer, and sits at ``host_base + h * block_bytes + c * chip_bytes`` for
    host block ``h``.

    Returns
    -------
    numpy.ndarray
        ``(chips * len(device_ids), 4)`` uint64, chip by chip, in the order of
        ``device_ids``.
    """
    device = np.asarray(device_ids, dtype=np.uint64)
    host = np.asarray(host_ids, dtype=np.uint64)
    chip = np.repeat(np.arange(chips, dtype=np.uint64), len(device))
    regions = np.empty((len(chip), 4), dtype=np.uint64)
    regions[:, 0] = chip
    regions[:, 1] = np.tile(device, chips) * chip_bytes
    regions[:, 2] = chip_bytes
    regions[:, 3] = host_base + np.tile(host, chips) * block_bytes + chip * chip_bytes
    return regions
