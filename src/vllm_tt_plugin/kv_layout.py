# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where each vLLM KV layer's blocks live on a TT mesh, for KV connectors.

A KV connector moves KV by vLLM layer name and block ID. A TT model keeps KV in
ttnn device tensors whose structure differs from model to model: K and V per
layer, one latent per layer (MLA), extra layers for an MTP drafter, heads split
over the chips or copied to all of them. ``KVLayout`` describes these cases the
same way, as long as each part is paged by vLLM block ID. For each layer name it
lists the paged device tensors, called parts here, that hold a block, how many
bytes one block of a part takes on one chip, and which chips hold which shard
of it. Connectors and the copy between chips and host only read the layout.

A model declares its layout with an optional ``get_kv_transfer_layout`` method
(see ``docs/MODEL_INTERFACE.md``). Without it, ``default_kv_layout`` derives
the layout of the common case: a ``[k, v]`` pair per attention layer, paged
``[blocks, heads, block_size, head_dim]`` in TILE layout, heads split evenly
over the chips in mesh order or copied to every chip.

Principal entry points: ``kv_transfer_layout`` and ``KVLayout``.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import ttnn
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig

_TILE = 32


@dataclass(frozen=True)
class KVPart:
    """One paged device tensor of a layer's KV.

    Block ``b`` of a part is bytes ``[b * block_bytes, (b + 1) * block_bytes)``
    of a chip's buffer: a paged cache is laid out block by block.

    Attributes
    ----------
    tensor
        The ttnn device tensor.
    block_bytes
        Bytes of one block on one chip.
    shards
        Per shard, in transfer order, the mesh devices (row-major index) that
        hold it. A copy reads a shard from its first device and writes it to
        all of them.
    """

    tensor: Any
    block_bytes: int
    shards: tuple[tuple[int, ...], ...]

    @property
    def transfer_bytes(self) -> int:
        """Bytes of one block of this part in a transfer: one copy per shard."""
        return len(self.shards) * self.block_bytes


@dataclass(frozen=True)
class KVLayout:
    """The parts of every layer a connector transfers, by vLLM layer name.

    A transferred block of a layer is its parts' blocks in order, each part
    shard after shard. Two servers exchange KV only when every layer's
    transferred block has the same bytes on both: same dtype, head size,
    block size and total heads. Their chip counts may differ when heads are
    split in chip order.
    """

    layers: Mapping[str, tuple[KVPart, ...]]

    def block_bytes(self, name: str) -> int:
        """Bytes of one transferred block of layer ``name``."""
        return sum(part.transfer_bytes for part in self.layers[name])


def kv_transfer_layout(
    model: Any, kv_caches: Any, kv_cache_config: KVCacheConfig
) -> KVLayout:
    """The model's declared KV layout, or the default one.

    Parameters
    ----------
    model
        The TT model; its optional ``get_kv_transfer_layout(kv_caches,
        kv_cache_config)`` returns, per layer name, a sequence of
        ``(tensor, block_bytes, shards)``.
    kv_caches
        What the model's KV allocator returned.
    kv_cache_config
        vLLM's KV cache config; every layer name in it must be described.

    Raises
    ------
    ValueError
        When the layout leaves out a layer name of ``kv_cache_config``, or
        a part's ``block_bytes`` or ``shards`` are empty.
    """
    declare = getattr(model, "get_kv_transfer_layout", None)
    if declare is None:
        return default_kv_layout(kv_caches, kv_cache_config)
    declared = declare(kv_caches, kv_cache_config)
    layers = {
        name: tuple(
            KVPart(tensor, int(block_bytes), tuple(tuple(s) for s in shards))
            for tensor, block_bytes, shards in parts
        )
        for name, parts in declared.items()
    }
    _check_covers(layers, kv_cache_config)
    for name, parts in layers.items():
        for part in parts:
            if part.block_bytes <= 0 or not part.shards or not all(part.shards):
                raise ValueError(f"Layer {name} declares an empty part: {part}.")
    return KVLayout(layers)


def default_kv_layout(kv_caches: Any, kv_cache_config: KVCacheConfig) -> KVLayout:
    """The layout of a ``[k, v]`` pair per attention layer.

    ``kv_caches`` is ``[k, v]`` per layer in layer-index order, or that list
    inside a one-entry list (one submesh). Layer ``model.layers.<i>...`` is
    entry ``i``. Each cache is ``[num_blocks, heads, block_size, head_size]``
    on every chip, with ``num_blocks`` from ``kv_cache_config`` and
    ``block_size`` and ``head_size`` from the layer's spec.

    Raises
    ------
    ValueError
        When the entries do not match the layer names, an entry is not a
        ``[k, v]`` pair, or a cache's shape, layout or dtype cannot be
        described here; such a model declares ``get_kv_transfer_layout``.
    """
    per_layer = kv_caches
    if (
        isinstance(per_layer, Sequence)
        and len(per_layer) == 1
        and isinstance(per_layer[0], Sequence)
        and isinstance(per_layer[0][0], Sequence)
    ):
        (per_layer,) = per_layer
    layers = {}
    for group in kv_cache_config.kv_cache_groups:
        specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        for name in group.layer_names:
            spec = specs[name] if specs else group.kv_cache_spec
            index = _layer_index(name)
            if not isinstance(per_layer, Sequence) or index >= len(per_layer):
                raise ValueError(
                    f"No [k, v] KV cache entry for layer {name}; the model "
                    "can declare its layout with get_kv_transfer_layout."
                )
            entry = per_layer[index]
            if not isinstance(entry, Sequence) or len(entry) != 2:
                found = (
                    f"{len(entry)} entries"
                    if isinstance(entry, Sequence)
                    else type(entry).__name__
                )
                raise ValueError(
                    f"Layer {name}: KV cache entry {index} is not a [k, v] pair "
                    f"but {found}; the model can declare its layout with "
                    "get_kv_transfer_layout."
                )
            k, v = entry
            layers[name] = (
                _paged_part(name, k, spec, kv_cache_config.num_blocks),
                _paged_part(name, v, spec, kv_cache_config.num_blocks),
            )
    if isinstance(per_layer, Sequence) and len(per_layer) != len(layers):
        raise ValueError(
            f"The model allocated {len(per_layer)} [k, v] entries for "
            f"{len(layers)} layer names; the model can declare its layout "
            "with get_kv_transfer_layout."
        )
    return KVLayout(layers)


def _paged_part(name: str, cache: Any, spec: AttentionSpec, num_blocks: int) -> KVPart:
    """A ``[num_blocks, heads, block_size, head_size]`` TILE cache as one part.

    ``heads`` is per chip: a mesh tensor's shape is the shape of each chip's
    shard. The other three dimensions must equal ``num_blocks`` and the spec's,
    or block ``b`` of the cache would not be vLLM's block ``b``.

    Raises
    ------
    ValueError
        When the cache is not 4-D, a dimension differs from ``num_blocks`` or
        ``spec``, or its dtype, layout or head split cannot be described here.
    """
    shape = tuple(cache.shape)
    if len(shape) != 4:
        raise ValueError(
            f"Layer {name}: a KV cache of shape {shape} is not [num_blocks, "
            "heads, block_size, head_size]; the model can declare its layout "
            "with get_kv_transfer_layout."
        )
    blocks, heads, block_size, head_dim = shape
    for field, found, expected in zip(
        ("num_blocks", "block_size", "head_size"),
        (blocks, block_size, head_dim),
        (num_blocks, spec.block_size, spec.head_size),
    ):
        if found != expected:
            raise ValueError(
                f"Layer {name}: the KV cache of shape {shape} has "
                f"{field}={found}, but kv_cache_config has {field}={expected}; "
                "the model can declare its layout with get_kv_transfer_layout."
            )
    num_kv_heads = spec.num_kv_heads
    tile_bytes = _tile_bytes(cache.dtype)
    if (
        tile_bytes is None
        or cache.layout != ttnn.TILE_LAYOUT
        or block_size % _TILE
        or head_dim % _TILE
    ):
        raise ValueError(
            f"Layer {name}: cannot describe a {cache.dtype} {cache.layout} cache "
            f"of shape {shape}; the model can declare its layout "
            "with get_kv_transfer_layout."
        )
    chips = cache.device().get_num_devices()
    if heads * chips == num_kv_heads:
        shards = tuple((chip,) for chip in range(chips))
    elif heads == num_kv_heads:
        shards = (tuple(range(chips)),)
    else:
        raise ValueError(
            f"Layer {name}: {heads} heads per chip on {chips} chips is neither "
            f"a split nor a copy of {num_kv_heads} KV heads; the model can "
            "declare its layout with get_kv_transfer_layout."
        )
    block_bytes = heads * (block_size // _TILE) * (head_dim // _TILE) * tile_bytes
    return KVPart(cache, block_bytes, shards)


def _tile_bytes(dtype: Any) -> int | None:
    """Bytes of one 32x32 tile of ``dtype``, or ``None`` for an unlisted dtype.

    bfloat8_b and bfloat4_b keep one shared exponent byte per row of 16 values.
    The table is built on call, so importing this module needs no ttnn dtypes.
    """
    sizes = {
        ttnn.bfloat16: 2048,
        ttnn.float32: 4096,
        ttnn.bfloat8_b: 1088,
        ttnn.bfloat4_b: 576,
    }
    return sizes.get(dtype)


def _check_covers(layers: Mapping[str, Any], kv_cache_config: KVCacheConfig) -> None:
    names = {n for g in kv_cache_config.kv_cache_groups for n in g.layer_names}
    if missing := sorted(names - layers.keys()):
        raise ValueError(f"The KV layout does not describe layers {missing}.")


def _layer_index(name: str) -> int:
    match = re.search(r"layers\.(\d+)", name)
    if match is None:
        raise ValueError(
            f"Layer name {name} has no layers.<index>; the model can declare "
            "its layout with get_kv_transfer_layout."
        )
    return int(match.group(1))
