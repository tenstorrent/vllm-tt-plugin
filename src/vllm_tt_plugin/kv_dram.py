# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where each vLLM KV layer's blocks are in TT card DRAM, for NIXL.

With ``NixlConnector`` on TT, NIXL's Tenstorrent build moves KV between the
DRAM of two servers' cards directly. It addresses card memory by card and DRAM
bank: card N of ``/dev/tenstorrent/N``, address ``(bank_id << 32) | offset``
within the bank. ``kv_dram_layout`` turns a model's ``KVLayout`` into such
addresses: per layer name, one ``DramRegion`` per DRAM bank of each shard of
each part, and the banks that hold them.

ttnn stores an interleaved DRAM tensor one page per slot and deals the pages
round-robin over a chip's DRAM banks: page ``i`` is in bank ``i % 8``, at
``(i // 8)`` pages past the tensor's buffer address plus the bank's base
address. When one block of a part is a multiple of 8 pages on a chip, every
block starts in bank 0, so block ``b`` is one byte range in each bank, ``b``
such ranges past the first block's.

Principal entry points: ``kv_dram_layout``, ``DramRegion`` and
``KVDramLayout``. This module only reads the layout (``kv_layout``) and ttnn's
cluster facts: each chip's DRAM bank table and the cluster descriptor. It does
not register memory with NIXL or move KV.
"""

from typing import NamedTuple

import ttnn
import yaml

from vllm_tt_plugin.kv_layout import KVLayout, KVPart

# NIXL's Tenstorrent build numbers a chip's banks 0 to 7, which are ttnn's
# bank IDs only when all 8 are in use.
_DRAM_BANKS = 8
# Bytes of a bank NIXL's Tenstorrent build maps; the top 16 MiB of the bank's
# 4 GiB window is not DRAM.
_NIXL_BANK_BYTES = 0xFF000000


class DramRegion(NamedTuple):
    """One shard's byte range in one DRAM bank, as a NIXL transfer names it.

    Block ``b`` of the region is the ``block_bytes`` bytes at
    ``addr + b * block_stride``.

    Attributes
    ----------
    card
        N of ``/dev/tenstorrent/N``.
    addr
        ``(bank_id << 32) | offset`` of the region's block 0 in the bank.
    block_bytes
        Bytes of one block in the region.
    block_stride
        Bytes from one block to the next in the bank: the shard's whole block
        in that bank, more than ``block_bytes`` when ``parts`` splits a shard.
    """

    card: int
    addr: int
    block_bytes: int
    block_stride: int


class KVDramLayout(NamedTuple):
    """Where the model's KV is in card DRAM.

    Attributes
    ----------
    regions
        Per layer name, the layer's regions in transfer order (see
        ``kv_dram_layout``).
    banks
        ``(bank_id << 32, bank_size, card)`` of every bank that holds a region,
        each once, in order of first use. NIXL registers whole banks.
    """

    regions: dict[str, list[DramRegion]]
    banks: list[tuple[int, int, int]]


def kv_dram_layout(
    layout: KVLayout, num_blocks: int, parts: int | None = None
) -> KVDramLayout:
    """The DRAM regions of each layer's KV, and the banks that hold them.

    Parameters
    ----------
    layout
        The model's KV transfer layout. Every part's tensor must be DRAM
        interleaved, and every shard held by one chip.
    num_blocks
        vLLM's KV block count; every region must hold that many blocks.
    parts
        How many equal parts to list a KV part in, over all its shards.
        ``None`` lists one per shard. A server with fewer chips than the other
        server of a transfer sets it to the other's shard count: with
        ``parts=2``, a 1-chip server lists each per-bank range as two halves,
        the bytes that each chip of a 2-chip server holds in that bank. It must
        be a positive multiple of each part's shard count.

    Returns
    -------
    KVDramLayout
        Per layer name, the regions in this order: the layout's parts in
        order; in a part, its shards in order; in a shard, its ``parts``
        subparts in order; in a subpart, the chip's banks in bank-table order.
        A server with more chips lists the same bytes in the same order when
        the KV heads are split over the chips in order.

    Raises
    ------
    ValueError
        With the layer name and the offending values, when a part's tensor is
        not DRAM interleaved; its ``block_bytes`` is not a whole number of
        pages; its pages per block do not divide over 8 banks, or into
        ``parts`` subparts of whole pages in each bank; ``parts`` is not a
        positive multiple of its shard count; a shard is on several chips; a
        chip does not use exactly 8 DRAM banks or is not a PCIe card of this
        host; or ``num_blocks`` blocks do not fit a bank, or end past the
        0xFF000000 bytes NIXL maps of one.

    Notes
    -----
    A shard copied to several chips is refused: a pull reads one remote range
    into one local range, so a copy has no single range on each side to pair.

    ``get_device_ids()`` lists a mesh's devices in row-major order, the order
    of ``KVPart.shards`` indices. A mesh tensor has the same buffer address on
    every chip.
    """
    cards = _card_by_device()
    regions: dict[str, list[DramRegion]] = {}
    banks: dict[tuple[int, int], tuple[int, int, int]] = {}
    for name, layer_parts in layout.layers.items():
        regions[name] = []
        for part in layer_parts:
            subparts, block_stride = _block_in_bank(name, part, parts)
            block_bytes = block_stride // subparts
            device_ids = part.tensor.device().get_device_ids()
            address = part.tensor.buffer_address()
            for shard in part.shards:
                device_id, card, table = _shard_chip(name, shard, device_ids, cards)
                for bank in table:
                    start = address + bank["base_addr"]
                    end = start + num_blocks * block_stride
                    limit = min(bank["bank_size"], _NIXL_BANK_BYTES)
                    if end > limit:
                        raise ValueError(
                            f"Layer {name}: {num_blocks} blocks of {block_stride} "
                            f"bytes from {start:#x} end at {end:#x}, past "
                            f"{limit:#x} in DRAM bank {bank['bank_id']} of "
                            f"device {device_id}."
                        )
                    banks.setdefault(
                        (card, bank["bank_id"]),
                        (bank["bank_id"] << 32, bank["bank_size"], card),
                    )
                regions[name] += [
                    DramRegion(
                        card,
                        (bank["bank_id"] << 32)
                        | (address + bank["base_addr"] + subpart * block_bytes),
                        block_bytes,
                        block_stride,
                    )
                    for subpart in range(subparts)
                    for bank in table
                ]
    return KVDramLayout(regions, list(banks.values()))


def _block_in_bank(name: str, part: KVPart, parts: int | None) -> tuple[int, int]:
    """Subparts per shard of ``part``, and a shard's bytes of a block in a bank.

    Raises
    ------
    ValueError
        When the tensor is not DRAM interleaved, or a block or ``parts`` does
        not divide into whole pages in each bank.
    """
    config = part.tensor.memory_config()
    if (
        config.buffer_type != ttnn.BufferType.DRAM
        or config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise ValueError(
            f"Layer {name}: a {config.buffer_type} {config.memory_layout} tensor "
            "is not DRAM interleaved, so its pages are not dealt over the DRAM "
            "banks."
        )
    page = part.tensor.buffer_aligned_page_size()
    pages, rest = divmod(part.block_bytes, page)
    if rest:
        raise ValueError(
            f"Layer {name}: a block of {part.block_bytes} bytes is not a whole "
            f"number of {page}-byte pages."
        )
    if pages % _DRAM_BANKS:
        raise ValueError(
            f"Layer {name}: a block of {pages} pages does not divide over "
            f"{_DRAM_BANKS} DRAM banks, so it is not one byte range in each bank."
        )
    shards = len(part.shards)
    subparts = 1
    if parts is not None:
        if parts <= 0 or parts % shards:
            raise ValueError(
                f"Layer {name}: parts={parts} is not a positive multiple of the "
                f"part's {shards} shards."
            )
        subparts = parts // shards
    if pages % (_DRAM_BANKS * subparts):
        raise ValueError(
            f"Layer {name}: parts={parts} splits a block of {pages} pages into "
            f"{subparts} subparts, which are not whole pages in each of "
            f"{_DRAM_BANKS} DRAM banks."
        )
    return subparts, pages // _DRAM_BANKS * page


def _shard_chip(
    name: str,
    shard: tuple[int, ...],
    device_ids: list[int],
    cards: dict[int, int],
) -> tuple[int, int, list[dict[str, int]]]:
    """The device ID, card and DRAM bank table of the one chip holding ``shard``.

    Raises
    ------
    ValueError
        When the shard is on several chips, or its chip does not use exactly
        8 DRAM banks or is not a PCIe card of this host.
    """
    if len(shard) != 1:
        raise ValueError(
            f"Layer {name}: a shard copied to mesh devices {list(shard)} has no "
            "single DRAM range to pull from or into; NIXL transfers need each "
            "shard on one chip."
        )
    device_id = device_ids[shard[0]]
    table = ttnn.cluster.get_dram_bank_table(device_id)
    if len(table) != _DRAM_BANKS:
        raise ValueError(
            f"Layer {name}: device {device_id} has {len(table)} DRAM banks in "
            f"use; NIXL numbers a chip's banks 0 to 7, which are ttnn's bank IDs "
            f"only when all {_DRAM_BANKS} are in use."
        )
    if device_id not in cards:
        raise ValueError(
            f"Layer {name}: device {device_id} is not a PCIe card of this host "
            f"(cards by device: {cards}), so NIXL cannot reach its DRAM."
        )
    return device_id, cards[device_id], table


def _card_by_device() -> dict[int, int]:
    """N of ``/dev/tenstorrent/N`` for each ttnn device ID of this process.

    The two differ: ttnn numbers the cards it may open from 0, in PCI address
    order. ttnn's cluster descriptor lists the card of each device that has
    one; a chip reached through another chip has none.
    """
    with open(ttnn.cluster.serialize_cluster_descriptor()) as f:
        descriptor = yaml.load(f, Loader=yaml.BaseLoader)
    return {
        int(device_id): int(card)
        for entry in descriptor.get("chips_with_mmio", [])
        for device_id, card in entry.items()
    }
