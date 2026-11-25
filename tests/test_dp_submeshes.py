# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Placement policy of the plugin's standard-DP submesh split.

Lane mode (``tt_data_parallel``) places devices with the model package's own
``create_submeshes``; standard DP places them with this copy. The table below
pins the shared policy so the two cannot drift apart silently.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import ttnn

from vllm_tt_plugin.utils import dp_discovery
from vllm_tt_plugin.utils.dp_submeshes import create_dp_submeshes, dp_submesh_shapes

# (mesh shape, DP) -> (parent shape after any reshape, submesh shape)
PLACEMENTS = [
    # Galaxy: viewed as 4x8, split by rows; groups under 8 devices are 1xN.
    ((4, 8), 2, (4, 8), (2, 8)),
    ((4, 8), 4, (4, 8), (1, 8)),
    ((4, 8), 8, (4, 8), (1, 4)),
    ((4, 8), 16, (4, 8), (1, 2)),
    ((4, 8), 32, (4, 8), (1, 1)),
    ((8, 4), 4, (4, 8), (1, 8)),
    ((1, 32), 2, (4, 8), (2, 8)),
    ((2, 16), 8, (4, 8), (1, 4)),
    # Everything else: 1xN rows of the parent, which keeps its shape.
    ((1, 8), 2, (1, 8), (1, 4)),
    ((1, 8), 4, (1, 8), (1, 2)),
    ((1, 8), 8, (1, 8), (1, 1)),
    ((2, 4), 2, (2, 4), (1, 4)),
    ((1, 4), 2, (1, 4), (1, 2)),
    ((1, 4), 4, (1, 4), (1, 1)),
    ((1, 2), 2, (1, 2), (1, 1)),
]


class FakeMesh(ttnn.MeshDevice):
    """Records the reshape and split calls a real mesh would receive.

    Subclassing ``ttnn.MeshDevice`` lets the lane-mode helper's ``isinstance``
    check accept it. Native ttnn's ``shape`` is a read-only property, so the
    fake keeps its own, and it reads a ``MeshShape`` by iterating it because
    native ``MeshShape.dims`` is a method.
    """

    def __init__(self, shape):
        self._shape = tuple(shape)
        self.calls = []

    @property
    def shape(self):
        return self._shape

    def reshape(self, mesh_shape):
        self.calls.append(("reshape", tuple(mesh_shape)))
        self._shape = tuple(mesh_shape)

    def create_submeshes(self, mesh_shape):
        self.calls.append(("create_submeshes", tuple(mesh_shape)))
        rows, cols = self._shape
        sub_rows, sub_cols = mesh_shape
        return [object()] * ((rows * cols) // (sub_rows * sub_cols))


@pytest.mark.parametrize(("mesh_shape", "dp", "parent", "submesh"), PLACEMENTS)
def test_placement_policy(mesh_shape, dp, parent, submesh):
    assert dp_submesh_shapes(mesh_shape, dp) == (parent, submesh)


@pytest.mark.parametrize(("mesh_shape", "dp"), [((1, 8), 3), ((4, 8), 5), ((1, 2), 0)])
def test_uneven_split_is_rejected(mesh_shape, dp):
    with pytest.raises(ValueError, match="Unsupported device split"):
        dp_submesh_shapes(mesh_shape, dp)


@pytest.mark.parametrize(("mesh_shape", "dp", "parent", "submesh"), PLACEMENTS)
def test_create_dp_submeshes_reshapes_only_when_needed(mesh_shape, dp, parent, submesh):
    mesh = FakeMesh(mesh_shape)

    submeshes = create_dp_submeshes(mesh, dp)

    expected = [("create_submeshes", submesh)]
    if parent != mesh_shape:
        expected.insert(0, ("reshape", parent))
    assert mesh.calls == expected
    assert len(submeshes) == dp


def test_create_dp_submeshes_keeps_the_parent_for_dp1():
    mesh = FakeMesh((4, 8))

    assert create_dp_submeshes(mesh, 1) == [mesh]
    assert mesh.calls == []


def test_discovery_needs_no_model_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Standard-DP discovery imports only ttnn and the plugin, so it works in an
    environment that has no tt-metal ``models`` tree on the path."""
    monkeypatch.setitem(sys.modules, "models", None)
    opened = _pcie_cluster(monkeypatch, tmp_path, P150X8_CARDS)

    groups = dp_discovery._discover_standard_dp_visible_device_groups("P150x8", 4)

    assert groups == [
        (_cards(0, 1), (1, 2)),
        (_cards(2, 3), (1, 2)),
        (_cards(4, 5), (1, 2)),
        (_cards(6, 7), (1, 2)),
    ]
    assert opened[0].calls == [("create_submeshes", (1, 2))]


def _cluster_descriptor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, addresses: list[str]
) -> None:
    """Serves a cluster descriptor that gives chip ``i`` the address
    ``addresses[i]``, written unquoted as UMD writes it."""
    descriptor = tmp_path / "cluster_descriptor.yaml"
    descriptor.write_text(
        "chip_pci_bdfs:\n"
        + "".join(f"  {chip}: {address}\n" for chip, address in enumerate(addresses))
    )
    monkeypatch.setattr(
        ttnn.cluster, "serialize_cluster_descriptor", lambda: str(descriptor)
    )


# PCI addresses of a T3K's four N300 cards. The first is all digits, which
# YAML 1.1 would read as a number.
T3K_CARDS = ["0000:31:00.0", "0000:4b:00.0", "0000:b1:00.0", "0000:ca:00.0"]


def _n300_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, addresses: list[str]
) -> list:
    """N300 cards as UMD numbers them in this process: the cards' PCIe chips
    first, as chips 0 to n-1, then their remote chips, each reached through card
    2n-1 - chip.
    """
    num_cards = len(addresses)
    _cluster_descriptor(monkeypatch, tmp_path, addresses)
    opened = []
    monkeypatch.setattr(ttnn, "get_num_devices", lambda: 2 * num_cards, raising=False)
    monkeypatch.setattr(
        ttnn,
        "GetPCIeDeviceID",
        lambda chip: chip if chip < num_cards else 2 * num_cards - 1 - chip,
        raising=False,
    )
    monkeypatch.setattr(
        ttnn, "open_mesh_device", lambda shape: opened.append(shape), raising=False
    )
    monkeypatch.setattr(ttnn, "close_mesh_device", lambda mesh: None, raising=False)
    return opened


def test_n300_cluster_binds_one_whole_card_per_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """TT_VISIBLE_DEVICES names PCIe cards, and opening one opens its remote chip
    too, so on a T3K each rank gets one card as a 1x2 mesh. No parent mesh is
    opened: its submeshes pair chips of different cards, which no rank can open.
    """
    opened = _n300_cluster(monkeypatch, tmp_path, T3K_CARDS)

    groups = dp_discovery._discover_standard_dp_visible_device_groups("T3K", 4)

    assert groups == [(address, (1, 2)) for address in T3K_CARDS]
    assert opened == []


def test_n300_cards_keep_the_operators_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Under TT_VISIBLE_DEVICES="2,3" the host's third and fourth cards are chips
    0 and 1 of this process, while a worker given "0" or "1" would open the
    host's first two cards. Each rank gets its card's PCI address instead.
    """
    _n300_cluster(monkeypatch, tmp_path, T3K_CARDS[2:])

    groups = dp_discovery._discover_standard_dp_visible_device_groups("T3K", 2)

    assert groups == [("0000:b1:00.0", (1, 2)), ("0000:ca:00.0", (1, 2))]


@pytest.mark.parametrize("dp", [2, 8])
def test_n300_cluster_rejects_a_split_that_is_not_one_card_per_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dp: int
):
    opened = _n300_cluster(monkeypatch, tmp_path, T3K_CARDS)

    with pytest.raises(ValueError, match=r"need data_parallel_size=4, got " + str(dp)):
        dp_discovery._discover_standard_dp_visible_device_groups("T3K", dp)
    assert opened == []


# PCI addresses of an 8x P150 host's cards, in the order of the host's integer
# ids: integer k in TT_VISIBLE_DEVICES names the k-th card by PCI address.
P150X8_CARDS = [
    "0000:01:00.0",
    "0000:25:00.0",
    "0000:41:00.0",
    "0000:61:00.0",
    "0000:81:00.0",
    "0000:a1:00.0",
    "0000:c1:00.0",
    "0000:e1:00.0",
]


def _cards(*host_ids: int) -> str:
    """The ``TT_VISIBLE_DEVICES`` value that names these host cards by address."""
    return ",".join(P150X8_CARDS[host_id] for host_id in host_ids)


def _umd_selection(visible: str | None) -> list[str]:
    """Addresses of the cards UMD opens for ``visible``, in the order it numbers
    them: an integer entry k is the host's k-th card, any other entry selects the
    cards whose address contains it, and the selection is sorted by address."""
    if visible is None:
        return P150X8_CARDS
    selected = set()
    for entry in visible.split(","):
        if entry.isdecimal():
            selected.add(P150X8_CARDS[int(entry)])
        else:
            selected.update(card for card in P150X8_CARDS if entry in card)
    return sorted(selected)


def _pcie_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, addresses: list[str]
) -> list:
    """P150 cards as UMD numbers them in this process: every chip on its own PCIe
    link, as chips 0 to n-1 at ``addresses``. Returns the meshes discovery opens.
    """
    num_chips = len(addresses)

    class Mesh(FakeMesh):
        def create_submeshes(self, mesh_shape):
            super().create_submeshes(mesh_shape)
            shape = tuple(mesh_shape)
            size = shape[0] * shape[1]
            return [
                SimpleNamespace(
                    shape=shape,
                    get_device_ids=lambda first=first: range(first, first + size),
                )
                for first in range(0, num_chips, size)
            ]

    opened = []

    def open_mesh_device(mesh_shape):
        opened.append(Mesh(tuple(mesh_shape)))
        return opened[-1]

    _cluster_descriptor(monkeypatch, tmp_path, addresses)
    monkeypatch.setattr(ttnn, "get_num_devices", lambda: num_chips, raising=False)
    monkeypatch.setattr(ttnn, "GetPCIeDeviceID", lambda chip: chip, raising=False)
    monkeypatch.setattr(ttnn, "open_mesh_device", open_mesh_device, raising=False)
    monkeypatch.setattr(ttnn, "close_mesh_device", lambda mesh: None, raising=False)
    monkeypatch.setattr(
        ttnn.cluster, "get_cluster_type", lambda: ttnn.cluster.ClusterType.P150
    )
    return opened


@pytest.mark.parametrize(
    ("visible", "dp", "host_groups"),
    [
        pytest.param(None, 2, [(0, 1, 2, 3), (4, 5, 6, 7)], id="whole-host"),
        pytest.param("2,3", 2, [(2,), (3,)], id="integer-subset"),
        pytest.param("3,1", 2, [(1,), (3,)], id="unsorted-integers"),
        pytest.param("0,1,4,5", 2, [(0, 1), (4, 5)], id="integer-gap"),
        pytest.param(_cards(*range(8)), 8, [(i,) for i in range(8)], id="pci-host"),
        pytest.param(_cards(2, 3), 2, [(2,), (3,)], id="pci-subset"),
        pytest.param("1,61:00.0", 2, [(1,), (3,)], id="mixed-partial-pci"),
    ],
)
def test_pcie_groups_keep_the_operators_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    visible: str | None,
    dp: int,
    host_groups: list[tuple[int, ...]],
):
    """Chip ids of the discovery process are local to its inherited
    TT_VISIBLE_DEVICES: under "2,3" its chips 0 and 1 are the host's cards 2 and
    3, while a worker given "0" or "1" would open the host's first two cards.
    Each rank gets its cards' PCI addresses instead, whatever form the
    operator's selection takes.
    """
    _pcie_cluster(monkeypatch, tmp_path, _umd_selection(visible))

    groups = dp_discovery._discover_standard_dp_visible_device_groups("P150", dp)

    assert groups == [(_cards(*ids), (1, len(ids))) for ids in host_groups]


def test_pcie_chip_without_pci_address_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A chip the cluster descriptor gives no address cannot be named for a
    worker, so discovery fails before opening any device."""
    opened = _pcie_cluster(monkeypatch, tmp_path, _umd_selection("2,3"))
    _cluster_descriptor(monkeypatch, tmp_path, _umd_selection("2"))

    with pytest.raises(RuntimeError, match=r"no PCI address for PCIe chips \[1\]"):
        dp_discovery._discover_standard_dp_visible_device_groups("P150", 2)
    assert opened == []


@pytest.mark.parametrize(("mesh_shape", "dp", "parent", "submesh"), PLACEMENTS)
def test_matches_tt_transformers_lane_mode_policy(mesh_shape, dp, parent, submesh):
    """Where tt-transformers is installed, its lane-mode split must agree."""
    mesh_utils = pytest.importorskip("tt_transformers.mesh_utils")
    plugin_mesh = FakeMesh(mesh_shape)
    model_mesh = FakeMesh(mesh_shape)

    create_dp_submeshes(plugin_mesh, dp)
    mesh_utils.create_submeshes(model_mesh, dp)

    assert plugin_mesh.calls == model_mesh.calls
