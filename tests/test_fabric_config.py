# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""get_fabric_config must be able to name every fabric ttnn supports.

The 2D torus variants matter beyond completeness: a mesh whose graph
descriptor declares a torus cannot run a 2D collective under FABRIC_1D_RING
at all -- all_gather asserts fabric_is_2d for a non-degenerate mesh unless
given an explicit cluster_axis -- so a spec that could not name
FABRIC_2D_TORUS_XY could not describe a working torus.
"""

import pytest
import ttnn

from vllm_tt_plugin.worker import get_fabric_config


@pytest.fixture(autouse=True)
def _no_cluster_probe(monkeypatch):
    """get_fabric_config computes a hardware-derived default before applying
    the override, and that probe needs a device."""
    monkeypatch.setattr(
        ttnn.cluster, "get_cluster_type", lambda: ttnn.cluster.ClusterType.GALAXY
    )


@pytest.mark.parametrize(
    "name",
    [
        "DISABLED",
        "FABRIC_1D",
        "FABRIC_1D_RING",
        "FABRIC_2D",
        "FABRIC_2D_TORUS_X",
        "FABRIC_2D_TORUS_Y",
        "FABRIC_2D_TORUS_XY",
        "CUSTOM",
    ],
)
def test_every_ttnn_fabric_config_is_nameable(name):
    got = get_fabric_config({"fabric_config": name}, num_devices=32)
    assert got == getattr(ttnn.FabricConfig, name)


def test_torus_xy_is_two_dimensional():
    """The value the Blackhole Galaxy torus needs is distinct from the 1D ring
    it was previously pinned to."""
    torus = get_fabric_config({"fabric_config": "FABRIC_2D_TORUS_XY"}, num_devices=32)
    ring = get_fabric_config({"fabric_config": "FABRIC_1D_RING"}, num_devices=32)
    assert torus != ring


def test_unknown_fabric_config_is_rejected():
    with pytest.raises(AssertionError, match="Invalid fabric_config"):
        get_fabric_config({"fabric_config": "FABRIC_3D_HYPERTORUS"}, num_devices=32)


def test_single_device_ignores_explicit_request():
    assert get_fabric_config({"fabric_config": "FABRIC_2D_TORUS_XY"}, num_devices=1) is None
