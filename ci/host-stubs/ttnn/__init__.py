# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Import-only ``ttnn`` stand-in so the unit suite can run without tt-metal.

``vllm_tt_plugin.entrypoints.platform_plugin`` gates the whole TT platform on
``import ttnn``, and several plugin modules import it at module scope, so a
stock runner cannot even collect the tests without something answering to that
name. This module supplies only the attributes the plugin touches.

Every entry point that would reach a device raises. The suite either monkey-
patches these (``patch("vllm_tt_plugin.worker.ttnn.get_arch_name", ...)``) or
never reaches them, so a test that starts depending on real hardware fails
loudly here instead of silently passing against a fake device.

Put this directory's parent on ``PYTHONPATH`` to use it:

    PYTHONPATH=ci/host-stubs pytest tests/ --ignore=tests/tt
"""

from enum import Enum

# Hardware-fixed tile dimension used by tt-metal operator shapes. The host CI
# only needs this constant for import-time provenance checks; device entry
# points below continue to require real hardware.
TILE_SIZE = 32


def _no_device(name):
    def raise_no_device(*args, **kwargs):
        raise RuntimeError(
            f"ttnn.{name} needs Tenstorrent hardware; this is the host-CI stub. "
            "Patch it in the test, or move the test to a TT environment."
        )

    raise_no_device.__name__ = name
    return raise_no_device


class FabricConfig(Enum):
    DISABLED = "disabled"
    FABRIC_1D = "fabric_1d"
    FABRIC_1D_RING = "fabric_1d_ring"
    FABRIC_2D = "fabric_2d"
    FABRIC_2D_TORUS_XY = "fabric_2d_torus_xy"
    CUSTOM = "custom"


class FabricReliabilityMode(Enum):
    STRICT_INIT = "strict_init"
    RELAXED_INIT = "relaxed_init"


class DispatchCoreAxis(Enum):
    ROW = "row"
    COL = "col"


class DispatchCoreConfig:
    def __init__(self, axis=None):
        self.axis = axis


class MeshShape:
    def __init__(self, *dims):
        self.dims = dims


class MeshDevice:
    """Annotation target only (``TTModelRunner.__init__``); never instantiated."""


class ClusterType(Enum):
    INVALID = "invalid"
    N150 = "n150"
    N300 = "n300"
    T3K = "t3k"
    GALAXY = "galaxy"
    P100 = "p100"
    P150 = "p150"
    P150_X2 = "p150_x2"
    P150_X4 = "p150_x4"
    BLACKHOLE_GALAXY = "blackhole_galaxy"
    SIMULATOR_WORMHOLE_B0 = "simulator_wormhole_b0"
    SIMULATOR_BLACKHOLE = "simulator_blackhole"


class _Cluster:
    ClusterType = ClusterType
    get_cluster_type = staticmethod(_no_device("cluster.get_cluster_type"))


cluster = _Cluster()

get_arch_name = _no_device("get_arch_name")
get_num_devices = _no_device("get_num_devices")
using_distributed_env = _no_device("using_distributed_env")
open_mesh_device = _no_device("open_mesh_device")
close_mesh_device = _no_device("close_mesh_device")
set_fabric_config = _no_device("set_fabric_config")
SetDefaultDevice = _no_device("SetDefaultDevice")
GetDefaultDevice = _no_device("GetDefaultDevice")
ReadDeviceProfiler = _no_device("ReadDeviceProfiler")
event_synchronize = _no_device("event_synchronize")
copy_host_to_device_tensor = _no_device("copy_host_to_device_tensor")
