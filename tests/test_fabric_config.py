# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_tt_plugin import worker

ttnn = worker.ttnn


@pytest.fixture
def fabric(monkeypatch):
    monkeypatch.setattr(
        ttnn.cluster, "get_cluster_type", lambda: ttnn.cluster.ClusterType.P150_X4
    )
    initialize = Mock()
    monkeypatch.setattr(ttnn, "set_fabric_config", initialize)
    return initialize


@pytest.mark.parametrize(
    "cluster, expected",
    [
        ("P150_X4", "FABRIC_1D"),
        ("GALAXY", "FABRIC_1D_RING"),
        ("BLACKHOLE_GALAXY", "FABRIC_2D_TORUS_XY"),
    ],
)
def test_models_without_fabric_capability_keep_hardware_defaults(
    monkeypatch, fabric, cluster, expected
):
    monkeypatch.setattr(
        ttnn.cluster,
        "get_cluster_type",
        lambda: getattr(ttnn.cluster.ClusterType, cluster),
    )
    worker.set_fabric(None, 4)
    assert fabric.call_args.kwargs == {
        "config": getattr(ttnn.FabricConfig, expected),
        "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
    }


def test_model_fabric_kwargs_pass_through_without_mutating_capability(fabric):
    router = object()
    model_config = {
        "config": ttnn.FabricConfig.FABRIC_1D_RING,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "num_planes": 2,
        "router_config": router,
    }
    original = model_config.copy()
    worker.set_fabric(None, 4, model_config)
    assert fabric.call_args.kwargs == original
    assert model_config == original


def test_launch_overrides_win_without_discarding_other_model_defaults(fabric):
    model_config = {
        "config": ttnn.FabricConfig.FABRIC_1D_RING,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "num_planes": 2,
    }
    original = model_config.copy()
    worker.set_fabric(
        {"fabric_config": "FABRIC_2D", "fabric_reliability_mode": "STRICT_INIT"},
        4,
        model_config,
    )
    assert fabric.call_args.kwargs == {
        "config": ttnn.FabricConfig.FABRIC_2D,
        "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
        "num_planes": 2,
    }
    assert model_config == original


def test_partial_model_config_inherits_hardware_defaults(fabric):
    worker.set_fabric(None, 4, {"num_planes": 2})
    assert fabric.call_args.kwargs == {
        "config": ttnn.FabricConfig.FABRIC_1D,
        "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
        "num_planes": 2,
    }


def test_single_device_does_not_initialize_fabric(fabric):
    worker.set_fabric(
        {"fabric_config": "FABRIC_2D"},
        1,
        {"config": ttnn.FabricConfig.FABRIC_1D_RING},
    )
    fabric.assert_not_called()


def test_ttnn_validation_failure_prevents_mesh_open(monkeypatch, fabric):
    fabric.side_effect = ValueError("invalid fabric configuration")
    monkeypatch.setattr(worker, "get_mesh_grid", lambda: (1, 4))
    open_mesh = Mock()
    monkeypatch.setattr(ttnn, "open_mesh_device", open_mesh)
    with pytest.raises(ValueError, match="invalid fabric configuration"):
        worker.open_mesh_device(None, "off", model_fabric_config={"num_planes": 0})
    open_mesh.assert_not_called()


def test_model_fabric_is_applied_before_mesh_open(monkeypatch, fabric):
    monkeypatch.setattr(worker, "get_mesh_grid", lambda: (1, 4))
    monkeypatch.setattr(worker, "get_dispatch_core_config", lambda _cfg: None)
    steps = []
    fabric.side_effect = lambda **kwargs: steps.append(("fabric", kwargs))
    mesh = SimpleNamespace(get_num_devices=lambda: 4)

    def open_mesh(*args, **kwargs):
        steps.append(("mesh", None))
        return mesh

    monkeypatch.setattr(ttnn, "open_mesh_device", open_mesh)
    result = worker.open_mesh_device(
        None, "off", model_fabric_config={"config": ttnn.FabricConfig.FABRIC_1D_RING}
    )
    assert result is mesh
    assert [step for step, _ in steps] == ["fabric", "mesh"]
    assert steps[0][1]["config"] == ttnn.FabricConfig.FABRIC_1D_RING
