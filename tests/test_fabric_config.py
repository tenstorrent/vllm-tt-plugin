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


def test_fabric_and_mesh_open_hold_the_host_device_lock(monkeypatch, fabric, tmp_path):
    monkeypatch.setattr(worker, "get_mesh_grid", lambda: (1, 2))
    monkeypatch.setattr(worker, "get_dispatch_core_config", lambda _cfg: None)
    monkeypatch.setattr(
        worker, "_device_open_lock_path", lambda: str(tmp_path / "open.lock")
    )
    steps = []
    real_flock = worker.fcntl.flock

    def flock(fd, op):
        steps.append("lock" if op == worker.fcntl.LOCK_EX else "unlock")
        real_flock(fd, op)

    monkeypatch.setattr(worker.fcntl, "flock", flock)
    fabric.side_effect = lambda **kwargs: steps.append("fabric")
    mesh = SimpleNamespace(get_num_devices=lambda: 2)

    def open_mesh(*args, **kwargs):
        steps.append("mesh")
        return mesh

    monkeypatch.setattr(ttnn, "open_mesh_device", open_mesh)
    assert worker.open_mesh_device(None, "off") is mesh
    assert steps == ["lock", "fabric", "mesh", "unlock"]

    steps.clear()
    monkeypatch.setattr(
        ttnn, "open_mesh_device", Mock(side_effect=RuntimeError("flush timeout"))
    )
    with pytest.raises(RuntimeError, match="flush timeout"):
        worker.open_mesh_device(None, "off")
    assert steps == ["lock", "fabric", "unlock"]


def test_class_level_fabric_declaration_applies_without_a_resolver():
    ring = {"config": ttnn.FabricConfig.FABRIC_1D_RING}
    model_class = SimpleNamespace(model_capabilities={"fabric_config": ring})

    # A class without the resolver never needs the served model's name.
    assert worker._model_fabric_config(model_class, SimpleNamespace()) is ring
    assert worker._model_fabric_config(SimpleNamespace(), SimpleNamespace()) is None


def test_a_class_serving_several_checkpoints_resolves_the_served_ones_fabric():
    """One registered class can serve checkpoints that need different fabrics, so
    it answers for the checkpoint being served, and that answer wins over a
    class-level declaration."""
    ring = {"config": ttnn.FabricConfig.FABRIC_1D_RING}
    asked = []

    class Dispatcher:
        model_capabilities = {"fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D}}

        @classmethod
        def fabric_config_for(cls, model_name):
            asked.append(model_name)
            return ring if model_name == "org/ring-model" else None

    ring_model = SimpleNamespace(model="org/ring-model")
    line_model = SimpleNamespace(model="org/line-model")
    assert worker._model_fabric_config(Dispatcher, ring_model) is ring
    assert worker._model_fabric_config(Dispatcher, line_model) is None
    assert asked == ["org/ring-model", "org/line-model"]
