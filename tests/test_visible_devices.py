# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Host tests for the standard-DP ``TT_VISIBLE_DEVICES`` group lifecycle.

Covers discovery of the per-rank device groups, their mesh-order fixup, the
mesh grid they resolve to, the handoff to vLLM's device-index assignment, and
the per-rank env binding the worker applies.
"""

import os
from types import SimpleNamespace

import pytest
import ttnn
from vllm.v1.engine import utils as engine_utils

from vllm_tt_plugin import platform as tt_platform
from vllm_tt_plugin.platform import (
    TTPlatform,
    _resolve_standard_dp_visible_device_groups,
)
from vllm_tt_plugin.utils.dp_discovery import (
    _maybe_reorder_standard_dp_visible_device_groups,
)
from vllm_tt_plugin.worker import _bind_visible_devices_env, _resolve_mesh_grid

EVAR = TTPlatform.device_control_env_var


def _discovered_groups_config(*visible_groups: str) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={"_tt_standard_dp_visible_groups": list(visible_groups)}
    )


def test_standard_dp_discovery_target_uses_helper_module() -> None:
    """Keep the spawned discovery target outside the platform module."""
    assert (
        tt_platform._run_standard_dp_visible_device_group_discovery.__module__
        == "vllm_tt_plugin.utils.dp_discovery"
    )


def test_standard_dp_discovery_timeout_terminates_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config: SimpleNamespace,
) -> None:
    vllm_config.parallel_config.data_parallel_size = 4

    class FakeConn:
        def poll(self, timeout: float) -> bool:
            return False

        def recv(self):
            raise AssertionError("recv should not be called after timeout")

        def close(self) -> None:
            return

    class FakeProc:
        def __init__(self) -> None:
            self.exitcode = None
            self.join_timeouts: list[float | None] = []
            self.terminated = False
            self.killed = False

        def start(self) -> None:
            return

        def join(self, timeout: float | None = None) -> None:
            self.join_timeouts.append(timeout)

        def is_alive(self) -> bool:
            return self.terminated and not self.killed

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    fake_parent_conn = FakeConn()
    fake_child_conn = SimpleNamespace(close=lambda: None)
    fake_proc = FakeProc()

    class FakeContext:
        def Pipe(self, duplex: bool = False):
            assert not duplex
            return fake_parent_conn, fake_child_conn

        def Process(self, **_kwargs):
            return fake_proc

    monkeypatch.setattr(
        tt_platform.multiprocessing, "get_context", lambda _mode: FakeContext()
    )

    with pytest.raises(RuntimeError, match="timed out after"):
        _resolve_standard_dp_visible_device_groups(vllm_config)

    assert fake_proc.terminated
    assert fake_proc.killed


def test_standard_dp_discovery_join_timeout_terminates_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config: SimpleNamespace,
) -> None:
    vllm_config.parallel_config.data_parallel_size = 4

    class FakeConn:
        def poll(self, timeout: float) -> bool:
            return True

        def recv(self):
            return ("ok", ["0", "1", "2", "3"])

        def close(self) -> None:
            return

    class FakeProc:
        def __init__(self) -> None:
            self.exitcode = None
            self.join_timeouts: list[float | None] = []
            self.terminated = False
            self.killed = False

        def start(self) -> None:
            return

        def join(self, timeout: float | None = None) -> None:
            self.join_timeouts.append(timeout)

        def is_alive(self) -> bool:
            return self.terminated and not self.killed or not self.terminated

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    fake_parent_conn = FakeConn()
    fake_child_conn = SimpleNamespace(close=lambda: None)
    fake_proc = FakeProc()

    class FakeContext:
        def Pipe(self, duplex: bool = False):
            assert not duplex
            return fake_parent_conn, fake_child_conn

        def Process(self, **_kwargs):
            return fake_proc

    monkeypatch.setattr(
        tt_platform.multiprocessing, "get_context", lambda _mode: FakeContext()
    )

    with pytest.raises(
        RuntimeError,
        match="did not exit after returning device groups",
    ):
        _resolve_standard_dp_visible_device_groups(vllm_config)

    assert fake_proc.terminated
    assert fake_proc.killed


def test_wh_galaxy_dp4_groups_follow_known_good_mesh_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ttnn.cluster,
        "get_cluster_type",
        lambda: ttnn.cluster.ClusterType.GALAXY,
    )

    groups = [
        ("0,1,2,3,4,5,6,7", (1, 8)),
        ("8,9,10,11,12,13,14,15", (1, 8)),
        ("16,17,18,19,20,21,22,23", (1, 8)),
        ("24,25,26,27,28,29,30,31", (1, 8)),
    ]

    assert _maybe_reorder_standard_dp_visible_device_groups(
        groups,
        (4, 8),
        4,
    ) == [
        ("0,1,2,3,4,5,6,7", (1, 8)),
        ("16,17,18,19,20,21,22,23", (1, 8)),
        ("24,25,26,27,28,29,30,31", (1, 8)),
        ("8,9,10,11,12,13,14,15", (1, 8)),
    ]


def test_visible_devices_override_full_machine_mesh_preset() -> None:
    assert _resolve_mesh_grid("TG", 1, "0") == (1, 1)
    assert _resolve_mesh_grid("TG", 8, "0,1,2,3,4,5,6,7") == (1, 8)
    assert _resolve_mesh_grid("P150x8", 8, "3") == (1, 1)


def test_visible_devices_use_discovered_submesh_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TTPlatform, "_standard_dp_mesh_grids", {"0,1,2,3": (2, 2)})

    assert _resolve_mesh_grid("TG", 4, "0,1,2,3") == (2, 2)


def test_standard_dp_visible_device_groups_feed_upstream_gpu_id_assignment(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config: SimpleNamespace,
) -> None:
    monkeypatch.setattr(engine_utils, "current_platform", TTPlatform)
    monkeypatch.setattr(
        TTPlatform,
        "_standard_dp_visible_device_groups",
        ["24,25,26,27,3,2,1,0", "16,17,18,19,20,21,22,23"],
    )

    engine_utils.set_assigned_physical_gpu_ids_for_dp_rank(vllm_config, local_dp_rank=1)

    assert vllm_config.parallel_config.assigned_physical_gpu_ids == [
        "16,17,18,19,20,21,22,23"
    ]


@pytest.mark.parametrize("inherited", [None, "", "0,1,2,3,4,5,6,7", "9,9"])
def test_discovered_group_overrides_inherited_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
    inherited: str | None,
) -> None:
    """An inherited value belongs to the launching process, not to this rank."""
    if inherited is None:
        monkeypatch.delenv(EVAR, raising=False)
    else:
        monkeypatch.setenv(EVAR, inherited)

    discovered = _discovered_groups_config("0,1,2,3", "4,5,6,7")

    _bind_visible_devices_env(discovered, SimpleNamespace(data_parallel_index=0))
    assert os.environ[EVAR] == "0,1,2,3"

    _bind_visible_devices_env(discovered, SimpleNamespace(data_parallel_index=1))
    assert os.environ[EVAR] == "4,5,6,7"


def test_explicit_mpi_launch_keeps_inherited_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPI rank binding stores no groups and owns its own env."""
    monkeypatch.setenv(EVAR, "5")

    _bind_visible_devices_env(
        SimpleNamespace(additional_config={}),
        SimpleNamespace(data_parallel_index=0),
    )

    assert os.environ[EVAR] == "5"


def test_rank_without_discovered_group_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EVAR, "0,1,2,3")

    with pytest.raises(RuntimeError, match="no discovered device group"):
        _bind_visible_devices_env(
            _discovered_groups_config("0,1,2,3"),
            SimpleNamespace(data_parallel_index=1),
        )
