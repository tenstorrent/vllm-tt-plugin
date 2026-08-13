# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Host tests for the standard-DP ``TT_VISIBLE_DEVICES`` group lifecycle.

Covers discovery of the per-rank device groups, their mesh-order fixup, the
mesh grid they resolve to, the handoff to ``assigned_physical_gpu_ids``, and
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
    _resolve_parent_mesh_grid,
)
from vllm_tt_plugin.worker import _bind_visible_devices_env, _resolve_mesh_grid

EVAR = TTPlatform.device_control_env_var


@pytest.fixture(autouse=True)
def _restore_visible_devices_env():
    """The function under test writes ``os.environ`` outside monkeypatch."""
    inherited = os.environ.get(EVAR)
    yield
    if inherited is None:
        os.environ.pop(EVAR, None)
    else:
        os.environ[EVAR] = inherited


def _discovered_groups_config(
    *visible_groups: str,
    local_dp_rank: int | None = 0,
    assigned_physical_gpu_ids: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={"_tt_standard_dp_visible_groups": list(visible_groups)},
        parallel_config=SimpleNamespace(
            assigned_physical_gpu_ids=assigned_physical_gpu_ids,
            data_parallel_rank_local=local_dp_rank,
        ),
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


def test_bh_galaxy_resolves_like_wh_galaxy() -> None:
    assert _resolve_mesh_grid("BH-Galaxy", 32, None) == (8, 4)
    assert _resolve_mesh_grid("BH-Galaxy", 32, None) == _resolve_mesh_grid(
        "TG", 32, None
    )
    assert _resolve_parent_mesh_grid("BH-Galaxy", 32) == _resolve_parent_mesh_grid(
        "TG", 32
    )


def test_visible_devices_use_discovered_submesh_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TTPlatform, "_standard_dp_mesh_grids", {"0,1,2,3": (2, 2)})

    assert _resolve_mesh_grid("TG", 4, "0,1,2,3") == (2, 2)


def test_stored_groups_suppress_rediscovery_on_config_reapply(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config: SimpleNamespace,
) -> None:
    """``TTWorker.init_device`` re-applies the hook with one group bound.

    Rediscovery there would resolve against the narrowed cluster: a Galaxy DP=2
    rank sees 16 chips, so ``create_submeshes`` would split them again and the
    real ``(2, 8)`` submesh shape would be replaced by two ``(1, 8)`` halves.
    """
    group_0 = ",".join(str(device_id) for device_id in range(16))
    group_1 = ",".join(str(device_id) for device_id in range(16, 32))

    vllm_config.parallel_config.data_parallel_size = 2
    vllm_config.additional_config = {
        "_tt_standard_dp_visible_groups": [group_0, group_1],
        "_tt_standard_dp_mesh_grids": {group_0: [2, 8], group_1: [2, 8]},
    }

    def _fail_discovery(_vllm_config):
        raise AssertionError("discovery must only run where the whole machine is seen")

    monkeypatch.setattr(
        "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
        _fail_discovery,
    )
    monkeypatch.setattr(
        "vllm_tt_plugin.platform.register_tt_models", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
        lambda: ["TTDummyModel"],
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.get_model_architecture",
        lambda _model_config: (type("DummyModel", (), {}), None),
    )

    TTPlatform.check_and_update_config(vllm_config)

    assert TTPlatform._standard_dp_visible_device_groups == [group_0, group_1]
    assert TTPlatform._standard_dp_mesh_grids == {group_0: (2, 8), group_1: (2, 8)}
    assert _resolve_mesh_grid("TG", 16, group_1) == (2, 8)


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
def test_matching_assigned_group_overrides_inherited_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
    inherited: str | None,
) -> None:
    """An inherited value belongs to the launching process, not to this rank."""
    if inherited is None:
        monkeypatch.delenv(EVAR, raising=False)
    else:
        monkeypatch.setenv(EVAR, inherited)

    _bind_visible_devices_env(
        _discovered_groups_config(
            "0,1,2,3",
            "4,5,6,7",
            local_dp_rank=1,
            assigned_physical_gpu_ids=["4,5,6,7"],
        )
    )

    assert os.environ[EVAR] == "4,5,6,7"


@pytest.mark.parametrize(
    ("local_dp_rank", "assigned_physical_gpu_ids"),
    [(0, ["0"]), (1, ["1"])],
)
def test_flat_assigned_device_ids_are_rejected_for_standard_dp(
    monkeypatch: pytest.MonkeyPatch,
    local_dp_rank: int,
    assigned_physical_gpu_ids: list[str],
) -> None:
    monkeypatch.setenv(EVAR, "0,1,2,3,4,5,6,7")

    with pytest.raises(
        RuntimeError,
        match=r"TT standard data parallelism does not support `--device-ids`",
    ):
        _bind_visible_devices_env(
            _discovered_groups_config(
                "0,1,2,3",
                "4,5,6,7",
                local_dp_rank=local_dp_rank,
                assigned_physical_gpu_ids=assigned_physical_gpu_ids,
            )
        )

    assert os.environ[EVAR] == "0,1,2,3,4,5,6,7"


def test_discovered_group_binds_when_upstream_left_ids_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback keys the stored groups by local DP rank, as init_device does."""
    monkeypatch.setenv(EVAR, "0,1,2,3,4,5,6,7")

    _bind_visible_devices_env(_discovered_groups_config("0,1,2,3", "4,5,6,7"))
    assert os.environ[EVAR] == "0,1,2,3"

    _bind_visible_devices_env(
        _discovered_groups_config("0,1,2,3", "4,5,6,7", local_dp_rank=1)
    )
    assert os.environ[EVAR] == "4,5,6,7"


def test_explicit_mpi_launch_keeps_inherited_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPI rank binding populates neither channel and owns its own env."""
    monkeypatch.setenv(EVAR, "5")

    # The fork vLLM the MPI launcher targets has no assigned_physical_gpu_ids.
    _bind_visible_devices_env(
        SimpleNamespace(
            additional_config={},
            parallel_config=SimpleNamespace(data_parallel_rank_local=0),
        )
    )

    assert os.environ[EVAR] == "5"


@pytest.mark.parametrize("local_dp_rank", [1, -1, None])
def test_rank_without_discovered_group_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
    local_dp_rank: int | None,
) -> None:
    monkeypatch.setenv(EVAR, "0,1,2,3")

    with pytest.raises(RuntimeError, match="No TT device group for local DP rank"):
        _bind_visible_devices_env(
            _discovered_groups_config("0,1,2,3", local_dp_rank=local_dp_rank)
        )

    assert os.environ[EVAR] == "0,1,2,3"


def test_empty_discovery_result_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero groups is a broken discovery, not a licence to keep the inherited value."""
    monkeypatch.setenv(EVAR, "0,1,2,3")

    with pytest.raises(RuntimeError, match="No TT device group for local DP rank"):
        _bind_visible_devices_env(_discovered_groups_config())

    assert os.environ[EVAR] == "0,1,2,3"
