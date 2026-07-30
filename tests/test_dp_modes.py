# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Unit tests for TT standard-DP routing and launch semantics."""

import pathlib
from types import SimpleNamespace

import pytest
import ttnn
from vllm.v1.core.sched import interface as sched_interface
from vllm.v1.engine import utils as engine_utils

from vllm_tt_plugin import platform as tt_platform
from vllm_tt_plugin import worker
from vllm_tt_plugin.launcher import parse_tt_mpi_params
from vllm_tt_plugin.platform import (
    TTPlatform,
    _resolve_standard_dp_visible_device_groups,
)
from vllm_tt_plugin.utils.dp_discovery import (
    _maybe_reorder_standard_dp_visible_device_groups,
)
from vllm_tt_plugin.worker import TTWorker, _resolve_mesh_grid

if not hasattr(sched_interface, "PauseState"):
    sched_interface.PauseState = type("PauseState", (), {})


class TestDPModes:
    @pytest.fixture
    def vllm_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            additional_config={},
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_size_local=1,
                data_parallel_rank=0,
                data_parallel_rank_local=0,
                data_parallel_index=0,
                data_parallel_external_lb=False,
                data_parallel_hybrid_lb=False,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                worker_cls="auto",
                data_parallel_backend="mp",
                nnodes=1,
                node_rank=0,
            ),
            model_config=SimpleNamespace(
                model="dummy",
                hf_config=SimpleNamespace(architectures=["DummyModel"]),
                max_logprobs=10,
                max_model_len=4,
                original_max_model_len=None,
                is_moe=False,
                get_sliding_window=lambda: None,
            ),
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                async_scheduling=False,
                scheduler_cls=None,
                max_num_seqs=4,
                max_num_batched_tokens=4,
                verify_max_model_len=lambda _max_model_len: None,
            ),
            speculative_config=None,
            lora_config=None,
            cache_config=SimpleNamespace(enable_prefix_caching=False),
        )

    @pytest.fixture
    def dummy_model_class(self) -> type:
        return type(
            "DummyModel",
            (),
            {"__module__": "models.tt_transformers.tt.generator_vllm"},
        )

    @staticmethod
    def register_dummy_model(
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
        visible_device_groups: list[str] | None = None,
    ) -> None:
        with monkeypatch.context() as m:
            m.setattr(
                "vllm_tt_plugin.platform.register_tt_models",
                lambda *args, **kwargs: None,
            )
            m.setattr(
                "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
                lambda _cfg: visible_device_groups,
            )
            m.setattr(
                "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
                lambda: ["TTDummyModel"],
            )
            m.setattr(
                "vllm.model_executor.model_loader.utils.get_model_architecture",
                lambda _model_config: (dummy_model_class, None),
            )

            TTPlatform.check_and_update_config(vllm_config)

    @pytest.mark.parametrize("original_max_model_len", [8192, -1])
    def test_check_and_update_config_preserves_explicit_max_model_len(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
        original_max_model_len: int,
    ) -> None:
        """An explicit --max-model-len is never rewritten by the TT platform.

        A numeric value must reach upstream's override-aware capacity check
        unchanged (so an oversized value fails loudly instead of being silently
        clamped), and an explicit -1 must stay -1.
        """
        vllm_config.model_config.original_max_model_len = original_max_model_len

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert vllm_config.model_config.original_max_model_len == original_max_model_len

    def test_check_and_update_config_preserves_omitted_max_model_len(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        """An omitted --max-model-len keeps upstream's HF-derived default."""
        vllm_config.model_config.original_max_model_len = None

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert vllm_config.model_config.original_max_model_len is None

    def test_update_max_model_len_syncs_worker_model_config(self) -> None:
        worker_instance = TTWorker.__new__(TTWorker)
        worker_instance.model_config = SimpleNamespace(max_model_len=262_144)

        TTWorker.update_max_model_len(worker_instance, 131_072)

        assert worker_instance.model_config.max_model_len == 131_072

    def test_upstream_dp_engine_core_is_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert (
            vllm_config.parallel_config.engine_core_cls
            == "vllm.v1.engine.core.EngineCore"
        )
        assert (
            vllm_config.parallel_config.engine_core_proc_cls
            == "vllm.v1.engine.core.EngineCoreProc"
        )
        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm.v1.engine.core.DPEngineCoreProc"
        )

    def test_lane_mode_keeps_upstream_dp_engine_core(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.additional_config = {"_tt_resolved_lane_count": 2}
        vllm_config.parallel_config.data_parallel_size = 1

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert (
            vllm_config.parallel_config.engine_core_cls
            == "vllm.v1.engine.core.EngineCore"
        )
        assert (
            vllm_config.parallel_config.engine_core_proc_cls
            == "vllm.v1.engine.core.EngineCoreProc"
        )
        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm.v1.engine.core.DPEngineCoreProc"
        )

    def test_collapsed_standard_dp_rank_warms_up_model(self) -> None:
        worker = TTWorker.__new__(TTWorker)
        worker.enable_model_warmup = True
        worker.parallel_config = SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank_local=7,
            data_parallel_index=7,
        )
        warmup_calls: list[str] = []
        worker.model_runner = SimpleNamespace(
            warmup_model=lambda: warmup_calls.append("warmup")
        )

        timings = TTWorker.compile_or_warm_up_model(worker)

        assert warmup_calls == ["warmup"]
        assert timings.language_model >= 0.0

    def test_visible_devices_override_full_machine_mesh_preset(self) -> None:
        assert _resolve_mesh_grid("TG", 1, "0") == (1, 1)
        assert _resolve_mesh_grid("TG", 8, "0,1,2,3,4,5,6,7") == (1, 8)
        assert _resolve_mesh_grid("P150x8", 8, "3") == (1, 1)

    def test_visible_devices_use_discovered_submesh_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(TTPlatform, "_standard_dp_mesh_grids", {"0,1,2,3": (2, 2)})

        assert _resolve_mesh_grid("TG", 4, "0,1,2,3") == (2, 2)

    def test_single_host_standard_dp_uses_upstream_launcher(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.parallel_config.data_parallel_size = 4

        self.register_dummy_model(
            monkeypatch,
            vllm_config,
            dummy_model_class,
            visible_device_groups=["24,25", "26,27", "3,2", "1,0"],
        )

        assert (
            vllm_config.parallel_config.engine_core_launcher_cls
            == "vllm.v1.engine.utils.CoreEngineLauncher"
        )
        assert TTPlatform._standard_dp_visible_device_groups == [
            "24,25",
            "26,27",
            "3,2",
            "1,0",
        ]

    def test_wh_galaxy_dp4_groups_follow_known_good_mesh_order(
        self,
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

    def test_rank_binding_keeps_tt_launcher(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.parallel_config.data_parallel_size = 4
        vllm_config.additional_config = {
            "tt": {"rank_binding": "/tmp/rank_binding.yaml"}
        }

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert (
            vllm_config.parallel_config.engine_core_launcher_cls
            == "vllm_tt_plugin.launcher.TTCoreEngineLauncher"
        )
        assert TTPlatform._standard_dp_visible_device_groups is None

    def test_standard_dp_discovery_target_uses_helper_module(self) -> None:
        """Keep the spawned discovery target outside the platform module."""
        assert (
            tt_platform._run_standard_dp_visible_device_group_discovery.__module__
            == "vllm_tt_plugin.utils.dp_discovery"
        )

    def test_tt_platform_set_device_uses_ttnn_default_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assigned_devices: list[object] = []

        monkeypatch.setattr(ttnn, "GetDefaultDevice", lambda: None, raising=False)
        monkeypatch.setattr(
            ttnn,
            "SetDefaultDevice",
            lambda device: assigned_devices.append(device),
            raising=False,
        )

        mesh_device = object()
        TTPlatform.set_device(None)
        TTPlatform.set_device(mesh_device)

        assert assigned_devices == [mesh_device]

    def test_init_device_tracks_mesh_as_worker_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mesh_device = SimpleNamespace(get_num_devices=lambda: 8)
        model_runner = SimpleNamespace()

        worker_instance = TTWorker.__new__(TTWorker)
        worker_instance.vllm_config = SimpleNamespace()
        worker_instance.parallel_config = SimpleNamespace(
            data_parallel_size=4,
            data_parallel_rank_local=0,
            data_parallel_index=0,
        )
        worker_instance.device_config = SimpleNamespace(device=None)
        worker_instance.trace_mode = "all"
        worker_instance.enable_model_warmup = True

        monkeypatch.setattr(TTPlatform, "check_and_update_config", lambda _cfg: None)
        monkeypatch.setattr(worker, "get_tt_config", lambda _cfg: {})
        monkeypatch.setattr(
            worker,
            "open_mesh_device",
            lambda _tt_config, _trace_mode, _local_dp_rank: mesh_device,
        )
        monkeypatch.setattr(worker, "TTModelRunner", lambda **_kwargs: model_runner)

        TTWorker.init_device(worker_instance)

        assert worker_instance.mesh_device is mesh_device
        assert worker_instance.device is mesh_device
        assert worker_instance.device_config.device is mesh_device
        assert worker_instance.model_runner is model_runner

    def test_standard_dp_discovery_timeout_terminates_subprocess(
        self,
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
        self,
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

    def test_standard_dp_visible_device_groups_feed_upstream_env_assignment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(engine_utils, "current_platform", TTPlatform)
        monkeypatch.setattr(
            TTPlatform,
            "_standard_dp_visible_device_groups",
            ["24,25,26,27,3,2,1,0", "16,17,18,19,20,21,22,23"],
        )

        assert (
            engine_utils.get_device_indices(
                TTPlatform.device_control_env_var,
                local_dp_rank=1,
                world_size=1,
            )
            == "16,17,18,19,20,21,22,23"
        )

    def test_legacy_gathered_override_is_ignored_by_platform(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.additional_config = {"tt": {"tt_data_parallel_size": 4}}
        vllm_config.parallel_config.data_parallel_size = 4

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert vllm_config.parallel_config.data_parallel_size == 4
        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm.v1.engine.core.DPEngineCoreProc"
        )
        assert (
            vllm_config.scheduler_config.scheduler_cls
            == "vllm_tt_plugin.scheduler.TTScheduler"
        )

    def test_standard_dp_rejects_moe_models(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.parallel_config.data_parallel_size = 4
        vllm_config.model_config.is_moe = True

        with monkeypatch.context() as m:
            m.setattr(
                "vllm_tt_plugin.platform.register_tt_models",
                lambda *args, **kwargs: None,
            )
            m.setattr(
                "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
                lambda _cfg: None,
            )
            m.setattr(
                "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
                lambda: ["TTDummyModel"],
            )
            m.setattr(
                "vllm.model_executor.model_loader.utils.get_model_architecture",
                lambda _model_config: (dummy_model_class, None),
            )

            with pytest.raises(
                ValueError,
                match="TT standard DP does not support MoE models yet",
            ):
                TTPlatform.check_and_update_config(vllm_config)

    def test_standard_dp_uses_all_device_ranks(
        self,
        tmp_path: pathlib.Path,
        vllm_config: SimpleNamespace,
    ) -> None:
        rank_binding = tmp_path / "rank_binding.json"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "    mesh_id: 0\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "0"\n'
            "\n"
            "  - rank: 1\n"
            "    mesh_id: 1\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "3"\n'
            "\n"
            "  - rank: 2\n"
            "    mesh_id: 2\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "1"\n'
            "\n"
            "  - rank: 3\n"
            "    mesh_id: 3\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "2"\n'
        )

        vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
        vllm_config.parallel_config.data_parallel_backend = "mp"
        vllm_config.parallel_config.data_parallel_size = 4

        parsed_rank_binding, non_device_dp_ranks = parse_tt_mpi_params(vllm_config)

        assert parsed_rank_binding == str(rank_binding)
        assert non_device_dp_ranks == set()

    def test_standard_dp_rejects_mismatched_mpi_world(
        self,
        tmp_path: pathlib.Path,
        vllm_config: SimpleNamespace,
    ) -> None:
        rank_binding = tmp_path / "rank_binding.json"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "    mesh_id: 0\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "0, 1"\n'
            "\n"
            "  - rank: 1\n"
            "    mesh_id: 1\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "2, 3"\n'
        )

        vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
        vllm_config.parallel_config.data_parallel_backend = "mp"
        vllm_config.parallel_config.data_parallel_size = 4

        with pytest.raises(
            RuntimeError,
            match="Standard DP mode requires one TT MPI rank per DP rank",
        ):
            parse_tt_mpi_params(vllm_config)

    def test_explicit_mpi_args_require_rank_binding(
        self,
        vllm_config: SimpleNamespace,
    ) -> None:
        vllm_config.additional_config = {"tt": {"mpi_args": "--host hostA"}}
        vllm_config.parallel_config.data_parallel_backend = "mp"
        vllm_config.parallel_config.data_parallel_size = 4

        with pytest.raises(
            RuntimeError,
            match="TT explicit MPI launch requires tt.rank_binding",
        ):
            parse_tt_mpi_params(vllm_config)

    def test_multinode_requires_rank_binding(
        self,
        vllm_config: SimpleNamespace,
    ) -> None:
        vllm_config.additional_config = {"tt": {}}
        vllm_config.parallel_config.data_parallel_backend = "mp"
        vllm_config.parallel_config.data_parallel_size = 4
        vllm_config.parallel_config.nnodes = 2

        with pytest.raises(
            RuntimeError,
            match="TT explicit MPI launch requires tt.rank_binding",
        ):
            parse_tt_mpi_params(vllm_config)

    def test_rank_binding_requires_visible_devices(
        self,
        tmp_path: pathlib.Path,
        vllm_config: SimpleNamespace,
    ) -> None:
        rank_binding = tmp_path / "rank_binding.json"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "    mesh_id: 0\n"
            "    env_overrides:\n"
            "      OTHER_ENV: foo\n"
            "  - rank: 1\n"
            "    mesh_id: 1\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "1"\n'
        )

        vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
        vllm_config.parallel_config.data_parallel_backend = "mp"
        vllm_config.parallel_config.data_parallel_size = 2

        with pytest.raises(RuntimeError, match="TT_VISIBLE_DEVICES"):
            parse_tt_mpi_params(vllm_config)

    def test_rank_binding_rejects_overlapping_visible_devices(
        self,
        tmp_path: pathlib.Path,
        vllm_config: SimpleNamespace,
    ) -> None:
        rank_binding = tmp_path / "rank_binding.json"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "    mesh_id: 0\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "0, 1"\n'
            "  - rank: 1\n"
            "    mesh_id: 1\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "1, 2"\n'
        )

        vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
        vllm_config.parallel_config.data_parallel_backend = "mp"
        vllm_config.parallel_config.data_parallel_size = 2

        with pytest.raises(
            RuntimeError,
            match="overlaps TT_VISIBLE_DEVICES assignments",
        ):
            parse_tt_mpi_params(vllm_config)

    def test_rank_binding_rejects_duplicate_rank_ids(
        self,
        tmp_path: pathlib.Path,
        vllm_config: SimpleNamespace,
    ) -> None:
        rank_binding = tmp_path / "rank_binding.json"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "    mesh_id: 0\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "0"\n'
            "  - rank: 0\n"
            "    mesh_id: 1\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "1"\n'
        )

        vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
        vllm_config.parallel_config.data_parallel_backend = "mp"
        vllm_config.parallel_config.data_parallel_size = 2

        with pytest.raises(RuntimeError, match="duplicate rank 0"):
            parse_tt_mpi_params(vllm_config)

    def test_legacy_gathered_override_is_ignored_by_launcher(
        self,
        tmp_path: pathlib.Path,
        vllm_config: SimpleNamespace,
    ) -> None:
        rank_binding = tmp_path / "rank_binding.json"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "    mesh_id: 0\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "0"\n'
            "\n"
            "  - rank: 1\n"
            "    mesh_id: 1\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "1"\n'
            "\n"
            "  - rank: 2\n"
            "    mesh_id: 2\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "2"\n'
            "\n"
            "  - rank: 3\n"
            "    mesh_id: 3\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "3"\n'
        )

        vllm_config.additional_config = {
            "tt": {
                "rank_binding": str(rank_binding),
                "tt_data_parallel_size": 4,
            }
        }
        vllm_config.parallel_config.data_parallel_backend = "mp"
        vllm_config.parallel_config.data_parallel_size = 4

        parsed_rank_binding, non_device_dp_ranks = parse_tt_mpi_params(vllm_config)

        assert parsed_rank_binding == str(rank_binding)
        assert non_device_dp_ranks == set()
