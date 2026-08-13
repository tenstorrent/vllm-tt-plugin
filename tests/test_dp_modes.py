# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Unit tests for how ``check_and_update_config`` routes a DP config.

Engine-core/launcher/scheduler selection and the worker's device lifecycle. The
``TT_VISIBLE_DEVICES`` group lifecycle lives in ``test_visible_devices.py``, and
the DP-to-lanes conversion in ``test_galaxy_dp_conversion.py``.
"""

from types import SimpleNamespace

import pytest
import ttnn
from vllm.v1.core.sched import interface as sched_interface

from vllm_tt_plugin import worker
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.worker import TTWorker

if not hasattr(sched_interface, "PauseState"):
    sched_interface.PauseState = type("PauseState", (), {})


class TestDPModes:
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

    @pytest.mark.parametrize("original_max_model_len", [8192, -1, None])
    def test_check_and_update_config_never_rewrites_max_model_len(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
        original_max_model_len: int | None,
    ) -> None:
        """The TT platform leaves vLLM's max_model_len policy alone.

        A numeric value must reach upstream's override-aware capacity check
        unchanged (so an oversized value fails loudly instead of being silently
        clamped), an explicit -1 must stay -1 so upstream auto-fits, and an
        omitted value must stay None so upstream keeps its HF-derived default.
        """
        vllm_config.model_config.original_max_model_len = original_max_model_len

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert vllm_config.model_config.original_max_model_len == original_max_model_len

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

    def test_single_host_standard_dp_uses_builtin_launch_path(
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

        assert not hasattr(
            vllm_config.parallel_config,
            "engine_core_launcher_cls",
        )
        assert TTPlatform._standard_dp_visible_device_groups == [
            "24,25",
            "26,27",
            "3,2",
            "1,0",
        ]

    @pytest.mark.parametrize(
        ("tt_config", "parallel_overrides"),
        [
            ({"rank_binding": "/tmp/rank_binding.yaml"}, {}),
            ({"mpi_args": "--host hostA"}, {}),
            ({}, {"nnodes": 2}),
            ({}, {"node_rank": 1}),
        ],
    )
    def test_explicit_tt_launch_fails_fast_on_vllm_024(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
        tt_config: dict,
        parallel_overrides: dict,
    ) -> None:
        vllm_config.parallel_config.data_parallel_size = 4
        vllm_config.additional_config = {"tt": tt_config}
        for key, value in parallel_overrides.items():
            setattr(vllm_config.parallel_config, key, value)

        with pytest.raises(RuntimeError, match="does not invoke.*CoreEngineLauncher"):
            self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

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

        parallel_config = SimpleNamespace(
            data_parallel_size=4,
            data_parallel_rank_local=0,
            data_parallel_index=0,
            assigned_physical_gpu_ids=None,
        )

        worker_instance = TTWorker.__new__(TTWorker)
        worker_instance.vllm_config = SimpleNamespace(
            additional_config={}, parallel_config=parallel_config
        )
        # WorkerBase aliases the field; keep the test's view identical.
        worker_instance.parallel_config = parallel_config
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

    def test_legacy_tt_dp_override_is_ignored_by_platform(
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
