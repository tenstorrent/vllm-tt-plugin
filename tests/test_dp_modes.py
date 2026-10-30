# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Unit tests for how ``check_and_update_config`` routes a DP config.

Engine-core/launcher/scheduler selection and the worker's device lifecycle. The
``TT_VISIBLE_DEVICES`` group lifecycle lives in ``test_visible_devices.py``, and
the DP-to-lanes conversion in ``test_galaxy_dp_conversion.py``.
"""

import asyncio
import itertools
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import ttnn
from vllm.utils.torch_utils import OMP_NUM_THREADS_SET_BY_VLLM
from vllm.v1.core.sched import interface as sched_interface
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import core_client

from vllm_tt_plugin import worker
from vllm_tt_plugin.config import (
    TT_LANE_SCHEDULER_CLS,
    TT_SCHEDULER_CLS,
    TT_WORKER_CLS,
    get_tt_supports_stage_inputs,
    get_tt_supports_stage_outputs,
    resolve_tt_scheduler_cls,
)
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.scheduler import TTScheduler
from vllm_tt_plugin.worker import TTWorker

if not hasattr(sched_interface, "PauseState"):
    sched_interface.PauseState = type("PauseState", (), {})


# Same class layout as vLLM-Omni's stage schedulers, which write themselves into
# every stage's engine args; vLLM-Omni is not a test dependency.
class OmniARScheduler(Scheduler):
    pass


class OmniARAsyncScheduler(OmniARScheduler, AsyncScheduler):
    pass


class OmniGenerationScheduler(Scheduler):
    pass


class TTStageARScheduler(OmniARAsyncScheduler, TTScheduler):
    pass


class TTStageGenerationScheduler(OmniGenerationScheduler):
    pass


class TTStageARWorker(TTWorker):
    tt_scheduler_cls = TTStageARScheduler


class TTRefusingStageARScheduler(TTStageARScheduler):
    seen: list = []

    @classmethod
    def validate_tt_config(cls, vllm_config) -> None:
        cls.seen.append(vllm_config.scheduler_config.scheduler_cls)
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("this stage runs one sequence")


class TTRefusingStageARWorker(TTWorker):
    tt_scheduler_cls = TTRefusingStageARScheduler


class TTStageGenerationWorker(TTWorker):
    tt_scheduler_cls = TTStageGenerationScheduler


class TTDefaultSchedulerWorker(TTWorker):
    pass


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

    def test_image_normalization_stays_in_the_cpu_processor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        # vLLM >= 0.28 enables this for upstream Qwen2/2.5-VL.
        vllm_config.model_config.multimodal_config = SimpleNamespace(
            mm_device_do_normalize=True
        )

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert (
            vllm_config.model_config.multimodal_config.mm_device_do_normalize is False
        )

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

    def test_engine_core_classes_are_left_to_upstream(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        # ``ParallelConfig`` accepts arbitrary attribute writes, so naming an
        # engine-core class upstream does not define reads as configuration but
        # does nothing. The plugin selects a worker and a scheduler; engine-core
        # selection is upstream's.
        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert not hasattr(vllm_config.parallel_config, "engine_core_cls")
        assert not hasattr(vllm_config.parallel_config, "engine_core_proc_cls")
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

        assert not hasattr(vllm_config.parallel_config, "engine_core_cls")
        assert not hasattr(vllm_config.parallel_config, "engine_core_proc_cls")
        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm.v1.engine.core.DPEngineCoreProc"
        )

    def test_check_and_update_config_stores_stage_input_support(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        dummy_model_class.model_capabilities = {"supports_stage_inputs": True}

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert get_tt_supports_stage_inputs(vllm_config) is True

    def test_check_and_update_config_refuses_stage_inputs_in_lane_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.additional_config = {"_tt_resolved_lane_count": 2}
        vllm_config.parallel_config.data_parallel_size = 1
        dummy_model_class.model_capabilities = {"supports_stage_inputs": True}

        with pytest.raises(ValueError, match="lane mode does not pass stage inputs"):
            self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

    def test_check_and_update_config_stores_stage_output_support(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        dummy_model_class.model_capabilities = {"supports_stage_outputs": True}
        vllm_config.scheduler_config.max_num_seqs = 1

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert get_tt_supports_stage_outputs(vllm_config) is True

    def test_check_and_update_config_refuses_stage_outputs_in_lane_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.additional_config = {"_tt_resolved_lane_count": 2}
        vllm_config.parallel_config.data_parallel_size = 1
        dummy_model_class.model_capabilities = {"supports_stage_outputs": True}

        with pytest.raises(ValueError, match="needs one-token front-packed"):
            self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

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

    @pytest.mark.parametrize("set_by_vllm", [True, False])
    def test_warmup_drops_vllm_sized_worker_threads(
        self, monkeypatch: pytest.MonkeyPatch, set_by_vllm: bool
    ) -> None:
        worker = TTWorker.__new__(TTWorker)
        worker.enable_model_warmup = False
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        if set_by_vllm:
            monkeypatch.setenv(OMP_NUM_THREADS_SET_BY_VLLM, "1")
        else:
            monkeypatch.delenv(OMP_NUM_THREADS_SET_BY_VLLM, raising=False)
        threads = [4]

        def set_num_threads(n: int) -> None:
            threads[0] = n

        monkeypatch.setattr(torch, "get_num_threads", lambda: threads[0])
        monkeypatch.setattr(torch, "set_num_threads", set_num_threads)

        TTWorker.compile_or_warm_up_model(worker)

        assert threads == [1 if set_by_vllm else 4]

    def test_single_host_standard_dp_leaves_the_launcher_to_upstream(
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

        assert not hasattr(vllm_config.parallel_config, "engine_core_launcher_cls")
        assert TTPlatform._standard_dp_visible_device_groups == [
            "24,25",
            "26,27",
            "3,2",
            "1,0",
        ]
        # Request routing stays upstream's balancer, with the prefix-cache
        # routing patch in front of it.
        assert hasattr(core_client, "_tt_original_get_core_engine_for_request")

    @pytest.mark.parametrize(
        ("tt_config", "parallel_overrides"),
        [
            ({"rank_binding": "/tmp/rank_binding.yaml"}, {}),
            ({"mpi_args": "--host hostA"}, {}),
            ({}, {"nnodes": 2}),
            ({}, {"node_rank": 1}),
        ],
    )
    def test_explicit_tt_launch_is_rejected_without_the_launcher_hook(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
        tt_config: dict,
        parallel_overrides: dict,
    ) -> None:
        # Upstream vLLM defines no ``engine_core_launcher_cls``, and writing one
        # anyway is silently ignored, which would quietly single-host a run the
        # user asked to spread over several nodes. Every explicit-MPI trigger
        # (rank_binding, mpi_args, nnodes > 1, node_rank > 0) must fail fast.
        vllm_config.parallel_config.data_parallel_size = 4
        vllm_config.additional_config = {"tt": tt_config}
        for key, value in parallel_overrides.items():
            setattr(vllm_config.parallel_config, key, value)

        with pytest.raises(NotImplementedError, match="engine-core launcher hook"):
            self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

    def test_rank_binding_selects_the_tt_launcher_when_the_hook_exists(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.parallel_config.data_parallel_size = 4
        vllm_config.parallel_config.engine_core_launcher_cls = "auto"
        vllm_config.additional_config = {
            "tt": {"rank_binding": "/tmp/rank_binding.yaml"}
        }

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert (
            vllm_config.parallel_config.engine_core_launcher_cls
            == "vllm_tt_plugin.launcher.TTCoreEngineLauncher"
        )
        assert TTPlatform._standard_dp_visible_device_groups is None

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

    @staticmethod
    def _init_device_worker(
        monkeypatch: pytest.MonkeyPatch,
        worker_cls: type[TTWorker] = TTWorker,
        model_capabilities: dict | None = None,
    ) -> tuple[TTWorker, SimpleNamespace, list[str]]:
        """A worker whose init_device steps run against test doubles.

        Returns the worker, its mesh double, and the list each step appends to.
        """
        mesh_device = SimpleNamespace(get_num_devices=lambda: 8)
        parallel_config = SimpleNamespace(
            data_parallel_size=4,
            data_parallel_rank_local=0,
            data_parallel_index=0,
            assigned_physical_gpu_ids=None,
        )

        worker_instance = worker_cls.__new__(worker_cls)
        worker_instance.vllm_config = SimpleNamespace(
            additional_config={}, parallel_config=parallel_config
        )
        # WorkerBase aliases the field; keep the test's view identical.
        worker_instance.parallel_config = parallel_config
        worker_instance.device_config = SimpleNamespace(device=None)
        worker_instance.model_config = SimpleNamespace()
        worker_instance.trace_mode = "all"
        worker_instance.enable_model_warmup = True
        worker_instance._num_tt_blocks = None

        monkeypatch.setattr(TTPlatform, "check_and_update_config", lambda _cfg: None)
        monkeypatch.setattr(worker, "get_tt_config", lambda _cfg: {})
        model_class = SimpleNamespace()
        if model_capabilities is not None:
            model_class.model_capabilities = model_capabilities
        monkeypatch.setattr(
            worker, "get_model_architecture", lambda _cfg: (model_class, "model")
        )

        steps: list[str] = []
        fabric_config = (model_capabilities or {}).get("fabric_config")

        def open_mesh(_tt_config, _trace_mode, _local_dp_rank, *, model_fabric_config):
            assert model_fabric_config is fabric_config
            steps.append("mesh")
            return mesh_device

        def size_pool(_cfg, _num_devices):
            steps.append("size")
            return 7

        def fit_max_model_len(_cfg, _num_tt_blocks):
            steps.append("fit")

        monkeypatch.setattr(worker, "open_mesh_device", open_mesh)
        monkeypatch.setattr(worker, "get_num_available_blocks_tt", size_pool)
        monkeypatch.setattr(
            worker, "_fit_block_output_max_model_len", fit_max_model_len
        )
        return worker_instance, mesh_device, steps

    @pytest.mark.parametrize("declares_fabric", [False, True])
    def test_init_device_tracks_mesh_as_worker_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
        declares_fabric: bool,
    ) -> None:
        model_capabilities = (
            {"fabric_config": {"config": worker.ttnn.FabricConfig.FABRIC_1D_RING}}
            if declares_fabric
            else None
        )
        worker_instance, mesh_device, steps = self._init_device_worker(
            monkeypatch, model_capabilities=model_capabilities
        )
        model_runner = SimpleNamespace()

        def build_runner(**kwargs):
            assert kwargs["mesh_device"] is mesh_device
            steps.append("runner")
            return model_runner

        monkeypatch.setattr(TTWorker, "model_runner_cls", staticmethod(build_runner))

        try:
            TTWorker.init_device(worker_instance)

            assert worker_instance.mesh_device is mesh_device
            assert worker_instance.device is mesh_device
            assert worker_instance.device_config.device is mesh_device
            assert worker_instance.model_runner is model_runner
            # The KV pool is sized and --max-model-len settled before the runner
            # exists, so the model sees the fitted length when load_model runs.
            assert steps == ["mesh", "size", "fit", "runner"]
            assert worker_instance._num_tt_blocks == 7
        finally:
            # The test double cannot be passed to TT device cleanup.
            worker_instance.mesh_device = None

    def test_init_device_builds_the_subclass_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class StageRunner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class StageWorker(TTWorker):
            model_runner_cls = StageRunner

        worker_instance, mesh_device, steps = self._init_device_worker(
            monkeypatch, worker_cls=StageWorker
        )

        try:
            worker_instance.init_device()

            assert type(worker_instance.model_runner) is StageRunner
            assert worker_instance.model_runner.kwargs["mesh_device"] is mesh_device
            assert worker_instance.model_runner.kwargs["num_devices"] == 8
            assert steps == ["mesh", "size", "fit"]
        finally:
            worker_instance.mesh_device = None

    def test_init_device_skips_pool_sizing_when_subclass_overrides_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class NoKVWorker(TTWorker):
            def _init_kv_pool(self) -> None:
                pass

        worker_instance, mesh_device, steps = self._init_device_worker(
            monkeypatch, worker_cls=NoKVWorker
        )
        model_runner = SimpleNamespace()
        monkeypatch.setattr(
            NoKVWorker, "model_runner_cls", staticmethod(lambda **_kw: model_runner)
        )

        try:
            worker_instance.init_device()

            assert worker_instance.mesh_device is mesh_device
            assert worker_instance.model_runner is model_runner
            assert steps == ["mesh"]
            assert worker_instance._num_tt_blocks is None
        finally:
            worker_instance.mesh_device = None

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

    @pytest.mark.parametrize(
        ("requested", "worker_cls", "expected"),
        [
            (None, TT_WORKER_CLS, TT_SCHEDULER_CLS),
            (Scheduler, TT_WORKER_CLS, TT_SCHEDULER_CLS),
            (OmniARScheduler, TTStageARWorker, TTStageARScheduler),
            (OmniARAsyncScheduler, TTStageARWorker, TTStageARScheduler),
            (
                OmniGenerationScheduler,
                TTStageGenerationWorker,
                TTStageGenerationScheduler,
            ),
        ],
    )
    def test_scheduler_pairing_selects_the_workers_tt_scheduler(
        self, requested: type | None, worker_cls: str | type, expected: str | type
    ) -> None:
        assert resolve_tt_scheduler_cls(requested, worker_cls, False) == expected
        # The worker re-applies the config to the scheduler chosen above.
        assert resolve_tt_scheduler_cls(expected, worker_cls, False) == expected

    @pytest.mark.parametrize(
        ("requested", "worker_cls"),
        [
            (OmniARAsyncScheduler, TT_WORKER_CLS),
            (OmniGenerationScheduler, TT_WORKER_CLS),
            (OmniGenerationScheduler, TTStageARWorker),
            (OmniARScheduler, TTStageGenerationWorker),
            (TTStageARScheduler, TTStageGenerationWorker),
        ],
    )
    def test_scheduler_pairing_rejects_a_mismatched_pair(
        self, requested: type, worker_cls: str | type
    ) -> None:
        with pytest.raises(ValueError, match="cannot run on TT worker"):
            resolve_tt_scheduler_cls(requested, worker_cls, False)

    def test_scheduler_pairing_rejects_a_worker_without_a_tt_scheduler(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="declares no tt_scheduler_cls"):
            resolve_tt_scheduler_cls(OmniARScheduler, Scheduler, False)

    @pytest.mark.parametrize(
        ("requested", "worker_cls"),
        [
            (OmniARAsyncScheduler, TTStageARWorker),
            (OmniGenerationScheduler, TTStageGenerationWorker),
            (OmniARAsyncScheduler, TT_WORKER_CLS),
        ],
    )
    def test_scheduler_pairing_rejects_lane_mode_with_a_stage(
        self, requested: type, worker_cls: str | type
    ) -> None:
        with pytest.raises(ValueError, match="TT lane mode runs TTLaneCoordinator"):
            resolve_tt_scheduler_cls(requested, worker_cls, True)

    @pytest.mark.parametrize("requested", [None, Scheduler, TT_LANE_SCHEDULER_CLS])
    @pytest.mark.parametrize(
        "worker_cls",
        [TT_WORKER_CLS, f"{__name__}.TTDefaultSchedulerWorker"],
    )
    def test_scheduler_pairing_keeps_the_lane_coordinator(
        self, requested: str | type | None, worker_cls: str
    ) -> None:
        assert (
            resolve_tt_scheduler_cls(requested, worker_cls, True)
            == TT_LANE_SCHEDULER_CLS
        )

    @pytest.mark.parametrize(
        ("requested", "worker_cls", "expected"),
        [
            (None, "auto", TT_SCHEDULER_CLS),
            (OmniARAsyncScheduler, TTStageARWorker, TTStageARScheduler),
            (
                OmniGenerationScheduler,
                TTStageGenerationWorker,
                TTStageGenerationScheduler,
            ),
        ],
    )
    def test_check_and_update_config_keeps_the_pair_on_worker_reapply(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
        requested: type | None,
        worker_cls: str | type,
        expected: str | type,
    ) -> None:
        vllm_config.scheduler_config.scheduler_cls = requested
        vllm_config.parallel_config.worker_cls = worker_cls

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)
        assert vllm_config.scheduler_config.scheduler_cls == expected

        # TTWorker.init_device re-applies the hook in the worker process.
        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)
        assert vllm_config.scheduler_config.scheduler_cls == expected

    def test_the_paired_scheduler_validates_the_resolved_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        monkeypatch.setattr(TTRefusingStageARScheduler, "seen", [])
        vllm_config.scheduler_config.scheduler_cls = OmniARAsyncScheduler
        vllm_config.parallel_config.worker_cls = TTRefusingStageARWorker
        vllm_config.scheduler_config.max_num_seqs = 1

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert TTRefusingStageARScheduler.seen == [TTRefusingStageARScheduler]

        vllm_config.scheduler_config.max_num_seqs = 2
        with pytest.raises(ValueError, match="this stage runs one sequence"):
            self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

    def test_check_and_update_config_rejects_a_mismatched_pair(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.scheduler_config.scheduler_cls = OmniGenerationScheduler
        vllm_config.parallel_config.worker_cls = TTStageARWorker

        with pytest.raises(ValueError, match="cannot run on TT worker"):
            self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)


# ---------------------------------------------------------------------------
# Prefix-cache DP engine routing
# ---------------------------------------------------------------------------

BLOCK = 16
_OWN_TAILS = itertools.count(1)


def _dp_client(num_engines=3, max_num_seqs=32, routing=True, counts=None):
    """A ``DPLBAsyncMPClient`` stand-in carrying what its balancer reads."""
    engines = [f"engine{i}".encode() for i in range(num_engines)]
    return SimpleNamespace(
        core_engines=engines,
        lb_engines=[list(c) for c in (counts or [[0, 0, 0.0]] * num_engines)],
        engine_inflight=Counter(),
        reqs_in_flight={},
        client_count=1,
        client_index=0,
        eng_start_index=0,
        vllm_config=SimpleNamespace(
            additional_config={"tt": {"dp_prefix_cache_routing": routing}},
            cache_config=SimpleNamespace(block_size=BLOCK),
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        ),
    )


def _dp_request(request_id, shared_blocks=0, own_blocks=1, data_parallel_rank=None):
    """A prompt of ``shared_blocks`` blocks every such prompt shares, then its own."""
    tail = next(_OWN_TAILS) * 1000
    prompt = list(range(shared_blocks * BLOCK)) + [
        10_000 + tail + i for i in range(own_blocks * BLOCK)
    ]
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt,
        mm_features=None,
        pooling_params=None,
        data_parallel_rank=data_parallel_rank,
    )


def _route(client, request):
    """Route through the patched balancer; returns the engine index."""
    from vllm_tt_plugin.platform import _install_tt_dp_prefix_cache_routing_patch

    _install_tt_dp_prefix_cache_routing_patch()
    engine = core_client.DPLBAsyncMPClient.get_core_engine_for_request(client, request)
    return client.core_engines.index(engine)


def _snapshot(client, *counts):
    """The coordinator's stats update: ``[waiting, running, kv_usage]`` per engine."""
    client.lb_engines = [list(c) for c in counts]


def _finish(client, *request_ids):
    """What ``process_engine_outputs`` does when these requests finish."""
    for request_id in request_ids:
        client.engine_inflight[client.reqs_in_flight.pop(request_id)] -= 1


def test_dp_request_is_routed_to_the_engine_that_was_sent_its_prefix():
    client = _dp_client(counts=[[1, 0, 0.0], [1, 0, 0.0], [0, 0, 0.0]])
    # The balancer sends the document to the least-loaded engine, 2.
    assert _route(client, _dp_request("doc", shared_blocks=4)) == 2
    _snapshot(client, [0, 0, 0.0], [0, 0, 0.0], [0, 1, 0.0])

    follower = _dp_request("q", shared_blocks=4)

    assert _route(client, follower) == 2
    assert follower.data_parallel_rank == 2
    # Queued there in the balancer's counts, as its own path records.
    assert client.lb_engines[2][0] == 1


def test_dp_the_engine_sent_the_longest_prefix_wins():
    client = _dp_client()
    _route(client, _dp_request("short", shared_blocks=2, data_parallel_rank=0))
    _route(client, _dp_request("long", shared_blocks=6, data_parallel_rank=1))
    _snapshot(client, [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0])

    assert _route(client, _dp_request("q", shared_blocks=6)) == 1


def test_dp_uncached_prompt_is_left_to_the_load_balancer():
    client = _dp_client(counts=[[0, 2, 0.0], [0, 1, 0.0], [0, 3, 0.0]])
    _route(client, _dp_request("doc", shared_blocks=4, data_parallel_rank=2))

    cold = _dp_request("cold")

    assert _route(client, cold) == 1
    assert cold.data_parallel_rank is None


def test_dp_a_request_without_prompt_tokens_is_left_to_the_load_balancer():
    client = _dp_client(counts=[[0, 2, 0.0], [0, 1, 0.0], [0, 3, 0.0]])
    embeds = _dp_request("embeds")
    embeds.prompt_token_ids = None

    assert _route(client, embeds) == 1
    assert embeds.data_parallel_rank is None


def test_dp_a_burst_of_followers_of_a_long_prefix_all_join_its_engine():
    # 20 questions on one 3968-token document: each queues only its 128 own
    # tokens, far below the prefix they reuse, so none spills.
    client = _dp_client()
    _route(client, _dp_request("doc", 248, own_blocks=0, data_parallel_rank=0))
    _snapshot(client, [0, 1, 0.0], [0, 0, 0.0], [0, 0, 0.0])

    picks = [
        _route(client, _dp_request(f"q{i}", shared_blocks=248, own_blocks=8))
        for i in range(20)
    ]

    assert picks == [0] * 20


def test_dp_followers_of_a_short_prefix_spill_once_the_queue_outweighs_the_hit():
    # A 256-token shared system prompt in 1024-token prompts: the first follower
    # queues 768 uncached tokens on engine 0, more than the 256 the next one
    # would save, so the next one takes the least-loaded engine.
    client = _dp_client()
    _route(client, _dp_request("seed", 16, own_blocks=48, data_parallel_rank=0))
    _snapshot(client, [0, 1, 0.0], [0, 0, 0.0], [0, 0, 0.0])

    picks = [
        _route(client, _dp_request(f"q{i}", shared_blocks=16, own_blocks=48))
        for i in range(3)
    ]

    assert picks == [0, 1, 2]


def test_dp_finished_requests_drop_their_queued_prefill_record():
    client = _dp_client()
    _route(client, _dp_request("seed", 16, own_blocks=48, data_parallel_rank=0))
    _snapshot(client, [0, 1, 0.0], [0, 0, 0.0], [0, 0, 0.0])
    assert _route(client, _dp_request("q0", shared_blocks=16, own_blocks=48)) == 0
    _finish(client, "seed", "q0")
    # A stale snapshot still counts one waiting request on engine 0, but
    # nothing in flight there queues any prefill.
    _snapshot(client, [1, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0])

    assert _route(client, _dp_request("q1", shared_blocks=16, own_blocks=48)) == 0


def test_dp_a_full_cached_engine_loses_to_an_engine_with_a_free_slot():
    client = _dp_client(max_num_seqs=2)
    _route(client, _dp_request("doc", shared_blocks=4, data_parallel_rank=0))
    _snapshot(client, [0, 2, 0.0], [0, 1, 0.0], [0, 2, 0.0])
    assert _route(client, _dp_request("q", shared_blocks=4)) == 1
    # Every engine full: the cached engine still wins (the wait is the same).
    client = _dp_client(max_num_seqs=2)
    _route(client, _dp_request("doc", shared_blocks=4, data_parallel_rank=0))
    _snapshot(client, [0, 2, 0.0], [0, 2, 0.0], [0, 2, 0.0])
    assert _route(client, _dp_request("q", shared_blocks=4)) == 0


def test_dp_engines_sent_the_same_prefix_resolve_on_load():
    client = _dp_client()
    for engine in (0, 1):
        _route(client, _dp_request(f"doc{engine}", 4, data_parallel_rank=engine))
    _snapshot(client, [1, 1, 0.0], [0, 1, 0.0], [0, 0, 0.0])

    assert _route(client, _dp_request("q", shared_blocks=4)) == 1


def test_dp_an_explicit_rank_is_honored_and_recorded():
    client = _dp_client()
    _route(client, _dp_request("doc", shared_blocks=4, data_parallel_rank=2))
    pinned = _dp_request("pinned", shared_blocks=4, data_parallel_rank=0)

    assert _route(client, pinned) == 0
    assert pinned.data_parallel_rank == 0
    # Engine 0 was sent the prefix too, so an equal hit now resolves on load.
    _snapshot(client, [0, 0, 0.0], [0, 0, 0.0], [0, 1, 0.0])
    assert _route(client, _dp_request("q", shared_blocks=4)) == 0


def test_dp_routing_off_leaves_the_balancer_alone():
    client = _dp_client(routing=False)
    _route(client, _dp_request("doc", shared_blocks=4, data_parallel_rank=0))
    _snapshot(client, [0, 1, 0.0], [0, 0, 0.0], [0, 0, 0.0])

    follower = _dp_request("q", shared_blocks=4)

    assert _route(client, follower) == 1
    assert follower.data_parallel_rank is None
    assert client._tt_prefix_cache_mirror is None


def test_dp_a_prefix_cache_reset_forgets_what_the_engines_were_sent():
    client = _dp_client()
    _route(client, _dp_request("doc", shared_blocks=4, data_parallel_rank=2))
    _snapshot(client, [0, 0, 0.0], [0, 0, 0.0], [0, 1, 0.0])
    utility_calls = []

    async def call_utility_async(method, *args):
        utility_calls.append((method, args))

    client.call_utility_async = call_utility_async
    asyncio.run(
        core_client.DPLBAsyncMPClient.reset_prefix_cache_async(
            client, reset_running_requests=True
        )
    )

    assert [call[0] for call in utility_calls] == ["reset_prefix_cache"]
    follower = _dp_request("q", shared_blocks=4)
    assert _route(client, follower) == 0
    assert follower.data_parallel_rank is None


def test_dp_routing_patch_installs_once():
    from vllm_tt_plugin.platform import _install_tt_dp_prefix_cache_routing_patch

    _install_tt_dp_prefix_cache_routing_patch()
    installed = core_client.DPLBAsyncMPClient.get_core_engine_for_request
    _install_tt_dp_prefix_cache_routing_patch()

    assert core_client.DPLBAsyncMPClient.get_core_engine_for_request is installed
    assert core_client._tt_original_get_core_engine_for_request is not installed


def test_dp_prefix_cache_routing_config_key():
    from vllm_tt_plugin.config import get_tt_dp_prefix_cache_routing

    assert get_tt_dp_prefix_cache_routing(SimpleNamespace(additional_config={})) is True
    off = SimpleNamespace(additional_config={"tt": {"dp_prefix_cache_routing": False}})
    assert get_tt_dp_prefix_cache_routing(off) is False
    with pytest.raises(ValueError, match="must be a boolean"):
        get_tt_dp_prefix_cache_routing(
            SimpleNamespace(additional_config={"tt": {"dp_prefix_cache_routing": 1}})
        )
