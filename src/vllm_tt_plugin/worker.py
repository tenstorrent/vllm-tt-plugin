# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import math
import os
import time
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import ttnn
from vllm.config import VllmConfig
from vllm.model_executor.model_loader import get_model_architecture
from vllm.tasks import SupportedTask
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_groups,
    get_uniform_page_size,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from vllm_tt_plugin.config import (
    get_tt_config,
    get_tt_data_parallel_size,
    get_tt_output_tokens_per_step,
    get_tt_per_lane_max_num_seqs,
    is_tt_block_output_model,
)
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import (
    _STANDARD_DP_VISIBLE_GROUPS_KEY,
    _TT_TOKEN_TILE_SIZE,
    TTPlatform,
    _load_standard_dp_visible_groups,
    _min_block_output_max_model_len,
    _should_pre_register_tt_test_models_from_cli,
    register_tt_models,
)
from vllm_tt_plugin.utils.dp_discovery import (
    format_tt_visible_devices,
    parse_mesh_grid,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = init_tt_logger(__name__)

# Ensure TT model architectures are registered in this process as early as
# possible. `WorkerWrapperBase.init_worker` imports the worker class module
# before initializing multimodal caches; without this, early architecture
# inspection may fail for TT-prefixed architectures.
register_tt_models(register_test_models=_should_pre_register_tt_test_models_from_cli())


def _bind_visible_devices_env(vllm_config: VllmConfig) -> None:
    """Bind ``TT_VISIBLE_DEVICES`` to this rank's device group.

    The engine-core launcher writes ``parallel_config.assigned_physical_gpu_ids``
    rather than exporting a per-rank env var. tt-metal reads only the env var, so
    the worker materializes it here; otherwise every rank keeps the launcher's
    value and they share chips.

    Standard-DP discovery owns the rank-to-submesh topology. A nonempty
    assignment must agree with the discovered group for the local rank. MPI
    launches populate neither and keep the inherited value.

    Raises:
        RuntimeError: discovery holds no group for this rank or an assignment
            conflicts with its discovered group.
    """
    parallel_config = vllm_config.parallel_config
    # Absent on the fork vLLM that the explicit MPI launcher targets.
    assigned_physical_gpu_ids = getattr(
        parallel_config, "assigned_physical_gpu_ids", None
    )
    groups = _load_standard_dp_visible_groups(vllm_config)

    if groups is not None:
        local_dp_rank = parallel_config.data_parallel_rank_local
        if local_dp_rank is None or not 0 <= local_dp_rank < len(groups):
            raise RuntimeError(
                f"No TT device group for local DP rank {local_dp_rank}: "
                f"discovery stored {len(groups)} group(s) under "
                f"additional_config[{_STANDARD_DP_VISIBLE_GROUPS_KEY!r}]"
            )

        visible_devices = groups[local_dp_rank]
        if assigned_physical_gpu_ids:
            assigned_visible_devices = format_tt_visible_devices(
                assigned_physical_gpu_ids
            )
            if assigned_visible_devices != visible_devices:
                raise RuntimeError(
                    "TT standard-DP assignment conflicts with discovery: "
                    f"local DP rank {local_dp_rank} was assigned "
                    f"{assigned_visible_devices!r}, but discovery requires "
                    f"{visible_devices!r}."
                )
    elif assigned_physical_gpu_ids:
        visible_devices = format_tt_visible_devices(assigned_physical_gpu_ids)
    else:
        return

    evar = TTPlatform.device_control_env_var
    inherited = os.environ.get(evar)
    os.environ[evar] = visible_devices

    logger.info(
        "Bound %s=%s for local DP rank %s (inherited %r)",
        evar,
        visible_devices,
        parallel_config.data_parallel_rank_local,
        inherited,
    )


def _resolve_mesh_grid(
    mesh_device_env: str | None,
    num_devices_available: int,
    visible_devices_env: str | None,
) -> tuple[int, int]:
    mesh_grid = parse_mesh_grid(
        mesh_device_env,
        num_devices_available,
        tg_mesh_grid=(8, 4),
    )

    if visible_devices_env:
        stored_mesh_grid = TTPlatform._standard_dp_mesh_grids.get(visible_devices_env)
        if stored_mesh_grid is not None:
            return stored_mesh_grid

        visible_count = len([d for d in visible_devices_env.split(",") if d.strip()])
        if visible_count > 0 and mesh_grid[0] * mesh_grid[1] != visible_count:
            mesh_grid = (1, visible_count)
        elif (
            visible_count == 0 and mesh_grid[0] * mesh_grid[1] != num_devices_available
        ):
            mesh_grid = (1, num_devices_available)

    return mesh_grid


def _available_kv_cache_memory_bytes_for_num_blocks(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    num_blocks: int,
) -> int:
    """Returns a byte budget that reconstructs ``num_blocks`` upstream.

    Standard-DP now uses vLLM's upstream multiprocess executor, so mutating
    ``cache_config.num_gpu_blocks_override`` inside the worker subprocess is not
    sufficient on its own: the engine-side KV planner lives in a different
    process. Instead, return the exact amount of "available memory" that makes
    upstream's grouping logic resolve the desired TT block count.
    """
    kv_cache_groups = get_kv_cache_groups(vllm_config, dict(kv_cache_spec))
    if not kv_cache_groups:
        return 0

    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        return kv_cache_groups[0].kv_cache_spec.page_size_bytes * num_blocks

    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    page_size = get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    return page_size * num_blocks * group_size


class TTWorker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )

        # Initialized by init_device
        self.mesh_device = None
        self._num_tt_blocks: int | None = None

        # Whether to use ttnn tracing for model execution
        tt_config = get_tt_config(self.vllm_config)
        trace_key = "trace_mode"
        self.trace_mode = "all"
        if tt_config and trace_key in tt_config:
            assert tt_config[trace_key] in ["decode_only", "all", "none"], (
                f"Invalid {trace_key}: {tt_config[trace_key]}"
            )
            self.trace_mode = tt_config[trace_key]

        enable_model_warmup_key = "enable_model_warmup"
        self.enable_model_warmup = True
        if tt_config and enable_model_warmup_key in tt_config:
            assert tt_config[enable_model_warmup_key] in [True, False], (
                f"Invalid {enable_model_warmup_key}: \
                {tt_config[enable_model_warmup_key]}"
            )

            self.enable_model_warmup = tt_config[enable_model_warmup_key]

    def init_device(self) -> None:
        # tt-metal latches the visible set at first cluster construction and never
        # re-reads the env var, so bind before `check_and_update_config` ->
        # `get_model_architecture` imports the tt-metal model module.
        _bind_visible_devices_env(self.vllm_config)

        # Validate/apply TT config in this worker process (multiprocessing
        # means platform class attrs + config mutations must be applied per
        # subprocess) before runner init.
        TTPlatform.check_and_update_config(self.vllm_config)

        local_dp_rank = self.parallel_config.data_parallel_rank_local
        logger.info(
            "TT worker standard-DP binding: data_parallel_index=%s "
            "data_parallel_rank_local=%s %s=%s MESH_DEVICE=%s",
            self.parallel_config.data_parallel_index,
            local_dp_rank,
            TTPlatform.device_control_env_var,
            os.environ.get(TTPlatform.device_control_env_var),
            os.environ.get("MESH_DEVICE"),
        )
        self.mesh_device = open_mesh_device(
            get_tt_config(self.vllm_config), self.trace_mode, local_dp_rank
        )
        self.device = self.mesh_device
        self.device_config.device = self.mesh_device
        assert self.mesh_device is not None
        self.num_devices = self.mesh_device.get_num_devices()

        # Size the KV pool and settle --max-model-len here, not in
        # determine_available_memory: upstream runs the whole executor init
        # (init_device, then load_model) before it asks for available memory
        # (EngineCore.__init__ -> _initialize_kv_caches), so a model that sizes
        # its own KV state at load time would otherwise see the unfitted
        # length -- for --max-model-len -1 that is the full HF context.
        # The count is cached rather than recomputed later because
        # get_max_tokens_all_users derives the budget from max_model_len,
        # which the fit below may have shrunk.
        self._num_tt_blocks = get_num_available_blocks_tt(
            self.vllm_config, self.num_devices
        )
        _fit_block_output_max_model_len(self.vllm_config, self._num_tt_blocks)

        # Init ModelRunner here, so that we have access to self.mesh_device.
        self.model_runner: TTModelRunner = TTModelRunner(
            vllm_config=self.vllm_config,
            mesh_device=self.mesh_device,
            trace_mode=self.trace_mode,
            enable_model_warmup=self.enable_model_warmup,
            num_devices=self.num_devices,
        )

    def load_model(self):
        self.model_runner.load_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        For the GPU/TPU backends, this method generates the KVCacheSpec by
        parsing the kv cache format from each Attention module in the static
        forward context (compilation_config.static_forward_context).
        core/kv_cache_utils.py uses the KVCacheSpec along with available
        memory info from a profiling run to determine num blocks.

        For the TT backend, the static forward context is not populated since
        the modelling code is independent. Two paths are supported:

        1. Hybrid models (mixed sliding-window + full-attention layers, e.g.
           Gemma3/4 / GPT-OSS) opt in by defining a ``get_kv_cache_spec``
           classmethod on the registered TT model class:

               @classmethod
               def get_kv_cache_spec(
                   cls, vllm_config
               ) -> dict[str, KVCacheSpec] | None:
                   ...

           The returned dict maps a layer name to its per-layer spec. This
           lets upstream's hybrid kv cache manager pack each attention type
           into its own group with its own per-request block budget.

        2. Models without the hook (and models that return ``None``) fall
           back to a single homogeneous spec under the dummy ``"foo"`` layer
           name, the same behaviour the TT backend has always had. As before
           we don't run profiling for available memory and instead override
           num blocks via ``self.cache_config.num_gpu_blocks_override``.
        """
        spec_from_hook = self._try_get_spec_from_model_hook()
        if spec_from_hook is not None:
            return spec_from_hook

        return self._build_default_kv_cache_spec()

    def _try_get_spec_from_model_hook(self) -> dict[str, KVCacheSpec] | None:
        """If the resolved TT model class implements ``get_kv_cache_spec``,
        invoke it and return the result. Returns ``None`` when the hook is
        absent or explicitly returns ``None`` (signalling fallback to the
        single-spec default).
        """
        from vllm.model_executor.models.registry import ModelRegistry

        # ``ModelConfig.architecture`` (singular) is computed in
        # ``ModelConfig.__post_init__`` from ``hf_config.architectures``
        # *before* :meth:`TTPlatform.check_and_update_config` prepends ``"TT"``
        # to the architectures list. As a result the cached property still
        # holds the upstream (e.g. CUDA) name, and resolving it would find
        # upstream's vLLM model class — which doesn't have our
        # ``get_kv_cache_spec`` hook. Prefer the prefixed entry from the
        # ``architectures`` list (which the platform modifies in-place) and
        # fall back to prepending ``"TT"`` when neither is available.
        arch = next(
            (a for a in self.model_config.architectures if a.startswith("TT")),
            None,
        )
        if arch is None:
            arch = self.model_config.architecture
            if not arch.startswith("TT"):
                arch = "TT" + arch
        model_cls, _ = ModelRegistry.resolve_model_cls(
            arch, model_config=self.model_config
        )
        hook = getattr(model_cls, "get_kv_cache_spec", None)
        if hook is None:
            return None
        spec = hook(self.vllm_config)
        if spec is None:
            return None
        if not isinstance(spec, dict) or not all(
            isinstance(k, str) and isinstance(v, KVCacheSpec) for k, v in spec.items()
        ):
            raise TypeError(
                f"{model_cls.__name__}.get_kv_cache_spec() must return "
                f"dict[str, KVCacheSpec] or None, got {type(spec).__name__}"
            )
        return spec

    def _build_default_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Single-layer spec used by the legacy non-hybrid path. Downstream
        sizing is overridden via ``cache_config.num_gpu_blocks_override``.
        """
        model_config = self.model_config
        parallel_config = self.parallel_config
        cache_config = self.cache_config

        # Excludes TP factor since that is handled on the model side for TT.
        total_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )

        use_mla = model_config.use_mla
        sliding_window = model_config.get_sliding_window()
        attn_spec: KVCacheSpec
        if use_mla:
            assert not sliding_window, "MLA not supported for sliding window"
            attn_spec = MLAAttentionSpec(
                block_size=cache_config.block_size,
                num_kv_heads=total_num_kv_heads,
                head_size=head_size,
                dtype=dtype,
            )
        else:
            attn_spec = FullAttentionSpec(
                block_size=cache_config.block_size,
                num_kv_heads=total_num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                sliding_window=sliding_window,
            )
        return {"foo": attn_spec}

    def determine_available_memory(self) -> int:
        """
        For the GPU/TPU backends, this method runs profiling to determine
        available memory for the KV cache. The available memory is then used
        in conjunction with the output of get_kv_cache_spec to determine
        the number of kv cache blocks (total memory / page_size / num layers).

        NOTE: TT does not profile device memory yet. ``init_device`` computed the
              target TT KV block count before the model loaded; this returns a
              synthetic byte budget that makes the upstream KV planner
              reconstruct that same block count in the engine process.
        """
        if self._num_tt_blocks is None:
            raise RuntimeError(
                "The TT KV pool has not been sized; determine_available_memory "
                "ran before init_device"
            )
        num_tt_blocks = self._num_tt_blocks
        kv_cache_spec = self.get_kv_cache_spec()
        self.cache_config.num_gpu_blocks_override = num_tt_blocks
        return _available_kv_cache_memory_bytes_for_num_blocks(
            self.vllm_config,
            kv_cache_spec,
            num_tt_blocks,
        )

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate TT KV cache and initialize persistent input batch.

        Every standard-DP rank owns its own TT mesh/KV cache, while
        single-process lane mode has only one rank.
        """
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def update_max_model_len(self, max_model_len: int) -> None:
        # The engine calls this via collective_rpc when --max-model-len -1
        # auto-fit reduces max_model_len to the KV cache capacity.
        # WorkerBase has no such hook (only the GPU worker defines one), so
        # TTWorker must provide it or the RPC raises AttributeError.
        # TTModelRunner reads self.model_config.max_model_len directly when it
        # builds the persistent input batch in initialize_kv_cache -- which the
        # engine calls after this RPC -- so updating the shared model config is
        # sufficient; it keeps no separate cached copy.
        self.model_config.max_model_len = max_model_len

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # The executor reduces per-worker timings returned here into
        # compilation_config.compilation_time. TT does device warmup rather than
        # graph compilation, so report the warmup wall time as the language-model
        # figure and zero for the (absent) encoder phase.
        if not self.enable_model_warmup:
            logger.warning("Skipping model warmup")
            return CompilationTimes(language_model=0.0, encoder=0.0)

        start = time.perf_counter()
        self.model_runner.warmup_model()
        elapsed = time.perf_counter() - start

        return CompilationTimes(language_model=elapsed, encoder=0.0)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        """Run the device forward for a non-DP or lane-DP step.

        Returns ``None``: the forward leaves a pending sampler that the engine
        finalizes via ``sample_tokens``. The runner dispatches plain
        single-process vs lane-DP internally on the scheduler's step plan, so
        the worker does not need to know which is active.
        """
        assert self.is_driver_worker, "There should only be one Worker for TT"
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(
        self,
        grammar_output: "GrammarOutput | None",
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Sample the forward deferred by ``execute_model``.

        Called by the engine exactly once after ``execute_model`` returns
        ``None``, matching the vLLM V1 forward-then-sample flow. The grammar
        bitmask is reordered and applied here, at sample time. Returns an async
        wrapper for overlapped decode, otherwise a completed output.
        """
        assert self.is_driver_worker, "There should only be one Worker for TT"
        return self.model_runner.sample_tokens(grammar_output)

    def check_health(self) -> None:
        # Worker will always be healthy as long as it's running.
        return

    def shutdown(self) -> None:
        """Release model-owned captures and close the mesh before exit.

        This is the hook upstream guarantees on every orderly shutdown path:
        EngineCore's SIGTERM/SIGINT handler ends run_busy_loop and the
        surrounding ``finally`` reaches ``executor.shutdown()`` even on fatal
        errors. ``__del__`` alone is not guaranteed at interpreter exit, and a
        mesh left open wedges the board's ethernet cores for the *next*
        process ("Timed out while waiting for active ethernet core ... Try
        resetting the board").
        """
        runner = getattr(self, "model_runner", None)
        if runner is not None:
            runner.shutdown()
        mesh_device = getattr(self, "mesh_device", None)
        if mesh_device is not None:
            close_mesh_device(mesh_device, get_tt_config(self.vllm_config))
            # Idempotence: __del__ (and a second shutdown call) must not
            # close the mesh again.
            self.mesh_device = None
        # Release the process-level admission handle when it points at this
        # engine's config, so an in-process successor engine is admitted
        # without waiting for garbage collection to clear the weakref.
        vllm_config = getattr(self, "vllm_config", None)
        if (
            vllm_config is not None
            and TTPlatform._resolve_tt_admission_handle() is vllm_config
        ):
            TTPlatform._tt_vllm_config = None

    # ---- Destructor (used to close devices) ----

    def __del__(self):
        # Delete model runner first in case there are model artifacts.
        # Separate suppress blocks: init_device raises between opening the
        # mesh and assigning model_runner (sizing validation), and a missing
        # model_runner must not short-circuit closing the open mesh.
        with suppress(AttributeError):
            # attributes may be already torn down when destructor is called
            del self.model_runner

        with suppress(AttributeError):
            if self.mesh_device:
                close_mesh_device(self.mesh_device, get_tt_config(self.vllm_config))
                del self.mesh_device

        if hasattr(super(), "__del__"):
            super().__del__()  # type: ignore


def get_num_available_blocks_tt(vllm_config: VllmConfig, num_devices: int = 1) -> int:
    """
    Used to set the number of available blocks for the TT KV cache as we
    currently do not run profiling to determine available memory.

    Pure sizing query: validates the budget (raising when a block-output pool
    cannot hold an explicitly configured ``max_model_len`` plus one canvas) but
    never mutates the config; the ``--max-model-len -1`` fit, and the
    servability check on the length it writes, live in
    ``_fit_block_output_max_model_len``.

    ``num_devices`` is the runtime-discovered physical device count.
    """

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config

    # region Get default or model- and device-specific `max_tokens_all_users`
    model_class = None
    try:
        tt_data_parallel = get_tt_data_parallel_size(vllm_config)
        model_class, _ = get_model_architecture(model_config)
        # Pass the per-submesh batch (the requests one submesh actually serves),
        # not the global engine capacity, so a model that derives a per-user
        # token budget from ``max_num_seqs`` computes the same value whether
        # parallelism is expressed as multi-process DP (each rank its own
        # engine) or single-process lane mode. This matches the padding term
        # below, which also uses ``get_tt_per_lane_max_num_seqs``, and keeps the
        # KV shape identical across both modes.
        max_tokens_all_users = model_class.get_max_tokens_all_users(
            model_name=model_config.model,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            max_model_len=model_config.max_model_len,
            max_num_seqs=get_tt_per_lane_max_num_seqs(vllm_config),
        )

        logger.info(
            "Getting max_tokens_all_users=%d for number of blocks in KV cache "
            "from generator '%s'.",
            max_tokens_all_users,
            model_class,
        )
    except AttributeError:
        max_tokens_all_users = 131_072

        logger.warning(
            "Setting max_tokens_all_users=%d for number of blocks in KV cache "
            "using rules in `get_num_available_blocks_tt`.",
            max_tokens_all_users,
        )
    # endregion

    # To fit a max batch with (max_tokens_all_users / max batch) per user,
    # allocate one worst-case output reservation per user. AR models need one
    # extra cache block; block-output models commit their entire output canvas
    # atomically and therefore need at least one full canvas of headroom.
    #
    # ``num_blocks`` is applied to each submesh KV cache un-divided, so the
    # padding must use the *per-lane/per-rank* batch -- the number of requests
    # a single submesh actually serves -- not the global engine capacity. In
    # multi-process DP this is ``max_num_seqs`` (each rank is its own engine);
    # in single-process lane mode it is ``max_num_seqs // lane count``.
    # Both reduce to the same per-submesh value, keeping the KV shape identical
    # regardless of how parallelism is expressed.
    max_batch = get_tt_per_lane_max_num_seqs(vllm_config)
    output_tokens_per_step = get_tt_output_tokens_per_step(vllm_config)
    per_user_output_reservation = max(cache_config.block_size, output_tokens_per_step)
    max_tokens_all_users += per_user_output_reservation * max_batch

    # Hybrid attention models (Gemma3/4, GPT-OSS, ...) normally split layers
    # into multiple kv_cache_groups: a full-attention group plus several
    # sliding-window groups. Upstream's hybrid manager packs these into
    # ``group_size = min(layer_counts_per_type)`` buffers and indexes them via
    # per-group block tables, so each request consumes
    # ``full_blocks_per_request + Σ sliding_blocks_per_request`` block IDs.
    #
    # Whether a given model actually emits SlidingWindowSpec (and therefore
    # needs this sliding-window headroom) is decided per model class via
    # ``_HYBRID_KV_CACHE_GROUPS_ENABLED``. Gemma4 re-enables it (it ships the
    # bounded sliding-window decode fix); Gemma3 / GPT-OSS keep it ``False`` and
    # emit FullAttentionSpec for every layer, so adding headroom for them would
    # over-allocate full-size KV blocks and can OOM Gemma3-27B on T3K. Read the
    # resolved model class's flag rather than a single global so re-enabling for
    # one model doesn't regress the others; default to ``False`` when the class
    # can't be resolved.
    hybrid_kv_cache_groups_enabled = getattr(
        model_class, "_HYBRID_KV_CACHE_GROUPS_ENABLED", False
    )
    sliding_window = model_config.get_sliding_window()
    if hybrid_kv_cache_groups_enabled and sliding_window is not None:
        # Conservative cap: assume up to a few sliding groups per buffer
        # (typical for Gemma3 5:1 / GPT-OSS 1:1 hybrid patterns) and add
        # ``sliding_window * max_batch`` worth of tokens per group as
        # padding. The exact number of sliding groups isn't known here
        # (the spec hook hasn't run yet); bound it with a small constant
        # rather than walking the model layer types from raw HF config.
        _MAX_SLIDING_GROUPS_HEURISTIC = 8
        max_tokens_all_users += (
            sliding_window * max_batch * _MAX_SLIDING_GROUPS_HEURISTIC
        )

    num_tt_blocks = math.ceil(max_tokens_all_users / cache_config.block_size)
    if is_tt_block_output_model(vllm_config):
        resolved_kv_tokens = num_tt_blocks * cache_config.block_size
        required_kv_tokens = model_config.max_model_len + output_tokens_per_step
        # ``--max-model-len -1`` asked for whatever the pool allows, so a pool
        # this length does not fit is not an error yet: the fit that follows
        # shrinks the length and validates what it writes.
        if (
            resolved_kv_tokens < required_kv_tokens
            and getattr(model_config, "original_max_model_len", None) != -1
        ):
            raise ValueError(
                "Block-output KV budget is too small: resolved "
                f"{resolved_kv_tokens} tokens, but max_model_len="
                f"{model_config.max_model_len} plus output_tokens_per_step="
                f"{output_tokens_per_step} requires at least "
                f"{required_kv_tokens}. Fix the model's "
                "get_max_tokens_all_users budget."
            )

    return num_tt_blocks


def _fit_block_output_max_model_len(
    vllm_config: VllmConfig, num_tt_blocks: int
) -> None:
    """Fit ``--max-model-len -1`` to the block-output KV pool.

    Deliberately pre-empts vLLM's ``_auto_fit_max_model_len``
    (vllm/v1/core/kv_cache_utils.py:2092): the TT pool is a fixed budget
    rather than profiled memory, and the upstream fit would hand the whole
    pool to ``max_model_len``, while a block-output request also needs one
    full output canvas of headroom. The engine snapshots ``max_model_len``
    only after ``determine_available_memory`` returns
    (``EngineCore._initialize_kv_caches``), so upstream's fit sees the
    fitted value as its baseline and, since it only ever shrinks, keeps it.

    Also owns the servability check for the fitted length: the sizing query
    deliberately lets an oversized ``-1`` request through, so this is where a
    pool too small to hold a prompt tile plus a canvas is rejected.
    """
    model_config = vllm_config.model_config
    output_tokens_per_step = get_tt_output_tokens_per_step(vllm_config)
    auto_fit_requested = getattr(model_config, "original_max_model_len", None) == -1
    if not auto_fit_requested or not is_tt_block_output_model(vllm_config):
        return
    resolved_kv_tokens = num_tt_blocks * vllm_config.cache_config.block_size
    fitted_max_model_len = resolved_kv_tokens - output_tokens_per_step
    if fitted_max_model_len >= model_config.max_model_len:
        return
    min_max_model_len = _min_block_output_max_model_len(output_tokens_per_step)
    if fitted_max_model_len < min_max_model_len:
        raise ValueError(
            "Block-output KV budget is too small to auto-fit --max-model-len: "
            f"the pool holds {resolved_kv_tokens} tokens, leaving "
            f"max_model_len={fitted_max_model_len} after reserving one "
            f"{output_tokens_per_step}-token output canvas, but serving a "
            f"request also needs a {_TT_TOKEN_TILE_SIZE}-token prompt tile, so "
            f"max_model_len must be at least {min_max_model_len}. Raise the "
            "model's get_max_tokens_all_users budget."
        )
    logger.info(
        "Auto-fitting block-output max_model_len from %d to %d so "
        "the KV budget covers one full output canvas.",
        model_config.max_model_len,
        fitted_max_model_len,
    )
    model_config.max_model_len = fitted_max_model_len


# TT-NN utilities


def get_dispatch_core_config(tt_config):
    dispatch_core_axis: ttnn.DispatchCoreAxis = None
    if tt_config is not None and "dispatch_core_axis" in tt_config:
        assert tt_config["dispatch_core_axis"] in ["row", "col"], (
            "Invalid dispatch_core_axis:"
            f"{tt_config['dispatch_core_axis']}. "
            "Expected: row, col."
        )
        dispatch_core_axis = (
            ttnn.DispatchCoreAxis.COL
            if tt_config["dispatch_core_axis"] == "col"
            else ttnn.DispatchCoreAxis.ROW
        )

    return ttnn.DispatchCoreConfig(axis=dispatch_core_axis)


def get_fabric_config(tt_config, num_devices):
    if num_devices == 1:
        # Ignore any explicit fabric request for single-device meshes.
        return None

    # Wormhole Galaxy (6U) uses a 1D ring. Blackhole Galaxy needs a 2D torus:
    # column-axis collectives have no wraparound path on 1D fabrics.
    cluster_type = ttnn.cluster.get_cluster_type()
    if cluster_type == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
        fabric_config = ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    elif cluster_type == ttnn.cluster.ClusterType.GALAXY:
        fabric_config = ttnn.FabricConfig.FABRIC_1D_RING
    else:
        fabric_config = ttnn.FabricConfig.FABRIC_1D

    # Override fabric_config if specified in TT plugin config. Resolve the name
    # from ttnn.FabricConfig so newly added fabrics (e.g. FABRIC_2D_TORUS_XY)
    # work without a plugin allow-list update.
    if tt_config is not None and "fabric_config" in tt_config:
        fabric_config_str = tt_config["fabric_config"]
        fabric_config = ttnn.FabricConfig.__members__.get(fabric_config_str)
        assert fabric_config is not None, (
            f"Invalid fabric_config: {fabric_config_str}. "
            f"Expected one of {list(ttnn.FabricConfig.__members__)}."
        )
    return fabric_config


def get_reliability_mode(tt_config):
    # Default to strict init and override if specified in TT plugin config.
    reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT
    if tt_config is not None and "fabric_reliability_mode" in tt_config:
        reliability_mode_str = tt_config["fabric_reliability_mode"]
        reliability_mode_map = {
            "STRICT_INIT": ttnn.FabricReliabilityMode.STRICT_INIT,
            "RELAXED_INIT": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
        reliability_mode = reliability_mode_map.get(reliability_mode_str)
        assert reliability_mode is not None, (
            f"Invalid fabric_reliability_mode: {reliability_mode_str}. "
            f"Expected one of {list(reliability_mode_map.keys())}."
        )
    return reliability_mode


# From tt-metal/conftest.py:
# Set fabric config to passed in value
# Do nothing if not set
# Must be called before creating the mesh device
def set_fabric(tt_config, num_devices):
    fabric_config = get_fabric_config(tt_config, num_devices)
    if fabric_config:
        reliability_mode = get_reliability_mode(tt_config)
        logger.info(
            "Setting fabric config: %s, reliability mode: %s",
            fabric_config,
            reliability_mode,
        )
        ttnn.set_fabric_config(fabric_config, reliability_mode)


# From tt-metal/conftest.py:
# Reset fabric config to DISABLED if not None, and do nothing otherwise
# Temporarily require previous state to be passed
# in as even setting it to DISABLED might be unstable
# This is to ensure that we don't propagate
# the instability to the rest of CI
def reset_fabric(tt_config, num_devices):
    fabric_config = get_fabric_config(tt_config, num_devices)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def device_params_from_tt_config(tt_config, trace_mode):
    device_params = {}

    if trace_mode in ["all", "decode_only"]:
        # Set the most common value as default, override later
        device_params["trace_region_size"] = 50000000
        if tt_config and "trace_region_size" in tt_config:
            device_params["trace_region_size"] = tt_config["trace_region_size"]

    if tt_config and "worker_l1_size" in tt_config:
        device_params["worker_l1_size"] = tt_config["worker_l1_size"]

    if tt_config and "l1_small_size" in tt_config:
        device_params["l1_small_size"] = tt_config["l1_small_size"]

    return device_params


def get_mesh_grid(*args: Any, **kwargs: Any):
    if args or kwargs.get("local_dp_rank") is not None:
        warnings.warn(
            "get_mesh_grid() ignores deprecated local_dp_rank; mesh selection "
            "now derives from MESH_DEVICE and TT_VISIBLE_DEVICES",
            UserWarning,
            stacklevel=2,
        )

    num_devices_available = ttnn.get_num_devices()
    mesh_grid = _resolve_mesh_grid(
        os.environ.get("MESH_DEVICE"),
        num_devices_available,
        os.environ.get(TTPlatform.device_control_env_var),
    )

    assert ttnn.using_distributed_env() or (
        mesh_grid[0] * mesh_grid[1] <= num_devices_available
    ), (
        f"Requested mesh grid shape {mesh_grid} is larger than "
        f"number of available devices {num_devices_available}"
    )

    return mesh_grid


def open_mesh_device(tt_config, trace_mode, local_dp_rank=0):
    mesh_grid = get_mesh_grid()
    logger.info("Attempting to open mesh device with grid shape %s", mesh_grid)

    device_params = device_params_from_tt_config(tt_config, trace_mode)

    # Set fabric before opening the device
    num_devices_requested = mesh_grid[0] * mesh_grid[1]
    set_fabric(tt_config, num_devices_requested)

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(*mesh_grid),
        dispatch_core_config=get_dispatch_core_config(tt_config),
        **device_params,
    )
    logger.info(
        "multidevice with %d devices and grid %s is created",
        mesh_device.get_num_devices(),
        mesh_grid,
    )
    return mesh_device


def close_mesh_device(mesh_device, tt_config):
    # Read device profiler (no-op if not profiling with tracy)
    ttnn.ReadDeviceProfiler(mesh_device)

    # Close devices
    num_devices = mesh_device.get_num_devices()
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)

    # Reset fabric
    reset_fabric(tt_config, num_devices)
