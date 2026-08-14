# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import inspect
import json
import multiprocessing
import os
import sys
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

import torch
from vllm.platforms.interface import Platform, PlatformEnum

from vllm_tt_plugin.config import (
    get_tt_config,
    get_tt_data_parallel_size,
    get_tt_output_tokens_per_step,
    store_tt_lane_count,
    store_tt_output_tokens_per_step,
    uses_tt_lane_coordinator,
    validate_tt_lane_config,
)
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.utils.dp_discovery import (
    StandardDPAssignmentT,
    _run_standard_dp_visible_device_group_discovery,
    _split_standard_dp_discovery_result,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.inputs import EngineInput
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = object

logger = init_tt_logger(__name__)

_STANDARD_DP_DISCOVERY_RECV_TIMEOUT_S = 60.0
_STANDARD_DP_DISCOVERY_JOIN_TIMEOUT_S = 5.0
_STANDARD_DP_MESH_GRIDS_KEY = "_tt_standard_dp_mesh_grids"
_STANDARD_DP_VISIBLE_GROUPS_KEY = "_tt_standard_dp_visible_groups"

TT_SCHEDULER_CLS = "vllm_tt_plugin.scheduler.TTScheduler"
TT_LANE_SCHEDULER_CLS = "vllm_tt_plugin.lane_scheduler.TTLaneCoordinator"


class _MaxModelLenConfig(Protocol):
    max_model_len: int


# TT model versions backed by the single-execute Galaxy generator
# (models.demos.llama3_70b_galaxy.tt.generator:Generator). For these,
# --data_parallel_size folds into single-process TT lanes. Maps the selecting
# env var to the version value that routes through that generator.
_GALAXY_GENERATOR_VERSIONS = {
    "TT_LLAMA_TEXT_VER": "llama3_70b_galaxy",
    "TT_QWEN3_TEXT_VER": "qwen3_32b_galaxy",
}


def _galaxy_generator_version() -> str | None:
    """Return the active Galaxy-generator model version, or None.

    Both ``llama3_70b_galaxy`` (Llama3 70B) and ``qwen3_32b_galaxy`` (Qwen3-32B)
    are served by ``models.demos.llama3_70b_galaxy.tt.generator:Generator``.
    """
    for env_var, version in _GALAXY_GENERATOR_VERSIONS.items():
        if os.getenv(env_var) == version:
            return version
    return None


def _uses_explicit_tt_mpi_launch(vllm_config: "VllmConfig") -> bool:
    """Returns whether TT must use explicit MPI-based placement."""
    tt_config = get_tt_config(vllm_config)
    parallel_config = vllm_config.parallel_config
    return bool(
        tt_config.get("rank_binding")
        or tt_config.get("mpi_args")
        or getattr(parallel_config, "nnodes", 1) > 1
        or getattr(parallel_config, "node_rank", 0) > 0
    )


def _store_standard_dp_mesh_grids(
    vllm_config: "VllmConfig",
    mesh_grids: dict[str, tuple[int, int]],
) -> None:
    """Store discovered mesh-grid hints on the vLLM config."""
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        additional_config = {}
        vllm_config.additional_config = additional_config
    additional_config[_STANDARD_DP_MESH_GRIDS_KEY] = {
        visible_devices: [mesh_grid[0], mesh_grid[1]]
        for visible_devices, mesh_grid in mesh_grids.items()
    }


def _store_standard_dp_visible_groups(
    vllm_config: "VllmConfig",
    visible_groups: list[str],
) -> None:
    """Store the per-rank visible-device group list on the vLLM config.

    Indexed by DP rank, and kept on ``additional_config`` because that is a
    declared config field and so survives pickling into the worker subprocess.
    """
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        additional_config = {}
        vllm_config.additional_config = additional_config
    additional_config[_STANDARD_DP_VISIBLE_GROUPS_KEY] = list(visible_groups)


def _load_standard_dp_visible_groups(
    vllm_config: "VllmConfig",
) -> list[str] | None:
    """Load the per-rank visible-device group list from the vLLM config.

    ``None`` means nothing was stored. An empty list means discovery stored
    nothing usable, which callers must not treat as "keep the inherited value".
    """
    additional_config = getattr(vllm_config, "additional_config", None) or {}

    if not isinstance(additional_config, dict):
        return None

    groups = additional_config.get(_STANDARD_DP_VISIBLE_GROUPS_KEY)
    if not isinstance(groups, list):
        return None

    return [str(g) for g in groups]


def _load_standard_dp_mesh_grids(
    vllm_config: "VllmConfig",
) -> dict[str, tuple[int, int]]:
    """Load stored mesh-grid hints from the vLLM config."""
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional_config, dict):
        return {}

    raw_mesh_grids = additional_config.get(_STANDARD_DP_MESH_GRIDS_KEY, {})
    if not isinstance(raw_mesh_grids, dict):
        return {}

    mesh_grids: dict[str, tuple[int, int]] = {}
    for visible_devices, mesh_grid in raw_mesh_grids.items():
        if not isinstance(visible_devices, str):
            continue
        if not isinstance(mesh_grid, (list, tuple)) or len(mesh_grid) != 2:
            continue
        try:
            mesh_grids[visible_devices] = (int(mesh_grid[0]), int(mesh_grid[1]))
        except (TypeError, ValueError):
            continue

    return mesh_grids


def _terminate_discovery_process(proc: multiprocessing.Process) -> None:
    """Terminates a lingering discovery subprocess."""
    proc.terminate()
    proc.join(timeout=_STANDARD_DP_DISCOVERY_JOIN_TIMEOUT_S)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=_STANDARD_DP_DISCOVERY_JOIN_TIMEOUT_S)


def _join_discovery_process_or_raise(
    proc: multiprocessing.Process,
    message: str,
) -> None:
    """Joins a discovery subprocess or raises if it stays alive."""
    proc.join(timeout=_STANDARD_DP_DISCOVERY_JOIN_TIMEOUT_S)
    if proc.is_alive():
        _terminate_discovery_process(proc)
        raise RuntimeError(message)


def _resolve_standard_dp_visible_device_groups(
    vllm_config: "VllmConfig",
) -> list[StandardDPAssignmentT] | None:
    """Resolves single-host TT device groups for standard DP.

    Notes
    -----
    The discovery work runs in a spawned helper so the parent avoids holding TT
    chip locks before worker startup.

    Examples
    --------
    >>> _resolve_standard_dp_visible_device_groups(vllm_config)
    [("0,1,2,3", (1, 4)), ...]
    """
    parallel_config = vllm_config.parallel_config
    if parallel_config.data_parallel_size <= 1 or _uses_explicit_tt_mpi_launch(
        vllm_config
    ):
        return None

    mp_ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
    proc = mp_ctx.Process(
        target=_run_standard_dp_visible_device_group_discovery,
        args=(
            child_conn,
            os.environ.get("MESH_DEVICE"),
            parallel_config.data_parallel_size,
        ),
        name="TTVisibleDevicesDiscovery",
    )
    proc.start()
    child_conn.close()

    try:
        if not parent_conn.poll(_STANDARD_DP_DISCOVERY_RECV_TIMEOUT_S):
            _terminate_discovery_process(proc)
            raise RuntimeError(
                "TT standard-DP device discovery subprocess timed out after "
                f"{_STANDARD_DP_DISCOVERY_RECV_TIMEOUT_S:.1f}s waiting for "
                "device groups"
            )
        status, payload = parent_conn.recv()

    except EOFError as exc:
        _join_discovery_process_or_raise(
            proc,
            "TT standard-DP device discovery subprocess did not exit cleanly "
            "after closing its result pipe",
        )
        raise RuntimeError(
            "TT standard-DP device discovery subprocess exited before returning "
            "device groups"
        ) from exc

    finally:
        parent_conn.close()

    _join_discovery_process_or_raise(
        proc,
        "TT standard-DP device discovery subprocess did not exit after "
        "returning device groups",
    )
    if proc.exitcode not in (0, None):
        raise RuntimeError(
            "TT standard-DP device discovery subprocess failed with exit code "
            f"{proc.exitcode}"
        )
    if status != "ok":
        raise RuntimeError(f"TT standard-DP device discovery failed: {payload}")

    return payload


# GPT-OSS is served by the tt_transformers generator, which drives data
# parallelism inside a single process -- either user-row sharding on multi-row
# meshes or one Generator over per-DP submeshes. Multi-process DP is therefore
# unnecessary: --data_parallel_size folds into in-process TT lanes, the same as
# the Galaxy generators.
_GPT_OSS_ARCH = "GptOssForCausalLM"


def _model_folds_dp_into_lanes(model_class) -> bool:
    """Whether the model's ``--data_parallel_size`` folds into in-process lanes.

    True for the Galaxy generators and for GPT-OSS, both of which drive data
    parallelism within a single process. Other models keep standard
    multi-process DP.
    """
    if _galaxy_generator_version() is not None:
        return True
    return getattr(model_class, "__name__", None) == _GPT_OSS_ARCH


def _collapse_parallel_config_to_single_process(parallel_config) -> None:
    """Reset DP-derived ParallelConfig fields to single-process values.

    ``ParallelConfig.__post_init__`` has already derived multi-process DP state
    (rank, local size, master port, LB mode) from ``data_parallel_size`` by the
    time the platform hook runs. When we fold DP into single-process TT lanes we
    must undo that so vLLM does not stand up multi-process DP coordination.
    ``world_size`` stays 1 because the TT backend requires
    ``tensor_parallel_size == pipeline_parallel_size == 1`` and DP does not
    multiply it (no external launcher), so ``world_size_across_dp`` collapses to
    1 automatically once ``data_parallel_size`` is reset.

    ``data_parallel_rank_local`` must be ``0`` here because a single-process
    lane run owns one device mesh. Standard DP ownership is handled separately:
    upstream rewrites each dense DP subprocess to a local DP=1 view while
    preserving its shard identity, and the TT worker uses that preserved shard
    index to decide that every standard-DP rank owns its own mesh/model/KV.
    ``0`` is also the value a genuine single-process run resolves to
    (``ParallelConfig.__post_init__`` defaults ``data_parallel_rank_local``
    from ``VLLM_DP_RANK_LOCAL`` / ``VLLM_DP_RANK``, both ``0``).
    """
    parallel_config.data_parallel_size = 1
    parallel_config.data_parallel_size_local = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_rank_local = 0
    parallel_config.data_parallel_index = 0
    parallel_config.data_parallel_external_lb = False
    parallel_config.data_parallel_hybrid_lb = False

    # A single-process lane run owns one in-process worker, so pin the uniproc
    # executor. Newer vLLM derives it from ``world_size_across_dp`` and latches
    # "mp" from the user's --data_parallel_size before this hook runs; pinning
    # "uni" keeps lane-DP single-process there too, so the worker's runtime
    # ``num_gpu_blocks_override`` still reaches the engine's KV-cache sizing.
    parallel_config.distributed_executor_backend = "uni"


def _convert_dp_to_lanes(vllm_config: "VllmConfig", model_class=None) -> None:
    """Transparently convert multi-process DP into in-process TT lanes.

    Models that run as a single shared device execute on one mesh -- the Galaxy
    generators (``llama3_70b_galaxy``, ``qwen3_32b_galaxy``) and GPT-OSS under
    user-row sharding -- do not need multi-process DP. Rather than asking users
    to migrate flags, we run ``--data_parallel_size N`` as ``N`` in-process
    lanes: record the resolved lane count and reset ``data_parallel_size`` to 1.

    ``max_num_seqs`` is per-DP-rank under multi-process DP but global under lane
    mode, so it is scaled by the lane count on the way in. Lane mode then
    partitions that global capacity evenly across lanes, keeping the per-lane
    capacity at the value the user asked for (e.g. ``--data_parallel_size 4
    --max_num_seqs 8`` becomes 4 lanes, each with max 8 seqs, global max 32).

    No-op unless ``data_parallel_size > 1`` and the model folds DP into lanes
    (``_model_folds_dp_into_lanes``). Idempotent: after conversion
    ``data_parallel_size == 1``, so re-entry short-circuits.
    """
    parallel_config = vllm_config.parallel_config
    data_parallel_size = parallel_config.data_parallel_size
    if data_parallel_size <= 1:
        return
    if not _model_folds_dp_into_lanes(model_class):
        return

    lanes = data_parallel_size
    scheduler_config = vllm_config.scheduler_config
    per_lane_max_num_seqs = int(scheduler_config.max_num_seqs)
    global_max_num_seqs = per_lane_max_num_seqs * lanes

    store_tt_lane_count(vllm_config, lanes)
    scheduler_config.max_num_seqs = global_max_num_seqs
    _collapse_parallel_config_to_single_process(parallel_config)

    logger.info(
        "Model requested DP (--data_parallel_size=%d) but runs as a "
        "single device execute; running single-process TT lane-DP instead "
        "(%d lanes, per-lane max_num_seqs=%d, global max_num_seqs=%d).",
        data_parallel_size,
        lanes,
        per_lane_max_num_seqs,
        global_max_num_seqs,
    )


def _register_model_if_missing(ModelRegistry, model_arch: str, model_path: str) -> None:
    """Register `model_arch` only if not already registered.

    This keeps TT model registration idempotent across multiple call sites
    (e.g. APIServer pre-register, TT worker import, and platform config hook).
    """
    if model_arch not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(model_arch, model_path)


def _should_pre_register_tt_test_models_from_cli() -> bool:
    """Return True iff CLI TT config enables TT test models.

    `TTPlatform.pre_register_and_update()` runs before `VllmConfig` is
    constructed, but ModelConfig may inspect architectures early.
    """
    argv = list(sys.argv[1:])

    def _parse_namespaced_config(raw: str) -> dict | None:
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    cfg = None
    for i, arg in enumerate(argv):
        if "=" in arg:
            flag, value = arg.split("=", 1)
            if flag.replace("_", "-") == "--additional-config":
                cfg = _parse_namespaced_config(value) or cfg
        elif arg.replace("_", "-") == "--additional-config" and i + 1 < len(argv):
            cfg = _parse_namespaced_config(argv[i + 1]) or cfg

    tt_config = cfg.get("tt", {}) if cfg else {}
    return bool(
        isinstance(tt_config, dict) and tt_config.get("register_test_models") is True
    )


def _install_tt_harmony_truncation_patch() -> None:
    """Use right truncation for TT GPT-OSS tokenizers.

    GPT-OSS harmony prompts have important template/control tokens at the
    beginning. Left truncation can remove those tokens when prompt truncation is
    requested, so TT keeps the prefix and truncates from the right for these
    models.

    TODO: remove this once fixed in vLLM core.
    """
    import vllm.tokenizers.registry as tokenizer_registry

    if hasattr(tokenizer_registry, "_tt_original_tokenizer_args_from_config"):
        return

    original = tokenizer_registry.tokenizer_args_from_config
    tokenizer_registry._tt_original_tokenizer_args_from_config = original

    def tokenizer_args_from_config_tt(config, **kwargs):
        tokenizer_mode, tokenizer_name, args, tokenizer_kwargs = original(
            config, **kwargs
        )
        if (
            "truncation_side" not in tokenizer_kwargs
            and config.runner_type in ("generate", "draft")
            and "gpt-oss" in str(tokenizer_name or "").lower()
        ):
            tokenizer_kwargs["truncation_side"] = "right"
        return tokenizer_mode, tokenizer_name, args, tokenizer_kwargs

    tokenizer_registry.tokenizer_args_from_config = tokenizer_args_from_config_tt

    renderer_registry = sys.modules.get("vllm.renderers.registry")
    if renderer_registry is not None:
        renderer_registry.tokenizer_args_from_config = tokenizer_args_from_config_tt


def _neutralize_model_owned_sampling(params) -> list[str]:
    """Reset HTTP sampling controls on the cloned per-request SamplingParams.

    The model owns its Gumbel sampler and temperature schedule, but common
    OpenAI clients still send transport sampling controls, so they are
    accepted and ignored. Returns the neutralized fields for logging.
    """
    ignored = []
    if params.temperature != 1.0:
        ignored.append(f"temperature={params.temperature!r}")
        params.temperature = 1.0
    if params.top_p != 1.0:
        ignored.append(f"top_p={params.top_p!r}")
        params.top_p = 1.0
    if params.top_k not in (0, -1):
        ignored.append(f"top_k={params.top_k!r}")
        params.top_k = 0
    if params.min_p != 0.0:
        ignored.append(f"min_p={params.min_p!r}")
        params.min_p = 0.0
    if params.seed is not None:
        ignored.append(f"seed={params.seed!r}")
        params.seed = None
    if params.presence_penalty != 0.0:
        ignored.append(f"presence_penalty={params.presence_penalty!r}")
        params.presence_penalty = 0.0
    if params.frequency_penalty != 0.0:
        ignored.append(f"frequency_penalty={params.frequency_penalty!r}")
        params.frequency_penalty = 0.0
    if params.repetition_penalty != 1.0:
        ignored.append(f"repetition_penalty={params.repetition_penalty!r}")
        params.repetition_penalty = 1.0
    return ignored


def _install_block_output_input_processor_patch() -> None:
    """Reject unsupported resumable requests and own block-output defaults.

    vLLM 0.24 validates caller-owned SamplingParams before cloning them, then
    resolves max_tokens=None on the clone. Patch that narrow boundary so TT can
    reject resumable streaming-input chunks before EngineCore admits any, and
    neutralize model-owned sampling controls and round the whole-canvas default
    on the clone, without mutating the shared caller-owned input.
    (vLLM creates the streaming-input session wrapper without the ``resumable``
    flag, so the session opens and the client sees the error on its first
    chunk.)

    TODO: remove once vLLM exposes a platform hook for per-request defaults.
    """
    import vllm.v1.engine.input_processor as input_processor
    from vllm.sampling_params import SamplingParams

    if hasattr(input_processor, "_tt_original_process_inputs"):
        return

    original = input_processor.InputProcessor.process_inputs
    input_processor._tt_original_process_inputs = original

    # ``resumable`` is positional-or-keyword in vLLM 0.24, so a positional
    # caller lands it in the wrapper's ``*args``; locate it in the original
    # signature once. The wrapper binds self/request_id/prompt/params itself.
    original_parameters = list(inspect.signature(original).parameters)
    resumable_args_index = (
        original_parameters.index("resumable") - 4
        if "resumable" in original_parameters
        else None
    )

    def process_inputs_tt(self, request_id, prompt, params, *args, **kwargs):
        output_size = get_tt_output_tokens_per_step(self.vllm_config)
        if output_size > 1:
            resumable = (
                args[resumable_args_index]
                if resumable_args_index is not None and len(args) > resumable_args_index
                else kwargs.get("resumable", False)
            )
            if resumable:
                raise ValueError(
                    "TT block-output models do not support resumable "
                    "streaming-input requests"
                )

        unresolved_max_tokens = (
            isinstance(params, SamplingParams) and params.max_tokens is None
        )
        request = original(self, request_id, prompt, params, *args, **kwargs)

        cloned_params = request.sampling_params
        if output_size == 1 or cloned_params is None:
            return request

        ignored = _neutralize_model_owned_sampling(cloned_params)
        if ignored:
            logger.warning_once(
                "This block-output model uses its model-owned sampler; HTTP "
                "sampling controls are accepted but ignored."
            )
            logger.debug(
                "Ignoring unsupported sampling controls for block-output model: %s",
                "; ".join(ignored),
            )

        if unresolved_max_tokens and cloned_params.max_tokens is not None:
            rounded = cloned_params.max_tokens // output_size * output_size
            if rounded == 0:
                # Re-validate the resolved default: inputs without
                # prompt_token_ids (e.g. prompt embeds) skip the canvas check
                # in validate_request, and a whole-canvas default of zero
                # would otherwise be admitted.
                raise ValueError(
                    f"Request leaves fewer than one physical {output_size}"
                    "-token output canvas within the max model length. Use a "
                    "shorter prompt or a larger max model length."
                )
            cloned_params.max_tokens = rounded
        return request

    input_processor.InputProcessor.process_inputs = process_inputs_tt


def _iter_extra_model_bundles():
    """Yield ``(folder, arch, main_class)`` for each bundle under ``EXTRA_MODELS_DIR``.

    ``EXTRA_MODELS_DIR`` is a directory of self-contained per-model bundle
    folders. Each folder holds a ``vllm_metadata.json`` (``arch`` = HF
    architecture name, ``main_class`` = ``"module:Class"`` implementing the vLLM
    generator adapter) plus the adapter class and its dependencies. Any
    distribution tool (e.g. tt-kernel) can drop a bundle folder here and have it
    registered with no source edit to this plugin. Malformed / incomplete folders
    are skipped with a warning.
    """
    base = os.getenv("EXTRA_MODELS_DIR")
    if not base:
        return

    if not os.path.isdir(base):
        logger.warning("EXTRA_MODELS_DIR=%s is not a directory; ignoring.", base)
        return

    for name in sorted(os.listdir(base)):
        folder = os.path.join(base, name)
        if not os.path.isdir(folder):
            continue

        meta_path = os.path.join(folder, "vllm_metadata.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path) as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Skipping %s: cannot read vllm_metadata.json (%s)", folder, exc
            )
            continue

        arch = data.get("arch")
        main_class = data.get("main_class")
        if not arch or not main_class:
            logger.warning(
                "Skipping %s: vllm_metadata.json needs 'arch' and 'main_class'.",
                folder,
            )
            continue

        yield folder, arch, main_class


def _register_models_from_extra_dir(ModelRegistry) -> int:
    """Register every model found under ``EXTRA_MODELS_DIR``; return the count.

    Registration is lazy (a ``"module:Class"`` string resolved by vLLM later), so
    a bundle folder that carries its own adapter module must stay importable when
    that resolution happens. We ``append`` the folder to ``sys.path`` (never
    ``insert(0)``, so an installed package of the same name always wins and
    nothing is shadowed); built-in adapters given as a full dotted path resolve
    normally and need no path entry. The arch is registered under the plugin's
    ``TT``-prefixed convention (mirroring ``check_and_update_config``).
    """
    count = 0

    for folder, arch, main_class in _iter_extra_model_bundles():
        if folder not in sys.path:
            sys.path.append(folder)

        tt_arch = arch if arch.startswith("TT") else "TT" + arch
        _register_model_if_missing(ModelRegistry, tt_arch, main_class)

        logger.info(
            "Registered TT model %s -> %s (from EXTRA_MODELS_DIR/%s)",
            tt_arch,
            main_class,
            os.path.basename(folder),
        )

        count += 1

    return count


def _builtin_models_enabled() -> bool:
    """Whether to register the built-in (hard-coded) TT model map.

    Defaults to enabled for backward compatibility. Set
    ``TT_VLLM_BUILTIN_MODELS=0`` to rely solely on ``EXTRA_MODELS_DIR`` (the
    intended end-state once all models ship as bundles). Any of 0/false/no/off
    disables it.
    """
    val = os.getenv("TT_VLLM_BUILTIN_MODELS")
    if val is None:
        return True
    return val.strip().lower() not in ("0", "false", "no", "off")


def register_tt_models(register_test_models=False) -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    # Dynamic hook: register any bundles dropped under EXTRA_MODELS_DIR. Runs
    # first so a distributed bundle can supply a model without touching this file.
    _register_models_from_extra_dir(ModelRegistry)

    # Built-in map. Kept for compatibility; disable with TT_VLLM_BUILTIN_MODELS=0.
    if not _builtin_models_enabled():
        if register_test_models:
            register_tt_test_models()
        return

    llama_text_version = os.getenv("TT_LLAMA_TEXT_VER", "tt_transformers")
    if llama_text_version == "tt_transformers":
        path_llama_text = "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    elif llama_text_version == "llama3_70b_galaxy":
        path_llama_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:LlamaForCausalLM"
        )
    elif llama_text_version == "llama2_70b":
        path_llama_text = (
            "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported TT Llama version: {llama_text_version}, "
            "pick one of [tt_transformers, llama3_70b_galaxy, llama2_70b]"
        )

    # Llama3.1/3.2 - Text
    _register_model_if_missing(ModelRegistry, "TTLlamaForCausalLM", path_llama_text)

    # Llama3.2 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTMllamaForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration",
    )

    # Qwen2.5 - Text
    path_qwen_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    _register_model_if_missing(ModelRegistry, "TTQwen2ForCausalLM", path_qwen_text)

    # Qwen3 - Text
    qwen3_text_version = os.getenv("TT_QWEN3_TEXT_VER", "tt_transformers")
    if qwen3_text_version == "tt_transformers":
        path_qwen3_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    elif qwen3_text_version == "qwen3_32b_galaxy":
        path_qwen3_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:QwenForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported TT Qwen3 version: {qwen3_text_version}, "
            "pick one of [tt_transformers, qwen3_32b_galaxy]"
        )

    _register_model_if_missing(ModelRegistry, "TTQwen3ForCausalLM", path_qwen3_text)

    # Qwen3.5 - Text
    qwen35_text_version = os.getenv("TT_QWEN35_TEXT_VER", "qwen36_blackhole")
    if qwen35_text_version == "qwen36_blackhole":
        path_qwen35_text = (
            "models.demos.blackhole.qwen36.tt.qwen36_vllm:Qwen36ForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported TT Qwen3.5 version: {qwen35_text_version}, "
            "pick one of [qwen36_blackhole]"
        )

    _register_model_if_missing(
        ModelRegistry, "TTQwen3_5ForConditionalGeneration", path_qwen35_text
    )

    # Qwen2.5 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTQwen2_5_VLForConditionalGeneration",
        "models.demos.qwen25_vl.tt.generator_vllm:Qwen2_5_VLForConditionalGeneration",
    )

    # Qwen3 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTQwen3VLForConditionalGeneration",
        "models.demos.qwen3_vl.tt.generator_vllm:Qwen3VLForConditionalGeneration",
    )

    # Mistral - Text only
    _register_model_if_missing(
        ModelRegistry,
        "TTMistralForCausalLM",
        "models.tt_transformers.tt.generator_vllm:MistralForCausalLM",
    )

    # Mistral 3 - Multimodal (Vision + Text)
    _register_model_if_missing(
        ModelRegistry,
        "TTMistral3ForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:Mistral3ForConditionalGeneration",
    )

    # Gemma3
    _register_model_if_missing(
        ModelRegistry,
        "TTGemma3ForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration",
    )

    # Gemma4 — text-only TT bridge.
    #
    # Gemma4 isn't in vLLM's upstream registry, so without an entry here
    # the upstream architecture resolver falls back to
    # ``TransformersMultiModalForCausalLM`` (because ``hf_config !=
    # hf_text_config`` for Gemma4's nested config — see
    # ``ModelConfig._get_transformers_backend_cls``) and crashes on the
    # ``_processor_factory`` assertion in the multimodal registry. The
    # plugin's later ``TT``-prefix logic runs after that resolution, so
    # it can't help.
    #
    # We register the plain HF arch names directly so upstream resolution
    # finds our class. Since ``Gemma4ForCausalLM`` (the TT class) does not
    # use ``SupportsMultiModal``, vLLM's ``_model_info.supports_multimodal``
    # is False, ``multimodal_config`` is not populated, and the request
    # path stays text-only — which matches what the TT model implements.
    # The ``TT``-prefixed aliases satisfy the plugin's later validation
    # in ``check_and_update_config`` so no override is needed.
    #
    # The 12B checkpoint is the "unified" multimodal variant: its config
    # declares ``architectures: ['Gemma4UnifiedForConditionalGeneration']``
    # with ``model_type: gemma4_unified`` and nested text/vision/audio
    # configs. Without the unified arch registered, the same nested-config
    # fallback resolves it to ``TransformersMultiModalForCausalLM``. We map
    # the unified arch (and its ``TT`` alias) to the same text-only TT class
    # so text-only inference runs on the unified checkpoint.
    _gemma4_target = "models.demos.gemma4.tt.generator_vllm:Gemma4ForCausalLM"
    for arch in (
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
        "TTGemma4ForCausalLM",
        "TTGemma4ForConditionalGeneration",
        "TTGemma4UnifiedForConditionalGeneration",
    ):
        _register_model_if_missing(ModelRegistry, arch, _gemma4_target)

    # DiffusionGemma emits one complete 256-token canvas per model step. Both
    # the checkpoint's HF architecture and TT-prefixed aliases must resolve
    # before ModelConfig falls back to an unrelated Transformers backend.
    _diffusion_gemma_target = (
        "models.experimental.diffusion_gemma.tt.generator_vllm:"
        "DiffusionGemmaForCausalLM"
    )
    for arch in (
        "DiffusionGemmaForBlockDiffusion",
        "DiffusionGemmaForCausalLM",
        "TTDiffusionGemmaForBlockDiffusion",
        "TTDiffusionGemmaForCausalLM",
    ):
        _register_model_if_missing(ModelRegistry, arch, _diffusion_gemma_target)

    # DeepseekV3
    _register_model_if_missing(
        ModelRegistry,
        "TTDeepseekV3ForCausalLM",
        "models.demos.deepseek_v3.tt.generator_vllm:DeepseekV3ForCausalLM",
    )

    # GPT-OSS
    _register_model_if_missing(
        ModelRegistry,
        "TTGptOssForCausalLM",
        "models.tt_transformers.tt.generator_vllm:GptOssForCausalLM",
    )

    # Optionally register test models if explicitly enabled
    if register_test_models:
        register_tt_test_models()


def register_tt_test_models():
    """Register non-production TT models which are only used for testing."""
    from vllm.model_executor.models.registry import ModelRegistry

    # Fake model for testing multi-process inference on T3000
    _register_model_if_missing(
        ModelRegistry,
        "TTDummyT3000MultiProcessModel",
        "models.vllm_test_utils.t3000_multiproc_test.test_model:DummyT3000MultiProcessModel",
    )

    # Fake model which does nothing, for measuring vLLM host overheads
    _register_model_if_missing(
        ModelRegistry,
        "TTDummyNoOpModel",
        "models.vllm_test_utils.no_op_test.test_model:DummyNoOpModel",
    )

    # Fake model for testing multi-host inference on dual Galaxy
    _register_model_if_missing(
        ModelRegistry,
        "TTDummyDualGlxModel",
        "models.vllm_test_utils.dual_glx_ccl_test.test_model:DummyDualGlxModel",
    )


class TTPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "tt"
    device_type: str = "tt"
    device_control_env_var: str = "TT_VISIBLE_DEVICES"
    _standard_dp_visible_device_groups: ClassVar[list[str] | None] = None
    _standard_dp_mesh_grids: ClassVar[dict[str, tuple[int, int]]] = {}
    sample_on_device_mode: ClassVar[Literal["all", "decode_only"] | None] = None
    output_tokens_per_step: ClassVar[int] = 1
    block_model_config: ClassVar[_MaxModelLenConfig | None] = None
    # Disable torch.compile on TT platform - the triton version in tt-metal
    # is incompatible with torch's inductor backend.
    simple_compile_backend: str = "eager"

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int) -> str | int:
        """Map a DP rank to its whole comma-joined TT device group.

        Deviates from upstream, where ``device_id`` is a logical device index and
        the return is one physical id. Sound only because TT pins ``world_size``
        to 1, so upstream asks for exactly one id per DP rank and keeps it as a
        one-element list: ``assigned_physical_gpu_ids`` carries a group string,
        not the ``list[int]`` it declares. Anything reading that list as
        per-device, its length as a device count or its entries as ints, is
        wrong for TT.
        """
        groups = cls._standard_dp_visible_device_groups
        if groups is not None:
            return groups[device_id]
        return super().device_id_to_physical_device_id(device_id)

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        # Hybrid models (Gemma3/4, GPT-OSS) opt in to upstream's HMA via
        # ``HybridAttentionForCausalLM.get_kv_cache_spec`` so layers from
        # different attention groups can share DRAM tensors. Without this
        # override the base ``Platform`` returns ``False`` and HMA collapses
        # every ``SlidingWindowSpec`` back to ``FullAttentionSpec`` in
        # ``unify_hybrid_kv_cache_specs`` — defeating the entire point.
        return True

    @classmethod
    def pre_register_and_update(
        cls, parser: FlexibleArgumentParser | None = None
    ) -> None:
        # Called during CLI/parser setup (APIServer). ModelConfig may
        # validate/inspect architectures before VllmConfig is constructed in
        # this process, so we must ensure TT test models are registered early
        # when explicitly requested via CLI override.
        super().pre_register_and_update(parser)
        _install_tt_harmony_truncation_patch()
        if _should_pre_register_tt_test_models_from_cli():
            register_tt_test_models()

    @classmethod
    def import_kernels(cls) -> None:
        # Do not import vllm._C or vllm._moe_C
        pass

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return True

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def set_device(cls, device) -> None:
        if device is None:
            return

        import ttnn

        get_default_device = getattr(ttnn, "GetDefaultDevice", None)
        current_device = get_default_device() if callable(get_default_device) else None
        if current_device is not device:
            ttnn.SetDefaultDevice(device)

    @classmethod
    def _resolve_output_tokens_per_step(cls, model_class: type) -> int:
        """Validate and return a model's committed output-width capability."""
        model_capabilities: dict | None = getattr(
            model_class, "model_capabilities", None
        )
        output_tokens_per_step = (
            model_capabilities.get("output_tokens_per_step", 1)
            if model_capabilities
            else 1
        )
        if (
            isinstance(output_tokens_per_step, bool)
            or not isinstance(output_tokens_per_step, int)
            or output_tokens_per_step < 1
        ):
            raise ValueError(
                f"Invalid output_tokens_per_step={output_tokens_per_step!r} for "
                f"{model_class.__module__}.{model_class.__name__}; "
                "expected an integer >= 1"
            )
        return output_tokens_per_step

    @classmethod
    def _get_block_model_max_len(cls) -> int | None:
        """Read the live frontend limit, including vLLM auto-fit updates."""
        model_config = cls.block_model_config
        if model_config is None:
            return None
        return int(model_config.max_model_len)

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        # The standalone TT plugin implements the vLLM V1 runner only. vLLM
        # 0.24 otherwise force-selects its Triton-only V2 runner for every HF
        # config with ``canvas_length`` before applying the normal no-Triton
        # fallback. Pinning this in the platform hook keeps diffusion models on
        # the runner the TT worker actually implements.
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        _install_tt_harmony_truncation_patch()
        cls._standard_dp_visible_device_groups = None
        cls._standard_dp_mesh_grids = {}
        if _uses_explicit_tt_mpi_launch(vllm_config):
            raise RuntimeError(
                "Explicit TT MPI/rank-binding/multinode launch is unsupported "
                "with vLLM 0.24: that release does not invoke the "
                "CoreEngineLauncher extension required to start tt-run. "
                "Remove tt.rank_binding/tt.mpi_args and multinode settings."
            )
        if vllm_config.scheduler_config.enable_chunked_prefill:
            logger.info("Chunked prefill is not yet supported for TT backend")
            vllm_config.scheduler_config.enable_chunked_prefill = False
            # vLLM does this bump silently earlier
            # if chunked prefill is already disabled,
            # and max_num_batched_tokens is not explicitly set.
            # We can't know if it was specified
            # or the default, hence the warning.
            if (
                vllm_config.scheduler_config.max_num_batched_tokens
                < vllm_config.model_config.max_model_len
            ):
                logger.warning(
                    "max_num_batched_tokens=%d < max_model_len=%d with chunked prefill "
                    "disabled, bumping max_num_batched_tokens to match.",
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    vllm_config.model_config.max_model_len,
                )
                vllm_config.scheduler_config.max_num_batched_tokens = (
                    vllm_config.model_config.max_model_len
                )

        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for TT backend"
        )
        assert (
            vllm_config.parallel_config.tensor_parallel_size == 1
            and vllm_config.parallel_config.pipeline_parallel_size == 1
        ), "TT backend does not support distributed execution"
        assert not vllm_config.lora_config, "LoRA is not supported for TT backend"

        # Device computes top-32 logprobs but the OpenAI API limits to 20
        MAX_TOP_K = 20

        model_config = vllm_config.model_config
        if model_config.max_logprobs > MAX_TOP_K:
            logger.warning(
                "max_logprobs=%d exceeds TT device limit of %d, clamping to %d",
                model_config.max_logprobs,
                MAX_TOP_K,
                MAX_TOP_K,
            )
            model_config.max_logprobs = MAX_TOP_K

        # Force the grammar backends to emit compact JSON. xgrammar and guidance
        # allow arbitrary inter-field whitespace by default; under greedy decoding
        # the model can pick a whitespace token as the argmax indefinitely,
        # exhausting the token budget before it emits a property name and
        # returning truncated, unparseable JSON. Masking whitespace out of the
        # grammar makes that loop structurally impossible for any decoding
        # strategy. Backend stays "auto" so schemas xgrammar cannot compile still
        # fall back to guidance (which also honors this flag); outlines and
        # lm-format-enforcer ignore it.
        vllm_config.structured_outputs_config.disable_any_whitespace = True

        # Import and register models from tt-metal.
        #
        # NOTE: We also register TT models early in `vllm_tt_plugin.worker`
        # (at module import time). That registration is required to handle
        # engine/worker subprocess startup ordering where model architectures
        # may be inspected (e.g. multimodal processor cache init) before this
        # `check_and_update_config()` hook is reached in that process.
        tt_config = get_tt_config(vllm_config)
        register_test_models = False
        if tt_config and "register_test_models" in tt_config:
            register_test_models = tt_config["register_test_models"]
            assert register_test_models in [True, False], (
                f"Invalid option register_test_models: {register_test_models}"
            )
        register_tt_models(register_test_models)

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_tt_plugin.worker.TTWorker"
        parallel_config.engine_core_cls = "vllm.v1.engine.core.EngineCore"
        parallel_config.engine_core_proc_cls = "vllm.v1.engine.core.EngineCoreProc"

        # For TT models, prepend "TT" to the architecture name,
        # e.g. "TTLlamaForCausalLM"
        arch_names = vllm_config.model_config.hf_config.architectures
        is_diffusion_gemma = any(
            arch.removeprefix("TT")
            in ("DiffusionGemmaForBlockDiffusion", "DiffusionGemmaForCausalLM")
            for arch in arch_names
        )
        for i in range(len(arch_names)):
            if not arch_names[i].startswith("TT"):
                arch_names[i] = "TT" + arch_names[i]

        # Verify that the TT architecture is registered in the model registry
        from vllm.model_executor.models.registry import ModelRegistry

        supported_archs = ModelRegistry.get_supported_archs()
        if not any(arch_name in supported_archs for arch_name in arch_names):
            tt_archs = sorted(
                [arch for arch in supported_archs if arch.startswith("TT")]
            )
            raise ValueError(
                f"No TT model architecture is registered for "
                f"model: '{vllm_config.model_config.model}'. "
                f"Available TT architectures: {tt_archs}"
            )

        # Setting attributes on the class level is kind of hacky, but
        # it's the only way to make validate_request depend on vllm_config
        # This is needed to catch incompatible requests early enough
        # to return an error instead of crashing.
        # TODO move this to tt_model_runner when request validation
        # stops depending on vllm_config

        if tt_config is not None and "sample_on_device_mode" in tt_config:
            sample_on_device_mode = tt_config["sample_on_device_mode"]
            assert sample_on_device_mode in [
                "all",
                "decode_only",
            ], f"Invalid sample_on_device_mode: {sample_on_device_mode}"
        else:
            sample_on_device_mode = None
        cls.sample_on_device_mode = sample_on_device_mode  # type: ignore[attr-defined]

        # Compat sampling uses the full vLLM sampling pipeline,
        # with logit processors and sampler, instead of our custom sampling.
        # It is enabled only if any of the requests in the batch requires it,
        # or if always_compat_sampling is enabled.

        always_compat_sampling = False
        if tt_config is not None and "always_compat_sampling" in tt_config:
            always_compat_sampling = tt_config["always_compat_sampling"]
            assert always_compat_sampling in [True, False], (
                "always_compat_sampling must be a boolean"
            )
            if always_compat_sampling:
                raise ValueError(
                    "always_compat_sampling is not yet supported for V1 TT backend."
                )
        cls.always_compat_sampling = always_compat_sampling  # type: ignore[attr-defined]

        # must perform local import to get around circular import
        from vllm.model_executor.model_loader.utils import get_model_architecture

        model_class, _ = get_model_architecture(vllm_config.model_config)

        # Get model capabilities from the class
        model_capabilities: dict | None = getattr(
            model_class, "model_capabilities", None
        )
        output_tokens_per_step = cls._resolve_output_tokens_per_step(model_class)
        is_block_output_model = output_tokens_per_step > 1
        if is_diffusion_gemma and not is_block_output_model:
            raise ValueError(
                "DiffusionGemma must declare output_tokens_per_step > 1 "
                "in model_capabilities"
            )
        if is_block_output_model:
            # Upstream 0.24 recognizes DiffusionGemma's original HF
            # architecture before this platform hook runs and injects a
            # DiffusionConfig(canvas_length=256). Scheduler interprets that
            # canvas as speculative draft tokens, while this plugin separately
            # reserves the model-owned output block. Clear the upstream
            # diffusion path so one physical canvas is accounted exactly once.
            if vllm_config.diffusion_config is not None:
                logger.info(
                    "Block-output model owns diffusion scheduling; disabling "
                    "upstream DiffusionConfig speculative-token accounting."
                )
                vllm_config.diffusion_config = None
            if model_config.max_model_len < output_tokens_per_step:
                raise ValueError(
                    f"max_model_len={model_config.max_model_len} must be at least "
                    f"output_tokens_per_step={output_tokens_per_step}"
                )
            if vllm_config.scheduler_config.max_num_seqs != 1:
                raise ValueError(
                    "Block-output models currently own one model-side request "
                    "state and require --max-num-seqs 1"
                )
            if (
                parallel_config.data_parallel_size != 1
                or get_tt_data_parallel_size(vllm_config) != 1
            ):
                raise ValueError(
                    "Block-output models do not yet support data parallelism; "
                    "use --data-parallel-size 1"
                )
            if vllm_config.cache_config.enable_prefix_caching:
                raise ValueError(
                    "Block-output models do not support vLLM automatic prefix "
                    "caching; disable prefix caching"
                )
            if model_config.logits_processors:
                raise ValueError(
                    "Block-output models do not support --logits-processors "
                    "because output is sampled inside the model"
                )
            if vllm_config.scheduler_config.async_scheduling:
                raise ValueError(
                    "Block-output models currently support synchronous serving "
                    "only; launch with --no-async-scheduling"
                )
            vllm_config.scheduler_config.long_prefill_token_threshold = 0
            if model_config.generation_config == "auto":
                logger.info(
                    "Block-output model owns generation defaults; normalizing "
                    "--generation-config auto to vllm."
                )
                model_config.generation_config = "vllm"

        store_tt_output_tokens_per_step(vllm_config, output_tokens_per_step)
        cls.output_tokens_per_step = output_tokens_per_step
        if is_block_output_model:
            _install_block_output_input_processor_patch()
        # vLLM 0.24 syncs an auto-fitted max_model_len back into this same
        # frontend ModelConfig object through EngineCoreReadyResponse. Retain
        # the object rather than snapshotting its pre-fit integer.
        cls.block_model_config = model_config if is_block_output_model else None

        # A model either supports the full on-device sampling pipeline or it
        # doesn't — there is no greedy-only mode. Models opt in by setting
        # `supports_sample_on_device` in their `model_capabilities` dict.
        supports_sample_on_device = (
            model_capabilities.get("supports_sample_on_device", False)
            if model_capabilities
            else False
        )
        if sample_on_device_mode is not None and not supports_sample_on_device:
            raise ValueError(
                f"sample_on_device_mode={sample_on_device_mode!r} was requested, "
                f"but model {model_class.__name__} "
                f"({model_class.__module__}) does not support on-device sampling. "
                "Unset sample_on_device_mode or use a model that supports it."
            )
        if is_block_output_model and sample_on_device_mode != "all":
            raise ValueError(
                "Block-output models emit complete multi-token outputs from "
                "their model-owned sampler and require "
                f'sample_on_device_mode="all"; got {sample_on_device_mode!r}'
            )

        # Model-gated async scheduling. Async overlap requires generators that
        # support split decode submission via `decode_forward(...,
        # read_from_device=False)` followed by `read_decode_output(...,
        # async_read=True)`.
        supports_async_decode = (
            model_capabilities.get("supports_async_decode", False)
            if model_capabilities
            else False
        )
        if vllm_config.scheduler_config.async_scheduling and not supports_async_decode:
            logger.warning(
                "Async scheduling was requested, but TT model %s (%s) does not "
                "declare support (`model_capabilities['supports_async_decode']`). "
                "Disabling async scheduling.",
                model_class.__name__,
                model_class.__module__,
            )
            vllm_config.scheduler_config.async_scheduling = False

        # Single-execute models (Galaxy generators, GPT-OSS under user-row
        # sharding) run one shared device execute on the full mesh, so
        # multi-process DP is folded transparently into single-process TT lanes
        # -- users keep passing --data_parallel_size with no other flag changes.
        # Must run before the validation/routing below so the lane path is
        # selected. model_class carries the single-execute decision for GPT-OSS.
        _convert_dp_to_lanes(vllm_config, model_class)

        is_lane_mode = uses_tt_lane_coordinator(vllm_config)
        if (
            getattr(model_config, "is_moe", False)
            and parallel_config.data_parallel_size > 1
            and not is_lane_mode
        ):
            raise ValueError(
                "TT standard DP does not support MoE models yet. "
                "Use data_parallel_size=1."
            )

        if is_lane_mode:
            # Fail fast on misconfiguration: lane mode requires max_num_seqs to
            # split evenly across the internal TT lanes.
            validate_tt_lane_config(vllm_config)
            vllm_config.scheduler_config.scheduler_cls = TT_LANE_SCHEDULER_CLS
            logger.info(
                "Using TTLaneCoordinator with %d in-process TT lanes",
                get_tt_data_parallel_size(vllm_config),
            )
        else:
            vllm_config.scheduler_config.scheduler_cls = TT_SCHEDULER_CLS

        if not is_lane_mode:
            cls._standard_dp_mesh_grids = _load_standard_dp_mesh_grids(vllm_config)
            cls._standard_dp_visible_device_groups = _load_standard_dp_visible_groups(
                vllm_config
            )
            # Discovery opens the parent mesh, so only a process that still sees
            # the whole machine may run it. This hook also re-runs in the worker,
            # after `TT_VISIBLE_DEVICES` is narrowed to one group; there it must
            # consume what `VllmConfig` carries, or it would rediscover against
            # the narrowed cluster and overwrite the real submesh shapes.
            if cls._standard_dp_visible_device_groups is None:
                discovery_result = _resolve_standard_dp_visible_device_groups(
                    vllm_config
                )
                (
                    cls._standard_dp_visible_device_groups,
                    resolved_mesh_grids,
                ) = _split_standard_dp_discovery_result(discovery_result)
                if resolved_mesh_grids:
                    cls._standard_dp_mesh_grids = resolved_mesh_grids
                    _store_standard_dp_mesh_grids(vllm_config, resolved_mesh_grids)
                if cls._standard_dp_visible_device_groups is not None:
                    _store_standard_dp_visible_groups(
                        vllm_config, cls._standard_dp_visible_device_groups
                    )

        if vllm_config.cache_config.enable_prefix_caching:
            # Check prefix caching support from capabilities (default to False)
            supports_prefix_caching = (
                model_capabilities.get("supports_prefix_caching", False)
                if model_capabilities
                else False
            )

            if not supports_prefix_caching:
                vllm_config.cache_config.enable_prefix_caching = False
                logger.warning(
                    "Prefix caching is not supported in TT backend for %s, "
                    "disabling it",
                    model_class.__module__,
                )
            else:
                # Check if the model architecture uses sliding window
                uses_sliding_window = (
                    vllm_config.model_config.get_sliding_window() is not None
                )
                if uses_sliding_window:
                    vllm_config.cache_config.enable_prefix_caching = False
                    logger.warning(
                        "Prefix caching is not supported in TT backend for "
                        "models with sliding window, disabling it"
                    )

        logger.info(
            "Automatic prefix caching is %s",
            "enabled" if vllm_config.cache_config.enable_prefix_caching else "disabled",
        )
        # Check that all invariants are satisfied after all rewriting
        vllm_config.scheduler_config.verify_max_model_len(
            vllm_config.model_config.max_model_len
        )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # The regular v0 vLLM sampling code tries
        # to use pinned memory in case we're using GPUs.
        return False

    @classmethod
    def uses_host_device_handling(cls) -> bool:
        return True

    @classmethod
    def _fit_whole_canvas_default(
        cls, prompt_len: int, output_size: int, max_model_len: int
    ) -> int:
        """Largest whole-canvas output that fits after the prompt; raises if
        not even one physical canvas fits."""
        remaining = max_model_len - prompt_len
        if remaining < output_size:
            raise ValueError(
                f"Prompt length {prompt_len} leaves {max(0, remaining)} tokens "
                f"within max_model_len={max_model_len}, but this model commits "
                f"physical {output_size}-token output canvases. Use a shorter "
                "prompt or a larger max model length."
            )
        return remaining // output_size * output_size

    def get_max_output_tokens(self, prompt_len: int) -> int:
        """Clamp the platform default to complete physical output canvases."""
        output_size = type(self).output_tokens_per_step
        max_model_len = type(self)._get_block_model_max_len()
        if output_size == 1 or max_model_len is None:
            return super().get_max_output_tokens(prompt_len)
        return self._fit_whole_canvas_default(prompt_len, output_size, max_model_len)

    @classmethod
    def validate_request(
        cls,
        processed_inputs: "EngineInput",
        params: "SamplingParams | PoolingParams",
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        from vllm.sampling_params import SamplingParams

        dev = cls.device_name

        if isinstance(params, SamplingParams) and params.prompt_logprobs is not None:
            raise ValueError(f"Not yet supporting prompt_logprobs on {dev}")

        output_size = cls.output_tokens_per_step
        if not isinstance(params, SamplingParams) or output_size == 1:
            return

        prompt_token_ids = processed_inputs.get("prompt_token_ids")
        max_model_len = cls._get_block_model_max_len()
        if prompt_token_ids is not None and max_model_len is not None:
            prompt_len = len(prompt_token_ids)
            max_tokens = params.max_tokens
            if max_tokens is None:
                # OpenAI serving resolves max_tokens before this hook, but
                # offline callers can pass max_tokens=None (the processor
                # defaults it only after validation). Validate that eventual
                # per-request default locally; mutating the caller-owned object
                # here makes LLM.generate([...], one_params) prompt-order
                # dependent because vLLM clones only after this hook.
                max_tokens = cls._fit_whole_canvas_default(
                    prompt_len, output_size, max_model_len
                )
            physical_output_tokens = (
                (max_tokens + output_size - 1) // output_size
            ) * output_size
            if prompt_len + physical_output_tokens > max_model_len:
                raise ValueError(
                    "Block output is committed in physical "
                    f"{output_size}-token canvases: prompt length {prompt_len} "
                    f"plus max_tokens={max_tokens} requires "
                    f"{physical_output_tokens} physical output tokens, exceeding "
                    f"max_model_len={max_model_len}. Reduce max_tokens or use a "
                    "shorter prompt."
                )

        # Reject unsupported response-contract controls. Model-owned sampling
        # controls (temperature etc.) are instead accepted and neutralized on
        # the per-request clone in _install_block_output_input_processor_patch.
        unsupported = []
        if params.n != 1:
            unsupported.append(f"n={params.n!r} (accepted: 1)")
        if params.logprobs is not None:
            unsupported.append(f"logprobs={params.logprobs!r} (accepted: None)")
        if params.logprob_token_ids is not None:
            unsupported.append("logprob_token_ids (accepted: omitted/None)")
        if params.flat_logprobs:
            unsupported.append("flat_logprobs=True (accepted: False)")
        if params.bad_words:
            unsupported.append("bad_words (accepted: omitted/empty)")
        if params.structured_outputs is not None:
            unsupported.append("structured_outputs (accepted: omitted/None)")
        if params.logit_bias is not None:
            unsupported.append("logit_bias (accepted: omitted/None)")
        if params.allowed_token_ids is not None:
            unsupported.append("allowed_token_ids (accepted: omitted/None)")
        if params.min_tokens != 0:
            unsupported.append(f"min_tokens={params.min_tokens!r} (accepted: 0)")
        if params.thinking_token_budget is not None:
            unsupported.append("thinking_token_budget (accepted: omitted/None)")
        if params.repetition_detection is not None:
            unsupported.append("repetition_detection (accepted: omitted/None)")
        if params.extra_args:
            unsupported.append("extra_args (accepted: omitted/empty)")

        if unsupported:
            raise ValueError(
                "This block-output model owns its Gumbel sampling and does not "
                "support these request parameters: " + "; ".join(unsupported)
            )

    @staticmethod
    def compat_sampling_required(sampling_params, num_devices) -> bool:
        # Device logprobs only supported on multi-device setups and only
        # the sampled token's logprob is returned (not top-k alternatives).
        # Single device: any logprobs require host sampling.
        # Multi-device: logprobs > 1 requires host sampling because device
        # can only return the sampled token's logprob.
        # https://github.com/tenstorrent/tt-metal/issues/34077
        if (
            sampling_params.logprobs is not None
            and sampling_params.logprobs > 0
            and (num_devices == 1 or sampling_params.logprobs > 1)
        ):
            return True

        # all of the following sampling params require compat sampling
        return (
            sampling_params.min_p != 0.0
            or (
                sampling_params.bad_words is not None
                and len(sampling_params.bad_words) > 0
            )
            or sampling_params.prompt_logprobs is not None
            or sampling_params.structured_outputs is not None
            or sampling_params.logit_bias is not None
            or sampling_params.allowed_token_ids is not None
            or sampling_params.min_tokens != 0
        )
