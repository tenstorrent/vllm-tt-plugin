# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from typing import TYPE_CHECKING, Any

from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_tt_logger(__name__)


def _extract_tt_config(
    config: dict[str, Any], config_name: str
) -> tuple[dict[str, Any], bool]:
    if not isinstance(config, dict):
        raise ValueError(f"{config_name} must be a JSON object")
    if "tt" not in config:
        return {}, False
    tt_config = config["tt"]
    if not isinstance(tt_config, dict):
        raise ValueError(f"{config_name}['tt'] must be a JSON object")
    return tt_config, True


def get_tt_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Return TT config from vLLM's generic additional config namespace."""
    additional_config, _ = _extract_tt_config(
        getattr(vllm_config, "additional_config", {}) or {}, "additional_config"
    )
    return dict(additional_config)


# Internal key recording the resolved TT lane count. Stored at the top level of
# additional_config -- deliberately outside the user "tt" namespace -- so it
# never collides with user config and reads as platform-derived state rather
# than user input. Written by store_tt_lane_count, read by
# get_tt_data_parallel_size.
_RESOLVED_LANE_COUNT_KEY = "_tt_resolved_lane_count"
_OUTPUT_TOKENS_PER_STEP_KEY = "_tt_output_tokens_per_step"


def get_tt_data_parallel_size(vllm_config: "VllmConfig") -> int:
    """Effective TT lane count for batching, KV sizing, and merged execution.

    Standard multi-process DP runs one independent TT mesh per rank, so the TT
    model itself sees no internal DP and the effective TT lane count remains 1.
    With a single engine (``data_parallel_size == 1``) the value is the lane
    count resolved by the Galaxy DP-to-lanes conversion (see ``platform.py``)
    and recorded via ``store_tt_lane_count``; absent that, the count is 1.
    Not user-facing.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_RESOLVED_LANE_COUNT_KEY, 1))


def store_tt_lane_count(vllm_config: "VllmConfig", lanes: int) -> None:
    """Record the resolved in-process TT lane count on the config.

    Writes an internal, top-level key into ``additional_config`` (kept out of
    the user "tt" namespace) so ``get_tt_data_parallel_size`` observes it both
    here and in the worker subprocess -- ``additional_config`` is a declared
    config field, so it survives the copy/pickle to that process. Internal
    handoff from the Galaxy DP-to-lanes conversion; not user-facing.
    """
    if lanes < 1:
        raise ValueError(f"resolved TT lane count must be >= 1, got {lanes}")
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_RESOLVED_LANE_COUNT_KEY] = lanes


def get_tt_output_tokens_per_step(vllm_config: "VllmConfig") -> int:
    """Return the normalized model output width, defaulting to AR behavior.

    ``TTPlatform.check_and_update_config`` resolves the model capability once
    and stores it on the serializable vLLM config. Scheduler and worker
    construction therefore do not need to import model-loader code.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_OUTPUT_TOKENS_PER_STEP_KEY, 1))


def require_tt_output_tokens_per_step(vllm_config: "VllmConfig") -> int:
    """Return the resolved output width, failing if setup did not store it."""
    additional = getattr(vllm_config, "additional_config", None)
    if (
        not isinstance(additional, dict)
        or _OUTPUT_TOKENS_PER_STEP_KEY not in additional
    ):
        raise RuntimeError(
            "TT output_tokens_per_step was not initialized on VllmConfig"
        )
    return int(additional[_OUTPUT_TOKENS_PER_STEP_KEY])


def is_tt_block_output_model(vllm_config: "VllmConfig") -> bool:
    """Whether this config describes a model that commits multi-token blocks."""
    return get_tt_output_tokens_per_step(vllm_config) > 1


# Platform-derived, like _RESOLVED_LANE_COUNT_KEY above: written from the
# MODEL's declared capability, never read from operator input. The leading
# underscore keeps an operator-passed --additional-config entry of the same
# name from reading as configuration this code honours.
_ADAPTIVE_BLOCK_OUTPUT_KEY = "_tt_adaptive_block_output"


def is_tt_adaptive_block_output_model(vllm_config: "VllmConfig") -> bool:
    """Whether the model commits a block ONLY when it decodes alone (batch==1).

    A plain block-output model owns a single request state and requires
    ``max_num_seqs 1`` / no data-parallelism. An ADAPTIVE block-output model
    emits its multi-token block only on steps that schedule exactly one decode
    request, and falls back to plain 1-token baseline decode whenever two or
    more requests batch together. That lifts the ``max_num_seqs 1`` and
    data-parallel restrictions: at low concurrency each request gets the block
    speedup, at higher concurrency the server is a plain batched baseline
    (never worse). ``TTScheduler._update_after_schedule`` owns the reservation
    predicate and reserves the K-token placeholder block only when the step is
    solo, is a decode, and the request owns the model's speculative session --
    which is where ``tt_adaptive_block_max_prompt_tokens`` was already applied,
    at the prefill that armed that session.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional.get(_ADAPTIVE_BLOCK_OUTPUT_KEY, False))


def store_tt_adaptive_block_output(vllm_config: "VllmConfig", flag: bool) -> None:
    """Record the model's adaptive block-output capability on the config.

    Internal platform-to-runtime handoff, not user-facing: the value comes from
    the model's ``tt_adaptive_block_output`` capability, which
    ``TTPlatform.check_and_update_config`` has already validated against
    ``output_tokens_per_step > 1``.
    """
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_ADAPTIVE_BLOCK_OUTPUT_KEY] = bool(flag)


# Platform-derived; see _ADAPTIVE_BLOCK_OUTPUT_KEY.
_ADAPTIVE_BLOCK_MAX_PROMPT_KEY = "_tt_adaptive_block_max_prompt_tokens"


def get_tt_adaptive_block_max_prompt_tokens(vllm_config: "VllmConfig") -> int:
    """Prompt-length frontier for the adaptive block path (0 = no limit).

    An adaptive block model whose fused capture cannot fit beyond some prompt
    length serves longer prompts as plain baseline (width-1 steps) for their
    whole lifetime. The scheduler must reserve width-1 for those requests even
    on solo decode steps, so the model declares the SAME frontier it gates on.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_ADAPTIVE_BLOCK_MAX_PROMPT_KEY, 0))


def store_tt_adaptive_block_max_prompt_tokens(
    vllm_config: "VllmConfig", limit: int
) -> None:
    """Record the adaptive block path's prompt-length frontier on the config.

    Internal platform-to-runtime handoff, not user-facing: the value comes from
    the model's ``tt_adaptive_block_max_prompt_tokens`` capability. Validated
    here, next to its sibling ``store_tt_output_tokens_per_step``, rather than
    in the platform -- the pairing rule that a non-zero frontier requires the
    adaptive capability stays in the platform, because only the platform sees
    both capabilities.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError(
            "resolved TT adaptive_block_max_prompt_tokens must be an integer "
            f">= 0, got {limit!r}"
        )
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_ADAPTIVE_BLOCK_MAX_PROMPT_KEY] = int(limit)


# Platform-derived; see _ADAPTIVE_BLOCK_OUTPUT_KEY.
_BLOCK_KV_EXTENT_KEY = "_tt_block_kv_extent_tokens"


def get_tt_block_kv_extent_tokens(vllm_config: "VllmConfig") -> int:
    """Total KV positions ONE block-output decode step may touch (0 = undeclared).

    A block-output step runs internal verify iterations against vLLM-owned KV, so
    the positions it writes run past the tokens it commits. The emitted width does
    not bound that: the physical extent is the committed block PLUS the last
    iteration's verification rows plus any accepted tokens carried beyond the
    emitted width, and a model may be configured with a verification width LARGER
    than its output block (dFlash at ``SERVE_BLOCK=2``, ``VERIFY=7`` emits 2 and
    writes 8 verification rows). Reserving a multiple of the output width then
    under-allocates and the step writes positions with no request block
    (vllm-tt-plugin#118 review, finding 2).

    So the model declares the extent it actually touches and the scheduler
    reserves at least that much lookahead. 0 means the model did not declare one,
    and the scheduler falls back to its output-width multiple.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_BLOCK_KV_EXTENT_KEY, 0))


def store_tt_block_kv_extent_tokens(vllm_config: "VllmConfig", extent: int) -> None:
    """Record the block-output step's physical KV extent on the config.

    Internal platform-to-runtime handoff: the value comes from the model's
    ``tt_block_kv_extent_tokens`` capability. Validated here alongside its
    siblings; the rule that it must cover the output width lives in the platform,
    which is the only place that sees both.
    """
    if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
        raise ValueError(
            "resolved TT block_kv_extent_tokens must be an integer >= 0, got "
            f"{extent!r}"
        )
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_BLOCK_KV_EXTENT_KEY] = int(extent)


def store_tt_output_tokens_per_step(
    vllm_config: "VllmConfig", output_tokens_per_step: int
) -> None:
    """Store the validated per-request output width on the vLLM config."""
    if (
        isinstance(output_tokens_per_step, bool)
        or not isinstance(output_tokens_per_step, int)
        or output_tokens_per_step < 1
    ):
        raise ValueError(
            "resolved TT output_tokens_per_step must be an integer >= 1, got "
            f"{output_tokens_per_step!r}"
        )
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_OUTPUT_TOKENS_PER_STEP_KEY] = output_tokens_per_step


def get_tt_max_batch_size(vllm_config: "VllmConfig") -> int:
    """Return the global TT batch capacity for model/KV sizing.

    Standard DP is per-rank and single-process lane mode already stores the
    global engine capacity in ``max_num_seqs`` after the Galaxy conversion, so
    the TT model should always size itself to the visible engine-local batch.
    """
    return int(vllm_config.scheduler_config.max_num_seqs)


def get_tt_per_lane_max_num_seqs(vllm_config: "VllmConfig") -> int:
    """Return the per-lane/per-rank scheduling and wire-format capacity.

    Outside lane mode the global ``max_num_seqs`` is already the per-rank
    capacity. In single-process lane mode it is the validated per-lane split
    (see ``validate_tt_lane_config``).
    """
    if not uses_tt_lane_coordinator(vllm_config):
        return int(vllm_config.scheduler_config.max_num_seqs)
    return validate_tt_lane_config(vllm_config)


def validate_tt_lane_config(vllm_config: "VllmConfig") -> int:
    """Validate single-process lane-mode batch sizing; return per-lane capacity.

    Lane mode partitions the global ``max_num_seqs`` evenly across the lanes
    (one in-process DP replica each), so the global value must be a positive
    multiple of the lane count; raises ``ValueError`` otherwise. Assumes lane
    mode is active (callers gate on ``uses_tt_lane_coordinator``).

    Exposed as a named helper so ``platform.check_and_update_config`` can run
    this check at config time -- calling it for its raising side effect so a
    misconfiguration fails fast with a clear message -- rather than calling the
    per-lane getter and discarding its result.
    """
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    lanes = get_tt_data_parallel_size(vllm_config)
    if max_num_seqs % lanes != 0:
        raise ValueError(
            "max_num_seqs must be divisible by the TT lane count in "
            f"single-process lane mode; got max_num_seqs={max_num_seqs}, "
            f"lanes={lanes}."
        )
    per_lane = max_num_seqs // lanes
    if per_lane < 1:
        raise ValueError(
            "max_num_seqs must provide at least one request per TT lane; got "
            f"max_num_seqs={max_num_seqs}, lanes={lanes}."
        )
    return per_lane


def uses_tt_lane_coordinator(vllm_config: "VllmConfig") -> bool:
    return (
        vllm_config.parallel_config.data_parallel_size == 1
        and get_tt_data_parallel_size(vllm_config) > 1
    )


# Decode-interleave policy. TT executes a step as all-prefill or all-decode, so
# a run of prefill steps stalls every running decode for its whole duration.
# These bound that run. Read by the scheduler and by the lane coordinator.
_DECODE_INTERLEAVE_ENABLED_KEY = "decode_interleave_enabled"
_DECODE_INTERLEAVE_PREFILL_STEPS_KEY = "decode_interleave_prefill_steps"
_DECODE_INTERLEAVE_DECODE_STEPS_KEY = "decode_interleave_decode_steps"

_DECODE_INTERLEAVE_ENABLED_DEFAULT = True
_DECODE_INTERLEAVE_PREFILL_STEPS_DEFAULT = 2
_DECODE_INTERLEAVE_DECODE_STEPS_DEFAULT = 1


def _read_tt_bool(tt_config: dict[str, Any], key: str, default: bool) -> bool:
    value = tt_config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(
            f"additional_config['tt']['{key}'] must be a boolean, got {value!r}"
        )
    return value


def _read_tt_positive_int(tt_config: dict[str, Any], key: str, default: int) -> int:
    value = tt_config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"additional_config['tt']['{key}'] must be an integer >= 1, got {value!r}"
        )
    return value


def get_tt_decode_interleave_config(
    vllm_config: "VllmConfig",
) -> tuple[bool, int, int]:
    """Return ``(enabled, prefill_steps, decode_steps)`` for decode interleave.

    ``prefill_steps`` is the number of consecutive prefill steps allowed before
    a decode-only step is inserted; ``decode_steps`` is how many decode-only
    steps that insertion runs before prefill is required again. Both are step
    counts rather than token counts because the interleave granularity on TT is
    a whole step: the device cannot mix prefill and decode rows in one batch.
    """
    tt_config = get_tt_config(vllm_config)
    return (
        _read_tt_bool(
            tt_config,
            _DECODE_INTERLEAVE_ENABLED_KEY,
            _DECODE_INTERLEAVE_ENABLED_DEFAULT,
        ),
        _read_tt_positive_int(
            tt_config,
            _DECODE_INTERLEAVE_PREFILL_STEPS_KEY,
            _DECODE_INTERLEAVE_PREFILL_STEPS_DEFAULT,
        ),
        _read_tt_positive_int(
            tt_config,
            _DECODE_INTERLEAVE_DECODE_STEPS_KEY,
            _DECODE_INTERLEAVE_DECODE_STEPS_DEFAULT,
        ),
    )
