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


# ``SamplingParams`` fields that can force a step onto host sampling, and the
# value that does not. Host sampling yields one token where a canvas is needed,
# so ``_get_output_tokens`` raises and takes EngineCore with it.
#
# Mirrored from ``check_perform_device_sampling``; add a request-driven branch
# there and you must add its field here. Pinned behaviorally, not by a second
# list -- see ``tests/test_lane_input_batch.py``.
BLOCK_HOST_SAMPLING_FORCERS: tuple[tuple[str, Any], ...] = (
    ("min_p", 0.0),
    ("min_tokens", 0),
    ("logit_bias", None),
    ("allowed_token_ids", None),
    ("bad_words", None),
    ("_bad_words_token_ids", None),
    # Forces host sampling on any device count outside {8, 32} -- P150x4
    # included -- and ``logprobs=0`` (OpenAI ``logprobs: true``) is enough. The
    # worker-side ``disable_logprobs`` sentinel already blocks it; this entry is
    # the second layer, so the table mirrors the predicate rather than depending
    # on that sentinel.
    ("logprobs", None),
)

# Response-shape controls a block-output model cannot honor. Kept out of
# ``BLOCK_HOST_SAMPLING_FORCERS`` so that table stays an exact mirror of the
# predicate.
BLOCK_UNSUPPORTED_RESPONSE_CONTROLS: tuple[tuple[str, Any], ...] = (
    ("prompt_logprobs", None),
)

# Penalties neither force host sampling nor change generation -- the model owns
# its sampler and drops what it is handed. The cost is host waste: a non-neutral
# value flips ``InputBatch.no_penalties``, and every block decode step then
# rebuilds the prompt and output token-history tensors in
# ``_prepare_model_inputs``, the output one growing with the session. The
# frontend neutralizes them; mirror it here.
BLOCK_PENALTY_NEUTRAL: tuple[tuple[str, Any], ...] = (
    ("presence_penalty", 0.0),
    ("frequency_penalty", 0.0),
    ("repetition_penalty", 1.0),
)

# What TTScheduler applies to a request that reached the engine without frontend
# validation.
BLOCK_UNVALIDATED_SAMPLING_NEUTRAL: tuple[tuple[str, Any], ...] = (
    BLOCK_HOST_SAMPLING_FORCERS
    + BLOCK_PENALTY_NEUTRAL
    + BLOCK_UNSUPPORTED_RESPONSE_CONTROLS
)


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
