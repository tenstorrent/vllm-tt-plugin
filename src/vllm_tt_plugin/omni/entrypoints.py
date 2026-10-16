# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


def omni_platform_plugin() -> str | None:
    """Return the TT Omni platform class when TT runtime libraries are present."""
    try:
        import ttnn  # noqa: F401
    except Exception as exc:
        logger.debug("TT Omni platform is not available because: %s", exc)
        return None
    return "vllm_tt_plugin.omni.platform.TTOmniPlatform"


def register() -> None:
    """Register the TT stage models and bring-up pipelines with vLLM-Omni.

    vLLM-Omni validates a stage's architecture against ``OmniModelRegistry``, not
    vLLM's ``ModelRegistry``, and loads this general plugin before it does.
    """
    try:
        import ttnn  # noqa: F401
    except Exception:
        return
    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    from vllm_tt_plugin.omni.pipelines import register_tt_pipelines
    from vllm_tt_plugin.platform import (
        TT_QWEN3_TTS_TALKER_ARCH,
        TT_QWEN3_TTS_TALKER_TARGET,
    )

    OmniModelRegistry.register_model(
        TT_QWEN3_TTS_TALKER_ARCH, TT_QWEN3_TTS_TALKER_TARGET
    )
    register_tt_pipelines()
