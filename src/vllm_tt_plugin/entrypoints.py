# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


def register() -> None:
    """Register TT models in every vLLM process."""
    from vllm_tt_plugin.model_registry import register_tt_models_from_plugin

    register_tt_models_from_plugin()
    _register_tt_reasoning_parsers()
    _register_tt_tool_parsers()


def _register_tt_reasoning_parsers() -> None:
    """Register reasoning parsers for TT-served models that vLLM lacks.

    Lazy, and only when the name is free, so a future upstream parser wins.
    """
    try:
        from vllm.reasoning import ReasoningParserManager
    except Exception as exc:  # pragma: no cover - vLLM without reasoning support
        logger.debug("Skipping TT reasoning parser registration: %s", exc)
        return
    name = "k2_horizon"
    if name in ReasoningParserManager.list_registered():
        return
    ReasoningParserManager.register_lazy_module(
        name,
        "vllm_tt_plugin.k2_horizon_reasoning_parser",
        "K2HorizonReasoningParser",
    )


def _register_tt_tool_parsers() -> None:
    """Register tool-call parsers for TT-served models that vLLM lacks.

    Lazy, and only when the name is free, so a future upstream parser wins.
    """
    try:
        from vllm.tool_parsers import ToolParserManager
    except Exception as exc:  # pragma: no cover - vLLM without tool-parser support
        logger.debug("Skipping TT tool parser registration: %s", exc)
        return
    name = "k2_horizon"
    if name in ToolParserManager.list_registered():
        return
    ToolParserManager.register_lazy_module(
        name,
        "vllm_tt_plugin.k2_horizon_tool_parser",
        "K2HorizonToolParser",
    )


def platform_plugin() -> str | None:
    """Return the TT platform class when TT runtime libraries are present."""
    try:
        import ttnn  # noqa: F401
    except Exception as exc:
        logger.debug("TT plugin platform is not available because: %s", exc)
        return None

    logger.debug("Confirmed TT plugin platform is available because ttnn is found.")
    return "vllm_tt_plugin.platform.TTPlatform"
