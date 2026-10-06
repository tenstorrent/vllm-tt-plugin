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


# Parsers for TT-served models that upstream vLLM lacks. Registered lazily and
# only while the name is free, so a future upstream parser wins.
_TT_REASONING_PARSERS = {
    "kolibri1": (
        "vllm_tt_plugin.kolibri1_reasoning_parser",
        "Kolibri1ReasoningParser",
    ),
}
# Kolibri 1 shares the Hermes `<tool_call>...</tool_call>` format.
_TT_TOOL_PARSERS = {
    "kolibri1": ("vllm.tool_parsers.hermes_tool_parser", "Hermes2ProToolParser"),
}


def _register_tt_reasoning_parsers() -> None:
    try:
        from vllm.reasoning import ReasoningParserManager
    except Exception as exc:  # pragma: no cover - vLLM without reasoning support
        logger.debug("Skipping TT reasoning parser registration: %s", exc)
        return
    registered = set(ReasoningParserManager.list_registered())
    for name, (module_path, class_name) in _TT_REASONING_PARSERS.items():
        if name in registered:
            continue
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)


def _register_tt_tool_parsers() -> None:
    try:
        from vllm.tool_parsers import ToolParserManager
    except Exception as exc:  # pragma: no cover - vLLM without tool parsers
        logger.debug("Skipping TT tool parser registration: %s", exc)
        return
    registered = set(ToolParserManager.list_registered())
    for name, (module_path, class_name) in _TT_TOOL_PARSERS.items():
        if name in registered:
            continue
        ToolParserManager.register_lazy_module(name, module_path, class_name)


def platform_plugin() -> str | None:
    """Return the TT platform class when TT runtime libraries are present."""
    try:
        import ttnn  # noqa: F401
    except Exception as exc:
        logger.debug("TT plugin platform is not available because: %s", exc)
        return None

    logger.debug("Confirmed TT plugin platform is available because ttnn is found.")
    return "vllm_tt_plugin.platform.TTPlatform"
