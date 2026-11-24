# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Registers model-output parser adapters without replacing upstream names.

``register_parsers`` is called by the plugin's general entry point in every
vLLM process. Registration stores import targets; vLLM loads a parser class
when selected. Model and platform registration remain owned by their modules.
"""

from vllm_tt_plugin.logger import init_tt_logger

__all__ = ("register_parsers",)

logger = init_tt_logger(__name__)

# Registry names map to (module path, class name) for vLLM's lazy loaders.
_REASONING_PARSERS: dict[str, tuple[str, str]] = {
    "kolibri1": ("vllm_tt_plugin.parsers.kolibri1", "Kolibri1ReasoningParser"),
}
_TOOL_PARSERS: dict[str, tuple[str, str]] = {
    "kolibri1": ("vllm.tool_parsers.hermes_tool_parser", "Hermes2ProToolParser"),
}


def register_parsers() -> None:
    """Registers missing reasoning parsers and tool-parser aliases in this process.

    Notes
    -----
    vLLM owns the process-wide registries and the lifetime of their entries.
    Existing names win, including unresolved lazy registrations. Repeated
    calls preserve both loaded classes and lazy targets. If importing either
    manager fails, its registrations are skipped with a debug diagnostic;
    the other manager is still tried.

    Examples
    --------
    Use an installed vLLM environment; no device is opened.

    >>> import vllm.config
    >>> from vllm_tt_plugin.parsers.registration import register_parsers
    >>> register_parsers()
    >>> from vllm.reasoning import ReasoningParserManager
    >>> "kolibri1" in ReasoningParserManager.list_registered()
    True
    """
    _register_reasoning_parsers()
    _register_tool_parsers()


def _register_reasoning_parsers() -> None:
    """Adds lazy reasoning targets only for names absent from vLLM's registry."""
    try:
        from vllm.reasoning import ReasoningParserManager
    except Exception as exc:  # pragma: no cover - vLLM without reasoning support
        logger.debug("Skipping TT reasoning parser registration: %s", exc)
        return
    registered = set(ReasoningParserManager.list_registered())
    for name, (module_path, class_name) in _REASONING_PARSERS.items():
        if name in registered:
            continue
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)


def _register_tool_parsers() -> None:
    """Adds lazy tool targets, reusing upstream implementations for shared formats."""
    try:
        from vllm.tool_parsers import ToolParserManager
    except Exception as exc:  # pragma: no cover - vLLM without tool parsers
        logger.debug("Skipping TT tool parser registration: %s", exc)
        return
    registered = set(ToolParserManager.list_registered())
    for name, (module_path, class_name) in _TOOL_PARSERS.items():
        if name in registered:
            continue
        ToolParserManager.register_lazy_module(name, module_path, class_name)
