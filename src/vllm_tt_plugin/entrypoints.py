# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


def register() -> None:
    """Register TT models, reasoning parsers and tool parsers in every vLLM process."""
    from vllm_tt_plugin import detok_prime
    from vllm_tt_plugin.model_registry import register_tt_models_from_plugin

    register_tt_models_from_plugin()
    _register_tt_reasoning_parsers()
    _register_tt_tool_parsers()
    _register_tt_tokenizers()
    # TT_DETOK_PRIME_TOKENS: the streaming detokenizer primed with the prompt's
    # tail; installed here because the API server process, where the
    # detokenizer runs, loads general plugins and no platform code.
    detok_prime.install()


def _register_tt_reasoning_parsers() -> None:
    """Register reasoning parsers for TT-served models that aren't upstream.

    Kept in the plugin (rather than patched into ``vllm.reasoning``) so it
    carries over unchanged when switching to upstream vLLM. Registered lazily so
    the parser module is only imported when ``--reasoning-parser`` selects it.
    """
    from vllm.reasoning import ReasoningParserManager

    ReasoningParserManager.register_lazy_module(
        "gemma4",
        "vllm_tt_plugin.gemma4_reasoning_parser",
        "Gemma4ReasoningParser",
    )


def _register_tt_tool_parsers() -> None:
    """Register tool-call parsers for TT-served models that aren't upstream.

    Kept in the plugin (rather than patched into ``vllm.tool_parsers``) so it
    carries over unchanged when switching to upstream vLLM. Registered lazily so
    the parser module is only imported when ``--tool-call-parser`` selects it.
    """
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    ToolParserManager.register_lazy_module(
        "gemma4",
        "vllm_tt_plugin.gemma4_tool_parser",
        "Gemma4ToolParser",
    )
    ToolParserManager.register_lazy_module(
        "deepseek_v41",
        "vllm_tt_plugin.deepseek_v41_tool_parser",
        "DeepSeekV41ToolParser",
    )


def _register_tt_tokenizers() -> None:
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    TokenizerRegistry.register(
        "deepseek_v41", "vllm_tt_plugin.deepseek_v41_tokenizer", "DeepseekV41Tokenizer"
    )
    RENDERER_REGISTRY.register("deepseek_v41", "vllm.renderers.hf", "HfRenderer")


def platform_plugin() -> str | None:
    """Return the TT platform class when TT runtime libraries are present."""
    try:
        import ttnn  # noqa: F401
    except Exception as exc:
        logger.debug("TT plugin platform is not available because: %s", exc)
        return None

    logger.debug("Confirmed TT plugin platform is available because ttnn is found.")
    return "vllm_tt_plugin.platform.TTPlatform"
