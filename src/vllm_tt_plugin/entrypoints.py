# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


def register() -> None:
    """Register TT models, reasoning parsers and tool parsers in every vLLM process."""
    from vllm_tt_plugin.model_registry import register_tt_models_from_plugin

    register_tt_models_from_plugin()
    _register_tt_reasoning_parsers()
    _register_tt_tool_parsers()


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
    ReasoningParserManager.register_lazy_module(
        "diffusion_gemma",
        "vllm_tt_plugin.gemma4_reasoning_parser",
        "Gemma4ReasoningParser",
    )
    _install_gemma4_complete_parser_adapter()


def _install_gemma4_complete_parser_adapter() -> None:
    """Adapt vLLM 0.24 parsing for both plugin-owned Gemma 4 aliases.

    vLLM 0.24 supplies ``model_output_token_ids`` to ``Parser.parse`` but its
    generated ``DelegatingParser`` ignores them and does not own the final
    ``finish_streaming`` flush. ``Gemma4ReasoningParser`` needs both compatibility
    paths under the plugin's ``diffusion_gemma`` and ``gemma4`` aliases.
    """
    from collections.abc import Sequence
    from typing import Any

    from vllm.parser.parser_manager import ParserManager

    if getattr(ParserManager, "_tt_gemma4_complete_parser_installed", False):
        return

    original_get_parser = ParserManager.get_parser

    @classmethod
    def get_parser(
        cls,
        tool_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
        enable_auto_tools: bool = False,
        model_name: str | None = None,
        is_harmony: bool = False,
    ):
        del cls
        parser_cls = original_get_parser(
            tool_parser_name=tool_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            enable_auto_tools=enable_auto_tools,
            model_name=model_name,
            is_harmony=is_harmony,
        )
        if parser_cls is None or reasoning_parser_name not in {
            "diffusion_gemma",
            "gemma4",
        }:
            return parser_cls

        class Gemma4CompleteParser(parser_cls):
            def parse(
                self,
                model_output: str,
                request: Any,
                enable_auto_tools: bool = False,
                model_output_token_ids: Sequence[int] = (),
            ):
                reasoning_parser = self.reasoning_parser
                token_extractor = getattr(
                    reasoning_parser, "extract_reasoning_from_token_ids", None
                )
                # The token-ID domain is authoritative whenever the output
                # carries real structural marker tokens: a marker spelled as
                # ordinary text (including a text-spelled quote marker, which
                # would blind the text scan's quote toggling) is then literal
                # data, matching the streaming path's ID-domain scans. The
                # extractor decodes content with skip_special_tokens=False, so
                # literal <|tool_call> frames survive for the tool parser. The
                # text path remains the fallback for tokenizers that expose
                # markers only as text pieces (no structural token IDs).
                structural_token_ids = {
                    getattr(reasoning_parser, "start_token_id", None),
                    getattr(reasoning_parser, "end_token_id", None),
                    getattr(reasoning_parser, "tool_call_token_id", None),
                }
                structural_token_ids.discard(None)
                if (
                    model_output_token_ids
                    and callable(token_extractor)
                    and any(
                        token_id in structural_token_ids
                        for token_id in model_output_token_ids
                    )
                ):
                    reasoning, content = token_extractor(
                        model_output_token_ids, model_output
                    )
                else:
                    reasoning, content = self.extract_reasoning(model_output, request)
                tool_calls, content = self._extract_tool_calls(
                    content=content,
                    request=request,
                    enable_auto_tools=enable_auto_tools,
                )
                return reasoning, content, tool_calls

            def finalize_generation(
                self,
                delta_message,
                request: Any,
                state,
            ):
                result = super().finalize_generation(delta_message, request, state)
                for parser in (self.reasoning_parser, self.tool_parser):
                    finish_streaming = getattr(parser, "finish_streaming", None)
                    if not callable(finish_streaming):
                        continue
                    flushed = finish_streaming()
                    if flushed is None:
                        continue
                    if result is None:
                        result = flushed
                        continue

                    if flushed.reasoning:
                        result.reasoning = (result.reasoning or "") + flushed.reasoning
                    if flushed.content:
                        result.content = (result.content or "") + flushed.content
                    for flushed_call in flushed.tool_calls or []:
                        existing_call = next(
                            (
                                call
                                for call in result.tool_calls or []
                                if call.index == flushed_call.index
                            ),
                            None,
                        )
                        if (
                            existing_call is not None
                            and existing_call.function is not None
                            and flushed_call.function is not None
                        ):
                            if flushed_call.function.name:
                                existing_call.function.name = (
                                    existing_call.function.name
                                    or flushed_call.function.name
                                )
                            if flushed_call.function.arguments:
                                existing_call.function.arguments = (
                                    existing_call.function.arguments or ""
                                ) + flushed_call.function.arguments
                        else:
                            result.tool_calls = list(result.tool_calls or [])
                            result.tool_calls.append(flushed_call)
                return result

        return Gemma4CompleteParser

    ParserManager.get_parser = get_parser
    ParserManager._tt_gemma4_complete_parser_installed = True


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


def platform_plugin() -> str | None:
    """Return the TT platform class when TT runtime libraries are present."""
    try:
        import ttnn  # noqa: F401
    except Exception as exc:
        logger.debug("TT plugin platform is not available because: %s", exc)
        return None

    logger.debug("Confirmed TT plugin platform is available because ttnn is found.")
    return "vllm_tt_plugin.platform.TTPlatform"
