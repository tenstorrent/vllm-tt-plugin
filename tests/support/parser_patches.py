# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Restores the vLLM parser attributes that the plugin's tool-call patches replace.

``vllm_tt_plugin.parsers.tool_schema`` and ``parsers.qwen3_arguments`` replace
module and class attributes of vLLM's parser engine and mark each owner with
``_tt_original_*`` attributes. A test that installs them uses ``snapshot`` so
later tests parse with upstream behavior again.
"""

from collections.abc import Callable


def snapshot() -> Callable[[], None]:
    """Records the patch targets and markers; returns a function that restores them.

    The restore function puts back each recorded attribute, deletes markers
    that did not exist when ``snapshot`` ran, and clears the ``qwen3_config``
    cache, whose configurations hold the converter they were built with.
    """
    from vllm.parser import qwen3
    from vllm.parser.engine import parser_engine
    from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine

    targets = (
        (
            parser_engine,
            "find_tool_properties",
            "_tt_original_find_tool_properties",
            "coerce_to_schema_type",
            "_tt_original_coerce_to_schema_type",
        ),
        (
            qwen3,
            "_qwen3_arg_converter",
            "_tt_original_qwen3_arg_converter",
            "_qwen3_arg_value_extender",
            "_tt_original_qwen3_arg_value_extender",
        ),
        (
            StreamingParserEngine,
            "_on_terminal",
            "_tt_original_on_terminal",
            "_emit_for_state",
            "_tt_original_emit_for_state",
        ),
    )
    unset = object()
    saved = [
        (owner, name, owner.__dict__.get(name, unset))
        for owner, *names in targets
        for name in names
    ]

    def restore() -> None:
        for owner, name, value in saved:
            if value is not unset:
                setattr(owner, name, value)
            elif name in owner.__dict__:
                delattr(owner, name)
        qwen3.qwen3_config.cache_clear()

    return restore
