# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Contains model-output adapters and their lazy vLLM registration.

``registration.register_parsers`` installs missing reasoning and tool-parser
names; model modules implement only behavior absent from the pinned vLLM.
``tool_schema`` and ``qwen3_arguments`` patch upstream parser-engine defects;
the TT platform hook installs them. This initializer imports nothing so parser
discovery stays lazy.
"""
