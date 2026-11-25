# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Contains model-output adapters and their lazy vLLM registration.

``registration.register_parsers`` installs missing reasoning and tool-parser
names; model modules implement only behavior absent from the pinned vLLM.
This initializer imports nothing so parser discovery stays lazy.
"""
