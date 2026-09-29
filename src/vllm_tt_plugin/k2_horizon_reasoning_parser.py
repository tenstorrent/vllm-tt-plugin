# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for IFM/K2-Horizon models served by the TT plugin.

The K2-Horizon chat template places the opening ``<ifm|think>`` marker in the
prompt. DeepSeek-R1's parser already handles an implicit opener and keeps an
unfinished reasoning response as reasoning with no final answer, so only the
marker strings differ. Upstream vLLM (v0.26.0) ships no K2-Horizon parser.
"""

from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


class K2HorizonReasoningParser(DeepSeekR1ReasoningParser):
    @property
    def start_token(self) -> str:
        return "<ifm|think>"

    @property
    def end_token(self) -> str:
        return "</ifm|think>"
