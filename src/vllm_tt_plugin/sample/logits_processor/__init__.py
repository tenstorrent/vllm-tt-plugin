# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Host logits processors.

``build_logitsprocs`` is adapted from vLLM v0.31.0
``vllm/v1/sample/logits_processor/__init__.py``. Custom processors use vLLM's
V2 interface; see ``custom.py``.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm_tt_plugin.sample.logits_processor.custom import build_custom_logitsprocs
from vllm_tt_plugin.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
from vllm_tt_plugin.sample.logits_processor.state import (
    BatchUpdateBuilder,
    LogitsProcessors,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_tt_logger(__name__)

STR_SPEC_DEC_REJECTS_LOGITSPROCS = (
    "Custom logits processors are not supported when speculative decoding is enabled."
)


def build_logitsprocs(
    vllm_config: "VllmConfig",
    custom_logitsprocs: Sequence[str | type] = (),
) -> LogitsProcessors:
    if vllm_config.speculative_config:
        if custom_logitsprocs:
            raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
        logger.warning(
            "min_p and logit_bias parameters won't work with speculative decoding."
        )
        return LogitsProcessors([MinTokensLogitsProcessor()])

    return LogitsProcessors(
        [
            MinTokensLogitsProcessor(),
            LogitBiasLogitsProcessor(),
            MinPLogitsProcessor(vllm_config.scheduler_config.max_num_seqs),
        ],
        custom=build_custom_logitsprocs(vllm_config, custom_logitsprocs),
    )


__all__ = [
    "BatchUpdate",
    "BatchUpdateBuilder",
    "LogitBiasLogitsProcessor",
    "LogitsProcessor",
    "LogitsProcessors",
    "MinPLogitsProcessor",
    "MinTokensLogitsProcessor",
    "MoveDirectionality",
    "build_logitsprocs",
]
