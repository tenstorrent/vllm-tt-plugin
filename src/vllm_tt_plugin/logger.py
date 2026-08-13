# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

# Stdlib-only by necessity: importing `vllm` here triggers platform plugin
# loading, which re-enters this module before it has finished executing.
import logging

_TT_LOGGER_ROOT = "vllm.tt"
_PACKAGE_PREFIX = "vllm_tt_plugin."


class _TTFileTag(logging.Filter):
    """Marks records so TT lines are distinguishable in vLLM's output."""

    def filter(self, record: logging.LogRecord) -> bool:
        # vLLM's format prints `[filename:lineno]` and no logger name, and most
        # of the plugin's file names also exist in vLLM core.
        record.filename = f"tt/{record.filename}"
        return True


def init_tt_logger(name: str) -> logging.Logger:
    """Initializes a TT plugin logger under vLLM's configured logger tree.

    Records keep vLLM's own format and stream-e.g., `vllm.tt.engine` renders as::

        INFO 08-12 15:28:15 [tt/engine.py:167] DP device ranks: [0, 1]

    NOTE: a ``VLLM_LOGGING_CONFIG_PATH`` config that omits the ``vllm`` logger
          disables every logger created before vLLM applies it, so calling this
          ahead of ``import vllm`` silences TT records entirely.
    """
    logger = logging.getLogger(
        f"{_TT_LOGGER_ROOT}.{name.removeprefix(_PACKAGE_PREFIX)}"
    )
    if not any(isinstance(f, _TTFileTag) for f in logger.filters):
        logger.addFilter(_TTFileTag())
    return logger
