# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

# Stdlib-only by necessity: importing `vllm` here triggers platform plugin
# loading, which re-enters this module before it has finished executing.
import functools
import logging
from types import MethodType

_TT_LOGGER_ROOT = "vllm.tt"
_PACKAGE_PREFIX = "vllm_tt_plugin."


@functools.lru_cache
def _log_once(logger: logging.Logger, level: int, msg: str, args: tuple) -> None:
    # stacklevel=3 attributes the record to the *_once caller's line.
    logger.log(level, msg, *args, stacklevel=3)


def _warning_once(self: logging.Logger, msg: str, *args, **_kwargs) -> None:
    _log_once(self, logging.WARNING, msg, args)


def _info_once(self: logging.Logger, msg: str, *args, **_kwargs) -> None:
    _log_once(self, logging.INFO, msg, args)


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
    # vLLM patches warning_once/info_once per instance in init_logger; this
    # stdlib logger needs its own so *_once call sites work on every model.
    if not hasattr(logger, "warning_once"):
        logger.warning_once = MethodType(_warning_once, logger)
        logger.info_once = MethodType(_info_once, logger)
    return logger
