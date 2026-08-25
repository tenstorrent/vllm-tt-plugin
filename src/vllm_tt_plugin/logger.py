# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

# Stdlib-only by necessity: importing `vllm` here triggers platform plugin
# loading, which re-enters this module before it has finished executing.
import functools
import logging
from dataclasses import dataclass, field
from types import MethodType

_TT_LOGGER_ROOT = "vllm.tt"
_PACKAGE_PREFIX = "vllm_tt_plugin."


@dataclass(frozen=True)
class _LogOnceCall:
    cache_key: tuple[str, str]
    msg: str = field(compare=False, hash=False)
    args: tuple = field(compare=False, hash=False)


@functools.lru_cache
def _log_once(logger: logging.Logger, level: int, call: _LogOnceCall) -> None:
    # stacklevel=3 attributes the record to the *_once caller's line.
    logger.log(level, call.msg, *call.args, stacklevel=3)


def _warning_once(self: logging.Logger, msg: str, *args, **_kwargs) -> None:
    if not self.isEnabledFor(logging.WARNING):
        return

    record = logging.LogRecord(
        self.name,
        logging.WARNING,
        pathname="",
        lineno=0,
        msg=msg,
        args=args,
        exc_info=None,
    )
    try:
        cache_key = ("rendered", record.getMessage())
    except Exception:
        # Normal logging defers interpolation to the handler, where malformed
        # format arguments are reported without escaping from Logger.warning.
        # Keep that behavior while still producing a hashable deduplication key.
        try:
            raw_call = repr((msg, args))
        except Exception:
            raw_call = f"{msg!r} with unrepresentable arguments"
        cache_key = ("unrenderable", raw_call)
    _log_once(
        self,
        logging.WARNING,
        _LogOnceCall(cache_key, msg, args),
    )


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
    # Bind unconditionally: transformers installs an lru_cache warning_once on
    # logging.Logger keyed on the raw args, which raises TypeError for any
    # unhashable format argument. The instance binding shadows that class
    # attribute so TT loggers dedup on the rendered message instead.
    logger.warning_once = MethodType(_warning_once, logger)
    return logger
