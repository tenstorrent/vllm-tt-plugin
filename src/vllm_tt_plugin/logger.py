# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Integrates TT loggers with vLLM's logger hierarchy and deduplicates warnings.

``init_tt_logger`` reuses standard logging instances, marks their records with
a TT filename prefix, and attaches ``warning_once``. Handlers and configuration
remain owned by the application. This module uses only the standard library:
importing vLLM here would re-enter plugin discovery before logger setup finishes.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from types import MethodType
from typing import cast

__all__ = ("init_tt_logger",)

_TT_LOGGER_ROOT = "vllm.tt"
_PACKAGE_PREFIX = "vllm_tt_plugin."


def init_tt_logger(name: str) -> logging.Logger:
    """Returns a TT logger using the application's vLLM logging configuration.

    Parameters
    ----------
    name : str
        Module name, normally ``__name__``. A leading ``vllm_tt_plugin.`` is
        removed before placing the name beneath ``vllm.tt``.

    Returns
    -------
    logging.Logger
        Shared instance from the logging registry, with a TT filename filter
        and a bound ``warning_once(msg, *args, **kwargs)`` method. Repeated
        setup adds no duplicate filters and restores the TT warning method.

    Notes
    -----
    Setup leaves handlers, levels, propagation, and disabled state unchanged.
    The filter mutates each emitted record's filename to ``tt/<filename>``;
    vLLM's existing format and streams determine how it is displayed.
    Later dictionary logging configuration can disable existing loggers if
    it omits their hierarchy and enables ``disable_existing_loggers``.

    ``warning_once`` accepts standard warning arguments, including exception
    information, extra context, and caller-relative stack levels. It deduplicates
    rendered text per logger through a process-wide 128-entry LRU cache. Evicted
    warnings may be emitted again. Different context for repeated text does not
    create another entry: the first call's arguments and options are retained
    until eviction. Disabled warnings are neither formatted nor remembered.
    Enabled calls are cached even if a filter or handler suppresses output.
    Concurrent first calls may both emit before their cache entry is stored.

    Examples
    --------
    Initialization needs neither vLLM imports nor a TT runtime.

    >>> from vllm_tt_plugin.logger import init_tt_logger
    >>> logger = init_tt_logger("vllm_tt_plugin.logger_example")
    >>> logger.name
    'vllm.tt.logger_example'
    >>> init_tt_logger("vllm_tt_plugin.logger_example") is logger
    True
    >>> callable(logger.warning_once)
    True
    """
    logger = logging.getLogger(
        f"{_TT_LOGGER_ROOT}.{name.removeprefix(_PACKAGE_PREFIX)}"
    )
    if not any(isinstance(item, _TTFileTag) for item in logger.filters):
        logger.addFilter(_TTFileTag())
    logger.warning_once = MethodType(_warning_once, logger)
    return logger


class _TTFileTag(logging.Filter):
    """Prefixes record filenames so TT sources are identifiable in vLLM output."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Marks the record in place and allows it through without filtering."""
        record.filename = f"tt/{record.filename}"
        return True


def _warning_once(
    self: logging.Logger,
    msg: object,
    *args: object,
    **kwargs: object,
) -> None:
    """Emits an enabled warning once per cached rendered message for this logger.

    Arguments follow ``logging.Logger.warning``. The two wrapper frames are
    added to the requested stacklevel so caller attribution remains relative
    to the code invoking ``warning_once``. An instance binding takes precedence
    over class-level warning extensions with different deduplication behavior.
    """
    if not self.isEnabledFor(logging.WARNING):
        return

    kwargs["stacklevel"] = cast(int, kwargs.get("stacklevel", 1)) + 2
    _log_once(
        self, logging.WARNING, _LogOnceCall(_message_key(msg, args), msg, args, kwargs)
    )


def _message_key(msg: object, args: tuple[object, ...]) -> tuple[str, str]:
    """Builds a hashable key while leaving format-error reporting to handlers.

    A LogRecord applies the same interpolation rules as logging, including a
    single dictionary argument. If interpolation fails, a tagged representation
    distinguishes the raw call from successfully rendered text. Emission keeps
    the original message and arguments so the handler can report their error.
    """
    record = logging.LogRecord(
        name="",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg=msg,
        args=args,
        exc_info=None,
    )
    try:
        return "rendered", record.getMessage()
    except Exception:
        try:
            raw_call = repr((msg, args))
        except Exception:
            raw_call = f"{msg!r} with unrepresentable arguments"
        return "unrenderable", raw_call


@dataclass(frozen=True)
class _LogOnceCall:
    """Retains the first call's arguments while hashing only its message key.

    ``cache_key`` contains tagged text. ``msg``, ``args``, and ``kwargs`` retain
    the original objects, which may be unhashable, without participating in
    equality or hashing. The cache owns these references until eviction.
    """

    cache_key: tuple[str, str]
    msg: object = field(compare=False, hash=False)
    args: tuple[object, ...] = field(compare=False, hash=False)
    kwargs: dict[str, object] = field(compare=False, hash=False)


@functools.lru_cache(maxsize=128)
def _log_once(logger: logging.Logger, level: int, call: _LogOnceCall) -> None:
    """Emits a successful call once for its logger, level, and cached message key.

    The shared cache retains at most 128 calls across loggers and levels.
    Emission errors propagate and are not cached.
    """
    logger.log(level, call.msg, *call.args, **call.kwargs)
