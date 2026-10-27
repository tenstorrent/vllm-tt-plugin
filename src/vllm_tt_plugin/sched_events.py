# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Per-request scheduling events, for calibrating a scheduler model against a served
run.

Enabled by ``TT_SCHED_EVENT_LOG=<path>`` in the engine's environment; unset, every call
returns at once
and nothing else changes. The file (appended, one line per event, line-buffered so a
killed engine loses
at most one line) reads::

    <monotonic s> open wall=<time.time()> pid=<pid>
    <monotonic s> arrive id=<request id> prompt=<tokens> max_tokens=<n>
    <monotonic s> admit id=<request id>
    <monotonic s> preempt id=<request id>
    <monotonic s> chunk id=<request id> start=<computed tokens> end=<computed +
    scheduled>
    <monotonic s> forward_start kind=prefill|decode b=<rows> reqs=<id,id,...>
    <monotonic s> forward_end kind=prefill|decode b=<rows>
    <monotonic s> first_token id=<request id>
    <monotonic s> finish id=<request id> status=<RequestStatus name> output=<tokens>

``arrive`` is the scheduler's ``add_request``; ``admit`` and ``chunk`` are the
scheduler's decision of a
step (a new request, a prefill row with the chunk it was given); ``forward_*`` bracket
the model runner's
synchronous device forward of that step (``b`` the rows it ran); ``first_token`` is the
first output token
the scheduler appended to the request; ``finish`` is the scheduler freeing it. The
monotonic clock is the
engine process's; the ``open`` line gives its offset to the wall clock so the file can
be joined with a
client's wall-clock records (AIPerf's ``request_start_ns``). AIPerf's ``X-Request-ID``
header is the suffix
of vLLM's request id (``chatcmpl-<X-Request-ID>``).
"""

from __future__ import annotations

import functools
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from typing import IO, Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

_PATH: str | None = os.environ.get("TT_SCHED_EVENT_LOG")
_FILE: IO[str] | None = None


def enabled() -> bool:
    return _PATH is not None


def emit(event: str, **fields: object) -> None:
    """Append one event line; a no-op unless ``TT_SCHED_EVENT_LOG`` is set.
    The first parameter is positional-only in effect: a field may be named
    ``kind`` (the forward events' ``kind=prefill|decode``), so it must not
    share the parameter's name."""
    global _FILE
    if _PATH is None:
        return
    now = time.monotonic()
    if _FILE is None:
        # Kept open for the process's life.
        _FILE = open(_PATH, "a", buffering=1)  # noqa: SIM115
        _FILE.write(f"{now:.6f} open wall={time.time():.6f} pid={os.getpid()}\n")
    _FILE.write(
        f"{now:.6f} {event} " + " ".join(f"{k}={v}" for k, v in fields.items()) + "\n"
    )


def ids(req_ids: Iterable[str]) -> str:
    return ",".join(req_ids)


def guarded(fn: F) -> F:
    """Decorator: with the event log enabled, an exception escaping ``fn`` is
    written to the log and to stderr (synchronously) before it propagates. The
    engine's own ``logger.exception`` goes through an asynchronous file
    handler that a dying process does not flush, so a step's traceback was
    lost. Disabled, ``fn`` is called as is."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _PATH is None:
            return fn(*args, **kwargs)
        try:
            return fn(*args, **kwargs)
        except BaseException:
            text = traceback.format_exc()
            emit("exception", where=fn.__qualname__)
            if _FILE is not None:
                _FILE.write(text if text.endswith("\n") else text + "\n")
                _FILE.flush()
            sys.stderr.write(f"[sched_events] exception in {fn.__qualname__}:\n{text}")
            sys.stderr.flush()
            raise

    return cast(F, wrapper)
