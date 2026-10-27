# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Per-request scheduling events, for calibrating a scheduler model against a served run.

Enabled by ``TT_SCHED_EVENT_LOG=<path>`` in the engine's environment; unset, every call returns at once
and nothing else changes. The file (appended, one line per event, line-buffered so a killed engine loses
at most one line) reads::

    <monotonic s> open wall=<time.time()> pid=<pid>
    <monotonic s> arrive id=<request id> prompt=<tokens> max_tokens=<n>
    <monotonic s> admit id=<request id>
    <monotonic s> preempt id=<request id>
    <monotonic s> chunk id=<request id> start=<computed tokens> end=<computed + scheduled>
    <monotonic s> forward_start kind=prefill|decode b=<rows> reqs=<id,id,...>
    <monotonic s> forward_end kind=prefill|decode b=<rows>
    <monotonic s> first_token id=<request id>
    <monotonic s> finish id=<request id> status=<RequestStatus name> output=<tokens>

``arrive`` is the scheduler's ``add_request``; ``admit`` and ``chunk`` are the scheduler's decision of a
step (a new request, a prefill row with the chunk it was given); ``forward_*`` bracket the model runner's
synchronous device forward of that step (``b`` the rows it ran); ``first_token`` is the first output token
the scheduler appended to the request; ``finish`` is the scheduler freeing it. The monotonic clock is the
engine process's; the ``open`` line gives its offset to the wall clock so the file can be joined with a
client's wall-clock records (AIPerf's ``request_start_ns``). AIPerf's ``X-Request-ID`` header is the suffix
of vLLM's request id (``chatcmpl-<X-Request-ID>``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import IO

_PATH: str | None = os.environ.get("TT_SCHED_EVENT_LOG")
_FILE: IO[str] | None = None


def enabled() -> bool:
    return _PATH is not None


def emit(kind: str, **fields: object) -> None:
    """Append one event line; a no-op unless ``TT_SCHED_EVENT_LOG`` is set."""
    global _FILE
    if _PATH is None:
        return
    now = time.monotonic()
    if _FILE is None:
        _FILE = open(_PATH, "a", buffering=1)  # noqa: SIM115 - kept open for the process's life
        _FILE.write(f"{now:.6f} open wall={time.time():.6f} pid={os.getpid()}\n")
    _FILE.write(f"{now:.6f} {kind} " + " ".join(f"{k}={v}" for k, v in fields.items()) + "\n")


def ids(req_ids: Iterable[str]) -> str:
    return ",".join(req_ids)
