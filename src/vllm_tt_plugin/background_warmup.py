# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmup work a model defers past engine start (e.g. MiMo's MIMO_DEFER_TRACES), run while the engine serves.

A model that defers part of its warmup exposes ``background_warmup_step()`` (does one unit; returns whether more
remain) and ``background_warmup_pending()``. Units run on the EngineCore thread, never inside a step's own work:

- idle: ``install()`` wraps ``EngineCoreProc._process_input_queue``; while the engine has no work (no request, no
  batch in flight) and nothing is queued, it asks the worker for one unit at a time. A request that arrives waits for
  at most the unit in progress.
- busy: the worker calls ``busy_step(model)`` at the start of every ``execute_model``; once
  ``TT_BACKGROUND_WARMUP_BUSY_INTERVAL_S`` (default 5) seconds have passed since the last unit it runs one, before the
  step's forward. So an engine that is never idle (steady load, or a pool whose engines are all still warming) still
  finishes, in about (units x interval); every running request stalls for that unit (0.3-0.6 s for a MiMo prefill
  trace capture). 0 disables it (units only while idle). The model must accept a unit at that point (MiMo flushes
  the previous step's pending draws and synchronizes the device before it captures, as its own recapture of the
  decode traces after a new prefill program does in the middle of serving).

``TT_READINESS_FILE`` (optional): the worker writes ``{"pending": n, "full": bool, "t": epoch}`` there after warmup
and after every unit, so a host agent can tell an engine serving with deferred work left from a fully warmed one.
"""

import json
import os
import time

from vllm.logger import init_logger

logger = init_logger(__name__)
READINESS_FILE = os.environ.get("TT_READINESS_FILE")
BUSY_INTERVAL_S = float(os.environ.get("TT_BACKGROUND_WARMUP_BUSY_INTERVAL_S", "5") or 0)

# last: monotonic time of the last unit (or of the end of warmup); None until warmup ended with deferred work.
_state = {"last": None, "busy_units": 0, "done": False}


def write_readiness(pending):
    if not READINESS_FILE:
        return
    tmp = f"{READINESS_FILE}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"pending": int(pending), "full": pending == 0, "t": time.time()}, f)
        os.replace(tmp, READINESS_FILE)
    except OSError as e:
        logger.warning("could not write %s: %s", READINESS_FILE, e)


def pending(model):
    fn = getattr(model, "background_warmup_pending", None)
    return int(fn()) if fn is not None else 0


def started():
    """Warmup ended with deferred work: the busy interval counts from now."""
    _state["last"] = time.monotonic()


def step(model):
    """One deferred unit of ``model``'s warmup; returns whether more remain."""
    fn = getattr(model, "background_warmup_step", None)
    if fn is None or _state["done"]:
        return False
    more = bool(fn())
    _state["last"] = time.monotonic()
    if not more:
        _state["done"] = True
    write_readiness(pending(model))
    return more


def busy_step(model, interval_s=None):
    """At the start of a step: one unit if deferred work remains and ``interval_s`` passed since the last one (or
    since warmup ended). Returns whether a unit ran."""
    interval_s = BUSY_INTERVAL_S if interval_s is None else interval_s
    if _state["done"] or interval_s <= 0 or _state["last"] is None:
        return False
    if time.monotonic() - _state["last"] < interval_s or not pending(model):
        return False
    _state["busy_units"] += 1
    if not step(model):
        logger.info("background warmup finished (%d units ran between busy steps)", _state["busy_units"])
    return True


_installed = False


def wrap(original):
    """``EngineCoreProc._process_input_queue`` that first runs deferred units while the engine is idle."""

    def _process_input_queue(self):
        if getattr(self, "_tt_background_pending", True):
            t0, units = time.time(), 0
            while not self.has_work() and self.is_running() and self.input_queue.empty():
                more = self.model_executor.collective_rpc("background_warmup_step")
                units += 1
                if not any(more):
                    self._tt_background_pending = False
                    logger.info("background warmup finished (%d units in this idle stretch)", units)
                    break
            if units and getattr(self, "_tt_background_pending", True):
                logger.debug("background warmup paused after %d units, %.2f s", units, time.time() - t0)
        return original(self)

    return _process_input_queue


def install():
    global _installed
    if _installed:
        return
    from vllm.v1.engine import core

    core.EngineCoreProc._process_input_queue = wrap(core.EngineCoreProc._process_input_queue)
    _installed = True
