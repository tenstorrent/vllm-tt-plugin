# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmup work a model defers past engine start (e.g. MiMo's MIMO_DEFER_TRACES), run while the engine is idle.

A model that defers part of its warmup exposes ``background_warmup_step()`` (does one unit; returns whether more
remain) and ``background_warmup_pending()``. vLLM's EngineCore only calls into the worker to run steps, so
``install()`` wraps ``EngineCoreProc._process_input_queue``: while the engine has no work (no request, no batch in
flight) and nothing is queued, it asks the worker for one unit at a time. A request that arrives waits for at most
the unit in progress. Everything runs on the EngineCore thread, so no device work overlaps a step.

``TT_READINESS_FILE`` (optional): the worker writes ``{"pending": n, "full": bool, "t": epoch}`` there after warmup
and after every unit, so a host agent can tell an engine serving with deferred work left from a fully warmed one.
"""

import json
import os
import time

from vllm.logger import init_logger

logger = init_logger(__name__)
READINESS_FILE = os.environ.get("TT_READINESS_FILE")


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


def step(model):
    """One deferred unit of ``model``'s warmup; returns whether more remain."""
    fn = getattr(model, "background_warmup_step", None)
    if fn is None:
        return False
    more = bool(fn())
    write_readiness(model.background_warmup_pending())
    return more


def pending(model):
    fn = getattr(model, "background_warmup_pending", None)
    return int(fn()) if fn is not None else 0


_installed = False


def install():
    global _installed
    if _installed:
        return
    from vllm.v1.engine import core

    original = core.EngineCoreProc._process_input_queue

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

    core.EngineCoreProc._process_input_queue = _process_input_queue
    _installed = True
