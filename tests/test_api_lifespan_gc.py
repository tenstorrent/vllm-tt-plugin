# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Keep process-global heap freezing out of the pytest process."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("failure", ["none", "body", "serving"])
def test_api_lifespan_reclaims_frozen_state(failure):
    code = """
import asyncio
import gc
import sys
import types
import weakref
from contextlib import asynccontextmanager

from fastapi import FastAPI
from vllm.entrypoints.serve.utils import server_utils
from vllm_tt_plugin.platform import _install_api_lifespan_gc_patch

class Cycle:
    def __init__(self):
        self.link = self

class Serving:
    def shutdown(self):
        raise RuntimeError("serving")

# Exercise the upstream contract even on runtimes with a downstream GC patch.
@asynccontextmanager
async def original(app):
    gc.freeze()
    try:
        yield
    finally:
        try:
            serving = getattr(app.state, "openai_serving_transcription", None)
            if serving is not None:
                serving.shutdown()
        finally:
            del app.state

server_utils.lifespan = original
alias = types.ModuleType("vllm.entrypoints.openai.api_server")
alias.lifespan = original
sys.modules[alias.__name__] = alias
sys.modules["__main__"].lifespan = original
_install_api_lifespan_gc_patch()
wrapped = server_utils.lifespan
_install_api_lifespan_gc_patch()
assert server_utils.lifespan is wrapped
assert alias.lifespan is wrapped
assert sys.modules["__main__"].lifespan is wrapped

async def run():
    app = FastAPI()
    app.state.cycle = Cycle()
    retained = weakref.ref(app.state.cycle)
    failure = sys.argv[1]
    if failure == "serving":
        app.state.openai_serving_transcription = Serving()
    try:
        async with alias.lifespan(app):
            assert retained() is not None
            if failure == "body":
                raise RuntimeError("body")
    except RuntimeError as exc:
        assert str(exc) == failure
    else:
        assert failure == "none"
    assert not hasattr(app, "state")
    assert retained() is None

try:
    asyncio.run(run())
finally:
    gc.unfreeze()
    gc.collect()
"""
    subprocess.run([sys.executable, "-c", code, failure], check=True, timeout=90)
