# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Keep process-global heap freezing out of the pytest process."""

import subprocess
import sys
from pathlib import Path

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


def test_example_installs_lifespan_patch_in_runpy_globals():
    code = r"""
import importlib.abc
import importlib.util
import runpy
import sys
from contextlib import asynccontextmanager

from vllm.entrypoints.serve.utils import server_utils
from vllm_tt_plugin.platform import _install_api_lifespan_gc_patch

@asynccontextmanager
async def original(app):
    yield

server_utils.lifespan = original
name = "vllm.entrypoints.openai.api_server"
sys.modules.pop(name, None)

class API(importlib.abc.MetaPathFinder, importlib.abc.InspectLoader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == name:
            return importlib.util.spec_from_loader(fullname, self)

    def get_source(self, fullname):
        return (
            'from vllm.entrypoints.serve.utils import server_utils\n'
            'from vllm.entrypoints.serve.utils.server_utils import lifespan\n'
            'from vllm_tt_plugin.platform import _install_api_lifespan_gc_patch\n'
            '_install_api_lifespan_gc_patch()\n'
            'assert lifespan is server_utils.lifespan\n'
            'assert getattr(lifespan, "_tt_lifespan_gc_patch", False)\n'
        )

    def is_package(self, fullname):
        return False

sys.meta_path.insert(0, API())
runpy.run_path(sys.argv[1], run_name="__main__")
"""
    example = Path(__file__).resolve().parents[1] / "examples/server_example_tt.py"
    subprocess.run([sys.executable, "-c", code, str(example)], check=True, timeout=90)
