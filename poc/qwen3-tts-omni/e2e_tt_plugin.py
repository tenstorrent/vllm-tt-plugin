# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run vLLM-Omni's own Qwen3-TTS online-serving tests against the TT deployment.

The tests start their server from vLLM-Omni's CUDA deploy config, which puts both
stages on device 0. This pytest plugin swaps in the TT deploy config, and optionally
another checkpoint, on every collected test before its server starts. The test files
themselves run unchanged. From a vLLM-Omni v0.30.0 checkout:

    PYTHONPATH=<this dir> python -m pytest -p e2e_tt_plugin --run-level full_model \\
        tests/e2e/online_serving/test_qwen3_tts_base.py \\
        tests/e2e/online_serving/test_qwen3_tts_base_expansion.py \\
        [--tt-model Qwen/Qwen3-TTS-12Hz-1.7B-Base]
"""

from pathlib import Path

TT_DEPLOY_CONFIG = Path(__file__).parent / "deploy" / "tt_qwen3_tts.yaml"


def pytest_addoption(parser):
    parser.addoption(
        "--tt-deploy-config",
        default=str(TT_DEPLOY_CONFIG),
        help="deploy config the tests' servers start from",
    )
    parser.addoption(
        "--tt-model",
        default=None,
        help="checkpoint to serve instead of the one each test names",
    )


def pytest_collection_modifyitems(config, items):
    deploy_config = config.getoption("--tt-deploy-config")
    model = config.getoption("--tt-model")
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        server = params.get("omni_server")
        if server is None:
            continue
        params["omni_server"] = server._replace(
            stage_config_path=deploy_config, model=model or server.model
        )
