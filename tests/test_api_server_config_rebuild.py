# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Every ``vllm serve`` API server must be able to rebuild the TT config.

With more than one API server (``vllm serve``'s default under
``--data-parallel-size N``), the launcher builds a VllmConfig from the parsed
CLI args, then pickles those same args to each API server process, which
builds its own VllmConfig from them. EngineArgs hands some sub-config objects
from the args to VllmConfig by reference, so the TT config hook running in the
launcher must leave the args in a state that still builds a valid config.
"""

import pickle
from pathlib import Path
from unittest.mock import patch

import pytest
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_tt_plugin.config import get_tt_data_parallel_size

LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"
DATA_PARALLEL_SIZE = 2
API_SERVER_COUNT = DATA_PARALLEL_SIZE


def _create_engine_config(args, model_class: type, api_process_rank: int):
    """Build a VllmConfig from CLI args the way ``vllm serve`` does.

    Rank -1 is the launcher (``run_multi_api_server``); ranks 0..N-1 are the
    API server processes (``build_async_engine_client``).
    """
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args._api_process_count = API_SERVER_COUNT
    engine_args._api_process_rank = api_process_rank
    with (
        # Each of these builds runs in its own process under `vllm serve`.
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch(
            "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            return_value=(model_class, None),
        ),
    ):
        return engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER
        )


@pytest.mark.parametrize(
    "folds_dp_into_lanes", [True, False], ids=["lane-dp", "standard-dp"]
)
def test_api_servers_rebuild_config_from_launcher_args(folds_dp_into_lanes):
    model_class = type(
        "StubModel",
        (),
        {"model_capabilities": {"folds_dp_into_lanes": folds_dp_into_lanes}},
    )
    args = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser()).parse_args(
        [
            "--model",
            str(LOCAL_MODEL_CONFIG),
            "--skip-tokenizer-init",
            "--dtype",
            "float16",
            "--max-model-len",
            "256",
            "--max-num-seqs",
            "4",
            "--data-parallel-size",
            str(DATA_PARALLEL_SIZE),
        ]
    )

    # Lane-DP folds the DP ranks into lanes of one engine; standard DP keeps
    # one engine per rank.
    engines, lanes = (
        (1, DATA_PARALLEL_SIZE) if folds_dp_into_lanes else (DATA_PARALLEL_SIZE, 1)
    )

    launcher_config = _create_engine_config(args, model_class, -1)

    assert launcher_config.parallel_config.data_parallel_size == engines
    assert get_tt_data_parallel_size(launcher_config) == lanes
    assert launcher_config.structured_outputs_config.disable_any_whitespace is True

    for rank in range(API_SERVER_COUNT):
        api_server_args = pickle.loads(pickle.dumps(args))

        api_server_config = _create_engine_config(api_server_args, model_class, rank)

        # The API server picks its engine client from its own config, so it
        # must resolve the engine layout the launcher started.
        assert api_server_config.parallel_config.data_parallel_size == engines
        assert get_tt_data_parallel_size(api_server_config) == lanes
        assert (
            api_server_config.structured_outputs_config.disable_any_whitespace is True
        )
