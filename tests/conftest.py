# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest

# `TTPlatform.check_and_update_config` records its results on the class rather
# than on the config it is handed, so one test that calls it configures every
# test that runs after it, across files.
_TT_PLATFORM_CONFIG_ATTRS = (
    "_standard_dp_visible_device_groups",
    "_standard_dp_mesh_grids",
    "sample_on_device_mode",
    "always_compat_sampling",
    "output_tokens_per_step",
    "block_model_config",
)


@pytest.fixture(autouse=True)
def reset_tt_platform_class_state():
    # Deferred: importing the platform from conftest runs before vLLM has
    # finished resolving its platform plugins, and that import is circular.
    import os

    import vllm.v1.engine.input_processor as input_processor

    from vllm_tt_plugin.platform import TTPlatform

    unset = object()
    saved = {
        name: TTPlatform.__dict__.get(name, unset) for name in _TT_PLATFORM_CONFIG_ATTRS
    }
    saved_process_inputs = input_processor.InputProcessor.process_inputs
    saved_original_process_inputs = input_processor.__dict__.get(
        "_tt_original_process_inputs", unset
    )
    saved_v2_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER", unset)

    yield

    input_processor.InputProcessor.process_inputs = saved_process_inputs
    if saved_original_process_inputs is unset:
        input_processor.__dict__.pop("_tt_original_process_inputs", None)
    else:
        input_processor._tt_original_process_inputs = saved_original_process_inputs

    if saved_v2_runner is unset:
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    else:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = saved_v2_runner

    for name, value in saved.items():
        if value is unset:
            if name in TTPlatform.__dict__:
                delattr(TTPlatform, name)
        else:
            setattr(TTPlatform, name, value)


@pytest.fixture
def vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank=0,
            data_parallel_rank_local=0,
            data_parallel_index=0,
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            world_size=1,
            local_world_size=1,
            assigned_physical_gpu_ids=None,
            worker_cls="auto",
            data_parallel_backend="mp",
            nnodes=1,
            node_rank=0,
            # Seeded with vLLM's own default so the assertions below catch a
            # plugin-side override rather than the absence of one.
            dp_engine_core_proc_cls="vllm.v1.engine.core.DPEngineCoreProc",
        ),
        model_config=SimpleNamespace(
            model="dummy",
            hf_config=SimpleNamespace(architectures=["DummyModel"]),
            max_logprobs=10,
            max_model_len=4,
            original_max_model_len=None,
            is_moe=False,
            get_sliding_window=lambda: None,
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False,
            async_scheduling=False,
            scheduler_cls=None,
            max_num_seqs=4,
            max_num_batched_tokens=4,
            verify_max_model_len=lambda _max_model_len: None,
        ),
        speculative_config=None,
        lora_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        structured_outputs_config=SimpleNamespace(disable_any_whitespace=False),
    )
