# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest


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
