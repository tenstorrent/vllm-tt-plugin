# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host tests for ``MambaSpec`` KV cache groups on TT.

The fake hybrid model has the per-device layout of Qwen3.6-27B at TP=4: 64
decoder layers, every fourth one full attention (16 layers, 1 KV head of 256,
bf16) and the rest gated delta network layers (48 layers, conv history
3 x 2560 bf16 and recurrent state 12 x 128 x 128 fp32). vLLM groups them into
three Mamba groups and one attention group of 16 layers each and overlays all
four groups on the same memory, so the i-th layer of every group shares one
layer slot.

The KV cache configs come from vLLM's own planner (``get_kv_cache_configs``),
so the tests also pin the vLLM behavior the TT runner relies on.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.worker import (
    _available_kv_cache_memory_bytes_for_num_blocks,
    get_num_available_blocks_tt,
)

NUM_LAYERS = 64
NUM_SLOTS = 16
BLOCK_SIZE = 1024
MAX_MODEL_LEN = 20_000
MAX_BATCH = 32
GDN_SHAPES = ((3, 2560), (12, 128, 128))
GDN_DTYPES = (torch.bfloat16, torch.float32)
# 1 KV head x head_size 256 x (K + V) x 2 bytes.
ATTENTION_BYTES_PER_TOKEN = 1024


def _is_attention(layer: int) -> bool:
    return layer % 4 == 3


def _layer_specs(
    attention_block_size: int = BLOCK_SIZE,
    mamba_cache_mode: str = "none",
    num_speculative_blocks: int = 0,
) -> dict:
    attention = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.bfloat16,
    )
    gdn = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=GDN_SHAPES,
        dtypes=GDN_DTYPES,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    return {
        (
            f"model.layers.{i}.self_attn"
            if _is_attention(i)
            else f"model.layers.{i}.linear_attn"
        ): (attention if _is_attention(i) else gdn)
        for i in range(NUM_LAYERS)
    }


class _CacheConfig(SimpleNamespace):
    def get_resolved_kv_cache_layout(self):
        return KVCacheLayout[self.kv_cache_layout]


def _vllm_config(mamba_cache_mode: str = "none") -> SimpleNamespace:
    """The fields vLLM's KV planner and the TT sizing read, as at engine start."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="hybrid-mamba-test",
            max_model_len=MAX_MODEL_LEN,
            original_max_model_len=MAX_MODEL_LEN,
            get_sliding_window=lambda: None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=MAX_BATCH,
            max_num_batched_tokens=MAX_MODEL_LEN,
            disable_hybrid_kv_cache_manager=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        cache_config=_CacheConfig(
            block_size=64,
            num_gpu_blocks_override=None,
            kv_cache_layout=KVCacheLayout.LBNHC.name,
            prefix_cache_retention_interval=None,
            mamba_cache_mode=mamba_cache_mode,
            enable_prefix_caching=mamba_cache_mode != "none",
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        additional_config={},
        speculative_config=None,
        kv_transfer_config=None,
        max_in_flight_tokens=MAX_MODEL_LEN,
    )


def _model_class(specs: dict, max_tokens_all_users: int):
    class HybridMambaModel:
        @staticmethod
        def get_max_tokens_all_users(**_kwargs):
            return max_tokens_all_users

        @staticmethod
        def get_kv_cache_spec(_config):
            return specs

    return HybridMambaModel


def _kv_cache_config(cfg, specs: dict, num_blocks: int):
    """vLLM's own plan for ``specs`` at exactly ``num_blocks`` blocks."""
    cfg.cache_config.num_gpu_blocks_override = num_blocks
    available = _available_kv_cache_memory_bytes_for_num_blocks(cfg, specs, num_blocks)
    [kv_cache_config] = get_kv_cache_configs(cfg, [specs], [available])
    assert kv_cache_config.num_blocks == num_blocks
    return kv_cache_config


class _RecordingModel:
    def __init__(self):
        self.per_layer_specs = None

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        self.per_layer_specs = per_layer_specs
        return ["kv"]


def _runner(cfg, model=None, tt_data_parallel_size: int = 1) -> TTModelRunner:
    """A runner with only the state ``initialize_kv_cache`` reads."""
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.model_config = cfg.model_config
    runner.parallel_config = cfg.parallel_config
    runner.scheduler_config = cfg.scheduler_config
    runner.cache_config = cfg.cache_config
    runner.tt_max_batch_size = MAX_BATCH
    runner.tt_per_lane_max_num_seqs = MAX_BATCH // tt_data_parallel_size
    runner.tt_data_parallel_size = tt_data_parallel_size
    runner.num_devices = tt_data_parallel_size
    runner.vocab_size = 32
    runner._host_logitsprocs = None
    runner._is_block_output_model = False
    runner._output_tokens_per_step = 1
    runner._mamba_groups = None
    runner.model = model if model is not None else _RecordingModel()
    return runner


@pytest.mark.parametrize(
    ("attention_block_size", "expected_blocks"),
    [
        # Unified page = the 1 MiB attention page; the GDN page is padded to
        # it. 131072 tokens * 1024 B / 1 MiB = 128 blocks, plus one output
        # block per request (32) and one state block per request in each of
        # the three GDN groups (96).
        (1024, 128 + 32 + 96),
        # A 64-token attention page (64 KiB) does not divide the 801,792 B GDN
        # page, so vLLM pads the attention page to it: each block costs
        # 801,792 B per layer slot. ceil(131072 * 1024 / 801792) = 168.
        (64, 168 + 32 + 96),
    ],
)
def test_mamba_pool_is_sized_from_bytes_per_layer_slot(
    attention_block_size, expected_blocks
):
    """The pool holds the model's attention budget in bytes, plus GDN state.

    The token formula for attention-only models, ``max_tokens_all_users /
    cache_config.block_size``, ignores both the page padding and the GDN
    state blocks; for the 64-token case it gives 2080 blocks.
    """
    cfg = _vllm_config()
    # Short enough that one request (128 64-token blocks) stays under the
    # budget, so the admission floor does not decide.
    cfg.model_config.max_model_len = 8192
    specs = _layer_specs(attention_block_size)
    with patch(
        "vllm_tt_plugin.worker.get_model_architecture",
        return_value=(_model_class(specs, 131_072), "hybrid-mamba-test"),
    ):
        blocks = get_num_available_blocks_tt(cfg)

    assert blocks == expected_blocks
    # vLLM's planner reconstructs that count from the reported byte budget.
    kv_cache_config = _kv_cache_config(cfg, specs, blocks)
    assert [type(g.kv_cache_spec) for g in kv_cache_config.kv_cache_groups] == [
        MambaSpec,
        MambaSpec,
        MambaSpec,
        FullAttentionSpec,
    ]
    assert all(len(g.layer_names) == NUM_SLOTS for g in kv_cache_config.kv_cache_groups)


def test_mamba_pool_floor_admits_one_full_length_request():
    """A tiny model budget still admits one max_model_len request."""
    cfg = _vllm_config()
    cfg.scheduler_config.max_num_seqs = 1
    specs = _layer_specs()
    with patch(
        "vllm_tt_plugin.worker.get_model_architecture",
        return_value=(_model_class(specs, 1024), "hybrid-mamba-test"),
    ):
        blocks = get_num_available_blocks_tt(cfg)

    # The budget gives 1 attention block, 1 output block and 3 state blocks.
    # One request needs ceil(20000 / 1024) = 20 attention blocks, one state
    # block in each GDN group and the null block, plus the output block.
    assert blocks == 20 + 3 + 1 + 1
    _kv_cache_config(cfg, specs, blocks)


def test_gdn_layers_share_the_layer_slot_of_their_group_position():
    """Layer ``4j + k`` of every group lands in layer slot ``j``."""
    cfg = _vllm_config()
    specs = _layer_specs()
    kv_cache_config = _kv_cache_config(cfg, specs, 256)
    model = _RecordingModel()
    runner = _runner(cfg, model)

    TTModelRunner.initialize_kv_cache(runner, kv_cache_config)

    per_layer = model.per_layer_specs
    assert len(per_layer) == NUM_LAYERS
    for layer, (shape, dtype, tensor_idx) in enumerate(per_layer):
        assert tensor_idx == layer // 4
        if _is_attention(layer):
            assert shape == (256, 1, BLOCK_SIZE, 256)
            assert dtype == torch.bfloat16
        else:
            assert (shape, dtype) == (GDN_SHAPES, GDN_DTYPES)
    # Group order follows the spec dict: the three GDN groups, then attention.
    assert runner._layer_to_group_idx == [layer % 4 for layer in range(NUM_LAYERS)]
    assert runner._mamba_groups.group_indices == (0, 1, 2)


def test_mamba_page_tables_are_sized_from_groups_and_padded_to_8_columns():
    """The page-table width comes from the group block sizes, rounded to 8.

    ``cache_config.block_size`` (64) is no group's block size here; it would
    give ceil(20000 / 64) = 313 columns.
    """
    cfg = _vllm_config()
    kv_cache_config = _kv_cache_config(cfg, _layer_specs(), 256)
    runner = _runner(cfg)

    TTModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # ceil(20000 / 1024) = 20, padded to 24.
    assert runner.max_num_blocks_per_req == 24
    per_group = runner.input_batch.block_tables_for_rows(
        [0], runner.max_num_blocks_per_req
    )
    per_layer = TTModelRunner._block_tables_per_layer(runner, per_group)
    assert len(per_layer) == NUM_LAYERS
    assert {tuple(t.shape) for t in per_layer} == {(MAX_BATCH, 24)}


def test_attention_only_page_table_width_is_unchanged():
    """Without MambaSpec groups the width stays min(cdiv(len, bs), blocks)."""
    cfg = _vllm_config()
    common = dict(block_size=64, num_kv_heads=1, head_size=256, dtype=torch.bfloat16)
    specs = {
        f"model.layers.{i}.self_attn": (
            FullAttentionSpec(**common)
            if i % 2
            else SlidingWindowSpec(**common, sliding_window=512)
        )
        for i in range(4)
    }
    kv_cache_config = _kv_cache_config(cfg, specs, 1000)
    runner = _runner(cfg)
    runner.model_config.get_num_layers_by_block_type = lambda *_: 4

    TTModelRunner.initialize_kv_cache(runner, kv_cache_config)

    assert runner._mamba_groups is None
    # ceil(20000 / 64) = 313, not a multiple of 8, kept as before.
    assert runner.max_num_blocks_per_req == 313


@pytest.mark.parametrize(
    "specs_kwargs",
    [
        # Prefix-cache checkpoints (mode align) are not implemented yet.
        dict(mamba_cache_mode="align"),
        dict(num_speculative_blocks=1),
    ],
)
def test_unsupported_mamba_configs_fail_at_kv_cache_init(specs_kwargs):
    cfg = _vllm_config(specs_kwargs.get("mamba_cache_mode", "none"))
    kv_cache_config = _kv_cache_config(cfg, _layer_specs(**specs_kwargs), 256)

    with pytest.raises(ValueError):
        TTModelRunner.initialize_kv_cache(_runner(cfg), kv_cache_config)


def test_mamba_spec_mode_must_match_the_cache_config():
    cfg = _vllm_config("none")
    kv_cache_config = _kv_cache_config(cfg, _layer_specs(mamba_cache_mode="align"), 256)

    with pytest.raises(ValueError):
        TTModelRunner.initialize_kv_cache(_runner(cfg), kv_cache_config)


def test_lane_dp_rejects_mamba_groups():
    cfg = _vllm_config()
    kv_cache_config = _kv_cache_config(cfg, _layer_specs(), 256)

    with pytest.raises(NotImplementedError):
        TTModelRunner.initialize_kv_cache(
            _runner(cfg, tt_data_parallel_size=2), kv_cache_config
        )


def test_gdn_layer_without_an_attention_layer_in_its_slot_is_rejected():
    """A layer slot that holds GDN state but no attention layer is rejected.

    15 attention layers and 10 GDN layers: vLLM uses groups of 10 layers, so
    the attention layers split into groups of 8 and 7 and the GDN layers at
    group positions 8 and 9 have no attention layer in their slot to size its
    buffer.
    """
    cfg = _vllm_config()
    specs = _layer_specs()
    attention = specs["model.layers.3.self_attn"]
    gdn = specs["model.layers.0.linear_attn"]
    specs = {
        **{f"model.layers.{i}.self_attn": attention for i in range(15)},
        **{f"model.layers.{i}.linear_attn": gdn for i in range(15, 25)},
    }
    kv_cache_config = _kv_cache_config(cfg, specs, 256)
    assert [len(g.layer_names) for g in kv_cache_config.kv_cache_groups] == [8, 7, 10]

    with pytest.raises(ValueError):
        TTModelRunner.initialize_kv_cache(_runner(cfg), kv_cache_config)
