# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Unit tests for ``TTModelRunner._kv_cache_shape``.

The per-device KV buffer is allocated here and written by the tt-metal
model, in a different repository, with no shared source of truth for how
many KV heads land on each chip. The two derivations have to agree
exactly: the paged-cache ops reinterpret a buffer whose head count
disagrees with the layer's, which preserves bytes but silently shrinks
the tokens-per-block, so a mismatch shows up as a tile-alignment assert
inside a kernel rather than as a shape error.

The model side shards KV heads along the mesh's *column* axis and
replicates down the rows (``ShardTensor2dMesh(..., dims=(None, dim))``,
``tp = mesh_device.shape[1]``). Every mesh the plugin had run on before
Blackhole Galaxy was a single row, where the column count and the device
count are the same number -- so dividing by the device count was correct
by coincidence. These tests pin the distinction.
"""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.model_runner import TTModelRunner


def _runner(num_devices: int, mesh_shape, tt_data_parallel_size: int = 1):
    """A runner carrying only the attributes ``_kv_cache_shape`` reads.

    ``__new__`` skips ``__init__``, which wants a real ``VllmConfig`` and a
    real mesh; the method under test touches three attributes.
    """
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.num_devices = num_devices
    runner.tt_data_parallel_size = tt_data_parallel_size
    runner.mesh_device = SimpleNamespace(shape=mesh_shape)
    return runner


def _spec(num_kv_heads: int, head_size: int = 256, block_size: int = 64):
    return SimpleNamespace(
        num_kv_heads=num_kv_heads, head_size=head_size, block_size=block_size
    )


# (id, num_devices, mesh_shape, dp, expected tp)
#
# The mesh shape is the one in force when the KV cache is allocated, which
# is after ``load_model``. That matters for the data-parallel rows: a
# 32-device mesh is reshaped from (8, 4) to (4, 8) during model load,
# before being carved into ``1 x (32 // dp)`` submeshes, so the parent
# reports 8 columns while the model runs on 4 or 8.
_MESHES = [
    ("t3k", 8, (1, 8), 1, 8),
    ("p300x2", 2, (1, 2), 1, 2),
    ("p150x8", 8, (1, 8), 1, 8),
    ("galaxy_lane_dp4", 32, (4, 8), 4, 8),
    ("galaxy_lane_dp8", 32, (4, 8), 8, 4),
    ("bh_galaxy", 32, (8, 4), 1, 4),
    ("single", 1, (1, 1), 1, 1),
]


@pytest.mark.parametrize(
    "num_devices,mesh_shape,dp,expected_tp",
    [pytest.param(*m[1:], id=m[0]) for m in _MESHES],
)
def test_tensor_parallel_width(num_devices, mesh_shape, dp, expected_tp):
    assert _runner(num_devices, mesh_shape, dp)._tensor_parallel_width() == expected_tp


@pytest.mark.parametrize(
    "num_devices,mesh_shape,dp,expected_tp",
    [pytest.param(*m[1:], id=m[0]) for m in _MESHES],
)
def test_tp_never_exceeds_the_submesh(num_devices, mesh_shape, dp, expected_tp):
    """A model cannot shard across more devices than its submesh holds."""
    runner = _runner(num_devices, mesh_shape, dp)
    assert runner._tensor_parallel_width() <= num_devices // dp


def test_single_row_meshes_are_unchanged_by_the_column_derivation():
    """Regression guard for every mesh that worked before Blackhole Galaxy.

    On a single-row mesh the column count *is* the device count, so the
    superseded ``num_kv_heads // min(devices, num_kv_heads)`` and the
    column-based derivation have to agree. If they ever diverge here, the
    fix has changed behaviour on shipped hardware.
    """
    for _id, num_devices, mesh_shape, dp, _tp in _MESHES:
        if mesh_shape[0] != 1:
            continue
        runner = _runner(num_devices, mesh_shape, dp)
        devices = num_devices // dp
        for num_kv_heads in (1, 2, 4, 8, 16):
            superseded = num_kv_heads // min(devices, num_kv_heads)
            assert runner._kv_cache_shape(_spec(num_kv_heads), 8)[1] == superseded


def test_blackhole_galaxy_sliding_layer_gets_the_model_s_head_count():
    """gemma-4-31B-it, 8x4, 16 KV heads: the model uses ``16 // tp(4) = 4``.

    Allocating 1 here (the device-count derivation, ``16 // min(32, 16)``)
    is what trips ``paged_fill_cache``'s ``effective_block_size %
    TILE_HEIGHT`` check during warmup prefill.
    """
    runner = _runner(num_devices=32, mesh_shape=(8, 4))
    assert runner._kv_cache_shape(_spec(16, head_size=256), 4128) == (4128, 4, 64, 256)


def test_blackhole_galaxy_full_attention_layer_is_unchanged():
    """The 10 full-attention layers carry 4 KV heads at head_dim 512.

    ``4 // tp(4) == 1``, which is what they were already getting -- the fix
    must move the sliding layers only.
    """
    runner = _runner(num_devices=32, mesh_shape=(8, 4))
    assert runner._kv_cache_shape(_spec(4, head_size=512), 4128) == (4128, 1, 64, 512)


def test_fewer_kv_heads_than_tp_clamps_to_one():
    """Mirrors the models' GQA fallback: ``1 if kv_replicated else kv // tp``.

    ``kv_replicated`` is defined as ``num_key_value_heads < tp``, where each
    device holds the single KV head its Q heads map to. Integer division
    would allocate zero.
    """
    runner = _runner(num_devices=8, mesh_shape=(1, 8))
    assert runner._kv_cache_shape(_spec(4), 8)[1] == 1
    assert runner._kv_cache_shape(_spec(1), 8)[1] == 1


def test_num_blocks_and_layout_pass_through():
    runner = _runner(num_devices=32, mesh_shape=(8, 4))
    shape = runner._kv_cache_shape(_spec(16, head_size=256, block_size=128), 2080)
    assert shape == (2080, 4, 128, 256)


def test_missing_mesh_shape_falls_back_to_the_device_count():
    """Tests and single-device runs hand over objects with no ``shape``."""
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.num_devices = 8
    runner.tt_data_parallel_size = 1
    runner.mesh_device = object()
    assert runner._tensor_parallel_width() == 8


def test_reinterpret_is_a_no_op_at_the_corrected_head_count():
    """The invariant the tt-metal helper inverts, restated as a test.

    ``effective_block_size`` solves
    ``input_kv * eff_bs * input_hd == cache_kv * cache_bs * cache_hd``.
    When the allocation matches the layer's own view the result equals the
    declared block size, so no reinterpret happens and the kernel's
    tile-alignment constraint is satisfied for any legal block size. Under
    the device-count derivation ``cache_kv`` was 1 and ``eff_bs`` came out
    at ``block_size / 4`` -- 16 at the default 64, below TILE_HEIGHT.
    """
    block_size, head_dim, model_kv_heads = 64, 256, 4
    runner = _runner(num_devices=32, mesh_shape=(8, 4))
    _, cache_kv, cache_bs, cache_hd = runner._kv_cache_shape(
        _spec(16, head_size=head_dim, block_size=block_size), 4128
    )

    eff_bs = (cache_kv * cache_bs * cache_hd) // (model_kv_heads * head_dim)

    assert eff_bs == block_size
    assert eff_bs % 32 == 0  # TILE_HEIGHT
