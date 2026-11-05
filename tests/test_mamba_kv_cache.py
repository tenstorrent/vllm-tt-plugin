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
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy, get_kv_cache_configs
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
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


def _layer_specs(attention_block_size: int = BLOCK_SIZE, **mamba_kwargs) -> dict:
    attention = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.bfloat16,
    )
    mamba_kwargs = {"mamba_cache_mode": "none", **mamba_kwargs}
    gdn = MambaSpec(
        # Upstream resolves mamba_block_size to max_model_len in mode none.
        block_size=MAX_MODEL_LEN
        if mamba_kwargs["mamba_cache_mode"] == "none"
        else BLOCK_SIZE,
        shapes=GDN_SHAPES,
        dtypes=GDN_DTYPES,
        **mamba_kwargs,
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
    """The fields vLLM's KV planner, the TT sizing and the runner read."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="hybrid-mamba-test",
            max_model_len=MAX_MODEL_LEN,
            original_max_model_len=MAX_MODEL_LEN,
            get_sliding_window=lambda: None,
            is_encoder_decoder=False,
            is_multimodal_model=False,
            uses_mrope=False,
            hf_config=SimpleNamespace(model_type="hybrid-mamba-test"),
            get_vocab_size=lambda: 32,
            logits_processors=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=MAX_BATCH,
            max_num_batched_tokens=MAX_MODEL_LEN,
            disable_hybrid_kv_cache_manager=False,
            async_scheduling=False,
        ),
        lora_config=None,
        load_config=None,
        observability_config=None,
        device_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
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
        num_prefill_lookahead_tokens=0,
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


class _AllocatingModel:
    """A model with the allocation hook only."""

    def __init__(self):
        self.per_layer_specs = None
        self.calls: list[tuple[str, dict]] = []

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        self.per_layer_specs = per_layer_specs
        return ["kv"]

    def prefill_forward(self, **kwargs):
        self.calls.append(("prefill_forward", kwargs))

    def decode_forward(self, **kwargs):
        self.calls.append(("decode_forward", kwargs))


class _ZeroingModel(_AllocatingModel):
    def zero_kv_cache_blocks(self, **kwargs):
        self.calls.append(("zero_kv_cache_blocks", kwargs))


class _RecordingModel(_ZeroingModel):
    """A model with every hook a MambaSpec model can need, recording each call."""

    def copy_kv_cache_blocks(self, **kwargs):
        self.calls.append(("copy_kv_cache_blocks", kwargs))


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
    runner._num_speculative_tokens = 0
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
    ("mamba_kwargs", "match"),
    [
        # The launch speculates with 0 tokens here.
        (dict(num_speculative_blocks=1), "differs from the launch's 0"),
        # Mode none: vLLM places prefill checkpoints in mode align only.
        (dict(num_prefill_checkpoint_blocks=1), "prefill checkpoints need mode"),
    ],
)
def test_unsupported_mamba_configs_fail_at_kv_cache_init(mamba_kwargs, match):
    cfg = _vllm_config()
    kv_cache_config = _kv_cache_config(cfg, _layer_specs(**mamba_kwargs), 256)

    with pytest.raises(ValueError, match=match):
        TTModelRunner.initialize_kv_cache(_runner(cfg), kv_cache_config)


@pytest.mark.parametrize(
    ("mamba_cache_mode", "model_cls", "fails"),
    [
        ("none", _AllocatingModel, True),
        ("none", _ZeroingModel, False),
        ("align", _ZeroingModel, True),
        ("align", _RecordingModel, False),
    ],
)
def test_mamba_models_must_implement_the_block_hooks(
    mamba_cache_mode, model_cls, fails
):
    """Mode none needs zero_kv_cache_blocks; mode align also copy_kv_cache_blocks."""
    cfg = _vllm_config(mamba_cache_mode)
    specs = _layer_specs(mamba_cache_mode=mamba_cache_mode)
    kv_cache_config = _kv_cache_config(cfg, specs, 256)
    runner = _runner(cfg, model_cls())

    if fails:
        with pytest.raises(ValueError):
            TTModelRunner.initialize_kv_cache(runner, kv_cache_config)
    else:
        TTModelRunner.initialize_kv_cache(runner, kv_cache_config)


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


# region State blocks, zeroing and copies


def test_state_block_selection_matches_upstream():
    """The port picks the same column as ``mamba_get_block_table_tensor``."""
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

    from vllm_tt_plugin.mamba_state import MambaCacheGroups

    block_table = torch.arange(1, 6 * 24 + 1, dtype=torch.int32).reshape(6, 24)
    seq_lens = torch.tensor([0, 1, 1024, 1025, 5000, 20_000])
    for mode in ("none", "align"):
        spec = _layer_specs(mamba_cache_mode=mode)["model.layers.0.linear_attn"]
        groups = MambaCacheGroups(
            group_indices=(0,),
            mamba_cache_mode=mode,
            block_size=spec.block_size,
            attention_group_index=1,
        )
        expected = mamba_get_block_table_tensor(block_table, seq_lens, spec, mode)
        # Mode none passes the whole row through; its state is column 0.
        assert torch.equal(
            groups.state_block_ids(block_table, seq_lens), expected[:, :1]
        )


def _live_runner(cfg, specs, model) -> TTModelRunner:
    """A runner built by ``TTModelRunner.__init__``, KV cache initialized."""
    runner = TTModelRunner(
        vllm_config=cfg,
        mesh_device=None,
        trace_mode="none",
        enable_model_warmup=False,
        num_devices=1,
    )
    runner.model = model
    runner.initialize_kv_cache(_kv_cache_config(cfg, specs, 256))
    return runner


def _new_request(req_id: str, prompt_len: int, block_ids) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=block_ids,
        num_computed_tokens=0,
        lora_request=None,
    )


def _step(
    num_scheduled: dict[str, int],
    new=(),
    cached=(),
    zero=None,
    copies=None,
) -> SchedulerOutput:
    """``cached`` holds ``(req_id, num_computed_tokens, new_block_ids)``."""
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = list(new)
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=[req_id for req_id, _, _ in cached],
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in cached],
        all_token_ids={},
        new_block_ids=[block_ids for _, _, block_ids in cached],
        num_computed_tokens=[computed for _, computed, _ in cached],
        num_output_tokens=[0 for _ in cached],
    )
    output.num_scheduled_tokens = dict(num_scheduled)
    output.total_num_scheduled_tokens = sum(num_scheduled.values())
    output.new_block_ids_to_zero = zero
    output.kv_cache_block_copies = copies
    return output


def _state_ids(model_input, layer: int) -> list[int]:
    return model_input.state_block_ids_per_layer[layer].reshape(-1).tolist()


def test_mode_none_reads_state_from_column_0():
    """Mode none: a request's single GDN block per group is its state block.

    Requests A and B prefill, then decode. Every GDN layer gets the block of
    its own group for each row, attention layers get none, and the padding
    rows get the null block 0. vLLM lists the new attention blocks in
    ``new_block_ids_to_zero``, but no GDN group has owned them, so the model
    zeroes none.
    """
    cfg = _vllm_config()
    model = _RecordingModel()
    runner = _live_runner(cfg, _layer_specs(), model)
    # Group order: GDN groups 0, 1, 2, then attention group 3.
    blocks_a = ([10], [11], [12], [20, 21])
    blocks_b = ([30], [31], [32], [40])

    prefill = runner.build_model_input(
        _step(
            {"a": 1500, "b": 700},
            new=[_new_request("a", 1500, blocks_a), _new_request("b", 700, blocks_b)],
            zero=[20, 21, 40],
        ),
        None,
    )

    assert model.calls == []
    assert len(prefill.state_block_ids_per_layer) == NUM_LAYERS
    for layer in range(NUM_LAYERS):
        if _is_attention(layer):
            assert prefill.state_block_ids_per_layer[layer] is None
            continue
        group = layer % 4
        assert prefill.state_block_ids_per_layer[layer].shape == (MAX_BATCH, 1)
        ids = _state_ids(prefill, layer)
        assert ids[:2] == [blocks_a[group][0], blocks_b[group][0]]
        assert set(ids[2:]) == {0}

    runner.submit_prefill(prefill, [2])
    name, kwargs = model.calls[-1]
    assert name == "prefill_forward"
    assert kwargs["state_block_ids_per_layer"] is prefill.state_block_ids_per_layer

    decode = runner.build_model_input(
        _step({"a": 1, "b": 1}, cached=[("a", 1500, None), ("b", 700, None)]),
        None,
    )
    assert decode.prompt_lens is None
    assert _state_ids(decode, 0)[:2] == [10, 30]
    assert _state_ids(decode, 6)[:2] == [12, 32]
    runner.async_decode.submit_decode(decode, read_from_device=True)
    name, kwargs = model.calls[-1]
    assert name == "decode_forward"
    assert kwargs["state_block_ids_per_layer"] is decode.state_block_ids_per_layer


def _zeroed(model) -> list[list[int]]:
    return [
        kwargs["block_ids"]
        for name, kwargs in model.calls
        if name == "zero_kv_cache_blocks"
    ]


def test_only_attention_blocks_a_gdn_group_owned_are_zeroed():
    """vLLM lists every new attention block in ``new_block_ids_to_zero``.

    The model allocates its KV buffers zeroed and attention writes finite
    values, so only a block a GDN group owned since its last zeroing can read
    as NaN at the positions attention masks, and only if the prefill that
    gets it does not write it whole. The runner hands the model those blocks
    and no others: a block passes from request A's GDN group 0 to request B's
    attention group and is zeroed; a block that passes from A's attention
    group to B's is not; a block zeroed once and kept by attention is not
    zeroed again; a GDN block that a running request gets with its new blocks
    counts too; a GDN block that a prefill chunk writes whole is not zeroed.
    """
    cfg = _vllm_config()
    model = _RecordingModel()
    runner = _live_runner(cfg, _layer_specs(), model)
    assert runner._mamba_groups.zeroes_only_state_blocks

    runner.build_model_input(
        _step(
            {"a": 1500},
            new=[_new_request("a", 1500, ([10], [11], [12], [20, 21]))],
            zero=[20, 21],
        ),
        None,
    )
    assert _zeroed(model) == []

    model.calls.clear()
    step = _step(
        {"b": 700},
        new=[_new_request("b", 700, ([20], [30], [31], [10, 21]))],
        zero=[10, 21],
    )
    step.finished_req_ids = {"a"}
    runner.build_model_input(step, None)
    assert _zeroed(model) == [[10]]

    model.calls.clear()
    step = _step(
        {"b": 1, "c": 700},
        new=[_new_request("c", 700, ([50], [51], [52], [11, 12]))],
        cached=[("b", 700, ([], [], [60], []))],
        zero=[11, 12],
    )
    runner.build_model_input(step, None)
    assert _zeroed(model) == [[11, 12]]

    model.calls.clear()
    step = _step(
        {"d": 700},
        new=[_new_request("d", 700, ([70], [71], [72], [10, 20, 60]))],
        zero=[10, 20, 60],
    )
    step.finished_req_ids = {"b", "c"}
    runner.build_model_input(step, None)
    assert _zeroed(model) == [[20, 60]]

    # Prefill chunks that write a block whole leave no position to read
    # unwritten: D's GDN blocks 70 and 71 go to E's first chunk, and 72 to its
    # second chunk, unzeroed.
    model.calls.clear()
    step = _step(
        {"e": 2048},
        new=[_new_request("e", 4096, ([90], [91], [92], [70, 71]))],
        zero=[70, 71],
    )
    step.finished_req_ids = {"d"}
    runner.build_model_input(step, None)
    assert _zeroed(model) == []

    model.calls.clear()
    step = _step(
        {"e": 2048}, cached=[("e", 2048, ([], [], [], [72, 21]))], zero=[72, 21]
    )
    runner.build_model_input(step, None)
    assert _zeroed(model) == []


def test_mixed_attention_precision_zeroes_every_listed_block():
    """Without ``zeroes_only_state_blocks`` (attention groups of different KV
    precisions, which vLLM also zeroes for) the runner hands the model every
    block vLLM lists."""
    cfg = _vllm_config()
    model = _RecordingModel()
    runner = _live_runner(cfg, _layer_specs(), model)
    runner._mamba_groups.zeroes_only_state_blocks = False

    runner.build_model_input(
        _step(
            {"a": 1500},
            new=[_new_request("a", 1500, ([10], [11], [12], [20, 21]))],
            zero=[20, 21],
        ),
        None,
    )
    assert _zeroed(model) == [[20, 21]]


def test_page_table_is_the_attention_groups_table():
    """``page_table`` is an attention table although group 0 is a GDN group.

    vLLM orders the groups by spec, so the GDN groups come first here. A model
    that reads ``page_table`` for attention must get the attention group's
    rows in prefill and in decode.
    """
    cfg = _vllm_config()
    model = _RecordingModel()
    runner = _live_runner(cfg, _layer_specs(), model)
    assert isinstance(
        runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec, MambaSpec
    )
    attention_group = 3
    blocks_a = ([10], [11], [12], [20, 21])
    blocks_b = ([30], [31], [32], [40])

    prefill = runner.build_model_input(
        _step(
            {"a": 1500, "b": 700},
            new=[_new_request("a", 1500, blocks_a), _new_request("b", 700, blocks_b)],
        ),
        None,
    )
    runner.submit_prefill(prefill, [2])
    decode = runner.build_model_input(
        _step({"a": 1, "b": 1}, cached=[("a", 1500, None), ("b", 700, None)]),
        None,
    )
    runner.async_decode.submit_decode(decode, read_from_device=True)

    forwards = [kwargs for name, kwargs in model.calls if name.endswith("_forward")]
    assert len(forwards) == 2
    for model_input, kwargs in zip((prefill, decode), forwards):
        page_table = kwargs["page_table"]
        assert torch.equal(
            page_table, model_input.block_tables_per_group[attention_group]
        )
        assert page_table[0, :2].tolist() == [20, 21]
        assert page_table[1, :1].tolist() == [40]


def test_attention_only_model_without_hooks_ignores_block_zeroing():
    """A model with no MambaSpec groups and no hooks behaves as before."""
    cfg = _vllm_config()
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
    )
    specs = {f"model.layers.{i}.self_attn": spec for i in range(4)}
    cfg.model_config.get_num_layers_by_block_type = lambda *_: 4
    model = _AllocatingModel()
    runner = _live_runner(cfg, specs, model)

    model_input = runner.build_model_input(
        _step({"a": 100}, new=[_new_request("a", 100, ([5],))], zero=[5]), None
    )

    assert model.calls == []
    assert model_input.state_block_ids_per_layer is None


def test_a_prefill_hands_the_model_the_states_it_must_leave_inside_its_chunk():
    """``MambaSpec`` prefill checkpoint blocks: the scheduler passes vLLM's
    ``(position, column)`` list per request (``_attach_mamba_checkpoints``);
    the prefill gets, per GDN layer, the block at each column of its group,
    and the positions. A row without an entry holds block 0 and position 0,
    and a step that is not a prefill carries neither keyword."""
    cfg = _vllm_config("align")
    model = _RecordingModel()
    specs = _layer_specs(
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=128,
    )
    runner = _live_runner(cfg, specs, model)
    assert runner._mamba_groups.prefill_checkpoints
    step = _step(
        {"a": 2500, "b": 700},
        new=[
            _new_request("a", 2500, ([0, 10, 11], [0, 20, 21], [0, 30, 31], [40])),
            _new_request("b", 700, ([50], [60], [70], [80])),
        ],
    )
    step._tt_mamba_checkpoints = {"a": ((2048, 1),)}
    prefill = runner.build_model_input(step, None)

    assert prefill.state_checkpoint_positions[:3].tolist() == [[2048], [0], [0]]
    for layer in range(NUM_LAYERS):
        ids = prefill.state_checkpoint_ids_per_layer[layer]
        if _is_attention(layer):
            assert ids is None
            continue
        assert ids.shape == (MAX_BATCH, 1)
        assert ids[:2, 0].tolist() == [(10, 20, 30)[layer % 4], 0]
    runner.submit_prefill(prefill, [2, 0])
    name, kwargs = model.calls[-1]
    assert name == "prefill_forward"
    assert kwargs["state_checkpoint_positions"] is prefill.state_checkpoint_positions
    assert (
        kwargs["state_checkpoint_ids_per_layer"]
        is prefill.state_checkpoint_ids_per_layer
    )

    decode = runner.build_model_input(
        _step({"a": 1, "b": 1}, cached=[("a", 2500, None), ("b", 700, None)]), None
    )
    assert decode.state_checkpoint_positions is None


@pytest.mark.parametrize(
    ("mamba_kwargs", "match"),
    [
        (dict(num_prefill_checkpoint_blocks=1), "an alignment"),
        (
            dict(num_prefill_checkpoint_blocks=2, prefill_checkpoint_alignment=128),
            "exactly 1 block",
        ),
    ],
)
def test_prefill_checkpoints_need_one_block_and_an_alignment(mamba_kwargs, match):
    cfg = _vllm_config("align")
    specs = _layer_specs(mamba_cache_mode="align", **mamba_kwargs)
    kv_cache_config = _kv_cache_config(cfg, specs, 256)
    with pytest.raises(ValueError, match=match):
        TTModelRunner.initialize_kv_cache(_runner(cfg), kv_cache_config)


def test_mode_align_moves_state_into_the_block_of_each_new_column():
    """Mode align: the running state moves forward one block-table column.

    Request A prefills 2500 tokens in chunks of 1024, 1024 and 452. Column p
    holds the state after (p + 1) * 1024 tokens, so each chunk that crosses
    into a new column first copies the state from the previous column's block
    to the new one, in every GDN group, after vLLM's own block copies, and the
    forward reads the new block.
    """
    cfg = _vllm_config("align")
    model = _RecordingModel()
    runner = _live_runner(cfg, _layer_specs(mamba_cache_mode="align"), model)
    chunk1 = runner.build_model_input(
        _step({"a": 1024}, new=[_new_request("a", 2500, ([10], [20], [30], [40]))]),
        None,
    )
    assert model.calls == []
    assert [_state_ids(chunk1, layer)[0] for layer in (0, 1, 2)] == [10, 20, 30]

    chunk2 = runner.build_model_input(
        _step(
            {"a": 1024},
            cached=[("a", 1024, ([11], [21], [31], [41]))],
            copies=[KVCacheBlockCopy(src_block_id=5, dst_block_id=6)],
        ),
        None,
    )
    assert model.calls == [
        ("copy_kv_cache_blocks", {"kv_cache": ["kv"], "copies": [(5, 6)]}),
        (
            "copy_kv_cache_blocks",
            {"kv_cache": ["kv"], "copies": [(10, 11), (20, 21), (30, 31)]},
        ),
    ]
    assert [_state_ids(chunk2, layer)[0] for layer in (0, 1, 2)] == [11, 21, 31]

    model.calls.clear()
    chunk3 = runner.build_model_input(
        _step({"a": 452}, cached=[("a", 2048, ([12], [22], [32], [42]))]), None
    )
    assert model.calls == [
        (
            "copy_kv_cache_blocks",
            {"kv_cache": ["kv"], "copies": [(11, 12), (21, 22), (31, 32)]},
        )
    ]
    assert [_state_ids(chunk3, layer)[0] for layer in (0, 1, 2)] == [12, 22, 32]

    # A finished request leaves no column behind for a reused request id.
    finished = _step({})
    finished.finished_req_ids = {"a"}
    runner.build_model_input(finished, None)
    assert runner._mamba_groups.state_columns == {}


@pytest.mark.parametrize("declared", [True, False])
def test_the_model_capability_turns_on_chained_state_moves(declared):
    """``supports_mamba_chained_state_moves`` is read when the cache is set up;
    absent, chained steps keep the host's moves (and the boundary blocker)."""
    cfg = _vllm_config()
    kv_cache_config = _kv_cache_config(cfg, _layer_specs(), 256)
    model = _RecordingModel()
    if declared:
        model.model_capabilities = {"supports_mamba_chained_state_moves": True}
    runner = _runner(cfg, model)

    TTModelRunner.initialize_kv_cache(runner, kv_cache_config)

    assert runner._mamba_groups.chained_moves_on_device is declared
