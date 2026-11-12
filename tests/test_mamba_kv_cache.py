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
from vllm_tt_plugin.config import store_tt_lane_count, store_tt_output_tokens_per_step
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
    # Upstream resolves mamba_block_size to max_model_len in mode none.
    mamba_kwargs.setdefault(
        "block_size",
        MAX_MODEL_LEN if mamba_kwargs["mamba_cache_mode"] == "none" else BLOCK_SIZE,
    )
    gdn = MambaSpec(shapes=GDN_SHAPES, dtypes=GDN_DTYPES, **mamba_kwargs)
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


def _num_blocks(cfg, specs: dict, max_tokens_all_users: int) -> int:
    with patch(
        "vllm_tt_plugin.worker.get_model_architecture",
        return_value=(_model_class(specs, max_tokens_all_users), "hybrid-mamba-test"),
    ):
        return get_num_available_blocks_tt(cfg)


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


class _RecordingModel(_AllocatingModel):
    """A model with every hook a MambaSpec model needs, recording each call."""

    def zero_kv_cache_blocks(self, **kwargs):
        self.calls.append(("zero_kv_cache_blocks", kwargs))


def _runner(cfg, model=None) -> TTModelRunner:
    """A runner with only the state ``initialize_kv_cache`` reads."""
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.vllm_config = cfg
    runner.model_config = cfg.model_config
    runner.parallel_config = cfg.parallel_config
    runner.scheduler_config = cfg.scheduler_config
    runner.cache_config = cfg.cache_config
    runner.tt_max_batch_size = MAX_BATCH
    runner.tt_per_lane_max_num_seqs = MAX_BATCH
    runner.tt_data_parallel_size = 1
    runner.num_devices = 1
    runner.vocab_size = 32
    runner._host_logitsprocs = None
    runner._is_block_output_model = False
    runner._output_tokens_per_step = 1
    runner._mamba_groups = None
    runner.model = model if model is not None else _RecordingModel()
    return runner


# region Pool sizing


@pytest.mark.parametrize(
    ("attention_block_size", "expected_blocks"),
    [
        # Unified page = the 1 MiB attention page; the GDN page is padded to
        # it. 131072 tokens * 1024 B / 1 MiB = 128 blocks, plus one output
        # block per request (32), one state block per request in each of
        # the three GDN groups (96) and the null block.
        (1024, 128 + 32 + 96 + 1),
        # A 64-token attention page (64 KiB) does not divide the 801,792 B GDN
        # page, so vLLM pads the attention page to it: each block costs
        # 801,792 B per layer slot. ceil(131072 * 1024 / 801792) = 168.
        (64, 168 + 32 + 96 + 1),
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
    # budget, so the admission check passes.
    cfg.model_config.max_model_len = 8192
    specs = _layer_specs(attention_block_size, block_size=8192)

    blocks = _num_blocks(cfg, specs, 131_072)

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


def test_mamba_pool_below_one_full_length_request_fails_at_startup():
    """A model budget that cannot admit one max_model_len request is rejected.

    The budget gives 1 attention block, 1 output block, 3 state blocks and the
    null block. One request needs ceil(20000 / 1024) = 20 attention blocks,
    one state block in each GDN group and the null block, plus the output
    block: 25. The pool is not grown past the budget, as it is for
    sliding-window models.
    """
    cfg = _vllm_config()
    cfg.scheduler_config.max_num_seqs = 1
    specs = _layer_specs()

    with pytest.raises(ValueError, match="holds 6 blocks.*needs 25 blocks"):
        _num_blocks(cfg, specs, 1024)

    # A budget of exactly one request at max_model_len (20 blocks) admits it:
    # the pool adds the output, state and null blocks.
    assert _num_blocks(cfg, specs, 20 * BLOCK_SIZE) == 20 + 1 + 3 + 1
    # --max-model-len -1 keeps the budget; vLLM fits the length to the pool.
    cfg.model_config.original_max_model_len = -1
    assert _num_blocks(cfg, specs, 1024) == 6


@pytest.mark.parametrize(
    ("unsupported", "match"),
    [
        ("prefix_caching", "launch with --no-enable-prefix-caching"),
        ("speculative", "launch without --speculative-config"),
        ("lane_dp", "lane-DP"),
        ("block_output", "Block-output models"),
    ],
)
def test_unsupported_mamba_launches_fail_before_the_model_loads(unsupported, match):
    """The worker sizes the pool before it loads the model and rejects there.

    Prefix caching resolves ``mamba_cache_mode`` to ``align``, speculative
    decoding needs speculative state blocks, and lane-DP and block-output
    execution have no MambaSpec path.
    """
    mode = "align" if unsupported == "prefix_caching" else "none"
    cfg = _vllm_config(mode)
    if unsupported == "speculative":
        cfg.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    elif unsupported == "lane_dp":
        store_tt_lane_count(cfg, 2)
    elif unsupported == "block_output":
        store_tt_output_tokens_per_step(cfg, 4)

    with pytest.raises(ValueError, match=match):
        _num_blocks(cfg, _layer_specs(mamba_cache_mode=mode), 131_072)


# endregion

# region Allocation


def test_gdn_layers_share_the_layer_slot_of_their_group_position():
    """Layer ``4j + k`` of every group lands in layer slot ``j``."""
    cfg = _vllm_config()
    kv_cache_config = _kv_cache_config(cfg, _layer_specs(), 256)
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
    groups = kv_cache_config.kv_cache_groups
    assert runner._mamba_groups.layer_names == tuple(
        tuple(groups[i].layer_names) for i in (0, 1, 2)
    )


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
        (dict(num_speculative_blocks=1), "neither speculative nor prefill"),
        (dict(num_prefill_checkpoint_blocks=1), "neither speculative nor prefill"),
        # MambaManager would hold one state block per 1024 tokens.
        (dict(block_size=BLOCK_SIZE), "below max_model_len"),
        (dict(mamba_cache_mode="align"), "must use the resolved cache mode"),
    ],
)
def test_unsupported_mamba_specs_fail_at_kv_cache_init(mamba_kwargs, match):
    cfg = _vllm_config()
    kv_cache_config = _kv_cache_config(cfg, _layer_specs(**mamba_kwargs), 256)

    with pytest.raises(ValueError, match=match):
        TTModelRunner.initialize_kv_cache(_runner(cfg), kv_cache_config)


def test_lane_dp_rejects_mamba_groups_at_kv_cache_init():
    cfg = _vllm_config()
    kv_cache_config = _kv_cache_config(cfg, _layer_specs(), 256)
    store_tt_lane_count(cfg, 2)

    with pytest.raises(ValueError, match="lane-DP"):
        TTModelRunner.initialize_kv_cache(_runner(cfg), kv_cache_config)


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

    with pytest.raises(ValueError, match="share no buffer with an attention layer"):
        TTModelRunner.initialize_kv_cache(_runner(cfg), kv_cache_config)


# endregion


def test_mamba_models_must_implement_zero_kv_cache_blocks():
    """vLLM asks to zero new attention blocks whenever Mamba layers exist."""
    cfg = _vllm_config()
    kv_cache_config = _kv_cache_config(cfg, _layer_specs(), 256)

    with pytest.raises(ValueError, match="zero_kv_cache_blocks"):
        TTModelRunner.initialize_kv_cache(
            _runner(cfg, _AllocatingModel()), kv_cache_config
        )


# region State blocks, page table and zeroing


def test_state_blocks_match_upstream_selection():
    """``step_state`` picks the block upstream ``mamba_get_block_table_tensor``
    hands the GDN kernel in mode none, and block 0 for padding rows."""
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

    from vllm_tt_plugin.mamba_state import MambaCacheGroups

    spec = _layer_specs()["model.layers.0.linear_attn"]
    tables = [
        torch.arange(1, 4 * 24 + 1, dtype=torch.int32).reshape(4, 24) + 100 * g
        for g in range(2)
    ]
    groups = MambaCacheGroups(
        group_indices=(0, 1),
        layer_names=(("model.layers.0.linear_attn",), ("model.layers.1.linear_attn",)),
        attention_group_index=2,
    )

    state = groups.step_state(tables, ["a", "b", "c"], num_rows=6)

    seq_lens = torch.tensor([1, 1024, 20_000, 0])
    for g, group in enumerate(state.groups):
        upstream = mamba_get_block_table_tensor(tables[g], seq_lens, spec, "none")
        assert group.windows.dtype == torch.int32
        assert group.windows.shape == (6, 1)
        assert group.windows[:3, 0].tolist() == upstream[:3, 0].tolist()
        assert group.windows[3:, 0].tolist() == [0, 0, 0]
    # A scheduled row without a state block is a bookkeeping fault.
    tables[1][2, 0] = 0
    with pytest.raises(RuntimeError, match=r"requests \['c'\] have no state block"):
        groups.step_state(tables, ["a", "b", "c"], num_rows=6)


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


def _group_state(model_input, layer: int):
    """The ``MambaGroupState`` whose layers include GDN layer ``layer``."""
    name = f"model.layers.{layer}.linear_attn"
    [group] = [g for g in model_input.mamba_state.groups if name in g.layer_names]
    return group


def _state_ids(model_input, layer: int) -> list[int]:
    return _group_state(model_input, layer).windows[:, 0].tolist()


def _zeroed(model) -> list[list[int]]:
    return [
        kwargs["block_ids"]
        for name, kwargs in model.calls
        if name == "zero_kv_cache_blocks"
    ]


def test_prefill_and_decode_get_each_rows_state_block_per_group():
    """Requests A and B prefill, then decode.

    Every GDN layer's group gets the block of that group for each row, and the
    padding rows get the null block 0. vLLM lists the new attention blocks in
    ``new_block_ids_to_zero``, and the runner hands the model the whole list
    before the step.
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

    assert model.calls == [
        ("zero_kv_cache_blocks", {"kv_cache": ["kv"], "block_ids": [20, 21, 40]})
    ]
    groups = runner.kv_cache_config.kv_cache_groups
    assert [g.layer_names for g in prefill.mamba_state.groups] == [
        tuple(groups[i].layer_names) for i in (0, 1, 2)
    ]
    for layer in range(NUM_LAYERS):
        if _is_attention(layer):
            continue
        group = layer % 4
        assert _group_state(prefill, layer).windows.shape == (MAX_BATCH, 1)
        ids = _state_ids(prefill, layer)
        assert ids[:2] == [blocks_a[group][0], blocks_b[group][0]]
        assert set(ids[2:]) == {0}

    runner.submit_prefill(prefill, [2])
    name, kwargs = model.calls[-1]
    assert name == "prefill_forward"
    assert kwargs["mamba_state"] is prefill.mamba_state

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
    assert kwargs["mamba_state"] is decode.mamba_state
    assert _zeroed(model) == [[20, 21, 40]]


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


def test_mamba_model_refuses_block_copies():
    """vLLM queues block copies only for prefix-cache hits, which mode none
    never has; a copy that arrives anyway fails instead of being dropped."""
    cfg = _vllm_config()
    runner = _live_runner(cfg, _layer_specs(), _RecordingModel())

    with pytest.raises(RuntimeError, match=r"block copies \(src, dst\)"):
        runner.build_model_input(
            _step(
                {"a": 100},
                new=[_new_request("a", 100, ([10], [11], [12], [20]))],
                copies=[KVCacheBlockCopy(src_block_id=5, dst_block_id=6)],
            ),
            None,
        )


def test_attention_only_model_without_hooks_is_unchanged():
    """A model with no MambaSpec groups gets no zeroing and no ``mamba_state``."""
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
    runner.submit_prefill(model_input, [1])

    assert model_input.mamba_state is None
    [(name, kwargs)] = model.calls
    assert name == "prefill_forward"
    assert "mamba_state" not in kwargs


# endregion
