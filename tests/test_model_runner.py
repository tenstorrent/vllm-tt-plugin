# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for ``TTModelRunner`` (non-lane path)."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.worker import TTWorker

# region Constants
VOCAB_SIZE = 64
BLOCK_SIZE = 16
MAX_MODEL_LEN = 32
MAX_NUM_SEQS = MAX_NUM_REQS = 4
DP_SIZE = 1
SAMPLED_TOKEN_ID = 42
# endregion Constants

# region Test helpers


def _batch_with_one_request(
    prompt_len: int,
    output_len: int,
    num_computed_tokens: int,
    sampling_params: SamplingParams | None = None,
) -> tuple[InputBatch, CachedRequestState]:
    """Creates a batch with a single request, for testing purposes."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    request = CachedRequestState(
        req_id="r",
        prompt_token_ids=list(range(prompt_len)),
        mm_features=None,
        sampling_params=(
            sampling_params
            if sampling_params is not None
            else SamplingParams(temperature=0.0)
        ),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=num_computed_tokens,
        output_token_ids=list(range(prompt_len, prompt_len + output_len)),
    )
    batch.add_request(request)
    return batch, request


def _fake_runner(batch: InputBatch, request: CachedRequestState) -> SimpleNamespace:
    """Creates a fake runner with the given batch and request, for testing purposes."""
    runner = SimpleNamespace(
        input_batch=batch,
        requests={"r": request},
        _num_speculative_tokens=0,
        _output_tokens_per_step=1,
        _is_adaptive_block_output=False,
        tt_per_lane_max_num_seqs=MAX_NUM_SEQS,
        tt_data_parallel_size=DP_SIZE,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        model_config=SimpleNamespace(is_multimodal_model=False),
        check_perform_device_sampling=lambda **_: False,
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        _sampling_params_for_padded_decode=lambda params, req_indices, n: params,
        _decode_layout_changed_since_last_decode=False,
        _build_host_generators=TTModelRunner._build_host_generators,
    )
    # The output-commit methods resolve the step's committed width through
    # _tt_committed_width (reads _is_adaptive_block_output + _output_tokens_per_step).
    runner._tt_committed_width = lambda toks: TTModelRunner._tt_committed_width(
        runner, toks
    )
    return runner


def test_hybrid_kv_cache_uses_v029_tensor_layers():
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=1024,
                layers=["model.layers.0.self_attn", "model.layers.1.self_attn"],
                layer_stride=512,
                block_stride=256,
            ),
            KVCacheTensor(
                size=1024,
                layers=["model.layers.2.self_attn"],
                layer_stride=512,
                block_stride=256,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["model.layers.0.self_attn", "model.layers.1.self_attn"], spec
            ),
            KVCacheGroupSpec(["model.layers.2.self_attn"], spec),
        ],
    )
    runner = SimpleNamespace(
        num_devices=1,
        tt_data_parallel_size=1,
    )
    runner._kv_cache_shape = lambda spec, num_blocks: TTModelRunner._kv_cache_shape(
        runner, spec, num_blocks
    )

    per_layer = TTModelRunner._build_per_layer_specs(runner, config, 3)

    # Groups alias from byte 0: the i-th layer of each group shares buffer i.
    assert [entry[2] for entry in per_layer] == [0, 1, 0]


def test_kv_cache_layout_resolves_to_lbnhc_for_mixed_specs():
    specs = [
        FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=num_kv_heads,
            head_size=8,
            dtype=torch.float32,
        )
        for num_kv_heads in (1, 4)
    ]
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_layout=None),
        kv_transfer_config=None,
    )
    supported = TTWorker.get_supported_kv_cache_layouts(TTWorker.__new__(TTWorker))

    layout = resolve_kv_cache_layout(vllm_config, [supported], specs)

    assert supported == ["LBNHC"]
    assert layout.name == "LBNHC"


def test_hybrid_tables_pad_once_per_native_group():
    runner = SimpleNamespace(
        _layer_to_group_idx=[1, 0, 1, 0],
        tt_max_batch_size=4,
        max_num_blocks_per_req=8,
    )
    source = [torch.tensor([[2, 3]]), torch.tensor([[7, 9]])]
    tables = TTModelRunner._block_tables_per_layer(runner, source)
    assert tables[0] is tables[2]
    assert tables[1] is tables[3]
    assert tables[0] is not tables[1]
    assert tables[0].shape == (4, 8)
    assert tables[0][0].tolist() == [7, 9, 0, 0, 0, 0, 0, 0]
    assert tables[1][0].tolist() == [2, 3, 0, 0, 0, 0, 0, 0]
    source[1][0, 1] = 11
    changed = TTModelRunner._block_tables_per_layer(runner, source)
    assert changed[0][0, 1] == 11
    assert tables[0][0, 1] == 9


@pytest.mark.parametrize("groups", [None, [1, 0, 1]])
def test_adapter_group_configuration_precedes_cache_allocation(groups):
    calls = []
    model = SimpleNamespace(
        configure_kv_cache_groups=lambda mapping: calls.append(("groups", mapping)),
        allocate_kv_cache_per_layer=lambda specs: calls.append(("allocate", specs)),
    )
    runner = SimpleNamespace(
        model=model,
        model_config=SimpleNamespace(get_num_layers_by_block_type=lambda *args: 3),
        parallel_config=None,
        _layer_to_group_idx=groups,
        _build_per_layer_specs=lambda *args: "specs",
    )
    TTModelRunner._allocate_kv_caches(runner, None)
    assert calls == [("groups", tuple(groups or [0, 0, 0])), ("allocate", "specs")]


def _prepare(runner, *rows):
    """Run _prepare_model_inputs for cached rows of
    (req_id, num_scheduled, num_computed, num_output)."""
    out = SchedulerOutput.make_empty()
    out.num_scheduled_tokens = {r: s for r, s, _, _ in rows}
    out.total_num_scheduled_tokens = sum(s for _, s, _, _ in rows)
    out.scheduled_cached_reqs = CachedRequestData(
        req_ids=[r for r, *_ in rows],
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in rows],
        all_token_ids={},
        new_block_ids=[None for _ in rows],
        num_computed_tokens=[c for _, _, c, _ in rows],
        num_output_tokens=[o for *_, o in rows],
    )
    return TTModelRunner._prepare_model_inputs(runner, out, None)


# endregion Test helpers

# region Prefill classification


@pytest.mark.parametrize(
    "num_computed_tokens, intermediate_prefill_mask",
    [(0, True), (2, True), (5, True), (6, False)],
)
def test_cached_chunked_prefill_classification(
    num_computed_tokens: int,
    intermediate_prefill_mask: bool,
):
    """Classify a cached chunked-prefill continuation and its boundary."""
    batch, request = _batch_with_one_request(
        prompt_len=8,
        output_len=0,
        num_computed_tokens=num_computed_tokens,
    )
    runner = _fake_runner(batch, request)
    num_scheduled_tokens = 2

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": num_scheduled_tokens}
    scheduler_output.total_num_scheduled_tokens = num_scheduled_tokens
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[num_computed_tokens],
        num_output_tokens=[0],
    )
    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    assert model_input.prompt_lens.tolist() == [
        num_computed_tokens + num_scheduled_tokens
    ]
    assert model_input.intermediate_prefill_mask.tolist() == [intermediate_prefill_mask]


def test_resumed_replay_past_prompt_length_remains_prefill():
    """Resumed replay past the prompt length remains prefill."""
    # 4-token prompt, 6 previously generated tokens (num_tokens=10). Resumed
    # from preemption and already replayed its first chunk up to position 5,
    # which is past the prompt but short of num_tokens: still mid-replay.
    num_computed_tokens = 5
    batch, request = _batch_with_one_request(
        prompt_len=4,
        output_len=6,
        num_computed_tokens=num_computed_tokens,
    )
    runner = _fake_runner(batch, request)
    num_scheduled_tokens = 3

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": num_scheduled_tokens}
    scheduler_output.total_num_scheduled_tokens = num_scheduled_tokens

    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[num_computed_tokens],
        num_output_tokens=[6],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    # A decode step would set `prompt_lens=None` and sample a token.
    # This step must still be a prefill chunk (position 5..8 of a 10-token replay).
    assert model_input.prompt_lens is not None
    assert model_input.prompt_lens.tolist() == [
        num_computed_tokens + num_scheduled_tokens
    ]
    assert model_input.intermediate_prefill_mask.tolist() == [True]


# endregion Prefill classification

# region Decode input construction


@pytest.mark.parametrize("output_len", [0, 1, 2, 5])
def test_completed_cached_request_builds_decode_input(output_len: int):
    """Classify a cached request that has computed everything but its last token."""
    prompt_len = 8
    num_tokens = prompt_len + output_len
    num_computed_tokens = max(prompt_len, num_tokens - 1)
    batch, request = _batch_with_one_request(
        prompt_len=prompt_len,
        output_len=output_len,
        num_computed_tokens=num_computed_tokens,
    )
    runner = _fake_runner(batch, request)
    num_scheduled_tokens = 1

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"r": num_scheduled_tokens}
    scheduler_output.total_num_scheduled_tokens = num_scheduled_tokens
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[num_computed_tokens],
        num_output_tokens=[output_len],
    )

    model_input = TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)

    assert model_input is not None
    assert model_input.prompt_lens is None
    assert model_input.input_positions.tolist() == [num_tokens - 1] + [-1] * (
        MAX_NUM_SEQS - 1
    )
    assert model_input.input_tokens.shape == (MAX_NUM_SEQS, 1)
    assert model_input.input_tokens[0, 0] == batch.token_ids_cpu[0, num_tokens - 1]


def test_structured_decode_build_captures_expected_grammar_request_ids():
    prompt_len = 8
    output_len = 1
    num_computed_tokens = prompt_len
    batch, request = _batch_with_one_request(
        prompt_len=prompt_len,
        output_len=output_len,
        num_computed_tokens=num_computed_tokens,
        sampling_params=SamplingParams(
            temperature=0.0,
            structured_outputs=StructuredOutputsParams(json_object=True),
        ),
    )
    runner = _fake_runner(batch, request)
    runner.check_perform_device_sampling = lambda **_: True

    model_input = _prepare(
        runner,
        ("r", 1, num_computed_tokens, output_len),
    )

    assert model_input.prompt_lens is None
    assert model_input.grammar_bitmask == [None]
    assert model_input.structured_output_req_ids == frozenset({"r"})
    assert model_input.defer_device_sampling is True


def test_final_one_token_prompt_chunk_stays_prefill():
    """Even a one-token prompt remainder is prefill work: the prefill path
    owns chunk bookkeeping (intermediate mask, vision rope state)."""
    batch, request = _batch_with_one_request(8, 0, num_computed_tokens=7)

    model_input = _prepare(_fake_runner(batch, request), ("r", 1, 7, 0))

    assert model_input.prompt_lens.tolist() == [8]
    assert model_input.intermediate_prefill_mask.tolist() == [False]


def test_mixed_prefill_batch_keeps_scheduled_decode_row():
    """A scheduled row must never be dropped from a prefill step: the output
    row count desyncs and the dropped token is never rescheduled."""
    batch, chunking = _batch_with_one_request(8, 0, num_computed_tokens=2)
    steady = CachedRequestState(
        req_id="r2",
        prompt_token_ids=list(range(8)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([1],),
        num_computed_tokens=8,
        output_token_ids=[100],
    )
    batch.add_request(steady)

    model_input = _prepare(
        _fake_runner(batch, chunking), ("r", 2, 2, 0), ("r2", 1, 8, 1)
    )

    # Chunk end 2+2=4 (mid-prompt) and the steady row's 8+1=9 both stay.
    assert sorted(model_input.prompt_lens.tolist()) == [4, 9]


@pytest.mark.parametrize("capability", [None, False, True])
@pytest.mark.parametrize("otherwise_eligible", [False, True])
def test_partial_prefill_device_sampling_requires_opt_in(
    capability, otherwise_eligible
):
    batch, intermediate = _batch_with_one_request(8, 0, num_computed_tokens=2)
    final = CachedRequestState(
        req_id="final",
        prompt_token_ids=list(range(8)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([1],),
        num_computed_tokens=6,
        output_token_ids=[],
    )
    batch.add_request(final)
    runner = _fake_runner(batch, intermediate)
    runner.requests["final"] = final
    runner.model = SimpleNamespace(
        model_capabilities=(
            {}
            if capability is None
            else {"supports_partial_prefill_device_sampling": capability}
        )
    )
    runner.check_perform_device_sampling = lambda **_: otherwise_eligible

    model_input = _prepare(runner, ("r", 2, 2, 0), ("final", 2, 6, 0))

    assert model_input.intermediate_prefill_mask.tolist() == [True, False]
    assert model_input.perform_device_sampling is (
        otherwise_eligible and capability is True
    )


@pytest.mark.parametrize("capability", [None, False, True])
@pytest.mark.parametrize("otherwise_eligible", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_partial_prefill_mask_forwarding_and_output_suppression(
    capability, otherwise_eligible, mixed
):
    batch, request = _batch_with_one_request(8, 0, num_computed_tokens=2)
    runner = _fake_runner(batch, request)
    rows = [("r", 2, 2, 0)]
    if mixed:
        final = CachedRequestState(
            req_id="final",
            prompt_token_ids=list(range(8)),
            mm_features=None,
            sampling_params=SamplingParams(temperature=0.0),
            generator=None,
            block_ids=([1],),
            num_computed_tokens=6,
            output_token_ids=[],
        )
        batch.add_request(final)
        runner.requests["final"] = final
        rows.append(("final", 2, 6, 0))
    submitted = []
    noted = []
    applied = []
    placeholder = torch.full((len(rows), 1), 42, dtype=torch.int32)

    def prefill_forward(**kwargs):
        submitted.append(kwargs)
        return placeholder

    runner.model = SimpleNamespace(
        model_capabilities=(
            {}
            if capability is None
            else {"supports_partial_prefill_device_sampling": capability}
        ),
        prefill_forward=prefill_forward,
    )
    runner.check_perform_device_sampling = lambda **_: otherwise_eligible
    runner.kv_caches = object()
    runner.trace_mode = "all"
    runner.request_specific_rope = False
    runner.async_decode = SimpleNamespace(
        note_prefill_submitted=lambda: noted.append(True)
    )
    runner._apply_sampled_tokens_to_state = lambda *a, **kw: applied.append((a, kw))
    model_input = _prepare(runner, *rows)

    tt_out = TTModelRunner.submit_prefill(runner, model_input, [len(rows)])
    output = TTModelRunner._build_chunked_prefill_output(
        runner,
        req_ids=[row[0] for row in rows],
        sampled_token_ids=tt_out,
        logprobs=None,
        intermediate_mask=model_input.intermediate_prefill_mask.numpy(),
    )

    assert len(submitted) == 1
    assert submitted[0]["kv_cache"] is runner.kv_caches
    if capability is True:
        assert (
            submitted[0]["intermediate_prefill_mask"]
            is model_input.intermediate_prefill_mask
        )
    else:
        assert "intermediate_prefill_mask" not in submitted[0]
    assert ("sampling_params" in submitted[0]) is (
        otherwise_eligible and capability is True
    )
    assert noted == [True]
    assert output.sampled_token_ids == ([[], [42]] if mixed else [[]])
    if mixed:
        assert len(applied) == 1
        assert applied[0][0][0].tolist() == [[42]]
        assert applied[0][1] == {"req_ids": ["final"]}
    else:
        assert applied == []


def test_prefill_schedule_hook_reports_exact_mixed_work_before_forward():
    batch, intermediate = _batch_with_one_request(8, 0, num_computed_tokens=2)
    runner = _fake_runner(batch, intermediate)
    rows = [("r", 2, 2, 0)]
    for name, computed, outputs, scheduled in (
        ("decode", 8, [10], 1),
        ("replay", 8, [10, 11, 12], 2),
        ("final", 7, [], 1),
    ):
        request = CachedRequestState(
            req_id=name,
            prompt_token_ids=list(range(8)),
            mm_features=None,
            sampling_params=SamplingParams(temperature=0.0),
            generator=None,
            block_ids=([len(rows)],),
            num_computed_tokens=computed,
            output_token_ids=outputs,
        )
        batch.add_request(request)
        runner.requests[name] = request
        rows.append((name, scheduled, computed, len(outputs)))
    model_input = _prepare(runner, *rows)
    calls = []

    def record_schedule(actual_input, active_decode_req_ids):
        assert actual_input is model_input
        assert actual_input.row_req_ids == ["r", "decode", "replay", "final"]
        assert actual_input.input_positions.tolist() == [2, 8, 8, 7]
        assert actual_input.prompt_lens.tolist() == [4, 9, 10, 8]
        assert active_decode_req_ids == ["decode"]
        calls.append("schedule")

    runner.model = SimpleNamespace(
        record_vllm_prefill_schedule=record_schedule,
        prefill_forward=lambda **kwargs: calls.append("forward"),
    )
    runner.kv_caches = object()
    runner.trace_mode = "all"
    runner.request_specific_rope = False
    runner.async_decode = SimpleNamespace(
        note_prefill_submitted=lambda: calls.append("submitted")
    )

    TTModelRunner.submit_prefill(runner, model_input, [4])

    assert calls == ["schedule", "forward", "submitted"]


def test_prefill_schedule_hook_retains_decode_hidden_by_prefill_only_batch():
    batch, anchor = _batch_with_one_request(8, 0, num_computed_tokens=0)
    runner = _fake_runner(batch, anchor)
    seen = []
    runner.model = SimpleNamespace(
        record_vllm_prefill_schedule=lambda inputs, active: seen.append(active),
        prefill_forward=lambda **kwargs: None,
    )
    runner.kv_caches = object()
    runner.trace_mode = "all"
    runner.request_specific_rope = False
    runner.async_decode = SimpleNamespace(note_prefill_submitted=lambda: None)
    runner.encoder_cache = {}
    runner._req_state_slot = {"r": 0}
    runner._release_dead_state_slots = lambda output: (
        TTModelRunner._release_dead_state_slots(runner, output)
    )

    # Readiness comes from this real successful final-prefill transition.
    initial = _prepare(runner, ("r", 8, 0, 0))
    TTModelRunner.submit_prefill(runner, initial, [1])
    assert seen == [[]]
    assert runner._diagnostic_decode_ready_req_ids == {"r"}

    # Cached computed is a pre-step cursor. Async output publication can leave
    # it more than one token behind, without turning a decode into prefill.
    anchor.num_computed_tokens = 9
    anchor.output_token_ids.extend([10, 11, 12])
    assert anchor.num_computed_tokens + 1 < anchor.num_tokens
    newcomer = CachedRequestState(
        req_id="new",
        prompt_token_ids=list(range(16)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0),
        generator=None,
        block_ids=([1],),
        num_computed_tokens=2,
        output_token_ids=[],
    )
    runner.requests["new"] = newcomer
    runner._req_state_slot["new"] = 1
    step = SchedulerOutput.make_empty()
    step.num_scheduled_tokens = {"new": 2}
    step.total_num_scheduled_tokens = 2
    step.scheduled_cached_reqs = CachedRequestData(
        req_ids=["new"],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[2],
        num_output_tokens=[0],
    )

    # Exercise the actual operation that hid the resident decode in serving.
    TTModelRunner._update_states(runner, step)
    assert "r" not in batch.req_id_to_index
    assert "r" in runner.requests and "r" in runner._req_state_slot
    inputs = TTModelRunner._prepare_model_inputs(runner, step, None)
    TTModelRunner.submit_prefill(runner, inputs, [1])
    assert seen[-1] == ["r"]
    assert runner._diagnostic_decode_ready_req_ids == {"r"}

    # Residency loss must invalidate the diagnostic readiness before a later
    # resume can reuse its slot for an intermediate history-replay chunk.
    preemption = SchedulerOutput.make_empty()
    preemption.preempted_req_ids = {"r"}
    TTModelRunner._release_dead_state_slots(runner, preemption)
    assert not runner._diagnostic_decode_ready_req_ids
    TTModelRunner.submit_prefill(runner, inputs, [1])
    assert seen[-1] == []


def test_steady_block_step_builds_decode_input():
    """One whole canvas outstanding is the block steady state; dispatching it
    as prompt work re-encodes the entire session per canvas."""
    batch, request = _batch_with_one_request(8, 16, num_computed_tokens=8)
    runner = _fake_runner(batch, request)
    runner._output_tokens_per_step = 16

    model_input = _prepare(runner, ("r", 16, 8, 16))

    assert model_input.prompt_lens is None


def test_block_resume_replay_past_one_canvas_remains_prefill():
    """More than one canvas outstanding is uncomputed replay history."""
    batch, request = _batch_with_one_request(8, 16, num_computed_tokens=8)
    runner = _fake_runner(batch, request)
    runner._output_tokens_per_step = 8

    model_input = _prepare(runner, ("r", 8, 8, 16))

    assert model_input.prompt_lens.tolist() == [16]


# endregion Decode input construction

# region Output state


def test_apply_sampled_token_updates_request_state():
    try:
        import torch
    except ImportError:
        pytest.xfail("torch is required for sampled-token tensor inputs")

    prompt_len = 8
    batch, request = _batch_with_one_request(
        prompt_len=prompt_len,
        output_len=0,
        num_computed_tokens=prompt_len,
    )
    runner = _fake_runner(batch, request)
    runner.model_config.max_model_len = MAX_MODEL_LEN
    runner._apply_sampled_tokens_to_state = lambda **kwargs: (
        TTModelRunner._apply_sampled_tokens_to_state(runner, **kwargs)
    )
    runner._build_runner_output = lambda **kwargs: TTModelRunner._build_runner_output(
        runner, **kwargs
    )

    output = TTModelRunner.apply_and_build_runner_output(
        runner,
        sampled_token_ids=torch.tensor([[SAMPLED_TOKEN_ID]], dtype=torch.int32),
    )

    assert batch.num_tokens[0] == prompt_len + 1
    assert batch.token_ids_cpu[0, prompt_len] == SAMPLED_TOKEN_ID
    assert request.output_token_ids == [SAMPLED_TOKEN_ID]
    assert output.req_ids == ["r"]
    assert output.sampled_token_ids == [[SAMPLED_TOKEN_ID]]


# endregion Output state


@pytest.mark.parametrize("declared", [None, True, False])
@pytest.mark.parametrize("has_penalties", [False, True])
def test_device_penalties_follow_model_capability(declared, has_penalties):
    capabilities = {} if declared is None else {"supports_device_penalties": declared}
    runner = SimpleNamespace(
        sample_on_device_mode="all",
        num_devices=4,
        tt_data_parallel_size=1,
        model=SimpleNamespace(model_capabilities=capabilities),
        model_config=SimpleNamespace(logits_processors=[]),
        input_batch=SimpleNamespace(
            no_penalties=not has_penalties,
            no_allowed_token_ids=True,
            max_num_logprobs=None,
            sampling=SimpleNamespace(
                bad_words_token_ids={}, has_active_logitsprocs=lambda: False
            ),
        ),
    )
    expected = not (has_penalties and declared is False)
    assert TTModelRunner.check_perform_device_sampling(runner, True, False) is expected
