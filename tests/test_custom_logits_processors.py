# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Custom logits processors run through vLLM's V2 interface on the host sampler.

A V2 processor keeps per-request state by request slot and reads a
slot-indexed token history. ``CustomLogitsProcessors`` gives each request a
slot and maps persistent-batch rows to slots, so these tests move requests
between rows, commit output tokens, leave lane slots empty, and check what the
processor sees and changes.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.input_batch import InputBatch, TTLaneInputBatch
from vllm_tt_plugin.request_state import CachedRequestState
from vllm_tt_plugin.sample.logits_processor import build_logitsprocs
from vllm_tt_plugin.sample.metadata import SamplingMetadata
from vllm_tt_plugin.sample.sampler import Sampler

VOCAB = 16
BLOCK = 16
MAX_MODEL_LEN = 64
BOOST = 100.0


class BoostFromParams(LogitsProcessor):
    """Adds BOOST to the token named by ``extra_args["boost"]``, per slot."""

    def __init__(self, vllm_config, req_states):
        self.boost = torch.full((req_states.max_num_reqs,), -1, dtype=torch.long)

    def add_request(self, req_idx, sampling_params):
        token = (sampling_params.extra_args or {}).get("boost")
        self.boost[req_idx] = -1 if token is None else token
        return token is not None

    def apply(self, logits, ctx):
        tokens = self.boost[ctx.expanded_idx_mapping.long()]
        rows = torch.nonzero(tokens >= 0).flatten()
        logits[rows, tokens[rows]] += BOOST
        return logits


class Recorder(LogitsProcessor):
    """Records each call's context, history and logits."""

    def __init__(self, vllm_config, req_states):
        self.req_states = req_states
        self.calls = []

    def apply(self, logits, ctx):
        slots = ctx.idx_mapping_np
        total = self.req_states.total_len.np[slots]
        self.calls.append(
            SimpleNamespace(
                slots=slots.tolist(),
                history=[
                    self.req_states.all_token_ids.np[slot, :n].tolist()
                    for slot, n in zip(slots, total, strict=True)
                ],
                prompt_len=self.req_states.prompt_len.np[slots].tolist(),
                prefill_len=self.req_states.prefill_len.np[slots].tolist(),
                input_ids=ctx.input_ids.tolist(),
                pos=ctx.pos.tolist(),
                seq_lens=ctx.seq_lens_upper_bound_np.tolist(),
                logits=logits.clone(),
            )
        )
        return logits


def _config(max_num_seqs):
    return SimpleNamespace(
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        model_config=SimpleNamespace(
            max_model_len=MAX_MODEL_LEN, get_vocab_size=lambda: VOCAB
        ),
    )


def _batch(max_num_reqs=4, processors=(BoostFromParams, Recorder)):
    return InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=MAX_MODEL_LEN,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
        logitsprocs=build_logitsprocs(
            _config(max_num_reqs), custom_logitsprocs=list(processors)
        ),
    )


def _request(req_id, prompt, boost=None, **params):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=None,
        sampling_params=SamplingParams(
            extra_args=None if boost is None else {"boost": boost}, **params
        ),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=len(prompt),
        output_token_ids=[],
    )


def _processors(batch):
    boost, recorder = batch.sampling.logitsprocs.custom.processors
    return boost, recorder


def test_processor_state_follows_its_request_when_condense_moves_it():
    batch = _batch()
    batch.add_request(_request("a", [1, 2], boost=5))
    batch.add_request(_request("b", [3], boost=7))
    batch.refresh_logitsprocs()
    _, recorder = _processors(batch)
    batch.sampling.logitsprocs.custom.apply(torch.zeros(2, VOCAB))
    slot_b = recorder.calls[-1].slots[1]

    freed = batch.remove_request("a")
    batch.condense([freed])
    batch.refresh_logitsprocs()
    assert batch.req_ids == ["b"]

    logits = batch.sampling.logitsprocs.custom.apply(torch.zeros(1, VOCAB))

    assert recorder.calls[-1].slots == [slot_b]
    assert int(logits[0].argmax()) == 7
    assert float(logits[0, 7]) == BOOST


def test_reused_row_gets_a_fresh_slot_without_the_previous_requests_state():
    batch = _batch()
    batch.add_request(_request("old", [1], boost=5))
    batch.refresh_logitsprocs()
    freed = batch.remove_request("old")
    batch.add_request(_request("new", [2]), freed)
    batch.refresh_logitsprocs()

    logits = batch.sampling.logitsprocs.custom.apply(torch.zeros(1, VOCAB))

    assert torch.count_nonzero(logits) == 0


def test_processor_reads_prompt_and_committed_output_history():
    batch = _batch()
    request = _request("a", [1, 2, 3])
    request.output_token_ids.extend([9])
    batch.add_request(request)
    batch.refresh_logitsprocs()
    _, recorder = _processors(batch)

    # The runner commits tokens by extending the request's output list.
    request.output_token_ids.extend([10, 11])
    batch.sampling.logitsprocs.custom.apply(torch.zeros(1, VOCAB))

    call = recorder.calls[-1]
    assert call.history == [[1, 2, 3, 9, 10, 11]]
    assert call.prompt_len == [3]
    assert call.prefill_len == [4]
    assert call.pos == [5]
    assert call.input_ids == [11]
    assert call.seq_lens == [6]


def test_lane_gap_rows_are_not_shown_to_the_processors():
    num_lanes, per_lane = 2, 4
    batch = TTLaneInputBatch(
        num_lanes=num_lanes,
        per_lane=per_lane,
        max_model_len=MAX_MODEL_LEN,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
        logitsprocs=build_logitsprocs(
            _config(num_lanes * per_lane),
            custom_logitsprocs=[BoostFromParams, Recorder],
        ),
    )
    batch.add_request_to_row(_request("a", [1], boost=5), 6)
    batch.refresh_logitsprocs()
    _, recorder = _processors(batch)

    logits = batch.sampling.logitsprocs.custom.apply(torch.zeros(8, VOCAB))

    assert len(recorder.calls[-1].slots) == 1
    assert float(logits[6, 5]) == BOOST
    others = torch.cat([logits[:6], logits[7:]])
    assert torch.count_nonzero(others) == 0


def test_sampler_applies_custom_processors_after_penalties_before_temperature():
    """Token 3 gets logit_bias 5.0, then repetition penalty 2.0 halves it.

    The processor sees 2.5: after the bias and the penalty, and before the
    temperature of 2.0 would make it 1.25.
    """
    batch = _batch(processors=(Recorder,))
    batch.add_request(
        _request(
            "a",
            [3],
            logit_bias={3: 5.0},
            repetition_penalty=2.0,
            temperature=2.0,
        )
    )
    batch.refresh_logitsprocs()
    (recorder,) = batch.sampling.logitsprocs.custom.processors
    metadata = SamplingMetadata(
        temperature=torch.tensor([2.0]),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=False,
        prompt_token_ids=torch.tensor([[3]], dtype=torch.int64),
        frequency_penalties=torch.tensor([0.0]),
        presence_penalties=torch.tensor([0.0]),
        repetition_penalties=torch.tensor([2.0]),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=batch.sampling.logitsprocs,
    )

    Sampler()(torch.zeros(1, VOCAB), metadata)

    assert float(recorder.calls[-1].logits[0, 3]) == pytest.approx(2.5)


def test_processor_not_needed_by_any_row_is_not_applied():
    batch = _batch(processors=(BoostFromParams,))
    batch.add_request(_request("a", [1]))
    batch.refresh_logitsprocs()
    custom = batch.sampling.logitsprocs.custom
    calls = []
    custom.processors[0].apply = lambda logits, ctx: calls.append(ctx) or logits

    custom.apply(torch.zeros(1, VOCAB))

    assert calls == []


def test_non_v2_processor_class_is_rejected_when_the_runner_builds_processors():
    class V1Style:
        def update_state(self, batch_update):
            pass

    with pytest.raises(ValueError, match="V1Style is not a subclass"):
        build_logitsprocs(_config(4), custom_logitsprocs=[V1Style])


def test_a_tt_engine_config_accepts_v2_custom_logits_processors():
    """``TTPlatform.check_and_update_config`` runs inside
    ``VllmConfig.__post_init__`` and lets V2-interface processors through."""
    from vllm.config import ModelConfig, VllmConfig

    from tests.test_block_scheduler import LOCAL_MODEL_CONFIG, _stub_model_resolution

    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        logits_processors=[BoostFromParams],
    )
    with _stub_model_resolution():
        config = VllmConfig(model_config=model_config)

    assert config.model_config.logits_processors == [BoostFromParams]


def test_installed_entry_point_processor_loads_without_logits_processors_flag(
    monkeypatch,
):
    from vllm.v1.worker.gpu.sample.logits_processor import loader

    from vllm_tt_plugin.sample.logits_processor import custom

    entry_point = SimpleNamespace(
        name="boost", value="tests:BoostFromParams", load=lambda: BoostFromParams
    )
    monkeypatch.setattr(custom, "entry_points", lambda group: [entry_point])
    monkeypatch.setattr(loader, "entry_points", lambda group: [entry_point])

    logitsprocs = build_logitsprocs(_config(4))

    assert [type(p) for p in logitsprocs.custom.processors] == [BoostFromParams]


def test_no_custom_processors_leaves_custom_unset(monkeypatch):
    from vllm_tt_plugin.sample.logits_processor import custom

    monkeypatch.setattr(custom, "entry_points", lambda group: [])

    assert build_logitsprocs(_config(4)).custom is None


def test_custom_processors_are_refused_with_speculative_decoding():
    config = _config(4)
    config.speculative_config = object()

    with pytest.raises(ValueError, match="speculative decoding"):
        build_logitsprocs(config, custom_logitsprocs=[BoostFromParams])
