# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A synchronous runner over ``SampledTarget``, shared by the sampled spec tests.

``_runner`` binds ``TTModelRunner``'s own methods to a stand-in over the real
``InputBatch``, so every step goes through ``_prepare_model_inputs``,
``submit_decode`` and ``_finish_front_packed_sync``.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.spec_decode import VerifyOutput

from .sampled_target import (
    SAMPLED_VOCAB_SIZE,
    SampledTarget,
    base_token,
    support,
)

BLOCK_SIZE = 16
MAX_MODEL_LEN = 512
MAX_NUM_REQS = 4
PROMPT_LEN = 4
DRAFT_LEN = 3


def _runner(model: SampledTarget, num_speculative_tokens: int = DRAFT_LEN):
    """A synchronous runner over the real InputBatch, speculating or not."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=SAMPLED_VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    plan = type(model).spec_plan(None, MAX_NUM_REQS, max(num_speculative_tokens, 1))
    runner = SimpleNamespace(
        model=model,
        input_batch=batch,
        vocab_size=SAMPLED_VOCAB_SIZE,
        requests={},
        kv_caches=object(),
        trace_mode="decode_only",
        request_specific_rope=False,
        supports_stage_outputs=False,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        _num_speculative_tokens=num_speculative_tokens,
        _spec_accept_modes=plan.accept_modes if num_speculative_tokens else (),
        async_decode_scheduling=False,
        # The drafts come from each test through the scheduler output, so a
        # case can make them right or wrong exactly where it means to.
        _spec_method=None,
        _spec_supports_narrow_decode=plan.supports_narrow_decode,
        _spec_drafts_from_model=False,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        _ngram_proposer=None,
        vllm_config=SimpleNamespace(
            speculative_config=None,
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
            scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_REQS),
        ),
        tt_per_lane_max_num_seqs=MAX_NUM_REQS,
        tt_data_parallel_size=1,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        supports_stage_inputs=False,
        model_config=SimpleNamespace(
            is_multimodal_model=False, max_model_len=MAX_MODEL_LEN
        ),
        check_perform_device_sampling=lambda **_: False,
        # The real sampler: the reference arm's tokens have to come from the
        # ordinary host-sampling tail.
        host_sampler=Sampler(),
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
    )
    for name, member in vars(TTModelRunner).items():
        if hasattr(runner, name):
            continue
        if isinstance(member, staticmethod):
            setattr(runner, name, member.__func__)
        elif isinstance(member, classmethod):
            setattr(runner, name, member.__func__.__get__(TTModelRunner))
        elif inspect.isfunction(member):
            setattr(runner, name, member.__get__(runner))
    return runner


def _add_request(runner, req_id: str, params: SamplingParams, first_token: int = 1):
    """A fully prefilled request, with its own generator if it is seeded."""
    generator = None
    if params.sampling_type == SamplingType.RANDOM_SEED:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(params.seed)
    request = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[
            (first_token + index) % SAMPLED_VOCAB_SIZE for index in range(PROMPT_LEN)
        ],
        mm_features=None,
        sampling_params=params,
        generator=generator,
        block_ids=([0],),
        num_computed_tokens=PROMPT_LEN,
        output_token_ids=[],
    )
    runner.requests[req_id] = request
    runner.input_batch.add_request(request)


def _build(runner, *req_ids: str, drafts: dict[str, list[int]] | None = None):
    for req_id in req_ids:
        row = runner.input_batch.req_id_to_index[req_id]
        runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[
            row
        ]
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_spec_decode_tokens = dict(drafts or {})
    scheduler_output.num_scheduled_tokens = dict.fromkeys(req_ids, 1)
    scheduler_output.total_num_scheduled_tokens = len(req_ids)
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in req_ids],
        all_token_ids={},
        new_block_ids=[None for _ in req_ids],
        num_computed_tokens=[
            int(runner.input_batch.num_tokens[runner.input_batch.req_id_to_index[r]])
            for r in req_ids
        ],
        num_output_tokens=[0 for _ in req_ids],
    )
    return TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)


def _step(runner, *req_ids: str, drafts: dict[str, list[int]] | None = None):
    """One whole synchronous decode step; returns each request's committed ids."""
    model_input = _build(runner, *req_ids, drafts=drafts)
    submission = TTAsyncDecodeController(runner).submit_decode(
        model_input, read_from_device=True
    )
    fwd = _SyncForward(
        tt_out=submission.tt_out,
        tt_log_probs=None,
        sampling_params=model_input.tt_sampling_params,
        model_input=model_input,
        batch_size_per_dp=[len(req_ids)],
        perform_device_sampling=model_input.perform_device_sampling,
        is_decode=True,
        spec_hidden=submission.spec_hidden,
    )
    output = runner._finish_front_packed_sync(None, fwd=fwd)
    return {
        req_id: list(output.sampled_token_ids[output.req_id_to_index[req_id]])
        for req_id in req_ids
    }


def _tail(runner, req_id: str) -> tuple[int, int]:
    row = runner.input_batch.req_id_to_index[req_id]
    length = int(runner.input_batch.num_tokens[row])
    return int(runner.input_batch.token_ids_cpu[row, length - 1]), length - 1


def _output(runner, req_id: str) -> list[int]:
    return list(runner.requests[req_id].output_token_ids)


def _base_chain(token: int, position: int, count: int) -> list[int]:
    """The target's argmax continuation, which is what a greedy request emits."""
    chain = []
    for _ in range(count):
        token = base_token(token, position)
        position += 1
        chain.append(token)
    return chain


def _argmax_drafts(runner, req_id: str) -> list[int]:
    return _base_chain(*_tail(runner, req_id), DRAFT_LEN)


def _run_plain(params: SamplingParams, count: int, first_token: int = 1) -> list[int]:
    """The reference arm: a runner that does not speculate, one request."""
    runner = _runner(SampledTarget(), num_speculative_tokens=0)
    _add_request(runner, "r", params, first_token)
    while len(_output(runner, "r")) < count:
        _step(runner, "r")
    return _output(runner, "r")[:count]


def _assert_in_support(prompt_tail: tuple[int, int], tokens: list[int]) -> list[int]:
    """Every token is one the target allows after its predecessor; returns ranks."""
    token, position = prompt_tail
    ranks = []
    for index, emitted in enumerate(tokens):
        allowed = support(token, position)
        assert emitted in allowed, (
            f"token {index} ({emitted}) has probability zero after {token} at "
            f"position {position}; the target allows {allowed}"
        )
        ranks.append(allowed.index(emitted))
        token, position = emitted, position + 1
    return ranks


GREEDY = SamplingParams(temperature=0.0, ignore_eos=True)


class _DraftingTarget(SampledTarget):
    """``SampledTarget`` with its own drafter: the rule's argmax chain.

    Each verify hands out a fresh hidden handle that the next proposal must
    receive back.
    """

    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": ["device_propose", "hidden_feed"],
        "spec_hidden_handoff": ["roundtrip"],
    }

    def __init__(self) -> None:
        super().__init__()
        self.hidden = None
        self.proposals = 0

    def answer(self, spec_mode, logits):
        out = super().answer(spec_mode, logits)
        self.hidden = object()
        return VerifyOutput(
            spec_mode=out.spec_mode,
            argmax_ids=out.argmax_ids,
            logits=out.logits,
            hidden=self.hidden,
        )

    def propose_draft_tokens(
        self,
        num_drafts,
        committed_tokens,
        committed_positions,
        accepted_counts,
        hidden=None,
    ):
        from vllm_tt_plugin.spec_decode import DraftOutput

        assert hidden is self.hidden, "the proposal got another step's hidden handle"
        self.proposals += 1
        index = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        token = committed_tokens.to(torch.int64).gather(1, index)
        position = committed_positions.to(torch.int64).gather(1, index)
        columns = []
        for _ in range(num_drafts):
            token = (token * 31 + position * 7 + 11) % SAMPLED_VOCAB_SIZE
            position = position + 1
            columns.append(token)
        return DraftOutput(draft_token_ids=torch.cat(columns, dim=1).to(torch.int32))


def _drafting_runner(model):
    runner = _runner(model)
    runner._spec_drafts_from_model = True
    runner._spec_method = "custom_class"
    return runner
