# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Token-masked verifies (``supports_spec_device_mask``) through the runner.

A model declaring ``supports_spec_device_mask`` takes the verify of a step
whose rows carry a grammar, ``allowed_token_ids``, ``bad_words`` or a pending
``min_tokens`` as an ``argmax_ids`` verify with deferred sampling:
``decode_forward(defer_device_sampling=True)`` at ``execute_model``, then
``sample_deferred_decode`` at ``sample_tokens`` with the composed token mask,
and draws each column from the masked target. These tests drive whole
synchronous steps (``_prepare_model_inputs``, ``submit_decode``,
``_finish_front_packed_sync``) against ``MaskedTarget``, a ``SampledTarget``
that masks its own logits, and hold them to the same steps of a runner whose
model does not declare the capability, which verifies in ``logits`` mode and
walks the same controls on the host.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import _SyncForward
from vllm_tt_plugin.spec_admission import validate_spec_device_mask
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    PLACEHOLDER_TOKEN_ID,
    VerifyOutput,
)
from vllm_tt_plugin.structured_output import grammar_mask_stride

from . import sampled_sync_harness as sync_harness
from .sampled_reference import _pack
from .sampled_target import SAMPLED_VOCAB_SIZE, SampledTarget, rule_logits, support

DRAFT_LEN = sync_harness.DRAFT_LEN


class MaskedTarget(SampledTarget):
    """``SampledTarget`` serving token-masked verifies with deferred sampling.

    ``decode_forward(defer_device_sampling=True)`` keeps the block's rule
    logits as the payload; ``sample_deferred_decode`` masks them with the token
    mask and answers each column with the masked argmax for a greedy row and a
    draw from the masked distribution for a sampled one. It records every mask
    it was handed.
    """

    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": [],
        "supports_sample_on_device": True,
        "spec_ids_are_sampled": True,
        "supports_spec_device_mask": True,
    }

    def __init__(self) -> None:
        super().__init__()
        self.payload = None
        self.masks: list[torch.Tensor | None] = []
        self.num_valid: list[list[int]] = []
        self.generator = torch.Generator().manual_seed(0)

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        if not kwargs.pop("defer_device_sampling", False):
            return super().decode_forward(tokens, start_pos, spec_mode, **kwargs)
        if spec_mode != ACCEPT_MODE_ARGMAX_IDS or "sampling_params" in kwargs:
            raise ValueError("a deferred verify is argmax_ids, sampled later")
        if self.payload is not None:
            raise RuntimeError("a second deferred verify before the first sampled")
        self.verify_modes.append(spec_mode)
        self.payload = rule_logits(tokens, start_pos)
        return self.payload

    def sample_deferred_decode(
        self, payload, sampling_params=None, grammar_bitmask=None, num_valid_drafts=None
    ):
        if payload is not self.payload:
            raise ValueError("a stale payload")
        self.payload = None
        self.masks.append(None if grammar_bitmask is None else grammar_bitmask.clone())
        self.num_valid.append(num_valid_drafts.tolist())
        logits = payload
        if grammar_bitmask is not None:
            logits = logits.masked_fill(~_allowed(grammar_bitmask), float("-inf"))
        ids = logits.argmax(dim=-1)
        for row, temperature in enumerate(sampling_params.temperature):
            if row < ids.shape[0] and temperature > 0:
                probs = torch.softmax(logits[row] / temperature, dim=-1)
                ids[row] = torch.multinomial(probs, 1, generator=self.generator)[:, 0]
        return VerifyOutput(spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids.int())


def _allowed(mask: torch.Tensor) -> torch.Tensor:
    bits = torch.arange(32, dtype=torch.int32)
    allowed = ((mask.unsqueeze(-1) >> bits) & 1).bool().reshape(*mask.shape[:-1], -1)
    return allowed[..., :SAMPLED_VOCAB_SIZE]


# region A grammar the scheduler fills


def _grammar_allows(history: list[int], token: int) -> bool:
    """A grammar whose state is the output length: one residue mod 3 is out."""
    return token % 3 != len(history) % 3


def _validated(history: list[int], drafts: list[int]) -> list[int]:
    """``StructuredOutputGrammar.validate_tokens``: the longest valid prefix."""
    valid: list[int] = []
    for token in drafts:
        if not _grammar_allows(history + valid, token):
            break
        valid.append(token)
    return valid


def _grammar_output(runner, scheduled: dict[str, list[int]]) -> GrammarOutput | None:
    """``StructuredOutputManager.grammar_bitmask`` for the step's structured rows."""
    structured = [
        req_id
        for req_id in runner.input_batch.req_ids[: runner.input_batch.num_reqs]
        if runner.requests[req_id].sampling_params.structured_outputs is not None
    ]
    if not structured:
        return None
    stride = grammar_mask_stride(runner._num_speculative_tokens)
    blocks = []
    for req_id in structured:
        state = list(runner.requests[req_id].output_token_ids)
        rows = []
        for token in [*scheduled.get(req_id, []), None]:
            rows.append(
                torch.tensor(
                    [_grammar_allows(state, t) for t in range(SAMPLED_VOCAB_SIZE)]
                )
            )
            if token is not None:
                state.append(token)
        rows.extend(torch.zeros(SAMPLED_VOCAB_SIZE) for _ in range(stride - len(rows)))
        blocks.append(_pack(torch.stack(rows).bool()))
    return GrammarOutput(structured, torch.cat(blocks).numpy())


# endregion

# region Runners and steps


def _runner(model: SampledTarget):
    """A synchronous runner whose plain steps sample on the host and whose
    token-masked verifies sample on the device."""
    runner = sync_harness._runner(model)
    runner.input_batch = InputBatch(
        max_num_reqs=sync_harness.MAX_NUM_REQS,
        max_model_len=sync_harness.MAX_MODEL_LEN,
        max_num_batched_tokens=sync_harness.MAX_MODEL_LEN,
        vocab_size=SAMPLED_VOCAB_SIZE,
        block_sizes=[sync_harness.BLOCK_SIZE],
        logitsprocs=LogitsProcessors(
            [MinTokensLogitsProcessor(None, torch.device("cpu"), False)]
        ),
    )
    runner.structured_output_arange = torch.arange(0, 32)
    runner.sample_on_device_mode = "decode_only"
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner.check_perform_device_sampling = lambda token_masks=False, **_: token_masks
    runner.async_decode = TTAsyncDecodeController(runner)
    return runner


def _step(runner, *req_ids: str, drafts: dict[str, list[int]] | None = None):
    """One synchronous step with its grammar; the scheduler validated the drafts."""
    drafts = dict(drafts or {})
    for req_id in list(drafts):
        if runner.requests[req_id].sampling_params.structured_outputs is not None:
            drafts[req_id] = _validated(
                list(runner.requests[req_id].output_token_ids), drafts[req_id]
            )
    runner.input_batch.refresh_logitsprocs()
    model_input = sync_harness._build(runner, *req_ids, drafts=drafts)
    grammar = _grammar_output(runner, drafts)
    submission = runner.async_decode.submit_decode(
        model_input, read_from_device=not model_input.defer_device_sampling
    )
    fwd = _SyncForward(
        tt_out=submission.tt_out,
        tt_log_probs=None,
        sampling_params=model_input.tt_sampling_params,
        model_input=model_input,
        batch_size_per_dp=[len(req_ids)],
        perform_device_sampling=model_input.perform_device_sampling,
        is_decode=True,
        decode_submission=submission,
        spec_hidden=submission.spec_hidden,
    )
    runner._finish_front_packed_sync(grammar, fwd=fwd)
    return model_input


def _run(runner, req_id: str, count: int, drafts_for=None) -> list[int]:
    while len(sync_harness._output(runner, req_id)) < count:
        drafts = {} if drafts_for is None else {req_id: drafts_for(runner)}
        _step(runner, req_id, drafts=drafts)
    return sync_harness._output(runner, req_id)[:count]


def _controlled_greedy(params: SamplingParams, first_token: int, count: int):
    """The greedy output under every control, independently of either runner."""
    token = first_token + sync_harness.PROMPT_LEN - 1
    position = sync_harness.PROMPT_LEN - 1
    history: list[int] = []
    for _ in range(count):
        allowed = [t for t in support(token, position) if _allows(params, history, t)]
        token, position = allowed[0], position + 1
        history.append(token)
    return history


def _allows(params: SamplingParams, history: list[int], token: int) -> bool:
    if params.structured_outputs is not None and not _grammar_allows(history, token):
        return False
    if params.allowed_token_ids and token not in params.allowed_token_ids:
        return False
    for word in params.bad_words_token_ids or ():
        prefix = word[:-1]
        if token == word[-1] and history[len(history) - len(prefix) :] == prefix:
            return False
    pending = params.min_tokens > len(history)
    return not (pending and token in params.all_stop_token_ids)


def _params(case: str, **sampling) -> SamplingParams:
    chain = _controlled_greedy(SamplingParams(temperature=0.0), 1, 8)
    controls = {
        "structured": {"structured_outputs": StructuredOutputsParams(regex="[a-z]+")},
        "allowed": {"allowed_token_ids": list(range(0, SAMPLED_VOCAB_SIZE, 2))},
        "bad-words": {},
        "min-tokens": {"min_tokens": 6, "stop_token_ids": [chain[1], chain[4]]},
    }[case]
    params = SamplingParams(ignore_eos=True, max_tokens=64, **controls, **sampling)
    if case == "bad-words":
        # What ``update_from_tokenizer`` leaves; the test has no tokenizer.
        params._bad_words_token_ids = [[chain[2], chain[3]], [chain[5]]]
    return params


CASES = ("structured", "allowed", "bad-words", "min-tokens")

# endregion

# region The mode and the output


@pytest.mark.parametrize("drafts", ["right", "wrong"])
@pytest.mark.parametrize("case", CASES)
def test_a_greedy_row_with_a_token_control_verifies_argmax_ids_to_the_logits_output(
    case, drafts
):
    """Every verify of the row is a token-masked ``argmax_ids`` verify, and the
    output is the ``logits`` runner's, which equals the greedy chain under the
    controls. Right drafts are accepted, so every column's mask has to be that
    column's; wrong drafts are rejected, so the correction is the masked argmax.
    """
    params = _params(case, temperature=0.0)
    expected = _controlled_greedy(params, 1, 40)
    assert expected[:30] != _controlled_greedy(SamplingParams(), 1, 30), case

    def right(runner):
        done = len(sync_harness._output(runner, "c"))
        return expected[done : done + DRAFT_LEN]

    def wrong(runner):
        bent = right(runner)
        if len(bent) > 1:
            bent[1] = (bent[1] + 1) % SAMPLED_VOCAB_SIZE
        return bent

    drafts_for = right if drafts == "right" else wrong
    outputs = {}
    for name, model in (("masked", MaskedTarget()), ("logits", SampledTarget())):
        runner = _runner(model)
        sync_harness._add_request(runner, "c", params)
        outputs[name] = _run(runner, "c", 30, drafts_for)
        if name == "masked":
            masked = model
    assert outputs["masked"] == outputs["logits"] == expected[:30]
    assert masked.masks and masked.payload is None
    assert ACCEPT_MODE_LOGITS not in masked.verify_modes
    if case == "min-tokens":
        # Reached at token 6: plain argmax_ids verifies after it, no mask.
        assert len(masked.masks) < masked.verify_modes.count(ACCEPT_MODE_ARGMAX_IDS)
    else:
        assert len(masked.masks) == len(masked.verify_modes)


def test_the_mask_holds_each_column_s_grammar_and_static_controls():
    """One step, three rows: a grammar row (its scheduler row per column), a row
    with a bad word whose prefix is its first draft and a pending min_tokens
    (each column's own static controls), and an unconstrained neighbor."""
    model = MaskedTarget()
    runner = _runner(model)
    grammar = StructuredOutputsParams(regex="[a-z]+")
    sync_harness._add_request(runner, "g", SamplingParams(temperature=0.0), 21)
    sync_harness._add_request(
        runner, "j", SamplingParams(temperature=0.0, structured_outputs=grammar)
    )
    controls = SamplingParams(temperature=0.0, min_tokens=2, stop_token_ids=[9])
    controls._bad_words_token_ids = [[30, 13]]
    sync_harness._add_request(runner, "b", controls, first_token=7)
    drafts = {"g": [1, 2, 3], "j": [4, 5, 7], "b": [30, 31, 32]}
    _step(runner, "g", "j", "b", drafts=drafts)
    assert model.verify_modes == [ACCEPT_MODE_ARGMAX_IDS]
    allowed = _allowed(model.masks[-1])
    rows = runner.input_batch.req_id_to_index
    assert tuple(allowed.shape) == (sync_harness.MAX_NUM_REQS, DRAFT_LEN + 1, 64)
    assert bool(allowed[rows["g"]].all())
    valid = _validated([], drafts["j"])
    for column in range(len(valid) + 1):
        want = torch.tensor([_grammar_allows(valid[:column], t) for t in range(64)])
        assert torch.equal(allowed[rows["j"], column], want), column
    banned = [
        (~allowed[rows["b"], column]).nonzero().reshape(-1).tolist()
        for column in range(DRAFT_LEN + 1)
    ]
    # min_tokens 2 bans 9 at columns 0 and 1; draft 0 is the bad word's prefix.
    assert banned == [[9], [9, 13], [], []]


def test_a_row_the_mask_cannot_certify_keeps_its_verify_in_logits_mode():
    """Logprobs need the logits, and an allowlist that a pending min_tokens
    empties can allow nothing: both verify in ``logits`` mode."""
    for params in (
        SamplingParams(temperature=0.0, logprobs=1, allowed_token_ids=[2, 4]),
        SamplingParams(
            temperature=0.0, allowed_token_ids=[2], min_tokens=4, stop_token_ids=[2]
        ),
    ):
        model = MaskedTarget()
        runner = _runner(model)
        sync_harness._add_request(runner, "c", params)
        _step(runner, "c")
        _step(runner, "c", drafts={"c": [2, 2, 2]})
        assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS} and not model.masks


def test_a_sampled_row_draws_inside_its_controls_on_the_device():
    model = MaskedTarget()
    runner = _runner(model)
    params = SamplingParams(
        temperature=1.0, seed=3, allowed_token_ids=list(range(0, SAMPLED_VOCAB_SIZE, 3))
    )
    sync_harness._add_request(runner, "s", params, first_token=5)
    sync_harness._add_request(runner, "g", SamplingParams(temperature=0.0), 21)
    while len(sync_harness._output(runner, "s")) < 40:
        _step(runner, "s", "g", drafts={"g": sync_harness._argmax_drafts(runner, "g")})
    emitted = sync_harness._output(runner, "s")
    assert set(emitted) <= set(params.allowed_token_ids)
    assert set(model.verify_modes) == {ACCEPT_MODE_ARGMAX_IDS} and model.masks


def test_without_the_capability_a_token_control_asks_for_logits():
    model = SampledTarget()
    runner = _runner(model)
    sync_harness._add_request(runner, "c", _params("allowed", temperature=0.0))
    _step(runner, "c")
    _step(runner, "c", drafts={"c": [0, 2, 4]})
    assert set(model.verify_modes) == {ACCEPT_MODE_LOGITS}


def test_the_submitted_mask_is_ignored_by_a_row_past_its_drafts():
    """A column past a row's valid drafts stays unconstrained in the walk:
    the counts the model sees are the submitted ones."""
    model = MaskedTarget()
    runner = _runner(model)
    sync_harness._add_request(runner, "c", _params("allowed", temperature=0.0))
    _step(runner, "c")
    model_input = _step(runner, "c", drafts={"c": [0]})
    assert model_input.num_valid_drafts.tolist()[0] == 1
    assert model.num_valid[-1][0] == 1
    assert model_input.draft_token_ids[0, 1:].tolist() == [PLACEHOLDER_TOKEN_ID] * 2


# endregion

# region Admission


@pytest.mark.parametrize(
    "capabilities, error",
    [
        pytest.param(
            {"supports_spec_device_mask": "yes"}, "is a bool", id="not-a-bool"
        ),
        pytest.param(
            {"supports_spec_device_mask": True}, "spec_ids_are_sampled", id="no-draws"
        ),
        pytest.param(
            {"supports_spec_device_mask": True, "spec_ids_are_sampled": True},
            None,
            id="valid",
        ),
        pytest.param({}, None, id="absent"),
    ],
)
def test_the_capability_is_a_bool_that_needs_device_draws(capabilities, error):
    if error is None:
        validate_spec_device_mask(capabilities, "M")
        return
    with pytest.raises(ValueError, match=error):
        validate_spec_device_mask(capabilities, "M")


# endregion
