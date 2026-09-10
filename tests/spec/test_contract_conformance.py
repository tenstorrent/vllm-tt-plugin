# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The speculative-decoding contract enforces its own shapes and modes.

These tests exercise the two mechanisms the contract types own: the inclusive
domain of ``accepted_counts``, and the pairing between a ``spec_mode`` and the
verify-return fields it requires. They also drive ``FakeSpecModel`` through the
inputs the contract forbids, so a later change to the runner that builds a
ragged block, a zero count or a mismatched width fails here rather than inside
an accept walk.
"""

import pytest
import torch

# vLLM's own bootstrap resolves the platform plugin, which imports plugin
# modules. Letting a plugin import trigger that bootstrap deadlocks the cycle
# on a half-built module, so let vLLM finish importing itself first.
import vllm  # noqa: F401

from tests.spec.fake_spec_model import (
    FAKE_VOCAB_SIZE,
    FakeSpecModel,
    make_fake_spec_model,
)
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_FUSED_SAMPLE,
    ACCEPT_MODE_LOGITS,
    ACCEPT_MODES,
    DRAFTER_STATE_INTERNAL,
    HIDDEN_HANDOFFS,
    SPEC_REQUIREMENTS,
    SpecPlan,
    SpecReject,
    VerifyOutput,
    normalize_declared_values,
)


def _plan(**overrides) -> SpecPlan:
    fields = {
        "effective_k": 3,
        "lanes_per_request": 4,
        "extra_bytes_per_seq": 0,
        "extra_bytes_per_token": 0,
        "accept_modes": (ACCEPT_MODE_ARGMAX_IDS,),
        "drafter_state": DRAFTER_STATE_INTERNAL,
    }
    fields.update(overrides)
    return SpecPlan(**fields)


def _block(rows: int, block_width: int, base: int = 10):
    tokens = (
        torch.arange(rows * block_width, dtype=torch.int32).reshape(rows, block_width)
        + base
    )
    positions = torch.arange(rows * block_width, dtype=torch.int32).reshape(
        rows, block_width
    )
    return tokens, positions


def _counts(rows: int, value: int):
    return torch.full((rows,), value, dtype=torch.int32)


# --- the contract's derived quantities ------------------------------------


@pytest.mark.parametrize("effective_k", [1, 3, 7, 11])
def test_accepted_counts_range_is_one_based_and_never_zero(effective_k):
    plan = _plan(effective_k=effective_k, lanes_per_request=effective_k + 1)
    assert plan.accepted_counts_range == (1, 1 + effective_k)
    assert plan.block_width == 1 + effective_k


# --- SpecPlan rejects what the runner cannot budget with ------------------


@pytest.mark.parametrize(
    "overrides, offending",
    [
        ({"effective_k": 0}, "0"),
        ({"effective_k": -1}, "-1"),
        ({"lanes_per_request": 0}, "0"),
        ({"extra_bytes_per_seq": -1}, "-1"),
        ({"extra_bytes_per_token": -8}, "-8"),
        ({"accept_modes": ()}, "mode"),
        ({"accept_modes": ("nonsense",)}, "nonsense"),
        (
            {"accept_modes": (ACCEPT_MODE_LOGITS, ACCEPT_MODE_LOGITS)},
            ACCEPT_MODE_LOGITS,
        ),
        ({"drafter_state": "elsewhere"}, "elsewhere"),
    ],
)
def test_spec_plan_raises_and_names_the_offending_value(overrides, offending):
    with pytest.raises(ValueError) as excinfo:
        _plan(**overrides)
    assert offending in str(excinfo.value)


def test_spec_plan_normalizes_accept_modes_to_a_tuple():
    # A list field on a frozen value object would be shared mutable state.
    declared = [ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS]
    plan = _plan(accept_modes=declared)
    declared.append(ACCEPT_MODE_FUSED_SAMPLE)
    assert plan.accept_modes == (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS)


def test_spec_plan_accepts_fused_sample_because_admission_owns_that_policy():
    # The type is the contract surface. Refusing a mode no model implements is
    # a config-time decision, not a property of the value object.
    plan = _plan(accept_modes=(ACCEPT_MODE_FUSED_SAMPLE,))
    assert plan.accept_modes == (ACCEPT_MODE_FUSED_SAMPLE,)


# --- SpecReject carries a usable refusal ----------------------------------


def test_spec_reject_requires_a_reason():
    with pytest.raises(ValueError):
        SpecReject(reason="   ")


@pytest.mark.parametrize("supported_k", [(0,), (-2,), (3, 3)])
def test_spec_reject_rejects_an_unusable_supported_set(supported_k):
    with pytest.raises(ValueError):
        SpecReject(reason="no", supported_k=supported_k)


def test_spec_reject_normalizes_supported_k_to_a_tuple():
    assert SpecReject(reason="no", supported_k=[3, 7]).supported_k == (3, 7)


# --- the mode-to-return pairing -------------------------------------------


def test_every_accept_mode_has_a_declared_return_shape():
    # A mode with no required fields would construct empty and fail later.
    for mode in ACCEPT_MODES:
        with pytest.raises(ValueError):
            VerifyOutput(spec_mode=mode)


def test_verify_output_accepts_each_mode_with_its_own_fields():
    ids = torch.zeros(2, 4, dtype=torch.int32)
    logits = torch.zeros(2, 4, FAKE_VOCAB_SIZE)
    counts = _counts(2, 1)
    assert (
        VerifyOutput(spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids).argmax_ids is ids
    )
    assert VerifyOutput(spec_mode=ACCEPT_MODE_LOGITS, logits=logits).logits is logits
    fused = VerifyOutput(
        spec_mode=ACCEPT_MODE_FUSED_SAMPLE,
        accepted_token_ids=ids,
        accepted_counts=counts,
    )
    assert fused.accepted_counts is counts


def test_verify_output_rejects_a_mode_carrying_the_wrong_field():
    with pytest.raises(ValueError):
        VerifyOutput(
            spec_mode=ACCEPT_MODE_LOGITS,
            argmax_ids=torch.zeros(2, 4, dtype=torch.int32),
        )
    with pytest.raises(ValueError):
        VerifyOutput(
            spec_mode=ACCEPT_MODE_FUSED_SAMPLE,
            accepted_token_ids=torch.zeros(2, 4, dtype=torch.int32),
        )


def test_verify_output_rejects_an_unknown_mode():
    with pytest.raises(ValueError) as excinfo:
        VerifyOutput(spec_mode="device")
    assert "device" in str(excinfo.value)


def test_hidden_handle_stays_opaque():
    # The runner holds it and passes it back without interpreting it, so any
    # object must survive the round trip, including None for on-device retention.
    sentinel = object()
    ids = torch.zeros(1, 2, dtype=torch.int32)
    assert (
        VerifyOutput(
            spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids, hidden=sentinel
        ).hidden
        is sentinel
    )
    assert VerifyOutput(spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids).hidden is None


# --- declared capability lists --------------------------------------------


def test_absent_declaration_is_empty_not_an_error():
    assert normalize_declared_values(None, SPEC_REQUIREMENTS, "spec_requirements") == ()


def test_declared_capability_typo_raises_rather_than_disabling_silently():
    with pytest.raises(ValueError) as excinfo:
        normalize_declared_values(
            ["devise_propose"], SPEC_REQUIREMENTS, "spec_requirements"
        )
    assert "devise_propose" in str(excinfo.value)


def test_declared_capability_duplicate_raises():
    with pytest.raises(ValueError):
        normalize_declared_values(
            ["on_device", "on_device"], HIDDEN_HANDOFFS, "spec_hidden_handoff"
        )


# --- the stand-in model conforms and declares well-formed capabilities ----


def test_fake_model_capabilities_use_known_values():
    capabilities = FakeSpecModel.model_capabilities
    assert capabilities["supports_spec_decode"] is True
    assert normalize_declared_values(
        capabilities["spec_requirements"], SPEC_REQUIREMENTS, "spec_requirements"
    )
    assert normalize_declared_values(
        capabilities["spec_hidden_handoff"], HIDDEN_HANDOFFS, "spec_hidden_handoff"
    )


def test_spec_plan_reduces_an_over_large_requested_k():
    plan = FakeSpecModel.spec_plan(None, max_num_seqs=1, requested_k=9)
    assert isinstance(plan, SpecPlan)
    assert plan.effective_k == 7
    assert plan.lanes_per_request == plan.block_width


def test_spec_plan_refuses_a_draft_length_below_every_supported_one():
    reject = FakeSpecModel.spec_plan(None, max_num_seqs=1, requested_k=2)
    assert isinstance(reject, SpecReject)
    assert reject.supported_k == (3, 7, 11)


def test_spec_plan_refuses_a_concurrency_the_model_cannot_serve():
    reject = FakeSpecModel.spec_plan(None, max_num_seqs=8, requested_k=3)
    assert isinstance(reject, SpecReject)
    assert "8" in reject.reason


def test_configured_variant_does_not_leak_into_the_base_class():
    variant = make_fake_spec_model(max_supported_num_seqs=4, supported_k=(1,))
    assert isinstance(variant.spec_plan(None, max_num_seqs=4, requested_k=1), SpecPlan)
    assert isinstance(
        FakeSpecModel.spec_plan(None, max_num_seqs=4, requested_k=1), SpecReject
    )


def test_unknown_knob_raises_rather_than_being_ignored():
    with pytest.raises(ValueError) as excinfo:
        make_fake_spec_model(effective_k=3)
    assert "effective_k" in str(excinfo.value)


# --- both accept modes agree ----------------------------------------------


def test_the_two_accept_modes_return_the_same_token_ids():
    # A test of one mode proves nothing about the other unless they agree.
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    valid = torch.full((rows,), num_drafts, dtype=torch.int32)
    counts = _counts(rows, 1)
    ids = model.decode_forward(
        tokens, positions, valid, counts, spec_mode=ACCEPT_MODE_ARGMAX_IDS
    ).argmax_ids
    logits = model.decode_forward(
        tokens, positions, valid, counts, spec_mode=ACCEPT_MODE_LOGITS
    ).logits
    assert torch.equal(logits.argmax(dim=-1).to(torch.int32), ids)


def test_fused_sample_is_refused_because_no_hardware_runs_it():
    model = FakeSpecModel()
    tokens, positions = _block(1, 4)
    with pytest.raises(ValueError):
        model.decode_forward(
            tokens,
            positions,
            torch.full((1,), 3, dtype=torch.int32),
            _counts(1, 1),
            spec_mode=ACCEPT_MODE_FUSED_SAMPLE,
        )


# --- the stand-in refuses every input the contract forbids ----------------


def test_propose_and_verify_take_the_same_padded_width():
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    counts = _counts(rows, 1)
    model.propose_draft_tokens(num_drafts, tokens, positions, counts)
    model.decode_forward(
        tokens, positions, torch.full((rows,), num_drafts, dtype=torch.int32), counts
    )
    assert model.propose_calls[0]["rows"] == model.verify_calls[0]["rows"]
    assert model.verify_calls[0]["block_width"] == 1 + num_drafts


def test_propose_refuses_a_ragged_committed_block():
    model = FakeSpecModel()
    tokens, positions = _block(2, 4)
    with pytest.raises(ValueError):
        model.propose_draft_tokens(3, tokens[:, :3], positions, _counts(2, 1))


def test_propose_refuses_a_block_narrower_than_one_plus_k():
    model = FakeSpecModel()
    tokens, positions = _block(2, 3)
    with pytest.raises(ValueError):
        model.propose_draft_tokens(3, tokens, positions, _counts(2, 1))


@pytest.mark.parametrize("count", [0, 5])
def test_verify_refuses_an_accepted_count_outside_the_inclusive_domain(count):
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    with pytest.raises(ValueError) as excinfo:
        model.decode_forward(
            tokens,
            positions,
            torch.full((rows,), num_drafts, dtype=torch.int32),
            _counts(rows, count),
        )
    assert str(count) in str(excinfo.value)


def test_verify_accepts_the_lower_bound_that_follows_a_prefill():
    # 1 means only the input token stood, which is also the value after a
    # prefill and after a non-speculating step.
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    out = model.decode_forward(
        tokens,
        positions,
        torch.zeros(rows, dtype=torch.int32),
        _counts(rows, 1),
    )
    assert out.argmax_ids.shape == (rows, 1 + num_drafts)


def test_verify_refuses_a_non_int32_accepted_counts():
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    with pytest.raises(ValueError):
        model.decode_forward(
            tokens,
            positions,
            torch.full((rows,), num_drafts, dtype=torch.int32),
            torch.ones(rows, dtype=torch.int64),
        )


def test_verify_refuses_a_none_accepted_counts_without_a_fused_previous_step():
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    with pytest.raises(ValueError):
        model.decode_forward(
            tokens,
            positions,
            torch.full((rows,), num_drafts, dtype=torch.int32),
            None,
        )


@pytest.mark.parametrize("valid", [-1, 4])
def test_verify_refuses_a_per_row_draft_count_outside_zero_to_k(valid):
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    with pytest.raises(ValueError):
        model.decode_forward(
            tokens,
            positions,
            torch.full((rows,), valid, dtype=torch.int32),
            _counts(rows, 1),
        )


def test_verify_refuses_a_mis_sized_per_row_draft_count():
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    with pytest.raises(ValueError):
        model.decode_forward(
            tokens,
            positions,
            torch.full((rows + 1,), num_drafts, dtype=torch.int32),
            _counts(rows, 1),
        )
