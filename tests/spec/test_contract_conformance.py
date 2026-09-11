# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The speculative-decoding contract enforces its own shapes and modes.

These tests exercise the mechanisms the contract types own: the inclusive
domain of ``accepted_counts``, the pairing between a ``spec_mode`` and the
verify-return fields it requires, and the completeness of that pairing. They
also drive ``FakeSpecModel`` through the inputs the contract forbids, so a
later change to the runner that builds a ragged block, a zero count, a
mismatched width or a batch-wide draft count fails here rather than inside an
accept walk.
"""

import pytest
import torch

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
    MODE_REQUIRED_FIELDS,
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


def _per_row(rows: int, value: int):
    """One int32 entry per row, for accepted_counts and num_valid_drafts."""
    return torch.full((rows,), value, dtype=torch.int32)


def _verify(model, tokens, positions, valid, counts, mode=ACCEPT_MODE_ARGMAX_IDS):
    """Call the verify primitive with a mode always supplied.

    The primitive itself has no default, so that a runner cannot silently get
    greedy ids; a default here only keeps the input-validation tests short.
    """
    return model.decode_forward(tokens, positions, valid, counts, spec_mode=mode)


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


def test_every_accept_mode_declares_its_required_fields():
    # Pinned directly: a mode missing from the mapping would otherwise be
    # caught by the unknown-mode branch, which proves nothing about the
    # mapping's completeness.
    assert set(MODE_REQUIRED_FIELDS) == set(ACCEPT_MODES)
    assert all(fields for fields in MODE_REQUIRED_FIELDS.values())


@pytest.mark.parametrize("mode", sorted(ACCEPT_MODES))
def test_verify_output_requires_the_fields_its_mode_declares(mode):
    with pytest.raises(ValueError) as excinfo:
        VerifyOutput(spec_mode=mode)
    assert mode in str(excinfo.value)


def test_verify_output_accepts_each_mode_with_its_own_fields():
    ids = torch.zeros(2, 4, dtype=torch.int32)
    logits = torch.zeros(2, 4, FAKE_VOCAB_SIZE)
    counts = _per_row(2, 1)
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


def test_verify_output_rejects_a_mode_missing_a_required_field():
    ids = torch.zeros(2, 4, dtype=torch.int32)
    # argmax_ids is not what "logits" mode declares, so the required field is
    # still missing however many other fields are set.
    with pytest.raises(ValueError):
        VerifyOutput(spec_mode=ACCEPT_MODE_LOGITS, argmax_ids=ids)
    with pytest.raises(ValueError):
        VerifyOutput(spec_mode=ACCEPT_MODE_FUSED_SAMPLE, accepted_token_ids=ids)


def test_verify_output_tolerates_a_field_another_mode_uses():
    # Only the declared fields are required; a model that fills more is not
    # wrong, so the runner reads by mode rather than by presence.
    ids = torch.zeros(2, 4, dtype=torch.int32)
    logits = torch.zeros(2, 4, FAKE_VOCAB_SIZE)
    out = VerifyOutput(spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids, logits=logits)
    assert out.spec_mode == ACCEPT_MODE_ARGMAX_IDS
    assert out.logits is logits


def test_verify_output_rejects_an_unknown_mode():
    with pytest.raises(ValueError) as excinfo:
        VerifyOutput(spec_mode="device")
    assert "device" in str(excinfo.value)


def test_hidden_handle_stays_opaque():
    # The runner holds it and passes it back without interpreting it, so any
    # object must survive the round trip, including None for on-device
    # retention.
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


@pytest.mark.parametrize(
    "method", ["spec_plan", "propose_draft_tokens", "decode_forward"]
)
def test_a_contract_method_is_not_a_settable_knob(method):
    # Overriding one would replace a primitive with a plain value and fail at
    # the call site instead of here.
    with pytest.raises(ValueError) as excinfo:
        make_fake_spec_model(**{method: None})
    assert method in str(excinfo.value)


# --- both accept modes agree ----------------------------------------------


def test_the_two_accept_modes_return_the_same_token_ids():
    # A test of one mode proves nothing about the other unless they agree.
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    valid = _per_row(rows, num_drafts)
    counts = _per_row(rows, 1)
    ids = _verify(model, tokens, positions, valid, counts).argmax_ids
    logits = _verify(
        model, tokens, positions, valid, counts, mode=ACCEPT_MODE_LOGITS
    ).logits
    assert torch.equal(logits.argmax(dim=-1).to(torch.int32), ids)


def test_fused_sample_is_refused_because_no_implementation_serves_it():
    model = FakeSpecModel()
    tokens, positions = _block(1, 4)
    with pytest.raises(ValueError):
        _verify(
            model,
            tokens,
            positions,
            _per_row(1, 3),
            _per_row(1, 1),
            mode=ACCEPT_MODE_FUSED_SAMPLE,
        )


def test_verify_requires_an_explicit_spec_mode():
    # A runner that forgets the mode must fail, not receive greedy ids.
    model = FakeSpecModel()
    tokens, positions = _block(1, 4)
    with pytest.raises(TypeError):
        model.decode_forward(tokens, positions, _per_row(1, 3), _per_row(1, 1))


# --- acceptance is per row, never reduced across the batch ----------------


def test_each_row_accepts_to_its_own_valid_draft_count():
    # A batch-wide count would let one grammar-truncated request destroy every
    # other request's speculation, which the contract forbids.
    model = FakeSpecModel()
    tokens, positions = _block(2, 4)
    ids = _verify(
        model,
        tokens,
        positions,
        torch.tensor([3, 0], dtype=torch.int32),
        _per_row(2, 1),
    ).argmax_ids
    assert torch.equal(ids[0], tokens[0])
    assert int(ids[1][1]) != int(tokens[1][1])


def test_a_row_with_fewer_valid_drafts_diverges_only_past_its_own_count():
    model = FakeSpecModel()
    tokens, positions = _block(2, 4)
    ids = _verify(
        model,
        tokens,
        positions,
        torch.tensor([1, 3], dtype=torch.int32),
        _per_row(2, 1),
    ).argmax_ids
    assert int(ids[0][1]) == int(tokens[0][1])
    assert int(ids[0][2]) != int(tokens[0][2])
    assert torch.equal(ids[1], tokens[1])


def test_accept_depth_caps_a_row_without_reducing_across_the_batch():
    model = make_fake_spec_model(accept_depth=1)()
    tokens, positions = _block(2, 4)
    ids = _verify(model, tokens, positions, _per_row(2, 3), _per_row(2, 1)).argmax_ids
    for row in range(2):
        assert int(ids[row][1]) == int(tokens[row][1])
        assert int(ids[row][2]) != int(tokens[row][2])


# --- the stand-in refuses every input the contract forbids ----------------


def test_verify_then_propose_take_the_same_padded_width():
    # The contract's per-step order: verify first, then propose from that
    # step's hidden state.
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    counts = _per_row(rows, 1)
    _verify(model, tokens, positions, _per_row(rows, num_drafts), counts)
    model.propose_draft_tokens(num_drafts, tokens, positions, counts)
    assert model.verify_calls[0]["rows"] == model.propose_calls[0]["rows"]
    assert model.verify_calls[0]["block_width"] == 1 + num_drafts


@pytest.mark.parametrize(
    "tokens, positions, offender",
    [
        (torch.zeros(4, dtype=torch.int32), torch.zeros(4, dtype=torch.int32), "2-D"),
        (
            torch.zeros(2, 4, dtype=torch.int32),
            torch.zeros(3, 4, dtype=torch.int32),
            "same shape",
        ),
        (
            torch.zeros(2, 3, dtype=torch.int32),
            torch.zeros(2, 3, dtype=torch.int32),
            "uniform width",
        ),
        (
            torch.zeros(0, 4, dtype=torch.int32),
            torch.zeros(0, 4, dtype=torch.int32),
            "at least one row",
        ),
    ],
)
def test_each_structural_violation_trips_its_own_guard(tokens, positions, offender):
    # Each input is invalid in exactly one way, so a removed guard cannot hide
    # behind a later one.
    model = FakeSpecModel()
    with pytest.raises(ValueError) as excinfo:
        model.propose_draft_tokens(3, tokens, positions, _per_row(2, 1))
    assert offender in str(excinfo.value)


def test_verify_refuses_an_empty_batch():
    model = FakeSpecModel()
    empty = torch.zeros(0, 4, dtype=torch.int32)
    with pytest.raises(ValueError) as excinfo:
        _verify(
            model,
            empty,
            empty,
            torch.zeros(0, dtype=torch.int32),
            torch.zeros(0, dtype=torch.int32),
        )
    assert "at least one row" in str(excinfo.value)


@pytest.mark.parametrize("count", [0, 5])
def test_verify_refuses_an_accepted_count_outside_the_inclusive_domain(count):
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    with pytest.raises(ValueError) as excinfo:
        _verify(
            model, tokens, positions, _per_row(rows, num_drafts), _per_row(rows, count)
        )
    assert str(count) in str(excinfo.value)


def test_verify_accepts_the_lower_bound_that_follows_a_prefill():
    # 1 means only the input token stood, which is also the value after a
    # prefill and after a non-speculating step.
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    out = _verify(model, tokens, positions, _per_row(rows, 0), _per_row(rows, 1))
    assert out.argmax_ids.shape == (rows, 1 + num_drafts)


def test_both_primitives_accept_the_inclusive_upper_bound():
    # 1+K is full acceptance, the common speculative success, and it is the
    # exact value an exclusive upper bound would reject.
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    full = _per_row(rows, 1 + num_drafts)
    out = _verify(model, tokens, positions, _per_row(rows, num_drafts), full)
    assert out.argmax_ids.shape == (rows, 1 + num_drafts)
    drafts = model.propose_draft_tokens(
        num_drafts, tokens, positions, full
    ).draft_token_ids
    assert drafts.shape == (rows, num_drafts)


def test_verify_refuses_a_non_int32_accepted_per_row():
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    with pytest.raises(ValueError):
        _verify(
            model,
            tokens,
            positions,
            _per_row(rows, num_drafts),
            torch.ones(rows, dtype=torch.int64),
        )


def test_verify_refuses_a_none_accepted_counts_without_a_fused_previous_step():
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    with pytest.raises(ValueError):
        _verify(model, tokens, positions, _per_row(rows, num_drafts), None)


@pytest.mark.parametrize("valid", [-1, 4])
def test_verify_refuses_a_per_row_draft_count_outside_zero_to_k(valid):
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    with pytest.raises(ValueError):
        _verify(model, tokens, positions, _per_row(rows, valid), _per_row(rows, 1))


def test_verify_refuses_a_mis_sized_per_row_draft_count():
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    with pytest.raises(ValueError):
        _verify(
            model, tokens, positions, _per_row(rows + 1, num_drafts), _per_row(rows, 1)
        )


def test_propose_records_the_hidden_handoff_and_returns_no_scores():
    # On-device retention hands propose no tensor, and this stand-in declares
    # no drafter scores, so an accept rule needing them must not assume they
    # are there.
    model = FakeSpecModel()
    rows, num_drafts = 2, 3
    tokens, positions = _block(rows, 1 + num_drafts)
    out = model.propose_draft_tokens(
        num_drafts, tokens, positions, _per_row(rows, 1), hidden=None
    )
    assert out.draft_scores is None
    assert model.propose_calls[0]["hidden_was_none"] is True
    sentinel = object()
    model.propose_draft_tokens(
        num_drafts, tokens, positions, _per_row(rows, 1), hidden=sentinel
    )
    assert model.propose_calls[1]["hidden_was_none"] is False
