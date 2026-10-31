# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests of the masked-verify path (structured outputs with speculation, M1).

A structured request is served by the model's own verify when the switch
``VLLM_TT_STRUCTURED_SPEC=1`` is on and the model declares
``supports_verify_grammar_bitmask`` and a live ``verify_grammar_enabled``: the
request is certifiable, is offered no drafts, forces a verify step, and the
packed grammar mask reaches ``decode_forward`` as the ``grammar_bitmask``
keyword. Contract: docs/STRUCTURED_SPEC_INTERFACE.md.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from vllm_tt_plugin.model_runner import TTModelRunner, _step_verifies
from vllm_tt_plugin.spec_decode import VerifyOutput

from .fake_spec_model import FakeSpecModel  # noqa: F401
from .test_spec_inputs import (
    MAX_NUM_REQS,
    _add_decoding_request,
    _batch,
    _decode,
    _drafts,
    _fake_runner,
    _submit,
)

WORDS = 16  # a small packed width is enough for the pass-through


def _model(enabled=True, capable=True, positions=False):
    caps = {"supports_verify_grammar_bitmask": capable}
    return SimpleNamespace(
        model_capabilities=caps,
        verify_grammar_enabled=enabled,
        verify_grammar_positions=positions,
    )


def _runner(model, requests=None, drafts_ok=True):
    runner = SimpleNamespace(model=model, requests=requests or {})
    runner.input_batch = SimpleNamespace()
    runner._request_is_argmax_certifiable = TTModelRunner._request_is_argmax_certifiable.__get__(runner)
    runner._request_is_speculable = TTModelRunner._request_is_speculable.__get__(runner)
    runner._proposed_draft_token_ids = {}
    runner._spec_accept_modes = ("argmax_ids",)
    return runner


def _state(**params):
    return SimpleNamespace(
        sampling_params=SamplingParams(temperature=0.0, **params),
        output_token_ids=[],
    )


def _structured():
    return StructuredOutputsParams(regex="[a-z]+")


@pytest.fixture
def switch_on(monkeypatch):
    monkeypatch.setenv("VLLM_TT_STRUCTURED_SPEC", "1")


def test_active_needs_switch_capability_and_a_live_model(monkeypatch):
    monkeypatch.delenv("VLLM_TT_STRUCTURED_SPEC", raising=False)
    assert not TTModelRunner._structured_spec_active(_runner(_model()))
    monkeypatch.setenv("VLLM_TT_STRUCTURED_SPEC", "1")
    assert TTModelRunner._structured_spec_active(_runner(_model()))
    assert not TTModelRunner._structured_spec_active(_runner(_model(enabled=False)))
    assert not TTModelRunner._structured_spec_active(_runner(_model(capable=False)))
    assert not TTModelRunner._structured_spec_active(_runner(SimpleNamespace()))


def test_a_structured_row_is_certifiable_only_on_the_masked_path(monkeypatch):
    requests = {"s": _state(structured_outputs=_structured()), "lp": _state(logprobs=1, structured_outputs=_structured()), "g": _state()}
    monkeypatch.delenv("VLLM_TT_STRUCTURED_SPEC", raising=False)
    off = _runner(_model(), requests)
    assert not off._request_is_argmax_certifiable("s")
    monkeypatch.setenv("VLLM_TT_STRUCTURED_SPEC", "1")
    on = _runner(_model(), requests)
    assert on._request_is_argmax_certifiable("s")
    assert on._request_is_argmax_certifiable("g")
    # another control the ids cannot answer still keeps the row off the verify
    assert not on._request_is_argmax_certifiable("lp")


def test_a_masked_structured_row_is_offered_no_drafts(switch_on):
    requests = {"s": _state(structured_outputs=_structured()), "g": _state()}
    runner = _runner(_model(), requests)
    TTModelRunner._publish_draft(runner, "s", [1, 2, 3])
    TTModelRunner._publish_draft(runner, "g", [4, 5, 6])
    assert "s" not in runner._proposed_draft_token_ids
    assert runner._proposed_draft_token_ids["g"] == [4, 5, 6]
    assert TTModelRunner._request_is_masked_verify_structured(runner, "s")
    assert not TTModelRunner._request_is_masked_verify_structured(runner, "g")


def test_force_verify_sends_a_draftless_step_to_the_model():
    no_drafts = torch.zeros(2, dtype=torch.int32)
    counts = torch.ones(2, dtype=torch.int32)
    assert not _step_verifies(True, True, no_drafts, counts, 2)
    assert _step_verifies(True, True, no_drafts, counts, 2, force_verify=True)
    # a row that is not speculable still overrules it
    assert not _step_verifies(True, False, no_drafts, counts, 2, force_verify=True)


class _Model:
    decode_input_update_contract = 1
    model_capabilities = {"supports_async_decode": False}

    def __init__(self):
        self.calls = []

    def decode_forward(self, **kwargs):
        self.calls.append(kwargs)
        return VerifyOutput(
            spec_mode=kwargs["spec_mode"],
            argmax_ids=torch.zeros(kwargs["tokens"].shape, dtype=torch.int32),
        )


def _verify_input():
    batch = _batch()
    requests = {r: _add_decoding_request(batch, r) for r in ("a", "b")}
    runner = _fake_runner(batch, requests)
    return runner, _decode(runner, "a", "b", drafts=_drafts(a=[11, 12, 13]))


def test_the_packed_mask_reaches_decode_forward():
    runner, model_input = _verify_input()
    k1 = int(model_input.input_tokens.shape[1])
    mask = torch.full((MAX_NUM_REQS, k1, WORDS), -1, dtype=torch.int32)
    mask[1, 0, 0] = 5
    model = _Model()
    _submit(runner, model, replace(model_input, grammar_bitmask=[mask]))
    got = model.calls[0]["grammar_bitmask"]
    assert got.dtype == torch.int32 and got.shape == (MAX_NUM_REQS, k1, WORDS)
    assert torch.equal(got, mask)


def test_a_verify_without_a_mask_keeps_the_call_shape():
    runner, model_input = _verify_input()
    model = _Model()
    _submit(runner, model, model_input)
    assert "grammar_bitmask" not in model.calls[0]


def _plain_step_runner(speculable):
    calls = []
    runner = SimpleNamespace(
        _request_is_speculable=lambda r: speculable[r],
        _proposed_draft_token_ids={"a": [1], "b": [2]},
        _spec_drafts_from_model=True,
        _propose_model_drafts=lambda *args, **kw: calls.append(args),
    )
    return runner, calls


def _propose(runner, rows=("a", "b")):
    batch = SimpleNamespace(input_positions=torch.zeros(len(rows), 1, dtype=torch.int64))
    TTModelRunner.propose_after_plain_step(
        runner, batch, torch.zeros(len(rows), 1, dtype=torch.int32), list(rows)
    )


def test_a_plain_step_with_an_uncertifiable_row_proposes_nothing():
    """The model's drafter reseeds by replaying the prompt tail on every call for a row the step did not advance."""
    runner, calls = _plain_step_runner({"a": True, "b": False})
    _propose(runner)
    assert calls == [] and runner._proposed_draft_token_ids == {}


def test_a_plain_step_of_speculable_rows_still_proposes():
    runner, calls = _plain_step_runner({"a": True, "b": True})
    _propose(runner)
    assert len(calls) == 1


def test_m2_needs_the_model_attribute_and_the_plugin_switch(monkeypatch):
    monkeypatch.setenv("VLLM_TT_STRUCTURED_SPEC", "1")
    monkeypatch.delenv("VLLM_TT_STRUCTURED_M2", raising=False)
    assert not TTModelRunner._structured_m2_active(_runner(_model(positions=False)))
    assert TTModelRunner._structured_m2_active(_runner(_model(positions=True)))
    monkeypatch.setenv("VLLM_TT_STRUCTURED_M2", "0")
    assert not TTModelRunner._structured_m2_active(_runner(_model(positions=True)))
    monkeypatch.setenv("VLLM_TT_STRUCTURED_SPEC", "0")
    monkeypatch.delenv("VLLM_TT_STRUCTURED_M2", raising=False)
    assert not TTModelRunner._structured_m2_active(_runner(_model(positions=True)))


def test_under_m2_a_structured_row_speculates_but_still_forces_the_verify(monkeypatch):
    monkeypatch.setenv("VLLM_TT_STRUCTURED_SPEC", "1")
    monkeypatch.delenv("VLLM_TT_STRUCTURED_M2", raising=False)
    requests = {"s": _state(structured_outputs=_structured()), "g": _state()}
    runner = _runner(_model(positions=True), requests)
    TTModelRunner._publish_draft(runner, "s", [1, 2, 3])
    assert runner._proposed_draft_token_ids["s"] == [1, 2, 3]
    assert not TTModelRunner._request_is_masked_verify_structured(runner, "s")
    assert TTModelRunner._request_is_structured_verify_row(runner, "s")
    assert not TTModelRunner._request_is_structured_verify_row(runner, "g")
    assert runner._request_is_argmax_certifiable("s")


def test_mode_file_overrides_the_environment_live(monkeypatch, tmp_path):
    f = tmp_path / "mode"
    monkeypatch.setenv("VLLM_TT_STRUCTURED_MODE_FILE", str(f))
    monkeypatch.setenv("VLLM_TT_STRUCTURED_SPEC", "1")
    runner = _runner(_model(positions=True))
    assert TTModelRunner._structured_mode(runner) == "m2"  # no file: environment defaults
    for word, want in (("m1", "m1"), ("m0", "m0"), ("m2", "m2")):
        f.write_text(word + "\n")
        runner.__dict__.pop("_struct_mode_file_cache", None)  # the 0.2 s cache
        assert TTModelRunner._structured_mode(runner) == want
    f.write_text("garbage")
    runner.__dict__.pop("_struct_mode_file_cache", None)
    assert TTModelRunner._structured_mode(runner) == "m2"
    # a model without per-position masks falls back to m1; without the mask to m0
    f.write_text("m2")
    runner.__dict__.pop("_struct_mode_file_cache", None)
    assert TTModelRunner._structured_mode(_runner(_model(positions=False))) == "m1"
    assert TTModelRunner._structured_mode(_runner(_model(enabled=False))) == "m0"
