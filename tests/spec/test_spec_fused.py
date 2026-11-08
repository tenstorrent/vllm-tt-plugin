# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ``fused_sample`` accept mode, stage S1 (greedy rows only), on the host.

The device accepts and commits in this mode, so the runner has no walk to run:
it checks the two returned fields and commits them. What these tests pin is
that doing so is indistinguishable from the ``argmax_ids`` walk the same model
would have been given: same published tokens, same carried accepted counts,
same propose calls. The model is ``FakeSpecModel`` with ``serve_fused``, which
computes ids and counts by the greedy accept rule from its own deterministic
target.
"""

import importlib
import sys
from pathlib import Path

import pytest
import torch
from vllm.sampling_params import SamplingParams

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin import spec_accept
from vllm_tt_plugin.async_decode import (
    TTAsyncDecodeController,
    _verify_output_tensor,
)
from vllm_tt_plugin.model_runner import TTModelRunner, fused_spec_sampling
from vllm_tt_plugin.spec_accept import (
    SpecSamplingInputs,
    accept_sampled_support,
    fused_sample_enabled,
    fused_sampled_enabled,
    predraw_uniforms,
)
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_FUSED_SAMPLE,
    ACCEPT_MODE_LOGITS,
    ACCEPT_MODE_TOPK,
    PLACEHOLDER_TOKEN_ID,
    VerifyOutput,
)

from .fake_spec_model import FAKE_VOCAB_SIZE, make_fake_spec_model
from .test_spec_device_topk import (
    _topk_runner,
    _TopkTarget,
    exact_support,
)
from .test_spec_greedy_e2e import (
    DRAFT_LEN,
    MAX_NUM_REQS,
    PROMPT_LEN,
    _add_request,
    _model_drafter_runner,
    _runner,
    _scheduler_output_for,
    _step,
)
from .test_spec_sampled_runner import (
    GREEDY,
    _argmax_drafts,
    _build,
    _output,
)
from .test_spec_sampled_runner import _add_request as _add_sampled_request
from .test_spec_sampled_runner import _step as _sampled_step

FUSED_MODES = (ACCEPT_MODE_ARGMAX_IDS, ACCEPT_MODE_LOGITS, ACCEPT_MODE_FUSED_SAMPLE)


def _fused_model(**knobs):
    knobs.setdefault("accept_modes", FUSED_MODES)
    return make_fake_spec_model(serve_fused=True, **knobs)()


def _argmax_model(**knobs):
    return make_fake_spec_model(**knobs)()


def _with_modes(runner, model):
    runner._spec_accept_modes = tuple(model.accept_modes)
    return runner


@pytest.fixture(autouse=True)
def _fused_env(monkeypatch):
    monkeypatch.delenv("VLLM_TT_SPEC_FUSED", raising=False)


# region Mode choice


def test_all_greedy_rows_ask_for_fused_sample():
    model = _fused_model()
    runner = _with_modes(_runner(model), model)
    _add_request(runner, "a")
    _add_request(runner, "b", first_token=40)

    _step(runner, "a", "b")

    assert model.verify_calls[0]["spec_mode"] == ACCEPT_MODE_FUSED_SAMPLE


def test_one_sampled_row_falls_back_to_logits():
    model = _fused_model()
    runner = _with_modes(_runner(model), model)
    _add_request(runner, "a")
    _add_request(runner, "b", first_token=40, temperature=1.0)

    mode = TTModelRunner._spec_mode_for_step(runner, ["a", "b"])

    assert mode == ACCEPT_MODE_LOGITS


def test_env_flag_turns_fused_off(monkeypatch):
    model = _fused_model()
    runner = _with_modes(_runner(model), model)
    _add_request(runner, "a")
    monkeypatch.setenv("VLLM_TT_SPEC_FUSED", "0")

    assert not fused_sample_enabled()
    assert TTModelRunner._spec_mode_for_step(runner, ["a"]) == ACCEPT_MODE_ARGMAX_IDS
    monkeypatch.setenv("VLLM_TT_SPEC_FUSED", "1")
    assert fused_sample_enabled()
    assert TTModelRunner._spec_mode_for_step(runner, ["a"]) == ACCEPT_MODE_FUSED_SAMPLE


def test_a_model_without_the_mode_never_gets_it():
    model = _argmax_model()
    runner = _with_modes(_runner(model), model)
    _add_request(runner, "a")

    assert TTModelRunner._spec_mode_for_step(runner, ["a"]) == ACCEPT_MODE_ARGMAX_IDS


# endregion

# region Equivalence with the argmax_ids walk


def _drive(runner, steps=4, rows=("a", "b")):
    """Several drafted steps; returns per-step published tokens and counts."""
    for index, req in enumerate(rows):
        _add_request(runner, req, first_token=1 + 40 * index)
    published, counts = [], []
    drafts = None
    for _ in range(steps):
        output = _step(runner, *rows, drafts=drafts)
        published.append([list(ids) for ids in output.sampled_token_ids])
        counts.append(dict(runner._req_accepted_counts))
        # The drafts the model proposed travel to the next step as the
        # scheduler would deliver them.
        drafts = {
            req: list(runner._proposed_draft_token_ids.pop(req))
            for req in rows
            if req in runner._proposed_draft_token_ids
        }
    return published, counts


@pytest.mark.parametrize("accept_depth", [None, 0, 1, 2])
def test_fused_and_argmax_runs_are_identical(accept_depth):
    fused = _fused_model(accept_depth=accept_depth)
    plain = _argmax_model(accept_depth=accept_depth, accept_modes=FUSED_MODES)
    fused_runner = _with_modes(_model_drafter_runner(fused), fused)
    plain_runner = _model_drafter_runner(plain)
    plain_runner._spec_accept_modes = (ACCEPT_MODE_ARGMAX_IDS,)

    fused_published, fused_counts = _drive(fused_runner)
    plain_published, plain_counts = _drive(plain_runner)

    assert {c["spec_mode"] for c in fused.verify_calls} == {ACCEPT_MODE_FUSED_SAMPLE}
    assert {c["spec_mode"] for c in plain.verify_calls} == {ACCEPT_MODE_ARGMAX_IDS}
    assert fused_published == plain_published
    assert fused_counts == plain_counts
    assert len(fused.propose_calls) == len(plain.propose_calls)
    for a, b in zip(fused.propose_calls, plain.propose_calls):
        assert torch.equal(a["committed_tokens"], b["committed_tokens"])
        assert torch.equal(a["committed_positions"], b["committed_positions"])
        assert torch.equal(a["accepted_counts"], b["accepted_counts"])
        assert a["rows"] == b["rows"] == MAX_NUM_REQS


def test_the_deferred_completion_matches_the_argmax_walk():
    """``complete_spec_decode_step``, the async path's walk, takes the fields."""
    results = {}
    for name, model, modes in (
        ("fused", _fused_model(), FUSED_MODES),
        ("plain", _argmax_model(), (ACCEPT_MODE_ARGMAX_IDS,)),
    ):
        runner = _runner(model)
        runner._spec_accept_modes = modes
        _add_request(runner, "a")
        _add_request(runner, "b", first_token=40)
        last = int(runner.input_batch.token_ids_cpu[0, PROMPT_LEN - 1])
        drafts = {
            "a": [(last + 1 + j) % FAKE_VOCAB_SIZE for j in range(DRAFT_LEN)],
            "b": [3, 4],
        }
        for req in ("a", "b"):
            row = runner.input_batch.req_id_to_index[req]
            runner.input_batch.num_computed_tokens_cpu[row] = (
                runner.input_batch.num_tokens[row]
            )
        model_input = TTModelRunner._prepare_model_inputs(
            runner, _scheduler_output_for(runner, "a", "b", drafts=drafts), None
        )
        controller = TTAsyncDecodeController(runner)
        submission = controller.submit_decode(model_input, read_from_device=True)
        completed = controller.complete_spec_decode_step(
            submission, model_input, context=None
        )
        results[name] = completed
        assert model.verify_calls[0]["spec_mode"] == (
            ACCEPT_MODE_FUSED_SAMPLE if name == "fused" else ACCEPT_MODE_ARGMAX_IDS
        )
    assert torch.equal(results["fused"].committed, results["plain"].committed)
    assert torch.equal(results["fused"].counts, results["plain"].counts)
    assert results["fused"].prefixes == results["plain"].prefixes
    assert results["fused"].counts.tolist()[0] == DRAFT_LEN + 1


# endregion

# region Contract conformance of the new fields


def _verify(model, spec_mode=ACCEPT_MODE_FUSED_SAMPLE, valid=(3, 0), rows=2):
    width = DRAFT_LEN + 1
    tokens = torch.arange(rows * width, dtype=torch.int32).reshape(rows, width)
    positions = torch.arange(width, dtype=torch.int32).repeat(rows, 1)
    return model.decode_forward(
        tokens,
        positions,
        num_valid_drafts=torch.tensor(valid, dtype=torch.int32),
        accepted_counts=torch.ones(rows, dtype=torch.int32),
        spec_mode=spec_mode,
    )


def test_the_fake_returns_conformant_fields_and_pads_rows():
    out = _verify(_fused_model())
    assert out.spec_mode == ACCEPT_MODE_FUSED_SAMPLE
    assert out.accepted_token_ids.dtype == torch.int32
    assert out.accepted_counts.dtype == torch.int32
    assert out.accepted_token_ids.shape == (2, DRAFT_LEN + 1)
    # Row 0 accepts all three drafts and takes the bonus; row 1 carries no
    # draft, so it commits its column-0 argmax with count 1 (a padding row).
    assert out.accepted_counts.tolist() == [DRAFT_LEN + 1, 1]
    assert (out.accepted_token_ids[1, 1:] == PLACEHOLDER_TOKEN_ID).all()
    assert out.accepted_token_ids[1, 0] == out.argmax_ids[1, 0]


def test_the_unwrap_keeps_the_verify_output_for_fused_sample():
    out = _verify(_fused_model())
    assert _verify_output_tensor(out, "Fake", ACCEPT_MODE_FUSED_SAMPLE) is out
    with pytest.raises(NotImplementedError):
        _verify_output_tensor(out, "Fake", ACCEPT_MODE_ARGMAX_IDS)


def _walk(runner, out):
    model_input = TTModelRunner._prepare_model_inputs(
        runner, _scheduler_output_for(runner, "a"), None
    )
    return runner.walk_spec_acceptance(model_input, out), model_input


def _walk_runner():
    model = _fused_model()
    runner = _with_modes(_runner(model), model)
    _add_request(runner, "a")
    row = runner.input_batch.req_id_to_index["a"]
    runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[row]
    return runner


def _fused_out(ids, counts):
    return VerifyOutput(
        spec_mode=ACCEPT_MODE_FUSED_SAMPLE,
        accepted_token_ids=torch.tensor(ids, dtype=torch.int32),
        accepted_counts=torch.tensor(counts, dtype=torch.int32),
    )


def test_the_walk_takes_the_fields_as_the_walk_result():
    runner = _walk_runner()
    rows = MAX_NUM_REQS
    ids = [[7, 8, -1, -1]] + [[5, -1, -1, -1]] * (rows - 1)
    walk, _ = _walk(runner, _fused_out(ids, [2] + [1] * (rows - 1)))
    assert walk.logprobs is None
    assert walk.counts.tolist() == [2] + [1] * (rows - 1)
    assert walk.committed.tolist() == ids


@pytest.mark.parametrize(
    "ids, counts",
    [
        ([[7, 8, -1, -1]] * 4, [0, 1, 1, 1]),  # count 0
        ([[7, 8, -1, -1]] * 4, [5, 1, 1, 1]),  # count past width
        ([[7, -1, -1, -1]] * 4, [2, 1, 1, 1]),  # placeholder inside prefix
        ([[7, 8, 9, -1]] * 4, [2, 2, 2, 2]),  # id past the count
        ([[7, 8, -1, -1]] * 3, [2, 2, 2]),  # wrong row count
    ],
)
def test_the_walk_refuses_inconsistent_fields(ids, counts):
    runner = _walk_runner()
    with pytest.raises(ValueError):
        _walk(runner, _fused_out(ids, counts))


def test_the_walk_refuses_wrong_dtype():
    runner = _walk_runner()
    out = VerifyOutput(
        spec_mode=ACCEPT_MODE_FUSED_SAMPLE,
        accepted_token_ids=torch.zeros(MAX_NUM_REQS, 4, dtype=torch.int64),
        accepted_counts=torch.ones(MAX_NUM_REQS, dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="int32"):
        _walk(runner, out)


def test_padding_rows_commit_one_token_and_reach_propose():
    model = _fused_model()
    runner = _with_modes(_model_drafter_runner(model), model)
    _add_request(runner, "a")

    output = _step(runner, "a")

    assert [len(ids) for ids in output.sampled_token_ids] == [DRAFT_LEN + 1 - DRAFT_LEN]
    call = model.propose_calls[0]
    assert call["rows"] == MAX_NUM_REQS
    assert call["accepted_counts"].tolist() == [1] * MAX_NUM_REQS


# endregion


# region Stage S2: sampled rows on the fused mode


class _FusedTopkTarget(_TopkTarget):
    """The topk target that also serves ``fused_sample`` the way the device does.

    A fused verify accepts on its top-M support with the uniforms the runner
    passed in ``spec_sampling`` (``accept_sampled_support(uniforms=...)``), a
    greedy-only step by the argmax rule; it draws nothing itself."""

    accept_modes = (
        ACCEPT_MODE_ARGMAX_IDS,
        ACCEPT_MODE_LOGITS,
        ACCEPT_MODE_TOPK,
        ACCEPT_MODE_FUSED_SAMPLE,
    )

    def __init__(self):
        super().__init__()
        self.spec_sampling_calls = []

    def decode_forward(self, tokens, start_pos, spec_mode=None, **kwargs):
        if spec_mode != ACCEPT_MODE_FUSED_SAMPLE:
            return super().decode_forward(tokens, start_pos, spec_mode, **kwargs)
        rows, width = (int(s) for s in tokens.shape)
        ss = kwargs.get("spec_sampling")
        self.spec_sampling_calls.append(ss)
        self.verify_modes.append(spec_mode)
        logits = spec_logits(tokens, start_pos)
        values, ids = exact_support(logits, self.topk_candidates)
        argmax = logits.argmax(dim=-1).to(torch.int32)
        if ss is None:
            temperature = torch.zeros(rows)
            top_k = torch.ones(rows, dtype=torch.int32)
            top_p = torch.ones(rows)
        else:
            assert ss["uniforms"].shape == (rows, width + 1)
            temperature, top_k, top_p = ss["temperature"], ss["top_k"], ss["top_p"]
        sampling = SpecSamplingInputs(
            vocab_size=int(logits.shape[-1]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generators={},
        )
        result = accept_sampled_support(
            argmax,
            values,
            ids,
            tokens[:, 1:],
            kwargs["num_valid_drafts"],
            sampling,
            None,
            uniforms=None if ss is None else ss["uniforms"],
        )
        return VerifyOutput(
            spec_mode=spec_mode,
            argmax_ids=argmax,
            accepted_token_ids=result.committed_token_ids,
            accepted_counts=result.accepted_counts,
        )


def spec_logits(tokens, start_pos):
    from .sampled_target import rule_logits

    return rule_logits(tokens, start_pos)


def _fused_target_runner():
    return _topk_runner(_FusedTopkTarget())


SEEDED = SamplingParams(temperature=0.9, top_k=6, seed=11, ignore_eos=True)
UNSEEDED = SamplingParams(temperature=1.0, top_k=5, ignore_eos=True)


def _mode_after_first_step(runner, *req_ids, drafts=True):
    _sampled_step(runner, *req_ids)
    given = {r: _argmax_drafts(runner, r) for r in req_ids} if drafts else None
    return _build(runner, *req_ids, drafts=given)


def test_s2_mixed_batch_asks_for_fused_sample():
    model, runner = _fused_target_runner()
    _add_sampled_request(runner, "g", GREEDY, first_token=1)
    _add_sampled_request(runner, "s", SEEDED, first_token=9)
    mi = _mode_after_first_step(runner, "g", "s")
    assert mi.spec_mode == ACCEPT_MODE_FUSED_SAMPLE
    assert mi.perform_device_sampling is False
    ss = fused_spec_sampling(mi, runner._spec_plan)
    rows, width = mi.input_tokens.shape
    assert (
        ss["uniforms"].shape == (rows, width + 1)
        and ss["uniforms"].dtype == torch.float32
    )
    assert ss["temperature"].dtype == torch.float32 and ss["top_k"].dtype == torch.int32
    assert ss["top_p"].dtype == torch.float32 and ss["greedy"].dtype == torch.bool
    assert ss["greedy"].tolist() == [True, False] + [True] * (rows - 2)
    assert ss["temperature"][:2].tolist() == pytest.approx([0.0, 0.9])
    assert ss["top_k"].tolist()[:2] == [1, 6]
    assert not ss["uniforms"][2:].any()


def test_s2_greedy_only_step_sends_no_spec_sampling():
    model, runner = _fused_target_runner()
    _add_sampled_request(runner, "g", GREEDY)
    mi = _mode_after_first_step(runner, "g")
    assert mi.spec_mode == ACCEPT_MODE_FUSED_SAMPLE
    assert fused_spec_sampling(mi, runner._spec_plan) is None
    _sampled_step(runner, "g", drafts={"g": _argmax_drafts(runner, "g")})
    assert model.spec_sampling_calls[-1] is None


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"VLLM_TT_SPEC_FUSED_SAMPLED": "0"}, ACCEPT_MODE_TOPK),
        ({"VLLM_TT_SPEC_FUSED": "0"}, ACCEPT_MODE_TOPK),
        ({"VLLM_TT_SPEC_DEVICE_TOPK": "0"}, ACCEPT_MODE_LOGITS),
    ],
)
def test_s2_flags_keep_the_old_modes(monkeypatch, env, expected):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    model, runner = _fused_target_runner()
    _add_sampled_request(runner, "g", GREEDY, first_token=1)
    _add_sampled_request(runner, "s", SEEDED, first_token=9)
    assert _mode_after_first_step(runner, "g", "s").spec_mode == expected


def test_s2_a_row_the_support_cannot_serve_sends_the_step_to_logits():
    model, runner = _fused_target_runner()
    _add_sampled_request(runner, "s", SEEDED, first_token=1)
    _add_sampled_request(
        runner,
        "x",
        SamplingParams(temperature=1.0, top_k=0, ignore_eos=True),
        first_token=9,
    )
    assert _mode_after_first_step(runner, "s", "x").spec_mode == ACCEPT_MODE_LOGITS


def test_s2_a_seeded_row_with_no_drafts_keeps_topk():
    model, runner = _fused_target_runner()
    _add_sampled_request(runner, "s", SEEDED, first_token=1)
    _add_sampled_request(runner, "u", UNSEEDED, first_token=9)
    # no drafts offered: the seeded row has 0 valid drafts
    assert (
        _mode_after_first_step(runner, "s", "u", drafts=False).spec_mode
        == ACCEPT_MODE_TOPK
    )
    # unknown counts count as none for a seeded row
    assert runner._spec_mode_for_step(["s", "u"]) == ACCEPT_MODE_TOPK
    # an unseeded sampled row with no drafts is fine, and so are seeded rows with drafts
    assert (
        runner._spec_mode_for_step(["u"], torch.tensor([0])) == ACCEPT_MODE_FUSED_SAMPLE
    )
    assert (
        runner._spec_mode_for_step(["s", "u"], torch.tensor([2, 0]))
        == ACCEPT_MODE_FUSED_SAMPLE
    )
    assert (
        runner._spec_mode_for_step(["s", "u"], torch.tensor([0, 2])) == ACCEPT_MODE_TOPK
    )
    assert fused_sampled_enabled()


class _RecordTable:
    """A table recording the uniforms ``_walk_table`` feeds it.

    Every draft has probability 0."""

    def __init__(self):
        self.bonus = None
        self.corr = {}

    def sample(self, ids, u, exclude=None):
        if exclude is None:
            self.bonus = u.clone()
        else:
            for i, value in zip(ids.tolist(), u.tolist()):
                self.corr[i] = value
        return torch.zeros(len(ids), dtype=torch.int32)

    def prob_of(self, ids, tokens):
        return torch.zeros(len(ids))


@pytest.mark.parametrize("K", [3, 7])
def test_s2_predraw_equals_the_topk_walk_draws_and_states(monkeypatch, K):
    rows = 5
    valid = torch.tensor([K, 2, 0, 1, K], dtype=torch.int32)
    seeds = {0: 5, 1: 6, 3: 7, 4: 8}
    # the host walk's draws
    gens_walk = {r: torch.Generator().manual_seed(s) for r, s in seeds.items()}
    torch.manual_seed(99)
    table = _RecordTable()
    accepts = []
    real = spec_accept._uniform

    def recording(*args):
        accepts.append(real(*args))
        return accepts[-1]

    monkeypatch.setattr(spec_accept, "_uniform", recording)
    spec_accept._walk_table(
        table, torch.zeros(rows, K, dtype=torch.int32), valid, gens_walk, rows, K, 64
    )
    state_walk = torch.get_rng_state()
    # the runner's pre-draw, same seeds, same global state
    gens_pre = {r: torch.Generator().manual_seed(s) for r, s in seeds.items()}
    torch.manual_seed(99)
    uniforms = predraw_uniforms(gens_pre, rows, K, valid)
    assert torch.equal(torch.get_rng_state(), state_walk)
    for r in seeds:
        assert torch.equal(gens_pre[r].get_state(), gens_walk[r].get_state())
    clamp = lambda x: x.to(torch.float32).clamp(
        max=float(torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)))
    )
    assert torch.equal(uniforms[:, 0], clamp(table.bonus))
    assert torch.equal(uniforms[:, 1 : K + 1], clamp(accepts[0]))
    width = K + 1
    for r in range(rows):
        if int(valid[r]) > 0:  # the rows whose correction the walk used
            assert uniforms[r, K + 1] == clamp(torch.tensor(table.corr[r * width + 0]))


def test_s2_predraw_refuses_a_seeded_row_without_drafts():
    with pytest.raises(ValueError, match="carry no drafts"):
        predraw_uniforms(
            {1: torch.Generator().manual_seed(1)}, 2, 3, torch.tensor([2, 0])
        )


def test_s2_predraw_matches_the_tt_metal_copy_when_it_is_importable():
    root = Path(__file__).resolve().parents[3] / "tt-metal"
    if not root.is_dir():
        pytest.skip("no tt-metal checkout next to the plugin")
    sys.path.insert(0, str(root))
    try:
        theirs = importlib.import_module(
            "models.demos.blackhole.qwen36.tt.spec_fused_sampled"
        ).predraw_uniforms
    except Exception as exc:  # no ttnn here
        pytest.skip(f"tt-metal module not importable: {exc}")
    finally:
        sys.path.remove(str(root))
    valid = torch.tensor([3, 2, 1, 3])
    a = {r: torch.Generator().manual_seed(r + 1) for r in (0, 2)}
    b = {r: torch.Generator().manual_seed(r + 1) for r in (0, 2)}
    torch.manual_seed(5)
    x = predraw_uniforms(a, 4, 3, valid)
    sx = torch.get_rng_state()
    torch.manual_seed(5)
    y = theirs(b, 4, 3, valid)
    assert torch.equal(x, y) and torch.equal(sx, torch.get_rng_state())
    assert all(torch.equal(a[r].get_state(), b[r].get_state()) for r in a)


def _speculate_rows(runner, reqs, count):
    for _ in range(100):
        if all(len(_output(runner, r)) >= count for r in reqs):
            break
        _sampled_step(
            runner, *reqs, drafts={r: _argmax_drafts(runner, r) for r in reqs}
        )
    return {r: _output(runner, r)[:count] for r in reqs}


def test_s2_fused_commits_what_topk_commits_with_the_same_seeds(monkeypatch):
    def run(fused_sampled):
        monkeypatch.setenv("VLLM_TT_SPEC_FUSED_SAMPLED", "1" if fused_sampled else "0")
        model, runner = _fused_target_runner()
        torch.manual_seed(2024)
        _add_sampled_request(runner, "g", GREEDY, first_token=1)
        _add_sampled_request(runner, "s", SEEDED, first_token=9)
        _add_sampled_request(runner, "u", UNSEEDED, first_token=17)
        out = _speculate_rows(runner, ("g", "s", "u"), 40)
        return model, out, torch.get_rng_state()

    fused_model, fused_out, fused_rng = run(True)
    topk_model, topk_out, topk_rng = run(False)
    assert ACCEPT_MODE_FUSED_SAMPLE in set(fused_model.verify_modes)
    assert any(c is not None for c in fused_model.spec_sampling_calls)
    assert set(topk_model.verify_modes) <= {ACCEPT_MODE_TOPK, ACCEPT_MODE_ARGMAX_IDS}
    assert fused_out == topk_out
    # the global stream ends where the host walk leaves it
    assert torch.equal(fused_rng, topk_rng)


# endregion
