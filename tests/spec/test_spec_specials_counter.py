# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The production counter of special sampling features (``spec_specials``), on the host.

Pinned here:

- every feature is classified from the request's ``SamplingParams`` (and its
  images), and neutral values count as nothing;
- the counters are cumulative: requests at admission, steps once per feature a
  row carries, by path; a logits verify under one reason;
- the one log line (``TT specials: key=value ...``) round-trips through the
  parse script, which keeps the last line per worker pid and sums the ranks.

The counter reads no configuration and adds no capability.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin import async_decode
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.input_batch import build_cached_request_state
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_FUSED_SAMPLE,
    ACCEPT_MODE_LOGITS,
    ACCEPT_MODE_TOPK,
)
from vllm_tt_plugin.spec_specials import (
    FEATURES,
    SPECIALS_LINE_PREFIX,
    classify_params,
    logits_reason,
    reset_specials_stats,
    specials_report,
    specials_stats,
)

V = 152064


@pytest.fixture(autouse=True)
def _fresh_counters():
    reset_specials_stats()
    yield
    reset_specials_stats()


def _params(**kw):
    kw.setdefault("max_tokens", 64)
    return SamplingParams(**kw)


def _sampled(**kw):
    kw.setdefault("temperature", 0.8)
    return _params(**kw)


def _classify(params, **kw):
    return set(classify_params(params, V, **kw))


# ---- classification: one test per feature, from the request's own parameters


@pytest.mark.parametrize(
    "params, expected",
    [
        (_sampled(logprobs=0), {"logprobs", "topk_off_p1"}),
        (_sampled(logprobs=5, top_k=20), {"logprobs"}),
        (_sampled(prompt_logprobs=1, top_k=20), {"prompt_logprobs"}),
        (_sampled(top_p=0.9), {"topk_off_top_p"}),
        (_sampled(top_k=0, top_p=0.9), {"topk_off_top_p"}),
        (_sampled(top_k=V, top_p=0.9), {"topk_off_top_p"}),
        (_sampled(), {"topk_off_p1"}),
        (_sampled(top_k=-1), {"topk_off_p1"}),
        (_sampled(top_k=64), {"topk_ge64"}),
        (_sampled(top_k=64, top_p=0.9), {"topk_ge64"}),
        (_sampled(top_k=1000), {"topk_ge64"}),
        (_params(temperature=0.0), {"greedy"}),
        (_sampled(top_k=20, seed=7), {"seed"}),
        (
            _sampled(
                top_k=20,
                structured_outputs=StructuredOutputsParams(json={"type": "object"}),
            ),
            {"structured", "structured_json"},
        ),
        (
            _sampled(
                top_k=20, structured_outputs=StructuredOutputsParams(json_object=True)
            ),
            {"structured", "structured_json"},
        ),
        (
            _sampled(top_k=20, structured_outputs=StructuredOutputsParams(regex="a+")),
            {"structured", "structured_regex"},
        ),
        (
            _sampled(
                top_k=20,
                structured_outputs=StructuredOutputsParams(grammar="root ::= 'a'"),
            ),
            {"structured", "structured_grammar"},
        ),
        (
            _sampled(
                top_k=20, structured_outputs=StructuredOutputsParams(choice=["a", "b"])
            ),
            {"structured", "structured_grammar"},
        ),
        (_sampled(top_k=20, allowed_token_ids=[1, 2]), {"allowed_ids"}),
        (_sampled(top_k=20, bad_words=["x"]), {"bad_words"}),
        (_sampled(top_k=20, min_tokens=3), {"min_tokens"}),
        (_sampled(top_k=20, presence_penalty=0.5), {"penalty"}),
        (_sampled(top_k=20, frequency_penalty=-0.5), {"penalty"}),
        (_sampled(top_k=20, repetition_penalty=1.1), {"penalty"}),
        (_sampled(top_k=20, logit_bias={5: 1.0}), {"logit_bias"}),
        (_sampled(top_k=20, min_p=0.05), {"min_p"}),
        (_sampled(top_k=20, n=2), {"n_gt1"}),
        (_sampled(top_k=20, thinking_token_budget=100), {"think_budget"}),
    ],
)
def test_each_feature_is_classified(params, expected):
    assert _classify(params) == expected


def test_images_are_a_feature():
    assert _classify(_sampled(top_k=20), has_images=True) == {"image"}
    assert _classify(_sampled(top_k=20), has_images=False) == set()


def test_several_features_on_one_request():
    params = _sampled(logprobs=2, seed=3, presence_penalty=0.3, min_tokens=2, top_p=0.9)
    assert _classify(params) == {
        "logprobs",
        "seed",
        "penalty",
        "min_tokens",
        "topk_off_top_p",
    }


@pytest.mark.parametrize(
    "params",
    [
        _sampled(top_k=20),  # a finite top_k below 64, top_p 1
        _sampled(top_k=20, top_p=1.0),
        _sampled(
            top_k=20, top_p=0.9
        ),  # top_p with a finite top_k: not the certificate row
        _sampled(top_k=20, presence_penalty=0.0),
        _sampled(top_k=20, frequency_penalty=0.0),
        _sampled(top_k=20, repetition_penalty=1.0),
        _sampled(top_k=20, min_p=0.0),
        _sampled(top_k=20, min_tokens=0),
        _sampled(top_k=20, seed=None),
        _sampled(top_k=20, n=1),
        _sampled(top_k=20, logit_bias={}),
        _sampled(top_k=20, allowed_token_ids=[]),
        _sampled(top_k=20, bad_words=[]),
    ],
)
def test_neutral_values_are_not_special(params):
    assert classify_params(params, V) == ()


def test_greedy_ignores_top_k_and_top_p():
    assert _classify(_params(temperature=0.0, top_p=0.5, top_k=100)) == {"greedy"}


def test_every_classified_name_is_a_known_feature():
    params = _sampled(
        logprobs=1,
        prompt_logprobs=1,
        seed=1,
        min_p=0.1,
        n=2,
        min_tokens=1,
        presence_penalty=1.0,
        logit_bias={1: 1.0},
        allowed_token_ids=[1],
        thinking_token_budget=3,
        structured_outputs=StructuredOutputsParams(regex="a"),
    )
    assert set(classify_params(params, V, has_images=True)) <= set(FEATURES)


# ---- admission: counted once, kept on the request state


def _new_req(req_id, params, images=False):
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[object()] if images else [],
        sampling_params=params,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
    )


def test_admission_counts_each_request_once_and_cumulates():
    plain = build_cached_request_state(
        _new_req("a", _sampled(top_k=20)), supports_stage_inputs=False, vocab_size=V
    )
    special = build_cached_request_state(
        _new_req("b", _sampled(top_k=20, logprobs=1, seed=4), images=True),
        supports_stage_inputs=False,
        vocab_size=V,
    )
    assert plain.specials == ()
    assert set(special.specials) == {"logprobs", "seed", "image"}
    stats = specials_stats()
    assert stats["requests"] == 2
    assert stats["features"] == {"logprobs": 1, "seed": 1, "image": 1}
    build_cached_request_state(
        _new_req("c", _sampled(top_k=20, logprobs=1)),
        supports_stage_inputs=False,
        vocab_size=V,
    )
    stats = specials_stats()
    assert stats["requests"] == 3
    assert stats["features"]["logprobs"] == 2


# ---- steps: once per feature a row carries, by path


def _controller(requests):
    controller = object.__new__(TTAsyncDecodeController)
    controller.runner = SimpleNamespace(requests=requests)
    return controller


def _state(*features):
    return SimpleNamespace(specials=tuple(features))


def test_step_counts_once_per_feature_by_path():
    controller = _controller(
        {
            "a": _state("logprobs", "seed"),
            "b": _state("logprobs"),
            "c": _state(),
        }
    )
    step = SimpleNamespace(
        spec_mode=ACCEPT_MODE_FUSED_SAMPLE, row_req_ids=["a", "b", "c"]
    )
    controller._count_step_specials(step)
    controller._count_step_specials(step)
    controller._count_step_specials(
        SimpleNamespace(spec_mode=ACCEPT_MODE_LOGITS, row_req_ids=["b"])
    )
    controller._count_step_specials(
        SimpleNamespace(spec_mode=None, row_req_ids=["a", "missing"])
    )
    stats = specials_stats()
    assert stats["steps"] == 4
    assert stats["paths"] == {"fused": 2, "logits": 1, "plain": 1}
    assert stats["step_features"] == {
        ("logprobs", "fused"): 2,  # two rows carry it, one count per step
        ("seed", "fused"): 2,
        ("logprobs", "logits"): 1,
        ("logprobs", "plain"): 1,
        ("seed", "plain"): 1,
    }


@pytest.mark.parametrize(
    "mode, path",
    [
        (ACCEPT_MODE_FUSED_SAMPLE, "fused"),
        (ACCEPT_MODE_LOGITS, "logits"),
        (ACCEPT_MODE_TOPK, "topk"),
        (ACCEPT_MODE_ARGMAX_IDS, "ids"),
        (None, "plain"),
    ],
)
def test_every_path_has_a_name(mode, path):
    controller = _controller({"a": _state("seed")})
    controller._count_step_specials(SimpleNamespace(spec_mode=mode, row_req_ids=["a"]))
    assert specials_stats()["paths"] == {path: 1}


def test_a_step_without_row_ids_is_counted_with_no_feature():
    controller = _controller({})
    controller._count_step_specials(SimpleNamespace(spec_mode=None, row_req_ids=None))
    assert specials_stats()["steps"] == 1
    assert specials_stats()["step_features"] == {}


# ---- logits reasons


def _reason(params, output_len=0, **kw):
    args = {
        "candidates": 64,
        "vocab_size": V,
        "penalized": False,
        "grammar_ok": False,
        "device_logprobs": True,
        "device_top_p": True,
        "device_penalties": True,
    }
    args.update(kw)
    return logits_reason(params, output_len, **args)


def test_logits_reason_of_each_refusal():
    assert _reason(_sampled(top_k=20), candidates=None) == "no_candidates"
    assert (
        _reason(_sampled(top_k=20), penalized=True, device_penalties=False) == "penalty"
    )
    assert _reason(_sampled(top_k=20, logprobs=1), penalized=True) == "penalty"
    assert _reason(_sampled(top_k=100)) == "topk_wide"
    assert _reason(_sampled()) == "topk_off_p1"
    assert _reason(_sampled(top_p=0.9), device_top_p=False) == "topk_off_no_cert"
    assert _reason(_sampled(top_k=20, logprobs=1), device_logprobs=False) == "logprobs"
    assert _reason(_sampled(top_k=20, logprobs=21)) == "logprobs"
    grammar = StructuredOutputsParams(regex="a+")
    assert _reason(_sampled(top_k=20, structured_outputs=grammar)) == "structured"
    assert (
        _reason(_sampled(top_k=20, structured_outputs=grammar), grammar_ok=True)
        == "other"
    )
    assert _reason(_sampled(top_k=20, allowed_token_ids=[1])) == "token_filter"
    stop = _sampled(top_k=20, min_tokens=5, ignore_eos=False)
    stop._all_stop_token_ids = {2}
    assert _reason(stop, output_len=1) == "min_tokens"
    assert _reason(stop, output_len=5) == "other"
    assert _reason(_sampled(top_k=20)) == "other"


def _runner_double(*, certifiable=(), drafts_seeded=False, **attrs):
    requests = attrs.pop("requests")
    ns = SimpleNamespace(
        requests=requests,
        vocab_size=V,
        model_config=SimpleNamespace(logprobs_mode="raw_logprobs"),
        _spec_accept_modes=(
            ACCEPT_MODE_LOGITS,
            ACCEPT_MODE_TOPK,
            ACCEPT_MODE_FUSED_SAMPLE,
        ),
        _spec_plan=SimpleNamespace(topk_candidates=64),
        _spec_device_logprobs=True,
        _spec_device_top_p=True,
        _spec_device_penalties=True,
        _request_is_argmax_certifiable=lambda req_id, grammar_ok=False: req_id
        in certifiable,
        _request_has_penalties=lambda req_id: False,
        _seeded_row_without_drafts=lambda rows, valid: drafts_seeded,
    )
    ns.__dict__.update(attrs)
    return ns


def _count_runner_reason(monkeypatch, runner, rows):
    monkeypatch.setenv("VLLM_TT_SPEC_DEVICE_TOPK", "1")
    monkeypatch.delenv("VLLM_TT_SPEC_FUSED", raising=False)
    TTModelRunner._count_logits_step_reason(runner, rows, None, None)
    return specials_stats()["logits_reasons"]


def test_runner_names_the_first_refused_row(monkeypatch):
    requests = {
        "g": _state(),
        "w": _state(),
        "s": _state(),
    }
    requests["g"] = SimpleNamespace(
        sampling_params=_params(temperature=0.0), output_token_ids=[]
    )
    requests["w"] = SimpleNamespace(
        sampling_params=_sampled(top_k=200), output_token_ids=[]
    )
    requests["s"] = SimpleNamespace(
        sampling_params=_sampled(top_k=20, logprobs=50), output_token_ids=[]
    )
    runner = _runner_double(requests=requests, certifiable={"g"})
    # the certifiable greedy row is skipped; the first refused row decides
    assert _count_runner_reason(monkeypatch, runner, ["g", "w", "s"]) == {
        "topk_wide": 1
    }
    reset_specials_stats()
    assert _count_runner_reason(monkeypatch, runner, ["g", "s"]) == {"logprobs": 1}


def test_runner_names_launch_and_seed_reasons(monkeypatch):
    requests = {
        "a": SimpleNamespace(sampling_params=_sampled(top_k=20), output_token_ids=[])
    }
    runner = _runner_double(requests=requests, drafts_seeded=True)
    assert _count_runner_reason(monkeypatch, runner, ["a"]) == {"seed_no_draft": 1}
    reset_specials_stats()
    monkeypatch.setenv("VLLM_TT_SPEC_FUSED", "0")
    TTModelRunner._count_logits_step_reason(runner, ["a"], None, None)
    assert specials_stats()["logits_reasons"] == {"launch_off": 1}


# ---- the log line and the parse script


def _load_parser():
    path = pathlib.Path(__file__).with_name("_specials_parse.py")
    spec = importlib.util.spec_from_file_location("specials_parse", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fill(requests_by_features, steps):
    reset_specials_stats()
    for features in requests_by_features:
        build_cached_request_state(
            _new_req("r", features), supports_stage_inputs=False, vocab_size=V
        )
    states = {"a": _state("logprobs", "seed"), "b": _state()}
    controller = _controller(states)
    for mode in steps:
        controller._count_step_specials(
            SimpleNamespace(spec_mode=mode, row_req_ids=["a", "b"])
        )


def test_report_line_is_one_line_and_stable():
    _fill(
        [_sampled(top_k=20, logprobs=1), _sampled(top_k=20)], [ACCEPT_MODE_FUSED_SAMPLE]
    )
    line = specials_report()
    assert "\n" not in line
    assert line == (
        f"{SPECIALS_LINE_PREFIX} reqs=2 steps=1 req_logprobs=1 path_fused=1 "
        "logprobs@fused=1 seed@fused=1"
    )


def test_line_round_trips_through_the_parse_script():
    parse = _load_parser()
    _fill(
        [_sampled(top_k=20, logprobs=1), _sampled(top_p=0.9), _sampled(top_k=20)],
        [ACCEPT_MODE_FUSED_SAMPLE, ACCEPT_MODE_FUSED_SAMPLE, ACCEPT_MODE_LOGITS],
    )
    from vllm_tt_plugin.spec_specials import count_logits_reason

    count_logits_reason("topk_wide")
    prefix = "(Worker pid=1003) INFO 10-28 20:00:00 tt/async_decode.py:1"
    line = f"{prefix} {specials_report()}"
    pid, counts = parse.parse_line(line)
    assert pid == 1003
    assert counts == {
        "reqs": 3,
        "steps": 3,
        "req_logprobs": 1,
        "req_topk_off_top_p": 1,
        "path_fused": 2,
        "path_logits": 1,
        "logprobs@fused": 2,
        "seed@fused": 2,
        "logprobs@logits": 1,
        "seed@logits": 1,
        "why_topk_wide": 1,
    }
    assert parse.parse_line("INFO TT submissions: 128 ordinary decode") is None


def test_parse_keeps_the_last_line_per_pid_and_sums_ranks():
    parse = _load_parser()
    lines = [
        "(Worker pid=1) INFO x TT specials: reqs=1 steps=2 req_seed=1 path_fused=2",
        "(Worker pid=2) INFO x TT specials: reqs=4 steps=8 req_seed=2 path_fused=8",
        "noise line",
        "(Worker pid=1) INFO x TT specials: reqs=3 steps=6 req_seed=2 path_fused=5 "
        "path_logits=1 why_topk_wide=1",
    ]
    last = parse.last_per_pid(lines)
    assert set(last) == {1, 2}
    total = parse.total(last)
    assert total["reqs"] == 7
    assert total["steps"] == 14
    assert total["req_seed"] == 4
    assert total["path_fused"] == 13
    assert total["path_logits"] == 1
    text = parse.table(last)
    assert "workers=2 requests=7 steps=14" in text
    assert "req_seed" in text and "why_topk_wide" in text
    assert parse.table({}) == "no TT specials lines"


def test_the_line_comes_with_the_submissions_line(monkeypatch):
    lines = []
    monkeypatch.setattr(
        async_decode.logger, "info", lambda fmt, *args: lines.append(fmt % args)
    )
    monkeypatch.setattr(async_decode, "_SUBMISSION_LOG_INTERVAL", 1)
    controller = _controller({"a": _state("seed")})
    for name in (
        "_ordinary_decode_submissions",
        "_verify_submissions",
        "_logits_verify_submissions",
        "_topk_verify_submissions",
        "_fused_verify_submissions",
        "_verify_rows",
        "_valid_drafts_submitted",
        "_accepted_drafts",
        "_overlapped_submissions",
    ):
        setattr(controller, name, 0)
    controller._chain_enabled = False
    controller.chain_enabled = lambda: False
    controller.count_decode_submission(
        SimpleNamespace(spec_mode=None, row_req_ids=["a"], num_valid_drafts=None)
    )
    assert any(line.startswith("TT submissions:") for line in lines)
    specials = [line for line in lines if line.startswith(SPECIALS_LINE_PREFIX)]
    assert specials == [
        f"{SPECIALS_LINE_PREFIX} reqs=0 steps=1 path_plain=1 seed@plain=1"
    ]
