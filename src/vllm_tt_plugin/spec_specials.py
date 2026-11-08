# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A production counter of "special" sampling features and the path that served them.

Every request that carries a feature the fused path treats differently from a
plain sampled or greedy request is counted once, when it is admitted
(``build_cached_request_state``). Every decode step is counted once per feature
that at least one of its rows carries, under the path that served the step
(``fused``, ``logits``, ``topk``, ``ids``, ``plain`` for an ordinary decode).
A logits-mode verify is also counted under the reason it was not served by the
fused path (``logits_reason``).

Nothing here reads configuration: only the request's own ``SamplingParams``
and the step's rows. The counters are module-level, like
``spec_device_penalties``, because the submission line reads them from the
controller. One line per worker, on the cadence of the ``TT submissions`` line,
cumulative since start (``specials_report``); ``tests/spec/_specials_parse.py`` in the
research scratch reads it.

Not visible to a worker, so not counted: ``n`` above 1 (the frontend splits it
into child requests that carry n=1; ``n_gt1`` stays at 0 unless a caller sends
a prebuilt request), ``tool_choice`` (it arrives as a structured output, counted
under ``structured``), chat-template thinking (only ``thinking_token_budget`` is
a parameter), and a ``logit_bias`` that the speculative validator refused (that
runs in the API server process).
"""

from __future__ import annotations

import threading
from typing import Any

from vllm_tt_plugin.spec_decode import MAX_DEVICE_LOGPROBS

# Stable names; the report and the parse script key on them. Order is the
# order of the report.
FEATURES = (
    "logprobs",
    "prompt_logprobs",
    "topk_off_top_p",  # sampled, top_k off, top_p < 1: the certificate row
    "topk_off_p1",  # sampled, top_k off, top_p == 1
    "topk_ge64",  # sampled, 64 <= top_k < vocab
    "greedy",
    "seed",
    "structured",
    "structured_json",
    "structured_regex",
    "structured_grammar",
    "allowed_ids",
    "bad_words",
    "min_tokens",
    "penalty",
    "logit_bias",
    "min_p",
    "n_gt1",
    "think_budget",
    "image",
)

# The paths a step can take, in report order.
PATHS = ("fused", "logits", "topk", "ids", "plain")

_MIN_WIDE_TOP_K = 64
_GREEDY_BELOW = 1e-5


def classify_params(
    params: Any, vocab_size: int | None, *, has_images: bool = False
) -> tuple[str, ...]:
    """The special features one request carries, from its ``SamplingParams``.

    Neutral values count as nothing: penalties at 0 / 1, ``min_p`` 0,
    ``min_tokens`` 0, a finite ``top_k`` below 64, top_p 1 with a finite top_k.
    A greedy request ignores top_k and top_p, so only the sampled ones are
    classified by them. Without a vocabulary size only ``top_k <= 0`` is off.
    """
    vocab_size = 2**31 if vocab_size is None else int(vocab_size)
    found: list[str] = []
    if params.logprobs is not None:
        found.append("logprobs")
    if params.prompt_logprobs is not None:
        found.append("prompt_logprobs")
    if float(params.temperature) < _GREEDY_BELOW:
        found.append("greedy")
    else:
        top_k = int(params.top_k)
        if 0 < top_k < vocab_size:
            if top_k >= _MIN_WIDE_TOP_K:
                found.append("topk_ge64")
        elif float(params.top_p) < 1.0:
            found.append("topk_off_top_p")
        else:
            found.append("topk_off_p1")
    if params.seed is not None and int(params.seed) != -1:
        found.append("seed")
    structured = getattr(params, "structured_outputs", None)
    if structured is not None:
        found.append("structured")
        if structured.json is not None or structured.json_object:
            found.append("structured_json")
        elif structured.regex is not None:
            found.append("structured_regex")
        else:
            found.append("structured_grammar")
    if params.allowed_token_ids:
        found.append("allowed_ids")
    if params.bad_words:
        found.append("bad_words")
    if params.min_tokens > 0:
        found.append("min_tokens")
    if (
        float(params.presence_penalty) != 0.0
        or float(params.frequency_penalty) != 0.0
        or float(params.repetition_penalty) != 1.0
    ):
        found.append("penalty")
    if params.logit_bias:
        found.append("logit_bias")
    if float(params.min_p) > 0.0:
        found.append("min_p")
    if int(params.n or 1) > 1:
        found.append("n_gt1")
    if getattr(params, "thinking_token_budget", None) is not None:
        found.append("think_budget")
    if has_images:
        found.append("image")
    return tuple(found)


def logits_reason(
    params: Any,
    output_len: int,
    *,
    candidates: int | None,
    vocab_size: int,
    penalized: bool,
    grammar_ok: bool,
    device_logprobs: bool,
    device_top_p: bool,
    device_penalties: bool,
) -> str:
    """The first reason a fused step could not carry this row (its logits step).

    Mirrors ``params_fused_servable`` in the order it tests, one name per test,
    so the breakdown adds up to the logits steps. ``other`` is a row that rule
    accepts (the step was held in logits by a launch setting or another row).
    """
    if candidates is None:
        return "no_candidates"
    if penalized and not (device_penalties and params.logprobs is None):
        return "penalty"
    if float(params.temperature) >= _GREEDY_BELOW:
        top_k = int(params.top_k)
        if 0 < top_k < int(vocab_size):
            if top_k > candidates - 1:
                return "topk_wide"
        elif not (device_top_p and float(params.top_p) < 1.0):
            return "topk_off_p1" if float(params.top_p) >= 1.0 else "topk_off_no_cert"
    if params.logprobs is not None and not (
        device_logprobs and 0 <= int(params.logprobs) <= MAX_DEVICE_LOGPROBS
    ):
        return "logprobs"
    if params.structured_outputs is not None and not grammar_ok:
        return "structured"
    if params.allowed_token_ids or params.bad_words_token_ids:
        return "token_filter"
    if params.min_tokens > output_len and params.all_stop_token_ids:
        return "min_tokens"
    return "other"


class _SpecialsStats:
    """Requests per feature, steps per feature and path, logits steps per reason."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.requests = 0
            self.by_feature: dict[str, int] = {}
            self.steps = 0
            self.step_paths: dict[str, int] = {}
            self.step_features: dict[tuple[str, str], int] = {}
            self.logits_reasons: dict[str, int] = {}

    def add_request(self, features: tuple[str, ...]) -> None:
        with self._lock:
            self.requests += 1
            for name in features:
                self.by_feature[name] = self.by_feature.get(name, 0) + 1

    def add_step(self, path: str, features: frozenset[str] | set[str]) -> None:
        with self._lock:
            self.steps += 1
            self.step_paths[path] = self.step_paths.get(path, 0) + 1
            for name in features:
                key = (name, path)
                self.step_features[key] = self.step_features.get(key, 0) + 1

    def add_logits_reason(self, reason: str) -> None:
        with self._lock:
            self.logits_reasons[reason] = self.logits_reasons.get(reason, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": self.requests,
                "features": dict(self.by_feature),
                "steps": self.steps,
                "paths": dict(self.step_paths),
                "step_features": dict(self.step_features),
                "logits_reasons": dict(self.logits_reasons),
            }


_STATS = _SpecialsStats()


def specials_stats() -> dict[str, Any]:
    """A copy of the counters (``requests``, ``features``, ``steps``, ``paths``,
    ``step_features`` keyed ``(feature, path)``, ``logits_reasons``)."""
    return _STATS.snapshot()


def reset_specials_stats() -> None:
    _STATS.reset()


def count_request_specials(features: tuple[str, ...]) -> None:
    """One admitted request and the features it carries."""
    _STATS.add_request(features)


def count_step_specials(path: str, features: frozenset[str] | set[str]) -> None:
    """One decode step on ``path`` and the features its rows carry."""
    _STATS.add_step(path, features)


def count_logits_reason(reason: str) -> None:
    """One logits-mode verify and why the fused path did not take it."""
    _STATS.add_logits_reason(reason)


SPECIALS_LINE_PREFIX = "TT specials:"


def specials_report() -> str:
    """The one-line report: ``TT specials: key=value ...``, cumulative.

    Keys: ``reqs``, ``steps``, ``req_<feature>``, ``path_<path>``,
    ``<feature>@<path>`` (steps carrying the feature, by path) and
    ``why_<reason>`` (logits steps by reason). A zero count is left out, so the
    line grows only with what production sends.
    """
    stats = _STATS.snapshot()
    parts = [f"reqs={stats['requests']}", f"steps={stats['steps']}"]
    parts += [
        f"req_{name}={stats['features'][name]}"
        for name in FEATURES
        if stats["features"].get(name)
    ]
    parts += [
        f"path_{path}={stats['paths'][path]}"
        for path in PATHS
        if stats["paths"].get(path)
    ]
    parts += [
        f"{name}@{path}={stats['step_features'][(name, path)]}"
        for name in FEATURES
        for path in PATHS
        if stats["step_features"].get((name, path))
    ]
    parts += [
        f"why_{reason}={count}"
        for reason, count in sorted(stats["logits_reasons"].items())
    ]
    return f"{SPECIALS_LINE_PREFIX} " + " ".join(parts)
