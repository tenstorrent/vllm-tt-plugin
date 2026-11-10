"""VLLM_TT_PREFIX_DEBUG=1 logs one INFO line per ``_get_local_prefix_cache_hit`` call (request id, num_tokens, vLLM's raw num_local); flag off = silent."""

import types

from vllm_tt_plugin import scheduler as S
from vllm_tt_plugin.scheduler import TTScheduler


class _Fake(TTScheduler):
    def __init__(self):
        self._prefix_hit_policy = ""
        self._prefill_chunk_alignment = 0


def _call(monkeypatch, num_local):
    base = S.TTScheduler.__mro__[1]
    monkeypatch.setattr(base, "_get_local_prefix_cache_hit", lambda self, req: ("blocks", num_local, 0, False))
    lines = []
    monkeypatch.setattr(S.logger, "info", lambda msg, *a, **k: lines.append(msg % a))
    req = types.SimpleNamespace(request_id="r1", num_tokens=1234)
    out = _Fake()._get_local_prefix_cache_hit(req)
    return out, lines


def test_silent_when_off(monkeypatch):
    monkeypatch.delenv("VLLM_TT_PREFIX_DEBUG", raising=False)
    out, lines = _call(monkeypatch, 512)
    assert out == ("blocks", 512, 0, False) and lines == []
    monkeypatch.setenv("VLLM_TT_PREFIX_DEBUG", "0")
    assert _call(monkeypatch, 512)[1] == []


def test_one_line_per_call_when_on(monkeypatch):
    monkeypatch.setenv("VLLM_TT_PREFIX_DEBUG", "1")
    out, lines = _call(monkeypatch, 0)  # cold prompt: still logged
    assert out == ("blocks", 0, 0, False)
    assert lines == ["TT prefix debug: req=r1 num_tokens=1234 num_local=0"]
    assert _call(monkeypatch, 4096)[1] == ["TT prefix debug: req=r1 num_tokens=1234 num_local=4096"]
