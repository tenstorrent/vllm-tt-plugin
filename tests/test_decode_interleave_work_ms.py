# SPDX-License-Identifier: Apache-2.0
"""Hy4 chunked prefill: work_ms burst policy + live switch file of TTDecodeInterleavePolicy (default off: the step-count policy is unchanged).
Runs without vllm: the policy class is extracted from scheduler.py by ast and exec'ed with a stub config reader."""
import ast
import os
import time
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src" / "vllm_tt_plugin" / "scheduler.py"


def load_policy():
    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "_env_float") or (isinstance(n, ast.ClassDef) and n.name == "TTDecodeInterleavePolicy")]
    ns = {
        "os": os,
        "time": time,
        "get_tt_decode_interleave_config": lambda cfg: cfg.tt,
        "VllmConfig": object,
    }
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["TTDecodeInterleavePolicy"]


P = load_policy()


def mk(enabled=True, ps=2, ds=1):
    return P(SimpleNamespace(tt=(enabled, ps, ds)))


def phases(pol, n, sleep_ms=0.0):
    out = ""
    for _ in range(n):
        d = pol.wants_decode_step(has_pending_prefill=True, has_running_decode=True)
        pol.record_step(is_decode=d, prefill_pending=True)
        out += "D" if d else "P"
        if not d and sleep_ms:
            time.sleep(sleep_ms / 1000)
    return out


def test_default_is_the_step_policy(monkeypatch):
    for k in ("TT_DECODE_INTERLEAVE_WORK_MS", "TT_DECODE_INTERLEAVE_FILE", "TT_DECODE_INTERLEAVE_BURST"):
        monkeypatch.delenv(k, raising=False)
    assert phases(mk(), 9) == "PPDPPDPPD"


def test_work_ms_burst(monkeypatch):
    monkeypatch.setenv("TT_DECODE_INTERLEAVE_WORK_MS", "30")
    monkeypatch.setenv("TT_DECODE_INTERLEAVE_BURST", "3")
    monkeypatch.delenv("TT_DECODE_INTERLEAVE_FILE", raising=False)
    pol = mk()
    s = phases(pol, 14, sleep_ms=12)  # a prefill step takes ~12 ms: burst after 3 of them (36 ms >= 30)
    assert s.startswith("PPP") and "DDD" in s, s
    assert pol.bursts >= 1 and pol.work_decode_steps >= 3
    # a burst is exactly `burst` decode steps, then prefill resumes
    i = s.index("DDD")
    assert s[i + 3] == "P" or len(s) == i + 3


def test_no_burst_before_work_budget(monkeypatch):
    monkeypatch.setenv("TT_DECODE_INTERLEAVE_WORK_MS", "100000")
    monkeypatch.delenv("TT_DECODE_INTERLEAVE_FILE", raising=False)
    assert phases(mk(), 6) == "PPPPPP"


def test_no_decoders_no_burst(monkeypatch):
    monkeypatch.setenv("TT_DECODE_INTERLEAVE_WORK_MS", "1")
    pol = mk()
    time.sleep(0.01)
    assert not pol.wants_decode_step(has_pending_prefill=True, has_running_decode=False)
    assert not pol.wants_decode_step(has_pending_prefill=False, has_running_decode=True)


def test_live_file_switch(monkeypatch, tmp_path):
    f = tmp_path / "il"
    monkeypatch.setenv("TT_DECODE_INTERLEAVE_FILE", str(f))
    monkeypatch.delenv("TT_DECODE_INTERLEAVE_WORK_MS", raising=False)
    pol = mk()
    f.write_text("off")
    pol._file_read = 0
    assert phases(pol, 6) == "PPPPPP"  # off: prefill never yields
    f.write_text("work_ms=5,burst=2")
    pol._file_read = 0
    s = phases(pol, 12, sleep_ms=8)
    assert "DD" in s, s
    f.write_text("0")
    pol._file_read = 0
    assert "D" not in phases(pol, 5)


def test_decode_with_nothing_pending_clears(monkeypatch):
    monkeypatch.setenv("TT_DECODE_INTERLEAVE_WORK_MS", "10")
    pol = mk()
    pol.record_step(is_decode=False, prefill_pending=True)
    time.sleep(0.02)
    pol.record_step(is_decode=True, prefill_pending=False)
    assert pol._prefill_ms == 0.0 and pol._decode_run == 0
