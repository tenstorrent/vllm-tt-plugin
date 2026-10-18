# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import sys

import pytest

from vllm_tt_plugin import video_guard

# A helper that segfaults as soon as it is given work, like the OpenCV decoder
# does on a bad video.
_CRASHING_WORKER = [
    sys.executable,
    "-c",
    "import os, signal, sys; sys.stdin.buffer.read(1); "
    "os.kill(os.getpid(), signal.SIGSEGV)",
]
# A helper that answers every job with the decoded payload size.
_ECHO_WORKER = [
    sys.executable,
    "-c",
    "import os, sys\n"
    "from vllm_tt_plugin import video_guard as g\n"
    "out = os.fdopen(os.dup(1), 'wb'); os.dup2(2, 1)\n"
    "while True:\n"
    "    try:\n"
    "        data, n, kw = g._recv(sys.stdin.buffer)\n"
    "    except EOFError:\n"
    "        break\n"
    "    g._send(out, ('err', ValueError('bad')) if data == b'bad' "
    "else ('ok', len(data)))\n",
]


def _run(pool, data=b"abc"):
    return pool.run(data, 4, {}, timeout=30)


def test_decoder_crash_fails_only_that_request(monkeypatch):
    monkeypatch.setattr(video_guard, "_WORKER_ARGV", _CRASHING_WORKER)
    pool = video_guard._Pool(2)
    with pytest.raises(ValueError, match="decoder crashed"):
        _run(pool)

    # The pool keeps serving after a crash.
    monkeypatch.setattr(video_guard, "_WORKER_ARGV", _ECHO_WORKER)
    assert _run(pool) == 3


def test_worker_is_reused_and_decode_errors_keep_it_alive(monkeypatch):
    monkeypatch.setattr(video_guard, "_WORKER_ARGV", _ECHO_WORKER)
    pool = video_guard._Pool(1)
    assert _run(pool) == 3
    worker = pool._idle.queue[-1]
    with pytest.raises(ValueError, match="bad"):
        _run(pool, b"bad")
    assert _run(pool) == 3
    assert pool._idle.queue[-1] is worker


def test_install_wraps_molmo2_backend_once(monkeypatch):
    backend = pytest.importorskip("vllm.multimodal.video").Molmo2VideoBackend
    original = backend.__dict__["load_bytes"]
    monkeypatch.delenv("TT_VIDEO_GUARD", raising=False)
    monkeypatch.delenv(video_guard._WORKER_ENV, raising=False)
    try:
        video_guard.install_molmo2_video_guard()
        wrapped = backend.__dict__["load_bytes"]
        assert wrapped is not original
        video_guard.install_molmo2_video_guard()
        assert backend.__dict__["load_bytes"] is wrapped
    finally:
        backend.load_bytes = original
