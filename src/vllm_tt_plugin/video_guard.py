# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Crash isolation for Molmo2 video decoding.

vLLM decodes request videos with OpenCV inside the API server process. Some
files make the OpenCV/FFmpeg decoder segfault, which kills the whole API server
and leaves the engine core orphaned on the devices. This module runs
``Molmo2VideoBackend.load_bytes`` in a small pool of persistent helper
processes so a decoder crash fails only the one request (HTTP 400).

Set ``TT_VIDEO_GUARD=0`` to disable, ``TT_VIDEO_GUARD_TIMEOUT`` (seconds,
default 900) to bound a single decode, ``TT_VIDEO_GUARD_WORKERS`` (default 4)
to bound the number of helper processes.
"""

import contextlib
import os
import pickle
import queue
import struct
import subprocess
import sys
import threading

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

_WORKER_ENV = "TT_VIDEO_GUARD_WORKER"
_HEADER = struct.Struct("<Q")
_WORKER_ARGV = [sys.executable, "-m", "vllm_tt_plugin.video_guard"]


def _send(stream, obj) -> None:
    payload = pickle.dumps(obj, protocol=5)
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _recv(stream):
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        raise EOFError
    (size,) = _HEADER.unpack(header)
    payload = stream.read(size)
    if len(payload) < size:
        raise EOFError
    return pickle.loads(payload)


class _Worker:
    def __init__(self) -> None:
        env = dict(os.environ, **{_WORKER_ENV: "1"})
        self.proc = subprocess.Popen(
            _WORKER_ARGV,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )

    def run(self, data, num_frames, kwargs, timeout):
        timer = threading.Timer(timeout, self.kill)
        timer.start()
        try:
            _send(self.proc.stdin, (data, num_frames, kwargs))
            status, value = _recv(self.proc.stdout)
        except (EOFError, BrokenPipeError, OSError):
            code = self.proc.wait()
            raise ValueError(
                f"Video decoder crashed (exit code {code}); the video is "
                "corrupt or unsupported"
            ) from None
        finally:
            timer.cancel()
        if status == "err":
            raise value
        return value

    def kill(self) -> None:
        with contextlib.suppress(OSError):
            self.proc.kill()

    def close(self) -> None:
        self.kill()
        self.proc.wait()
        for stream in (self.proc.stdin, self.proc.stdout):
            with contextlib.suppress(OSError):
                stream.close()


class _Pool:
    def __init__(self, size: int) -> None:
        self._idle: queue.LifoQueue = queue.LifoQueue()
        self._slots = threading.Semaphore(size)

    def run(self, data, num_frames, kwargs, timeout):
        with self._slots:
            try:
                worker = self._idle.get_nowait()
            except queue.Empty:
                worker = _Worker()
            try:
                result = worker.run(data, num_frames, kwargs, timeout)
            except ValueError as exc:
                if "decoder crashed" in str(exc):
                    logger.warning("%s", exc)
                    worker.close()
                    raise
                self._idle.put(worker)
                raise
            except BaseException:
                worker.close()
                raise
            self._idle.put(worker)
            return result


def install_molmo2_video_guard() -> None:
    """Route ``Molmo2VideoBackend.load_bytes`` through helper processes."""
    if os.environ.get(_WORKER_ENV) or os.environ.get("TT_VIDEO_GUARD") == "0":
        return
    try:
        from vllm.multimodal.video import Molmo2VideoBackend
    except ImportError:
        return
    if getattr(Molmo2VideoBackend.load_bytes, "_tt_video_guard", False):
        return

    pool = _Pool(int(os.environ.get("TT_VIDEO_GUARD_WORKERS", "4")))
    timeout = float(os.environ.get("TT_VIDEO_GUARD_TIMEOUT", "900"))

    def load_bytes(cls, data, num_frames=-1, **kwargs):
        return pool.run(data, num_frames, kwargs, timeout)

    load_bytes._tt_video_guard = True
    Molmo2VideoBackend.load_bytes = classmethod(load_bytes)
    logger.info("Molmo2 video decoding isolated in helper processes")


def _worker_main() -> None:
    # Keep the protocol pipe private so library output on fd 1 cannot corrupt it.
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    inp = sys.stdin.buffer

    from vllm.multimodal.video import Molmo2VideoBackend

    original = Molmo2VideoBackend.load_bytes.__func__
    while True:
        try:
            data, num_frames, kwargs = _recv(inp)
        except EOFError:
            return
        try:
            result = ("ok", original(Molmo2VideoBackend, data, num_frames, **kwargs))
        except Exception as exc:  # noqa: BLE001
            try:
                pickle.dumps(exc)
            except Exception:  # noqa: BLE001
                exc = RuntimeError(repr(exc))
            result = ("err", exc)
        del data
        _send(out, result)


if __name__ == "__main__":
    _worker_main()
