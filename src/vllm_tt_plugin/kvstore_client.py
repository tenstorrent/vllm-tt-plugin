# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Client of ttserve-kvstore, the per-host shared-memory KV page store (github.com/zhenghaoniTT/ttserve,
crates/ttserve-kvstore).

The store owns the slots of shared-memory files ("pools"); this client asks it for slots under 32-byte keys and maps
the pools' files, so the caller copies KV pages straight between the device and ``Pool.base + offset``. Protocol:
newline-delimited JSON over a Unix socket, one request, one response. This module is the whole boundary between the
TT KV connector and the store: another store (an LMCache server) would replace this class.

Thread-safe (one lock per client). Any socket error closes the connection and raises KVStoreError; the next call
reconnects. The store drops a connection's reservations and leases when it closes.
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import socket
import threading
from dataclasses import dataclass


class KVStoreError(RuntimeError):
    pass


@dataclass
class Pool:
    name: str
    path: str
    num_slots: int
    slot_bytes: int
    mm: mmap.mmap
    base: int  # address of the mapping


class KVStoreClient:
    def __init__(self, socket_path: str, timeout_s: float = 10.0) -> None:
        self.socket_path = socket_path
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._rfile = None
        self._lock = threading.Lock()
        self._pools: dict[str, Pool] = {}

    # -- transport ---------------------------------------------------------------------------------------------
    def _connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout_s)
        s.connect(self.socket_path)
        self._sock = s
        self._rfile = s.makefile("rb")

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        finally:
            self._sock = None
            self._rfile = None

    def call(self, timeout_s: float | None = None, **req) -> dict:
        with self._lock:
            try:
                if self._sock is None:
                    self._connect()
                self._sock.settimeout(timeout_s or self.timeout_s)
                self._sock.sendall((json.dumps(req) + "\n").encode())
                line = self._rfile.readline()
                if not line:
                    raise KVStoreError("store closed the connection")
                resp = json.loads(line)
            except (OSError, ValueError) as e:
                self._close()
                raise KVStoreError(f"{req.get('op')}: {e}") from e
        if not resp.get("ok"):
            raise KVStoreError(f"{req.get('op')}: {resp.get('error')}")
        return resp

    # -- pools ---------------------------------------------------------------------------------------------------
    def open_pool(self, name: str, slot_bytes: int, capacity_bytes: int) -> Pool:
        # Creating a pool reserves its memory up front: tens of seconds for tens of GiB.
        r = self.call(op="open_pool", pool=name, slot_bytes=slot_bytes, capacity_bytes=capacity_bytes, timeout_s=600)
        old = self._pools.get(name)
        if old is not None and old.path == r["path"] and old.num_slots == r["num_slots"]:
            return old
        size = r["num_slots"] * r["slot_bytes"]
        fd = os.open(r["path"], os.O_RDWR)
        try:
            mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        pool = Pool(name, r["path"], r["num_slots"], r["slot_bytes"], mm, base)
        self._pools[name] = pool
        return pool

    # -- entries -------------------------------------------------------------------------------------------------
    @staticmethod
    def _hex(keys: list[bytes]) -> list[str]:
        return [k.hex() for k in keys]

    def lookup(self, pool: str, keys: list[bytes]) -> int:
        """How many of ``keys``, from the first, are present."""
        if not keys:
            return 0
        return self.call(op="lookup", pool=pool, keys=self._hex(keys))["present"]

    def contains(self, pool: str, keys: list[bytes]) -> list[bool]:
        if not keys:
            return []
        return self.call(op="contains", pool=pool, keys=self._hex(keys))["present"]

    def acquire(self, pool: str, keys: list[bytes]) -> tuple[int, list[int]]:
        """Pin the leading present keys: (lease, their slots). Release the lease once their bytes are copied."""
        if not keys:
            return 0, []
        r = self.call(op="acquire", pool=pool, keys=self._hex(keys))
        return r["lease"], r["slots"]

    def release(self, lease: int) -> None:
        if lease:
            self.call(op="release", lease=lease)

    def reserve(self, pool: str, keys: list[bytes]) -> tuple[int, list[int | None]]:
        """Slots to write for the keys not yet present (None: present, being written elsewhere, or no room)."""
        if not keys:
            return 0, []
        r = self.call(op="reserve", pool=pool, keys=self._hex(keys))
        return r["reservation"], r["slots"]

    def commit(self, reservation: int) -> int:
        return self.call(op="commit", reservation=reservation)["published"] if reservation else 0

    def abort(self, reservation: int) -> None:
        if reservation:
            self.call(op="abort", reservation=reservation)

    def stats(self) -> dict:
        return self.call(op="stats")
