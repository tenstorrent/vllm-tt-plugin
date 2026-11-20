# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-lane tt-metal diagnostics environment for standard data-parallel launches.

A standard-DP server runs one tt-metal process per lane on the same host, each
restricted to its sub-mesh with ``TT_VISIBLE_DEVICES``. tt-metal separates its
Inspector RPC port and log directory by rank only for multi-host MPI launches
(``DistributedContext`` world size > 1), so standard-DP lanes all bind
``localhost:50051`` (three of four lose, non-fatally) and all write
``<TT_METAL_LOGS_PATH>/generated/inspector``.

When the dispatch watchdog fires in a lane that lost the bind, the auto-triage
command (``TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE``, run by tt-metal
through ``std::system``) connects to the default port and reads another lane's
Inspector. tt-exalens in that subprocess inherits the lane's
``TT_VISIBLE_DEVICES``, so none of the other lane's devices are visible,
``run_checks.devices`` resolves empty, and every dispatcher-backed dump is
skipped with ``No devices to inspect`` (tt-metal #63345). This module gives
each lane its own Inspector address and log directory and points the triage
command at them, mirroring tt-metal's own ``port + rank`` / ``_rank_<n>``
convention for MPI ranks. No existing tt-metal option does this for same-host
processes that are not MPI ranks; once tt-metal #63345 provides one, this
module can go.

The variables involved are tt-metal's own (``tt_metal/llrt/rtoptions.cpp``),
read from the inherited environment and overridden per lane; the plugin
introduces none of its own:

- ``TT_METAL_INSPECTOR_RPC``: ``0`` disables the Inspector RPC server
  (default enabled). Read only.
- ``TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS``: ``host:port`` or ``host`` the
  server listens on (default ``localhost:50051``). Overridden to
  ``host:port+<rank>``.
- ``TT_METAL_LOGS_PATH``: directory for everything tt-metal generates
  (inspector, watcher, dprint, profiler, NoC logs, reports; default the
  working directory). Overridden to ``<inherited>/dp_rank_<rank>``, so all of
  those land under the rank's directory and lanes stop overwriting each
  other's files.
- ``TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE``: shell command tt-metal
  runs on a dispatch timeout (default none; tt-metal documents
  ``./tools/tt-triage.py``). Overridden with this lane's Inspector flags
  inserted after the triage entry point.

All functions are pure; the worker applies the returned overrides to
``os.environ`` before the first device open, which is when tt-metal latches
its runtime options.
"""

import logging
import os
import re
import shlex
from collections.abc import Mapping

logger = logging.getLogger(__name__)

TT_METAL_INSPECTOR_RPC_ENV = "TT_METAL_INSPECTOR_RPC"
TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV = "TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS"
TT_METAL_LOGS_PATH_ENV = "TT_METAL_LOGS_PATH"
TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV = "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"

# tt-metal defaults (tt_metal/llrt/rtoptions.cpp): the Inspector RPC server
# listens on localhost:50051 and logs go under the current working directory.
DEFAULT_INSPECTOR_RPC_HOST = "localhost"
DEFAULT_INSPECTOR_RPC_PORT = 50051
INSPECTOR_LOG_SUBDIR = os.path.join("generated", "inspector")
LANE_LOGS_DIR_FORMAT = "dp_rank_{rank}"

# The triage entry point as one shell word: ``triage`` (console script),
# ``triage.py`` (tools/triage/triage.py) or ``tt-triage.py`` (the tools/
# wrapper tt-metal documents as the timeout command), with an optional
# directory prefix, optionally wrapped in matching single or double quotes.
# The whole word is matched so the flags are inserted after the closing quote
# and the word keeps its shell meaning.
_TRIAGE_WORD_RE = re.compile(
    r"""
    (?<!\S)                                      # start or after whitespace
    (?:
        (?:[^\s"']*/)?(?:tt-)?triage(?:\.py)?    # bare word
      | "(?:[^"]*/)?(?:tt-)?triage(?:\.py)?"     # double-quoted, may hold spaces
      | '(?:[^']*/)?(?:tt-)?triage(?:\.py)?'     # single-quoted, may hold spaces
    )
    (?=\s|$)
    """,
    re.VERBOSE,
)

_INSPECTOR_RPC_PORT_FLAG = "--inspector-rpc-port"
_INSPECTOR_RPC_HOST_FLAG = "--inspector-rpc-host"
_INSPECTOR_LOG_PATH_FLAG = "--inspector-log-path"
_INSPECTOR_FLAG_RE = re.compile(r"--inspector-(?:rpc-port|rpc-host|log-path)\b")


def parse_inspector_rpc_address(value: str | None) -> tuple[str, int] | None:
    """Parse ``TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS`` the way tt-metal does.

    Accepts ``host:port`` or a bare ``host`` (default port). Returns ``None``
    for a malformed port so the caller leaves the value alone and tt-metal
    reports the problem itself at init.
    """
    if value is None or value.strip() == "":
        return DEFAULT_INSPECTOR_RPC_HOST, DEFAULT_INSPECTOR_RPC_PORT
    value = value.strip()
    host, sep, port_text = value.partition(":")
    if not sep:
        return value, DEFAULT_INSPECTOR_RPC_PORT
    try:
        port = int(port_text)
    except ValueError:
        return None
    if not 0 < port < 65536:
        return None
    return host or DEFAULT_INSPECTOR_RPC_HOST, port


def lane_logs_path(base_logs_path: str, local_dp_rank: int) -> str:
    """Per-lane ``TT_METAL_LOGS_PATH``: ``<base>/dp_rank_<n>``."""
    return os.path.join(base_logs_path, LANE_LOGS_DIR_FORMAT.format(rank=local_dp_rank))


def rewrite_triage_command(
    command: str,
    *,
    rpc_host: str | None,
    rpc_port: int | None,
    inspector_log_path: str,
) -> str:
    """Point an auto-triage command at this lane's Inspector.

    Inserts ``--inspector-rpc-host`` / ``--inspector-rpc-port`` (when an RPC
    address is given) and ``--inspector-log-path`` right after the first
    ``triage`` / ``triage.py`` / ``tt-triage.py`` word, each as one shell
    word (``shlex.quote`` on the value), because tt-metal runs the command
    through ``std::system``.

    The command is the operator's and its syntax stays theirs: one that
    already names any of those flags is returned unchanged, with a warning
    that gives this lane's values, as is one with no triage word in it. The
    second rule also makes the rewrite idempotent.
    """
    match = _TRIAGE_WORD_RE.search(command)
    if match is None:
        logger.warning(
            "%s has no triage / triage.py / tt-triage.py word; leaving it "
            "unchanged. If it runs triage some other way, that triage reads the "
            "default Inspector address, not this lane's %s:%s / %s. Command: %s",
            TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV,
            rpc_host,
            rpc_port,
            inspector_log_path,
            command,
        )
        return command
    if _INSPECTOR_FLAG_RE.search(command):
        logger.warning(
            "%s already passes Inspector flags to triage; leaving it unchanged. "
            "This lane's Inspector is %s:%s with logs at %s; drop the flags to "
            "have the plugin set them per lane. Command: %s",
            TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV,
            rpc_host,
            rpc_port,
            inspector_log_path,
            command,
        )
        return command

    flags: list[str] = []
    if rpc_host is not None and rpc_port is not None:
        flags.append(f"{_INSPECTOR_RPC_HOST_FLAG}={shlex.quote(rpc_host)}")
        flags.append(f"{_INSPECTOR_RPC_PORT_FLAG}={rpc_port}")
    flags.append(f"{_INSPECTOR_LOG_PATH_FLAG}={shlex.quote(inspector_log_path)}")
    end = match.end()
    return command[:end] + "".join(" " + flag for flag in flags) + command[end:]


def compute_dp_lane_metal_env(
    local_dp_rank: int | None,
    num_lanes: int,
    env: Mapping[str, str],
    *,
    cwd: str | None = None,
) -> dict[str, str]:
    """Environment overrides that isolate this lane's tt-metal diagnostics.

    Returns an empty dict when there is nothing to separate (a single lane,
    an unknown rank). Otherwise:

    - ``TT_METAL_LOGS_PATH``: ``<base>/dp_rank_<n>`` where ``<base>`` is the
      inherited value or the working directory (tt-metal's own default), so
      each lane's Inspector writes ``.../dp_rank_<n>/generated/inspector`` and
      the serialized RPC data tt-metal dumps on a dispatch timeout is the hung
      lane's, not a neighbour's.
    - ``TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS``: inherited host and
      ``port + n`` (tt-metal's MPI-rank convention), unless
      ``TT_METAL_INSPECTOR_RPC=0`` disabled the server or the inherited address
      is malformed.
    - ``TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE``: the inherited triage
      command with this lane's Inspector host, port and log path, see
      :func:`rewrite_triage_command`.
    """
    if local_dp_rank is None or num_lanes <= 1:
        return {}

    overrides: dict[str, str] = {}

    base_logs_path = env.get(TT_METAL_LOGS_PATH_ENV) or (cwd or os.getcwd())
    logs_path = lane_logs_path(base_logs_path, local_dp_rank)
    overrides[TT_METAL_LOGS_PATH_ENV] = logs_path
    inspector_log_path = os.path.join(logs_path, INSPECTOR_LOG_SUBDIR)

    rpc_host: str | None = None
    rpc_port: int | None = None
    rpc_enabled = env.get(TT_METAL_INSPECTOR_RPC_ENV, "1").strip() != "0"
    if rpc_enabled:
        parsed = parse_inspector_rpc_address(
            env.get(TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV)
        )
        if parsed is not None:
            rpc_host, base_port = parsed
            rpc_port = base_port + local_dp_rank
            overrides[TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV] = (
                f"{rpc_host}:{rpc_port}"
            )

    command = env.get(TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV)
    if command:
        rewritten = rewrite_triage_command(
            command,
            rpc_host=rpc_host,
            rpc_port=rpc_port,
            inspector_log_path=inspector_log_path,
        )
        if rewritten != command:
            overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV] = rewritten

    return overrides
