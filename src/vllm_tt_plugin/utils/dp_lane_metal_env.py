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
convention for MPI ranks.

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
_TRIAGE_TOKEN_RE = re.compile(
    r"""
    (?<!\S)                          # start of string or after whitespace
    (?P<triage>
        (?:[^\s"']*/)?(?:tt-)?triage(?:\.py)?      # bare word, no spaces
      | "(?:[^"]*/)?(?:tt-)?triage(?:\.py)?"       # double-quoted, may hold spaces
      | '(?:[^']*/)?(?:tt-)?triage(?:\.py)?'       # single-quoted, may hold spaces
    )
    (?=\s|$)                         # the word ends at whitespace or EOS
    """,
    re.VERBOSE,
)

_INSPECTOR_RPC_PORT_FLAG = "--inspector-rpc-port"
_INSPECTOR_RPC_HOST_FLAG = "--inspector-rpc-host"
_INSPECTOR_LOG_PATH_FLAG = "--inspector-log-path"
_INSPECTOR_FLAGS = (
    _INSPECTOR_RPC_HOST_FLAG,
    _INSPECTOR_RPC_PORT_FLAG,
    _INSPECTOR_LOG_PATH_FLAG,
)

# An Inspector flag the command already carries, in either docopt spelling
# (``--flag=value`` or ``--flag value``), with a quoted or bare value.
_INSPECTOR_FLAG_RE = re.compile(
    r"""
    (?<!\S)
    (?P<flag>--inspector-(?:rpc-host|rpc-port|log-path))
    (?:=|\s+)
    (?P<value>'[^']*'|"[^"]*"|\S+)
    (?=\s|$)
    """,
    re.VERBOSE,
)


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


def _unquote(value: str) -> str:
    """The shell value of one quoted or bare word, as tt-metal's shell sees it."""
    words = shlex.split(value)
    return words[0] if len(words) == 1 else value


def _explicit_inspector_flags(command: str) -> dict[str, str]:
    """Inspector flags the command already carries, unquoted, by flag name."""
    return {
        match.group("flag"): _unquote(match.group("value"))
        for match in _INSPECTOR_FLAG_RE.finditer(command)
    }


def rewrite_triage_command(
    command: str,
    *,
    rpc_host: str | None,
    rpc_port: int | None,
    inspector_log_path: str,
    base_rpc_host: str | None = None,
    base_rpc_port: int | None = None,
    base_inspector_log_path: str | None = None,
) -> str:
    """Point an auto-triage command at this lane's Inspector.

    Inserts ``--inspector-rpc-host``/``--inspector-rpc-port`` (when an RPC
    address is given) and ``--inspector-log-path`` right after the triage
    entry point. Each generated argument is a single shell word
    (``shlex.quote``), because tt-metal runs the command through
    ``std::system``.

    A flag the command already carries is moved with the lane when its value
    is the pre-lane one (``base_*``: the server address and log directory the
    operator configured, which triage was pointing at consistently). Any other
    explicit value contradicts the Inspector this lane will run, so it raises
    before the worker opens a device rather than leaving triage reading an
    Inspector that is not this lane's. A command that does not invoke a
    recognised triage entry point is returned unchanged with a warning, since
    a triage hidden in it would still read the default address.
    """
    match = _TRIAGE_TOKEN_RE.search(command)
    if match is None:
        logger.warning(
            "%s does not invoke a recognised tt-triage entry point (triage, "
            "triage.py or tt-triage.py as one shell word); leaving it unchanged. "
            "If it runs triage some other way, that triage reads the default "
            "Inspector address, not this lane's %s:%s / %s. Command: %s",
            TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV,
            rpc_host,
            rpc_port,
            inspector_log_path,
            command,
        )
        return command

    lane_values: dict[str, str] = {}
    if rpc_host is not None and rpc_port is not None:
        lane_values[_INSPECTOR_RPC_HOST_FLAG] = rpc_host
        lane_values[_INSPECTOR_RPC_PORT_FLAG] = str(rpc_port)
    lane_values[_INSPECTOR_LOG_PATH_FLAG] = inspector_log_path
    base_values = {
        _INSPECTOR_RPC_HOST_FLAG: base_rpc_host,
        _INSPECTOR_RPC_PORT_FLAG: None if base_rpc_port is None else str(base_rpc_port),
        _INSPECTOR_LOG_PATH_FLAG: base_inspector_log_path,
    }

    explicit = _explicit_inspector_flags(command)
    for flag, value in explicit.items():
        lane_value = lane_values.get(flag)
        if lane_value is None or value == lane_value:
            continue  # nothing to move, or already the lane value (idempotent)
        if value != base_values[flag]:
            raise ValueError(
                f"{TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV} carries {flag}={value!r} "
                f"but this lane's Inspector is {flag}={lane_value!r} (configured "
                f"value before the per-lane offset: {base_values[flag]!r}). Drop "
                f"the flag so the plugin sets it per lane, or make it match the "
                f"configured {TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV} / "
                f"{TT_METAL_LOGS_PATH_ENV}. Command: {command!r}"
            )

    # Quote the value, not the whole ``--flag=value`` word: the shell still
    # sees one argument, and the flag stays recognisable on a second pass.
    extra = [
        f"{flag}={shlex.quote(value)}"
        for flag, value in lane_values.items()
        if flag not in explicit
    ]
    end = match.end("triage")
    rewritten = command[:end] + "".join(" " + arg for arg in extra) + command[end:]

    def move_explicit(flag_match: re.Match[str]) -> str:
        flag = flag_match.group("flag")
        lane_value = lane_values.get(flag)
        if lane_value is None:
            return flag_match.group(0)
        return f"{flag}={shlex.quote(lane_value)}"

    return _INSPECTOR_FLAG_RE.sub(move_explicit, rewritten)


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
    base_inspector_log_path = os.path.join(base_logs_path, INSPECTOR_LOG_SUBDIR)
    inspector_log_path = os.path.join(logs_path, INSPECTOR_LOG_SUBDIR)

    rpc_host: str | None = None
    rpc_port: int | None = None
    base_port: int | None = None
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
            base_rpc_host=rpc_host,
            base_rpc_port=base_port,
            base_inspector_log_path=base_inspector_log_path,
        )
        if rewritten != command:
            overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV] = rewritten

    return overrides
