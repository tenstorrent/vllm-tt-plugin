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

# The triage entry point is one shell word whose basename is ``triage``
# (console script), ``triage.py`` (tools/triage/triage.py) or ``tt-triage.py``
# (the tools/ wrapper tt-metal documents as the timeout command).
_TRIAGE_BASENAMES = frozenset({"triage", "triage.py", "tt-triage.py"})

_INSPECTOR_RPC_PORT_FLAG = "--inspector-rpc-port"
_INSPECTOR_RPC_HOST_FLAG = "--inspector-rpc-host"
_INSPECTOR_LOG_PATH_FLAG = "--inspector-log-path"
_INSPECTOR_FLAGS = (
    _INSPECTOR_RPC_HOST_FLAG,
    _INSPECTOR_RPC_PORT_FLAG,
    _INSPECTOR_LOG_PATH_FLAG,
)

# Shell control operators that end a simple command. Redirections (``2>&1``)
# are kept inside words: they are not Inspector flags and they do not end the
# triage invocation.
_OPERATOR_CHARS = "|&;()\n"


class _ShellWord:
    """One word of the command as written (``text``), with its span."""

    __slots__ = ("start", "end", "text", "operator")

    def __init__(self, start: int, end: int, text: str, operator: bool):
        self.start = start
        self.end = end
        self.text = text
        self.operator = operator

    def literal(self) -> str | None:
        """The word's value after quote removal, or ``None`` when the shell
        would still expand it (``$VAR``, ``$(...)``, backticks) or it cannot be
        read as one word."""
        if _has_deferred_expansion(self.text):
            return None
        try:
            parts = shlex.split(self.text)
        except ValueError:
            return None
        return parts[0] if len(parts) == 1 else None


def _has_deferred_expansion(text: str) -> bool:
    """True when ``$`` or a backtick appears outside single quotes."""
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'":
            close = text.find("'", i + 1)
            i = len(text) if close < 0 else close + 1
            continue
        if ch == "\\":
            i += 2
            continue
        if ch in "$`":
            return True
        i += 1
    return False


def _split_shell_words(command: str) -> list[_ShellWord]:
    """Split a shell command into words and control operators, keeping each
    word exactly as written (quotes, escapes, ``$(...)`` groups) with its span.

    Nothing is expanded or executed; the tokenizer only needs to know where
    one word ends and the next begins.
    """
    words: list[_ShellWord] = []
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch in " \t":
            i += 1
            continue
        if ch in _OPERATOR_CHARS:
            j = i + 1
            if ch in "|&;" and j < n and command[j] == ch:
                j += 1  # ``||``, ``&&``, ``;;``
            words.append(_ShellWord(i, j, command[i:j], operator=True))
            i = j
            continue
        start = i
        while i < n:
            ch = command[i]
            if ch in " \t":
                break
            if ch in _OPERATOR_CHARS:
                # ``>&`` / ``<&`` is a redirection, part of the word.
                if ch == "&" and i > start and command[i - 1] in "<>":
                    i += 1
                    continue
                break
            if ch == "\\":
                i += 2
            elif ch == "'":
                close = command.find("'", i + 1)
                i = n if close < 0 else close + 1
            elif ch == '"':
                i = _skip_double_quoted(command, i)
            elif ch == "`":
                close = command.find("`", i + 1)
                i = n if close < 0 else close + 1
            elif ch == "$" and i + 1 < n and command[i + 1] in "({":
                i = _skip_group(command, i + 1)
            else:
                i += 1
        words.append(_ShellWord(start, i, command[start:i], operator=False))
    return words


def _skip_double_quoted(command: str, i: int) -> int:
    """Index just past the double-quoted string opening at ``i``."""
    i += 1
    n = len(command)
    while i < n:
        ch = command[i]
        if ch == "\\":
            i += 2
        elif ch == '"':
            return i + 1
        elif ch == "$" and i + 1 < n and command[i + 1] in "({":
            i = _skip_group(command, i + 1)
        else:
            i += 1
    return n


def _skip_group(command: str, i: int) -> int:
    """Index just past the ``(...)`` or ``{...}`` group opening at ``i``."""
    opener = command[i]
    closer = ")" if opener == "(" else "}"
    depth = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch == "\\":
            i += 2
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        elif ch == "'":
            close = command.find("'", i + 1)
            i = n if close < 0 else close
        i += 1
    return n


def _is_triage_entry_point(word: _ShellWord) -> bool:
    literal = word.literal()
    return literal is not None and os.path.basename(literal) in _TRIAGE_BASENAMES


class _ExplicitFlag:
    """An Inspector flag the triage invocation already carries."""

    __slots__ = ("flag", "start", "end", "value", "written")

    def __init__(
        self, flag: str, start: int, end: int, value: str | None, written: str
    ):
        self.flag = flag
        self.start = start  # span of the flag word(s) in the command
        self.end = end
        self.value = value  # literal value, or None when the shell would expand it
        self.written = written  # the flag as the operator wrote it, for messages


def _explicit_inspector_flags(args: list[_ShellWord]) -> list[_ExplicitFlag]:
    """Inspector flags among the triage invocation's own arguments, in either
    docopt spelling (``--flag=value`` as one word, or ``--flag value``)."""
    found: list[_ExplicitFlag] = []
    i = 0
    while i < len(args):
        word = args[i]
        literal = word.literal()
        if literal is None:
            # ``--flag=$VAR``: the flag is readable even when the value is not.
            flag = next(
                (f for f in _INSPECTOR_FLAGS if word.text.startswith(f + "=")), None
            )
            if flag is not None:
                found.append(_ExplicitFlag(flag, word.start, word.end, None, word.text))
            i += 1
            continue
        flag = next(
            (
                f
                for f in _INSPECTOR_FLAGS
                if literal == f or literal.startswith(f + "=")
            ),
            None,
        )
        if flag is None:
            i += 1
            continue
        if literal != flag:
            found.append(
                _ExplicitFlag(
                    flag, word.start, word.end, literal[len(flag) + 1 :], word.text
                )
            )
            i += 1
            continue
        if i + 1 < len(args):
            value_word = args[i + 1]
            found.append(
                _ExplicitFlag(
                    flag,
                    word.start,
                    value_word.end,
                    value_word.literal(),
                    f"{word.text} {value_word.text}",
                )
            )
            i += 2
            continue
        i += 1
    return found


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
    base_rpc_host: str | None = None,
    base_rpc_port: int | None = None,
    base_inspector_log_path: str | None = None,
) -> str:
    """Point an auto-triage command at this lane's Inspector.

    The command is split into shell words without expanding anything. The
    first word whose basename is a triage entry point starts the triage
    invocation, which runs to the next control operator; only that
    invocation's arguments are read or edited, so flags given to other
    commands in a pipeline are untouched. Inserts ``--inspector-rpc-host``
    / ``--inspector-rpc-port`` (when an RPC address is given) and
    ``--inspector-log-path`` right after the entry point, each as one shell
    word (``shlex.quote`` on the value), because tt-metal runs the command
    through ``std::system``. Generated text is never re-parsed.

    An Inspector flag the invocation already carries is moved with the lane
    when its value is the pre-lane one (``base_*``: the server address and
    log directory the operator configured, which triage was pointing at
    consistently). A value the shell would still expand (``$VAR``,
    ``$(...)``) cannot be checked and is left as written, with a warning. Any
    other explicit value contradicts the Inspector this lane will run, so it
    raises before the worker opens a device rather than leaving triage
    reading an Inspector that is not this lane's. A command with no
    recognised entry point is returned unchanged with a warning, since a
    triage hidden in it would still read the default address.
    """
    words = _split_shell_words(command)
    entry_index = next(
        (
            i
            for i, w in enumerate(words)
            if not w.operator and _is_triage_entry_point(w)
        ),
        None,
    )
    if entry_index is None:
        logger.warning(
            "%s does not invoke a recognised tt-triage entry point (a shell word "
            "named triage, triage.py or tt-triage.py); leaving it unchanged. If it "
            "runs triage some other way, that triage reads the default Inspector "
            "address, not this lane's %s:%s / %s. Command: %s",
            TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV,
            rpc_host,
            rpc_port,
            inspector_log_path,
            command,
        )
        return command
    entry = words[entry_index]
    args: list[_ShellWord] = []
    for word in words[entry_index + 1 :]:
        if word.operator:
            break
        args.append(word)

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

    # (start, end, replacement) edits on the original text, applied last to
    # first so earlier spans stay valid. Generated text is never re-parsed.
    edits: list[tuple[int, int, str]] = []
    explicit = _explicit_inspector_flags(args)
    for item in explicit:
        lane_value = lane_values.get(item.flag)
        if lane_value is None or item.value == lane_value:
            continue  # nothing to move, or already the lane value (idempotent)
        if item.value is None:
            logger.warning(
                "%s passes %s to triage with a value the shell expands later, "
                "which the plugin cannot check against this lane's %s=%r; "
                "leaving it as written. Command: %s",
                TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV,
                item.written,
                item.flag,
                lane_value,
                command,
            )
            continue
        if item.value != base_values[item.flag]:
            raise ValueError(
                f"{TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV} passes "
                f"{item.flag}={item.value!r} to triage but this lane's Inspector is "
                f"{item.flag}={lane_value!r} (configured value before the per-lane "
                f"offset: {base_values[item.flag]!r}). Drop the flag so the plugin "
                f"sets it per lane, or make it match the configured "
                f"{TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV} / "
                f"{TT_METAL_LOGS_PATH_ENV}. Command: {command!r}"
            )
        edits.append((item.start, item.end, _flag_word(item.flag, lane_value)))

    present = {item.flag for item in explicit}
    extra = [
        _flag_word(flag, value)
        for flag, value in lane_values.items()
        if flag not in present
    ]
    if extra:
        edits.append((entry.end, entry.end, "".join(" " + arg for arg in extra)))

    rewritten = command
    for start, end, replacement in sorted(edits, reverse=True):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    return rewritten


def _flag_word(flag: str, value: str) -> str:
    """``--flag=value`` as one shell word: the value is quoted, the flag is
    left bare so the word is still readable as a flag on a later pass."""
    return f"{flag}={shlex.quote(value)}"


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
