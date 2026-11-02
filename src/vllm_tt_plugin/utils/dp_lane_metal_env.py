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

# The triage entry point is a shell word whose basename is ``triage``
# (console script), ``triage.py`` (tools/triage/triage.py) or ``tt-triage.py``
# (the tools/ wrapper tt-metal documents as the timeout command). It only
# counts when it sits where the shell would run it: as a simple command's
# executable, or as the script a Python interpreter is told to run.
_TRIAGE_BASENAMES = frozenset({"triage", "triage.py", "tt-triage.py"})
_PYTHON_INTERPRETER_RE = re.compile(r"python(3(\.\d+)?)?")
# Python options that consume the following word, so it is not the script.
_PYTHON_OPTIONS_WITH_ARG = frozenset({"-W", "-X", "--check-hash-based-pycs"})
# ``-c code`` and ``-m module`` leave no script position.
_PYTHON_OPTIONS_WITHOUT_SCRIPT = frozenset({"-c", "-m"})
# Wrappers that run the command that follows them: the options of theirs that
# consume the next word, and the number of operands before the command.
_COMMAND_WRAPPERS: dict[str, tuple[frozenset[str], int]] = {
    "env": (frozenset({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}), 0),
    "exec": (frozenset({"-a"}), 0),
    "nohup": (frozenset(), 0),
    "time": (frozenset({"-f", "--format", "-o", "--output"}), 0),
    "command": (frozenset(), 0),
    "timeout": (frozenset({"-s", "--signal", "-k", "--kill-after"}), 1),
}
# ``command -v``/``-V`` print where a command is rather than run it.
_COMMAND_LOOKUP_OPTIONS = frozenset({"-v", "-V"})
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
# Reserved words that may precede a simple command's executable.
_RESERVED_WORDS = frozenset(
    {"{", "}", "!", "if", "then", "else", "elif", "fi", "do", "done", "while", "until"}
)

_INSPECTOR_RPC_PORT_FLAG = "--inspector-rpc-port"
_INSPECTOR_RPC_HOST_FLAG = "--inspector-rpc-host"
_INSPECTOR_LOG_PATH_FLAG = "--inspector-log-path"
_INSPECTOR_FLAGS = (
    _INSPECTOR_RPC_HOST_FLAG,
    _INSPECTOR_RPC_PORT_FLAG,
    _INSPECTOR_LOG_PATH_FLAG,
)

# Shell control operators that end a simple command.
_OPERATOR_CHARS = "|&;()\n"

# Token kinds. A redirection token carries the operator and its operand
# (``2>&1``, ``>/dev/null``, ``> out.log``) as one span that is never read
# as an argument and never edited.
_WORD = "word"
_OPERATOR = "operator"
_REDIRECT = "redirect"

# Results of reading a word's static text after quote removal.
_COMPLETE = "complete"  # the whole word is known
_DEFERRED = "deferred"  # the shell expands the rest (``$VAR``, ``$(...)``, ``~``)
_MALFORMED = "malformed"  # unbalanced quotes or a trailing backslash


class _ShellWord:
    """One token of the command as written (``text``), with its span."""

    __slots__ = ("start", "end", "text", "kind")

    def __init__(self, start: int, end: int, text: str, kind: str):
        self.start = start
        self.end = end
        self.text = text
        self.kind = kind

    def unquoted(self) -> tuple[str, str]:
        """The word's static text after quote removal, and whether it is
        complete, cut short by a deferred expansion, or malformed."""
        return _unquote(self.text)

    def literal(self) -> str | None:
        """The word's full value after quote removal, or ``None`` when the
        shell would still expand part of it or it cannot be read."""
        text, status = self.unquoted()
        return text if status == _COMPLETE else None


def _unquote(text: str) -> tuple[str, str]:
    """Remove shell quoting from one word without expanding anything.

    Returns the text up to the first construct the shell expands at run time
    (``$`` or a backtick outside single quotes, a leading ``~``) and a status:
    ``_COMPLETE`` when nothing was deferred, ``_DEFERRED`` when the returned
    prefix is all that is known statically, ``_MALFORMED`` for unbalanced
    quotes or a trailing backslash.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    if text.startswith("~"):
        return "", _DEFERRED
    while i < n:
        ch = text[i]
        if ch == "\\":
            if i + 1 >= n:
                return "".join(out), _MALFORMED
            if text[i + 1] != "\n":  # backslash-newline is a line continuation
                out.append(text[i + 1])
            i += 2
        elif ch == "'":
            close = text.find("'", i + 1)
            if close < 0:
                return "".join(out), _MALFORMED
            out.append(text[i + 1 : close])
            i = close + 1
        elif ch == '"':
            i += 1
            while True:
                if i >= n:
                    return "".join(out), _MALFORMED
                ch = text[i]
                if ch == '"':
                    i += 1
                    break
                if ch == "\\":
                    # Inside double quotes a backslash only escapes these;
                    # backslash-newline is a line continuation there too.
                    if i + 1 < n and text[i + 1] == "\n":
                        i += 2
                    elif i + 1 < n and text[i + 1] in '$`"\\':
                        out.append(text[i + 1])
                        i += 2
                    else:
                        out.append("\\")
                        i += 1
                elif ch in "$`":
                    return "".join(out), _DEFERRED
                else:
                    out.append(ch)
                    i += 1
        elif ch in "$`":
            return "".join(out), _DEFERRED
        else:
            out.append(ch)
            i += 1
    return "".join(out), _COMPLETE


def _split_shell_words(command: str) -> list[_ShellWord]:
    """Split a shell command into words, control operators and redirections,
    keeping each token exactly as written (quotes, escapes, ``$(...)``
    groups) with its span.

    Nothing is expanded or executed; the tokenizer only needs to know where
    one token ends and the next begins. Redirections are recognised the way
    the shell does, with or without surrounding whitespace (``2>&1``,
    ``--flag=value>/dev/null``), so their syntax never lands in an argument.
    A backslash-newline continuation and a ``#`` comment produce no tokens.
    """
    tokens: list[_ShellWord] = []
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch in " \t":
            i += 1
            continue
        if ch == "\\" and command[i + 1 : i + 2] == "\n":
            i += 2  # line continuation between tokens
            continue
        if ch == "#":
            end = command.find("\n", i)
            i = n if end < 0 else end  # comment runs to the end of the line
            continue
        if _starts_redirect(command, i):
            tokens.append(_scan_redirect(command, i))
            i = tokens[-1].end
            continue
        if ch in _OPERATOR_CHARS:
            j = i + 1
            if ch in "|&;" and j < n and command[j] == ch:
                j += 1  # ``||``, ``&&``, ``;;``
            tokens.append(_ShellWord(i, j, command[i:j], _OPERATOR))
            i = j
            continue
        start = i
        i = _scan_word(command, i)
        if i < n and command[i] in "<>" and command[start:i].isdigit():
            # ``2>&1``: the digits are the redirected descriptor.
            tokens.append(_scan_redirect(command, start))
            i = tokens[-1].end
            continue
        tokens.append(_ShellWord(start, i, command[start:i], _WORD))
    return tokens


def _starts_redirect(command: str, i: int) -> bool:
    ch = command[i]
    if ch in "<>":
        return True
    return ch == "&" and i + 1 < len(command) and command[i + 1] == ">"


def _scan_redirect(command: str, start: int) -> _ShellWord:
    """The redirection starting at ``start`` (optional descriptor digits or
    ``&``, the operator, the operand) as one token."""
    n = len(command)
    i = start
    while i < n and command[i].isdigit():
        i += 1
    if i < n and command[i] == "&":
        i += 1  # ``&>``, ``&>>``
    if i < n and command[i] in "<>":
        op = command[i]
        i += 1
        if i < n and command[i] == op:
            i += 1  # ``>>``, ``<<``
            if op == "<" and i < n and command[i] == "<":
                i += 1  # ``<<<``
        elif i < n and (command[i] in "&|" or (op == "<" and command[i] == ">")):
            i += 1  # ``>&``, ``<&``, ``>|``, ``<>``
    while i < n and command[i] in " \t":
        i += 1
    i = _scan_word(command, i)
    return _ShellWord(start, i, command[start:i], _REDIRECT)


def _scan_word(command: str, i: int) -> int:
    """Index just past the word starting at ``i``: it ends at unquoted
    whitespace, a control operator or a redirection operator."""
    n = len(command)
    while i < n:
        ch = command[i]
        if ch in " \t" or ch in _OPERATOR_CHARS or ch in "<>":
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
    return i


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


def _program_basename(word: _ShellWord) -> str | None:
    """The basename of a word naming a program or script, read statically.

    A fully literal word gives its basename. When a directory prefix is
    expanded later (``$VIRTUAL_ENV/bin/python``, ``"$TT_METAL_HOME/tools/
    tt-triage.py"``) the final path component is still read, since that is
    what decides whether the word names Python or triage.
    """
    literal = word.literal()
    if literal is not None:
        return os.path.basename(literal)
    if "/" not in word.text:
        return None
    tail = word.text.rsplit("/", 1)[1]
    for quote in ('"', "'"):
        if tail.endswith(quote) and tail.count(quote) % 2 == 1:
            tail = tail[:-1]  # closing quote of a quoted whole word
    text, status = _unquote(tail)
    return text if status == _COMPLETE and text else None


def _is_triage_entry_point(word: _ShellWord) -> bool:
    return _program_basename(word) in _TRIAGE_BASENAMES


def _is_python_interpreter(word: _ShellWord) -> bool:
    name = _program_basename(word)
    return name is not None and _PYTHON_INTERPRETER_RE.fullmatch(name) is not None


def _python_script_word(args: list[_ShellWord]) -> _ShellWord | None:
    """The word a Python interpreter would run as its script, if any."""
    i = 0
    while i < len(args):
        text, status = args[i].unquoted()
        if status == _MALFORMED:
            return None
        if status == _DEFERRED:
            # ``"$TT_METAL_HOME/tools/tt-triage.py"``: an expanded word in
            # script position is the script unless it is spelled as an option.
            return None if text.startswith("-") else args[i]
        if text == "--":
            return args[i + 1] if i + 1 < len(args) else None
        if not text.startswith("-") or text == "-":
            return args[i]
        if text in _PYTHON_OPTIONS_WITHOUT_SCRIPT:
            return None
        i += 2 if text in _PYTHON_OPTIONS_WITH_ARG else 1
    return None


def _find_triage_entry(words: list[_ShellWord]) -> int | None:
    """Index of the triage entry point in ``words``, looking only at the
    positions the shell would execute: each simple command's executable
    (after leading assignments and reserved words such as ``{`` or ``then``,
    and through ``env``/``exec``/``nohup``/``time``/``command``/``timeout``)
    and the script given to a Python interpreter. A matching basename
    anywhere else, such as a path checked with ``test`` or ``command -v`` or
    created with ``mkdir``, is an argument, not triage.
    """
    for operands in _simple_commands(words):
        j = 0
        while j < len(operands):
            word = operands[j]
            if _ASSIGNMENT_RE.match(word.text) or word.text in _RESERVED_WORDS:
                j += 1  # ``NAME=value``, ``{``, ``!``, ``then`` ... before the command
                continue
            if _is_triage_entry_point(word):
                return word.index
            if _is_python_interpreter(word):
                script = _python_script_word(operands[j + 1 :])
                if script is not None and _is_triage_entry_point(script):
                    return script.index
                break
            wrapper_name = _program_basename(word)
            wrapper = _COMMAND_WRAPPERS.get(wrapper_name or "")
            if wrapper is None:
                break
            options_with_arg, operand_count = wrapper
            j += 1
            lookup_only = False
            while j < len(operands):
                text = operands[j].literal()
                if text is None or not text.startswith("-") or text == "--":
                    break
                if wrapper_name == "command" and text in _COMMAND_LOOKUP_OPTIONS:
                    lookup_only = True  # ``command -v triage``: a probe, not a run
                j += 2 if text in options_with_arg else 1
            if lookup_only:
                break
            if j < len(operands) and operands[j].literal() == "--":
                j += 1
            j += operand_count
    return None


class _Operand(_ShellWord):
    """A word of a simple command, remembering its index in the token list."""

    __slots__ = ("index",)

    def __init__(self, word: _ShellWord, index: int):
        super().__init__(word.start, word.end, word.text, word.kind)
        self.index = index


def _simple_commands(words: list[_ShellWord]) -> list[list[_Operand]]:
    """The words of each simple command (redirections and operators left
    out), in command order."""
    commands: list[list[_Operand]] = [[]]
    for index, word in enumerate(words):
        if word.kind == _OPERATOR:
            if commands[-1]:
                commands.append([])
        elif word.kind == _WORD:
            commands[-1].append(_Operand(word, index))
    return [c for c in commands if c]


class _ExplicitFlag:
    """An Inspector flag the triage invocation already carries."""

    __slots__ = ("flag", "start", "end", "value", "written", "value_only")

    def __init__(
        self,
        flag: str,
        start: int,
        end: int,
        value: str | None,
        written: str,
        value_only: bool,
    ):
        self.flag = flag
        self.start = start  # span to replace when the flag moves with the lane
        self.end = end
        self.value = value  # literal value, or None when the shell would expand it
        self.written = written  # the flag as the operator wrote it, for messages
        self.value_only = value_only  # ``--flag value``: only the value is replaced


def _explicit_inspector_flags(args: list[_ShellWord]) -> list[_ExplicitFlag]:
    """Inspector flags among the triage invocation's own arguments, in either
    docopt spelling (``--flag=value`` as one word, or ``--flag value``).

    The flag name is read after quote removal, so ``"--flag=$VAR"`` and
    ``'--flag='"$VAR"`` are recognised as the flag with a deferred value,
    the same as ``--flag="$VAR"``.
    """
    found: list[_ExplicitFlag] = []
    i = 0
    while i < len(args):
        word = args[i]
        text, status = word.unquoted()
        if status == _MALFORMED:
            i += 1
            continue
        flag = next(
            (
                f
                for f in _INSPECTOR_FLAGS
                if text.startswith(f + "=") or (text == f and status == _COMPLETE)
            ),
            None,
        )
        if flag is None:
            i += 1
            continue
        if text != flag:
            value = text[len(flag) + 1 :] if status == _COMPLETE else None
            found.append(
                _ExplicitFlag(flag, word.start, word.end, value, word.text, False)
            )
            i += 1
            continue
        if i + 1 < len(args):
            value_word = args[i + 1]
            found.append(
                _ExplicitFlag(
                    flag,
                    value_word.start,
                    value_word.end,
                    value_word.literal(),
                    f"{word.text} {value_word.text}",
                    True,
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

    The command is split into shell words, redirections and control
    operators without expanding anything. The triage invocation is the first
    simple command whose executable, or whose Python interpreter's script,
    is a triage entry point; a matching name in any other position (a path
    given to ``test`` or ``mkdir``) is an argument and is ignored. Only that
    invocation's own words are read or edited, so flags given to other
    commands in a pipeline, and redirection syntax anywhere, are untouched.
    Inserts ``--inspector-rpc-host`` / ``--inspector-rpc-port`` (when an RPC
    address is given) and ``--inspector-log-path`` right after the entry
    point, each as one shell word (``shlex.quote`` on the value), because
    tt-metal runs the command through ``std::system``. Generated text is
    never re-parsed.

    An Inspector flag the invocation already carries is moved with the lane
    when its value is the pre-lane one (``base_*``: the server address and
    log directory the operator configured, which triage was pointing at
    consistently). A value the shell would still expand (``$VAR``,
    ``$(...)``) cannot be checked and is left as written, with a warning,
    however the option is quoted. Any other explicit value contradicts the
    Inspector this lane will run, so it raises before the worker opens a
    device rather than leaving triage reading an Inspector that is not this
    lane's. A command with no recognised entry point is returned unchanged
    with a warning, since a triage hidden in it would still read the default
    address.
    """
    words = _split_shell_words(command)
    entry_index = _find_triage_entry(words)
    if entry_index is None:
        logger.warning(
            "%s does not invoke a recognised tt-triage entry point (triage, "
            "triage.py or tt-triage.py run as a command or as a Python script); "
            "leaving it unchanged. If it runs triage some other way, that triage "
            "reads the default Inspector address, not this lane's %s:%s / %s. "
            "Command: %s",
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
        if word.kind == _OPERATOR:
            break
        if word.kind == _WORD:
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
        replacement = (
            shlex.quote(lane_value)
            if item.value_only
            else _flag_word(item.flag, lane_value)
        )
        edits.append((item.start, item.end, replacement))

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
