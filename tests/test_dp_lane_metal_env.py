# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host tests for the per-lane tt-metal diagnostics environment.

Standard-DP lanes are separate tt-metal processes on one host. Without a
per-lane Inspector address and log directory they collide on
``localhost:50051`` and the auto-triage run on a dispatch timeout reads a
neighbouring lane (tt-metal #63345). These tests cover the override
computation, the triage command rewrite, and the worker binding that applies
them to ``os.environ``.
"""

import logging
import os
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from vllm_tt_plugin.utils.dp_lane_metal_env import (
    DEFAULT_INSPECTOR_RPC_PORT,
    TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV,
    TT_METAL_INSPECTOR_RPC_ENV,
    TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV,
    TT_METAL_LOGS_PATH_ENV,
    compute_dp_lane_metal_env,
    parse_inspector_rpc_address,
    rewrite_triage_command,
)
from vllm_tt_plugin.worker import _bind_dp_lane_metal_env

_TRIAGE_CMD = (
    "mkdir -p /logs && /opt/tt-metal/python_env/bin/python "
    "/opt/tt-metal/tools/triage/triage.py --disable-progress 2>&1 "
    "| tee /logs/tt-triage-$(date +%Y%m%d-%H%M%S).log"
)

_MANAGED_ENV_VARS = (
    TT_METAL_LOGS_PATH_ENV,
    TT_METAL_INSPECTOR_RPC_ENV,
    TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV,
    TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV,
)


@pytest.fixture(autouse=True)
def _restore_managed_env():
    """``_bind_dp_lane_metal_env`` writes ``os.environ`` outside monkeypatch."""
    inherited = {name: os.environ.get(name) for name in _MANAGED_ENV_VARS}
    yield
    for name, value in inherited.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ("localhost", DEFAULT_INSPECTOR_RPC_PORT)),
        ("", ("localhost", DEFAULT_INSPECTOR_RPC_PORT)),
        ("127.0.0.1", ("127.0.0.1", DEFAULT_INSPECTOR_RPC_PORT)),
        ("127.0.0.1:8080", ("127.0.0.1", 8080)),
        (":8080", ("localhost", 8080)),
        ("127.0.0.1:eighty", None),
        ("127.0.0.1:70000", None),
    ],
)
def test_parse_inspector_rpc_address_matches_tt_metal(value, expected) -> None:
    assert parse_inspector_rpc_address(value) == expected


def test_single_lane_and_unknown_rank_leave_env_alone() -> None:
    assert compute_dp_lane_metal_env(0, 1, {}, cwd="/w") == {}
    assert compute_dp_lane_metal_env(None, 4, {}, cwd="/w") == {}


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_each_lane_gets_its_own_port_and_logs_dir(rank: int) -> None:
    env = {
        TT_METAL_LOGS_PATH_ENV: "/logs",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: _TRIAGE_CMD,
    }
    overrides = compute_dp_lane_metal_env(rank, 4, env, cwd="/w")

    assert overrides[TT_METAL_LOGS_PATH_ENV] == f"/logs/dp_rank_{rank}"
    assert (
        overrides[TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV]
        == f"localhost:{DEFAULT_INSPECTOR_RPC_PORT + rank}"
    )
    command = overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV]
    assert (
        f"triage.py --inspector-rpc-host=localhost "
        f"--inspector-rpc-port={DEFAULT_INSPECTOR_RPC_PORT + rank} "
        f"--inspector-log-path=/logs/dp_rank_{rank}/generated/inspector "
        "--disable-progress"
    ) in command
    # The rest of the shell pipeline is untouched.
    assert command.startswith("mkdir -p /logs && ")
    assert command.endswith("| tee /logs/tt-triage-$(date +%Y%m%d-%H%M%S).log")


def test_lanes_do_not_share_an_address_or_logs_dir() -> None:
    env = {TT_METAL_LOGS_PATH_ENV: "/logs"}
    lanes = [compute_dp_lane_metal_env(rank, 4, env, cwd="/w") for rank in range(4)]
    addresses = {lane[TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV] for lane in lanes}
    logs = {lane[TT_METAL_LOGS_PATH_ENV] for lane in lanes}
    assert len(addresses) == 4
    assert len(logs) == 4


def test_logs_dir_defaults_to_working_directory_like_tt_metal() -> None:
    overrides = compute_dp_lane_metal_env(1, 2, {}, cwd="/work")
    assert overrides[TT_METAL_LOGS_PATH_ENV] == "/work/dp_rank_1"


def test_explicit_address_is_offset_by_rank() -> None:
    env = {TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV: "127.0.0.1:60000"}
    overrides = compute_dp_lane_metal_env(3, 4, env, cwd="/w")
    assert overrides[TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV] == "127.0.0.1:60003"


def test_malformed_address_is_passed_through_for_tt_metal_to_reject() -> None:
    env = {
        TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV: "127.0.0.1:eighty",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: _TRIAGE_CMD,
    }
    overrides = compute_dp_lane_metal_env(1, 2, env, cwd="/w")
    assert TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV not in overrides
    command = overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV]
    assert "--inspector-rpc-port" not in command
    assert "--inspector-log-path=/w/dp_rank_1/generated/inspector" in command


def test_disabled_rpc_still_separates_logs_and_points_triage_at_them() -> None:
    env = {
        TT_METAL_INSPECTOR_RPC_ENV: "0",
        TT_METAL_LOGS_PATH_ENV: "/logs",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: _TRIAGE_CMD,
    }
    overrides = compute_dp_lane_metal_env(2, 4, env, cwd="/w")
    assert TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV not in overrides
    assert overrides[TT_METAL_LOGS_PATH_ENV] == "/logs/dp_rank_2"
    command = overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV]
    assert "--inspector-rpc-port" not in command
    assert "--inspector-log-path=/logs/dp_rank_2/generated/inspector" in command


def test_rewrite_leaves_non_triage_commands_alone() -> None:
    command = "echo hang detected | tee /logs/hang.log"
    assert (
        rewrite_triage_command(
            command, rpc_host="localhost", rpc_port=50052, inspector_log_path="/x"
        )
        == command
    )


def _shell_words(command: str) -> list[str]:
    """The argv tt-metal's ``std::system`` shell would hand to triage."""
    return shlex.split(command)


def _inspector_options(words: list[str]) -> list[str]:
    """``--inspector-*`` options as docopt reads them, ``--flag value`` and
    ``--flag=value`` alike, in command order."""
    options = []
    i = 0
    while i < len(words):
        word = words[i]
        if word.startswith("--inspector-") and "=" not in word and i + 1 < len(words):
            options.append(f"{word}={words[i + 1]}")
            i += 2
            continue
        if word.startswith("--inspector-"):
            options.append(word)
        i += 1
    return options


@pytest.mark.parametrize(
    "logs_path",
    [
        "/logs/rank logs",
        "/logs/$HOME/x",
        "/logs/it's",
        "/logs/it's here",
        '/logs/say "hi"',
        "/logs/a;rm -rf b",
    ],
)
def test_generated_arguments_are_single_shell_words(logs_path: str) -> None:
    env = {
        TT_METAL_LOGS_PATH_ENV: logs_path,
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: "python triage.py --disable-progress",
    }
    overrides = compute_dp_lane_metal_env(1, 2, env, cwd="/w")
    words = _shell_words(overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV])
    expected_path = f"{logs_path}/dp_rank_1/generated/inspector"
    assert words == [
        "python",
        "triage.py",
        "--inspector-rpc-host=localhost",
        f"--inspector-rpc-port={DEFAULT_INSPECTOR_RPC_PORT + 1}",
        f"--inspector-log-path={expected_path}",
        "--disable-progress",
    ]


def test_only_the_triage_invocation_is_edited() -> None:
    """Flags on other commands in the pipeline are neither moved nor checked."""
    env = {
        TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV: "127.0.0.1:60000",
        TT_METAL_LOGS_PATH_ENV: "/logs",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: (
            "other --inspector-rpc-port=1 && "
            "python triage.py --inspector-rpc-port=60000; "
            "echo --inspector-rpc-port=2 2>&1 | tee /logs/t-$(date +%s).log"
        ),
    }
    overrides = compute_dp_lane_metal_env(2, 4, env, cwd="/w")
    rewritten = overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV]
    assert rewritten.startswith("other --inspector-rpc-port=1 && python triage.py ")
    assert rewritten.endswith(
        "--inspector-rpc-port=60002; echo --inspector-rpc-port=2 2>&1 "
        "| tee /logs/t-$(date +%s).log"
    )
    assert rewritten.count("--inspector-rpc-port=60002") == 1
    assert "--inspector-log-path=/logs/dp_rank_2/generated/inspector" in rewritten


@pytest.mark.parametrize(
    "written",
    [
        "--inspector-log-path=/logs/rank\\ logs/generated/inspector",
        "--inspector-log-path='/logs/rank logs/generated/inspector'",
        '"--inspector-log-path=/logs/rank logs/generated/inspector"',
        "--inspector-log-path '/logs/rank logs/generated/inspector'",
    ],
)
def test_explicit_flag_spellings_are_read_as_shell_words(written: str) -> None:
    env = {
        TT_METAL_LOGS_PATH_ENV: "/logs/rank logs",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: f"python triage.py {written} -v",
    }
    overrides = compute_dp_lane_metal_env(1, 2, env, cwd="/w")
    words = _shell_words(overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV])
    options = _inspector_options(words)
    assert (
        options.count(
            "--inspector-log-path=/logs/rank logs/dp_rank_1/generated/inspector"
        )
        == 1
    )
    assert not any(
        o.startswith("--inspector-log-path=/logs/rank logs/generated") for o in options
    )
    assert words[-1] == "-v"


def test_explicit_value_the_shell_expands_is_left_alone_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    env = {
        TT_METAL_LOGS_PATH_ENV: "/logs",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: (
            'python triage.py --inspector-log-path="$LOGS/generated/inspector" -v'
        ),
    }
    with caplog.at_level(
        logging.WARNING, logger="vllm_tt_plugin.utils.dp_lane_metal_env"
    ):
        overrides = compute_dp_lane_metal_env(1, 2, env, cwd="/w")
    rewritten = overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV]
    assert '--inspector-log-path="$LOGS/generated/inspector" -v' in rewritten
    assert rewritten.count("--inspector-log-path") == 1
    assert "--inspector-rpc-port=50052" in rewritten
    assert any("shell expands later" in r.getMessage() for r in caplog.records)


def test_generated_host_argument_is_quoted_too() -> None:
    env = {
        TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV: "my host:7000",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: "triage",
    }
    overrides = compute_dp_lane_metal_env(0, 2, env, cwd="/w")
    words = _shell_words(overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV])
    assert "--inspector-rpc-host=my host" in words
    assert "--inspector-rpc-port=7000" in words


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_explicit_consistent_client_flags_move_with_the_lane(rank: int) -> None:
    """A setup where triage already pointed at the configured server stays
    consistent on every rank: server and client ports move together."""
    env = {
        TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV: "127.0.0.1:60000",
        TT_METAL_LOGS_PATH_ENV: "/logs",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: (
            "python triage.py --inspector-rpc-host=127.0.0.1 "
            "--inspector-rpc-port 60000 "
            "--inspector-log-path='/logs/generated/inspector' --disable-progress"
        ),
    }
    overrides = compute_dp_lane_metal_env(rank, 4, env, cwd="/w")
    server_host, server_port = parse_inspector_rpc_address(
        overrides[TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV]
    )
    words = _shell_words(overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV])
    assert server_port == 60000 + rank
    assert sorted(_inspector_options(words)) == [
        f"--inspector-log-path=/logs/dp_rank_{rank}/generated/inspector",
        "--inspector-rpc-host=127.0.0.1",
        f"--inspector-rpc-port={server_port}",
    ]
    assert words[-1] == "--disable-progress"
    assert server_host == "127.0.0.1"


@pytest.mark.parametrize(
    "command",
    [
        "python triage.py --inspector-rpc-port=7000",
        "python triage.py --inspector-rpc-host=otherhost",
        "python triage.py --inspector-log-path=/elsewhere",
    ],
)
def test_explicit_contradictory_client_flags_are_rejected(command: str) -> None:
    env = {
        TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV: "127.0.0.1:60000",
        TT_METAL_LOGS_PATH_ENV: "/logs",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: command,
    }
    with pytest.raises(ValueError, match="--inspector-") as excinfo:
        compute_dp_lane_metal_env(2, 4, env, cwd="/w")
    message = str(excinfo.value)
    explicit_value = command.split("=")[-1]
    assert explicit_value in message
    # The message names the configured value the flag was expected to match.
    assert (
        "60000" in message
        or "127.0.0.1" in message
        or "/logs/generated/inspector" in message
    )
    assert command in message


def test_explicit_port_is_kept_when_rpc_is_disabled() -> None:
    """With the server off there is no lane port to move the client to."""
    env = {
        TT_METAL_INSPECTOR_RPC_ENV: "0",
        TT_METAL_LOGS_PATH_ENV: "/logs",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: "triage --inspector-rpc-port=7000",
    }
    overrides = compute_dp_lane_metal_env(1, 2, env, cwd="/w")
    words = _shell_words(overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV])
    assert "--inspector-rpc-port=7000" in words
    assert "--inspector-log-path=/logs/dp_rank_1/generated/inspector" in words


def test_rewrite_is_idempotent() -> None:
    once = rewrite_triage_command(
        _TRIAGE_CMD, rpc_host="localhost", rpc_port=50052, inspector_log_path="/x"
    )
    twice = rewrite_triage_command(
        once, rpc_host="localhost", rpc_port=50052, inspector_log_path="/x"
    )
    assert once == twice


@pytest.mark.parametrize(
    "command",
    [
        "triage --disable-progress",
        "/opt/tt-metal/tools/triage/triage.py",
        "python /opt/tt-metal/scripts/debugging_scripts/triage.py -v",
        "./tools/tt-triage.py",
        "python3 ws/tools/tt-triage.py --disable-progress",
        "python '/opt/tt metal/tools/tt-triage.py' -v",
        'python "ws/tools/triage.py"',
    ],
)
def test_rewrite_finds_the_triage_entry_point(command: str) -> None:
    rewritten = rewrite_triage_command(
        command, rpc_host="localhost", rpc_port=50052, inspector_log_path="/x"
    )
    words = _shell_words(rewritten)
    entry = next(i for i, w in enumerate(words) if w.endswith(("triage", "triage.py")))
    # Flags follow the entry point as whole words; the quoted path survives
    # as one word with its original shell meaning.
    assert words[entry + 1 : entry + 4] == [
        "--inspector-rpc-host=localhost",
        "--inspector-rpc-port=50052",
        "--inspector-log-path=/x",
    ]
    assert words[: entry + 1] == _shell_words(command)[: entry + 1]
    assert words[entry + 4 :] == _shell_words(command)[entry + 1 :]


def test_unrecognised_command_is_left_alone_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    command = "bash -c 'python -m tools.triage.triage'"
    with caplog.at_level(
        logging.WARNING, logger="vllm_tt_plugin.utils.dp_lane_metal_env"
    ):
        rewritten = rewrite_triage_command(
            command, rpc_host="localhost", rpc_port=50052, inspector_log_path="/x"
        )
    assert rewritten == command
    assert any(
        "recognised tt-triage entry point" in record.getMessage()
        and "50052" in record.getMessage()
        for record in caplog.records
    )


def test_rewrite_does_not_match_words_containing_triage() -> None:
    command = "python retriage.py --disable-progress"
    assert (
        rewrite_triage_command(
            command, rpc_host="localhost", rpc_port=50052, inspector_log_path="/x"
        )
        == command
    )


_SHELL_STUBS = """
triage() { printf '%s\\n' "$@" > "$ARGV_FILE"; }
python() { shift; triage "$@"; }
"""


def _argv_through_shell(
    command: str, tmp_path, env: dict[str, str] | None = None
) -> list[str]:
    """The argv the real ``/bin/sh`` hands to triage for ``command``, as
    tt-metal's ``std::system`` would, with ``triage``/``python triage.py``
    stubbed to record their arguments."""
    argv_file = tmp_path / "argv.txt"
    subprocess.run(
        ["/bin/sh", "-c", _SHELL_STUBS + command],
        check=True,
        cwd=tmp_path,
        env={**(env or {}), "PATH": os.environ["PATH"], "ARGV_FILE": str(argv_file)},
    )
    return argv_file.read_text().splitlines()


# The Inspector options as tools/triage/inspector_data.py declares them for
# docopt in tt-metal (usage wrapped; docopt reads it as one pattern).
_INSPECTOR_DATA_USAGE = """
Usage:
    inspector_data [--inspector-rpc-port=<inspector_rpc_port>]
                   [--inspector-rpc-host=<inspector_rpc_host>]
                   [--inspector-log-path=<inspector_log_path>]

Options:
    --inspector-rpc-port=<inspector_rpc_port>  RPC port. [default: 50051]
    --inspector-rpc-host=<inspector_rpc_host>  RPC host. [default: localhost]
    --inspector-log-path=<inspector_log_path>  Inspector log directory.
"""


def _docopt_inspector_options(argv: list[str]) -> dict[str, str]:
    """What triage's own option parser (docopt, a dev dependency) reads from
    the Inspector options in ``argv``, in whichever spelling they arrived."""
    import docopt

    inspector_argv = []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--inspector-"):
            inspector_argv.append(argv[i])
            if "=" not in argv[i] and i + 1 < len(argv):
                inspector_argv.append(argv[i + 1])
                i += 1
        i += 1
    parsed = docopt.docopt(_INSPECTOR_DATA_USAGE, argv=inspector_argv)
    return {k: v for k, v in parsed.items() if k.startswith("--inspector-")}


@pytest.mark.parametrize(
    ("prefix", "entry"),
    [
        ("test -f tools/triage.py && ", "python tools/triage.py"),
        ("mkdir -p /logs/triage && ", "triage"),
        ("cp tools/triage.py /tmp/triage.py; ", "python /tmp/triage.py"),
        ("ls triage | ", "triage"),
        ("command -v triage >/dev/null 2>&1 && ", "triage"),
    ],
)
def test_triage_named_as_an_argument_is_not_the_invocation(
    prefix: str, entry: str
) -> None:
    """A path checked with ``test`` or created with ``mkdir`` has the triage
    basename but is an argument; the invocation that follows is edited."""
    command = f"{prefix}{entry} -v"
    rewritten = rewrite_triage_command(
        command, rpc_host="localhost", rpc_port=50052, inspector_log_path="/x"
    )
    assert rewritten == (
        f"{prefix}{entry} --inspector-rpc-host=localhost "
        "--inspector-rpc-port=50052 --inspector-log-path=/x -v"
    )


@pytest.mark.parametrize(
    "command",
    [
        "ls triage",
        "test -f tools/triage.py",
        "mkdir -p /logs/triage.py",
        "echo tt-triage.py",
        "python -c 'import triage'",
        "python -m triage",
        "python -W triage.py other.py",
        "command -V triage",
        "python other.py # triage.py",
    ],
)
def test_argument_named_triage_alone_is_not_an_invocation(
    command: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(
        logging.WARNING, logger="vllm_tt_plugin.utils.dp_lane_metal_env"
    ):
        rewritten = rewrite_triage_command(
            command, rpc_host="localhost", rpc_port=50052, inspector_log_path="/x"
        )
    assert rewritten == command
    assert any(
        "recognised tt-triage entry point" in r.getMessage() for r in caplog.records
    )


@pytest.mark.parametrize(
    ("prefix", "entry"),
    [
        ("env TT_METAL_HOME=/opt/tt-metal python3 -u -W ignore ", "triage.py"),
        ("TT_METAL_HOME=/opt/tt-metal nohup ", "triage"),
        ("timeout -s KILL 300 python3.12 -B ", "/opt/tt-metal/tools/triage/triage.py"),
        ("exec python -- ", "triage.py"),
        ("$VIRTUAL_ENV/bin/python ", "tools/tt-triage.py"),
        ('"$VENV/bin/python" ', '"$TT_METAL_HOME/tools/tt-triage.py"'),
        ("env -u X -- ", "triage"),
        ("command ", "triage"),
        ("{ python ", "triage.py"),
        ("if [ -f x ]; then python ", "triage.py"),
        ("! ", "triage"),
        ("python \\\n", "triage.py"),
    ],
)
def test_entry_point_is_found_in_executable_and_script_positions(
    prefix: str, entry: str
) -> None:
    suffix = "; }" if prefix.startswith("{") else "; fi" if "then" in prefix else ""
    command = f"{prefix}{entry} -v{suffix}"
    rewritten = rewrite_triage_command(
        command, rpc_host="localhost", rpc_port=50052, inspector_log_path="/x"
    )
    assert rewritten == (
        f"{prefix}{entry} --inspector-rpc-host=localhost "
        f"--inspector-rpc-port=50052 --inspector-log-path=/x -v{suffix}"
    )


@pytest.mark.parametrize(
    "written",
    [
        '"--inspector-log-path=$LOGS/generated/inspector"',
        "'--inspector-log-path='\"$LOGS/generated/inspector\"",
        '--inspector-log-path="$LOGS/generated/inspector"',
        "--inspector-log-path=$LOGS/generated/inspector",
        '--inspector-log-path "$LOGS/generated/inspector"',
        '"--inspector-log-path" "$LOGS"/generated/inspector',
    ],
)
def test_quoted_deferred_option_reaches_triage_exactly_once(
    written: str, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """However the option is quoted, its name is read after quote removal,
    so the deferred value is kept as written and not inserted again."""
    env = {
        TT_METAL_LOGS_PATH_ENV: "/logs",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: f"python triage.py {written} -v",
    }
    with caplog.at_level(
        logging.WARNING, logger="vllm_tt_plugin.utils.dp_lane_metal_env"
    ):
        overrides = compute_dp_lane_metal_env(1, 2, env, cwd="/w")
    rewritten = overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV]
    assert written in rewritten
    assert rewritten.count("--inspector-log-path") == 1
    assert any("shell expands later" in r.getMessage() for r in caplog.records)
    argv = _argv_through_shell(rewritten, tmp_path, env={"LOGS": "/var/logs"})
    assert sorted(_inspector_options(argv)) == [
        "--inspector-log-path=/var/logs/generated/inspector",
        "--inspector-rpc-host=localhost",
        f"--inspector-rpc-port={DEFAULT_INSPECTOR_RPC_PORT + 1}",
    ]
    assert argv[-1] == "-v"


@pytest.mark.parametrize(
    "written",
    [
        '"--inspector-log-path=/logs/it\'s/generated/inspector"',
        '--inspector-log-path="/logs/it\'s/generated/inspector"',
        '--inspector-log-path "/logs/it\'s/generated/inspector"',
        "'--inspector-log-path=/logs/it'\"'\"'s/generated/inspector'",
        "--inspector-log-path=/logs/it\\'s/generated/inspector",
    ],
)
def test_apostrophe_inside_quoted_value_moves_with_the_lane(
    written: str, tmp_path
) -> None:
    env = {
        TT_METAL_LOGS_PATH_ENV: "/logs/it's",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: f"python triage.py {written} -v",
    }
    overrides = compute_dp_lane_metal_env(1, 2, env, cwd="/w")
    rewritten = overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV]
    argv = _argv_through_shell(rewritten, tmp_path)
    lane_path = "/logs/it's/dp_rank_1/generated/inspector"
    assert sorted(_inspector_options(argv)) == [
        f"--inspector-log-path={lane_path}",
        "--inspector-rpc-host=localhost",
        f"--inspector-rpc-port={DEFAULT_INSPECTOR_RPC_PORT + 1}",
    ]
    assert argv[-1] == "-v"


@pytest.mark.parametrize(
    ("written", "redirect"),
    [
        ("--inspector-rpc-port=60000>>OUT", ">>OUT"),
        ("--inspector-rpc-port=60000 2>/dev/null", "2>/dev/null"),
        ("--inspector-log-path /logs/generated/inspector>OUT 2>&1", ">OUT 2>&1"),
        ("--inspector-log-path=/logs/generated/inspector> OUT", "> OUT"),
        ("--inspector-rpc-port 2>&1 60000", "2>&1"),
    ],
)
def test_redirections_are_separated_from_inspector_values(
    written: str, redirect: str, tmp_path
) -> None:
    """``>`` needs no whitespace before it, so a redirection glued to an
    Inspector value is split off the way the shell splits it: the value is
    checked and moved on its own and the redirection text is kept."""
    out = str(tmp_path / "triage.log")
    written = written.replace("OUT", out)
    redirect = redirect.replace("OUT", out)
    env = {
        TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV: "127.0.0.1:60000",
        TT_METAL_LOGS_PATH_ENV: "/logs",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: f"python triage.py {written} -v",
    }
    overrides = compute_dp_lane_metal_env(2, 4, env, cwd="/w")
    rewritten = overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV]
    assert redirect in rewritten
    argv = _argv_through_shell(rewritten, tmp_path)
    assert sorted(_inspector_options(argv)) == [
        "--inspector-log-path=/logs/dp_rank_2/generated/inspector",
        "--inspector-rpc-host=127.0.0.1",
        "--inspector-rpc-port=60002",
    ]
    assert not any(">" in word or "<" in word for word in argv)
    assert argv[-1] == "-v"


def test_line_continuations_and_comments_follow_the_shell(tmp_path) -> None:
    """``\\<newline>`` joins lines and ``#`` starts a comment, so neither
    reaches a flag value or counts as an argument."""
    env = {
        TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV: "127.0.0.1:60000",
        TT_METAL_LOGS_PATH_ENV: "/logs",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: (
            "python triage.py --inspector-rpc-port \\\n    60000 \\\n"
            '    "--inspector-log-path=/logs/generated/\\\ninspector" \\\n'
            "    -v # --inspector-rpc-host=elsewhere\n"
        ),
    }
    overrides = compute_dp_lane_metal_env(2, 4, env, cwd="/w")
    rewritten = overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV]
    assert "# --inspector-rpc-host=elsewhere" in rewritten
    argv = _argv_through_shell(rewritten, tmp_path)
    assert sorted(_inspector_options(argv)) == [
        "--inspector-log-path=/logs/dp_rank_2/generated/inspector",
        "--inspector-rpc-host=127.0.0.1",
        "--inspector-rpc-port=60002",
    ]
    assert argv[-1] == "-v"


@pytest.mark.parametrize(
    ("command", "shell_env", "expected"),
    [
        (
            'python triage.py "--inspector-log-path=$LOGS/generated/inspector" -v',
            {"LOGS": "/var/logs"},
            {
                "--inspector-log-path": "/var/logs/generated/inspector",
                "--inspector-rpc-host": "127.0.0.1",
                "--inspector-rpc-port": "60002",
            },
        ),
        (
            'python triage.py "--inspector-log-path=/logs/it\'s/generated/inspector"',
            {},
            {
                "--inspector-log-path": "/logs/it's/dp_rank_2/generated/inspector",
                "--inspector-rpc-host": "127.0.0.1",
                "--inspector-rpc-port": "60002",
            },
        ),
        (
            "python triage.py --inspector-rpc-port 2>&1 60000 >>OUT",
            {},
            {
                "--inspector-log-path": "/logs/it's/dp_rank_2/generated/inspector",
                "--inspector-rpc-host": "127.0.0.1",
                "--inspector-rpc-port": "60002",
            },
        ),
    ],
)
def test_triage_option_parser_reads_the_rewritten_command(
    command: str, shell_env: dict[str, str], expected: dict[str, str], tmp_path
) -> None:
    """Consumer-side check: what the shell hands over is what tt-metal's own
    docopt usage for the Inspector options parses, each option once."""
    env = {
        TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV: "127.0.0.1:60000",
        TT_METAL_LOGS_PATH_ENV: "/logs/it's",
        TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV: command.replace(
            "OUT", str(tmp_path / "triage.log")
        ),
    }
    overrides = compute_dp_lane_metal_env(2, 4, env, cwd="/w")
    argv = _argv_through_shell(
        overrides[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV], tmp_path, env=shell_env
    )
    assert _docopt_inspector_options(argv) == expected


def _standard_dp_config(num_groups: int, local_dp_rank: int | None) -> SimpleNamespace:
    groups = [f"{8 * i},{8 * i + 1}" for i in range(num_groups)]
    return SimpleNamespace(
        additional_config={"_tt_standard_dp_visible_groups": groups},
        parallel_config=SimpleNamespace(data_parallel_rank_local=local_dp_rank),
    )


def test_worker_binding_applies_overrides_to_os_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TT_METAL_LOGS_PATH_ENV, "/logs")
    monkeypatch.setenv(TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV, _TRIAGE_CMD)
    monkeypatch.delenv(TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV, raising=False)
    monkeypatch.delenv(TT_METAL_INSPECTOR_RPC_ENV, raising=False)

    _bind_dp_lane_metal_env(_standard_dp_config(num_groups=4, local_dp_rank=2))

    assert os.environ[TT_METAL_LOGS_PATH_ENV] == "/logs/dp_rank_2"
    assert (
        os.environ[TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV]
        == f"localhost:{DEFAULT_INSPECTOR_RPC_PORT + 2}"
    )
    assert (
        f"--inspector-rpc-port={DEFAULT_INSPECTOR_RPC_PORT + 2}"
        in os.environ[TT_METAL_DISPATCH_TIMEOUT_COMMAND_ENV]
    )


def test_worker_binding_keeps_inherited_env_without_discovered_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPI launches store no groups; tt-metal offsets by MPI rank itself."""
    monkeypatch.setenv(TT_METAL_LOGS_PATH_ENV, "/logs")
    monkeypatch.setenv(TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV, "localhost:50051")
    config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(data_parallel_rank_local=1),
    )

    _bind_dp_lane_metal_env(config)

    assert os.environ[TT_METAL_LOGS_PATH_ENV] == "/logs"
    assert os.environ[TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV] == "localhost:50051"


def test_worker_binding_single_lane_keeps_inherited_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TT_METAL_LOGS_PATH_ENV, "/logs")
    monkeypatch.delenv(TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV, raising=False)

    _bind_dp_lane_metal_env(_standard_dp_config(num_groups=1, local_dp_rank=0))

    assert os.environ[TT_METAL_LOGS_PATH_ENV] == "/logs"
    assert TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS_ENV not in os.environ
