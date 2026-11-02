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


@pytest.mark.parametrize(
    "logs_path",
    [
        "/logs/rank logs",
        "/logs/$HOME/x",
        "/logs/it's",
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
    assert words.count("--inspector-rpc-host=127.0.0.1") == 1
    assert f"--inspector-rpc-port={server_port}" in words
    assert f"--inspector-log-path=/logs/dp_rank_{rank}/generated/inspector" in words
    assert len([w for w in words if w.startswith("--inspector-")]) == 3
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
