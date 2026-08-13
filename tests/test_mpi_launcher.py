# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Unit tests for ``tt.rank_binding`` parsing on the explicit TT MPI launch path.

The MPI launcher is not supported by the standalone plugin: it needs
``CoreEngineLauncher`` / ``EngineLaunchPlan``, which exist only in the
tenstorrent/vllm fork, so the whole module skips on stock vLLM.
"""

import pathlib
from types import SimpleNamespace

import pytest

try:
    from vllm_tt_plugin.launcher import parse_tt_mpi_params
except ImportError as exc:
    parse_tt_mpi_params = None
    _LAUNCHER_IMPORT_ERROR: str | None = str(exc)
else:
    _LAUNCHER_IMPORT_ERROR = None

pytestmark = pytest.mark.skipif(
    _LAUNCHER_IMPORT_ERROR is not None,
    reason=f"vllm_tt_plugin.launcher unavailable: {_LAUNCHER_IMPORT_ERROR}",
)


def test_standard_dp_uses_all_device_ranks(
    tmp_path: pathlib.Path,
    vllm_config: SimpleNamespace,
) -> None:
    rank_binding = tmp_path / "rank_binding.json"
    rank_binding.write_text(
        "rank_bindings:\n"
        "  - rank: 0\n"
        "    mesh_id: 0\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "0"\n'
        "\n"
        "  - rank: 1\n"
        "    mesh_id: 1\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "3"\n'
        "\n"
        "  - rank: 2\n"
        "    mesh_id: 2\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "1"\n'
        "\n"
        "  - rank: 3\n"
        "    mesh_id: 3\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "2"\n'
    )

    vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
    vllm_config.parallel_config.data_parallel_backend = "mp"
    vllm_config.parallel_config.data_parallel_size = 4

    parsed_rank_binding, non_device_dp_ranks = parse_tt_mpi_params(vllm_config)

    assert parsed_rank_binding == str(rank_binding)
    assert non_device_dp_ranks == set()


def test_standard_dp_rejects_mismatched_mpi_world(
    tmp_path: pathlib.Path,
    vllm_config: SimpleNamespace,
) -> None:
    rank_binding = tmp_path / "rank_binding.json"
    rank_binding.write_text(
        "rank_bindings:\n"
        "  - rank: 0\n"
        "    mesh_id: 0\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "0, 1"\n'
        "\n"
        "  - rank: 1\n"
        "    mesh_id: 1\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "2, 3"\n'
    )

    vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
    vllm_config.parallel_config.data_parallel_backend = "mp"
    vllm_config.parallel_config.data_parallel_size = 4

    with pytest.raises(
        RuntimeError,
        match="Standard DP mode requires one TT MPI rank per DP rank",
    ):
        parse_tt_mpi_params(vllm_config)


def test_explicit_mpi_args_require_rank_binding(
    vllm_config: SimpleNamespace,
) -> None:
    vllm_config.additional_config = {"tt": {"mpi_args": "--host hostA"}}
    vllm_config.parallel_config.data_parallel_backend = "mp"
    vllm_config.parallel_config.data_parallel_size = 4

    with pytest.raises(
        RuntimeError,
        match="TT explicit MPI launch requires tt.rank_binding",
    ):
        parse_tt_mpi_params(vllm_config)


def test_multinode_requires_rank_binding(
    vllm_config: SimpleNamespace,
) -> None:
    vllm_config.additional_config = {"tt": {}}
    vllm_config.parallel_config.data_parallel_backend = "mp"
    vllm_config.parallel_config.data_parallel_size = 4
    vllm_config.parallel_config.nnodes = 2

    with pytest.raises(
        RuntimeError,
        match="TT explicit MPI launch requires tt.rank_binding",
    ):
        parse_tt_mpi_params(vllm_config)


def test_rank_binding_requires_visible_devices(
    tmp_path: pathlib.Path,
    vllm_config: SimpleNamespace,
) -> None:
    rank_binding = tmp_path / "rank_binding.json"
    rank_binding.write_text(
        "rank_bindings:\n"
        "  - rank: 0\n"
        "    mesh_id: 0\n"
        "    env_overrides:\n"
        "      OTHER_ENV: foo\n"
        "  - rank: 1\n"
        "    mesh_id: 1\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "1"\n'
    )

    vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
    vllm_config.parallel_config.data_parallel_backend = "mp"
    vllm_config.parallel_config.data_parallel_size = 2

    with pytest.raises(RuntimeError, match="TT_VISIBLE_DEVICES"):
        parse_tt_mpi_params(vllm_config)


def test_rank_binding_rejects_overlapping_visible_devices(
    tmp_path: pathlib.Path,
    vllm_config: SimpleNamespace,
) -> None:
    rank_binding = tmp_path / "rank_binding.json"
    rank_binding.write_text(
        "rank_bindings:\n"
        "  - rank: 0\n"
        "    mesh_id: 0\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "0, 1"\n'
        "  - rank: 1\n"
        "    mesh_id: 1\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "1, 2"\n'
    )

    vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
    vllm_config.parallel_config.data_parallel_backend = "mp"
    vllm_config.parallel_config.data_parallel_size = 2

    with pytest.raises(
        RuntimeError,
        match="overlaps TT_VISIBLE_DEVICES assignments",
    ):
        parse_tt_mpi_params(vllm_config)


def test_rank_binding_rejects_duplicate_rank_ids(
    tmp_path: pathlib.Path,
    vllm_config: SimpleNamespace,
) -> None:
    rank_binding = tmp_path / "rank_binding.json"
    rank_binding.write_text(
        "rank_bindings:\n"
        "  - rank: 0\n"
        "    mesh_id: 0\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "0"\n'
        "  - rank: 0\n"
        "    mesh_id: 1\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "1"\n'
    )

    vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
    vllm_config.parallel_config.data_parallel_backend = "mp"
    vllm_config.parallel_config.data_parallel_size = 2

    with pytest.raises(RuntimeError, match="duplicate rank 0"):
        parse_tt_mpi_params(vllm_config)


def test_legacy_tt_dp_override_is_ignored_by_launcher(
    tmp_path: pathlib.Path,
    vllm_config: SimpleNamespace,
) -> None:
    rank_binding = tmp_path / "rank_binding.json"
    rank_binding.write_text(
        "rank_bindings:\n"
        "  - rank: 0\n"
        "    mesh_id: 0\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "0"\n'
        "\n"
        "  - rank: 1\n"
        "    mesh_id: 1\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "1"\n'
        "\n"
        "  - rank: 2\n"
        "    mesh_id: 2\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "2"\n'
        "\n"
        "  - rank: 3\n"
        "    mesh_id: 3\n"
        "    env_overrides:\n"
        '      TT_VISIBLE_DEVICES: "3"\n'
    )

    vllm_config.additional_config = {
        "tt": {
            "rank_binding": str(rank_binding),
            "tt_data_parallel_size": 4,
        }
    }
    vllm_config.parallel_config.data_parallel_backend = "mp"
    vllm_config.parallel_config.data_parallel_size = 4

    parsed_rank_binding, non_device_dp_ranks = parse_tt_mpi_params(vllm_config)

    assert parsed_rank_binding == str(rank_binding)
    assert non_device_dp_ranks == set()
