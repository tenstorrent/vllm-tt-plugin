# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""DP discovery utilities to support upstream vLLM's standard multi-process DP mode."""

import ast
import logging
from contextlib import suppress

logger = logging.getLogger(__name__)

StandardDPAssignmentT = tuple[str, tuple[int, int]]

_MESH_GRID_PRESETS = {
    "N150": (1, 1),
    "P100": (1, 1),
    "P150": (1, 1),
    "P150x2": (1, 2),
    "N300": (1, 2),
    "P300": (1, 2),
    "N150x4": (1, 4),
    "P150x4": (1, 4),
    "T3K": (1, 8),
    "P150x8": (1, 8),
    "P300x2": (1, 4),
}


def _parse_mesh_grid(
    mesh_device_env: str | None,
    num_devices_available: int,
    *,
    tg_mesh_grid: tuple[int, int],
) -> tuple[int, int]:
    """Parses one TT mesh preset or tuple into a concrete grid.

    Examples
    --------
    >>> _parse_mesh_grid("T3K", 8, tg_mesh_grid=(4, 8))
    (1, 8)
    >>> _parse_mesh_grid("(2, 4)", 8, tg_mesh_grid=(4, 8))
    (2, 4)
    """
    mesh_grid_dict = dict(_MESH_GRID_PRESETS)
    mesh_grid_dict["TG"] = tg_mesh_grid

    if mesh_device_env is None:
        return (1, num_devices_available)

    try:
        parsed_value = ast.literal_eval(mesh_device_env)
        if isinstance(parsed_value, (tuple, list)) and len(parsed_value) == 2:
            return tuple(int(dim) for dim in parsed_value)
        raise ValueError("Not a valid tuple")

    except (ValueError, SyntaxError, TypeError):
        mesh_grid = mesh_grid_dict.get(mesh_device_env)
        if mesh_grid is None:
            raise ValueError(
                f"Invalid MESH_DEVICE: {mesh_device_env}. "
                f"Expected one of: {list(mesh_grid_dict.keys())}"
            ) from None
        return mesh_grid


def _resolve_parent_mesh_grid(
    mesh_device_env: str | None,
    num_devices_available: int,
) -> tuple[int, int]:
    """Normalizes the parent mesh grid to the visible device count.

    Examples
    --------
    >>> _resolve_parent_mesh_grid("T3K", 8)
    (1, 8)
    """
    mesh_grid = _parse_mesh_grid(
        mesh_device_env,
        num_devices_available,
        tg_mesh_grid=(4, 8),
    )

    if mesh_grid[0] * mesh_grid[1] != num_devices_available:
        mesh_grid = (1, num_devices_available)

    return mesh_grid


def _maybe_reorder_standard_dp_visible_device_groups(
    device_groups: list[StandardDPAssignmentT],
    mesh_grid: tuple[int, int],
    data_parallel_size: int,
) -> list[StandardDPAssignmentT]:
    """Reorders TT single-host DP groups for known hardware layouts.

    Examples
    --------
    >>> groups = [("0,1", (1, 2)), ("2,3", (1, 2))]
    >>> _maybe_reorder_standard_dp_visible_device_groups(groups, (1, 2), 2) == groups
    True
    >>> # Example for WH Galaxy DP=4, which is a known special case where the default
    >>> # row-major order is not mesh-id order.
    >>> groups = [
    ...     ("0,1,2,3,4,5,6,7", (1, 8)),
    ...     ("8,9,10,11,12,13,14,15", (1, 8)),
    ...     ("16,17,18,19,20,21,22,23", (1, 8)),
    ...     ("24,25,26,27,28,29,30,31", (1, 8))]
    >>> _maybe_reorder_standard_dp_visible_device_groups(groups, (4, 8), 4)
    [('0,1,2,3,4,5,6,7', (1, 8)),
     ('16,17,18,19,20,21,22,23', (1, 8)),
     ('24,25,26,27,28,29,30,31', (1, 8)),
     ('8,9,10,11,12,13,14,15', (1, 8))]
    """
    import ttnn

    if (
        ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
        and mesh_grid == (4, 8)
        and data_parallel_size == 4
        and len(device_groups) == 4
    ):
        reordered_groups = [device_groups[index] for index in (0, 2, 3, 1)]
        logger.info(
            "Reordered TT single-host DP device groups for WH Galaxy DP=4 "
            "from row-major %s to mesh-id order %s",
            [visible_devices for visible_devices, _shape in device_groups],
            [visible_devices for visible_devices, _shape in reordered_groups],
        )
        return reordered_groups

    return device_groups


def _split_standard_dp_discovery_result(
    discovery_result: list[str] | list[StandardDPAssignmentT] | None,
) -> tuple[list[str] | None, dict[str, tuple[int, int]]]:
    """Splits discovery output into visible-device and mesh-grid views.

    Examples
    --------
    >>> _split_standard_dp_discovery_result(None)
    (None, {})
    >>> _split_standard_dp_discovery_result([("0,1", (1, 2))])
    (["0,1"], {"0,1": (1, 2)})
    """
    if discovery_result is None:
        return None, {}
    if not discovery_result:
        return [], {}

    first_entry = discovery_result[0]
    if isinstance(first_entry, str):
        return discovery_result, {}

    assignments = discovery_result
    return (
        [visible_devices for visible_devices, _mesh_grid in assignments],
        {visible_devices: mesh_grid for visible_devices, mesh_grid in assignments},
    )


def _discover_standard_dp_visible_device_groups(
    mesh_device_env: str | None,
    data_parallel_size: int,
) -> list[StandardDPAssignmentT]:
    """Discovers TT visible-device groups for one single-host DP layout.

    Notes
    -----
    This helper requires a live TT runtime and submesh creation support.

    Examples
    --------
    >>> _discover_standard_dp_visible_device_groups("T3K", 4)
    [("0,1,2,3,4,5,6,7", (1, 8)), ...]
    """
    import ttnn
    from models.tt_transformers.tt.generator import create_submeshes

    mesh_device = None
    submeshes = []

    try:
        num_devices_available = ttnn.get_num_devices()
        mesh_grid = _resolve_parent_mesh_grid(mesh_device_env, num_devices_available)
        mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_grid))
        submeshes = create_submeshes(mesh_device, data_parallel_size)
        if len(submeshes) != data_parallel_size:
            raise RuntimeError(
                "TT create_submeshes returned "
                f"{len(submeshes)} groups for data_parallel_size={data_parallel_size}"
            )

        device_groups = []
        for dp_rank, submesh in enumerate(submeshes):
            device_ids = list(submesh.get_device_ids())
            if not device_ids:
                raise RuntimeError(f"TT DP rank {dp_rank} resolved to an empty submesh")
            device_groups.append(
                (
                    ",".join(str(device_id) for device_id in device_ids),
                    tuple(int(dim) for dim in submesh.shape),
                )
            )

        device_groups = _maybe_reorder_standard_dp_visible_device_groups(
            device_groups,
            mesh_grid,
            data_parallel_size,
        )

        logger.info(
            "Resolved TT single-host DP device groups: %s",
            [
                f"{visible_devices}@{mesh_shape}"
                for visible_devices, mesh_shape in device_groups
            ],
        )

        return device_groups

    finally:
        for submesh in submeshes:
            with suppress(Exception):
                ttnn.close_mesh_device(submesh)
        if mesh_device is not None:
            with suppress(Exception):
                ttnn.close_mesh_device(mesh_device)


def _run_standard_dp_visible_device_group_discovery(
    conn,
    mesh_device_env: str | None,
    data_parallel_size: int,
) -> None:
    """Sends discovered TT device groups back over a pipe.

    This is the spawned child-process entrypoint for standard-DP discovery.
    """
    try:
        conn.send(
            (
                "ok",
                _discover_standard_dp_visible_device_groups(
                    mesh_device_env, data_parallel_size
                ),
            )
        )
    except Exception as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()
