# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Data-parallel submesh placement for standard multi-process DP discovery.

The policy matches the ``create_submeshes`` helper that tt-metal's model
generators and tt-transformers' ``mesh_utils`` use for in-process
(lane-mode) data parallelism, so a model placed by either path lands on the
same devices:

- a 32-device mesh is viewed as 4x8 and split into row-oriented groups, so
  DP=4 gives four 1x8 submeshes and DP=2 gives two 2x8 submeshes; groups of
  fewer than 8 devices are 1xN rows;
- any other mesh is split into 1xN rows.

The plugin owns this copy so discovery needs only ``ttnn``, not a model
package. ``tests/test_dp_submeshes.py`` pins every placement above; change
the shared policy and this copy together.
"""

__all__ = ("create_dp_submeshes", "dp_submesh_shapes")

import logging

logger = logging.getLogger(__name__)

_GALAXY_DEVICES = 32
_GALAXY_GRID = (4, 8)


def dp_submesh_shapes(
    mesh_shape: tuple[int, int], data_parallel: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``(parent_shape, submesh_shape)`` for splitting ``mesh_shape``.

    ``parent_shape`` is the shape the parent mesh must have before it is
    split; it differs from ``mesh_shape`` only for a 32-device mesh that is
    not already 4x8.

    Examples
    --------
    >>> dp_submesh_shapes((1, 8), 4)
    ((1, 8), (1, 2))
    >>> dp_submesh_shapes((8, 4), 4)
    ((4, 8), (1, 8))
    """
    num_rows, num_cols = (int(dim) for dim in mesh_shape)
    num_devices = num_rows * num_cols
    if data_parallel < 1 or num_devices % data_parallel != 0:
        raise ValueError(
            f"Unsupported device split: {num_devices} devices, {data_parallel} groups"
        )
    devices_per_group = num_devices // data_parallel

    if num_devices == _GALAXY_DEVICES:
        if devices_per_group >= 8 and devices_per_group % 8 == 0:
            return _GALAXY_GRID, (devices_per_group // 8, 8)
        return _GALAXY_GRID, (1, devices_per_group)

    return (num_rows, num_cols), (1, devices_per_group)


def create_dp_submeshes(mesh_device, data_parallel: int) -> list:
    """Split ``mesh_device`` into ``data_parallel`` submeshes.

    ``data_parallel == 1`` returns the parent mesh itself. A 32-device mesh is
    reshaped in place to 4x8 first.
    """
    if data_parallel == 1:
        return [mesh_device]

    import ttnn

    mesh_shape = tuple(int(dim) for dim in mesh_device.shape)
    parent_shape, submesh_shape = dp_submesh_shapes(mesh_shape, data_parallel)
    if parent_shape != mesh_shape:
        logger.info(
            "Reshaping %d-device mesh from %s to %s for DP submeshes",
            parent_shape[0] * parent_shape[1],
            mesh_shape,
            parent_shape,
        )
        mesh_device.reshape(ttnn.MeshShape(*parent_shape))
    return mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))
