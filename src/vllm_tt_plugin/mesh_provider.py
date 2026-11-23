# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Shared-mesh mode: a process-level source of the mesh every TT worker uses.

A chip can be opened by one process at a time, so several models that share
chips must share one process and one mesh. A host that runs several engines in
its process registers a provider before the first worker initializes its
device. Each ``TTWorker`` then acquires its mesh from the provider instead of
opening one, and at shutdown releases it to that provider instead of closing it
and resetting fabric. The provider's owner closes the mesh after every worker
has released it.

With no provider registered, the worker opens and closes its own mesh.
"""

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import ttnn


class MeshProvider(Protocol):
    def acquire(
        self,
        tt_config: dict[str, Any] | None,
        trace_mode: str,
        *,
        model_fabric_config: dict[str, Any] | None,
    ) -> "ttnn.MeshDevice":
        """Return the mesh for one worker.

        Receives the arguments the worker would otherwise pass to
        ``vllm_tt_plugin.worker.open_mesh_device``, so a provider can open the
        mesh exactly as the worker would, or check that a mesh it already
        holds suits this model.
        """
        ...

    def release(self, mesh_device: "ttnn.MeshDevice") -> None:
        """Take back a mesh the worker no longer uses; called once per acquire.

        The worker has already released its model's captures and KV caches.
        The mesh must stay open: other workers may still use it.
        """
        ...


_provider: MeshProvider | None = None


def set_mesh_provider(provider: MeshProvider | None) -> None:
    """Register the provider for workers that initialize after this call.

    ``None`` restores the default. A worker keeps the provider it acquired
    from, so changing it does not affect a worker that already holds a mesh.
    """
    global _provider
    _provider = provider


def get_mesh_provider() -> MeshProvider | None:
    return _provider
