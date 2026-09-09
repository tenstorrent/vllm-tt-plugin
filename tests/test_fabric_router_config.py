# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from unittest.mock import Mock

import pytest

from vllm_tt_plugin import worker


@pytest.mark.parametrize("payload", [None, 8192])
def test_router_payload_reaches_fabric_initialization(monkeypatch, payload):
    monkeypatch.setattr(
        worker, "get_fabric_config", lambda *_: worker.ttnn.FabricConfig.FABRIC_1D_RING
    )
    initialize = Mock()
    monkeypatch.setattr(worker.ttnn, "set_fabric_config", initialize)
    config = (
        {} if payload is None else {"fabric_max_packet_payload_size_bytes": payload}
    )
    worker.set_fabric(config, 4)
    expected = (
        worker.ttnn.FabricRouterConfig().max_packet_payload_size_bytes
        if payload is None
        else payload
    )
    assert (
        initialize.call_args.kwargs["router_config"].max_packet_payload_size_bytes
        == expected
    )


@pytest.mark.parametrize("payload", [True, 0, -1, "8192", 8192.0])
def test_invalid_payload_never_initializes_fabric(monkeypatch, payload):
    monkeypatch.setattr(
        worker, "get_fabric_config", lambda *_: worker.ttnn.FabricConfig.FABRIC_1D_RING
    )
    initialize = Mock()
    monkeypatch.setattr(worker.ttnn, "set_fabric_config", initialize)
    with pytest.raises(ValueError):
        worker.set_fabric({"fabric_max_packet_payload_size_bytes": payload}, 4)
    initialize.assert_not_called()
