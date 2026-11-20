# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The attention backend class vLLM is given when it asks TT for one.

TT models run their own attention on device, so no vLLM attention layer or
kernel exists. vLLM code outside the forward pass still asks the platform for
a backend class: ``NixlConnector`` reads its name to check that two servers
are compatible, and its kernel block sizes to size transferred blocks.
"""

from vllm.v1.attention.backend import AttentionBackend


class TTAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TT"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("TT models run their own attention on device.")

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError("TT models run their own attention on device.")
