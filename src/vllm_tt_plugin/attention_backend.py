# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The attention backend class vLLM gets when it asks TT for one.

TT models run their own attention on device, so no vLLM attention layer or
kernel exists. vLLM code outside the forward pass still asks the platform for
a backend class: ``NixlConnector`` compares its name between two servers and
reads its kernel block sizes to size transferred blocks.

Principal entry point: ``TTAttentionBackend``, named by
``TTPlatform.get_attn_backend_cls``.
"""

from vllm.v1.attention.backend import AttentionBackend


class TTAttentionBackend(AttentionBackend):
    """A backend that only names TT and keeps vLLM's default block sizes.

    Asking it for an implementation or metadata builder raises: nothing in
    vLLM runs attention on TT.
    """

    @staticmethod
    def get_name() -> str:
        """The name two servers compare before exchanging KV."""
        return "TT"

    @staticmethod
    def get_impl_cls():
        """Raise: TT models run their own attention on device.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("TT models run their own attention on device.")

    @staticmethod
    def get_builder_cls():
        """Raise: TT models build their own attention inputs.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("TT models run their own attention on device.")
