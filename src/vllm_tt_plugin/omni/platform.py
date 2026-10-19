# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The TT platform as vLLM-Omni sees it.

vLLM-Omni resolves its own platform beside vLLM's ``current_platform``, which stays
``TTPlatform``. Omni asks this one which worker runs each stage and which environment
variable confines a stage process to its devices; ``TTPlatform`` already answers the
second with ``TT_VISIBLE_DEVICES``.
"""

import glob

import torch
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

from vllm_tt_plugin.platform import TTPlatform

TT_OMNI_AR_WORKER_CLS = "vllm_tt_plugin.omni.worker.TTOmniARWorker"
TT_OMNI_GENERATION_WORKER_CLS = "vllm_tt_plugin.omni.worker.TTOmniGenerationWorker"


class TTOmniPlatform(OmniPlatform, TTPlatform):
    _omni_enum = OmniPlatformEnum.OOT

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return TT_OMNI_AR_WORKER_CLS

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return TT_OMNI_GENERATION_WORKER_CLS

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/deploy"

    @classmethod
    def get_device_count(cls) -> int:
        """Devices this host exposes, counted from the driver's device nodes.

        Opening a cluster here would fix the process's visible set before any stage
        binds its own.
        """
        return len(glob.glob("/dev/tenstorrent/[0-9]*"))

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        # Torch tensors in the TT path live on the host; ttnn owns the device.
        return torch.device("cpu")

    @classmethod
    def synchronize(cls) -> None:
        """Nothing torch-side to wait for; ttnn synchronizes its own work."""

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def supports_talker_mtp_graph_capture(cls) -> bool:
        return False
