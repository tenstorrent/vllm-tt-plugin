# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The workers for TT vLLM-Omni stages."""

from vllm_omni.worker.mixins import OmniWorkerMixin

from vllm_tt_plugin.omni.generation_model_runner import TTOmniGenerationModelRunner
from vllm_tt_plugin.omni.model_runner import TTOmniARModelRunner
from vllm_tt_plugin.omni.scheduler import TTOmniARScheduler
from vllm_tt_plugin.worker import TTWorker


class TTOmniARWorker(OmniWorkerMixin, TTWorker):
    model_runner_cls = TTOmniARModelRunner
    # Omni's AR schedulers are its bases, so the pairing rule replaces either of them
    # with this one.
    tt_scheduler_cls = TTOmniARScheduler


class TTOmniGenerationWorker(OmniWorkerMixin, TTWorker):
    model_runner_cls = TTOmniGenerationModelRunner
    # Each step runs every scheduled chunk whole, with no prefill or decode split,
    # so Omni's own generation scheduler runs the stage unchanged.
    tt_scheduler_cls = (
        "vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler"
    )
    uses_token_kv_cache = False
