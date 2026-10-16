# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The worker for a TT autoregressive vLLM-Omni stage."""

from vllm_omni.worker.mixins import OmniWorkerMixin

from vllm_tt_plugin.omni.model_runner import TTOmniARModelRunner
from vllm_tt_plugin.omni.scheduler import TTOmniARScheduler
from vllm_tt_plugin.worker import TTWorker


class TTOmniARWorker(OmniWorkerMixin, TTWorker):
    model_runner_cls = TTOmniARModelRunner
    # Omni's AR schedulers are its bases, so the pairing rule replaces either of them
    # with this one.
    tt_scheduler_cls = TTOmniARScheduler
