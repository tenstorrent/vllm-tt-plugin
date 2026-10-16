# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The scheduler for a TT autoregressive vLLM-Omni stage.

``OmniARAsyncScheduler`` comes first in the MRO, so Omni's request handling runs once
per step around everything else: aborts, stage-input gating, re-wrapping new requests
with their ``additional_information``, and sending each step's stage outputs onward.
``TTScheduler`` sits under it and splits the step into prefill or decode, as on any TT
model. The resolved order is TTOmniARScheduler, OmniARAsyncScheduler, OmniARScheduler,
OmniSchedulerMixin, TTScheduler, AsyncScheduler, Scheduler.

Two seams need joining:

  * Omni re-wraps the ``SchedulerOutput`` into an ``OmniSchedulerOutput`` from its
    dataclass fields alone, which drops the per-step state TT attaches beside them.
  * Omni's ``update_from_output`` replaces the base loop instead of extending it, so
    ``TTScheduler.update_from_output`` never runs; its commit bookkeeping runs here.
"""

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler

from vllm_tt_plugin.scheduler import TTScheduler

# What TT attaches to a SchedulerOutput beside its dataclass fields.
TT_SCHEDULER_OUTPUT_PREFIX = "_tt_"


class TTOmniARScheduler(OmniARAsyncScheduler, TTScheduler):
    def _wrap_omni_scheduler_output(self, base: SchedulerOutput, **kwargs):
        wrapped = super()._wrap_omni_scheduler_output(base, **kwargs)
        for name, value in vars(base).items():
            if name.startswith(TT_SCHEDULER_OUTPUT_PREFIX):
                setattr(wrapped, name, value)
        return wrapped

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        with self._committing_step(scheduler_output):
            return super().update_from_output(scheduler_output, model_runner_output)
