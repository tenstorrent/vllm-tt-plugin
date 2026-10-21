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
    ``TTScheduler.update_from_output`` never runs; its output-frame release runs here.
"""

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler

from vllm_tt_plugin.config import get_tt_output_tokens_per_step
from vllm_tt_plugin.scheduler import TTScheduler, carry_tt_scheduler_output_fields


class TTOmniARScheduler(OmniARAsyncScheduler, TTScheduler):
    @classmethod
    def validate_tt_config(cls, vllm_config) -> None:
        # Block output and speculative decoding commit several tokens per step
        # through Omni's update loop, which no TT stage has run. Lane mode is
        # refused earlier, when the scheduler is paired with the worker.
        conflicts = []
        output_tokens_per_step = get_tt_output_tokens_per_step(vllm_config)
        if output_tokens_per_step > 1:
            conflicts.append(
                f"block output (output_tokens_per_step={output_tokens_per_step})"
            )
        if vllm_config.speculative_config is not None:
            conflicts.append("speculative decoding")
        if conflicts:
            raise ValueError(
                "a TT vLLM-Omni AR stage runs one-token steps; got "
                + ", ".join(conflicts)
            )

    def _wrap_omni_scheduler_output(self, base: SchedulerOutput, **kwargs):
        wrapped = super()._wrap_omni_scheduler_output(base, **kwargs)
        carry_tt_scheduler_output_fields(base, wrapped)
        return wrapped

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        try:
            return super().update_from_output(scheduler_output, model_runner_output)
        finally:
            self.tt_release_output_frames(scheduler_output)
