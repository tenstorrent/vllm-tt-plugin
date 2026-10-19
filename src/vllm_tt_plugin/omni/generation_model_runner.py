# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The runner for a TT vLLM-Omni generation stage, such as Qwen3-TTS Code2Wav.

The stage has no tokens to sample and no token KV cache. Each step hands the stage
model every scheduled request's current chunk and returns the model's per-request
payload, as vLLM-Omni's ``GPUGenerationModelRunner`` (v1) does:

  * In: Omni's scheduler replaces a streamed request's prompt with each new chunk of
    codes and merges the chunk's metadata into its ``additional_information``. The
    model gets the chunks flat, split by ``seq_token_counts``, with each request's
    merged metadata and its scheduler id; a stateful decoder keys its state on that id.
  * Out: one payload per request, ``{"model_outputs": samples, "sr": rate}``.
  * Finished or aborted requests reach the model's ``on_requests_finished``, which
    frees their decoder state.

The model is the class registered for the stage's architecture, built on the host.
"""

import copy
from collections.abc import Mapping

import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.utils.mm_outputs import partition_payload_list
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)


def merge_info(merged: dict, update: dict) -> None:
    """Merge one level deep, as Omni's runner merges a request's buffer."""
    for key, value in update.items():
        if isinstance(value, dict):
            merged.setdefault(key, {}).update(value)
        else:
            merged[key] = value


def per_request_payloads(multimodal_outputs, num_reqs: int) -> list[dict]:
    """Split the model's ``{key: [one entry per request]}`` into one dict per row."""
    if not isinstance(multimodal_outputs, Mapping):
        raise TypeError(
            "a TT vLLM-Omni generation stage model returns multimodal_outputs as a "
            f"mapping of per-request lists, got {type(multimodal_outputs)}"
        )
    payloads: list[dict] = [{} for _ in range(num_reqs)]
    for key, values in multimodal_outputs.items():
        if not isinstance(values, list | tuple) or len(values) != num_reqs:
            raise ValueError(
                f"multimodal output {key!r} needs one entry per request: expected "
                f"{num_reqs}, got {values!r:.200}"
            )
        for payload, value in zip(payloads, values, strict=True):
            payload[key] = torch.as_tensor(value).detach().cpu().contiguous()
    return payloads


class TTOmniGenerationModelRunner(OmniConnectorModelRunnerMixin):
    # Omni's connector check requires the mixin. With the v1 runner and async_chunk,
    # Omni's scheduler receives the chunks itself, so this runner opens no connector.

    def __init__(
        self,
        vllm_config,
        mesh_device,
        trace_mode,
        enable_model_warmup,
        num_devices,
    ):
        model_config = vllm_config.model_config
        if getattr(model_config, "use_v2_model_runner", False):
            raise ValueError(
                "a TT vLLM-Omni stage needs `model_runner: v1` in its deploy config"
            )
        if not getattr(model_config, "async_chunk", False):
            raise ValueError(
                "a TT vLLM-Omni generation stage needs `async_chunk: true` in its "
                "deploy config"
            )
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError(
                "a TT vLLM-Omni generation stage needs `async_scheduling: false` in "
                "its deploy config"
            )
        self.vllm_config = vllm_config
        self.model_config = model_config
        self.mesh_device = mesh_device
        self.model = None
        # Per live request: its current chunk, tokens of it already run, and the
        # metadata merged across its chunks.
        self.prompt_token_ids: dict[str, list[int]] = {}
        self.num_computed_tokens: dict[str, int] = {}
        self.infos: dict[str, dict] = {}
        self._pending: tuple[list[str], list[dict]] | None = None

    def load_model(self) -> None:
        model_class, _ = get_model_architecture(self.model_config)
        # The worker points device_config at the mesh; the model's tensors live on
        # the host.
        host_config = copy.copy(self.vllm_config)
        host_config.device_config = copy.copy(self.vllm_config.device_config)
        host_config.device_config.device = torch.device("cpu")
        with set_current_vllm_config(host_config):
            model = model_class(vllm_config=host_config, prefix="")
            get_model_loader(host_config.load_config).load_weights(
                model, self.model_config
            )
        self.model = model.eval()

    def get_supported_tasks(self):
        return ("generate",)

    def initialize_kv_cache(self, kv_cache_config) -> None:
        if kv_cache_config.kv_cache_groups:
            raise ValueError(
                "a TT vLLM-Omni generation stage has no token KV cache, got KV "
                f"cache groups {kv_cache_config.kv_cache_groups}"
            )

    def warmup_model(self) -> None:
        """Nothing to capture: the stage model runs on the host."""

    def _update_states(self, scheduler_output) -> None:
        for req_id in scheduler_output.finished_req_ids:
            self.prompt_token_ids.pop(req_id, None)
            self.num_computed_tokens.pop(req_id, None)
            self.infos.pop(req_id, None)

        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            self.prompt_token_ids[req_id] = list(new_req.prompt_token_ids or [])
            self.num_computed_tokens[req_id] = new_req.num_computed_tokens
            info: dict = {}
            for payload in (
                getattr(new_req, "model_intermediate_buffer", None),
                getattr(new_req, "additional_information", None),
            ):
                merge_info(info, deserialize_additional_information(payload))
            self.infos[req_id] = info

        cached = scheduler_output.scheduled_cached_reqs
        prompts = getattr(cached, "prompt_token_ids", {})
        cached_infos = getattr(cached, "additional_information", {})
        for req_id, num_computed in zip(
            cached.req_ids, cached.num_computed_tokens, strict=True
        ):
            if req_id in prompts:
                self.prompt_token_ids[req_id] = list(prompts[req_id] or [])
            self.num_computed_tokens[req_id] = num_computed
            merge_info(
                self.infos[req_id],
                deserialize_additional_information(cached_infos.get(req_id)),
            )

    def execute_model(self, scheduler_output):
        if self._pending is not None:
            raise RuntimeError(
                "execute_model called twice without sample_tokens in between"
            )
        self._update_states(scheduler_output)
        if scheduler_output.finished_req_ids:
            self.model.on_requests_finished(scheduler_output.finished_req_ids)
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return EMPTY_MODEL_RUNNER_OUTPUT

        req_ids = list(scheduler_output.num_scheduled_tokens)
        input_ids: list[int] = []
        seq_token_counts: list[int] = []
        for req_id in req_ids:
            start = self.num_computed_tokens[req_id]
            chunk = self.prompt_token_ids[req_id][
                start : start + scheduler_output.num_scheduled_tokens[req_id]
            ]
            input_ids.extend(chunk)
            seq_token_counts.append(len(chunk))
        infos = [self.infos[req_id] for req_id in req_ids]
        with torch.inference_mode():
            output = self.model(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                positions=None,
                runtime_additional_information=infos,
                model_intermediate_buffer=infos,
                seq_token_counts=seq_token_counts,
                request_ids=req_ids,
            )
        self._pending = (
            req_ids,
            per_request_payloads(output.multimodal_outputs, len(req_ids)),
        )
        return None

    def sample_tokens(self, grammar_output):
        if self._pending is None:
            return OmniModelRunnerOutput.with_kv_conn_output_only(None)
        req_ids, payloads = self._pending
        self._pending = None
        inter_stage_outputs, multimodal_outputs = partition_payload_list(payloads)
        return OmniModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=[],
            multimodal_outputs=multimodal_outputs,
            inter_stage_outputs=inter_stage_outputs,
        )

    def shutdown(self) -> None:
        self.model = None
        self.prompt_token_ids.clear()
        self.num_computed_tokens.clear()
        self.infos.clear()
        self._pending = None
