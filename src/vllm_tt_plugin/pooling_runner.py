# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Model runner for TT pooling / embedding models.

Pooling models (text embedding and cross-encoder / reranker scoring) are far
simpler than generative models: they have no KV cache, no page tables, no
prefill/decode split and no sampling. Each request is a single forward pass
that turns tokenized input into a per-request vector (an embedding, or a
single relevance logit for a cross-encoder), returned to vLLM in the
``pooler_output`` field of :class:`ModelRunnerOutput`.

The generative :class:`~vllm_tt_plugin.model_runner.TTModelRunner` carries a
large amount of machinery (KV allocation, lane/DP orchestration, device
sampling, structured output) that pooling models neither need nor exercise.
Rather than thread ``runner_type == "pooling"`` branches through all of it,
the worker selects this dedicated runner for pooling models.

The TT model is a plain ``nn.Module`` whose ``forward(input_ids,
attention_mask)`` runs the encoder backbone on device and returns a host
tensor shaped ``[batch, hidden]`` (embedding) or ``[batch, 1]`` (reranker
logit); this runner only owns host-side batching and the vLLM output contract,
so it can be exercised without a device by substituting a fake model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import ttnn
from vllm.config import VllmConfig
from vllm.tasks import PoolingTask, SupportedTask
from vllm.v1.outputs import ModelRunnerOutput

from vllm_tt_plugin.loader import TTModelLoader
from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_tt_logger(__name__)


class TTPoolingModelRunner:
    """Simplified model runner for TT pooling / embedding models.

    Constructed with the same signature as
    :class:`~vllm_tt_plugin.model_runner.TTModelRunner` so the worker can pick
    either class without special-casing the call site. The generative-only
    arguments (``trace_mode``, ``enable_model_warmup``, ``num_devices``) are
    accepted and ignored: pooling runs a single un-traced forward per batch and
    does no device warmup.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        mesh_device: ttnn.MeshDevice,
        trace_mode: str,
        enable_model_warmup: bool,
        num_devices: int,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.mesh_device = mesh_device
        self.num_devices = num_devices
        # Accepted for signature parity with TTModelRunner; unused for pooling.
        self.trace_mode = trace_mode
        self.enable_model_warmup = enable_model_warmup
        logger.info("TTPoolingModelRunner initialized (trace/warmup ignored)")
        # req_id -> per-request bookkeeping (kept minimal for pooling).
        self.requests: dict[str, dict] = {}
        self.max_batch_size = self.scheduler_config.max_num_seqs
        # Model is set by load_model().
        self.model: nn.Module | None = None

    def load_model(self) -> None:
        """Load the pooling model via the shared TT model loader."""
        if self.model is not None:
            logger.info("Pooling model already loaded, skipping")
            return
        logger.info("Loading TT pooling model...")
        loader = TTModelLoader(self.load_config)
        self.model = loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        )

    def get_model(self) -> nn.Module:
        assert self.model is not None, "Model not loaded. Call load_model() first."
        return self.model

    def warmup_model(self) -> None:
        """No-op: pooling runs a single un-traced forward, nothing to warm up."""
        return

    def _prepare_model_inputs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list]:
        """Build padded ``(tokens, attention_mask, req_data_list)`` for one batch.

        Pooling requests are single-shot prefills: every scheduled request is a
        new request whose full prompt is embedded in one pass. Sequences are
        right-padded to the batch's longest prompt with a 0/1 attention mask so
        the model can ignore pad positions.
        """
        scheduled_reqs = scheduler_output.scheduled_new_reqs
        if not scheduled_reqs:
            return None, None, []

        token_ids_list = []
        max_seq_len = 0
        req_data_list = []
        for req_data in scheduled_reqs:
            prompt_token_ids = req_data.prompt_token_ids
            self.requests[req_data.req_id] = {
                "prompt_token_ids": prompt_token_ids,
                "pooling_params": req_data.pooling_params,
            }
            max_seq_len = max(max_seq_len, len(prompt_token_ids))
            token_ids_list.append(prompt_token_ids)
            req_data_list.append(req_data)

        batch_size = len(token_ids_list)
        tokens = torch.zeros((batch_size, max_seq_len), dtype=torch.int64)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
        for i, token_ids in enumerate(token_ids_list):
            seq_len = len(token_ids)
            tokens[i, :seq_len] = torch.tensor(token_ids, dtype=torch.int64)
            attention_mask[i, :seq_len] = 1.0
        return tokens, attention_mask, req_data_list

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """Run one pooling forward and return embeddings in ``pooler_output``.

        Unlike the generative runner, this returns a completed
        :class:`ModelRunnerOutput` directly (there is no deferred
        ``sample_tokens`` step for pooling models).
        """
        tokens, attention_mask, req_data_list = self._prepare_model_inputs(
            scheduler_output
        )
        # Evict finished requests regardless of whether anything is scheduled
        # this step (a step can finish requests while scheduling no new ones).
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        if tokens is None:
            return self._empty_output()

        assert self.model is not None, "Model not loaded. Call load_model() first."
        batch_size = tokens.shape[0]
        outputs = self.model.forward(
            input_ids=tokens,
            attention_mask=attention_mask,
        )
        # ``forward`` returns one vector per request: [batch, hidden] for
        # embeddings or [batch, 1] for a cross-encoder logit. vLLM expects
        # ``pooler_output`` as a list of one host tensor per request.
        pooler_output = [outputs[i].cpu() for i in range(batch_size)]

        req_ids = [req_data.req_id for req_data in req_data_list]
        req_id_to_index = {req_id: i for i, req_id in enumerate(req_ids)}
        sampled_token_ids: list[list[int]] = [[] for _ in range(batch_size)]

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )

    @staticmethod
    def _empty_output() -> ModelRunnerOutput:
        return ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        """Pooling models expose the ``embed`` task.

        Cross-encoder / reranker models are served through the same embed path
        (their per-request vector is a single relevance logit), so ``embed`` is
        the task advertised for every TT pooling model.
        """
        return ["embed"]

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return tuple(self.get_supported_pooling_tasks())
