# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Pooling models (``runner="pooling"``) on TT: one prefill per request, no sampling.

The model is a pooling model by vLLM's own interface (``is_pooling_model``, a
``pooler`` that names its tasks). Its ``prefill_forward`` returns, per user, the
per-token states its pooler takes, e.g. the per-token logits of a
``token_classify`` model whose classifier runs on device. The runner then applies
the model's pooler with each request's ``PoolingParams``, as vLLM's GPU runner does,
so task updates and ``use_activation`` act exactly as upstream. A request is
complete after that one step, so none of the generation runner's persistent batch
state is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput


def supported_pooling_tasks(model: Any) -> list:
    pooler = getattr(model, "pooler", None)
    return [] if pooler is None else sorted(pooler.get_supported_tasks())


def _multi_modal_kwargs(new_req: NewRequestData) -> dict[str, list]:
    """The request's multimodal fields as one user of a batch: field -> [per item]."""
    fields: dict[str, list] = {}
    items = [f.data for f in new_req.mm_features or () if f.data is not None]
    for item in items:
        for key in item:
            fields.setdefault(key, [])
    for key, values in fields.items():
        values.extend(item[key].data if key in item else None for item in items)
    return {key: [values] for key, values in fields.items()}


def execute_pooling_step(
    model: Any, scheduler_output: SchedulerOutput
) -> ModelRunnerOutput:
    if scheduler_output.scheduled_cached_reqs.req_ids:
        raise RuntimeError(
            "a TT pooling request completes in its first step; got continuing "
            f"requests {scheduler_output.scheduled_cached_reqs.req_ids}"
        )
    new_reqs = scheduler_output.scheduled_new_reqs
    if not new_reqs:
        return EMPTY_MODEL_RUNNER_OUTPUT
    pooler = model.pooler
    tasks = pooler.get_supported_tasks()
    req_ids, params, lens, states = [], [], [], []
    for new_req in new_reqs:
        pooling_params = new_req.pooling_params
        if pooling_params is None or pooling_params.task not in tasks:
            raise ValueError(
                f"request {new_req.req_id}: pooling task "
                f"{getattr(pooling_params, 'task', None)!r} is not one of the "
                f"model's tasks {sorted(tasks)}"
            )
        prompt = new_req.prompt_token_ids or []
        scheduled = scheduler_output.num_scheduled_tokens[new_req.req_id]
        if new_req.num_computed_tokens != 0 or scheduled != len(prompt):
            raise ValueError(
                f"request {new_req.req_id}: a TT pooling model takes its whole "
                f"prompt ({len(prompt)} tokens) in one step, got {scheduled} from "
                f"position {new_req.num_computed_tokens}"
            )
        pooler.get_pooling_updates(pooling_params.task).apply(pooling_params)
        tokens = torch.tensor([prompt], dtype=torch.long)
        (state,) = model.prefill_forward(
            tokens, prompt_lens=[len(prompt)], **_multi_modal_kwargs(new_req)
        )
        req_ids.append(new_req.req_id)
        params.append(pooling_params)
        lens.append(len(prompt))
        states.append(state)

    prompt_lens = torch.tensor(lens, dtype=torch.int32)
    metadata = PoolingMetadata(
        prompt_lens=prompt_lens,
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=params,
        pooling_states=[PoolingStates() for _ in params],
    )
    hidden_states = torch.cat(states)
    metadata.build_pooling_cursor(
        np.array(lens), prompt_lens, device=hidden_states.device
    )
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[] for _ in req_ids],
        pooler_output=pooler(hidden_states=hidden_states, pooling_metadata=metadata),
    )
