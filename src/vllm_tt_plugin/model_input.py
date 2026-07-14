# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Per-step input payloads handed from ``TTModelRunner`` to the TT model.

``TTModelInput`` is the prebuilt input for one execution step (prefill or
decode); ``TTSamplingParams`` carries the sampling tensors/lists that ride
along with it. Both are plain frozen dataclasses with no execution logic so
the runner, the lane executor, and the async decode controller can share them
without importing the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors


@dataclass(frozen=True)
class TTSamplingParams:
    """Sampling parameters for TT model execution.

    Host sampling uses tensors, while on-device sampling uses lists.
    """

    temperature: torch.Tensor | list[float]
    top_k: torch.Tensor | list[int]
    top_p: torch.Tensor | list[float]
    presence_penalty: torch.Tensor | list[float] | float = 0.0
    frequency_penalty: torch.Tensor | list[float] | float = 0.0
    repetition_penalty: torch.Tensor | list[float] | float = 1.0
    seed: torch.Tensor | list[int | None] | int = 0
    num_logprobs: torch.Tensor | list[int] | int | None = None
    enable_log_probs: torch.Tensor | list[bool] | None = None


def slice_tt_sampling_params(
    sampling: TTSamplingParams, rows: torch.Tensor | list[int]
) -> TTSamplingParams:
    """Select ``rows`` from the per-row sampling tensors of ``sampling``.

    ``num_logprobs >= 0`` encodes ``enable_log_probs`` (-2 means no logprobs, 0
    means the sampled token only).
    """
    num_logprobs = sampling.num_logprobs[rows]
    return TTSamplingParams(
        temperature=sampling.temperature[rows],
        top_k=sampling.top_k[rows],
        top_p=sampling.top_p[rows],
        presence_penalty=sampling.presence_penalty[rows],
        frequency_penalty=sampling.frequency_penalty[rows],
        repetition_penalty=sampling.repetition_penalty[rows],
        seed=sampling.seed[rows],
        num_logprobs=num_logprobs,
        enable_log_probs=num_logprobs >= 0,
    )


def normalize_greedy_device_sampling_params(
    sampling: TTSamplingParams,
) -> TTSamplingParams:
    """Convert greedy rows into the format TT device sampling expects.

    vLLM marks greedy requests with ``temperature == 0``. TT's device sampler
    does not use that convention, so for device sampling we rewrite greedy
    rows to ``top_k=1, top_p=0, temperature=1``. That keeps greedy rows as
    argmax, even when they share a batch with non-greedy rows.

    This is only for payloads that will be sampled on device. Host sampling
    should keep the original vLLM representation.

    Examples
    --------
    >>> inputs = TTSamplingParams(
    ...     temperature=[0.0, 0.8],
    ...     top_k=[32, 10],
    ...     top_p=[0.95, 0.9])
    >>> outputs = normalize_greedy_device_sampling_params(inputs)
    >>> outputs
    TTSamplingParams(
    ...     temperature=[1.0, 0.8],
    ...     top_k=[1, 10],
    ...     top_p=[0.0, 0.9], ...)
    """
    temperature = sampling.temperature
    top_k = sampling.top_k
    top_p = sampling.top_p

    if isinstance(temperature, torch.Tensor):
        greedy_mask = temperature == 0.0

        if not greedy_mask.any().item():
            return sampling

        top_k = torch.as_tensor(top_k)
        top_p = torch.as_tensor(top_p)

        return TTSamplingParams(
            temperature=torch.where(greedy_mask, torch.ones_like(temperature), temperature),
            top_k=torch.where(greedy_mask, torch.ones_like(top_k), top_k),
            top_p=torch.where(greedy_mask, torch.zeros_like(top_p), top_p),
            presence_penalty=sampling.presence_penalty,
            frequency_penalty=sampling.frequency_penalty,
            repetition_penalty=sampling.repetition_penalty,
            seed=sampling.seed,
            num_logprobs=sampling.num_logprobs,
            enable_log_probs=sampling.enable_log_probs,
        )

    if isinstance(temperature, list):
        changed = False
        new_temperature = list(temperature)
        new_top_k = list(top_k)
        new_top_p = list(top_p)

        for i, temp in enumerate(temperature):
            if temp == 0.0:
                new_temperature[i] = 1.0
                new_top_k[i] = 1
                new_top_p[i] = 0.0
                changed = True

        if not changed:
            return sampling

        return TTSamplingParams(
            temperature=new_temperature,
            top_k=new_top_k,
            top_p=new_top_p,
            presence_penalty=sampling.presence_penalty,
            frequency_penalty=sampling.frequency_penalty,
            repetition_penalty=sampling.repetition_penalty,
            seed=sampling.seed,
            num_logprobs=sampling.num_logprobs,
            enable_log_probs=sampling.enable_log_probs,
        )

    if temperature != 0.0:
        return sampling

    return TTSamplingParams(
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        presence_penalty=sampling.presence_penalty,
        frequency_penalty=sampling.frequency_penalty,
        repetition_penalty=sampling.repetition_penalty,
        seed=sampling.seed,
        num_logprobs=sampling.num_logprobs,
        enable_log_probs=sampling.enable_log_probs,
    )


@dataclass(frozen=True)
class TTModelInput:
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    prompt_lens: list[int] | None
    # Group-0 block table, retained as a tensor for back-compat with the
    # many DP padding/gather/pack paths that read it as ``block_tables``.
    # Hybrid models must additionally consult ``block_tables_per_group``
    # below; legacy single-group models can continue to use this field.
    block_tables: torch.Tensor
    # Per-group block tables in upstream's KVCacheConfig group order; one
    # entry for uniform models, ``len(kv_cache_groups)`` entries for
    # hybrid attention. Group g's tensor maps the model's layer-to-group
    # routing onto the right paged pool. We expand this into
    # ``block_tables_per_layer`` (one entry per decoder layer) before
    # handing it to hybrid models so they don't have to re-derive vLLM's
    # group construction order.
    block_tables_per_group: list[torch.Tensor]
    # Per-layer block tables, one entry per decoder layer in model
    # layer-index order. ``None`` for non-hybrid models (the runner only
    # populates this when ``self._layer_to_group_idx`` was set at
    # ``initialize_kv_cache`` time, which itself only fires when the
    # model class exposes ``get_kv_cache_spec``).
    block_tables_per_layer: list[torch.Tensor] | None
    unpadded_batch_size: int | list[int]  # List is used for DP
    tt_sampling_params: TTSamplingParams
    multi_modal_kwargs: dict[str, Any]

    # For DP gather, this is true only if all ranks can sample on device.
    perform_device_sampling: bool

    # always lists: single-element for non-DP, multi-element for DP
    # If not used, [None]
    grammar_bitmask: list[torch.Tensor | None]

    # Host-only sampling params - lists for DP (one per rank), single-element
    # for non-DP. These are used for host sampling when device sampling is not
    # supported.
    logitsprocs_list: list[LogitsProcessors | None]
    # bad_words_token_ids: list of dicts mapping req_index -> token_ids
    bad_words_token_ids_list: list[dict[int, list[list[int]]]]
    # allowed_token_ids_mask: list of (num_reqs, vocab_size) bool tensors
    allowed_token_ids_mask_list: list[torch.Tensor | None]
    # list of dicts mapping req_index -> generator for each DP rank
    # only gathered when host sampling
    generators_list: list[dict[int, torch.Generator]]
    # max_num_logprobs: per-DP-rank list of max logprobs values
    # None means no logprobs, 0 means sampled token only
    max_num_logprobs: list[int | None]

    # Optional: tokens for sampling with penalties during decode
    prompt_tokens: torch.Tensor | None = None
    output_tokens: torch.Tensor | None = None

    # Decode-only: indicates the padded decode-batch layout changed since the
    # previous step (used by on-device sampling).
    reset_batch: bool = False

    # Per-rank slot remap from condense - remap[i]=j means slot i's data came
    # from slot j. Identity when nothing moved. Shape: [total_B] (concat of
    # per-rank [B] tensors for DP).
    slot_remap: torch.Tensor | None = None

    # Single-process DP prefill only: global stable slots supplied by the
    # scheduler-owned step plan. ``None`` for non-DP, gathered-DP, and decode.
    prefill_empty_slots: list[int] | None = None
