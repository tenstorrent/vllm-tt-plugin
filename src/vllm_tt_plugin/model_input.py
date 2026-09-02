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
class TTDecodeReloadPlan:
    """Explicit host-to-device updates for one decode submission.

    The TT runner owns these decisions because it knows whether host
    token/position tensors are authoritative or intentionally one step behind
    an overlapped device-sampling decode. Contract-aware generators execute
    these flags as commands; they must not infer additional reloads from tensor
    equality, sampling mode, or their own previous-call state.
    """

    reload_inputs: bool
    reload_page_table: bool
    reload_sampling_params: bool
    reset_sampling_state: bool

    def __post_init__(self) -> None:
        # Sampling-state reset may align seeded counters from the host position,
        # which is authoritative only on a full forward-input reload.
        assert not (self.reset_sampling_state and not self.reload_inputs), (
            "reset_sampling_state requires reload_inputs"
        )
        # A full input reload already includes page tables. The two booleans
        # encode "everything", "page table only", or "nothing".
        assert not (self.reload_page_table and self.reload_inputs), (
            "reload_page_table is only valid without reload_inputs"
        )

    @property
    def overlap_safe(self) -> bool:
        """Whether a device-resident decode may be submitted host-stale."""
        return not (
            self.reload_inputs
            or self.reload_sampling_params
            or self.reset_sampling_state
        )


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


@dataclass(frozen=True)
class TTModelInput:
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    prompt_lens: list[int] | None
    # Group-0 block table, retained as a tensor for back-compat with the
    # padding paths that read it as ``block_tables``.
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

    # In lane mode this is true only if every lane can sample on device.
    perform_device_sampling: bool

    # Lists preserve the sampling-helper interface; each independent runner and
    # each merged lane step supplies one element.
    # If not used, [None]
    grammar_bitmask: list[torch.Tensor | None]

    # Host-only sampling params. Each independent rank supplies one list entry;
    # lane mode supplies one merged entry. Used when device sampling is absent.
    logitsprocs_list: list[LogitsProcessors | None]
    # bad_words_token_ids: list of dicts mapping req_index -> token_ids
    bad_words_token_ids_list: list[dict[int, list[list[int]]]]
    # allowed_token_ids_mask: list of (num_reqs, vocab_size) bool tensors
    allowed_token_ids_mask_list: list[torch.Tensor | None]
    # list of dicts mapping req_index -> generator for each DP rank
    # only populated when host sampling
    generators_list: list[dict[int, torch.Generator]]
    # max_num_logprobs: per-DP-rank list of max logprobs values
    # None means no logprobs, 0 means sampled token only
    max_num_logprobs: list[int | None]

    # Optional: tokens for sampling with penalties during decode
    prompt_tokens: torch.Tensor | None = None
    output_tokens: torch.Tensor | None = None

    # Decode-only runner lifecycle signal. Contract-v1 adapters receive the
    # explicit reload commands derived from this; legacy adapters receive it
    # translated to their historical ``reset_batch`` keyword.
    decode_layout_changed: bool = False

    # Decode-only: device state slot remap - row i reads slot remap[i]. From
    # ``_req_state_slot`` (lane mode: the condense-move remap). ``None`` means
    # identity. Shape: [total_B].
    slot_remap: torch.Tensor | None = None

    # Prefill-only: the device state slot each prefilling row writes to. Global for
    # single-process DP (supplied by the scheduler-owned step plan), local otherwise.
    prefill_empty_slots: list[int] | None = None

    # Prefill only: rows whose forward writes KV state but must not emit a
    # sampled token, because more prompt tokens remain after this chunk.
    # ``None`` for decode.
    intermediate_prefill_mask: torch.Tensor | None = None

    # Request id per forward row. A prefill build can drop rows, so forward row
    # order is not recoverable from the persistent batch; sample-time consumers
    # must resolve rows through this. ``None`` for lane builds, whose rows are
    # the persistent slots.
    row_req_ids: list[str] | None = None
