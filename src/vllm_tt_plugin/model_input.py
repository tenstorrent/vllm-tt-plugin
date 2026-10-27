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
class TTCompactedHostLogits:
    """Host logits in explicit scheduled-slot order, rather than full capacity."""

    logits: torch.Tensor
    rows: tuple[int, ...]


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
    # If not used, [None]. ``[rows, W]`` packed for a step that commits one
    # token per row; ``[rows, 1+K, W]``, one row per candidate column, for a
    # verify.
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

    # Decode-only, speculation only. ``draft_token_ids`` is ``[B, K]``, the
    # drafts the block's columns 1..K carry, kept alongside the block because
    # the accept walk needs them again after the verify returns. ``spec_mode``
    # names the accept mode the verify was asked for, and its presence is what
    # marks a step speculative. Both ``None`` otherwise.
    draft_token_ids: torch.Tensor | None = None
    spec_mode: str | None = None

    # Decode-only, speculation only. ``num_valid_drafts[i]`` is how many of row
    # i's ``input_tokens[i, 1:]`` columns are real drafts, in [0, K]; the
    # columns past it carry a padding marker and must not be verified.
    # ``accepted_counts[i]`` is how many tokens row i's previous step
    # committed, in [1, 1+K] and never 0, which is what a model reads to select
    # the candidate state slot it continues from. Both are ``[total_B]`` int32
    # in the same row order as ``input_tokens``, including its padding rows.
    # A verify sets both. An ordinary decode sets ``accepted_counts`` alone,
    # and only when a live row's previous step committed more than one token.
    # ``None`` on a prefill build and on a non-speculating launch.
    num_valid_drafts: torch.Tensor | None = None
    accepted_counts: torch.Tensor | None = None

    # Prefill only: rows whose forward writes KV state but must not emit a
    # sampled token, because more prompt tokens remain after this chunk.
    # ``None`` for decode.
    intermediate_prefill_mask: torch.Tensor | None = None

    # Request id per live forward row. A prefill build can drop rows, so
    # forward row order is not recoverable from the persistent batch;
    # sample-time consumers must resolve rows through this. A lane decode
    # build sets it to the scheduled requests in slot order, with ``live_rows``
    # naming their slots; ``None`` for a lane prefill build, whose rows are the
    # persistent slots.
    row_req_ids: list[str] | None = None

    # The block row each ``row_req_ids`` entry sits at. ``None`` means the live
    # rows are the first ``len(row_req_ids)`` rows of the block, which is the
    # front-packed layout. A lane decode holds every request at its own stable
    # slot, so its live rows are the scheduled slots and the padding rows sit
    # between them; the speculative consumers (the accept walk, the commit,
    # the proposal and the sampling capture) read rows through
    # ``live_row_index`` rather than assuming the front-packed layout.
    live_rows: tuple[int, ...] | None = None

    # Prefill-only: each row's stage inputs, passed to ``prefill_forward`` as
    # ``stage_inputs``. ``None`` unless the model declares
    # ``supports_stage_inputs``.
    stage_inputs: list[Any] | None = None

    # Decode-only: retain device logits until sample_tokens supplies the grammar,
    # then invoke the model's device sampler. Structured prefills stay on host.
    defer_device_sampling: bool = False

    # Requests in this forward that must receive a sample-time grammar row.
    # Captured with the forward so a partial GrammarOutput cannot silently leave
    # a structured row all-allowed after the persistent batch changes.
    structured_output_req_ids: frozenset[str] = frozenset()

    # Immutable row identity used for sample-time grammar remapping. Includes
    # padding/gap rows as None and is never reconstructed from a mutable batch.
    grammar_row_req_ids: tuple[str | None, ...] = ()

    # Speculating launch only, every build. How many grammar bitmask rows the
    # scheduler writes for each scheduled request: one per scheduled draft
    # position plus one for the bonus, so a structured request owns several
    # consecutive rows rather than one. ``None`` on a launch that does not
    # speculate, whose bitmask holds one row per structured request.
    grammar_rows_per_request: dict[str, int] | None = None

    # ``logits`` verify only. Live row position (the index into
    # ``row_req_ids``) to ``(remaining, stop_token_ids)`` for each live row
    # whose output is still shorter than its ``min_tokens``, captured at build
    # time, because the walk may run on the readback thread after the
    # request's output has grown.
    spec_min_tokens: dict[int, tuple[int, tuple[int, ...]]] | None = None


def live_row_index(
    model_input: TTModelInput, row_req_ids: list[str] | None = None
) -> torch.Tensor:
    """The block row of each live request, as an int64 index.

    ``tensor[live_row_index(model_input)]`` selects the live rows of any
    ``[B, ...]`` tensor of the step in ``row_req_ids`` order, whichever layout
    built the step. A front-packed build leaves ``live_rows`` unset and gets
    ``arange(len(row_req_ids))``. ``row_req_ids`` defaults to the step's own;
    a caller handed the live requests separately passes those, so the rows
    answer for the list it is iterating.
    """
    if row_req_ids is None:
        row_req_ids = model_input.row_req_ids
    if row_req_ids is None:
        raise RuntimeError(
            "live_row_index needs row_req_ids, which this build did not set"
        )
    live_rows = getattr(model_input, "live_rows", None)
    if live_rows is None:
        return torch.arange(len(row_req_ids), dtype=torch.int64)
    if len(live_rows) != len(row_req_ids):
        raise RuntimeError(
            f"live_rows names {len(live_rows)} rows for {len(row_req_ids)} "
            f"live requests {row_req_ids}: {live_rows}"
        )
    return torch.as_tensor(live_rows, dtype=torch.int64)


def penalty_history(output_tokens: torch.Tensor) -> list[list[int]]:
    """Each row's committed output, with TT's -1 padding removed."""
    return [[token for token in row.tolist() if token != -1] for row in output_tokens]


def penalty_prompt_ids(prompt_tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Prompt ids as vLLM's penalty op reads them.

    int64, with TT's -1 padding replaced by the vocabulary size, which is the
    one index the op's bin count ignores.
    """
    ids = prompt_tokens.to(torch.int64)
    return ids.masked_fill(ids == -1, vocab_size)
