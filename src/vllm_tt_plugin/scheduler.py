# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import bisect
import inspect
import math
import sys
import weakref
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_tt_plugin.config import (
    get_tt_adaptive_block_max_prompt_tokens,
    get_tt_block_kv_extent_tokens,
    get_tt_decode_interleave_config,
    get_tt_max_concurrent_partial_prefills,
    get_tt_output_tokens_per_step,
    get_tt_prefill_chunk_alignment,
    get_tt_prefill_chunk_cap_table,
    get_tt_prefill_over_pending_spec,
    get_tt_spec_plan,
    get_tt_supports_multimodal_prefill_resume,
    get_tt_supports_stage_outputs,
    is_tt_adaptive_block_output_model,
    is_tt_block_output_model,
    tt_requests_cannot_resume,
)
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.mamba_state import mamba_attention_block_size
from vllm_tt_plugin.partial_prefill_limit import (
    SHORT_FIRST_MAX_STEPS,
    HeldPrefills,
    PartialPrefillLimit,
)
from vllm_tt_plugin.spec_admission import MODEL_OWNED_DRAFT_METHOD
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

logger = init_tt_logger(__name__)


# ``SchedulerOutput`` has no backend metadata field. Like lane step state, this
# TT-only signal is attached to the mutable output object and therefore follows
# it through the executor serialization boundary. Counts, rather than a set of
# request IDs, identify exactly how many newest in-flight frames a wholesale
# prefix-cache reset made stale.
_TT_FORCED_RESET_DISCARD_COUNTS_ATTR = "_tt_forced_reset_discard_counts"
_TT_OUTPUT_FRAME_REQ_IDS_ATTR = "_tt_output_frame_req_ids"


def set_tt_forced_reset_discard_counts(
    scheduler_output: SchedulerOutput, counts: dict[str, int]
) -> None:
    if counts:
        set_tt_scheduler_output_field(
            scheduler_output, _TT_FORCED_RESET_DISCARD_COUNTS_ATTR, dict(counts)
        )


def get_tt_forced_reset_discard_counts(
    scheduler_output: SchedulerOutput,
) -> dict[str, int]:
    return dict(getattr(scheduler_output, _TT_FORCED_RESET_DISCARD_COUNTS_ATTR, {}))


# Per-step block-output reconciliation decisions, keyed by request id. Attached
# to the ``SchedulerOutput`` -- the ONLY object that re-associates a step's
# output with the scheduling decision that produced it: under async scheduling
# ``schedule()`` for step K+1 (which mutates per-``Request`` state) runs BEFORE
# ``update_from_output`` commits step K, so a single overwritten ``Request``
# slot would hand step K's commit the K+1 decision. The engine core holds each
# in-flight step's ``SchedulerOutput`` in its batch queue and passes it back to
# ``update_from_output``, so a map carried here always matches the output being
# committed (in sync mode the pairing is trivially the same step). Mirrors the
# forced-reset-count pattern above.
_TT_BLOCK_STEP_DECISIONS_ATTR = "_tt_block_step_decisions"


def set_tt_block_step_decisions(
    scheduler_output: SchedulerOutput, decisions: dict[str, bool]
) -> None:
    set_tt_scheduler_output_field(
        scheduler_output, _TT_BLOCK_STEP_DECISIONS_ATTR, dict(decisions)
    )


def get_tt_block_step_decisions(
    scheduler_output: SchedulerOutput,
) -> dict[str, bool]:
    return dict(getattr(scheduler_output, _TT_BLOCK_STEP_DECISIONS_ATTR, {}))


# The lane coordinator's per-step state; see ``lane_scheduler.py``.
_TT_STEP_STATE_ATTR = "_tt_step_state"

# Every attribute TT attaches to a ``SchedulerOutput`` beside its dataclass fields.
# Code that rebuilds a ``SchedulerOutput`` from those fields alone drops these, and
# carries them over with ``carry_tt_scheduler_output_fields``.
TT_SCHEDULER_OUTPUT_FIELDS = (
    _TT_FORCED_RESET_DISCARD_COUNTS_ATTR,
    _TT_OUTPUT_FRAME_REQ_IDS_ATTR,
    _TT_BLOCK_STEP_DECISIONS_ATTR,
    _TT_STEP_STATE_ATTR,
)


def set_tt_scheduler_output_field(
    scheduler_output: SchedulerOutput, name: str, value
) -> None:
    """Attach a TT field; ``carry_tt_scheduler_output_fields`` copies only these."""
    if name not in TT_SCHEDULER_OUTPUT_FIELDS:
        raise ValueError(
            f"TT SchedulerOutput field {name!r} is not in TT_SCHEDULER_OUTPUT_FIELDS "
            f"{TT_SCHEDULER_OUTPUT_FIELDS}"
        )
    setattr(scheduler_output, name, value)


def carry_tt_scheduler_output_fields(
    src: SchedulerOutput, dst: SchedulerOutput
) -> None:
    """Copy the TT fields set on ``src`` onto ``dst``."""
    for name in TT_SCHEDULER_OUTPUT_FIELDS:
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


class TTSchedulingMode(Enum):
    DEFAULT = "default"
    DECODE_ONLY = "decode_only"
    PREFILL_ONLY = "prefill_only"

    @classmethod
    def from_prefill_intent(cls, prefill_intent: int) -> "TTSchedulingMode":
        if prefill_intent == 0:
            return cls.DECODE_ONLY
        if prefill_intent == 1:
            return cls.PREFILL_ONLY
        raise ValueError(f"Invalid TT scheduling intent: {prefill_intent}")


def spec_lookahead_tokens(plan, num_spec_tokens: int, method: str | None) -> int:
    """KV slots to reserve past a step's own tokens for a model-owned drafter.

    Such a drafter proposes for the next step inside the current one: after
    the accept walk commits, its fused body runs over the anchor plus every
    draft and writes K/V for those K+1 positions, none of which the current
    step's allocation covers. Upstream reserves lookahead only for the drafter
    methods it knows, and ``custom_class`` is not one of them, so without this
    reservation the rows that cross into the next block land in the null block
    and the first token that reads them diverges from plain decode.

    An ngram launch also carries an admitted plan, but its drafts come from the
    plugin and its target verifies them inside the step's own allocation, so it
    reserves nothing here.
    """
    if plan is None or num_spec_tokens <= 0 or method != MODEL_OWNED_DRAFT_METHOD:
        return 0
    return num_spec_tokens + 1


class TTDecodeInterleavePolicy:
    """Bounds how many consecutive prefill steps may stall running decodes.

    A TT step carries either prefill rows or decode rows, never both, so a
    prompt split into N chunks occupies N consecutive prefill steps and every
    running decode waits for all of them. This policy inserts decode-only steps
    into such a run. ``docs/SCHEDULING.md`` covers why, how the two counts
    interact, and what the defaults rest on.

    Three invariants the code depends on:

    - Reaching the decode allowance never forces prefill. It only stops the
      policy choosing decode; whether prefill then runs is up to the scheduler
      and to whether prefill work is pending.
    - ``has_running_decode`` must exclude partial-prefill continuations. A
      decode step cannot advance one, so interleaving on its account trades a
      productive prefill step for an empty one.
    - A decode step taken with no prefill pending is an ordinary decode, not
      part of an insertion, and clears both counters.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        (
            self._enabled,
            self._prefill_steps,
            self._decode_steps,
        ) = get_tt_decode_interleave_config(vllm_config)
        self._prefill_run = 0
        self._decode_run = 0

    def wants_decode_step(
        self, *, has_pending_prefill: bool, has_running_decode: bool
    ) -> bool:
        """Whether to spend this step on decode although prefill work is pending.

        ``has_running_decode`` must exclude partial-prefill continuations: a
        decode step cannot advance one, so interleaving on its account would
        trade a productive prefill step for an empty one.
        """
        if not self._enabled or not has_pending_prefill or not has_running_decode:
            return False
        return (
            self._prefill_run >= self._prefill_steps
            and self._decode_run < self._decode_steps
        )

    def record_step(self, *, is_decode: bool, prefill_pending: bool) -> None:
        """Advance the counters with the phase the step actually ran.

        Called for every step, including one that scheduled no tokens. A
        decode-only step that schedules nothing (upstream's running loop skips a
        request whose async placeholders have already reached ``max_tokens``)
        still consumes its decode allowance; leaving the counters untouched
        there would re-pick decode on every following step and livelock the
        engine on empty steps.

        ``prefill_pending`` says whether any prefill work existed when the step
        was chosen. A decode step taken with nothing pending is an ordinary
        decode, not part of an insertion, so it clears both counters: counting
        it would leave ``_prefill_run`` at its bound with part of the decode
        allowance already spent, and the next arriving prompt would then lose
        its first prefill step to an insertion it was never part of.
        """
        if not prefill_pending:
            self._prefill_run = 0
            self._decode_run = 0
            return
        if is_decode:
            self._decode_run += 1
            if self._decode_run >= self._decode_steps:
                # Allowance spent: require a fresh run of prefill steps before
                # the next insertion, which is what stops the policy from
                # holding the device in decode while a prompt waits.
                self._decode_run = 0
                self._prefill_run = 0
        else:
            self._prefill_run += 1
            self._decode_run = 0


def _effective_prefill_step_limit(scheduler_config) -> tuple[int, str]:
    """The most tokens one scheduling step can hand a single request, named.

    Resolved as vLLM's ``Scheduler.schedule`` caps one request: by its token
    budget, ``max_num_scheduled_tokens`` when it is set and otherwise
    ``max_num_batched_tokens``, and by its input budget,
    ``max_num_batched_tokens``, which a larger ``max_num_scheduled_tokens``
    does not lift. A positive ``long_prefill_token_threshold`` then caps that
    limit. Returns ``(limit, name_of_the_binding_cap)``; the limit is
    ``sys.maxsize`` when nothing is configured.
    """
    scheduled = getattr(scheduler_config, "max_num_scheduled_tokens", None)
    batched = int(getattr(scheduler_config, "max_num_batched_tokens", 0) or 0)
    if scheduled is not None:
        limit, name = int(scheduled), "max_num_scheduled_tokens"
        if 0 < batched < limit:
            limit, name = batched, "max_num_batched_tokens"
    else:
        limit, name = (
            (batched, "max_num_batched_tokens")
            if batched > 0
            else (
                sys.maxsize,
                "max_num_batched_tokens",
            )
        )
    threshold = int(getattr(scheduler_config, "long_prefill_token_threshold", 0) or 0)
    if 0 < threshold < limit:
        limit, name = threshold, "long_prefill_token_threshold"
    return limit, name


def multimodal_prefill_limit(vllm_config) -> tuple[int, str]:
    """The longest multimodal prompt a model that declares
    ``supports_multimodal_prefill_resume`` False can prefill, and the setting
    that bounds it.

    ``TTScheduler._multimodal_prefill_grant`` prefills such a prompt in one
    step, past ``long_prefill_token_threshold``. vLLM takes a scheduled
    request's tokens out of ``max_num_scheduled_tokens`` (when set) and its
    tokens plus draft slots out of ``max_num_batched_tokens``. With chunked
    prefill on, the limit is also a multiple of the KV block size: a request
    replayed after preemption carries its output tokens and can outgrow one
    step, and its first chunk then ends on a KV block boundary at or after the
    end of its last multimodal item, within one step.

    ``cache_config.block_size`` is the attention block size when a request
    arrives, also for a model with ``MambaSpec`` groups: ``EngineCore`` sets
    it to the smallest block size of the KV cache groups, and a ``MambaSpec``
    group's block size is the attention block size (mode ``align``) or at
    least ``max_model_len`` (mode ``none``, ``check_mamba_launch``). The
    frontend takes it from the engine's ready response before it serves a
    request.
    """
    scheduler_config = vllm_config.scheduler_config
    batched = int(scheduler_config.max_num_batched_tokens)
    spec = vllm_config.speculative_config
    drafts = (
        int(getattr(spec, "max_num_new_slots_for_drafting", 0) or 0)
        if spec is not None
        else 0
    )
    limit = batched - drafts
    bound = f"max_num_batched_tokens={batched}"
    if drafts:
        bound += f" less {drafts} draft slots"
    scheduled = scheduler_config.max_num_scheduled_tokens
    if scheduled is not None and int(scheduled) < limit:
        limit, bound = int(scheduled), f"max_num_scheduled_tokens={scheduled}"
    if scheduler_config.enable_chunked_prefill:
        block = int(vllm_config.cache_config.block_size)
        if limit % block:
            limit = limit // block * block
            bound += f", rounded down to a multiple of the {block}-token KV block"
    return limit, bound


def _validate_prefill_chunk_alignment(align: int, scheduler_config) -> None:
    """Reject caps that can never let a new request reach the alignment grid.

    ``_align_prefill_chunk_end`` defers a new request whose chunk falls short
    of the grid; that only converges if a later step can offer a larger chunk.
    The binding cap is the one vLLM schedules a request with
    (``_effective_prefill_step_limit``): the smaller of
    ``max_num_scheduled_tokens``, when set, and ``max_num_batched_tokens``,
    further capped by a positive ``long_prefill_token_threshold``.
    """
    limit, name = _effective_prefill_step_limit(scheduler_config)
    if limit < align:
        raise ValueError(
            f"additional_config['tt']['prefill_chunk_alignment']={align} cannot be "
            f"reached: scheduler {name}={limit} is a permanent per-step cap "
            "below the "
            "alignment. Raise the cap to at least the alignment or set "
            "prefill_chunk_alignment to 0."
        )


def _validate_prefill_chunk_cap_table(
    table: Sequence[tuple[int, int]], align: int, kv_block: int
) -> None:
    """Reject a ``prefill_chunk_cap_table`` cap that a capped chunk cannot end on.

    ``TTScheduler._cap_prefill_chunk`` cuts a chunk to ``start + cap``, and
    ``_align_prefill_chunk_end`` then rounds the end of a partial chunk down to
    the alignment grid, so a cap must be a multiple of ``align`` (when it is
    above 1) for a capped chunk to keep its whole cap. vLLM's
    ``_mamba_block_aligned_split``, which runs before the cap for
    ``MambaSpec`` groups in mode ``align``, ends every partial chunk on a KV
    block boundary, where the GDN state is written; ``kv_block`` is that block
    size then, and 1 otherwise. A cap must keep that boundary too.
    """
    unit = math.lcm(max(align, 1), kv_block)
    for start, cap in table:
        if cap % unit:
            raise ValueError(
                f"additional_config['tt']['prefill_chunk_cap_table'] entry "
                f"[{start}, {cap}]: max_chunk_tokens must be a multiple of "
                f"{unit} (prefill_chunk_alignment={align}, KV block size of the "
                f"Mamba block-aligned split={kv_block})"
            )


def _require_kv_for_every_sequence(vllm_config, kv_cache_config) -> None:
    """Refuse a KV pool that could make vLLM preempt a stage-output request.

    vLLM preempts a running request when a step finds no free block, and a request
    of a model with ``supports_stage_outputs`` cannot resume. While ``max_num_seqs``
    requests at their peak fit in the pool at once, no step runs short. One
    request's peak is upstream's admission bound: the full context for a
    full-attention group, the window plus in-flight tokens for a sliding one.
    Every request is charged the whole in-flight budget, an over-count when
    several share it.
    """
    from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
    from vllm.v1.core.kv_cache_planning import (
        _max_memory_usage_bytes_from_groups,
        _pool_bytes_per_block_from_builder,
    )

    groups = kv_cache_config.kv_cache_groups
    if not groups:
        # An attention-free model allocates no blocks, so nothing preempts.
        return
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    max_model_len = vllm_config.model_config.max_model_len
    builder = get_kv_cache_config_builder(vllm_config)
    per_request = -(
        -_max_memory_usage_bytes_from_groups(builder, vllm_config, groups)
        // _pool_bytes_per_block_from_builder(builder, vllm_config, groups)
    )
    needed = max_num_seqs * per_request
    # Block 0 is vLLM's null block, never handed to a request.
    usable = kv_cache_config.num_blocks - 1
    if usable < needed:
        raise ValueError(
            "a model with supports_stage_outputs cannot resume a preempted request, "
            f"so its KV pool must hold max_num_seqs={max_num_seqs} requests at "
            f"max_model_len={max_model_len}: {needed} blocks across "
            f"{len(groups)} KV cache group(s), but only "
            f"{usable} are usable. Lower max_num_seqs or max_model_len, or raise "
            "the model's get_max_tokens_all_users budget."
        )


def scheduler_output_has_prefill_work(scheduler_output: SchedulerOutput) -> bool:
    """Whether the scheduler output contains any context-phase work.

    That is a new or resumed request, or a cached request with no output token
    yet. ``is_context_phase`` is the exact signal for a cached chunked-prefill
    continuation; token counts are ambiguous for a one-token final chunk and
    speculative decode. ``TTModelRunner`` runs such a step as a prefill.
    """
    cached_reqs = scheduler_output.scheduled_cached_reqs
    if scheduler_output.scheduled_new_reqs or cached_reqs.resumed_req_ids:
        return True
    return any(cached_reqs.is_context_phase(req_id) for req_id in cached_reqs.req_ids)


def prefill_qualifies_behind_verify(scheduler_output: SchedulerOutput) -> bool:
    """Whether a step is prefill work that a model declaring
    ``supports_prefill_over_pending_spec`` may run behind one unread verify:
    every scheduled request is a new request or a chunked-prefill
    continuation, none is resumed or preempted, and none carries multimodal
    input. ``TTScheduler`` and ``TTModelRunner`` decide with this one rule."""
    cached = scheduler_output.scheduled_cached_reqs
    if (
        cached.resumed_req_ids
        or scheduler_output.preempted_req_ids
        or get_tt_forced_reset_discard_counts(scheduler_output)
    ):
        return False
    if any(
        getattr(r, "mm_features", None) for r in scheduler_output.scheduled_new_reqs
    ):
        return False
    prefill = {r.req_id for r in scheduler_output.scheduled_new_reqs} | {
        r for r in cached.req_ids if cached.is_context_phase(r)
    }
    scheduled = set(scheduler_output.num_scheduled_tokens)
    return bool(scheduled) and scheduled <= prefill


def _carry_cleanup(earlier: SchedulerOutput, later: SchedulerOutput) -> None:
    """Add to ``later`` the cleanup that ``earlier`` drained.

    Every base ``schedule()`` pass drains ``finished_req_ids``, the freed
    encoder cache entries, and the requests preempted since the previous pass,
    even a pass that schedules nothing. When one ``TTScheduler.schedule()`` call
    runs several passes and returns only ``later``, the runner must still
    receive what ``earlier`` drained.
    """
    later.finished_req_ids |= earlier.finished_req_ids
    later.free_encoder_mm_hashes = (
        earlier.free_encoder_mm_hashes + later.free_encoder_mm_hashes
    )
    if earlier.preempted_req_ids:
        later.preempted_req_ids = (
            later.preempted_req_ids or set()
        ) | earlier.preempted_req_ids


# A model's bound ``confirm_prefix_cache_hit(token_ids, mm_items,
# num_computed_tokens)``: ``mm_items`` holds ``(identifier, offset, length,
# modality)`` per multimodal item, and the answer is the number of cached
# tokens the model serves (docs/MODEL_CAPABILITIES.md).
PrefixHitConfirmer = Callable[
    [Sequence[int], list[tuple[str, int, int, str]], int], int
]

# The loaded model's prefix-hit confirmer (its bound ``confirm_prefix_cache_hit``),
# published by ``TTWorker.load_model`` for the scheduler of its process: the
# engine core constructs its scheduler after the executor loaded the model, in
# the same process (a model that confirms hits is admitted only with the
# uniproc executor). None for a model without ``confirms_prefix_cache_hits``,
# and None again from ``TTWorker.shutdown``, so this module does not keep the
# model alive past the close of its mesh.
_published_confirmer: PrefixHitConfirmer | None = None


def publish_prefix_hit_confirmer(confirmer: PrefixHitConfirmer | None) -> None:
    """Record the loaded model's confirmer for the schedulers of this process."""
    global _published_confirmer
    _published_confirmer = confirmer


class TTScheduler(AsyncScheduler):
    """Scheduler for the TT (Tenstorrent) platform.

    TT constraints:
    - No mixed prefill+decode batches: each batch is either all-prefill
      or all-decode.
    - Token-chunked prefill is allowed: a long prefill may be split across
      steps. After a partial chunk the base scheduler keeps the request in
      ``running`` with ``is_prefill_chunk=True``; later prefill steps schedule
      the next chunk until the prompt is fully computed. Those continuations
      count as prefill work here, never as decodes.

    The base scheduler admits prefills from two queues: ``kv_holding_waiting``
    (requests that still hold KV blocks, e.g. a streaming-input session waiting
    for its next chunk), which it drains first, and ``waiting``. A temporarily
    blocked request (e.g. while a structured-output grammar compiles) stays in
    its queue, and the prefill loop skips it until its dependency is ready.
    Decode-only scheduling must hide both queues so that the base scheduler
    cannot admit a prefill into a decode step.

    Inherits from AsyncScheduler to get num_output_placeholders support.
    TT uses this scheduler in both sync and async execution modes:
    - with async_scheduling=False, it behaves as the single TT scheduler
      without execution overlap
    - with async_scheduling=True, placeholders allow decode requests to be
      re-scheduled before update_from_output processes the previous step's
      results, enabling host/device overlap
    - under async scheduling a preempted request keeps the tokens it had
      already scheduled but not yet received, and needs no TT-side handling
      for them. Those tokens are valid: the forward that produced them ran to
      completion before the preempt freed any block, device submits form a
      strict queue, and every async op is forced to complete before the next
      prefill, so no later write can reach the KV they were computed against.
      The base class marks them stale, appends them on arrival without
      touching the placeholder count, and keeps the request waiting until
      they drain, so the resumed prefill replays them instead of sampling
      against the old token history. Stale output is dropped
      (``Request.drop_stale_output``) only by the wholesale
      ``reset_prefix_cache`` teardown and when a resumable streaming-input
      request stops; wiring ordinary preemption into it drops valid tokens and
      silently truncates the response.

    Supports ``set_forced_mode`` for lane coordination:
    - ``TTSchedulingMode.DECODE_ONLY`` forces decode-only (even if waiting
      queue is non-empty).
    - ``TTSchedulingMode.PREFILL_ONLY`` forces prefill-only (and may return an
      empty batch when there is no pending prefill work).
    - ``TTSchedulingMode.DEFAULT`` uses the default policy: prefer prefill
      when pending prefill work exists, but fall back to decode-only if
      prefill cannot make progress and running decode requests exist.
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.aux_output_connector is not None:
            # The base update_from_output asserts an auxiliary output for every
            # request that emits tokens, and the TT model runner produces none.
            raise ValueError(
                "TT backend does not return auxiliary model outputs; remove "
                "--enable-return-routed-experts."
            )
        self._take_published_prefix_hit_confirmer()
        self._forced_mode = TTSchedulingMode.DEFAULT
        # Running decodes _schedule_prefill_only removes from ``running`` for
        # its base pass, and 0 outside it; see get_request_counts.
        self._num_hidden_decodes = 0
        # Requests _schedule_prefill_only holds back for
        # max_concurrent_partial_prefills during its base pass, and none
        # outside it; see get_request_counts.
        self._held_prefills = HeldPrefills()
        # Read only in DEFAULT mode. A lane coordinator sets a forced mode
        # before every one of its lanes' schedule() calls and owns the one
        # policy instance for the whole step, so this one stays dormant there.
        self._decode_interleave = TTDecodeInterleavePolicy(self.vllm_config)
        # The step the default mode returned last, while it produces output
        # tokens that update_from_output has not committed; and a prefill step
        # kept for the next schedule() call. See _schedule_release_step.
        self._unreleased_step: SchedulerOutput | None = None
        self._deferred_prefill: SchedulerOutput | None = None
        # The model's supports_prefill_over_pending_spec: a prefill may run
        # behind a held verify, with no release step before it.
        self._prefill_over_pending_spec = get_tt_prefill_over_pending_spec(
            self.vllm_config
        )
        # Cleanup that waits for the kept prefill to run; see
        # _order_cleanup_around_kept_prefill.
        self._finished_after_kept_prefill: set[str] = set()
        self._freed_encoder_after_kept_prefill: list[str] = []
        self._pending_forced_reset_discard_counts: dict[str, int] = {}
        # Block-step decisions for the step currently being committed, read off
        # its SchedulerOutput in update_from_output so _update_request_with_output
        # sees the decision that produced THIS output (not a later schedule).
        self._committing_block_step_decisions: dict[str, bool] = {}
        # On a speculating launch, the drafts scheduled for the step whose
        # output is being committed (_count_commit_past_scheduled_drafts reads
        # them); None otherwise.
        self._committing_spec_drafts: dict[str, list[int]] | None = None
        # Set by TTLaneCoordinator on its lanes; see make_stats.
        self.stats_made_by_coordinator = False
        # Request id that owns the adaptive model's SINGLE speculative
        # session, mirrored from scheduling-side facts (see
        # _mirror_spec_session). Only its owner can emit a block.
        self._spec_session_owner: str | None = None
        self._pending_async_output_frames: dict[str, int] = {}
        self._widest_decode_batch_size = 0
        mamba_block_size = mamba_attention_block_size(
            self.kv_cache_config.kv_cache_groups
        )
        # The attention KV block size when the model declares
        # supports_multimodal_prefill_resume False, whose multimodal requests
        # add_request and _multimodal_prefill_grant handle; 0 when it declares
        # True. self.block_size, the least common multiple of every group's
        # block size, is the attention block size only without MambaSpec
        # groups, whose block size can be max_model_len.
        self._multimodal_prefill_block_size = (
            0
            if get_tt_supports_multimodal_prefill_resume(self.vllm_config)
            else mamba_block_size or self.block_size
        )
        # Tokens, and tokens plus draft slots, granted through
        # _mamba_block_aligned_split in the current step.
        self._step_granted_tokens = 0
        self._step_granted_rows = 0
        self._warned_mm_split = False
        self._prefill_chunk_alignment = (
            get_tt_prefill_chunk_alignment(self.vllm_config, mamba_block_size)
            if self.scheduler_config.enable_chunked_prefill
            else 0
        )
        if self._prefill_chunk_alignment > 1:
            _validate_prefill_chunk_alignment(
                self._prefill_chunk_alignment, self.scheduler_config
            )
            logger.info(
                "TT scheduler: partial prefill chunks end on %d-token boundaries",
                self._prefill_chunk_alignment,
            )
        # Read here, not by the platform: the cap check needs the alignment,
        # which for a model with MambaSpec groups depends on the KV cache groups.
        self._prefill_chunk_cap_table = get_tt_prefill_chunk_cap_table(self.vllm_config)
        if self._prefill_chunk_cap_table:
            _validate_prefill_chunk_cap_table(
                self._prefill_chunk_cap_table,
                self._prefill_chunk_alignment,
                int(self.cache_config.block_size)
                if getattr(self, "_base_need_mamba_split", False)
                else 1,
            )
            logger.info(
                "TT scheduler: a text prefill chunk starting at depth "
                "[start_tokens] holds at most [max_chunk_tokens] tokens: %s",
                [list(pair) for pair in self._prefill_chunk_cap_table],
            )
        limit = get_tt_max_concurrent_partial_prefills(self.vllm_config)
        self._partial_prefill_limit = PartialPrefillLimit(limit) if limit else None
        if limit:
            logger.info(
                "TT scheduler: at most %d prefill(s) that need more than one "
                "chunk advance per prefill step, oldest first; waiting "
                "single-chunk prompts go ahead of them for at most %d steps",
                limit,
                SHORT_FIRST_MAX_STEPS,
            )
        # vLLM sets mamba_partial_cache_hit for MambaSpec groups in mode align
        # with a prefix_match_unit below the block size. It then makes
        # prefix-cache hits that end inside a block and queues a copy of the
        # block for each, and TT copies no blocks
        # (MambaCacheGroups.refuse_block_copies).
        if getattr(self, "mamba_partial_cache_hit", False):
            raise ValueError(
                f"cache_config.prefix_match_unit={self.hash_block_size} is below "
                f"the KV block size {self.block_size} with mamba_cache_mode "
                "'align', so vLLM would copy a KV cache block for every "
                "prefix-cache hit that ends inside a block. TT copies no KV cache "
                "blocks for a model with MambaSpec KV cache groups; leave "
                f"prefix_match_unit unset or set it to {self.block_size}."
            )
        self._output_tokens_per_step = get_tt_output_tokens_per_step(self.vllm_config)
        self._is_block_output_model = is_tt_block_output_model(self.vllm_config)
        self._requests_cannot_resume = tt_requests_cannot_resume(self.vllm_config)
        if get_tt_supports_stage_outputs(self.vllm_config):
            _require_kv_for_every_sequence(self.vllm_config, self.kv_cache_config)
        # Adaptive: emit the block only on a solo decode step; batch >1 decodes
        # as plain baseline. Lets max_num_seqs>1 coexist with block-output.
        self._is_adaptive_block = is_tt_adaptive_block_output_model(self.vllm_config)
        # Prompt-length frontier for the block path (0 = none): a longer prompt
        # is served as plain baseline by the model for its whole lifetime, so
        # its steps reserve width-1 even when solo.
        self._adaptive_block_max_prompt = get_tt_adaptive_block_max_prompt_tokens(
            self.vllm_config
        )
        speculative_config = self.vllm_config.speculative_config
        self.num_lookahead_tokens = max(
            self.num_lookahead_tokens,
            spec_lookahead_tokens(
                get_tt_spec_plan(self.vllm_config),
                self.num_spec_tokens,
                speculative_config.method if speculative_config is not None else None,
            ),
        )
        if self._is_block_output_model:
            # KV pages for the WHOLE step, not just the one token upstream
            # counts. A block-output decode runs several internal verify
            # iterations against vLLM-owned KV before it returns, and those
            # iterations write past the position upstream allocated for:
            # schedule() calls allocate_slots with num_lookahead_tokens, which
            # is 0 here because there is no vLLM speculative_config, and
            # raising Request.num_output_placeholders afterwards accounts for
            # pending OUTPUT tokens without allocating anything. The model's
            # refresh_page_tables then pads the missing columns with zero and
            # the verify reads and writes the null block.
            #
            # Reserved as twice the emitted width: the committed block, plus
            # headroom for the final iteration's verify positions and for
            # accepted tokens carried past the emitted width. Both are bounded
            # by the packed-verify width, which is far below the block width
            # (6 against 64 at the shipped default), so this is generous and
            # costs a page or two per request.
            #
            # Twice the width is NOT a bound in general, though: it only covers
            # the physical extent while the block is at least as wide as the
            # verification. A model configured the other way round -- dFlash at
            # GEMMA4_DFLASH_SERVE_BLOCK=2 with GEMMA4_DFLASH_VERIFY=7 emits 2
            # tokens and writes 8 verification rows -- needs 2 + 8 while this
            # reserves 4, and the step then writes positions that have no
            # request block (vllm-tt-plugin#118 review, finding 2). So a model
            # may declare the extent it actually touches and we honour whichever
            # is larger.
            declared_extent = get_tt_block_kv_extent_tokens(self.vllm_config)
            self.num_lookahead_tokens = max(
                int(getattr(self, "num_lookahead_tokens", 0) or 0),
                2 * int(self._output_tokens_per_step),
                int(declared_extent),
            )
            assert self.num_sampled_tokens_per_step == 1, (
                "Block-output accounting requires upstream to reserve exactly "
                "one sampled-token placeholder"
            )
            assert not self.kv_cache_manager.enable_caching, (
                "Block-output models must run with prefix caching disabled; "
                "TTPlatform validation disables it"
            )

    def _take_published_prefix_hit_confirmer(self) -> None:
        """Route the KV cache manager's hit lookup through the model's confirmer.

        A model that owns its prefix cache (device state per slot) confirms
        vLLM's longest hit before the scheduler commits it, so
        ``num_computed_tokens`` never exceeds what the device can serve and
        ``NewRequestData`` strips only the mm payload the model has resident.
        """
        confirmer = _published_confirmer
        if confirmer is None:
            self._prefix_hit_confirmer_ref = None
            return
        # Held weakly. EngineCore shuts the executor down before the scheduler,
        # so a strong reference here would keep the model, and every device
        # tensor it owns, alive past TTWorker.shutdown's close of the mesh. A
        # confirmer that is not a bound method holds no model instance.
        self._prefix_hit_confirmer_ref = (
            weakref.WeakMethod(confirmer)
            if inspect.ismethod(confirmer)
            else lambda: confirmer
        )
        manager = self.kv_cache_manager
        self._orig_get_computed_blocks = manager.get_computed_blocks
        manager.get_computed_blocks = self._confirmed_computed_blocks
        logger.info(
            "Prefix-cache hits are confirmed by the model before they are scheduled"
        )

    def _confirmed_computed_blocks(
        self, request: Request, max_cache_hit_length: int | None = None
    ) -> "tuple[KVCacheBlocks, int, int]":
        """``get_computed_blocks`` trimmed to the hit the model confirms.

        vLLM's longest block-hash hit is proposed to the model as
        ``confirm(token_ids, mm_items, num_new_computed_tokens)``; the model
        answers the number of cached tokens it will serve (0 = none), taken at
        most as the proposal and floored to the scheduler's block size (the KV
        cache manager carries none), and the computed blocks are cut to it. The
        shared-prefix boundary pin is dropped with a trimmed hit.

        ``token_ids`` is ``request.all_token_ids``, vLLM's read-only view of
        the sequence the hit was hashed over, handed over without a copy.
        ``max_cache_hit_length``, vLLM's bound on the hit (``num_tokens - 1``
        when None), goes to the lookup unchanged. The hit reaches up to that
        bound, so for a request resumed from preemption it runs past the prompt
        into the request's output.
        """
        blocks, num_new, boundary = self._orig_get_computed_blocks(
            request, max_cache_hit_length
        )
        if num_new <= 0:
            return blocks, num_new, boundary
        confirm = self._prefix_hit_confirmer_ref()
        if confirm is None:
            raise RuntimeError(
                "the model that confirms prefix-cache hits was released "
                "(TTWorker.shutdown) while its scheduler still schedules: "
                f"request {request.request_id} proposed a {num_new}-token hit"
            )
        mm_items = [
            (
                f.identifier,
                int(f.mm_position.offset),
                int(f.mm_position.length),
                f.modality,
            )
            for f in request.mm_features
        ]
        honoured = int(confirm(request.all_token_ids, mm_items, num_new))
        if honoured < 0:
            raise ValueError(
                f"confirm_prefix_cache_hit answered {honoured} cached tokens for the "
                f"{num_new}-token hit of request {request.request_id}"
            )
        honoured = min(honoured, num_new) // self.block_size * self.block_size
        if honoured == num_new:
            return blocks, num_new, boundary
        trimmed = self.kv_cache_manager.truncate_computed_blocks(blocks, honoured)
        return trimmed, honoured, 0

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

    # Host-only sampling controls and their neutral values: any of these
    # forces the step onto host sampling (check_perform_device_sampling),
    # which cannot construct a multi-token canvas and kills the engine.
    # Penalties force host sampling only for a model that declares
    # supports_device_penalties=False. Otherwise a non-neutral value flips
    # InputBatch.no_penalties and makes every block decode step build (and
    # discard) penalty token tensors that grow with the committed session
    # length; the frontend neutralizes them, so mirror that here for
    # prebuilt requests that bypassed it.
    # The predicate's other request-driven triggers cannot fire for a
    # block-output model: _strip_unsupported_block_controls strips structured
    # outputs, the block-output InputBatch drops logprobs (disable_logprobs),
    # and the platform refuses seeded_sampling_policy="host" and
    # max_device_top_k for block output at config time.
    _BLOCK_HOST_ONLY_SAMPLING_NEUTRAL = (
        ("min_p", 0.0),
        ("min_tokens", 0),
        ("logit_bias", None),
        ("allowed_token_ids", None),
        ("bad_words", None),
        ("_bad_words_token_ids", None),
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
        ("repetition_penalty", 1.0),
    )

    def add_request(self, request: Request) -> None:
        """Admit a request, first repairing a block-output request that skipped
        TT request validation.

        Block-output requests normally pass ``TTPlatform.validate_request``
        through the patched ``InputProcessor``, which rejects what a
        block-output model cannot serve. A prebuilt ``EngineCoreRequest``
        passed to ``AsyncLLM.add_request`` or ``LLMEngine.add_request`` (an
        API upstream deprecates), or sent through ``EngineCoreClient``, skips
        ``InputProcessor.process_inputs`` and so skips that validation. vLLM's
        own entrypoints never build such a request, and the platform refuses
        the Rust frontend for block-output models at startup, so this is a
        safety net for programmatic engine callers, not a path the served API
        reaches.

        An exception from ``add_request`` tears down EngineCore for every
        client. So instead of rejecting, the request repair layer
        (``_repair_unvalidated_block_request``) changes the request until it
        is servable, logs one warning per change, and lets the request finish
        normally; the caller is not told. Because the change is silent, the
        Rust frontend stays refused: it would send all traffic down this path
        instead of rejecting unsupported requests.

        A multimodal request of a model that declares
        ``supports_multimodal_prefill_resume`` False never reads the prefix
        cache: such a model computes a multimodal prompt only from position 0,
        and vLLM's ``strip_covered_mm_data`` drops the pixels of an item a
        prefix-cache hit covers. vLLM still caches the request's blocks: their
        hashes carry the item identifiers from the first item on, so only the
        text before the first item can match another request. Its first
        prefill chunk keeps the state at one block boundary only, the prefill
        checkpoint at the prompt's last cacheable position
        (``_multimodal_prefill_grant`` does not end the chunk on earlier
        boundaries), so another request reuses that text only when the first
        item starts after the checkpoint.
        """
        if getattr(self, "_multimodal_prefill_block_size", 0) and request.mm_features:
            # Request.__init__ derives the flag from the sampling parameters;
            # the KV cache manager reads it at the prefix-cache lookup.
            request.skip_reading_prefix_cache = True
        if self._is_block_output_model:
            existing = self.requests.get(request.request_id)
            if existing is not None and existing.streaming_queue is None:
                # A continuation (next input chunk or the closing sentinel) of
                # a session whose resumable flag _strip_unsupported_block_controls
                # scrubbed. The base scheduler asserts on the missing
                # streaming_queue, which would tear down EngineCore. The
                # session cannot accept more input, so drop the message; the
                # live request finishes and notifies its client on its own.
                logger.warning(
                    "Dropping streaming-input continuation for request %s: "
                    "block-output models do not support resumable sessions",
                    request.request_id,
                )
                return
            self._repair_unvalidated_block_request(request)
        super().add_request(request)

    def _repair_unvalidated_block_request(self, request: Request) -> None:
        """Make a block-output request that skipped validation servable.

        The prompt repairs run first, in this order, because each reads the
        prompt the previous one left. The max_tokens clamp needs the final
        prompt length. The control strip is independent of the prompt.
        """
        self._drop_unencodable_mm_features(request)
        self._replace_prompt_embeds_with_placeholders(request)
        self._pad_empty_prompt(request)
        self._truncate_overlong_prompt(request)
        self._clamp_max_tokens_to_whole_canvases(request)
        self._strip_unsupported_block_controls(request)

    def _drop_unencodable_mm_features(self, request: Request) -> None:
        if request.mm_features:
            # A text-only block model has a zero encoder budget: a feature at
            # offset 0 forces zero-token schedules forever (head-of-line
            # stall), and an interior offset carves a partial prefill chunk
            # that flips the step onto host sampling and kills the engine.
            # Dropped, the placeholder positions decode as ordinary tokens.
            logger.warning(
                "Request %s bypassed frontend validation with multimodal "
                "features a block-output model cannot encode; dropping them",
                request.request_id,
            )
            request.mm_features = []

    def _replace_prompt_embeds_with_placeholders(self, request: Request) -> None:
        if request.prompt_token_ids is None and request.num_prompt_tokens > 0:
            # The frontend rejects prompt_embeds for every TT model; admitted
            # bare, the worker's request-state builder raises
            # NotImplementedError out of execute_model. Replace with
            # placeholder tokens; an embeds-only Request already carries
            # [0] * num_prompt_tokens in _all_token_ids, so the replacement
            # keeps every derived view consistent.
            logger.warning(
                "Request %s bypassed frontend validation with a "
                "prompt_embeds-only prompt the TT backend does not support; "
                "replacing with %d placeholder tokens",
                request.request_id,
                request.num_prompt_tokens,
            )
            request.prompt_token_ids = [0] * request.num_prompt_tokens
            request.prompt_embeds = None
        if request.prompt_embeds is not None:
            # Mixed-mode prompts (token ids + embeds + prompt_is_token_ids
            # mask) carry placeholder ids at the embed positions, so the TT
            # model decodes them as ordinary tokens either way; scrub the
            # embeds so the prompt_len x hidden_size tensor is not pinned for
            # the request lifetime, and warn like the embeds-only branch.
            logger.warning(
                "Request %s bypassed frontend validation with mixed "
                "token/embeds prompt content the TT backend does not "
                "support; dropping the embeds (placeholder ids decode as "
                "ordinary tokens)",
                request.request_id,
            )
            request.prompt_embeds = None
            request.prompt_is_token_ids = None

    def _pad_empty_prompt(self, request: Request) -> None:
        if request.num_prompt_tokens == 0:
            # The frontend rejects empty prompts; admitted bare, the waiting
            # loop schedules zero new tokens and upstream's num_new_tokens
            # assert tears down the engine. Pad to one placeholder token so
            # the request schedules and finishes through the normal path.
            logger.warning(
                "Request %s bypassed frontend validation with an empty "
                "prompt; padding to one placeholder token",
                request.request_id,
            )
            request.prompt_token_ids = [0]
            request.prompt_embeds = None
            request._all_token_ids.append(0)
            request.num_prompt_tokens = 1

    def _truncate_overlong_prompt(self, request: Request) -> None:
        """Contain a bypassed prompt that leaves no room for a whole canvas.

        Frontend validation rejects such prompts; a prebuilt EngineCoreRequest
        skips it, and neither upstream EngineCore.add_request nor the base
        scheduler re-checks prompt length. Admitted untouched, the prompt
        either can never be scheduled (chunked prefill is disabled and the
        prompt exceeds the token budget: parked in WAITING forever,
        head-of-line blocking every later request), overflows the worker's
        max_model_len-wide token buffer, or, even when it fits
        max_model_len, trips the adapter's own canvas-capacity validation,
        which raises out of execute_model in eager (no-upfront-capture) mode
        where there is no graceful stop-canvas rejection. Truncate to the
        largest tile-aligned prompt that still fits one whole canvas below
        the tile-floored max_model_len, so every admitted request is
        genuinely servable and finishes through the normal notifying path.
        Prefix caching is disabled for block models (__init__ asserts it), so
        the request's stale block hashes stay inert.
        """
        from vllm_tt_plugin.platform import _TT_TOKEN_TILE_SIZE

        tile = _TT_TOKEN_TILE_SIZE
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        aligned_max_model_len = max_model_len // tile * tile
        keep = (aligned_max_model_len - self._output_tokens_per_step) // tile * tile
        if request.num_prompt_tokens <= keep:
            return
        logger.warning(
            "Request %s bypassed frontend validation with a %d-token prompt "
            "that leaves no room for a whole %d-token canvas within "
            "max_model_len; truncating to %d tokens so the request can "
            "finish length-capped",
            request.request_id,
            request.num_prompt_tokens,
            self._output_tokens_per_step,
            keep,
        )
        if request.prompt_token_ids is not None:
            request.prompt_token_ids = request.prompt_token_ids[:keep]
        if getattr(request, "prompt_embeds", None) is not None:
            request.prompt_embeds = request.prompt_embeds[:keep]
        del request._all_token_ids[keep:]
        request.num_prompt_tokens = keep

    def _clamp_max_tokens_to_whole_canvases(self, request: Request) -> None:
        """Clamp max_tokens so a bypassed EngineCoreRequest cannot overshoot.

        Frontend validation rejects an oversized limit. Prebuilt requests skip
        that path; the last canvas would then be applied past max_model_len and
        kill the engine. Shrink the logical cap to the largest whole-canvas
        budget that still fits. Lane mode reaches this through each lane.
        """
        from vllm_tt_plugin.platform import _fit_block_output_max_tokens

        prompt_ids = request.prompt_token_ids
        if prompt_ids is None or request.sampling_params is None:
            return
        fitted = _fit_block_output_max_tokens(
            len(prompt_ids),
            request.max_tokens,
            self._output_tokens_per_step,
            int(self.vllm_config.model_config.max_model_len),
        )
        if fitted == request.max_tokens:
            return
        # _truncate_overlong_prompt leaves room for one whole canvas, and the
        # platform's startup floor (_min_block_output_max_model_len) makes
        # that room exist. Deliberate fail-fast for developers: it fires only
        # if the repair steps are reordered, never for a client request.
        if fitted <= 0:
            raise RuntimeError(
                f"Block-output max_tokens clamp for request {request.request_id} "
                f"left no budget: prompt_len={len(prompt_ids)}, "
                f"max_tokens={request.max_tokens}, fitted={fitted}, "
                f"output_tokens_per_step={self._output_tokens_per_step}, "
                f"max_model_len={self.vllm_config.model_config.max_model_len}"
            )
        logger.debug(
            "Clamping block-output max_tokens from %s to %s for request %s "
            "so physical canvases fit max_model_len",
            request.max_tokens,
            fitted,
            request.request_id,
        )
        request.max_tokens = fitted
        request.sampling_params.max_tokens = fitted

    def _strip_unsupported_block_controls(self, request: Request) -> None:
        """Strip controls a block-output model cannot honor from a bypassed
        request: host-sampling forcers, structured outputs, and resumable
        streaming-input sessions.

        Frontend validation rejects these; a prebuilt EngineCoreRequest skips
        it, and any of them flips the step to host sampling mid-flight (which
        cannot construct a multi-token canvas) or parks the request forever.
        Raising here is no safer: an add_request exception also tears down
        EngineCore, so neutralize instead, the way the sampling controls are
        neutralized at the frontend.
        """
        params = request.sampling_params
        if params is None:
            return
        stripped = []
        if request.structured_output_request is not None:
            request.structured_output_request = None
            stripped.append("structured_outputs")
            # Request.__init__ parked the request on grammar compilation; with
            # the structured-output request gone no grammar ever arrives, and
            # the base prefill loop would skip the request forever, so restore
            # schedulability.
            if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
                request.status = RequestStatus.WAITING
        if params.structured_outputs is not None:
            params.structured_outputs = None
        if request.resumable:
            # A resumable session parks the stopped request to wait for more
            # input instead of finishing it, permanently leaking the
            # model-owned state slot. The frontend rejects it for block models.
            request.resumable = False
            stripped.append("resumable")
        for field, neutral in self._BLOCK_HOST_ONLY_SAMPLING_NEUTRAL:
            if getattr(params, field, neutral) not in (neutral, [], {}):
                setattr(params, field, neutral)
                stripped.append(field.lstrip("_"))
        if stripped:
            logger.warning(
                "Request %s bypassed frontend validation; stripped "
                "controls unsupported by block-output models: %s",
                request.request_id,
                ", ".join(stripped),
            )

    def _has_pending_prefill(self) -> bool:
        """Whether any request still needs prefill work.

        A blocked request in ``waiting`` or ``kv_holding_waiting`` still needs
        a future prefill pass: only the base scheduler's prefill loop checks
        whether its dependency became ready. In decode-only mode this check also
        ensures both waiting queues are hidden from the base scheduler.

        A running ``is_prefill_chunk`` request is a partial prefill whose next
        chunk can only be scheduled by a prefill step.
        """
        return (
            bool(self.waiting)
            or bool(self.kv_holding_waiting)
            or any(request.is_prefill_chunk for request in self.running)
        )

    # vLLM runs ``_mamba_block_aligned_split`` on every prefill chunk when
    # ``need_mamba_block_aligned_split`` is set; the TT scheduler reuses that
    # hook so chunks can be capped by depth and their ends aligned, and the
    # first chunk of a multimodal request sized (_multimodal_prefill_grant),
    # before any block is allocated.
    @property
    def need_mamba_block_aligned_split(self) -> bool:
        return (
            bool(getattr(self, "_base_need_mamba_split", False))
            or getattr(self, "_prefill_chunk_alignment", 0) > 1
            or getattr(self, "_multimodal_prefill_block_size", 0) > 0
            or bool(getattr(self, "_prefill_chunk_cap_table", ()))
        )

    @need_mamba_block_aligned_split.setter
    def need_mamba_block_aligned_split(self, value: bool) -> None:
        self._base_need_mamba_split = bool(value)

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        grant = self._multimodal_prefill_grant(request, start, num_new_tokens)
        if grant is None:
            if getattr(self, "_base_need_mamba_split", False):
                num_new_tokens = super()._mamba_block_aligned_split(
                    request,
                    num_new_tokens,
                    num_new_local_computed_tokens,
                    num_external_computed_tokens,
                )
            num_new_tokens = self._cap_prefill_chunk(request, start, num_new_tokens)
            grant = self._align_prefill_chunk_end(request, start, num_new_tokens)
        if grant > 0:
            self._step_granted_tokens = getattr(self, "_step_granted_tokens", 0) + grant
            self._step_granted_rows = (
                getattr(self, "_step_granted_rows", 0) + grant + self._draft_slots()
            )
        return grant

    def _draft_slots(self) -> int:
        """Input rows vLLM takes out of ``max_num_batched_tokens`` for each
        scheduled request beyond its tokens."""
        spec = getattr(getattr(self, "vllm_config", None), "speculative_config", None)
        if spec is None:
            return 0
        return int(getattr(spec, "max_num_new_slots_for_drafting", 0) or 0)

    def _multimodal_prefill_grant(
        self, request: Request, start: int, num_new_tokens: int
    ) -> int | None:
        """Tokens to grant the multimodal part of a prefill of a model that
        declares ``supports_multimodal_prefill_resume`` False, or ``None`` for
        any other chunk.

        Such a model computes a multimodal prompt from position 0
        (``add_request`` keeps the request off the prefix cache) with every
        multimodal item in the first prefill step, so the first chunk covers
        every item:

        - the whole rest of the request when it fits what is left of this step,
          past ``long_prefill_token_threshold``, which vLLM applied before this
          hook. What is left is counted from this step's grants through this
          hook (tokens, and tokens plus draft slots). In a prefill step every
          scheduled request passes here, and a grant vLLM later shrinks or
          drops still counts, so a grant never overdraws vLLM's budgets;
        - when it fits an empty step but not this one, 0: vLLM stops admitting
          for this step or skips the running request;
        - when it is longer than an empty step, which only a request replayed
          after preemption reaches (it carries its output tokens, and
          ``TTPlatform.validate_request`` rejects a longer prompt), the longest
          chunk that fits this step and ends on a KV block boundary at or after
          the end of the last item, or 0 to wait for a step with room.

        The chunks after the last item are text and take the ordinary rules.
        """
        block = getattr(self, "_multimodal_prefill_block_size", 0)
        if not block or num_new_tokens <= 0 or not request.mm_features:
            return None
        items_end = max(
            f.mm_position.offset + f.mm_position.length for f in request.mm_features
        )
        if start >= items_end:
            return None
        tokens_left = self.max_num_scheduled_tokens - self._step_granted_tokens
        rows_left = (
            self.scheduler_config.max_num_batched_tokens - self._step_granted_rows
        )
        drafts = self._draft_slots()
        room = min(tokens_left, rows_left - drafts)
        remaining = request.num_tokens - start
        if remaining <= room:
            return remaining
        step = min(
            self.max_num_scheduled_tokens,
            self.scheduler_config.max_num_batched_tokens - drafts,
        )
        if remaining <= step:
            return 0
        first_end = -(-items_end // block) * block
        if first_end - start <= step:
            end = (start + room) // block * block
            return end - start if end >= first_end else 0
        if not getattr(self, "_warned_mm_split", False):
            self._warned_mm_split = True
            logger.warning(
                "TT scheduler: request %s has multimodal items up to token %d, "
                "past what one step from token %d holds (%d tokens); its "
                "prefill is split inside an item, which a model that declares "
                "supports_multimodal_prefill_resume False cannot compute. "
                "TTPlatform.validate_request rejects such a prompt; this "
                "request skipped it.",
                request.request_id,
                items_end,
                start,
                step,
            )
        return None

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int], list[int]]:
        """vLLM's encoder scheduling, all or nothing for a model that declares
        ``supports_multimodal_prefill_resume`` False.

        When the encoder cache or this step's encoder budget cannot take an
        item of the chunk, vLLM shortens the chunk to end before that item. A
        multimodal chunk of such a model must hold every item
        (``_multimodal_prefill_grant``), so the request waits instead, with
        nothing allocated in the encoder cache.
        """
        scheduled = super()._try_schedule_encoder_inputs(
            request,
            num_computed_tokens,
            num_new_tokens,
            encoder_compute_budget,
            shift_computed_tokens=shift_computed_tokens,
        )
        if (
            getattr(self, "_multimodal_prefill_block_size", 0)
            and scheduled[1] < num_new_tokens
        ):
            return [], 0, encoder_compute_budget, [], []
        return scheduled

    def _cap_prefill_chunk(
        self, request: Request, start: int, num_new_tokens: int
    ) -> int:
        """Cut a prefill chunk to the ``prefill_chunk_cap_table`` cap of its
        start depth.

        The cap is the ``max_chunk_tokens`` of the pair with the largest
        ``start_tokens <= start``; no cap applies before the first pair. Unlike
        vLLM's ``long_prefill_token_threshold``, it also holds for a request
        that runs alone. Decode steps are left alone, and so are the chunks
        that ``_align_prefill_chunk_end`` leaves alone as multimodal: every
        chunk of a multimodal request of a model that declares
        ``supports_multimodal_prefill_resume`` True. The multimodal part of a
        prompt of a model that declares False never reaches here
        (``_multimodal_prefill_grant``); its later text chunks are capped.
        """
        cap = self._prefill_chunk_cap_at(start)
        if cap is None or num_new_tokens <= cap:
            return num_new_tokens
        # vLLM's own bound on the prefill, past the prompt for a request that
        # replays its output after preemption.
        if start >= max(request.num_prompt_tokens, request.num_tokens - 1):
            return num_new_tokens
        if not getattr(self, "_multimodal_prefill_block_size", 0) and (
            getattr(request, "mm_features", None)
            or getattr(request, "has_encoder_inputs", False)
        ):
            return num_new_tokens
        return cap

    def _prefill_chunk_cap_at(self, start: int) -> int | None:
        """The ``prefill_chunk_cap_table`` cap of a chunk that starts at
        ``start``, or None when no pair starts at or before it."""
        table = getattr(self, "_prefill_chunk_cap_table", ())
        index = bisect.bisect_right(table, start, key=lambda pair: pair[0])
        return table[index - 1][1] if index else None

    def _align_prefill_chunk_end(
        self, request: Request, start: int, num_new_tokens: int
    ) -> int:
        """Round a *partial* prefill chunk down so it ends on the alignment grid.

        Chunks that finish the prompt and decode steps are left alone, and so
        are multimodal prompts of a model that declares
        ``supports_multimodal_prefill_resume`` True (the multimodal part of a
        prompt of a model that declares False never reaches here; see
        ``_multimodal_prefill_grant``). A request whose aligned chunk would be
        empty waits for the next step: only this step's leftover budget can
        stop it, because a full step reaches the grid from any start
        (``__init__`` validates the per-step cap to be at least the alignment).
        """
        align = getattr(self, "_prefill_chunk_alignment", 0)
        if align <= 1 or num_new_tokens <= 0:
            return num_new_tokens
        end = start + num_new_tokens
        # A request resumed from preemption replays its output tokens too, so
        # only a chunk that reaches the end of the sequence is the last one.
        if end >= request.num_tokens:
            return num_new_tokens
        if not getattr(self, "_multimodal_prefill_block_size", 0) and (
            getattr(request, "mm_features", None)
            or getattr(request, "has_encoder_inputs", False)
        ):
            return num_new_tokens
        aligned_end = end // align * align
        if aligned_end <= start:
            cap = self._full_step_prefill_cap()
            if num_new_tokens >= cap:
                # A chunk of at least the alignment always crosses a grid
                # point, and the cap was validated to be at least the
                # alignment, so a full step never lands here.
                raise RuntimeError(
                    f"a full {num_new_tokens}-token prefill step (per-step cap "
                    f"{cap}) from position {start} misses the {align}-token "
                    "alignment grid; _validate_prefill_chunk_alignment admits "
                    "no cap below the alignment"
                )
            # This step's leftover budget is what stopped the request: it
            # waits for a step whose budget reaches the grid. A chunk that ends
            # short of the grid would make the next chunk start off it, which a
            # model aligning its prefill state to the grid cannot resume.
            return 0
        return aligned_end - start

    def _full_step_prefill_cap(self) -> int:
        """Most prompt tokens one scheduling step can give a single new request."""
        cfg = getattr(self, "scheduler_config", None)
        if cfg is None:
            return sys.maxsize
        limit, _ = _effective_prefill_step_limit(cfg)
        return limit

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # NOTE: `throttle_prefills` accepted for interface compatibility with the base
        #        scheduler but unused - TT separates prefill/decode explicitly.
        self._step_granted_tokens = 0
        self._step_granted_rows = 0
        has_pending_prefill = self._has_pending_prefill()
        # A partial prefill occupies ``running`` but cannot produce a decode
        # token, so it must not make the decode fallback below look viable.
        has_running_decode = any(
            not request.is_prefill_chunk for request in self.running
        )
        mode = self._forced_mode

        if mode == TTSchedulingMode.PREFILL_ONLY:
            # Forced mode is shared by every lane. Return an empty prefill
            # result unchanged so the coordinator can decide whether all lanes
            # should fall back to decode together.
            result = self._schedule_prefill_only()
            return self._finalize_scheduler_output(result, is_decode=False)
        if mode == TTSchedulingMode.DECODE_ONLY:
            if has_pending_prefill:
                # Hide the waiting queues and partial prefills so the base
                # scheduler cannot admit prefill work.
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result, is_decode=True)
            # No pending prefill: base scheduler naturally runs decode-only.
            result = super().schedule()
            return self._finalize_scheduler_output(result, is_decode=True)

        kept = getattr(self, "_deferred_prefill", None)
        result = self._schedule_default_mode(has_pending_prefill, has_running_decode)
        self._order_cleanup_around_kept_prefill(kept, result)
        self._unreleased_step = (
            result if getattr(result, _TT_OUTPUT_FRAME_REQ_IDS_ATTR, ()) else None
        )
        return result

    def _order_cleanup_around_kept_prefill(
        self, kept: SchedulerOutput | None, result: SchedulerOutput
    ) -> None:
        """Keep cleanup for a kept prefill's requests until that prefill has run.

        ``_schedule_default_mode`` returns another step while a prefill is kept
        only under a pause in mode "keep". The base pass of that step drains
        the cleanup of a request aborted during the pause.
        ``TTModelRunner._update_states`` applies ``finished_req_ids`` before it
        reads the scheduled rows. If a kept row's request were finished first,
        the kept prefill would recreate it from ``scheduled_new_reqs`` and
        nothing would remove it again, or, for a chunked-prefill continuation,
        would not find it in ``requests``. So those ``finished_req_ids``, and
        every freed encoder entry (a kept chunk can read its request's encoder
        output), go to the first step after the kept prefill. That is the order
        an abort between the two calls gets outside a pause.
        """
        finished = getattr(self, "_finished_after_kept_prefill", set())
        freed = getattr(self, "_freed_encoder_after_kept_prefill", [])
        if result is kept:
            return
        if kept is not None:
            kept_finished = result.finished_req_ids & kept.num_scheduled_tokens.keys()
            result.finished_req_ids -= kept_finished
            self._finished_after_kept_prefill = finished | kept_finished
            self._freed_encoder_after_kept_prefill = (
                freed + result.free_encoder_mm_hashes
            )
            result.free_encoder_mm_hashes = []
            return
        if finished or freed:
            result.finished_req_ids |= finished
            result.free_encoder_mm_hashes = freed + result.free_encoder_mm_hashes
            self._finished_after_kept_prefill = set()
            self._freed_encoder_after_kept_prefill = []

    def _schedule_default_mode(
        self, has_pending_prefill: bool, has_running_decode: bool
    ) -> SchedulerOutput:
        deferred = getattr(self, "_deferred_prefill", None)
        # Under a pause in mode "keep" (PAUSED_ALL) the base scheduler schedules
        # nothing, but the engine still calls schedule() while finished
        # requests wait to be reported; the kept prefill waits for the pause
        # to end as well.
        if deferred is not None and self.pause_state != PauseState.PAUSED_ALL:
            # Finalized when it was kept; see _schedule_release_step.
            self._deferred_prefill = None
            return deferred
        held = getattr(self, "_unreleased_step", None)
        if (
            held is not None
            and self.scheduler_config.async_scheduling
            and scheduler_output_has_prefill_work(held)
        ):
            # Whatever comes next would run before the engine commits the
            # prefill; see _schedule_release_step.
            return self._finalize_scheduler_output(
                self._schedule_release_step(), is_decode=False
            )

        # Default mode:
        # Prefer prefill whenever prefill work is pending, so new requests are
        # admitted and partial prefills advance.
        if has_pending_prefill:
            if self._decode_interleave.wants_decode_step(
                has_pending_prefill=True, has_running_decode=has_running_decode
            ):
                # A run of prefill steps has reached its bound. Spend this step
                # on the running decodes so their inter-token latency does not
                # scale with the pending prompt's length; the pending prefill
                # resumes on the next step.
                self._decode_interleave.record_step(
                    is_decode=True, prefill_pending=True
                )
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result, is_decode=True)
            # A held step here is a decode or a verify.
            release_step = (
                self._schedule_release_step()
                if held is not None and self.scheduler_config.async_scheduling
                else None
            )
            with self._skip_kv_connector_meta_for_discarded_prefill(has_running_decode):
                prefill_result = self._schedule_prefill_only()
            if release_step is not None and self._prefill_runs_behind_verify(
                held, prefill_result
            ):
                _carry_cleanup(release_step, prefill_result)
                self._decode_interleave.record_step(
                    is_decode=False, prefill_pending=True
                )
                return self._finalize_scheduler_output(prefill_result, is_decode=False)
            if release_step is not None:
                if prefill_result.total_num_scheduled_tokens > 0:
                    self._decode_interleave.record_step(
                        is_decode=False, prefill_pending=True
                    )
                    release_step = self._finalize_scheduler_output(
                        release_step, is_decode=False
                    )
                    self._deferred_prefill = self._finalize_scheduler_output(
                        prefill_result, is_decode=False
                    )
                    return release_step
                _carry_cleanup(release_step, prefill_result)
            # If prefill cannot make progress (e.g. KV pressure), do not stall
            # decode. Fall back to decode-only so running requests can advance
            # and free capacity for a later prefill admission.
            if prefill_result.total_num_scheduled_tokens == 0 and has_running_decode:
                self._decode_interleave.record_step(
                    is_decode=True, prefill_pending=True
                )
                self.finished_req_ids |= prefill_result.finished_req_ids
                self.reset_preempted_req_ids |= (
                    prefill_result.preempted_req_ids or set()
                )
                result = self._schedule_decode_only()
                # The decode pass already carries the finished and preempted
                # request IDs given back above; the freed encoder cache
                # entries still have to be added.
                result.free_encoder_mm_hashes = (
                    prefill_result.free_encoder_mm_hashes
                    + result.free_encoder_mm_hashes
                )
                return self._finalize_scheduler_output(result, is_decode=True)
            self._decode_interleave.record_step(is_decode=False, prefill_pending=True)
            return self._finalize_scheduler_output(prefill_result, is_decode=False)

        # No pending prefill work in default mode: run decode-only naturally.
        self._decode_interleave.record_step(is_decode=True, prefill_pending=False)
        result = super().schedule()
        return self._finalize_scheduler_output(result, is_decode=True)

    @contextmanager
    def _skip_kv_connector_meta_for_discarded_prefill(self, has_running_decode: bool):
        """``schedule`` discards a prefill pass when it schedules no tokens and
        decodes are running, then runs a decode pass instead. KV connector
        metadata built for the discarded pass would be lost with it. Skip
        ``_build_kv_connector_meta`` for that pass; the decode pass builds it.
        """
        original = self._build_kv_connector_meta

        def build_unless_discarded(connector, scheduler_output):
            no_prefill_tokens = scheduler_output.total_num_scheduled_tokens == 0

            if no_prefill_tokens and has_running_decode:
                return None
            return original(connector, scheduler_output)

        self._build_kv_connector_meta = build_unless_discarded
        try:
            yield
        finally:
            self._build_kv_connector_meta = original

    def _finalize_scheduler_output(
        self, scheduler_output: SchedulerOutput, *, is_decode: bool
    ) -> SchedulerOutput:
        if is_decode:
            rows = len(scheduler_output.num_scheduled_tokens)
            if rows > getattr(self, "_widest_decode_batch_size", 0):
                self._widest_decode_batch_size = rows
                logger.info(
                    "TT scheduler: widest decode batch reached %d request row(s)",
                    rows,
                )
        pending_reset_discards = getattr(
            self, "_pending_forced_reset_discard_counts", {}
        )
        # A reset while a prefill is kept counts the frames that prefill
        # reserved. The runner can discard a frame only after the step that
        # produces it has run, so the counts wait for a step handed out after
        # the kept prefill.
        if pending_reset_discards and getattr(self, "_deferred_prefill", None) is None:
            set_tt_forced_reset_discard_counts(scheduler_output, pending_reset_discards)
            self._pending_forced_reset_discard_counts = {}
        return scheduler_output

    def _schedule_prefill_only(self) -> SchedulerOutput:
        """Schedule prefill work: waiting requests and partial continuations.

        Temporarily hides the running *decode* requests so the base scheduler's
        running loop only advances partial prefills, and its waiting loop
        admits new ones. With ``max_concurrent_partial_prefills`` set, it also
        hides the partial prefills and waiting requests that
        ``PartialPrefillLimit`` holds back for this step. The hidden requests
        still hold their slots or still wait for this step, so the pass keeps
        two counts true for them:

        - Both running caps are lowered by the hidden decodes and hidden
          partial prefills, so the waiting loop respects the true capacity:
          the loop admits against ``max_num_active_reqs``, and the end-of-step
          check asserts ``max_num_running_reqs``.
        - ``_num_hidden_decodes`` and ``_held_prefills`` hold them, so
          ``get_request_counts`` still reports them and the base pass keeps
          ``long_prefill_token_threshold`` on a prompt they wait for.

        Every hidden request is back in its queue, in its place, when this
        returns, so a prefill kept in ``_deferred_prefill`` and
        ``prefill_qualifies_behind_verify`` see only what the pass scheduled.
        """
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]
        held = self._select_held_prefills(partial_prefills)
        waiting_order = list(self.waiting) if held.waiting else []

        saved_max = self.max_num_running_reqs
        saved_active = self.max_num_active_reqs
        num_hidden_running = len(pure_decodes) + len(held.running)
        self.running = [r for r in partial_prefills if r not in held.running]
        if held.waiting:
            self.waiting.remove_requests(held.waiting)
        self.max_num_running_reqs = max(0, saved_max - num_hidden_running)
        self.max_num_active_reqs = max(0, saved_active - num_hidden_running)
        self._num_hidden_decodes = len(pure_decodes)
        self._held_prefills = held
        try:
            result = super().schedule()
        finally:
            self._restore_held_prefills(held, partial_prefills, waiting_order)
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
            self.max_num_active_reqs = saved_active
            self._num_hidden_decodes = 0
            self._held_prefills = HeldPrefills()
        return result

    def _select_held_prefills(self, partial_prefills: list[Request]) -> HeldPrefills:
        """The requests ``max_concurrent_partial_prefills`` holds back from this
        prefill pass; none when the key is absent."""
        limit = getattr(self, "_partial_prefill_limit", None)
        if limit is None:
            return HeldPrefills()
        return limit.select(partial_prefills, list(self.waiting), self._is_multi_chunk)

    def _restore_held_prefills(
        self,
        held: HeldPrefills,
        partial_prefills: list[Request],
        waiting_order: list[Request],
    ) -> None:
        """Put the requests a prefill pass held back in their places again.

        A held partial prefill goes back to its place among the partial
        prefills, ahead of the requests the pass admitted. A held waiting
        request goes back to its place in ``waiting``: requests the pass put
        into ``waiting`` (preempted ones) stay in front, and requests it
        admitted stay out.
        """
        if held.running:
            kept = {r.request_id for r in self.running}
            known = {r.request_id for r in partial_prefills}
            self.running = [
                r for r in partial_prefills if r in held.running or r.request_id in kept
            ] + [r for r in self.running if r.request_id not in known]
        if held.waiting:
            remaining = list(self.waiting)
            present = {r.request_id for r in remaining}
            known = {r.request_id for r in waiting_order}
            order = [r for r in remaining if r.request_id not in known] + [
                r for r in waiting_order if r in held.waiting or r.request_id in present
            ]
            self.waiting.remove_requests(remaining)
            for request in order:
                self.waiting.add_request(request)

    def _is_multi_chunk(self, request: Request) -> bool:
        """Whether the uncached remainder of ``request`` needs more than one
        prefill chunk.

        A request that has computed nothing yet starts where its prefix-cache
        hit would end (``_estimated_cache_hit``).
        """
        start = request.num_computed_tokens or self._estimated_cache_hit(request)
        return request.num_tokens - start > self._prefill_chunk_limit(request, start)

    def _prefill_chunk_limit(self, request: Request, start: int) -> int:
        """Most tokens one prefill step gives ``request`` from ``start``.

        As ``Scheduler.schedule`` sizes a chunk: at most
        ``max_num_scheduled_tokens``, and ``max_num_batched_tokens`` less the
        draft slots; then a positive ``long_prefill_token_threshold`` (vLLM's
        lift for a lone request and its adaptive floor are not modelled), and
        the ``prefill_chunk_cap_table`` cap. The multimodal part of a prompt of
        a model that declares ``supports_multimodal_prefill_resume`` False
        takes a whole step instead (``_multimodal_prefill_grant``).
        """
        limit = min(
            self.max_num_scheduled_tokens,
            self.scheduler_config.max_num_batched_tokens - self._draft_slots(),
        )
        if getattr(self, "_multimodal_prefill_block_size", 0) and request.mm_features:
            items_end = max(
                f.mm_position.offset + f.mm_position.length for f in request.mm_features
            )
            if start < items_end:
                return limit
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < limit:
            limit = threshold
        return self._cap_prefill_chunk(request, start, limit)

    def _estimated_cache_hit(self, request: Request) -> int:
        """Prompt tokens vLLM would take from the prefix cache if it admitted
        ``request`` now.

        Asks the KV cache coordinator's ``find_longest_cache_hit`` with the
        bound ``get_computed_blocks`` uses (``num_tokens - 1``). For a hybrid
        model the hit must hold in every KV cache group, so a full-attention
        hit counts only up to a block boundary where each ``MambaSpec`` group
        also has a cached GDN state block. The lookup only reads the block
        pool: unlike ``get_computed_blocks``, it records no prefix-cache
        statistics and emits no KV cache events. A model's
        ``confirm_prefix_cache_hit`` and a KV connector's external tokens are
        not asked, so for such a launch the hit vLLM schedules can differ.
        """
        manager = self.kv_cache_manager
        if not manager.prefix_cache_lookup_enabled(request):
            return 0
        _, hit, _ = manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        return hit

    def _schedule_decode_only(self) -> SchedulerOutput:
        """Schedule only running decode requests.

        Temporarily hides both the ``waiting`` and ``kv_holding_waiting``
        queues so the base scheduler's waiting loop is a no-op and cannot admit
        a prefill (for example a grammar-ready structured-output request) into
        this decode step, and hides partial prefills so their next chunk is not
        scheduled into it either.  Any requests that get preempted during
        decode scheduling are merged back into the original queues afterwards.
        """
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_waiting = self.waiting
        saved_kv_holding = self.kv_holding_waiting
        self.waiting = create_request_queue(self.policy)
        self.kv_holding_waiting = create_request_queue(self.policy)
        if partial_prefills:
            self.running = [r for r in self.running if not r.is_prefill_chunk]
        try:
            result = super().schedule()
        finally:
            if self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            if self.kv_holding_waiting:
                saved_kv_holding.prepend_requests(self.kv_holding_waiting)
            self.waiting = saved_waiting
            self.kv_holding_waiting = saved_kv_holding
            if partial_prefills:
                self.running.extend(partial_prefills)
        return result

    def _prefill_runs_behind_verify(
        self, held: SchedulerOutput, prefill: SchedulerOutput
    ) -> bool:
        """Whether ``prefill`` goes to the runner ahead of the release step.

        With ``supports_prefill_over_pending_spec`` the model accepts a prefill
        while one verify it returned is unread, and ``TTModelRunner`` then
        submits the prefill behind that verify instead of reading it first
        (``spec_chain.prefill_over_pending``). The verify's tokens are then
        published after the prefill instead of before it. ``held`` must be a
        speculative decode step (every request carries scheduled drafts; the
        runner may still answer it as an ordinary decode, and then drains it)
        and ``prefill`` a step the runner puts behind a verify: prefill work
        only, with no resumed or preempted request, no multimodal input and no
        forced-reset discard counts waiting.
        """
        if not (
            self._prefill_over_pending_spec
            and self.scheduler_config.async_scheduling
            and prefill.total_num_scheduled_tokens > 0
        ):
            return False
        scheduled = set(held.num_scheduled_tokens)
        if (
            not scheduled
            or set(held.scheduled_spec_decode_tokens) != scheduled
            or scheduler_output_has_prefill_work(held)
        ):
            return False
        return prefill_qualifies_behind_verify(prefill) and not getattr(
            self, "_pending_forced_reset_discard_counts", {}
        )

    def _schedule_release_step(self) -> SchedulerOutput:
        """Schedule an empty step that lets the engine commit the held step.

        Under async scheduling ``EngineCore.step_with_batch_queue`` commits a
        step's output (``update_from_output``) only after it has scheduled and
        executed the next step. The engine executes this empty step instead;
        its two-entry batch queue is then full, so it commits the held step
        (``_unreleased_step``) before it returns. ``_schedule_default_mode``
        returns it in two cases.

        The held step is a prefill. ``TTModelRunner`` ran it to the end inside
        ``execute_model``, so its tokens are ready, and any next step would
        hold them back for its own ``execute_model``: a whole prefill, or the
        build and submission of a decode or verify, which after a prefill
        rebuilds its full inputs. A prefill chunk that does not complete its
        prompt produces no output, is never held, and is not followed by this
        step.

        The held step is a decode or a verify and a prefill comes next. The
        decode's tokens would wait for the whole prefill. Only the prefill
        pass can tell whether a prefill happens, so ``_schedule_default_mode``
        runs it right after this pass. When it schedules tokens, the prefill
        is kept in ``_deferred_prefill`` and the next ``schedule()`` call
        returns it. When it schedules nothing, there is no prefill to go ahead
        of, so this step is not returned and ``_carry_cleanup`` adds what it
        drained to the step returned instead; running decodes then keep their
        overlap with the engine. Nor is it returned when the prefill may run
        behind the held verify (``_prefill_runs_behind_verify``); the prefill
        then carries the cleanup. Running this pass first keeps that cleanup
        ahead of the prefill, in the order the runner receives the two steps.

        Between the two calls the engine commits the held step and handles
        client calls: aborts, pauses and prefix-cache resets. The kept prefill
        is finalized when it is kept and handed out unchanged, and while it is
        kept ``_finalize_scheduler_output`` holds back forced-reset discard
        counts: a reset in between counts the frames the prefill reserved, and
        the runner can discard them only after the prefill has run. A request
        aborted in between is still prefilled; the runner drops it with the
        next step's ``finished_req_ids``. Under a pause in mode "keep" the
        prefill is not handed out until the pause ends, and
        ``_order_cleanup_around_kept_prefill`` keeps the cleanup of its
        requests until after it.

        Every queue is hidden, so the base scheduler schedules nothing and
        preempts nothing, and the step carries only what the pass drains.
        """
        saved_waiting = self.waiting
        saved_kv_holding = self.kv_holding_waiting
        saved_running = self.running
        self.waiting = create_request_queue(self.policy)
        self.kv_holding_waiting = create_request_queue(self.policy)
        self.running = []
        try:
            return super().schedule()
        finally:
            self.waiting = saved_waiting
            self.kv_holding_waiting = saved_kv_holding
            self.running = saved_running

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Backstop against a stale-canvas resume vLLM's AR reset cannot represent.

        The engine layer (``_install_block_output_reset_abort_patch``) aborts
        running block requests and notifies their clients before delegating,
        so a reset requested through ``EngineCoreProc`` succeeds. Live block
        requests reach this guard only from engines that cannot notify the
        owning clients (a bare in-process ``EngineCore``) or callers that
        bypassed the engine layer; refusing here beats silently removing a
        request someone is still waiting on.
        """
        if self._requests_cannot_resume and reset_running_requests and self.running:
            message = (
                "Cannot reset prefix cache while a request that cannot resume "
                "(block output or stage outputs) is running; finish or abort the "
                "request first."
            )
            # pause_generation(mode="keep") retains live requests on purpose,
            # so a refused reset is an expected result there: return the
            # refusal the reset API defines instead of raising.
            # _install_block_output_pause_guard_patch refuses a keep-mode
            # pause that would clear the cache before it reaches this.
            if self.pause_state == PauseState.PAUSED_ALL:
                logger.error("%s", message)
                return False
            raise RuntimeError(message)
        # Upstream preempts these requests in drop mode and discards their stale
        # output itself. TT receives one output frame for a speculative
        # reservation of 1+K tokens, so publish frame counts to the runner,
        # which must keep those frames out of request state before
        # resumed-prefill inputs are built.
        reset_candidates = (
            [
                (
                    request,
                    self._pending_async_output_frames.get(request.request_id, 0),
                )
                for request in self.running
                if request.num_output_placeholders > 0
            ]
            if reset_running_requests
            else []
        )
        try:
            return super().reset_prefix_cache(reset_running_requests, reset_connector)
        finally:
            for request, frame_count in reset_candidates:
                if request.status == RequestStatus.PREEMPTED and frame_count > 0:
                    pending = self._pending_forced_reset_discard_counts
                    pending[request.request_id] = (
                        pending.get(request.request_id, 0) + frame_count
                    )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Reserve the physical output each scheduled step will commit.

        The platform removes the upstream diffusion marker, so vLLM reserves its
        normal one sampled-token placeholder. A plain block-output model commits
        a K-token canvas on every step, so reserve the remaining K-1 positions.
        An adaptive block-output model commits K only on a step that satisfies
        the ``block_step`` predicate below, and one token on every other step,
        so the extra reservation is per step. Each step's decision is recorded
        on its own SchedulerOutput because update_from_output must reconcile a
        commit against the decision that produced it.
        """
        self._drop_admission_spec_padding(scheduler_output)
        super()._update_after_schedule(scheduler_output)
        output_frame_req_ids = tuple(
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if not self.requests[req_id].is_prefill_chunk
        )
        # The parent computes whether this chunk produces output. Retain that
        # decision on this step because a later schedule changes the request.
        set_tt_scheduler_output_field(
            scheduler_output, _TT_OUTPUT_FRAME_REQ_IDS_ATTR, output_frame_req_ids
        )
        pending = getattr(self, "_pending_async_output_frames", None)
        if pending is None:
            pending = self._pending_async_output_frames = {}
        for req_id in output_frame_req_ids:
            pending[req_id] = pending.get(req_id, 0) + 1
        if self.num_spec_tokens and not self.scheduler_config.async_scheduling:
            # ``AsyncScheduler`` leaves every scheduled request holding
            # ``[-1] * num_spec_tokens``, which upstream's GPU runner
            # overwrites from its own state in ``_prepare_input_ids``. On a
            # synchronous launch the TT runner has no such step: it verifies
            # whatever the scheduler delivers, so a placeholder surviving here
            # becomes a draft the accept walk compares against, matches (the
            # model is handed the same placeholder), and commits as an output
            # token.
            #
            # Cleared rather than restored to the ids just scheduled: a
            # proposal is handed over once, so a request whose row proposed
            # nothing this step must speculate on nothing next step rather
            # than replay a spent proposal. Drafts reach a request only
            # through ``update_draft_token_ids``.
            #
            # Left standing on an asynchronous launch, because there they are
            # the only lookahead reservation the request gets: upstream stops
            # routing drafts through the scheduler (``EngineCore.post_step``
            # skips ``take_draft_token_ids``), the next schedule budgets
            # ``1 + len(spec_token_ids)`` positions for the request, and
            # ``TTModelRunner._drafts_to_verify`` reads the placeholders as
            # that reservation and verifies the proposal the runner holds.
            for req_id in scheduler_output.num_scheduled_tokens:
                self.requests[req_id].spec_token_ids = []
        if not self._is_block_output_model:
            return
        extra_placeholders = (
            self._output_tokens_per_step - self.num_sampled_tokens_per_step
        )
        # Adaptive: the model emits its block ONLY on a SOLO DECODE step; every
        # other step (batched decodes, and the prefill that host-samples the
        # anchor) commits exactly one plain token and reserves one placeholder.
        # This MUST match the model's own gate, which runs the spec block in
        # decode_forward when batch == 1 and a plain token otherwise:
        #   - solo == batch 1 (exactly one request scheduled this step);
        #   - decode == the prompt was fully computed BEFORE this step (see the
        #     decode test below, which subtracts this step's scheduled tokens).
        # The decision is attached to THIS step's SchedulerOutput (not a Request
        # slot): under async the next step's schedule() overwrites Request state
        # before this step's output commits, but the engine core hands the
        # matching SchedulerOutput back to update_from_output, so the map always
        # pairs with the output being committed.
        solo = len(scheduler_output.num_scheduled_tokens) == 1
        if self._is_adaptive_block:
            self._mirror_spec_session(scheduler_output, solo)
        decisions: dict[str, bool] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue
            if self._is_adaptive_block:
                # num_computed_tokens is already advanced by THIS step's
                # scheduled tokens here, so subtract them back out: a step is a
                # decode iff the prompt was fully computed BEFORE it. This is a
                # pure scheduling-side quantity -- output-commit timing (which
                # differs between sync tests and the pipelined engine loop)
                # cannot skew it. A resumed replay scheduling prompt+output
                # tokens lands back below the prompt boundary and correctly
                # stays a non-block step.
                scheduled = scheduler_output.num_scheduled_tokens[req_id]
                is_decode = (
                    request.num_computed_tokens - scheduled >= request.num_prompt_tokens
                )
                # Owning the model's single spec session is the whole gate:
                # the model serves a solo decode as plain baseline whenever it
                # has no session for THAT request, so reserving a block for a
                # non-owner would reserve a width the model cannot emit (see
                # _mirror_spec_session). The capture frontier is deliberately
                # NOT re-derived here -- it is a property of the PREFILL that
                # armed the session, already recorded in the ownership mirror,
                # and re-deriving it from the request's CURRENT length would
                # flip a live session's reservation to width 1 mid-generation
                # while the model keeps emitting blocks.
                block_step = solo and is_decode and self._spec_session_owner == req_id
            else:
                block_step = True
            if block_step:
                request.num_output_placeholders += extra_placeholders
            decisions[req_id] = block_step
        set_tt_block_step_decisions(scheduler_output, decisions)

    def _drop_admission_spec_padding(self, scheduler_output: SchedulerOutput) -> None:
        """Schedule a request admitted with one uncomputed token for that token.

        With speculative decoding on, ``Scheduler.schedule`` pads a request
        that it admits from a waiting queue with exactly one uncomputed token,
        such as a new request whose prefix-cache hit covers all but its last
        token or a preempted request that resumes with such a hit: it schedules
        ``1 + num_spec_tokens`` tokens and lists ``[-1] * num_spec_tokens`` for
        the request in ``scheduled_spec_decode_tokens``. It skips the padding
        in a few cases, among them a step that scheduled a partial prefill
        before the request. The padding keeps upstream's GPU runner on a
        uniform decode shape for its CUDA graph: that runner clamps the ``-1``
        input ids to 0, its rejection sampler rejects every ``-1`` draft, so
        the request's token comes from its last real token, and
        ``update_from_output`` rolls ``num_computed_tokens`` back by the
        rejected drafts.

        ``TTModelRunner`` runs every new and resumed request as a prefill, and
        a prefill verifies no drafts: it would prefill ``num_spec_tokens``
        positions past the request's last token from stale token ids and sample
        the next token after the last of them. So the padding is removed here,
        before the base class adds the scheduled tokens to
        ``num_computed_tokens`` and to the asynchronous output placeholders.
        The KV blocks vLLM allocated for the padded positions stay with the
        request; those positions follow its last token, and only its own later
        steps write them.
        """
        drafts_by_req = scheduler_output.scheduled_spec_decode_tokens
        if not drafts_by_req:
            return
        admitted = {new.req_id for new in scheduler_output.scheduled_new_reqs}
        admitted |= scheduler_output.scheduled_cached_reqs.resumed_req_ids
        for req_id in sorted(admitted & drafts_by_req.keys()):
            drafts = drafts_by_req[req_id]
            request = self.requests[req_id]
            scheduled = scheduler_output.num_scheduled_tokens[req_id]
            if (
                any(token != PLACEHOLDER_TOKEN_ID for token in drafts)
                or scheduled != 1 + len(drafts)
                or request.num_computed_tokens + 1 != request.num_tokens
            ):
                raise RuntimeError(
                    f"request {req_id} was admitted with {scheduled} scheduled "
                    f"tokens and drafts {drafts}, with "
                    f"{request.num_computed_tokens} of its {request.num_tokens} "
                    "tokens computed; vLLM gives an admitted request drafts "
                    "only as the padding of its one uncomputed token, which is "
                    "all TTScheduler can remove"
                )
            del drafts_by_req[req_id]
            scheduler_output.num_scheduled_tokens[req_id] = 1
            scheduler_output.total_num_scheduled_tokens -= len(drafts)

    def _spec_frontier_ok(self, measured_len: int) -> bool:
        """Whether the model arms a session for a prefill of ``measured_len``.

        The model measures its capture frontier against the ``prompt_lens`` the
        runner hands it, which is ``input_positions + chunk_lens`` -- the tokens
        computed INCLUDING this step. On a replay resumed from preemption that
        spans the generated output too, so it is NOT ``num_prompt_tokens``:
        measuring the prompt alone let a resumed request cross the frontier on
        the model side only, which drops the session while this scheduler still
        reserved a block, and the width check then kills the engine core
        (vllm-tt-plugin#118.2). The caller passes ``num_computed_tokens``, which
        ``_update_after_schedule`` has already advanced by this step.
        """
        return (
            self._adaptive_block_max_prompt == 0
            or measured_len <= self._adaptive_block_max_prompt
        )

    def _mirror_spec_session(self, scheduler_output, solo: bool) -> None:
        """Track which request owns the adaptive model's SINGLE spec session.

        The model keeps one global session: drafter taps are captured during a
        request's own prefill, and only the request that owns them can emit a
        multi-token block. Every transition of that session is driven by the
        shape of a step, so the scheduler can mirror it exactly without asking
        the model -- and that is what keeps the width this scheduler RESERVES
        equal to the width the model EMITS:

        * a PREFILL step re-seats the session. The model captures taps only for
          a solo, spec-eligible prefill and drops the session on any other
          prefill (batched, or a prompt over the capture frontier).
        * a BATCHED decode step destroys it. The model releases the decoder and
          serves plain baseline from then on, and it never re-arms, because
          taps only ever come from a prefill. This is the case that made a
          benchmark sweep fail: as concurrency drains back to one request, that
          request is solo and eligible but no longer owns a session.
        * a SOLO decode by the OWNER leaves ownership alone -- it bootstraps
          its pending taps and keeps the session for the rest of its life.
        * a SOLO decode by ANY OTHER request destroys it. Async scheduling can
          skip the owner once it has reached max_tokens (upstream guards that
          skip on num_output_placeholders), leaving a different request alone
          on the next step while the session is still armed. The model releases
          the session on that step for the same reason -- it would otherwise
          speculate from the owner's residual taps -- and this mirrors that
          release, so the reserved width stays 1 for the non-owner
          (vllm-tt-plugin#118.1).

        A TT step is never mixed prefill+decode (docs/SCHEDULING.md), so the
        first scheduled request classifies the whole step.
        """
        owner = self._spec_session_owner
        # A finished or aborted owner no longer holds the session: the model
        # clears it in release_request, and the id is gone from self.requests.
        if owner is not None and owner not in self.requests:
            owner = None
        scheduled_tokens = scheduler_output.num_scheduled_tokens
        first_id = next(iter(scheduled_tokens), None)
        if first_id is not None:
            request = self.requests[first_id]
            step_is_decode = (
                request.num_computed_tokens - scheduled_tokens[first_id]
                >= request.num_prompt_tokens
            )
            if not step_is_decode:
                # num_computed_tokens is advanced by this step already, so it is
                # exactly the prompt_lens the model measures its frontier on.
                owner = (
                    first_id
                    if (solo and self._spec_frontier_ok(request.num_computed_tokens))
                    else None
                )
            elif not solo or first_id != owner:
                owner = None
        self._spec_session_owner = owner

    def get_request_counts(self) -> tuple[int, int]:
        """Count the running and waiting requests, hidden ones included.

        Returns
        -------
        tuple[int, int]
            ``(num_running_reqs, num_waiting_reqs)`` as vLLM's
            ``Scheduler.get_request_counts`` counts the queues, plus the
            requests ``_schedule_prefill_only`` removes from them for its base
            pass: in the running count the decodes and the held partial
            prefills, in the waiting count the held waiting requests. Outside
            that pass no request is hidden and the counts are vLLM's.

        Notes
        -----
        vLLM's ``Scheduler.schedule`` applies ``long_prefill_token_threshold``
        only when these counts add up to more than one request: the cap stops a
        long prefill from starving the other requests, and a lone request has
        none to starve. A TT step cannot mix prefill and decode rows, so every
        decode hidden from a prefill pass waits for the whole step. Without
        them, a long prompt that arrives beside running decodes, and the only
        continuing partial prefill, count as one request and take the whole
        token budget in one step. The same sum sets the fair share at which
        ``long_prefill_token_threshold_adaptive`` floors the cap.

        A request ``max_concurrent_partial_prefills`` holds back from the pass
        waits for the whole step too.

        ``TTLaneCoordinator`` lane routing and ``make_stats`` call this between
        scheduling passes, when no request is hidden.
        """
        num_running, num_waiting = super().get_request_counts()
        held = self._held_prefills
        return (
            num_running + self._num_hidden_decodes + len(held.running),
            num_waiting + len(held.waiting),
        )

    def make_stats(self, *args, **kwargs) -> SchedulerStats | None:
        """None on a lane: TTLaneCoordinator.make_stats makes the step's one
        SchedulerStats for every lane. A lane's own would count only its
        requests, and the coordinator replaces it on the first client's output
        alone.
        """
        if self.stats_made_by_coordinator:
            return None
        return super().make_stats(*args, **kwargs)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Bind this step's block-step decisions before the base loop commits
        its outputs, so ``_update_request_with_output`` reads the decision that
        produced THIS output rather than a later schedule's overwrite, and
        consume only the output frames this scheduled step reserved. On a
        speculating launch the step's scheduled drafts are bound the same way,
        for ``_count_commit_past_scheduled_drafts``. A committed step is no
        longer the held step (``_schedule_release_step``)."""
        if self._is_block_output_model:
            self._committing_block_step_decisions = get_tt_block_step_decisions(
                scheduler_output
            )
        if self.num_spec_tokens:
            self._committing_spec_drafts = scheduler_output.scheduled_spec_decode_tokens
        try:
            return super().update_from_output(scheduler_output, model_runner_output)
        finally:
            self._committing_block_step_decisions = {}
            self._committing_spec_drafts = None
            self.tt_release_output_frames(scheduler_output)
            if scheduler_output is getattr(self, "_unreleased_step", None):
                self._unreleased_step = None

    def tt_release_output_frames(self, scheduler_output: SchedulerOutput) -> None:
        """Release the output frames ``scheduler_output`` reserved.

        Call it exactly once per committed step, as ``update_from_output`` does from
        its ``finally``. A subclass whose ``update_from_output`` does not reach this
        class's must call it the same way.
        """
        pending = self._pending_async_output_frames
        for req_id in getattr(scheduler_output, _TT_OUTPUT_FRAME_REQ_IDS_ATTR, ()):
            count = pending.get(req_id, 0)
            if count <= 1:
                pending.pop(req_id, None)
            else:
                pending[req_id] = count - 1

    def _count_commit_past_scheduled_drafts(
        self, request: Request, new_token_ids: list[int], is_stale: bool
    ) -> None:
        """Reconcile a commit past the drafts, or refuse it under a queued step.

        Schedule time advanced ``num_computed_tokens`` and
        ``num_output_placeholders`` by the step's 1 + d tokens. vLLM reconciles
        both with the commit only for a row it scheduled drafts for (d >= 1),
        by subtracting ``num_rejected = d - (len(new_token_ids) -
        num_sampled_tokens_per_step)``, which is negative, and so advances
        them, when the row commits past its drafts. A ``fused_sample`` verify
        may commit several tokens on a row scheduled none, where vLLM applies
        nothing: the request would hold c new tokens against counts advanced by
        one, so the placeholder count would underflow and the next step would
        schedule the c - 1 tokens again as new input. The same reconciliation
        is applied here, under vLLM's guards, just before the tokens are
        appended.

        Either correction is right only while no later step of the request is
        scheduled. A step scheduled before this commit started from this
        step's scheduled positions, so after a commit past the drafts it would
        start behind the request, and the runner would read the gap as prompt
        and prefill it. Only asynchronous scheduling schedules a step before
        the previous one's output, and ``TTPlatform`` refuses ``fused_sample``
        there, so such a commit is refused here rather than reconciled.

        Raises
        ------
        RuntimeError
            If the row committed past its drafts while a later step of the
            request was already scheduled.
        """
        drafts = self._committing_spec_drafts
        if drafts is None or is_stale:
            return
        num_drafts = len(drafts.get(request.request_id) or ())
        num_accepted = len(new_token_ids) - self.num_sampled_tokens_per_step
        if num_accepted <= num_drafts:
            return
        # update_from_output has already taken this step's tokens off the
        # count, so what remains belongs to steps scheduled after it.
        if request.num_in_flight_tokens > 0:
            raise RuntimeError(
                f"request {request.request_id} committed {len(new_token_ids)} "
                f"tokens on a step that scheduled {num_drafts} drafts, while "
                f"{request.num_in_flight_tokens} tokens of a later step were "
                "already scheduled from the positions before this commit. "
                "Only a fused_sample verify commits past its drafts, and "
                "TTPlatform serves it with synchronous scheduling only"
            )
        if num_drafts:
            return
        if request.num_computed_tokens > 0:
            request.num_computed_tokens += num_accepted
        if request.num_output_placeholders > 0:
            request.num_output_placeholders += num_accepted

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        """Commit one block and reconcile its full physical reservation."""
        if not self._is_block_output_model:
            self._count_commit_past_scheduled_drafts(request, new_token_ids, is_stale)
            return super()._update_request_with_output(
                request, new_token_ids, is_stale=is_stale
            )
        # Adaptive: a request that decoded batched (or over-frontier) this step
        # committed a single baseline token (no block was reserved) -> plain
        # reconciliation. The decision is read from THIS step's SchedulerOutput
        # (set in _update_after_schedule), which the engine core pairs with this
        # output even when async scheduling has already run a later schedule().
        if self._is_adaptive_block:
            block_step = self._committing_block_step_decisions.get(request.request_id)
            if block_step is None:
                raise RuntimeError(
                    "adaptive block-output request committed output without a "
                    f"scheduling decision: req_id={request.request_id!r}"
                )
            if not block_step:
                # The scheduler stamped this step width 1, so the model owes
                # exactly one token. Check before delegating: super() appends
                # whatever it is handed, so a model that returned a BLOCK here
                # would commit extra tokens against a single reserved
                # placeholder and the mismatch would surface later as a
                # placeholder leak or a corrupted continuation, far from its
                # cause. This is the mirror of the block-width check below.
                if len(new_token_ids) != self.num_sampled_tokens_per_step:
                    raise ValueError(
                        "Model output width violates the scheduled baseline "
                        f"width: got {len(new_token_ids)}, expected "
                        f"{self.num_sampled_tokens_per_step} "
                        f"(req_id={request.request_id!r}); the scheduler "
                        "stamped this step width 1 (batched, or a prompt over "
                        "the spec frontier), so the model must return a single "
                        "baseline token -- the scheduler and model block gates "
                        "disagree"
                    )
                return super()._update_request_with_output(
                    request, new_token_ids, is_stale=is_stale
                )
        if is_stale:
            # The request was preempted after this block was submitted, and
            # preemption zeroed the K placeholders the block would reconcile.
            # Block requests are never reset-preempted (reset_prefix_cache
            # raises while one runs) and solo spec steps are not KV-preempted,
            # so this must not happen -- fail loudly rather than commit a block
            # against a reservation that no longer exists.
            raise RuntimeError(
                "A stale async output reached block serving for a block step; "
                "block-output frames cannot be discarded (reset/preempt of a "
                "running block request is unsupported)"
            )
        if len(new_token_ids) != self._output_tokens_per_step:
            raise ValueError(
                "Model output width violates output_tokens_per_step: "
                f"{len(new_token_ids)} != {self._output_tokens_per_step} "
                f"(req_id={request.request_id!r}); the scheduler reserved a "
                "block but the model returned a different width -- the "
                "scheduler and model block gates disagree"
            )

        # Scheduler appends token-by-token and trims at EOS, stop tokens,
        # max_tokens, or max_model_len. It resolves one placeholder per returned
        # token, counted before the trim, so all K placeholders of the physical
        # reservation are consumed even when the client-visible block is trimmed.
        if request.num_output_placeholders < self._output_tokens_per_step:
            raise RuntimeError(
                "Output placeholders underflowed after block-output reconciliation"
            )
        return super()._update_request_with_output(request, new_token_ids)
