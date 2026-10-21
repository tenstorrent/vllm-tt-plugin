# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The host accept walk for speculative decoding.

Given the target model's logits over a ``[B, 1+K]`` candidate block and the
drafts that produced it, decide how long a prefix of each row commits and which
token ids those are. This is the step that makes speculation lossless: the
distribution of the tokens it commits equals the distribution the target model
would have produced without any drafting.

The plugin owns this code because upstream's cannot run here.
``vllm.v1.sample.rejection_sampler.rejection_sample`` launches Triton kernels
unconditionally, and ``vllm.triton_utils`` substitutes a placeholder that lets
the module import but not execute. The semantics below are upstream's, followed
deliberately so a kernel or a test ported from upstream needs no adjustment:

- Greedy rows compare each draft against the target's argmax and stop at the
  first mismatch. The corrected token at the mismatching position commits, so a
  row that rejects its very first draft still commits one token.
- Random rows accept a draft with probability ``min(1, p/q)``, where ``p`` is
  the target probability of the drafted id and ``q`` the drafter's. On a
  rejection the position commits a token drawn from the positive part of
  ``p - q``, which is what keeps the committed distribution equal to ``p``.
- A row that accepts every draft commits one more token, the bonus, drawn from
  the target distribution at the column past its last draft. The uniform
  ``1+K`` block width is what makes that column already present.
- ``accepted_counts`` is a count in ``[1, 1+K]``, never an index and never 0.

A drafter that reports no probabilities, such as an n-gram proposer, is handled
as a point mass at the drafted id: ``q`` is 1, so the accept test reduces to
``p >= u``, and the residual is ``p`` with the drafted id removed. That is
lossless for any drafter, sampling ones included, whose choice does not read
the walk's own random draws: the drafted id ``d`` commits with probability
``p(d)`` and any other id ``x`` with ``(1 - p(d)) * p(x) / (1 - p(d)) = p(x)``.

The target distribution is the one vLLM's ordinary sampler would use, built in
the ordinary sampler's order: the grammar bitmask, which the runner applies to
the logits before the sampler sees them; then allowed_token_ids, bad_words and
min_tokens; then the penalties; then temperature and top-k/top-p. min_p is
absent because vLLM refuses a min_p above 1e-5 on a speculating launch and
builds no min_p processor for it. ``accept_sampled_drafts`` applies everything
up to the penalties per candidate column, against column ``j``'s own state: the
committed output plus the first ``j`` drafts.

One place differs from upstream's rejection sampler, towards the ordinary
sampler. Upstream masks the stop tokens at the bonus position whenever the
committed output is below min_tokens, even where the drafts before it reach
min_tokens; this walk masks column ``j`` only while the output plus ``j``
drafts is below it, which is where the ordinary sampler would mask that token.

Logprobs are the raw logprobs, after the grammar bitmask and before any other
control, of each committed token at its own column. That is what the plugin's
ordinary host sampler returns, because it runs vLLM's ``Sampler`` in its
default ``raw_logprobs`` mode on logits the grammar has already masked.

The accept walk uses host PyTorch and the plugin's placeholder constant. Every
vLLM helper is imported inside the function that uses it. This module reads no
``model_capabilities`` key and admits no configuration, so synthetic
distributions are sufficient to exercise the acceptance arithmetic.
"""

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm_tt_plugin.model_input import penalty_history, penalty_prompt_ids
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

if TYPE_CHECKING:
    from vllm.v1.outputs import LogprobsTensors

    from vllm_tt_plugin.model_input import TTModelInput


@dataclass(frozen=True)
class AcceptResult:
    """One accept walk's committed tokens, their counts and optional logprobs.

    ``committed_token_ids`` is ``[B, 1+K]`` int32. Row ``i`` holds
    ``accepted_counts[i]`` committed ids followed by ``PLACEHOLDER_TOKEN_ID``,
    so a consumer can either truncate at the count or stop at the first
    placeholder and get the same answer. ``accepted_counts`` is ``[B]`` int32.
    ``committed_token_ids`` and ``accepted_counts`` are validated against each
    other on construction. An off-by-one in a walk corrupts output tokens
    rather than raising anywhere, so the pairing is checked where it is
    produced and not where it is consumed.

    ``logprobs``, when the walk was asked for them, has one row per committed
    token, row 0's tokens first, and ``cu_num_generated_tokens`` gives where
    each row's tokens start: vLLM's layout for a step that commits a different
    number of tokens per request.
    """

    committed_token_ids: torch.Tensor
    accepted_counts: torch.Tensor
    logprobs: "LogprobsTensors | None" = None

    def __post_init__(self) -> None:
        if self.committed_token_ids.dim() != 2:
            raise ValueError(
                "AcceptResult.committed_token_ids must be 2-D [B, 1+K], got "
                f"{tuple(self.committed_token_ids.shape)}"
            )
        rows, width = self.committed_token_ids.shape
        if self.accepted_counts.shape != (rows,):
            raise ValueError(
                f"AcceptResult.accepted_counts must be [{rows}] to match "
                f"committed_token_ids {tuple(self.committed_token_ids.shape)}, "
                f"got {tuple(self.accepted_counts.shape)}"
            )
        bad = self.accepted_counts[
            (self.accepted_counts < 1) | (self.accepted_counts > width)
        ]
        if bad.numel():
            raise ValueError(
                f"AcceptResult.accepted_counts entries must lie in [1, {width}]; "
                f"a count of 0 is never valid, got {bad.tolist()}"
            )
        committed = _within_count(rows, width, self.accepted_counts)
        if bool((self.committed_token_ids == PLACEHOLDER_TOKEN_ID)[committed].any()):
            raise ValueError(
                "AcceptResult has PLACEHOLDER_TOKEN_ID inside a row's committed "
                "prefix, so the count and the ids disagree: counts "
                f"{self.accepted_counts.tolist()}"
            )
        if bool((self.committed_token_ids != PLACEHOLDER_TOKEN_ID)[~committed].any()):
            raise ValueError(
                "AcceptResult has a real token id past a row's committed "
                "prefix, so the count and the ids disagree: counts "
                f"{self.accepted_counts.tolist()}"
            )


@dataclass(frozen=True)
class SpecPenalties:
    """The history-dependent controls of one verify, per row.

    ``prompt_token_ids`` is ``[B, P]`` int64 padded with the vocabulary size,
    which is the padding vLLM's penalty op ignores. ``output_token_ids`` holds
    each row's committed output before this step's candidate block; a draft
    joins the history only for the columns after it.
    """

    prompt_token_ids: torch.Tensor
    output_token_ids: list[list[int]]
    presence: torch.Tensor
    frequency: torch.Tensor
    repetition: torch.Tensor


@dataclass(frozen=True)
class SpecTokenFilters:
    """The controls of one verify that rule tokens out, per row.

    Each one sets a token's logit to ``-inf`` at a column, and none of them
    reads a probability, so a greedy row and a random row are filtered alike.

    - ``grammar_bitmask`` is ``[B, 1+K, W]`` int32, xgrammar's packed layout:
      bit ``v % 32`` of word ``v // 32`` is 1 where column ``j`` allows token
      ``v``. Row ``i`` column ``j`` is the bitmask the scheduler filled after
      the request's committed output and its first ``j`` drafts. A row or
      column the scheduler sent nothing for is all ones, which is -1.
    - ``allowed_token_ids_mask`` is ``[B, V]`` bool, True where a token is NOT
      allowed, which is vLLM's convention. It applies to every column.
    - ``bad_words_token_ids`` maps a row to its banned token sequences. A
      sequence bans its last token at a column whose history ends with the
      sequence's other tokens, and ``output_token_ids`` is each row's
      committed output, which that history starts from.
    - ``min_tokens`` maps a row to ``(remaining, stop_token_ids)``:
      ``remaining`` is ``min_tokens`` minus the committed output length, and
      column ``j`` may not commit a stop token while ``j < remaining``.
    """

    grammar_bitmask: torch.Tensor | None = None
    allowed_token_ids_mask: torch.Tensor | None = None
    bad_words_token_ids: dict[int, list[list[int]]] = field(default_factory=dict)
    output_token_ids: list[list[int]] | None = None
    min_tokens: dict[int, tuple[int, tuple[int, ...]]] = field(default_factory=dict)


@dataclass(frozen=True)
class SpecSamplingInputs:
    """One verify step's sampling controls, row-aligned with its block.

    ``generators`` maps a verify row to its request's own generator, which the
    walk advances; a row without one draws from the global stream.
    ``num_logprobs`` is the batch's largest logprobs request, vLLM's
    ``max_num_logprobs``: ``None`` computes none, 0 the committed token's own,
    and ``n`` adds the top ``n``.
    """

    vocab_size: int
    temperature: torch.Tensor
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    generators: dict[int, torch.Generator] = field(default_factory=dict)
    penalties: SpecPenalties | None = None
    filters: SpecTokenFilters | None = None
    num_logprobs: int | None = None


def spec_sampling_inputs(
    model_input: "TTModelInput", vocab_size: int
) -> SpecSamplingInputs:
    """The sampling a ``logits`` verify walks under, for the step's live rows.

    Every field is read off what ``_prepare_model_inputs`` copied when it built
    the step, and copied again here, so a walk on the readback thread reads
    nothing the persistent batch can have changed since. Padding rows are left
    out: they own no request, and sampling them would cost a vocabulary's work
    per row for tokens nothing reads.

    The grammar bitmask is the one exception to build-time capture: the
    scheduler computes it while the verify runs, and ``sample_tokens``
    attaches it to this step's input before any walk can read it.
    """
    live = len(model_input.row_req_ids)
    params = model_input.tt_sampling_params

    def per_row(values: Any, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=dtype).reshape(-1)
        return (
            tensor.expand(live).clone()
            if tensor.numel() == 1
            else tensor[:live].clone()
        )

    # Deferred for the same import cycle as the top-k/top-p helper below.
    from vllm.v1.sample.sampler import _SAMPLING_EPS

    # vLLM's sampler samples a row below this temperature greedily, and the
    # persistent batch keeps the raw value, so the walk is told the same.
    temperature = per_row(params.temperature, torch.float32)
    temperature = temperature.masked_fill(temperature < _SAMPLING_EPS, 0.0)
    top_k = per_row(params.top_k, torch.int32)
    top_p = per_row(params.top_p, torch.float32)

    penalties = None
    if model_input.prompt_tokens is not None and model_input.output_tokens is not None:
        penalties = SpecPenalties(
            prompt_token_ids=penalty_prompt_ids(
                model_input.prompt_tokens[:live], vocab_size
            ),
            output_token_ids=penalty_history(model_input.output_tokens[:live]),
            presence=per_row(params.presence_penalty, torch.float32),
            frequency=per_row(params.frequency_penalty, torch.float32),
            repetition=per_row(params.repetition_penalty, torch.float32),
        )
    return SpecSamplingInputs(
        vocab_size=vocab_size,
        temperature=temperature,
        # Left out when no row restricts them, which skips a sort of the whole
        # vocabulary per candidate column and changes no probability.
        top_k=top_k if bool((top_k < vocab_size).any()) else None,
        top_p=top_p if bool((top_p < 1.0).any()) else None,
        # ``_build_host_generators`` keys only live rows. The generators are
        # the requests' own, so the walk advances them.
        generators=dict(model_input.generators_list[0]),
        penalties=penalties,
        filters=_spec_token_filters(model_input, live),
        num_logprobs=model_input.max_num_logprobs[0],
    )


def _spec_token_filters(
    model_input: "TTModelInput", live: int
) -> SpecTokenFilters | None:
    """The step's token filters for its live rows, or ``None`` when it has none."""
    bitmask = model_input.grammar_bitmask[0]
    if bitmask is not None and bitmask.dim() != 3:
        raise RuntimeError(
            "a logits verify was handed a grammar bitmask of shape "
            f"{tuple(bitmask.shape)}; a verify reads one packed row per "
            "candidate column, [rows, 1+K, W], which "
            "spec_grammar_bitmask_for_tt_batch builds"
        )
    allowed = model_input.allowed_token_ids_mask_list[0]
    bad_words = {
        row: words
        for row, words in model_input.bad_words_token_ids_list[0].items()
        if row < live
    }
    min_tokens = dict(model_input.spec_min_tokens or {})
    if bitmask is None and allowed is None and not bad_words and not min_tokens:
        return None
    if bad_words and model_input.output_tokens is None:
        raise RuntimeError(
            "a logits verify carries bad_words but no output history; "
            "_prepare_model_inputs builds output_tokens for every decode with "
            "bad words, because a word's prefix is matched against it"
        )
    return SpecTokenFilters(
        grammar_bitmask=None if bitmask is None else bitmask[:live],
        allowed_token_ids_mask=None if allowed is None else allowed[:live],
        bad_words_token_ids=bad_words,
        output_token_ids=(
            penalty_history(model_input.output_tokens[:live]) if bad_words else None
        ),
        min_tokens=min_tokens,
    )


def accept_sampled_drafts(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_valid_drafts: torch.Tensor,
    sampling: SpecSamplingInputs,
) -> AcceptResult:
    """Walk acceptance under every control the ordinary sampler applies.

    The token filters and the penalties are applied here, per candidate
    column, because they read a column's own state: column ``j``'s history is
    the committed output plus the first ``j`` drafts. The remaining controls
    read no history and are applied by ``accept_speculated_tokens``.
    """
    if target_logits.dim() != 3 or int(target_logits.shape[-1]) != sampling.vocab_size:
        raise ValueError(
            "accept_sampled_drafts target_logits must be [B, 1+K, "
            f"{sampling.vocab_size}], the whole vocabulary at every candidate "
            f"column, got {tuple(target_logits.shape)}"
        )
    # Before the penalties, which index by draft id and would fail on a bad
    # one inside vLLM's op rather than here, by name.
    _check_inputs(
        target_logits, draft_token_ids, num_valid_drafts, sampling.temperature
    )
    _check_sampling(sampling, *target_logits.shape)
    logits = target_logits.to(torch.float32)
    filters = sampling.filters
    if filters is not None and filters.grammar_bitmask is not None:
        logits = _apply_grammar_bitmask(logits, filters.grammar_bitmask)
    # What logprobs are taken from: after the grammar, before everything else.
    raw_logits = logits
    if filters is not None:
        logits = apply_speculative_token_filters(
            logits, draft_token_ids, num_valid_drafts, filters
        )
    if sampling.penalties is not None:
        logits = apply_speculative_penalties(
            logits, draft_token_ids, num_valid_drafts, sampling.penalties
        )
    result = accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=draft_token_ids,
        num_valid_drafts=num_valid_drafts,
        temperature=sampling.temperature,
        top_k=sampling.top_k,
        top_p=sampling.top_p,
        generators=sampling.generators,
    )
    if sampling.num_logprobs is None:
        return result
    return AcceptResult(
        committed_token_ids=result.committed_token_ids,
        accepted_counts=result.accepted_counts,
        logprobs=committed_logprobs(
            raw_logits,
            result.committed_token_ids,
            result.accepted_counts,
            sampling.num_logprobs,
        ),
    )


def apply_speculative_token_filters(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_valid_drafts: torch.Tensor,
    filters: SpecTokenFilters,
) -> torch.Tensor:
    """allowed_token_ids, bad_words and min_tokens, column by column.

    In the ordinary sampler's order, although each only writes ``-inf`` and
    the order cannot change what they rule out. The grammar bitmask is not
    applied here: it comes before the raw logprobs, so the caller applies it.

    Returns a new tensor; the caller's logits are not modified.
    """
    rows, width, vocab = target_logits.shape
    # Contiguous whatever the caller's strides: the contract does not require
    # contiguous logits, and apply_bad_words writes in place through a flat
    # view below, which a transposed layout cannot give.
    logits = target_logits.clone(memory_format=torch.contiguous_format)
    if filters.allowed_token_ids_mask is not None:
        logits.masked_fill_(filters.allowed_token_ids_mask.unsqueeze(1), float("-inf"))

    if filters.bad_words_token_ids:
        # Deferred for the same import cycle as the top-k/top-p helper below.
        # vLLM's own op, so a multi-token word matches exactly as it does in
        # the ordinary sampler, which hands the same op a single column.
        from vllm.v1.sample.ops.bad_words import apply_bad_words

        if filters.output_token_ids is None:
            raise ValueError(
                "accept_sampled_drafts filters.bad_words_token_ids needs "
                "filters.output_token_ids, the history a word's prefix is "
                "matched against"
            )
        flat_words: dict[int, list[list[int]]] = {}
        flat_histories: list[list[int]] = [[] for _ in range(rows * width)]
        for row, words in filters.bad_words_token_ids.items():
            committed = list(filters.output_token_ids[row])
            valid = int(num_valid_drafts[row])
            drafts = [int(token) for token in draft_token_ids[row, :valid].tolist()]
            for column in range(width):
                flat_words[row * width + column] = words
                flat_histories[row * width + column] = (
                    committed + drafts[: min(column, valid)]
                )
        apply_bad_words(logits.view(rows * width, vocab), flat_words, flat_histories)

    for row, (remaining, stop_token_ids) in filters.min_tokens.items():
        columns = min(max(int(remaining), 0), width)
        if columns and stop_token_ids:
            logits[row, :columns, list(stop_token_ids)] = float("-inf")
    return logits


def committed_logprobs(
    raw_logits: torch.Tensor,
    committed_token_ids: torch.Tensor,
    accepted_counts: torch.Tensor,
    num_logprobs: int,
) -> "LogprobsTensors":
    """Each committed token's logprob, rank and top ``num_logprobs``, at its column.

    Only the committed columns are read, so a rejected draft's column costs no
    log-softmax. vLLM's ``Sampler`` does the arithmetic, which is what the
    ordinary host sampler calls for one column.
    """
    # Deferred for the same import cycle as the top-k/top-p helper below.
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.sample.sampler import Sampler

    rows, width, _ = raw_logits.shape
    committed = _within_count(rows, width, accepted_counts)
    # Boolean indexing reads row-major, so row 0's committed columns come first.
    logprobs = Sampler.compute_logprobs(raw_logits[committed])
    gathered = Sampler.gather_logprobs(
        logprobs, num_logprobs, committed_token_ids[committed].to(torch.int64)
    )
    starts = [0, *accepted_counts.to(torch.int64).cumsum(0).tolist()]
    return LogprobsTensors(*gathered[:3], cu_num_generated_tokens=starts)


def apply_speculative_penalties(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_valid_drafts: torch.Tensor,
    penalties: SpecPenalties,
) -> torch.Tensor:
    """Presence, frequency and repetition penalties, column by column.

    Column ``j`` is scored after the row's committed output and its first
    ``min(j, num_valid_drafts)`` drafts, so the bonus column reads every valid
    draft and a padding column reads no padding. vLLM's own penalty op does
    the arithmetic, which keeps it identical to the ordinary sampler's.

    Returns a new tensor; the caller's logits are not modified.
    """
    # Deferred for the same import cycle as the top-k/top-p helper below.
    from vllm.v1.sample.ops.penalties import apply_all_penalties

    rows, width, vocab = target_logits.shape
    histories: list[list[int]] = []
    for row in range(rows):
        committed = list(penalties.output_token_ids[row])
        valid = int(num_valid_drafts[row])
        drafts = [int(token) for token in draft_token_ids[row, :valid].tolist()]
        for column in range(width):
            histories.append(committed + drafts[: min(column, valid)])

    def per_column(per_row: torch.Tensor) -> torch.Tensor:
        return per_row.repeat_interleave(width, dim=0)

    flat = apply_all_penalties(
        target_logits.reshape(rows * width, vocab).clone(),
        per_column(penalties.prompt_token_ids),
        per_column(penalties.presence),
        per_column(penalties.frequency),
        per_column(penalties.repetition),
        histories,
    )
    return flat.reshape(rows, width, vocab)


def accept_speculated_tokens(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_valid_drafts: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    draft_probs: torch.Tensor | None = None,
    generators: dict[int, torch.Generator] | None = None,
) -> AcceptResult:
    """Walk acceptance over one ``[B, 1+K]`` candidate block.

    Args:
        target_logits: ``[B, 1+K, V]``. Column ``j`` is the target's
            distribution for the token at candidate position ``j``, so column 0
            scores the already-committed input token's successor and column
            ``j`` scores draft ``j``.
        draft_token_ids: ``[B, K]``. Row ``i``'s pending drafts. Entries inside
            ``num_valid_drafts[i]`` must be real token ids, because an accepted
            draft commits as it arrived. Entries past it are padding and may
            hold any value, including ``PLACEHOLDER_TOKEN_ID``: they are
            processed, since the walk is vectorised over the whole block and
            replaces them with an in-vocabulary id to gather with, but they
            cannot commit.
        num_valid_drafts: ``[B]`` in ``[0, K]``. How many of a row's drafts are
            real. A row with 0 commits exactly its bonus token.
        temperature: ``[B]``. A value of 0 selects greedy acceptance for that
            row, matching what the HTTP sampling parameter means. Greedy rows
            read ``target_logits`` as given; only random rows are rescaled.
        top_k: ``[B]`` or None. Applied to random rows before the softmax.
        top_p: ``[B]`` or None. Applied to random rows before the softmax.
        draft_probs: ``[B, K, V]`` or None. The drafter's own distribution per
            drafted position. None treats each draft as a point mass at its
            id, which is lossless for any drafter whose choice does not read
            this walk's random draws.
        generators: row index to a seeded ``torch.Generator``. A row named here
            draws its randomness from that generator, so a seeded request is
            reproducible; the rest draw from the global stream.

    Returns:
        An :class:`AcceptResult`.
    """
    rows, width, vocab = _check_inputs(
        target_logits, draft_token_ids, num_valid_drafts, temperature
    )
    num_drafts = width - 1
    generators = generators or {}
    is_greedy = temperature <= 0.0

    # Two paths, because the greedy one needs no softmax and no randomness and
    # a batch is usually entirely one or the other. A mixed batch selects the
    # greedy result per row and walks only its random rows on the random path,
    # whose softmax, top-k/top-p pass and full-vocabulary races a greedy row
    # never reads.
    accepted = torch.zeros((rows, num_drafts), dtype=torch.bool)
    candidates = torch.full((rows, width), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)

    if bool(is_greedy.any()):
        greedy_accepted, greedy_candidates = _greedy_walk(
            target_logits, draft_token_ids, num_drafts
        )
        accepted = torch.where(is_greedy.unsqueeze(1), greedy_accepted, accepted)
        candidates = torch.where(is_greedy.unsqueeze(1), greedy_candidates, candidates)

    random_rows = (~is_greedy).nonzero().view(-1).tolist()
    if random_rows:

        def pick(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None or len(random_rows) == rows:
                return tensor
            return tensor[random_rows]

        random_accepted, random_candidates = _random_walk(
            pick(target_logits),
            pick(draft_token_ids),
            pick(num_valid_drafts),
            pick(temperature),
            pick(top_k),
            pick(top_p),
            pick(draft_probs),
            {
                index: generators[row]
                for index, row in enumerate(random_rows)
                if row in generators
            },
            len(random_rows),
            num_drafts,
            vocab,
        )
        accepted[random_rows] = random_accepted
        candidates[random_rows] = random_candidates

    accepted_counts = _counts_from_accepted(accepted, num_valid_drafts, num_drafts)
    committed = torch.where(
        _within_count(rows, width, accepted_counts),
        candidates,
        torch.full_like(candidates, PLACEHOLDER_TOKEN_ID),
    )
    return AcceptResult(committed_token_ids=committed, accepted_counts=accepted_counts)


def _greedy_walk(
    target_logits: torch.Tensor, draft_token_ids: torch.Tensor, num_drafts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept while each draft equals the target's argmax at its position.

    The candidate block is the argmax at every column, which is correct for
    three different columns at once: an accepted position, where the argmax is
    the draft; the rejecting position, where it is the correction that commits
    in the draft's place; and the bonus position past a row's last draft.
    """
    argmax_ids = target_logits.argmax(dim=-1).to(torch.int32)
    accepted = argmax_ids[:, :num_drafts] == draft_token_ids.to(torch.int32)
    return accepted, argmax_ids


def _random_walk(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_valid_drafts: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    draft_probs: torch.Tensor | None,
    generators: dict[int, torch.Generator],
    rows: int,
    num_drafts: int,
    vocab: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection-sample each position, correcting a rejection from the residual."""
    noise, pending = _start_walk_noise(
        rows, num_drafts, vocab, num_valid_drafts, generators
    )
    probs = _constrained_probs(
        target_logits,
        temperature,
        top_k,
        top_p,
        seeded_rows=frozenset(generators),
    )
    # Replaced inside the vocabulary before any gather, because a draft column
    # past its row's count holds padding, the runner pads with
    # PLACEHOLDER_TOKEN_ID, and torch.gather refuses a negative index rather
    # than ignoring it. A replaced column cannot reach the output: every column
    # past the row's count is masked away by the count, and the column at the
    # count is overwritten by the bonus. Columns inside the count are already
    # known to be real token ids, because _check_inputs validated them.
    draft_ids = draft_token_ids.to(torch.int64).clamp(0, probs.shape[-1] - 1)

    for future in pending:
        future.result()
    bonus_probs = probs.gather(
        1, num_valid_drafts.to(torch.int64).view(rows, 1, 1).expand(rows, 1, vocab)
    )
    bonus = _race_argmax(bonus_probs, noise.bonus).squeeze(1)

    # The accept test, upstream's form: accept when p/q >= u. q is 0 only for a
    # drafter that proposed a token its own distribution rules out, which is a
    # drafter bug. The division is evaluated for those entries too and yields a
    # NaN or an infinity; the q > 0 term is what forces them to reject, rather
    # than the division being avoided. The p > 0 term rejects a draft the
    # filters ruled out even on the uniform draw of exactly 0: a draft the
    # grammar forbids reaches the walk whenever the scheduler did not validate
    # it, and only this test keeps it out of the output.
    target_of_draft = _gather_ids(probs[:, :num_drafts], draft_ids)
    if draft_probs is None:
        draft_of_draft = torch.ones_like(target_of_draft)
    else:
        _check_draft_probs(draft_probs, rows, num_drafts, vocab)
        draft_of_draft = _gather_ids(draft_probs, draft_ids)
    accepted = (
        (draft_of_draft > 0)
        & (target_of_draft > 0)
        & (target_of_draft.to(torch.float64) / draft_of_draft >= noise.uniform)
    )

    # The correction for a rejected position, drawn from the positive part of
    # p - q. Normalising it is unnecessary: the sampler below is scale free.
    if draft_probs is None:
        residual = probs[:, :num_drafts].clone()
        residual.scatter_(2, draft_ids.unsqueeze(2), 0.0)
    else:
        residual = (probs[:, :num_drafts] - draft_probs).clamp_min(0.0)
    recovered = _race_argmax(residual, noise.recovered)

    candidates = torch.cat(
        [
            torch.where(accepted, draft_token_ids.to(torch.int32), recovered),
            torch.full((rows, 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32),
        ],
        dim=1,
    )
    # A fully accepted row's bonus column is its own num_valid_drafts, which is
    # the extra column only when the row carried all K drafts.
    candidates.scatter_(
        1, num_valid_drafts.to(torch.int64).unsqueeze(1), bonus.unsqueeze(1)
    )
    return accepted, candidates


def _constrained_probs(
    target_logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    seeded_rows: frozenset[int] = frozenset(),
) -> torch.Tensor:
    """Target probabilities after temperature, top-k and top-p.

    The distribution the accept walk must preserve is the one the operator
    asked for, so the constraints belong before the accept test rather than
    after it. Every row is a random row: ``accept_speculated_tokens`` keeps
    greedy rows off this path.

    Top-k and top-p follow vLLM's PyTorch implementation, which sorts the
    whole vocabulary of every candidate column; that sort is most of the
    walk's cost. Two cases avoid it:

    - Top-k without top-p uses vLLM's ``apply_top_k_only``, which keeps every
      logit at least the k-th largest, the same set the sort keeps.
    - Top-k with top-p, for a row outside ``seeded_rows`` whose top-k is
      below the vocabulary size and at most ``_MAX_UNSORTED_TOP_K``, uses
      ``_top_k_top_p_without_sort`` whatever the other rows carry. Its top-p
      cut can differ from the sort's where two logits tie at the cut or where
      a cumulative sum lands within rounding of ``1 - p``. A seeded row keeps
      the sort, so it commits exactly what the ordinary host sampler, which
      uses the same sort, commits. A row with top-p but no top-k also keeps
      the sort.

    The caller's tensor is not modified: a caller that also wants logprobs
    needs the raw logits afterwards.
    """
    rows, width, vocab = target_logits.shape
    logits = target_logits.to(torch.float32) / temperature.view(rows, 1, 1)

    if top_k is not None or top_p is not None:
        # Deferred: vllm.v1.sample.ops resolves current_platform at import
        # time, which loads this plugin, so importing it at module scope from
        # inside the plugin is a cycle. The pytorch entry points are named
        # directly rather than the dispatching apply_top_k_top_p, which would
        # route to Triton on a machine that has it and to torch here, and
        # the two are not required to break ties identically.
        from vllm.v1.sample.ops.topk_topp_sampler import (
            apply_top_k_only,
            apply_top_k_top_p_pytorch,
        )

        flat = logits.reshape(rows * width, vocab)
        k = None if top_k is None else _expand_to_columns(top_k, width)
        p = None if top_p is None else _expand_to_columns(top_p, width)
        if p is None:
            flat = apply_top_k_only(flat, k)
        else:
            # Per row, so a seeded row, or a row without a top-k (the input
            # batch stores the vocabulary size), puts only itself on the sort.
            unsorted = (
                []
                if top_k is None
                else [
                    row
                    for row, row_k in enumerate(top_k.tolist())
                    if row not in seeded_rows
                    and row_k < vocab
                    and row_k <= _MAX_UNSORTED_TOP_K
                ]
            )
            unsorted_set = set(unsorted)
            sorted_rows = [row for row in range(rows) if row not in unsorted_set]
            for subset, apply in (
                (unsorted, _top_k_top_p_without_sort),
                (sorted_rows, apply_top_k_top_p_pytorch),
            ):
                if not subset:
                    continue
                if len(subset) == rows:
                    flat = apply(flat, k, p)
                    continue
                columns = [
                    row * width + column for row in subset for column in range(width)
                ]
                flat[columns] = apply(
                    flat[columns], None if k is None else k[columns], p[columns]
                )
        logits = flat.reshape(rows, width, vocab)

    return logits.softmax(dim=-1, dtype=torch.float32)


# The largest top-k ``_top_k_top_p_without_sort`` serves. Above it, its
# ``topk`` approaches the cost of the sort it replaces.
_MAX_UNSORTED_TOP_K = 1024


def _top_k_top_p_without_sort(
    logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor
) -> torch.Tensor:
    """vLLM's top-k then top-p masks on ``[N, V]`` logits, from the k largest.

    Top-k keeps every logit at least the k-th largest, as vLLM does. vLLM's
    top-p then drops a token when the probability at or below it, summed in
    ascending order over the top-k distribution, is at most ``1 - p``; this
    keeps a token while the probability strictly above it, summed in
    descending order, is below ``p``. That is the same cut unless two logits
    tie at it or a sum lands within rounding of the boundary. The largest
    token always stays, and a top-p of 1 keeps the whole top-k, as in vLLM.
    Modifies ``logits`` in place.
    """
    values = logits.topk(int(k.max()), dim=-1).values
    kth = values.gather(1, (k.to(torch.int64) - 1).unsqueeze(1))
    probs = values.masked_fill(values < kth, float("-inf")).softmax(dim=-1)
    above = probs.cumsum(dim=-1) - probs
    # At a top-p of 1 the descending sum can round to 1 before a top-k token
    # whose probability is below float32 rounding of 1, which vLLM keeps.
    within_p = (above < p.unsqueeze(1)) | (p.unsqueeze(1) >= 1.0)
    kept = within_p.sum(dim=-1, keepdim=True).clamp(min=1)
    cut = torch.maximum(kth, values.gather(1, kept - 1))
    return logits.masked_fill_(logits < cut, float("-inf"))


def _race_argmax(weights: torch.Tensor, race: torch.Tensor) -> torch.Tensor:
    """Draw one id per position from unnormalised ``[B, S, V]`` weights.

    Uses the exponential race: with independent ``e_v ~ Exponential(1)``, the
    argmax of ``w_v / e_v`` is distributed as ``w`` normalised. ``race`` is one
    ``[B, V]`` race per row rather than per position, matching upstream, so a
    seeded request draws the same number of values however many positions it
    carries.

    The weights are divided by the race, as vLLM's ``random_sample`` does, so a
    seeded row with no drafts draws exactly the token the ordinary sampler
    draws from the same distribution.
    """
    return (weights / race.unsqueeze(1)).argmax(dim=-1).to(torch.int32)


@dataclass(frozen=True)
class _WalkNoise:
    """Every random value one walk reads.

    ``bonus`` and ``recovered`` are ``[B, V]`` exponential races, for the bonus
    and for the correction at a rejected position. ``uniform`` is ``[B, K]``
    float64, one accept-test value per drafted position. float64 rather than
    float32 because float32 draws exact 0.0 often enough to matter, and a 0
    would accept a draft the target assigns no probability.
    """

    bonus: torch.Tensor
    uniform: torch.Tensor
    recovered: torch.Tensor


_NOISE_POOL: ThreadPoolExecutor | None = None
_NOISE_POOL_LOCK = threading.Lock()
# Below this vocabulary a row's draws cost less than handing them to a worker,
# so they run on the calling thread. The ordinary host sampler, TTTopKTopPSampler,
# uses the same bound for its worker-thread noise.
_PARALLEL_NOISE_MIN_VOCAB = 4096


def _noise_workers() -> int:
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    return max(1, min(16, available))


def _noise_pool() -> ThreadPoolExecutor:
    global _NOISE_POOL
    with _NOISE_POOL_LOCK:
        if _NOISE_POOL is None:
            _NOISE_POOL = ThreadPoolExecutor(
                max_workers=_noise_workers(), thread_name_prefix="tt-spec-noise"
            )
        return _NOISE_POOL


def _start_walk_noise(
    rows: int,
    num_drafts: int,
    vocab: int,
    num_valid_drafts: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> tuple[_WalkNoise, list[Future]]:
    """Start drawing every random value of one walk on worker threads.

    No value depends on the logits, so the walk computes the target
    probabilities while these are drawn. The full-vocabulary races are the
    largest cost of the walk, and torch draws exponential values on a single
    thread per call, so the rows draw on several workers. The caller waits on
    the returned futures before it reads the noise.

    A seeded row is one task that draws from its generator in the order a
    serial walk reads, so its stream is the one that walk would leave:

    - The bonus race first, so a seeded row's bonus depends only on the
      generator state the call started from. Drawing it after the accept
      uniforms would make it depend on how many drafts the row carried, which
      is how a row carrying no drafts would stop agreeing with an ordinary
      sampled step. Upstream samples bonus tokens before it calls
      rejection_sample.
    - Then one uniform per draft the row carries, not per column of the block,
      so a row's stream does not depend on the width of the block it travelled
      in. Upstream draws per request for the same reason.
    - Then the correction race. A row with no drafts takes no part in the
      accept test or the correction, so it draws neither, and its generator
      stays where an ordinary sampled step leaves it. Upstream skips both.
      Its correction race stays at 1 and is never read.

    The unseeded rows draw in chunks, one per worker, each from a private
    generator seeded from the global stream on the calling thread, so no two
    threads share a generator and a walk still repeats under
    ``torch.manual_seed``. Below ``_PARALLEL_NOISE_MIN_VOCAB`` everything is
    drawn on the calling thread, as one chunk.

    An unseeded row draws one race and uses it for both the bonus and the
    correction. A row commits either its bonus, when it accepts every draft,
    or its correction, at its first rejection, never both, and the accept
    test reads only the uniforms, so the committed token reads that race
    once. A seeded row cannot share: its stream must stay the one a serial
    walk leaves.
    """
    noise = _WalkNoise(
        bonus=torch.empty((rows, vocab), dtype=torch.float32),
        uniform=torch.ones((rows, num_drafts), dtype=torch.float64),
        recovered=torch.ones((rows, vocab), dtype=torch.float32),
    )
    counts = num_valid_drafts.tolist()

    def draw_seeded(row: int) -> None:
        generator = generators[row]
        noise.bonus[row].exponential_(generator=generator)
        if counts[row]:
            noise.uniform[row, : counts[row]] = torch.rand(
                (counts[row],), dtype=torch.float64, generator=generator
            )
            noise.recovered[row].exponential_(generator=generator)

    def draw_unseeded(chunk: list[int], seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        shape = (len(chunk), vocab)
        race = torch.empty(shape).exponential_(generator=generator)
        noise.bonus[chunk] = race
        noise.uniform[chunk] = torch.rand(
            (len(chunk), num_drafts), dtype=torch.float64, generator=generator
        )
        noise.recovered[chunk] = race

    unseeded = [row for row in range(rows) if row not in generators]
    parallel = vocab >= _PARALLEL_NOISE_MIN_VOCAB
    pool = _noise_pool() if parallel else None
    pieces = min(len(unseeded), _noise_workers()) if parallel else 1
    chunks = [unseeded[i::pieces] for i in range(pieces)] if unseeded else []
    seeds = torch.randint(0, 2**63 - 1, (len(chunks),)).tolist() if chunks else []

    tasks = [lambda row=row: draw_seeded(row) for row in generators] + [
        lambda chunk=chunk, seed=seed: draw_unseeded(chunk, seed)
        for chunk, seed in zip(chunks, seeds)
    ]
    if pool is None:
        for task in tasks:
            task()
        return noise, []
    return noise, [pool.submit(task) for task in tasks]


def _counts_from_accepted(
    accepted: torch.Tensor, num_valid_drafts: torch.Tensor, num_drafts: int
) -> torch.Tensor:
    """Committed token count per row, in ``[1, 1+K]``.

    A row that rejects at position ``j`` commits ``j+1`` tokens, because the
    correction at ``j`` commits in the rejected draft's place. A row that
    accepts every one of its ``num_valid_drafts`` drafts commits one more than
    that, the bonus. A row with no drafts commits 1.
    """
    rows = accepted.shape[0]
    valid = torch.arange(num_drafts).unsqueeze(0) < num_valid_drafts.unsqueeze(1)
    rejected = valid & ~accepted
    # argmax on a bool row of all False returns 0, so the "any rejection at
    # all" test has to gate it rather than reading the index directly.
    first_rejection = rejected.to(torch.int8).argmax(dim=1)
    counts = torch.where(
        rejected.any(dim=1), first_rejection + 1, num_valid_drafts.to(torch.int64) + 1
    )
    return counts.to(torch.int32).reshape(rows)


def _within_count(rows: int, width: int, accepted_counts: torch.Tensor) -> torch.Tensor:
    """``[B, width]`` mask, true where a column is inside its row's count."""
    columns = torch.arange(width).unsqueeze(0).expand(rows, width)
    return columns < accepted_counts.to(torch.int64).unsqueeze(1)


def _apply_grammar_bitmask(logits: torch.Tensor, bitmask: torch.Tensor) -> torch.Tensor:
    """``logits`` with every token a row's column bitmask forbids set to ``-inf``.

    Only rows with a constraint are unpacked: a row that is all ones, which is
    every row without a structured request, would cost ``1+K`` columns of 32
    integers per vocabulary word for nothing. Returns ``logits`` itself when
    no row is constrained, and a new tensor otherwise.
    """
    constrained = (bitmask != -1).flatten(1).any(dim=1).nonzero().view(-1)
    if not int(constrained.numel()):
        return logits
    masked = logits.clone()
    masked[constrained] = masked[constrained].masked_fill(
        _grammar_disallowed(bitmask[constrained], int(logits.shape[-1])),
        float("-inf"),
    )
    return masked


def _grammar_disallowed(bitmask: torch.Tensor, vocab: int) -> torch.Tensor:
    """``[..., V]`` bool from a packed ``[..., W]`` bitmask, True where disallowed.

    A bitmask narrower than the vocabulary covers the tokenizer's ids only, and
    a token no grammar can name is disallowed, so the tail is padded with True.
    """
    bits = torch.arange(32, dtype=torch.int32)
    allowed = ((bitmask.to(torch.int32).unsqueeze(-1) >> bits) & 1).bool()
    allowed = allowed.reshape(*bitmask.shape[:-1], -1)
    if int(allowed.shape[-1]) < vocab:
        pad = torch.zeros(
            (*allowed.shape[:-1], vocab - int(allowed.shape[-1])), dtype=torch.bool
        )
        allowed = torch.cat([allowed, pad], dim=-1)
    return ~allowed[..., :vocab]


def _gather_ids(probs: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``[B, S]`` probability that ``probs[b, s]`` assigns to ``ids[b, s]``."""
    return probs.gather(2, ids.unsqueeze(2)).squeeze(2)


def _expand_to_columns(per_row: torch.Tensor, width: int) -> torch.Tensor:
    """Repeat a per-row sampling parameter for every candidate column."""
    return per_row.unsqueeze(1).expand(per_row.shape[0], width).reshape(-1)


def _check_inputs(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_valid_drafts: torch.Tensor,
    temperature: torch.Tensor,
) -> tuple[int, int, int]:
    if target_logits.dim() != 3:
        raise ValueError(
            "accept_speculated_tokens target_logits must be 3-D [B, 1+K, V], "
            f"got {tuple(target_logits.shape)}"
        )
    rows, width, vocab = target_logits.shape
    if width < 1:
        raise ValueError(
            "accept_speculated_tokens target_logits must carry at least the "
            f"bonus column, got width {width}"
        )
    num_drafts = width - 1
    if draft_token_ids.shape != (rows, num_drafts):
        raise ValueError(
            f"accept_speculated_tokens draft_token_ids must be [{rows}, "
            f"{num_drafts}] to match target_logits "
            f"{tuple(target_logits.shape)}, got {tuple(draft_token_ids.shape)}"
        )
    for name, tensor in (
        ("num_valid_drafts", num_valid_drafts),
        ("temperature", temperature),
    ):
        if tensor.shape != (rows,):
            raise ValueError(
                f"accept_speculated_tokens {name} must be [{rows}], got "
                f"{tuple(tensor.shape)}"
            )
    out_of_range = num_valid_drafts[
        (num_valid_drafts < 0) | (num_valid_drafts > num_drafts)
    ]
    if out_of_range.numel():
        raise ValueError(
            "accept_speculated_tokens num_valid_drafts entries must lie in "
            f"[0, {num_drafts}], got {out_of_range.tolist()}"
        )
    _check_draft_ids(draft_token_ids, num_valid_drafts, num_drafts, vocab)
    return rows, width, vocab


def _check_draft_ids(
    draft_token_ids: torch.Tensor,
    num_valid_drafts: torch.Tensor,
    num_drafts: int,
    vocab: int,
) -> None:
    """Every draft inside a row's count must be a real token id.

    Checked because an accepted draft commits as it arrived, so an id outside
    the vocabulary would reach the detokenizer as an output token rather than
    failing anywhere. Only the valid prefix is checked: the columns past it are
    padding, whose value the caller is free to choose.
    """
    if num_drafts == 0:
        return
    valid = torch.arange(num_drafts).unsqueeze(0) < num_valid_drafts.unsqueeze(1)
    bad = valid & ((draft_token_ids < 0) | (draft_token_ids >= vocab))
    if not bool(bad.any()):
        return
    offenders = [
        (int(row), int(column), int(draft_token_ids[row, column]))
        for row, column in bad.nonzero().tolist()
    ]
    raise ValueError(
        "accept_speculated_tokens draft_token_ids entries inside a row's "
        f"num_valid_drafts must lie in [0, {vocab - 1}]; an accepted draft "
        "commits as it arrived, so an id outside the vocabulary would be "
        f"emitted as an output token. Offending (row, position, id): {offenders}"
    )


def _check_sampling(
    sampling: SpecSamplingInputs, rows: int, width: int, vocab: int
) -> None:
    """Every per-row control must have one entry per row of the block.

    Checked because a short tensor is not always refused further down: vLLM's
    penalty op scatters a one-row prompt into the first row only, and the other
    rows silently lose their repetition penalty, and a ``[1, V]`` allowlist
    broadcasts one request's allowlist over every row.
    """
    filters = sampling.filters
    if filters is not None:
        mask = filters.allowed_token_ids_mask
        if mask is not None and tuple(mask.shape) != (rows, vocab):
            raise ValueError(
                f"accept_sampled_drafts filters.allowed_token_ids_mask must be "
                f"[{rows}, {vocab}], got {tuple(mask.shape)}"
            )
        bitmask = filters.grammar_bitmask
        if bitmask is not None and (
            bitmask.dim() != 3 or tuple(bitmask.shape[:2]) != (rows, width)
        ):
            raise ValueError(
                f"accept_sampled_drafts filters.grammar_bitmask must be "
                f"[{rows}, {width}, W], one packed row per candidate column, "
                f"got {tuple(bitmask.shape)}"
            )
        if filters.output_token_ids is not None and (
            len(filters.output_token_ids) != rows
        ):
            raise ValueError(
                f"accept_sampled_drafts filters.output_token_ids must hold "
                f"{rows} rows, got {len(filters.output_token_ids)}"
            )
        outside = [
            row
            for row in (*filters.bad_words_token_ids, *filters.min_tokens)
            if not 0 <= int(row) < rows
        ]
        if outside:
            raise ValueError(
                f"accept_sampled_drafts filters name rows {sorted(set(outside))} "
                f"outside the block's {rows}"
            )
    per_row = {"top_k": sampling.top_k, "top_p": sampling.top_p}
    penalties = sampling.penalties
    if penalties is not None:
        per_row.update(
            presence=penalties.presence,
            frequency=penalties.frequency,
            repetition=penalties.repetition,
        )
        if penalties.prompt_token_ids.dim() != 2 or (
            int(penalties.prompt_token_ids.shape[0]) != rows
        ):
            raise ValueError(
                f"accept_sampled_drafts penalties.prompt_token_ids must be "
                f"[{rows}, P], got {tuple(penalties.prompt_token_ids.shape)}"
            )
        if len(penalties.output_token_ids) != rows:
            raise ValueError(
                f"accept_sampled_drafts penalties.output_token_ids must hold "
                f"{rows} rows, got {len(penalties.output_token_ids)}"
            )
    for name, tensor in per_row.items():
        if tensor is not None and tensor.shape != (rows,):
            raise ValueError(
                f"accept_sampled_drafts {name} must be [{rows}], got "
                f"{tuple(tensor.shape)}"
            )


def _check_draft_probs(
    draft_probs: torch.Tensor, rows: int, num_drafts: int, vocab: int
) -> None:
    if draft_probs.shape != (rows, num_drafts, vocab):
        raise ValueError(
            f"accept_speculated_tokens draft_probs must be [{rows}, "
            f"{num_drafts}, {vocab}], got {tuple(draft_probs.shape)}"
        )


__all__ = [
    "AcceptResult",
    "SpecPenalties",
    "SpecSamplingInputs",
    "SpecTokenFilters",
    "accept_sampled_drafts",
    "accept_speculated_tokens",
    "apply_speculative_penalties",
    "apply_speculative_token_filters",
    "committed_logprobs",
]
