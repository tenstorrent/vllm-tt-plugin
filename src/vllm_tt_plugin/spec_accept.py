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
lossless for any draft the walk's own random draws did not choose: the drafted
id commits with probability ``p(d)`` and any other id ``x`` with
``(1 - p(d)) * p(x) / (1 - p(d)) = p(x)``. So a drafter that samples its
proposals stays correct without reporting probabilities; reporting them would
only raise the acceptance probability from ``p(d)`` to ``min(1, p(d) / q(d))``.

The target distribution is the one vLLM's ordinary sampler would use, built in
the ordinary sampler's order: penalties on the raw logits, then temperature,
then min_p, then top-k/top-p. ``accept_sampled_drafts`` applies the penalties
per candidate column against that column's own history and hands the rest to
``accept_speculated_tokens``. Upstream's rejection sampler skips min_p at the
drafted positions; this walk does not, because the ordinary sampler applies it
at every position. vLLM 0.26 refuses min_p on a speculating launch, so the
runner never passes it; the parameter is there for when that refusal goes.

The accept walk uses host PyTorch and the plugin's placeholder constant.
Top-k/top-p filtering and the penalties lazily import vLLM's PyTorch helpers.
This module reads no ``model_capabilities`` key and admits no configuration,
so synthetic distributions are sufficient to exercise the acceptance
arithmetic.
"""

from dataclasses import dataclass, field

import torch

from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID


@dataclass(frozen=True)
class AcceptResult:
    """One accept walk's committed tokens and their counts.

    ``committed_token_ids`` is ``[B, 1+K]`` int32. Row ``i`` holds
    ``accepted_counts[i]`` committed ids followed by ``PLACEHOLDER_TOKEN_ID``,
    so a consumer can either truncate at the count or stop at the first
    placeholder and get the same answer. ``accepted_counts`` is ``[B]`` int32.

    The two are validated against each other on construction. An off-by-one in
    a walk corrupts output tokens rather than raising anywhere, so the pairing
    is checked where it is produced and not where it is consumed.
    """

    committed_token_ids: torch.Tensor
    accepted_counts: torch.Tensor

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
class SpecSamplingInputs:
    """One verify step's sampling controls, row-aligned with its block.

    Captured when the step is built, because the asynchronous path walks
    acceptance on whichever thread resolves the readback, after the persistent
    batch may have moved on. ``generators`` maps a verify row to its request's
    own generator, which the walk advances; a row without one draws from the
    global stream.
    """

    vocab_size: int
    temperature: torch.Tensor
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    min_p: torch.Tensor | None = None
    generators: dict[int, torch.Generator] = field(default_factory=dict)
    penalties: SpecPenalties | None = None


def accept_sampled_drafts(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_valid_drafts: torch.Tensor,
    sampling: SpecSamplingInputs,
) -> AcceptResult:
    """Walk acceptance under every control the ordinary sampler applies.

    The penalties are applied here, per candidate column, because they read a
    history and column ``j``'s history is the committed output plus the first
    ``j`` drafts. The remaining controls are history-free and are applied by
    ``accept_speculated_tokens``.
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
    _check_sampling(sampling, int(target_logits.shape[0]))
    logits = target_logits.to(torch.float32)
    if sampling.penalties is not None:
        logits = apply_speculative_penalties(
            logits, draft_token_ids, num_valid_drafts, sampling.penalties
        )
    return accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=draft_token_ids,
        num_valid_drafts=num_valid_drafts,
        temperature=sampling.temperature,
        top_k=sampling.top_k,
        top_p=sampling.top_p,
        min_p=sampling.min_p,
        generators=sampling.generators,
    )


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
    min_p: torch.Tensor | None = None,
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
        min_p: ``[B]`` or None. Applied to random rows after temperature and
            before top-k/top-p, the ordinary sampler's order. 0 disables it.
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
        target_logits, draft_token_ids, num_valid_drafts, temperature, min_p
    )
    num_drafts = width - 1
    generators = generators or {}
    is_greedy = temperature <= 0.0

    # Two paths, because the greedy one needs no softmax and no randomness and
    # a batch is usually entirely one or the other. A mixed batch computes both
    # and selects per row, which is what upstream's two kernels do by
    # early-exiting on the row's own flag.
    accepted = torch.zeros((rows, num_drafts), dtype=torch.bool)
    candidates = torch.full((rows, width), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)

    if bool(is_greedy.any()):
        greedy_accepted, greedy_candidates = _greedy_walk(
            target_logits, draft_token_ids, num_drafts
        )
        accepted = torch.where(is_greedy.unsqueeze(1), greedy_accepted, accepted)
        candidates = torch.where(is_greedy.unsqueeze(1), greedy_candidates, candidates)

    if bool((~is_greedy).any()):
        random_accepted, random_candidates = _random_walk(
            target_logits,
            draft_token_ids,
            num_valid_drafts,
            temperature,
            top_k,
            top_p,
            min_p,
            draft_probs,
            generators,
            rows,
            num_drafts,
            vocab,
        )
        select = (~is_greedy).unsqueeze(1)
        accepted = torch.where(select, random_accepted, accepted)
        candidates = torch.where(select, random_candidates, candidates)

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
    min_p: torch.Tensor | None,
    draft_probs: torch.Tensor | None,
    generators: dict[int, torch.Generator],
    rows: int,
    num_drafts: int,
    vocab: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection-sample each position, correcting a rejection from the residual."""
    probs = _constrained_probs(target_logits, temperature, top_k, top_p, min_p)
    # Replaced inside the vocabulary before any gather, because a draft column
    # past its row's count holds padding, the runner pads with
    # PLACEHOLDER_TOKEN_ID, and torch.gather refuses a negative index rather
    # than ignoring it. A replaced column cannot reach the output: every column
    # past the row's count is masked away by the count, and the column at the
    # count is overwritten by the bonus. Columns inside the count are already
    # known to be real token ids, because _check_inputs validated them.
    draft_ids = draft_token_ids.to(torch.int64).clamp(0, probs.shape[-1] - 1)

    # The bonus comes first, before any other draw, so a seeded row's bonus
    # depends only on the generator state the call started from. Drawing it
    # after the accept uniforms would make it depend on the block width and on
    # how many drafts the row happened to receive, which is how a row carrying
    # no drafts at all would stop agreeing with an ordinary sampled step.
    # Upstream orders it the same way, sampling bonus tokens in
    # RejectionSampler.forward before it calls rejection_sample.
    bonus_probs = probs.gather(
        1, num_valid_drafts.to(torch.int64).view(rows, 1, 1).expand(rows, 1, vocab)
    )
    bonus = _sample_per_row(bonus_probs, generators, rows, vocab).squeeze(1)

    # The accept test, upstream's form: accept when p/q >= u. q is 0 only for a
    # drafter that proposed a token its own distribution rules out, which is a
    # drafter bug. The division is evaluated for those entries too and yields a
    # NaN or an infinity; the q > 0 term is what forces them to reject, rather
    # than the division being avoided.
    target_of_draft = _gather_ids(probs[:, :num_drafts], draft_ids)
    if draft_probs is None:
        draft_of_draft = torch.ones_like(target_of_draft)
    else:
        _check_draft_probs(draft_probs, rows, num_drafts, vocab)
        draft_of_draft = _gather_ids(draft_probs, draft_ids)
    uniform = _uniform(rows, num_drafts, num_valid_drafts, generators)
    accepted = (draft_of_draft > 0) & (
        target_of_draft.to(torch.float64) / draft_of_draft >= uniform
    )

    # The correction for a rejected position, drawn from the positive part of
    # p - q. Normalising it is unnecessary: the sampler below is scale free.
    if draft_probs is None:
        residual = probs[:, :num_drafts].clone()
        residual.scatter_(2, draft_ids.unsqueeze(2), 0.0)
    else:
        residual = (probs[:, :num_drafts] - draft_probs).clamp_min(0.0)
    # A row with no drafts has nothing to correct, so its seeded generator must
    # not advance here. Upstream skips the same draw for the same reason.
    recovered = _sample_per_row(
        residual, _drafting_rows(num_valid_drafts, generators), rows, vocab
    )

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
    min_p: torch.Tensor | None = None,
) -> torch.Tensor:
    """Target probabilities after temperature, min_p, top-k and top-p.

    The distribution the accept walk must preserve is the one the operator
    asked for, so the constraints belong before the accept test rather than
    after it. A mixed batch also computes probabilities for greedy rows using
    a safe temperature of 1. The caller discards those random-path results and
    selects the raw-logit argmax results for greedy rows.

    The caller's tensor is not modified: a caller that also wants logprobs
    needs the raw logits afterwards.
    """
    rows, width, vocab = target_logits.shape
    safe_temperature = torch.where(
        temperature <= 0.0, torch.ones_like(temperature), temperature
    )
    logits = target_logits.to(torch.float32) / safe_temperature.view(rows, 1, 1)

    if min_p is not None and bool((min_p > 0).any()):
        # vLLM's MinPLogitsProcessor arithmetic, on the tempered logits. A row
        # at 0 has a threshold of 0, which nothing falls below.
        probs = logits.softmax(dim=-1)
        threshold = probs.amax(dim=-1, keepdim=True) * min_p.to(torch.float32).view(
            rows, 1, 1
        )
        logits = logits.masked_fill(probs < threshold, float("-inf"))

    if top_k is not None or top_p is not None:
        # Deferred: vllm.v1.sample.ops resolves current_platform at import
        # time, which loads this plugin, so importing it at module scope from
        # inside the plugin is a cycle. The pytorch entry point is named
        # directly rather than the dispatching apply_top_k_top_p, which would
        # route to Triton on a machine that has it and to torch here, and
        # the two are not required to break ties identically.
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

        flat = logits.reshape(rows * width, vocab)
        flat = apply_top_k_top_p_pytorch(
            flat,
            None if top_k is None else _expand_to_columns(top_k, width),
            None if top_p is None else _expand_to_columns(top_p, width),
        )
        logits = flat.reshape(rows, width, vocab)

    return logits.softmax(dim=-1, dtype=torch.float32)


def _sample_per_row(
    weights: torch.Tensor,
    generators: dict[int, torch.Generator],
    rows: int,
    vocab: int,
) -> torch.Tensor:
    """Draw one id per position from unnormalised ``[B, S, V]`` weights.

    Uses the exponential race: with independent ``e_v ~ Exponential(1)``, the
    argmax of ``w_v / e_v`` is distributed as ``w`` normalised. One race per
    row rather than per position, matching upstream, so a seeded request draws
    the same number of values however many positions it carries.

    Divided rather than multiplied by a reciprocal, because that is the
    ordinary sampler's arithmetic: a seeded row with no drafts then draws
    exactly the token the ordinary sampler draws from the same distribution.
    """
    race = torch.empty((rows, vocab), dtype=torch.float32)
    race.exponential_()
    for row, generator in generators.items():
        race[row].exponential_(generator=generator)
    scored = weights / race.unsqueeze(1)
    return scored.argmax(dim=-1).to(torch.int32)


def _drafting_rows(
    num_valid_drafts: torch.Tensor, generators: dict[int, torch.Generator]
) -> dict[int, torch.Generator]:
    """The seeded rows that carry at least one draft.

    A row with no drafts takes no part in the accept test or the correction, so
    its generator must not advance for either. Its unseeded values are drawn
    anyway and discarded, which costs nothing and keeps the shapes uniform.
    """
    return {
        row: generator
        for row, generator in generators.items()
        if int(num_valid_drafts[row]) > 0
    }


def _uniform(
    rows: int,
    num_drafts: int,
    num_valid_drafts: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Uniform values in [0, 1) for the accept test, one per drafted position.

    A seeded row advances its generator by its own draft count and not by the
    block width, so a row's stream does not depend on how wide the block it
    happened to travel in was. Upstream draws per request for the same reason.

    float64 rather than float32 because float32 draws exact 0.0 often enough to
    matter, and a 0 would accept a draft the target assigns no probability.
    """
    uniform = torch.rand((rows, num_drafts), dtype=torch.float64)
    for row, generator in generators.items():
        valid = int(num_valid_drafts[row])
        if valid == 0:
            continue
        uniform[row, :valid] = torch.rand(
            (valid,), dtype=torch.float64, generator=generator
        )
    return uniform


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
    min_p: torch.Tensor | None = None,
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
        ("min_p", min_p),
    ):
        if tensor is not None and tensor.shape != (rows,):
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


def _check_sampling(sampling: SpecSamplingInputs, rows: int) -> None:
    """Every per-row control must have one entry per row of the block.

    Checked because a short tensor is not always refused further down: vLLM's
    penalty op scatters a one-row prompt into the first row only, and the other
    rows silently lose their repetition penalty.
    """
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
    "accept_sampled_drafts",
    "accept_speculated_tokens",
    "apply_speculative_penalties",
]
