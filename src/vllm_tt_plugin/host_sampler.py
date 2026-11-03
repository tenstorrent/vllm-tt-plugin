# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Native CPU sampling with bounded, speculative noise for seeded requests."""

import os
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import torch
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p,
    random_sample,
    sample_with_exponential_noise,
)
from vllm.v1.sample.sampler import Sampler


@dataclass
class _PrefetchedNoise:
    state: torch.Tensor
    vocab: int
    size_bytes: int
    future: Future


# Candidates read before the full-vocabulary top-p sort. A row is sampled from
# them only when they provably contain its whole top-p set.
TOP_P_CANDIDATES = 1024


def _sample_top_p_candidates(logits, k, p):
    """Sample top-p rows from the largest logits without sorting the vocabulary.

    The top-p set of a row is the shortest descending prefix whose exclusive
    cumulative probability stays below ``p``. When the first candidates already
    reach ``p`` (or a top-k limit fits inside them), that prefix lies inside the
    candidates and the distribution equals ``apply_top_k_top_p`` up to ties at
    the boundary. Rows that do not qualify fall back to the full sort.
    """
    batch, vocab = logits.shape
    count = min(TOP_P_CANDIDATES, vocab)
    if count == vocab:
        return None
    values, indices = logits.topk(count, dim=-1)
    values = values.float()
    if k is None:
        limited = torch.zeros(batch, dtype=torch.bool)
    else:
        k = k.to(torch.long)
        limited = k < vocab
        if (limited & (k > count)).any():
            return None
        outside = torch.arange(count) >= k.unsqueeze(1)
        values.masked_fill_(limited.unsqueeze(1) & outside, -float("inf"))
    # Top-k rows renormalize over their candidates; others over the vocabulary.
    norm = torch.where(
        limited, values.logsumexp(dim=-1), logits.float().logsumexp(dim=-1)
    )
    probs = (values - norm.unsqueeze(1)).exp()
    covered = probs.cumsum(dim=-1)
    # p = 1 keeps every nonzero probability, which candidates cannot bound.
    fits = (covered[:, -1] >= p) & (p < 1)
    exact = (limited | fits) & torch.isfinite(norm)
    keep = (covered - probs) < p.unsqueeze(1)
    keep[:, 0] = True
    probs.masked_fill_(~keep, 0.0)
    tokens = torch.zeros(batch, dtype=torch.long)
    if exact.any():
        choice = torch.multinomial(probs[exact], num_samples=1)
        tokens[exact] = indices[exact].gather(1, choice).squeeze(1)
    return tokens, exact


class TTTopKTopPSampler(TopKTopPSampler):
    def __init__(self, logprobs_mode="raw_logprobs", use_fp64_gumbel=False):
        super().__init__(logprobs_mode, use_fp64_gumbel)
        self.forward = self.forward_native
        self._sampling_pool = None
        self._noise_cache = OrderedDict()
        self._cached_bytes = 0
        self._max_prefetch_bytes = 64 * 1024 * 1024
        self._sampling_lock = Lock()

    def _prepare_noise(self, state, vocab):
        # Workers only advance private clones. Preparing or discarding noise
        # must not consume a live request's RNG, including after cancellation.
        clone = torch.Generator(device="cpu").set_state(state)
        dtype = torch.float64 if self.use_fp64_gumbel else torch.float32

        def generate():
            noise = torch.empty(vocab, dtype=dtype)
            noise.exponential_(generator=clone)
            return noise, clone.get_state()

        size = vocab * (8 if self.use_fp64_gumbel else 4)
        return _PrefetchedNoise(
            state, vocab, size, self._sampling_pool.submit(generate)
        )

    def forward_native(self, logits, generators, k, p):
        batch, vocab = logits.shape
        if (
            logits.device.type == "cpu"
            and batch > 0
            # PyTorch multinomial supports at most 2**24 categories.
            and 4096 <= vocab <= 2**24
            and not generators
            and not self.use_fp64_gumbel
        ):
            return self._sample_unseeded_cpu(logits, k, p)
        if (
            logits.device.type != "cpu"
            or batch == 0
            or vocab < 4096
            or set(generators) != set(range(batch))
            or len({id(gen) for gen in generators.values()}) != batch
            or any(gen.device.type != "cpu" for gen in generators.values())
        ):
            return super().forward_native(logits, generators, k, p)

        # A request generator may move to a different batch row. Cache by the
        # generator itself, and verify its complete state before reusing noise.
        with self._sampling_lock:
            return self._sample_seeded(logits, generators, k, p)

    def _sample_unseeded_cpu(self, logits, k, p):
        if self.logprobs_mode == "raw_logprobs" and p is not None:
            fast = _sample_top_p_candidates(logits, k, p)
            if fast is not None:
                tokens, exact = fast
                if not exact.all():
                    rest = ~exact
                    tokens[rest] = self._sample_unseeded_full(
                        logits[rest], None if k is None else k[rest], p[rest]
                    )[0]
                return tokens, None
        return self._sample_unseeded_full(logits, k, p)

    def _sample_unseeded_full(self, logits, k, p):
        values = apply_top_k_top_p(logits, k, p)
        scores = None
        if self.logprobs_mode == "processed_logits":
            scores = values
        elif self.logprobs_mode == "processed_logprobs":
            scores = values.log_softmax(dim=-1, dtype=torch.float32)
        probs = values.softmax(dim=-1, dtype=torch.float32)
        totals = probs.sum(dim=-1)
        if not (
            torch.isfinite(totals).all() and (totals > 0).all() and (probs >= 0).all()
        ):
            # Preserve native handling without consuming RNG or filtering twice.
            return random_sample(probs, {}, False), scores
        # Two replacement draws avoid PyTorch's single-draw exponential race
        # over every vocabulary entry. Keeping the first draw preserves the
        # categorical marginal. Unseeded global RNG advancement changes; every
        # batch with request generators retains the existing path above.
        tokens = torch.multinomial(probs, num_samples=2, replacement=True)
        return tokens[:, 0].contiguous(), scores

    def _sample_seeded(self, logits, generators, k, p):
        batch, vocab = logits.shape
        if self._sampling_pool is None:
            available = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 1)
            )
            self._sampling_pool = ThreadPoolExecutor(
                max_workers=min(32, available), thread_name_prefix="tt-sampling"
            )

        prepared = []
        for row in range(batch):
            generator = generators[row]
            state = generator.get_state()
            entry = self._noise_cache.pop(generator, None)
            if entry is not None:
                self._cached_bytes -= entry.size_bytes
            if (
                entry is None
                or entry.vocab != vocab
                or not torch.equal(entry.state, state)
            ):
                if entry is not None:
                    entry.future.cancel()
                entry = self._prepare_noise(state, vocab)
            prepared.append(entry)

        # Resolve generation before submitting sampling tasks to the same pool;
        # a worker must never wait on work queued behind other waiting workers.
        noise_and_states = [entry.future.result() for entry in prepared]

        def sample_row(row):
            values = apply_top_k_top_p(
                logits[row : row + 1],
                k[row : row + 1] if k is not None else None,
                p[row : row + 1] if p is not None else None,
            )
            scores = None
            if self.logprobs_mode == "processed_logits":
                scores = values
            elif self.logprobs_mode == "processed_logprobs":
                scores = values.log_softmax(dim=-1, dtype=torch.float32)
            probs = values.softmax(dim=-1, dtype=torch.float32)
            noise, next_state = noise_and_states[row]
            generators[row].set_state(next_state)
            token = sample_with_exponential_noise(probs, noise.unsqueeze(0))
            return token, scores

        # Each Torch operation may also use intra-op threads. Group rows into
        # a few tasks rather than dispatching one competing Torch job per row.
        if batch == 1:
            results = [sample_row(0)]
        else:
            available = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 1)
            )
            workers = max(1, min(4, available // torch.get_num_threads(), batch))
            chunk_size = (batch + workers - 1) // workers

            def sample_chunk(start):
                return [
                    sample_row(row)
                    for row in range(start, min(start + chunk_size, batch))
                ]

            results = [
                result
                for chunk in self._sampling_pool.map(
                    sample_chunk, range(0, batch, chunk_size)
                )
                for result in chunk
            ]

        # This work can overlap the next TT forward pass. The cache is bounded
        # by noise bytes, including pending futures; evicted clones are harmless.
        noise_bytes = vocab * (8 if self.use_fp64_gumbel else 4)
        prefetch_count = min(batch, self._max_prefetch_bytes // noise_bytes)
        for row in range(batch - prefetch_count, batch):
            generator = generators[row]
            entry = self._prepare_noise(noise_and_states[row][1], vocab)
            self._noise_cache[generator] = entry
            self._cached_bytes += entry.size_bytes
            while self._cached_bytes > self._max_prefetch_bytes:
                _, old = self._noise_cache.popitem(last=False)
                self._cached_bytes -= old.size_bytes
                old.future.cancel()

        tokens = torch.cat([result[0] for result in results])
        scores = (
            torch.cat([result[1] for result in results])
            if results[0][1] is not None
            else None
        )
        return tokens, scores


class TTHostSampler(Sampler):
    @staticmethod
    def apply_temperature(logits, temp, all_random):
        # CPU metadata needs no synchronization. Avoid touching the entire
        # vocabulary when division would leave every value unchanged.
        if (
            all_random
            and logits.device.type == "cpu"
            and temp.device.type == "cpu"
            and torch.all(temp == 1.0).item()
        ):
            return logits
        return Sampler.apply_temperature(logits, temp, all_random)


def create_host_sampler():
    sampler = TTHostSampler()
    sampler.topk_topp_sampler = TTTopKTopPSampler(
        sampler.logprobs_mode, sampler.use_fp64_gumbel
    )
    return sampler
