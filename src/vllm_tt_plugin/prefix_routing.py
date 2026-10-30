# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefix-cache-aware routing of new requests, for lanes and for DP engines.

Each lane of ``TTLaneCoordinator`` and each engine of standard multi-process
DP keeps its own KV cache manager, so a prompt whose prefix an earlier request
computed only hits the prefix cache on the scheduler holding it; least-loaded
routing sends the follower wherever the load is lowest and prefills the prefix
again. :func:`pick_prefix_cache_target` is the one rule both layouts apply.
``TTLaneCoordinator._pick_lane`` feeds it each lane's own longest-hit lookup.
Standard DP cannot: vLLM's load balancer
(``DPLBAsyncMPClient.get_core_engine_for_request``) runs in the API server,
which reads no engine's cache, so :class:`PrefixCacheMirror` records which
engine each prompt block was sent to and :func:`route_dp_request` pins the
request to the holder through ``EngineCoreRequest.data_parallel_rank``, the
one input that balancer honors ahead of its own choice.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any

from vllm_tt_plugin.config import get_tt_dp_prefix_cache_routing

# Prompt blocks the mirror keeps before it forgets the oldest: at 32-token
# blocks, two million prompt tokens, past what the engines' caches hold.
DEFAULT_MAX_MIRRORED_BLOCKS = 1 << 16

_UNSET = object()


def pick_prefix_cache_target(
    least: int,
    scores: Sequence[float],
    hits: Sequence[int],
    is_full: Callable[[int], bool],
    queued_prefill_tokens: Callable[[int], int],
) -> int:
    """The target for a new request: the prefix holder, or the least-loaded.

    ``least`` is the target least-loaded routing picked and ``scores`` its
    load score per target, lower being less loaded. ``hits[i]`` is how many of
    the prompt's tokens target ``i`` already holds. The target holding the
    most of the prompt, among equals the less loaded and then the lower index,
    is chosen over ``least`` when it holds more than ``least`` does, unless
    (a) it has no free slot once its queued requests are admitted while
    ``least`` has one, because an open-ended wait for a slot costs more than a
    bounded prefill, or (b) the prefill queued ahead on it exceeds ``least``'s
    queue by more than the extra prefill the hit saves over ``least``'s own
    hit. Rule (b) is what keeps a hot prefix from piling every request onto
    one target: followers of a long prefix queue almost nothing, so they all
    follow it; followers of a short one spill as soon as the queue ahead
    outweighs the saving. Equal hits save nothing and resolve on load.
    ``is_full`` and ``queued_prefill_tokens`` are asked only when a holder is
    in play.
    """
    cached = max(range(len(hits)), key=lambda i: (hits[i], -scores[i], -i))
    extra_hit = hits[cached] - hits[least]
    if extra_hit <= 0 or cached == least:
        return least
    if is_full(cached) and not is_full(least):
        return least
    queued_ahead = queued_prefill_tokens(cached) - queued_prefill_tokens(least)
    return cached if queued_ahead <= extra_hit else least


class PrefixCacheMirror:
    """Which DP engine each prompt block was sent to, by chained block hash.

    The API server's stand-in for the engines' prefix caches, which it cannot
    read: an engine that was sent a prompt computed its blocks, so a later
    prompt sharing the leading blocks hits there, until the engine evicts
    them. The mirror sees dispatches only. It is not told of an eviction, and
    :meth:`clear` is called on a prefix-cache reset. Blocks are hashed in a
    chain, as the KV cache manager hashes them, so a block's hash covers every
    block before it and a prompt's hit on an engine is the run of leading
    blocks that engine was sent. The newest ``max_blocks`` blocks are kept.
    """

    def __init__(
        self,
        block_size: int,
        num_engines: int,
        max_blocks: int = DEFAULT_MAX_MIRRORED_BLOCKS,
    ) -> None:
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise TypeError(
                f"PrefixCacheMirror needs an integer block size, got {block_size!r}"
            )
        if block_size < 1 or num_engines < 1 or max_blocks < 1:
            raise ValueError(
                "PrefixCacheMirror needs a positive block size, engine count and "
                f"capacity, got {block_size}, {num_engines}, {max_blocks}"
            )
        self.block_size = block_size
        self.num_engines = num_engines
        self._max_blocks = max_blocks
        # Chained block hash -> bitmask of the engines it was sent to, oldest
        # first; a recorded block moves to the end.
        self._engines_by_block: dict[int, int] = {}
        # Each in-flight request's engine, the prompt tokens it had to prefill
        # when routed and its prompt length, in dispatch order.
        self._dispatched: dict[str, tuple[int, int, int]] = {}

    def block_hashes(
        self, prompt_token_ids: Sequence[int], mm_identifiers: Iterable[str] = ()
    ) -> list[int]:
        """The chained hashes of the prompt's full blocks.

        Multimodal item identifiers seed the chain, so prompts carrying
        different items share no block, as their KV blocks would not.
        """
        size = self.block_size
        parent = hash(("tt-prefix-cache-mirror", tuple(mm_identifiers)))
        hashes: list[int] = []
        for start in range(0, len(prompt_token_ids) - size + 1, size):
            parent = hash((parent, tuple(prompt_token_ids[start : start + size])))
            hashes.append(parent)
        return hashes

    def cached_tokens(self, hashes: Sequence[int], max_hit: int) -> list[int]:
        """Prompt tokens each engine holds: its run of leading ``hashes``.

        Capped at ``max_hit``, as the KV cache manager caps a lookup at the
        prompt length less one so a fully cached prompt still computes a token.
        """
        hits = [0] * self.num_engines
        alive = (1 << self.num_engines) - 1
        held_blocks = 0
        for block_hash in hashes:
            held = self._engines_by_block.get(block_hash, 0) & alive
            if held != alive:
                lost = alive & ~held
                for engine in range(self.num_engines):
                    if lost >> engine & 1:
                        hits[engine] = min(held_blocks * self.block_size, max_hit)
                alive = held
                if not alive:
                    return hits
            held_blocks += 1
        for engine in range(self.num_engines):
            if alive >> engine & 1:
                hits[engine] = min(held_blocks * self.block_size, max_hit)
        return hits

    def record(
        self,
        request_id: str,
        hashes: Iterable[int],
        engine: int,
        *,
        uncached_tokens: int,
        num_tokens: int,
    ) -> None:
        """Record that ``engine`` was sent the prompt whose blocks are ``hashes``."""
        blocks = self._engines_by_block
        bit = 1 << engine
        for block_hash in hashes:
            blocks[block_hash] = blocks.pop(block_hash, 0) | bit
        while len(blocks) > self._max_blocks:
            del blocks[next(iter(blocks))]
        self._dispatched[request_id] = (engine, uncached_tokens, num_tokens)

    def forget_finished(self, in_flight: Collection[str]) -> None:
        """Drop the dispatch records of requests no longer in flight."""
        for request_id in [rid for rid in self._dispatched if rid not in in_flight]:
            del self._dispatched[request_id]

    def queued_prefill_tokens(self, engine: int, waiting: int) -> int:
        """Prefill tokens the newest ``waiting`` in-flight requests of ``engine``
        had to compute when routed: the estimate of the prefill queued ahead
        there, which the lane coordinator reads off its lane's waiting queue."""
        if waiting <= 0:
            return 0
        total = counted = 0
        for routed_engine, uncached, _ in reversed(self._dispatched.values()):
            if routed_engine != engine:
                continue
            total += uncached
            counted += 1
            if counted >= waiting:
                break
        return total

    def clear(self) -> None:
        """Forget every block: the engines' caches were reset, so each in-flight
        request that has yet to prefill recomputes its whole prompt."""
        self._engines_by_block.clear()
        self._dispatched = {
            request_id: (engine, num_tokens, num_tokens)
            for request_id, (engine, _, num_tokens) in self._dispatched.items()
        }


def _dp_mirror(client: Any) -> PrefixCacheMirror | None:
    """The client's mirror, built on first use; ``None`` when routing is off."""
    mirror = getattr(client, "_tt_prefix_cache_mirror", _UNSET)
    if mirror is _UNSET:
        vllm_config = client.vllm_config
        mirror = None
        if get_tt_dp_prefix_cache_routing(vllm_config):
            mirror = PrefixCacheMirror(
                vllm_config.cache_config.block_size, len(client.core_engines)
            )
        client._tt_prefix_cache_mirror = mirror
    return mirror


def _upstream_load(client: Any) -> tuple[list[float], int]:
    """vLLM's load score per engine and the engine its balancer would pick.

    Computed as ``DPLBAsyncMPClient.get_core_engine_for_request`` computes
    them, from the same counts and in the same scan order, without its side
    effects: the first minimum from ``eng_start_index`` wins.
    """
    counts = client.lb_engines
    num_engines = len(counts)
    scores: list[float] = []
    for engine in range(num_engines):
        waiting, running, kv_cache_usage = counts[engine]
        inflight = client.engine_inflight[client.core_engines[engine]]
        score = float(max(client.client_count * inflight, waiting + running))
        if waiting:
            score += waiting * 6.0 * max(0.0, kv_cache_usage - 0.5)
        scores.append(score)
    start = client.eng_start_index
    offset = min(
        range(num_engines), key=lambda i: (scores[(start + i) % num_engines], i)
    )
    return scores, (start + offset) % num_engines


def _occupancy(client: Any, engine: int) -> int:
    """Running slots ``engine`` holds or has promised, as the balancer sees
    them: the greater of its exact in-flight count and the coordinator's
    waiting-plus-running snapshot."""
    waiting, running, _ = client.lb_engines[engine]
    inflight = client.engine_inflight[client.core_engines[engine]]
    return max(client.client_count * inflight, waiting + running)


def route_dp_request(client: Any, request: Any, original: Callable) -> Any:
    """``DPLBAsyncMPClient.get_core_engine_for_request`` with prefix routing.

    ``original`` is vLLM's load balancer. A request with no prompt tokens, or
    one sent while the engine set is changing, is left to it. Every other
    prompt is looked up in the mirror; a request not already pinned by its
    caller is pinned, through ``data_parallel_rank``, to the engine
    :func:`pick_prefix_cache_target` prefers over the balancer's own
    least-loaded choice, recomputed here from the counts the balancer reads.
    When the rule keeps the least-loaded engine the request stays unpinned and
    the balancer picks it as it would have. Either way the engine the balancer
    returns is recorded against the prompt's blocks.
    """
    mirror = _dp_mirror(client)
    prompt = request.prompt_token_ids
    engines = client.core_engines
    if (
        mirror is None
        or not prompt
        or len(engines) != mirror.num_engines
        or len(client.lb_engines) != len(engines)
    ):
        return original(client, request)
    hashes = mirror.block_hashes(
        prompt, (feature.identifier for feature in request.mm_features or ())
    )
    mirror.forget_finished(client.reqs_in_flight)
    hits = mirror.cached_tokens(hashes, len(prompt) - 1)
    if request.data_parallel_rank is None:
        scores, least = _upstream_load(client)
        max_num_seqs = client.vllm_config.scheduler_config.max_num_seqs
        target = pick_prefix_cache_target(
            least,
            scores,
            hits,
            lambda engine: _occupancy(client, engine) >= max_num_seqs,
            lambda engine: mirror.queued_prefill_tokens(
                engine, int(client.lb_engines[engine][0])
            ),
        )
        if target != least:
            request.data_parallel_rank = target
            # The local waiting increment the balancer applies on its own
            # path, so its next choice sees this request queued there.
            client.lb_engines[target][0] += client.client_count
    chosen = original(client, request)
    engine = engines.index(chosen)
    mirror.record(
        request.request_id,
        hashes,
        engine,
        uncached_tokens=len(prompt) - hits[engine],
        num_tokens=len(prompt),
    )
    return chosen


def forget_dp_prefix_mirror(client: Any) -> None:
    """Clear the client's mirror: its engines are resetting their prefix caches."""
    mirror = getattr(client, "_tt_prefix_cache_mirror", None)
    if mirror is not None:
        mirror.clear()
