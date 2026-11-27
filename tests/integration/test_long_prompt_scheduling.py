# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Long-prompt scheduling: ``additional_config['tt']`` keys
``max_concurrent_partial_prefills`` and ``prefill_chunk_cap_table``.

A TT step runs prefill rows or decode rows, never both, so every running
decode waits for each prefill step. ``max_concurrent_partial_prefills`` limits
how many prompts that need more than one chunk ("multi-chunk") advance per
prefill step, oldest first, and lets waiting single-chunk prompts go first for
a bounded number of steps. ``prefill_chunk_cap_table`` caps a text chunk by
the depth it starts at.

Most tests drive a real ``TTScheduler`` over a real ``KVCacheManager`` in the
order of the synchronous ``EngineCore.step``: schedule a step, then commit the
output the TT model runner gives for it. The chunk-hook tests call
``_mamba_block_aligned_split`` of a ``TTScheduler`` built without
``__init__``, as ``tests/test_prefill_chunk_alignment.py`` does.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.partial_prefill_limit import (
    SHORT_FIRST_MAX_STEPS,
    HeldPrefills,
    PartialPrefillLimit,
)
from vllm_tt_plugin.platform import _validate_long_prompt_scheduling
from vllm_tt_plugin.scheduler import (
    TTScheduler,
    _validate_prefill_chunk_cap_table,
    scheduler_output_has_prefill_work,
)

BLOCK_SIZE = 16
MAX_MODEL_LEN = 2048
MAX_NUM_SEQS = 12
# Token budget of one step (``max_num_batched_tokens``).
BUDGET = 1024
# A multiple of the default ``prefill_chunk_alignment`` (128), so a capped
# chunk already ends on the alignment grid.
THRESHOLD = 128
# Four chunks under the threshold: 128, 128, 128 and 16.
LONG = 400
SHORT = 16
SAMPLED_TOKEN = 7
LOCAL_MODEL_CONFIG = Path(__file__).parent.parent / "model_configs" / "qwen2"
# Prefix-cache hits need block hashes, which need vLLM's hash seed.
init_none_hash(sha256)


@contextmanager
def _stub_model_resolution(capabilities: dict) -> Iterator[None]:
    """Resolve a stub TT model class that declares ``capabilities`` instead of
    a tt-metal class while ``VllmConfig`` runs the platform hook."""
    model_class = type("_StubModel", (), {"model_capabilities": capabilities})
    with (
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch(
            "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            return_value=(model_class, None),
        ),
    ):
        yield


def _scheduler(
    tt: dict | None = None,
    *,
    threshold: int = THRESHOLD,
    hybrid: bool = False,
    async_scheduling: bool = False,
) -> TTScheduler:
    """A real ``TTScheduler`` with chunked prefill.

    ``tt`` is ``additional_config['tt']``; decode interleave is off unless it
    says otherwise, so prefill steps follow each other while prefill work is
    pending. ``hybrid`` gives the model a full-attention group and a
    ``MambaSpec`` group in mode ``align`` with prefix caching, and declares
    ``supports_prefill_resume_inside_kv_block`` False, as Qwen3.8 does, so
    chunks end on KV block boundaries.
    """
    capabilities = {"supports_chunked_prefill": True}
    if hybrid:
        capabilities["supports_prefill_resume_inside_kv_block"] = False
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=BUDGET,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=threshold,
        async_scheduling=async_scheduling,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    with _stub_model_resolution(capabilities):
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            additional_config={
                "tt": {"decode_interleave_enabled": False, **(tt or {})}
            },
        )
    config.scheduler_config.async_scheduling = async_scheduling
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        )
    ]
    if hybrid:
        cache_config.enable_prefix_caching = True
        cache_config.mamba_cache_mode = "align"
        groups.append(
            KVCacheGroupSpec(
                ["gdn"],
                MambaSpec(
                    block_size=BLOCK_SIZE,
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    mamba_cache_mode="align",
                    num_prefill_checkpoint_blocks=1,
                    prefill_checkpoint_alignment=BLOCK_SIZE,
                ),
            )
        )
    num_blocks = len(groups) * MAX_NUM_SEQS * MAX_MODEL_LEN // BLOCK_SIZE + 1
    cache_config.num_gpu_blocks = num_blocks
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
        ),
        block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(request_id: str, prompt: int | list[int]) -> Request:
    """A request of ``prompt`` tokens: a list, or a length of repeated 1s."""
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * prompt if isinstance(prompt, int) else prompt,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _commit(scheduler: TTScheduler, step: SchedulerOutput) -> None:
    """Commit ``step`` as the TT model runner reports it: a request whose
    scheduled tokens complete its prompt samples ``SAMPLED_TOKEN``; an
    intermediate prefill chunk samples nothing."""
    req_ids = list(step.num_scheduled_tokens)
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: row for row, req_id in enumerate(req_ids)},
            sampled_token_ids=[
                [] if scheduler.requests[req_id].is_prefill_chunk else [SAMPLED_TOKEN]
                for req_id in req_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _step(scheduler: TTScheduler) -> SchedulerOutput:
    step = scheduler.schedule()
    _commit(scheduler, step)
    return step


def _prefill_chunks(scheduler: TTScheduler, steps: int) -> list[dict[str, int]]:
    """The tokens each of the next ``steps`` prefill steps schedules, or the
    fewer that come before the scheduler runs out of prefill work."""
    chunks = []
    while len(chunks) < steps and scheduler._has_pending_prefill():
        step = _step(scheduler)
        if scheduler_output_has_prefill_work(step):
            chunks.append(dict(step.num_scheduled_tokens))
    return chunks


def _waiting(scheduler: TTScheduler) -> list[str]:
    return [request.request_id for request in scheduler.waiting]


class TestPartialPrefillLimit:
    """``max_concurrent_partial_prefills``: at most L multi-chunk prompts
    advance per prefill step, oldest first."""

    def test_without_the_key_long_prompts_advance_together(self):
        scheduler = _scheduler()
        scheduler.add_request(_request("a", LONG))
        scheduler.add_request(_request("b", LONG))

        assert _prefill_chunks(scheduler, 1) == [{"a": 128, "b": 128}]

    def test_multi_chunk_prompts_advance_one_at_a_time_oldest_first(self):
        scheduler = _scheduler({"max_concurrent_partial_prefills": 1})
        scheduler.add_request(_request("a", LONG))
        scheduler.add_request(_request("b", LONG))

        chunks = _prefill_chunks(scheduler, 7)

        # The last 16 tokens of "a" fit one chunk, so they no longer hold the
        # place, and "b" starts beside them.
        assert chunks == [
            {"a": 128},
            {"a": 128},
            {"a": 128},
            {"a": 16, "b": 128},
            {"b": 128},
            {"b": 128},
            {"b": 16},
        ]

    def test_the_threshold_holds_while_a_held_prompt_waits(self):
        # "b" waits for the step as much as a hidden decode does, so "a" is
        # not a lone request and keeps the 128-token threshold instead of the
        # whole budget.
        scheduler = _scheduler({"max_concurrent_partial_prefills": 1})
        scheduler.add_request(_request("a", LONG))
        scheduler.add_request(_request("b", LONG))

        step = _step(scheduler)

        assert step.num_scheduled_tokens == {"a": 128}
        # Between passes the counts are vLLM's: "a" runs, "b" waits.
        assert scheduler.get_request_counts() == (1, 1)

    def test_a_single_chunk_prompt_joins_a_running_partial_prefill(self):
        scheduler = _scheduler({"max_concurrent_partial_prefills": 1})
        scheduler.add_request(_request("a", LONG))
        scheduler.add_request(_request("b", LONG))
        assert _step(scheduler).num_scheduled_tokens == {"a": 128}
        scheduler.add_request(_request("c", SHORT))
        scheduler.add_request(_request("d", LONG))

        step = _step(scheduler)

        assert step.num_scheduled_tokens == {"a": 128, "c": SHORT}
        # The held prompts are back in the queue, in their order.
        assert _waiting(scheduler) == ["b", "d"]
        assert [r.request_id for r in scheduler.running] == ["a", "c"]

    def test_short_prompts_go_first_for_a_bounded_number_of_steps(self):
        scheduler = _scheduler({"max_concurrent_partial_prefills": 1})
        scheduler.add_request(_request("long", LONG))
        chunks = []
        for index in range(SHORT_FIRST_MAX_STEPS + 1):
            scheduler.add_request(_request(f"short-{index}", SHORT))
            chunks.append(_prefill_chunks(scheduler, 1)[0])

        assert chunks[:SHORT_FIRST_MAX_STEPS] == [
            {f"short-{index}": SHORT} for index in range(SHORT_FIRST_MAX_STEPS)
        ]
        assert chunks[SHORT_FIRST_MAX_STEPS] == {
            "long": 128,
            f"short-{SHORT_FIRST_MAX_STEPS}": SHORT,
        }

    def test_decode_interleave_still_sees_the_held_prompt(self):
        # "b" is held from every prefill pass but stays pending prefill work,
        # so the decode interleave keeps alternating with "a"'s chunks.
        scheduler = _scheduler(
            {
                "max_concurrent_partial_prefills": 1,
                "decode_interleave_enabled": True,
                "decode_interleave_prefill_steps": 1,
            }
        )
        scheduler.add_request(_request("decode", SHORT))
        _step(scheduler)
        _step(scheduler)
        scheduler.add_request(_request("a", LONG))
        scheduler.add_request(_request("b", LONG))

        steps = [dict(_step(scheduler).num_scheduled_tokens) for _ in range(4)]

        assert steps == [{"a": 128}, {"decode": 1}, {"a": 128}, {"decode": 1}]
        assert _waiting(scheduler) == ["b"]

    def test_a_kept_prefill_holds_only_what_its_pass_scheduled(self):
        # Under async scheduling a prefill that follows an uncommitted decode
        # is kept in _deferred_prefill while an empty step lets the engine
        # commit the decode. The held prompt is back in the queue before the
        # kept prefill is handed out.
        scheduler = _scheduler(
            {"max_concurrent_partial_prefills": 1}, async_scheduling=True
        )
        scheduler.add_request(_request("decode", SHORT))
        _commit(scheduler, scheduler.schedule())
        decode = scheduler.schedule()
        assert set(decode.num_scheduled_tokens) == {"decode"}
        scheduler.add_request(_request("a", LONG))
        scheduler.add_request(_request("b", LONG))

        release = scheduler.schedule()

        assert release.total_num_scheduled_tokens == 0
        assert _waiting(scheduler) == ["b"]
        assert scheduler.get_request_counts() == (2, 1)
        _commit(scheduler, decode)
        kept = scheduler.schedule()
        assert kept.num_scheduled_tokens == {"a": 128}
        assert _waiting(scheduler) == ["b"]


# Chunk of the fake base pass below.
CHUNK = 4096


class _FakeRequest:
    """A request for the fake base pass, equal only to itself like vLLM's
    ``Request``."""

    def __init__(self, request_id: str, prompt: int, computed: int = 0) -> None:
        self.request_id = request_id
        self.num_prompt_tokens = self.num_tokens = prompt
        self.num_computed_tokens = computed
        self.is_prefill_chunk = 0 < computed < prompt
        self.num_output_placeholders = 0
        self.mm_features = []
        self.has_encoder_inputs = False


def _bare(limit: int, waiting=(), running=()) -> TTScheduler:
    """A ``TTScheduler`` built without ``__init__``, with the state
    ``_schedule_prefill_only`` reads, the threshold at ``CHUNK`` and no
    prefix cache."""
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    for request in waiting:
        scheduler.waiting.add_request(request)
    scheduler.running = list(running)
    scheduler.max_num_running_reqs = scheduler.max_num_active_reqs = 8
    scheduler._num_hidden_decodes = 0
    scheduler._held_prefills = HeldPrefills()
    scheduler._partial_prefill_limit = PartialPrefillLimit(limit)
    scheduler._prefill_chunk_cap_table = ()
    scheduler._multimodal_prefill_block_size = 0
    scheduler.max_num_scheduled_tokens = 525312
    scheduler.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=525312, long_prefill_token_threshold=CHUNK
    )
    scheduler.kv_cache_manager = SimpleNamespace(
        prefix_cache_lookup_enabled=lambda request: False
    )
    return scheduler


def _fake_base_pass(monkeypatch, advanced: list[list[str]], pass_body=None) -> None:
    """Replace vLLM's ``schedule`` with a pass that gives every visible
    running request and every waiting request it admits one ``CHUNK``, as
    vLLM's pass does under the threshold, and logs the request ids it
    advanced. ``pass_body`` replaces that pass."""

    def chunk_each(self, throttle_prefills=False):
        step = [r.request_id for r in self.running]
        for request in self.running:
            request.num_computed_tokens = min(
                request.num_tokens, request.num_computed_tokens + CHUNK
            )
        while self.waiting and len(self.running) < self.max_num_running_reqs:
            request = self.waiting.pop_request()
            request.num_computed_tokens = min(request.num_tokens, CHUNK)
            step.append(request.request_id)
            self.running.append(request)
        for request in self.running:
            request.is_prefill_chunk = request.num_computed_tokens < request.num_tokens
        advanced.append(step)
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(AsyncScheduler, "schedule", pass_body or chunk_each)


class TestHidingAroundTheBasePass:
    """``_schedule_prefill_only`` hides the held requests from one base pass
    and puts them back in their places, with a fake base pass."""

    def test_long_prompts_run_one_at_a_time_after_waiting_short_ones(self, monkeypatch):
        longs = [_FakeRequest(f"L{i}", 3 * CHUNK + 100) for i in range(3)]
        scheduler = _bare(1, waiting=[longs[0], _FakeRequest("s0", 100), *longs[1:]])
        advanced = []
        _fake_base_pass(monkeypatch, advanced)
        for step in range(9):
            scheduler._schedule_prefill_only()
            if step == 0:
                scheduler.waiting.add_request(_FakeRequest("s1", 200))

        # The short prompts go first, each in the step after it arrives.
        assert advanced[:2] == [["s0"], ["s1"]]
        # L0 needs four chunks and advances alone while it needs more than
        # one; its last chunk fits one, so L1 starts beside it.
        assert advanced[2:] == [
            ["L0"],
            ["L0"],
            ["L0"],
            ["L0", "L1"],
            ["L1"],
            ["L1"],
            ["L1", "L2"],
        ]

    def test_running_partial_prefills_come_first_then_waiting_ones(self, monkeypatch):
        running = [_FakeRequest(f"R{i}", 10 * CHUNK, CHUNK) for i in range(3)]
        waiting = [_FakeRequest("W0", 9 * CHUNK), _FakeRequest("w", 50)]
        scheduler = _bare(2, waiting=waiting, running=running)
        advanced = []
        _fake_base_pass(monkeypatch, advanced)

        scheduler._schedule_prefill_only()

        assert advanced == [["R0", "R1", "w"]]
        # The held R2 is back after R0 and R1, ahead of the admitted w.
        assert [r.request_id for r in scheduler.running] == ["R0", "R1", "R2", "w"]
        assert [r.request_id for r in scheduler.waiting] == ["W0"]

    def test_waiting_requests_take_the_places_left(self, monkeypatch):
        running = [_FakeRequest("R0", 10 * CHUNK, CHUNK)]
        waiting = [_FakeRequest("W0", 9 * CHUNK), _FakeRequest("W1", 9 * CHUNK)]
        scheduler = _bare(2, waiting=waiting, running=running)
        advanced = []
        _fake_base_pass(monkeypatch, advanced)

        scheduler._schedule_prefill_only()

        assert advanced == [["R0", "W0"]]
        assert [r.request_id for r in scheduler.waiting] == ["W1"]

    def test_a_held_partial_prefill_is_not_preempted(self, monkeypatch):
        a = _FakeRequest("A", 10 * CHUNK, 3 * CHUNK)
        b = _FakeRequest("B", 10 * CHUNK, 3 * CHUNK)
        scheduler = _bare(1, running=[a, b])
        visible = []

        def preempt_the_last(self, throttle_prefills=False):
            # KV pressure: vLLM preempts the last visible running request.
            visible.append([r.request_id for r in self.running])
            self.waiting.prepend_request(self.running.pop())
            return SchedulerOutput.make_empty()

        _fake_base_pass(monkeypatch, [], preempt_the_last)

        scheduler._schedule_prefill_only()

        assert visible == [["A"]]
        assert scheduler.running == [b] and b.num_computed_tokens == 3 * CHUNK
        assert list(scheduler.waiting) == [a]

    def test_a_request_preempted_in_the_pass_stays_ahead_of_held_ones(
        self, monkeypatch
    ):
        scheduler = _bare(
            1, waiting=[_FakeRequest("L", 100_000), _FakeRequest("s1", 100)]
        )

        def admit_one_and_preempt(self, throttle_prefills=False):
            self.waiting.pop_request()
            self.waiting.prepend_request(_FakeRequest("P", 50))
            return SchedulerOutput.make_empty()

        _fake_base_pass(monkeypatch, [], admit_one_and_preempt)

        scheduler._schedule_prefill_only()

        assert [r.request_id for r in scheduler.waiting] == ["P", "L"]


class TestHybridCacheHitEstimate:
    """The multi-chunk decision counts a prefix-cache hit only where every KV
    cache group hits: for a ``MambaSpec`` group, only at a block boundary
    with a cached GDN state block."""

    # A prompt of six KV blocks. Prefilled in one chunk, it leaves the GDN
    # state cached at its prefill checkpoint, the last block boundary before
    # its last token (80), and at its end (96); the attention blocks are
    # cached at every block boundary.
    FIRST = list(range(100, 196))
    STATE_BOUNDARY = 80

    @staticmethod
    def _attention_only_hit(scheduler: TTScheduler, request: Request) -> int:
        """The hit of the full-attention group alone: its leading cached
        blocks."""
        pool = scheduler.kv_cache_manager.block_pool
        max_blocks = (request.num_tokens - 1) // BLOCK_SIZE
        hits = 0
        for block_hash in request.block_hashes[:max_blocks]:
            if not pool.get_cached_block(block_hash, [0]):
                break
            hits += 1
        return hits * BLOCK_SIZE

    def _scheduler_after_first_prompt(self) -> TTScheduler:
        scheduler = _scheduler({"max_concurrent_partial_prefills": 1}, hybrid=True)
        scheduler.add_request(_request("first", self.FIRST))
        assert _prefill_chunks(scheduler, 1) == [{"first": len(self.FIRST)}]
        return scheduler

    def test_an_attention_hit_without_a_gdn_state_block_is_not_counted(self):
        scheduler = self._scheduler_after_first_prompt()
        # Shares 48 tokens with "first": three cached attention blocks, but
        # no GDN state block at token 48. Its 128 uncached tokens would fit
        # one chunk after the attention hit, and need two from token 0.
        follow_up = _request("follow-up", self.FIRST[:48] + [7] * 128)
        assert self._attention_only_hit(scheduler, follow_up) == 48

        assert scheduler._estimated_cache_hit(follow_up) == 0
        assert scheduler._is_multi_chunk(follow_up)

        # vLLM computes it from token 0 as well: from a 48-token hit its first
        # chunk would hold 128 tokens. It ends that chunk at the shared prefix
        # (48) to cache the GDN state there for later prompts.
        scheduler.add_request(follow_up)
        assert _prefill_chunks(scheduler, 2) == [{"follow-up": 48}, {"follow-up": 128}]

    def test_a_hit_at_a_cached_gdn_state_block_is_counted(self):
        scheduler = self._scheduler_after_first_prompt()
        follow_up = _request("follow-up", self.FIRST[: self.STATE_BOUNDARY] + [7] * 128)

        assert scheduler._estimated_cache_hit(follow_up) == self.STATE_BOUNDARY
        assert not scheduler._is_multi_chunk(follow_up)

        # A single-chunk prompt joins the running partial prefill.
        scheduler.add_request(_request("long", LONG))
        _step(scheduler)
        scheduler.add_request(follow_up)
        assert _prefill_chunks(scheduler, 1) == [{"long": 128, "follow-up": 128}]


class TestPrefillChunkCapTable:
    """``prefill_chunk_cap_table``: a text chunk holds at most the cap of the
    depth it starts at."""

    TABLE = [[0, 256], [256, 128]]

    def test_chunks_shrink_with_depth_also_for_a_lone_prompt(self):
        # Without a threshold vLLM gives a lone 700-token prompt one step.
        plain = _scheduler(threshold=0)
        plain.add_request(_request("lone", 700))
        assert _prefill_chunks(plain, 1) == [{"lone": 700}]
        capped = _scheduler({"prefill_chunk_cap_table": self.TABLE}, threshold=0)
        capped.add_request(_request("lone", 700))

        chunks = _prefill_chunks(capped, 5)

        assert [chunk["lone"] for chunk in chunks] == [256, 128, 128, 128, 60]

    def test_a_cap_off_the_alignment_grid_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 192\].*multiple of 128"):
            _scheduler({"prefill_chunk_cap_table": [[0, 192]]})

    @pytest.mark.parametrize(
        ("cap", "align", "kv_block", "ok"),
        [
            pytest.param(256, 128, 1, True, id="on-the-grid"),
            pytest.param(192, 128, 1, False, id="off-the-grid"),
            pytest.param(48, 0, 1, True, id="no-alignment"),
            pytest.param(512, 128, 256, True, id="on-the-mamba-block"),
            pytest.param(384, 128, 256, False, id="off-the-mamba-block"),
        ],
    )
    def test_caps_must_keep_chunk_ends_aligned(self, cap, align, kv_block, ok):
        table = ((0, 1024), (4096, cap))
        if ok:
            _validate_prefill_chunk_cap_table(table, align, kv_block)
        else:
            with pytest.raises(ValueError, match=f"\\[4096, {cap}\\]"):
                _validate_prefill_chunk_cap_table(table, align, kv_block)


def _hook(table, alignment: int = 128, multimodal_block: int = 0) -> TTScheduler:
    """A ``TTScheduler`` built without ``__init__``, with the state its chunk
    hook reads."""
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler._prefill_chunk_alignment = alignment
    scheduler._prefill_chunk_cap_table = table
    scheduler._multimodal_prefill_block_size = multimodal_block
    scheduler.need_mamba_block_aligned_split = False
    scheduler.max_num_scheduled_tokens = 4096
    scheduler.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=4096, long_prefill_token_threshold=0
    )
    scheduler._step_granted_tokens = 0
    scheduler._step_granted_rows = 0
    return scheduler


def _prompt(tokens: int, computed: int = 0, image_end: int = 0) -> SimpleNamespace:
    """A request whose prompt holds an image up to ``image_end`` (none at 0)."""
    image = SimpleNamespace(mm_position=SimpleNamespace(offset=0, length=image_end))
    return SimpleNamespace(
        request_id="r",
        num_prompt_tokens=tokens,
        num_tokens=tokens,
        num_computed_tokens=computed,
        mm_features=[image] if image_end else [],
        has_encoder_inputs=bool(image_end),
    )


class TestChunkHook:
    """The cap in ``_mamba_block_aligned_split``: after vLLM's split, before
    ``_align_prefill_chunk_end``, and not on multimodal chunks."""

    TABLE = ((0, 256), (512, 128))

    @pytest.mark.parametrize(
        ("computed", "grant"),
        [
            pytest.param(0, 256, id="depth-0"),
            pytest.param(384, 256, id="last-pair-before-the-start"),
            pytest.param(512, 128, id="pair-at-the-start"),
            pytest.param(1024, 128, id="past-the-last-pair"),
        ],
    )
    def test_the_cap_of_the_start_depth_applies(self, computed, grant):
        hook = _hook(self.TABLE)
        assert hook._mamba_block_aligned_split(_prompt(4000, computed), 900) == grant

    def test_no_cap_before_the_first_pair(self):
        hook = _hook(((1024, 256),))
        assert hook._mamba_block_aligned_split(_prompt(4000), 900) == 896

    def test_a_capped_chunk_is_then_aligned(self):
        # A prefix-cache hit at 64 starts off the 128 grid: the capped chunk
        # ends at 320 and is rounded down to 256.
        hook = _hook(self.TABLE)
        assert hook._mamba_block_aligned_split(_prompt(4000, 64), 900) == 192

    def test_the_last_chunk_is_capped_too(self):
        hook = _hook(((0, 1024),))
        assert hook._mamba_block_aligned_split(_prompt(5000, 3072), 1928) == 1024

    def test_a_last_chunk_under_the_cap_is_left_alone(self):
        hook = _hook(self.TABLE)
        assert hook._mamba_block_aligned_split(_prompt(300, 128), 172) == 172

    def test_decode_steps_are_not_capped(self):
        # A verify of 1 + 3 tokens past the prompt, under a 2-token cap that
        # only a launch without alignment allows.
        hook = _hook(((0, 2),), alignment=0)
        verify = SimpleNamespace(
            num_prompt_tokens=300,
            num_tokens=310,
            num_computed_tokens=309,
            mm_features=[],
            has_encoder_inputs=False,
        )
        assert hook._mamba_block_aligned_split(verify, 4) == 4

    def test_multimodal_chunks_with_prefill_resume_are_not_capped(self):
        hook = _hook(self.TABLE)
        assert hook._mamba_block_aligned_split(_prompt(4000, 0, 600), 900) == 900

    def test_the_multimodal_part_without_prefill_resume_is_not_capped(self):
        # The first chunk holds every item and, as it fits the step, the
        # whole prompt (_multimodal_prefill_grant).
        hook = _hook(self.TABLE, multimodal_block=16)
        prompt = _prompt(1000, 0, 600)
        assert hook._mamba_block_aligned_split(prompt, 900) == 1000

    def test_text_chunks_after_the_items_are_capped(self):
        hook = _hook(self.TABLE, multimodal_block=16)
        prompt = _prompt(4000, 1024, 600)
        assert hook._mamba_block_aligned_split(prompt, 900) == 128


def _launch(tt: dict, chunked_prefill: bool) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={"tt": tt},
        scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked_prefill),
    )


class TestPlatformValidation:
    """The platform refuses the keys on a launch without chunked prefill."""

    @pytest.mark.parametrize(
        "tt",
        [
            pytest.param({"max_concurrent_partial_prefills": 1}, id="limit"),
            pytest.param({"prefill_chunk_cap_table": [[0, 256]]}, id="cap-table"),
        ],
    )
    def test_the_keys_need_chunked_prefill(self, tt):
        _validate_long_prompt_scheduling(_launch(tt, chunked_prefill=True))
        with pytest.raises(ValueError, match="need chunked prefill"):
            _validate_long_prompt_scheduling(_launch(tt, chunked_prefill=False))

    def test_absent_or_neutral_keys_need_nothing(self):
        neutral = {"max_concurrent_partial_prefills": 0, "prefill_chunk_cap_table": []}
        _validate_long_prompt_scheduling(_launch({}, chunked_prefill=False))
        _validate_long_prompt_scheduling(_launch(neutral, chunked_prefill=False))

    def test_a_malformed_value_fails_before_the_model_loads(self):
        launch = _launch({"prefill_chunk_cap_table": [[0, -1]]}, chunked_prefill=True)
        with pytest.raises(ValueError, match="max_chunk_tokens >= 1"):
            _validate_long_prompt_scheduling(launch)
