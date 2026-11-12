# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Multimodal requests of a model with ``MambaSpec`` layers.

Such a model (the Qwen3.8 adapter) builds the M-RoPE positions and vision rows
of a prefill step from that step's tokens and starts a multimodal prompt at
position 0. So ``TTScheduler`` keeps a multimodal request off the prefix cache
and schedules every multimodal item of it in its first prefill chunk, the whole
prompt when one step holds it, and ``TTPlatform.validate_request`` rejects a
multimodal prompt longer than one step. Text requests of the same launch keep
prefix caching and chunked prefill.

The tests drive a real ``TTScheduler`` with its ``KVCacheManager`` and a real
``TTModelRunner`` through ``_Harness`` (``test_mamba_state_lifecycle.py``),
which checks every step's state blocks against vLLM's allocation and a ledger
of the state each block holds, and in mode ``align`` that every state vLLM
keeps under a prefix hash is the state of exactly the tokens that hash covers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from tests.test_mamba_kv_cache import _kv_cache_config
from tests.test_mamba_state_lifecycle import (
    BLOCK_SIZE,
    EOS,
    MAX_NUM_SEQS,
    _Harness,
    _prompt,
    _scheduler,
)
from vllm_tt_plugin.config import store_tt_mamba_spec_layers
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.scheduler import multimodal_prefill_limit

STEP = 4096
ENCODER_BUDGET = 4096


def _item(identifier: str, offset: int, length: int, modality: str = "image"):
    """One multimodal item with a pixel payload, as the frontend sends it."""
    pixels = torch.full((length, 4), float(abs(hash(identifier)) % 97))
    return MultiModalFeatureSpec(
        data={
            "pixel_values": SimpleNamespace(data=pixels),
            "image_grid_thw": SimpleNamespace(data=torch.tensor([1, 2, length // 2])),
        },
        modality=modality,
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


class _MultimodalHarness(_Harness):
    """``_Harness`` with multimodal requests, an encoder budget, and a record
    of the pixels each prefill row hands the model."""

    def __init__(self, *args, encoder_budget: int = ENCODER_BUDGET, **kwargs):
        super().__init__(*args, **kwargs)
        self.runner.model_config.is_multimodal_model = True
        self.scheduler.max_num_encoder_input_tokens = encoder_budget
        self.scheduler.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_budget
        )
        # Request id -> per prefill step, the identifiers of the items whose
        # pixels the row carried.
        self.row_pixels: dict[str, list[list[str]]] = {}

    def add_mm(
        self,
        req_id: str,
        prompt: list[int],
        items: list[MultiModalFeatureSpec],
        max_tokens: int = 8,
    ) -> None:
        init_none_hash(sha256)
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0)
        params.update_from_generation_config({}, eos_token_id=EOS)
        self.num_added += 1
        request = Request(
            request_id=req_id,
            prompt_token_ids=prompt,
            sampling_params=params,
            pooling_params=None,
            mm_features=items,
            block_hasher=get_request_block_hasher(self.hash_block_size, sha256),
        )
        self.requests[req_id] = request
        self.scheduler.add_request(request)

    def finish_step(self, model_input, rows, is_prefill, verify) -> ModelRunnerOutput:
        if is_prefill:
            kwargs = model_input.multi_modal_kwargs
            for row, req_id in enumerate(model_input.row_req_ids):
                features = self.requests[req_id].mm_features
                pixels = kwargs["pixel_values"][row]
                if not features:
                    assert pixels is None
                    continue
                # Every item's pixels, on every prefill step of the request.
                assert len(pixels) == len(features)
                for feature, carried in zip(features, pixels):
                    assert torch.equal(carried, feature.data["pixel_values"].data)
                self.row_pixels.setdefault(req_id, []).append(
                    kwargs["mm_identifiers"][row]
                )
        return super().finish_step(model_input, rows, is_prefill, verify)


def _admit(config, prompt_len: int, items: dict[str, list[PlaceholderRange]]):
    """``TTPlatform.validate_request`` on a tokenized prompt of this launch."""
    inputs = {
        "type": "multimodal" if items else "token",
        "prompt_token_ids": list(range(prompt_len)),
    }
    if items:
        inputs["mm_placeholders"] = items
    previous = TTPlatform._tt_vllm_config
    TTPlatform._tt_vllm_config = config
    try:
        TTPlatform.validate_request(inputs, SamplingParams(max_tokens=8))
    finally:
        TTPlatform._tt_vllm_config = previous


# region Admission


@pytest.mark.parametrize(
    ("prompt_len", "image_offset", "admitted"),
    [
        (STEP - 1, 100, True),
        (STEP, STEP - 300, True),
        (STEP + 1, 100, False),
        # A small image inside a long prompt: the image fits a step, the
        # prompt does not.
        (10_000, 4000, False),
    ],
)
def test_a_multimodal_prompt_is_admitted_exactly_when_one_step_holds_it(
    prompt_len, image_offset, admitted
):
    """Admission checks the whole prompt against one scheduler step, and an
    admitted prompt is then prefilled in exactly one step."""
    h = _MultimodalHarness(
        num_blocks=64,
        max_num_batched_tokens=STEP,
        mode="align",
        checkpoint_alignment=BLOCK_SIZE,
    )
    config = h.scheduler.vllm_config
    store_tt_mamba_spec_layers(config, True)
    image = {"image": [PlaceholderRange(offset=image_offset, length=256)]}
    if not admitted:
        with pytest.raises(VLLMValidationError) as raised:
            _admit(config, prompt_len, image)
        assert str(raised.value) == (
            f"This multimodal prompt has {prompt_len} tokens. This model keeps "
            "recurrent state in MambaSpec blocks and prefills a multimodal "
            f"prompt in one scheduler step, which holds at most {STEP} prompt "
            f"tokens (max_num_batched_tokens={STEP}). Send a shorter prompt, "
            "or raise that limit at launch."
        )
        # A text prompt of the same length is chunked as before.
        _admit(config, prompt_len, {})
        return
    _admit(config, prompt_len, image)
    h.add_mm("m", _prompt(1, prompt_len), [_item("img", image_offset, 256)])
    h.run()
    assert h.prefill_spans["m"] == [(0, prompt_len)]
    assert h.checkpoints == [("m", 3072)]


def test_the_step_limit_counts_draft_slots_and_ends_on_a_kv_block():
    """vLLM takes a request's draft slots out of max_num_batched_tokens, and
    with chunked prefill the limit is a multiple of the KV block size."""
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=5000,
            max_num_scheduled_tokens=None,
            enable_chunked_prefill=True,
        ),
        speculative_config=SimpleNamespace(max_num_new_slots_for_drafting=3),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
    )
    assert multimodal_prefill_limit(config) == (
        4096,
        "max_num_batched_tokens=5000 less 3 draft slots, rounded down to a "
        "multiple of the 1024-token KV block",
    )
    config.scheduler_config.max_num_scheduled_tokens = 2048
    assert multimodal_prefill_limit(config) == (2048, "max_num_scheduled_tokens=2048")
    config.scheduler_config.enable_chunked_prefill = False
    config.scheduler_config.max_num_scheduled_tokens = None
    assert multimodal_prefill_limit(config)[0] == 4997


def test_a_model_without_mamba_spec_admits_long_multimodal_prompts():
    h = _MultimodalHarness(num_blocks=64, max_num_batched_tokens=STEP, mode="align")
    config = h.scheduler.vllm_config
    store_tt_mamba_spec_layers(config, False)
    _admit(config, 10_000, {"image": [PlaceholderRange(offset=4000, length=256)]})


# endregion

# region Scheduling


@pytest.mark.parametrize("mode", ["none", "align"])
def test_multimodal_prompts_are_prefilled_whole_while_text_prompts_chunk(mode):
    """With long_prefill_token_threshold 1024, the text prompt runs in chunks
    of at most 1024 tokens; each multimodal prompt, one of them at the step
    limit and one a long prompt with a small image, runs in one step past the
    threshold, after the step it shares with a text chunk."""
    h = _MultimodalHarness(
        num_blocks=64,
        max_num_batched_tokens=STEP,
        mode=mode,
        checkpoint_alignment=BLOCK_SIZE if mode == "align" else None,
    )
    h.scheduler.scheduler_config.long_prefill_token_threshold = 1024
    h.add("t", 6000)
    h.add_mm("m", _prompt(2, STEP), [_item("img-m", 3000, 256)])
    h.add_mm("s", _prompt(3, 3900), [_item("img-s", 2000, 64)])
    h.run()

    assert h.prefill_spans["m"] == [(0, STEP)]
    assert h.prefill_spans["s"] == [(0, 3900)]
    assert len(h.prefill_spans["t"]) >= 6
    assert max(b - a for a, b in h.prefill_spans["t"]) <= 1024
    assert h.row_pixels["m"] == [["img-m"]]
    assert h.row_pixels["s"] == [["img-s"]]
    if mode == "align":
        assert ("m", 3072) in h.checkpoints and ("s", 3072) in h.checkpoints
        assert h.cached_states_checked


def test_text_requests_reuse_the_text_before_an_image_and_mm_requests_never_read():
    """A multimodal request writes its prefix-cache blocks but reads none.

    ``m1`` has its image after its prefill checkpoint at 3072, so vLLM keeps
    the state after 3072 text tokens under a text-only hash: a text request
    with the same 3072 leading tokens starts from it, and the ledger finds the
    state of exactly those tokens there. ``m2`` has the same tokens and a
    different image, ``m3`` the same tokens and the same image: both start at
    position 0.
    """
    h = _MultimodalHarness(
        num_blocks=64,
        max_num_batched_tokens=STEP,
        mode="align",
        checkpoint_alignment=BLOCK_SIZE,
    )
    prompt = _prompt(4, 3400)
    h.add_mm("m1", prompt, [_item("img-a", 3100, 256)])
    h.run()
    h.add("t", prompt=prompt[:3072] + _prompt(5, 500))
    h.add_mm("m2", list(prompt), [_item("img-b", 3100, 256)])
    h.add_mm("m3", list(prompt), [_item("img-a", 3100, 256)])
    h.run()

    assert h.hits == {"t": [3072]}
    for req_id in ("m1", "m2", "m3"):
        assert h.prefill_spans[req_id] == [(0, 3400)]
        assert h.requests[req_id].skip_reading_prefix_cache
    assert not h.requests["t"].skip_reading_prefix_cache
    assert h.cached_states_checked


def test_different_media_with_identical_text_never_share_a_state():
    """Two requests with identical tokens and different images: the block
    hashes differ from the first block that holds an image on, and the second
    request starts at position 0."""
    h = _MultimodalHarness(
        num_blocks=64,
        max_num_batched_tokens=STEP,
        mode="align",
        checkpoint_alignment=BLOCK_SIZE,
    )
    prompt = _prompt(6, 3000)
    h.add_mm("a", list(prompt), [_item("img-a", 1500, 256)])
    h.run()
    h.add_mm("b", list(prompt), [_item("img-b", 1500, 256)])
    h.run()

    hashes_a, hashes_b = h.requests["a"].block_hashes, h.requests["b"].block_hashes
    assert hashes_a[0] == hashes_b[0]
    assert hashes_a[1] != hashes_b[1]
    assert not h.hits
    assert h.prefill_spans["b"] == [(0, 3000)]


def test_multiple_items_and_a_full_encoder_budget_make_a_request_wait():
    """``m1`` carries an image and a video (556 encoder tokens of a 600-token
    budget). ``m2``'s image does not fit the rest of that step's budget, and
    vLLM would end ``m2``'s chunk before its image; it waits for the next step
    and is then prefilled whole."""
    h = _MultimodalHarness(
        num_blocks=64,
        max_num_batched_tokens=STEP,
        mode="align",
        checkpoint_alignment=BLOCK_SIZE,
        encoder_budget=600,
    )
    h.add_mm(
        "m1",
        _prompt(7, 2000),
        [_item("img-1", 100, 256), _item("vid-1", 800, 300, modality="video")],
    )
    h.add_mm("m2", _prompt(8, 1500), [_item("img-2", 200, 256)])
    h.run()

    assert h.prefill_spans["m1"] == [(0, 2000)]
    assert h.prefill_spans["m2"] == [(0, 1500)]
    assert h.first_step["m2"] > h.first_step["m1"]
    assert h.row_pixels["m1"] == [["img-1", "vid-1"]]


@pytest.mark.parametrize("mode", ["none", "align"])
def test_a_preempted_multimodal_request_replays_from_zero_with_its_pixels(mode):
    """Four 1000-token multimodal requests in a 15-block pool decode past
    1024 tokens, and vLLM preempts some. A preempted request resumes at
    position 0 with every item's pixels. Its replay (prompt and output tokens)
    outgrows the 1024-token step, so its first chunk ends at 1024, after its
    image, and the rest follows as text."""
    h = _MultimodalHarness(num_blocks=15, max_num_batched_tokens=BLOCK_SIZE, mode=mode)
    for i in range(MAX_NUM_SEQS):
        h.add_mm(
            f"m{i}",
            _prompt(10 + i, 1000),
            [_item(f"img-{i}", 100, 256)],
            max_tokens=60 + 10 * i,
        )
    h.run(max_steps=2000)

    assert not h.scheduler.requests
    assert h.preempted and h.preempted <= h.resumed_from_zero
    assert not h.hits
    continued = False
    for req_id in h.preempted:
        spans = h.prefill_spans[req_id]
        assert spans[0] == (0, 1000)
        # A replay (from 0, possibly preempted again) ends its first chunk at
        # 1024; its later chunks are text after the image.
        for start, end in spans[1:]:
            assert (start, end) == (0, 1024) or start >= 1024, spans
            continued |= start == 1024
        assert all(ids == [f"img-{req_id[1:]}"] for ids in h.row_pixels[req_id])
    assert continued, "no replay continued after its first chunk"


# endregion

# region Models without MambaSpec layers


def _attention_only_scheduler(encoder_budget: int):
    """A real ``TTScheduler`` of a model with one attention group."""
    from tests.test_mamba_kv_cache import _vllm_config

    attention = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
    )
    specs = {f"model.layers.{i}.self_attn": attention for i in range(4)}
    scheduler = _scheduler(_kv_cache_config(_vllm_config(), specs, 64), STEP)
    scheduler.max_num_encoder_input_tokens = encoder_budget
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_budget)
    return scheduler


def _request(req_id: str, prompt_len: int, items) -> Request:
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=EOS)
    return Request(
        request_id=req_id,
        prompt_token_ids=_prompt(20, prompt_len),
        sampling_params=params,
        pooling_params=None,
        mm_features=items,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def test_multimodal_requests_of_a_model_without_mamba_spec_are_unchanged():
    """Without MambaSpec layers vLLM's rules stay: the request may read the
    prefix cache, a prompt longer than a step is chunked, and a full encoder
    budget ends the chunk before the item."""
    scheduler = _attention_only_scheduler(encoder_budget=200)
    request = _request("m", 6000, [_item("img", 1000, 256)])
    scheduler.add_request(request)
    out = scheduler.schedule()

    assert not request.skip_reading_prefix_cache
    assert out.num_scheduled_tokens == {"m": 1000}
