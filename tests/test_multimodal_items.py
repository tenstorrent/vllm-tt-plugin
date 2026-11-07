# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A multimodal model's items reach ``prefill_forward``, and only its modalities.

A model that declares ``model_capabilities["mm_modalities"]`` gets ``mm_items``
with every prefill: one entry per row, ``None`` for a text-only request, else the
request's items in prompt order with their placeholder ranges and processor
fields. Every other modality its vLLM processor accepts is limited to 0 items
per prompt, so vLLM refuses such a request while parsing it (HTTP 400). A
multimodal model without the declaration keeps the image-only ``pixel_values``
lists, and its processor's other modalities are refused the same way.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.config.multimodal import (
    AudioDummyOptions,
    MultiModalConfig,
    VideoDummyOptions,
)
from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.multimodal.processing.context import BaseProcessingInfo
from vllm.multimodal.utils import strip_covered_mm_data
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.config import get_tt_mm_modalities, resolve_tt_mm_modalities
from vllm_tt_plugin.input_batch import (
    InputBatch,
    TTLaneInputBatch,
    build_cached_request_state,
)
from vllm_tt_plugin.lane_scheduler import TTStepPlan
from vllm_tt_plugin.model_input import build_mm_items
from vllm_tt_plugin.model_runner import TTModelRunner

VOCAB_SIZE = 64
BLOCK_SIZE = 16
MAX_MODEL_LEN = 64
MAX_NUM_REQS = 4
ALL_MODALITIES = frozenset({"image", "video", "audio"})

# A prompt with an image at [4, 12), a video at [14, 24) whose placeholder range
# interleaves text (timestamps, vision start/end) with its embeddings, and an
# audio clip at [26, 30).
PROMPT_LEN = 32
# video start, two timestamp tokens, vision start, four frame embeddings, vision
# end, video end
VIDEO_IS_EMBED = torch.tensor([False] * 4 + [True] * 4 + [False] * 2)


# region Helpers


def _field(data: torch.Tensor) -> MultiModalFieldElem:
    return MultiModalFieldElem(data=data, field=MultiModalSharedField(batch_size=1))


def _feature(modality, offset, length, *, is_embed=None, identifier=None, **fields):
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem({k: _field(v) for k, v in fields.items()}),
        modality=modality,
        identifier=identifier or f"{modality}-{offset}",
        mm_position=PlaceholderRange(offset=offset, length=length, is_embed=is_embed),
    )


def _image(offset=4):
    return _feature(
        "image",
        offset,
        8,
        pixel_values=torch.full((32, 6), 1.0),
        image_grid_thw=torch.tensor([1, 4, 8]),
    )


def _video(offset=14):
    return _feature(
        "video",
        offset,
        10,
        is_embed=VIDEO_IS_EMBED,
        pixel_values_videos=torch.full((12, 6), 2.0),
        video_grid_thw=torch.tensor([1, 2, 6]),
        second_per_grid_ts=torch.tensor(1.0),
    )


def _audio(offset=26):
    return _feature(
        "audio",
        offset,
        4,
        audio_features=torch.full((16, 128), 3.0),
        audio_token_lens=torch.tensor(4),
    )


def _mm_features():
    return [_image(), _video(), _audio()]


class _Model:
    """Records every ``prefill_forward`` call's keywords."""

    def __init__(self):
        self.prefills: list[dict] = []

    def prefill_forward(self, **kwargs):
        self.prefills.append(kwargs)
        return object()


def _bind_runner_methods(runner: SimpleNamespace) -> SimpleNamespace:
    for name, member in vars(TTModelRunner).items():
        if hasattr(runner, name):
            continue
        if isinstance(member, staticmethod):
            setattr(runner, name, member.__func__)
        elif isinstance(member, classmethod):
            setattr(runner, name, member.__func__.__get__(TTModelRunner))
        elif inspect.isfunction(member):
            setattr(runner, name, member.__get__(runner))
    return runner


def _runner(mm_modalities, *, num_speculative_tokens=0) -> SimpleNamespace:
    """A front-packed runner fake; its lifecycle and prefill methods are real."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    runner = SimpleNamespace(
        model=_Model(),
        input_batch=batch,
        requests={},
        encoder_cache={},
        kv_caches=object(),
        trace_mode="none",
        request_specific_rope=False,
        async_decode=SimpleNamespace(note_prefill_submitted=lambda: None),
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        _num_speculative_tokens=num_speculative_tokens,
        async_decode_scheduling=False,
        _spec_method=None,
        _spec_supports_narrow_decode=False,
        _spec_drafts_from_model=False,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        _ngram_proposer=None,
        _req_state_slot={},
        vllm_config=SimpleNamespace(
            speculative_config=None,
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
            scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_REQS),
        ),
        tt_per_lane_max_num_seqs=MAX_NUM_REQS,
        tt_data_parallel_size=1,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        supports_stage_inputs=False,
        supports_stage_outputs=False,
        mm_modalities=mm_modalities,
        model_config=SimpleNamespace(
            is_multimodal_model=True, max_model_len=MAX_MODEL_LEN
        ),
        check_perform_device_sampling=lambda **_: False,
        _block_tables_per_layer=lambda _: None,
        _decode_state_slot_remap=lambda row_req_ids: None,
        _sampling_params_for_padded_decode=lambda params, req_indices, n: params,
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
    )
    return _bind_runner_methods(runner)


def _new_request(req_id, mm_features=(), num_computed_tokens=0) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(range(PROMPT_LEN)),
        mm_features=list(mm_features),
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=([0, 1],),
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
    )


def _cached(req_ids, num_computed_tokens) -> CachedRequestData:
    return CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in req_ids],
        all_token_ids={},
        new_block_ids=[None for _ in req_ids],
        num_computed_tokens=list(num_computed_tokens),
        num_output_tokens=[0 for _ in req_ids],
    )


def _step(*, new=(), cached=None, scheduled=None) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = list(new)
    if cached is not None:
        output.scheduled_cached_reqs = cached
    output.num_scheduled_tokens = dict(scheduled or {})
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    return output


def _run_prefill(runner, scheduler_output) -> dict:
    """Apply the step and submit its prefill; return the model's keywords."""
    runner._update_states(scheduler_output)
    model_input = runner._prepare_model_inputs(scheduler_output, None)
    assert model_input is not None
    runner.submit_prefill(model_input, [model_input.unpadded_batch_size])
    return runner.model.prefills[-1]


def _rows(runner, kwargs) -> dict:
    return dict(zip(runner.input_batch.req_ids, kwargs["mm_items"]))


def _assert_full_prompt_items(items):
    """The image, video and audio of ``_mm_features``, whole, in prompt order."""
    assert [item["modality"] for item in items] == ["image", "video", "audio"]
    assert [(item["offset"], item["length"]) for item in items] == [
        (4, 8),
        (14, 10),
        (26, 4),
    ]
    assert [item["identifier"] for item in items] == ["image-4", "video-14", "audio-26"]
    image, video, audio = items
    assert image["is_embed"] is None, "every image placeholder takes an embedding"
    assert torch.equal(video["is_embed"], VIDEO_IS_EMBED)
    assert audio["is_embed"] is None
    assert set(image["data"]) == {"pixel_values", "image_grid_thw"}
    assert torch.equal(image["data"]["image_grid_thw"], torch.tensor([1, 4, 8]))
    assert set(video["data"]) == {
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    }
    assert torch.equal(audio["data"]["audio_features"], torch.full((16, 128), 3.0))


class _ProcessingInfo(BaseProcessingInfo):
    """vLLM's own item-count validation over a processor of these modalities."""

    def __init__(self, mm_config: MultiModalConfig, supported=ALL_MODALITIES):
        super().__init__(SimpleNamespace(get_mm_config=lambda: mm_config))
        self._supported = supported

    def get_supported_mm_limits(self):
        return dict.fromkeys(sorted(self._supported))


def _model_class(capabilities=None, *, has_processor=True) -> type:
    attrs = {"__module__": "tests.fake_models"}
    if capabilities is not None:
        attrs["model_capabilities"] = capabilities
    if has_processor:
        # What MULTIMODAL_REGISTRY.register_processor sets on the class.
        attrs["_processor_factory"] = object()
    return type("FakeOmniForCausalLM", (), attrs)


def _vllm_config(mm_config: MultiModalConfig | None) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(multimodal_config=mm_config),
    )


@pytest.fixture
def processor_modalities(monkeypatch: pytest.MonkeyPatch):
    """Serve ``MULTIMODAL_REGISTRY.get_processing_info`` from a fake processor."""
    supported = {"modalities": ALL_MODALITIES}

    def get_processing_info(model_config):
        return _ProcessingInfo(model_config.multimodal_config, supported["modalities"])

    monkeypatch.setattr(MULTIMODAL_REGISTRY, "get_processing_info", get_processing_info)
    return supported


# endregion Helpers

# region Prefill delivery


def test_every_prefill_row_gets_its_requests_items_in_prompt_order():
    runner = _runner(ALL_MODALITIES)

    kwargs = _run_prefill(
        runner,
        _step(
            new=[_new_request("mm", _mm_features()), _new_request("text")],
            scheduled={"mm": PROMPT_LEN, "text": PROMPT_LEN},
        ),
    )

    assert "pixel_values" not in kwargs and "image_grid_thw" not in kwargs
    rows = _rows(runner, kwargs)
    assert rows["text"] is None, "a text-only request's row is None"
    _assert_full_prompt_items(rows["mm"])


def test_every_chunk_gets_all_items_whatever_part_of_the_prompt_it_covers():
    """The chunk boundaries fall inside the image and between the video and audio;
    each chunk still carries all three items with their whole-prompt ranges."""
    runner = _runner(ALL_MODALITIES)
    seen = []

    kwargs = _run_prefill(
        runner, _step(new=[_new_request("mm", _mm_features())], scheduled={"mm": 8})
    )
    seen.append(kwargs)
    for computed, length in ((8, 17), (25, PROMPT_LEN - 25)):
        seen.append(
            _run_prefill(
                runner,
                _step(cached=_cached(["mm"], [computed]), scheduled={"mm": length}),
            )
        )

    assert [kw["start_pos"].tolist() for kw in seen] == [[0], [8], [25]]
    assert [kw["prompt_lens"].tolist() for kw in seen] == [[8], [25], [PROMPT_LEN]]
    for kw in seen:
        (items,) = kw["mm_items"]
        _assert_full_prompt_items(items)


def test_an_item_prefix_caching_covers_arrives_without_its_data():
    """vLLM strips the fields of an item wholly inside the cached prefix
    (``strip_covered_mm_data``); the item itself still arrives, placed."""
    runner = _runner(ALL_MODALITIES)
    computed = 16  # the image [4, 12) is covered; the video [14, 24) is not
    stripped = strip_covered_mm_data(_mm_features(), computed)

    kwargs = _run_prefill(
        runner,
        _step(
            new=[_new_request("mm", stripped, num_computed_tokens=computed)],
            scheduled={"mm": PROMPT_LEN - computed},
        ),
    )

    (items,) = kwargs["mm_items"]
    assert kwargs["start_pos"].tolist() == [computed]
    assert [item["modality"] for item in items] == ["image", "video", "audio"]
    assert items[0]["data"] is None
    assert (items[0]["offset"], items[0]["length"]) == (4, 8)
    assert items[1]["data"] is not None and items[2]["data"] is not None


def test_a_speculative_launchs_prefill_gets_the_items_too():
    runner = _runner(ALL_MODALITIES, num_speculative_tokens=2)

    kwargs = _run_prefill(
        runner,
        _step(new=[_new_request("mm", _mm_features())], scheduled={"mm": PROMPT_LEN}),
    )

    (items,) = kwargs["mm_items"]
    _assert_full_prompt_items(items)


def _lane_runner(batch, requests, mm_modalities):
    runner = SimpleNamespace(
        input_batch=batch,
        requests=requests,
        mm_modalities=mm_modalities,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        _block_tables_per_layer=lambda _: None,
        check_perform_device_sampling=lambda **_: False,
        model_config=SimpleNamespace(is_multimodal_model=True),
    )
    runner._gather_multi_modal_inputs = (
        TTModelRunner._gather_multi_modal_inputs.__get__(runner)
    )
    return runner


def _lane_request(req_id, mm_features):
    return build_cached_request_state(
        _new_request(req_id, mm_features), supports_stage_inputs=False
    )


@pytest.mark.parametrize("num_lanes", [1, 2], ids=["row-engine", "two-lanes"])
def test_lane_prefill_rows_get_their_requests_items(num_lanes):
    """One-lane lane mode is a row engine; with two lanes each lane's rows are
    the requests placed in it, in plan order."""
    per_lane = 2
    batch = TTLaneInputBatch(
        num_lanes=num_lanes,
        per_lane=per_lane,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * num_lanes * per_lane,
        vocab_size=VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
        logitsprocs=None,
    )
    # A partial chunk of the multimodal request (it ends inside the video) and
    # a text-only request, in the last lane.
    requests = {
        "mm": _lane_request("mm", _mm_features()),
        "text": _lane_request("text", []),
    }
    last_lane_base = (num_lanes - 1) * per_lane
    rows = {"mm": last_lane_base, "text": last_lane_base + 1}
    for req_id, row in rows.items():
        batch.add_request_to_row(requests[req_id], row)
    scheduler_output = _step(scheduled={"mm": 20, "text": PROMPT_LEN})
    per_lane_counts = [0] * num_lanes
    per_lane_counts[-1] = 2
    plan = TTStepPlan(
        is_decode=False,
        capacity=num_lanes * per_lane,
        scheduled_req_ids=("mm", "text"),
        scheduled_rows=(rows["mm"], rows["text"]),
        input_rows=(rows["mm"], rows["text"]),
        req_id_to_row=rows,
        batch_size_per_dp=tuple(per_lane_counts),
        prefill_empty_slots=(rows["mm"], rows["text"]),
    )
    runner = _lane_runner(batch, requests, ALL_MODALITIES)

    model_input = batch.build_model_input(runner, scheduler_output, None, plan)

    assert model_input.prompt_lens.tolist() == [20, PROMPT_LEN]
    assert model_input.intermediate_prefill_mask.tolist() == [True, False]
    assert set(model_input.multi_modal_kwargs) == {"mm_items"}
    mm_row, text_row = model_input.multi_modal_kwargs["mm_items"]
    _assert_full_prompt_items(mm_row)
    assert text_row is None


# endregion Prefill delivery

# region Models without the declaration


def test_a_model_without_the_declaration_keeps_the_pixel_values_lists():
    runner = _runner(None)

    kwargs = _run_prefill(
        runner,
        _step(
            new=[_new_request("image", [_image()]), _new_request("text")],
            scheduled={"image": PROMPT_LEN, "text": PROMPT_LEN},
        ),
    )

    assert "mm_items" not in kwargs
    by_row = dict(zip(runner.input_batch.req_ids, kwargs["pixel_values"]))
    assert by_row["text"] is None
    (pixel_values,) = by_row["image"]
    assert torch.equal(pixel_values, torch.full((32, 6), 1.0))


# endregion Models without the declaration

# region Modality gate


def test_an_undeclared_modality_reaching_the_runner_is_a_broken_invariant():
    with pytest.raises(ValueError, match="'audio' item at prompt offset 26"):
        build_mm_items("r", _mm_features(), frozenset({"image", "video"}))


def test_a_declared_model_refuses_its_processors_other_modalities_at_request_time(
    processor_modalities,
):
    """vLLM's own item-count check refuses them: HTTP 400, not an engine error."""
    mm_config = MultiModalConfig()
    vllm_config = _vllm_config(mm_config)

    resolve_tt_mm_modalities(
        vllm_config, {"mm_modalities": {"image", "audio"}}, _model_class()
    )

    assert get_tt_mm_modalities(vllm_config) == frozenset({"image", "audio"})
    assert mm_config.get_limit_per_prompt("video") == 0
    # The class a processor's dummy-input builder asserts for the modality.
    assert isinstance(mm_config.limit_per_prompt["video"], VideoDummyOptions)
    assert "image" not in mm_config.limit_per_prompt
    assert "audio" not in mm_config.limit_per_prompt

    info = _ProcessingInfo(mm_config)
    info.validate_num_items("image", 3)
    info.validate_num_items("audio", 1)
    with pytest.raises(VLLMValidationError, match="At most 0 video"):
        info.validate_num_items("video", 1)


def test_a_model_declaring_every_processor_modality_is_not_limited(
    processor_modalities,
):
    mm_config = MultiModalConfig(limit_per_prompt={"image": 4})
    vllm_config = _vllm_config(mm_config)

    resolve_tt_mm_modalities(
        vllm_config, {"mm_modalities": ["image", "video", "audio"]}, _model_class()
    )

    assert get_tt_mm_modalities(vllm_config) == ALL_MODALITIES
    assert set(mm_config.limit_per_prompt) == {"image"}
    assert mm_config.get_limit_per_prompt("image") == 4


def test_a_model_without_the_declaration_is_refused_everything_but_images(
    processor_modalities,
):
    """Its prefill takes the image-only pixel_values lists."""
    mm_config = MultiModalConfig()
    vllm_config = _vllm_config(mm_config)

    resolve_tt_mm_modalities(vllm_config, {}, _model_class())

    assert get_tt_mm_modalities(vllm_config) is None
    assert mm_config.get_limit_per_prompt("image") == 999
    assert mm_config.get_limit_per_prompt("video") == 0
    assert isinstance(mm_config.limit_per_prompt["audio"], AudioDummyOptions)
    with pytest.raises(VLLMValidationError, match="At most 0 audio"):
        _ProcessingInfo(mm_config).validate_num_items("audio", 1)


def test_an_operator_limit_of_zero_on_an_unencoded_modality_is_kept(
    processor_modalities,
):
    mm_config = MultiModalConfig(
        limit_per_prompt={"video": {"count": 0, "num_frames": 8}}
    )

    resolve_tt_mm_modalities(
        _vllm_config(mm_config), {"mm_modalities": {"image"}}, _model_class()
    )

    assert mm_config.limit_per_prompt["video"].num_frames == 8


def test_an_operator_limit_allowing_an_unencoded_modality_is_refused(
    processor_modalities,
):
    with pytest.raises(ValueError, match="allows 2 audio item"):
        resolve_tt_mm_modalities(
            _vllm_config(MultiModalConfig(limit_per_prompt={"audio": 2})),
            {"mm_modalities": {"image", "video"}},
            _model_class(),
        )


def test_mm_embeds_are_refused_for_a_declaring_model(processor_modalities):
    with pytest.raises(ValueError, match="--enable-mm-embeds"):
        resolve_tt_mm_modalities(
            _vllm_config(MultiModalConfig(enable_mm_embeds=True)),
            {"mm_modalities": {"image"}},
            _model_class(),
        )


def test_a_declaration_vllm_serves_text_only_is_refused():
    """No registered processor: vLLM runs the model text-only."""
    with pytest.raises(ValueError, match="serves it text-only"):
        resolve_tt_mm_modalities(
            _vllm_config(MultiModalConfig()),
            {"mm_modalities": {"image"}},
            _model_class(has_processor=False),
        )


def test_a_text_only_model_has_nothing_to_gate():
    for mm_config, has_processor in ((None, True), (MultiModalConfig(), False)):
        vllm_config = _vllm_config(mm_config)

        resolve_tt_mm_modalities(
            vllm_config, None, _model_class(has_processor=has_processor)
        )

        assert get_tt_mm_modalities(vllm_config) is None


@pytest.mark.parametrize("declared", ["image", set(), ("image", 3), {"image": True}])
def test_a_malformed_declaration_is_refused(declared, processor_modalities):
    with pytest.raises(ValueError, match="non-empty set of modality names"):
        resolve_tt_mm_modalities(
            _vllm_config(MultiModalConfig()),
            {"mm_modalities": declared},
            _model_class(),
        )


# endregion Modality gate


def _check_and_update_config(monkeypatch, vllm_config, model_class) -> None:
    from vllm_tt_plugin.platform import TTPlatform

    with monkeypatch.context() as m:
        m.setattr(
            "vllm_tt_plugin.platform.register_tt_models", lambda *args, **kwargs: None
        )
        m.setattr(
            "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
            lambda _cfg: None,
        )
        m.setattr(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            lambda: ["TTDummyModel"],
        )
        m.setattr(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            lambda _model_config: (model_class, None),
        )
        TTPlatform.check_and_update_config(vllm_config)


def test_the_platform_resolves_the_declaration_at_config_time(
    monkeypatch: pytest.MonkeyPatch, vllm_config, processor_modalities
):
    mm_config = MultiModalConfig()
    vllm_config.model_config.multimodal_config = mm_config

    _check_and_update_config(
        monkeypatch,
        vllm_config,
        _model_class({"mm_modalities": frozenset({"image", "video"})}),
    )

    assert get_tt_mm_modalities(vllm_config) == frozenset({"image", "video"})
    assert mm_config.get_limit_per_prompt("audio") == 0
