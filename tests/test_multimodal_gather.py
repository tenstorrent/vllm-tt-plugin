# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`_gather_multi_modal_inputs` forwards every processor field by name."""

from types import SimpleNamespace

import pytest
import torch

from vllm_tt_plugin.model_runner import TTModelRunner


def _field(t):
    return SimpleNamespace(data=t)


def _feature(item, modality="image"):
    return SimpleNamespace(modality=modality, data=item)


def _runner(requests_by_id, req_ids):
    return SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=req_ids, num_reqs=len(req_ids)),
        requests=requests_by_id,
        _validate_mm_feature=lambda mm_feature: TTModelRunner._validate_mm_feature(
            None, mm_feature
        ),
        _MM_BASE_FIELDS=TTModelRunner._MM_BASE_FIELDS,
        _MM_PLACEHOLDER_FIELDS=TTModelRunner._MM_PLACEHOLDER_FIELDS,
    )


PLACEHOLDER_FIELDS = list(TTModelRunner._MM_PLACEHOLDER_FIELDS)


def test_qwen_shaped_items_keep_the_two_base_keys():
    pv, grid = torch.zeros(4, 3), torch.tensor([1, 2, 2])
    runner = _runner(
        {
            "text": SimpleNamespace(mm_features=[]),
            "img": SimpleNamespace(
                mm_features=[
                    _feature(
                        {"pixel_values": _field(pv), "image_grid_thw": _field(grid)}
                    )
                ]
            ),
        },
        ["text", "img"],
    )
    out = TTModelRunner._gather_multi_modal_inputs(runner)
    assert list(out) == ["pixel_values", "image_grid_thw"] + PLACEHOLDER_FIELDS
    assert out["pixel_values"] == [None, [pv]]
    assert out["image_grid_thw"] == [None, [grid]]


def test_extra_processor_fields_are_forwarded_by_name():
    # Gemma4's vLLM processor emits pixel_position_ids and no image_grid_thw.
    pv0, pos0 = torch.zeros(5, 3), torch.zeros(5, 2, dtype=torch.long)
    pv1, pos1 = torch.zeros(7, 3), torch.zeros(7, 2, dtype=torch.long)
    runner = _runner(
        {
            "a": SimpleNamespace(
                mm_features=[
                    _feature(
                        {
                            "pixel_values": _field(pv0),
                            "pixel_position_ids": _field(pos0),
                        }
                    ),
                    _feature(
                        {
                            "pixel_values": _field(pv1),
                            "pixel_position_ids": _field(pos1),
                        }
                    ),
                ]
            ),
            "b": SimpleNamespace(mm_features=[]),
        },
        ["a", "b"],
    )
    out = TTModelRunner._gather_multi_modal_inputs(runner)
    assert (
        list(out)
        == ["pixel_values", "image_grid_thw", "pixel_position_ids"] + PLACEHOLDER_FIELDS
    )
    assert out["pixel_values"] == [[pv0, pv1], None]
    assert out["image_grid_thw"] == [[None, None], None]
    assert out["pixel_position_ids"] == [[pos0, pos1], None]


def test_req_indices_select_and_order_rows():
    pv = torch.zeros(2, 3)
    runner = _runner(
        {
            "a": SimpleNamespace(mm_features=[]),
            "b": SimpleNamespace(mm_features=[_feature({"pixel_values": _field(pv)})]),
            "c": SimpleNamespace(mm_features=[_feature(None)]),
        },
        ["a", "b", "c"],
    )
    out = TTModelRunner._gather_multi_modal_inputs(runner, req_indices=[2, 1])
    assert out["pixel_values"] == [[None], [pv]]
    assert out["image_grid_thw"] == [[None], [None]]


def test_video_and_audio_items_are_forwarded_like_images():
    af, vf = torch.zeros(5, 640), torch.zeros(2, 4, 3)
    runner = _runner(
        {
            "a": SimpleNamespace(
                mm_features=[_feature({"input_features": _field(af)}, "audio")]
            ),
            "v": SimpleNamespace(
                mm_features=[_feature({"pixel_values_videos": _field(vf)}, "video")]
            ),
        },
        ["a", "v"],
    )
    out = TTModelRunner._gather_multi_modal_inputs(runner)
    assert out["input_features"] == [[af], [None]]
    assert out["pixel_values_videos"] == [[None], [vf]]
    assert out["pixel_values"] == [[None], [None]]


def test_unknown_item_kinds_are_rejected():
    feature = _feature({"points": _field(torch.zeros(1))}, "point_cloud")
    runner = _runner({"p": SimpleNamespace(mm_features=[feature])}, ["p"])
    with pytest.raises(NotImplementedError):
        TTModelRunner._gather_multi_modal_inputs(runner)


def test_each_items_placeholder_range_is_forwarded_after_the_processor_fields():
    pv0, pv1 = torch.zeros(5, 3), torch.zeros(7, 3)
    with_range = _feature({"pixel_values": _field(pv0)})
    with_range.mm_position = SimpleNamespace(offset=12, length=270, is_embed=None)
    stripped = _feature(None)  # inside the cached prefix: payload gone, range kept
    stripped.mm_position = SimpleNamespace(offset=300, length=256, is_embed=None)
    no_range = _feature({"pixel_values": _field(pv1)})
    runner = _runner(
        {
            "a": SimpleNamespace(mm_features=[with_range, stripped, no_range]),
            "b": SimpleNamespace(mm_features=[]),
        },
        ["a", "b"],
    )
    out = TTModelRunner._gather_multi_modal_inputs(runner)
    assert list(out)[-2:] == PLACEHOLDER_FIELDS
    assert out["pixel_values"] == [[pv0, None, pv1], None]
    assert out["mm_placeholder_offsets"] == [[12, 300, None], None]
    assert out["mm_placeholder_lengths"] == [[270, 256, None], None]
