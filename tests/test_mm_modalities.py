# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The transportable-modality set and the runner guard that reads it.

``_gather_multi_modal_inputs`` builds one fixed pair of kwargs
(``pixel_values`` / ``image_grid_thw``), so image is the only modality whose
payload can reach a model. ``SUPPORTED_MM_MODALITIES`` names that fact and
``_validate_mm_feature`` enforces it.

These pin the two together: widening the set without teaching the gather step
to collect the new kwargs would admit a payload the model never receives.
See https://github.com/tenstorrent/vllm-tt-plugin/issues/112.
"""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.config import SUPPORTED_MM_MODALITIES
from vllm_tt_plugin.model_runner import TTModelRunner


def _feature(modality):
    return SimpleNamespace(modality=modality)


def test_supported_set_matches_the_kwargs_the_gather_step_builds():
    # _gather_multi_modal_inputs hard-codes the image pair; any other entry
    # here would be advertised without a transport.
    assert frozenset({"image"}) == SUPPORTED_MM_MODALITIES


def test_validate_accepts_a_supported_modality():
    runner = object.__new__(TTModelRunner)

    for modality in SUPPORTED_MM_MODALITIES:
        runner._validate_mm_feature(_feature(modality))


@pytest.mark.parametrize("modality", ["video", "audio", "image_embeds", ""])
def test_validate_rejects_an_unsupported_modality(modality):
    runner = object.__new__(TTModelRunner)

    with pytest.raises(NotImplementedError):
        runner._validate_mm_feature(_feature(modality))
