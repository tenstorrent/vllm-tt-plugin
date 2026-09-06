# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Restricting the modality limits the plugin advertises to vLLM.

Upstream refuses a modality absent from ``get_supported_mm_limits`` with a 4xx,
so the platform intersects the model's declaration with
``SUPPORTED_MM_MODALITIES`` before vLLM reads it. See issue #112.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

# Let vLLM finish its own import before the plugin module pulls it in again.
import vllm  # noqa: F401

from vllm_tt_plugin.platform import _restrict_advertised_mm_modalities


@dataclass
class _Factories:
    """Stands in for vLLM's ``_ProcessorFactories`` (a frozen dataclass)."""

    info: type
    dummy_inputs: str = "dummy"
    processor: str = "processor"


def _model_cls(limits):
    class _Info:
        def get_supported_mm_limits(self):
            return dict(limits)

    class _Model:
        _processor_factory = _Factories(info=_Info)

    return _Model


def _config(is_multimodal=True):
    return SimpleNamespace(
        model_config=SimpleNamespace(is_multimodal_model=is_multimodal)
    )


@pytest.fixture
def registry(monkeypatch):
    """Point the registry lookup at whichever class a test installs."""
    import vllm.multimodal as mm

    holder = SimpleNamespace(model_cls=None)
    monkeypatch.setattr(
        mm.MULTIMODAL_REGISTRY,
        "_get_model_cls",
        lambda model_config: holder.model_cls,
        raising=False,
    )
    return holder


def _advertised(model_cls):
    return model_cls._processor_factory.info().get_supported_mm_limits()


def test_unsupported_modality_is_dropped(registry):
    registry.model_cls = _model_cls({"image": 1, "video": 1})

    _restrict_advertised_mm_modalities(_config())

    # Absent, not zero: upstream reads a missing key as unsupported.
    assert _advertised(registry.model_cls) == {"image": 1}


def test_supported_limits_pass_through_untouched(registry):
    registry.model_cls = _model_cls({"image": 10})

    _restrict_advertised_mm_modalities(_config())

    assert _advertised(registry.model_cls) == {"image": 10}


def test_a_model_declaring_nothing_servable_ends_up_empty(registry):
    registry.model_cls = _model_cls({"video": 1, "audio": 2})

    _restrict_advertised_mm_modalities(_config())

    assert _advertised(registry.model_cls) == {}


def test_wrapping_is_idempotent(registry):
    # check_and_update_config re-runs on the same class when the engine
    # rebuilds its config in-process; the registry keeps one factory per class.
    registry.model_cls = _model_cls({"image": 1, "video": 1})

    _restrict_advertised_mm_modalities(_config())
    wrapped_once = registry.model_cls._processor_factory.info
    _restrict_advertised_mm_modalities(_config())

    assert registry.model_cls._processor_factory.info is wrapped_once
    assert _advertised(registry.model_cls) == {"image": 1}


def test_a_shared_info_class_is_left_intact(registry):
    # tt-metal registers upstream classes directly (Mistral3ProcessingInfo,
    # Gemma3ProcessingInfo), and TT_Qwen3VLProcessingInfo backs two models.
    # Restricting one model must not reach the class or its other users.
    shared = _model_cls({"image": 1, "video": 1})._processor_factory.info

    class Owner:
        _processor_factory = _Factories(info=shared)

    class Other:
        _processor_factory = _Factories(info=shared)

    registry.model_cls = Owner
    _restrict_advertised_mm_modalities(_config())

    assert _advertised(Owner) == {"image": 1}
    assert _advertised(Other) == {"image": 1, "video": 1}
    assert shared().get_supported_mm_limits() == {"image": 1, "video": 1}


def test_wrap_keeps_the_identity_processor_builders_report(registry):
    # _build_llava_or_pixtral_hf_processor and friends dispatch on
    # isinstance(info, ...) and raise NotImplementedError(type(info)).
    base = _model_cls({"image": 1, "video": 1})._processor_factory.info
    registry.model_cls = _model_cls({"image": 1, "video": 1})
    registry.model_cls._processor_factory = _Factories(info=base)

    _restrict_advertised_mm_modalities(_config())

    wrapped = registry.model_cls._processor_factory.info
    assert isinstance(wrapped(), base)
    for attr in ("__name__", "__qualname__", "__module__", "__doc__"):
        assert getattr(wrapped, attr) == getattr(base, attr)


def test_wrap_does_not_stack_mro_levels(registry):
    registry.model_cls = _model_cls({"image": 1, "video": 1})
    depth = len(registry.model_cls._processor_factory.info.__mro__)

    for _ in range(5):
        _restrict_advertised_mm_modalities(_config())

    assert len(registry.model_cls._processor_factory.info.__mro__) == depth + 1


def test_other_factory_fields_survive_the_wrap(registry):
    registry.model_cls = _model_cls({"image": 1, "video": 1})

    _restrict_advertised_mm_modalities(_config())

    factories = registry.model_cls._processor_factory
    assert factories.dummy_inputs == "dummy"
    assert factories.processor == "processor"


def test_text_only_model_is_left_alone(registry):
    registry.model_cls = _model_cls({"image": 1, "video": 1})
    original = registry.model_cls._processor_factory

    _restrict_advertised_mm_modalities(_config(is_multimodal=False))

    assert registry.model_cls._processor_factory is original


def test_unresolvable_model_is_left_alone(monkeypatch):
    import vllm.multimodal as mm

    def _raise(model_config):
        raise ValueError("no architecture")

    monkeypatch.setattr(mm.MULTIMODAL_REGISTRY, "_get_model_cls", _raise, raising=False)

    # The arch check in check_and_update_config reports resolution failures;
    # this must not turn them into a different exception.
    _restrict_advertised_mm_modalities(_config())
