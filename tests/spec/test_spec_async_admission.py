# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A speculating TT launch survives real ``VllmConfig`` construction with
asynchronous scheduling still enabled.

Every other test in this directory calls ``TTPlatform.check_and_update_config``
directly, against a ``SimpleNamespace`` config. That cannot answer the question
this file exists for, because the decision under test is not the plugin's: vLLM
resolves asynchronous scheduling against the speculative method name inside
``VllmConfig.__post_init__``, and it does so before calling the platform hook.
An explicitly requested ``--async-scheduling`` raises for any method outside
EAGLE/MTP/draft_model/NGram GPU/DSpark; the automatic selection rewrites the
setting to ``False`` for the same set. ``custom_class`` is the only name
upstream accepts for a proposer it does not own, and it is therefore the name
the plugin's model-owned drafter has to use.

``_install_tt_async_spec_method_patch`` rebinds the one module-level name both
predicates read, so those two conditions admit the TT model-owned drafter and
nothing else changes. These tests build real ``ModelConfig``, ``CacheConfig``,
``SchedulerConfig``, ``SpeculativeConfig`` and ``VllmConfig`` objects so that
the gate, the patch and the TT hook all run for real, and they assert the
setting that survives rather than the call that was made.

The model is a registered architecture rather than a monkeypatched resolver,
again because the resolution has to be the real one: ``ModelConfig`` reads the
architecture from a checkpoint-shaped directory the test writes, and the
registry maps it to a stand-in that declares the capabilities the pairing
needs.
"""

from __future__ import annotations

import json
from typing import get_args

import pytest

# vLLM's bootstrap resolves the platform plugin, which imports plugin modules.
import vllm  # noqa: F401

from tests.spec.fake_spec_model import FakeSpecModel
from vllm_tt_plugin.config import get_tt_spec_plan
from vllm_tt_plugin.platform import (
    _install_tt_async_spec_method_patch,
    _uninstall_tt_async_spec_method_patch,
)
from vllm_tt_plugin.spec_admission import (
    MODEL_OWNED_DRAFT_METHOD,
    MODEL_OWNED_DRAFT_SENTINEL,
)

ARCHITECTURE = "TTAsyncAdmissionSpecModel"
REGISTERED_TARGET = "tests.spec.test_spec_async_admission:AsyncCapableSpecModel"
DRAFT_LEN = 3


class AsyncCapableSpecModel(FakeSpecModel):
    """A stand-in that declares everything the pairing needs of a model.

    Both async declarations, because the plugin's admission reads them
    separately: ``supports_async_decode`` is about the ordinary decode's split
    submission and readback, and ``supports_async_spec_decode`` is about the
    deferred verify. A concurrency above one, because the base stand-in refuses
    speculation past a single request and this file is not testing that.
    """

    max_supported_num_seqs = 8
    model_capabilities = {
        **FakeSpecModel.model_capabilities,
        "supports_async_decode": True,
        "supports_async_spec_decode": True,
    }

    # vLLM's registry classifies an architecture by inspecting the class for
    # these four members, and a class it cannot place is not a generative one,
    # which would send the automatic async-scheduling selection down its
    # pooling branch and have this file testing that instead. None of them is
    # ever called: the TT worker loads through the plugin's own loader.
    def __init__(self, *, vllm_config=None, prefix=""):  # noqa: D107
        del vllm_config, prefix
        super().__init__()

    def embed_input_ids(self, input_ids):
        raise NotImplementedError("the TT worker does not call this")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("the TT worker does not call this")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("the TT worker does not call this")


class NoAsyncSpecDecodeModel(AsyncCapableSpecModel):
    """Declares the ordinary async decode and not the speculative one."""

    model_capabilities = {
        **AsyncCapableSpecModel.model_capabilities,
        "supports_async_spec_decode": False,
    }


class NoAsyncDecodeModel(AsyncCapableSpecModel):
    """Declares the speculative async decode and not the ordinary one."""

    model_capabilities = {
        **AsyncCapableSpecModel.model_capabilities,
        "supports_async_decode": False,
    }


@pytest.fixture(scope="module", autouse=True)
def registered_architecture():
    """Map the checkpoint's architecture to the stand-in, once.

    Registered by dotted path rather than by class object, which is the form
    the plugin's own ``register_tt_models`` uses and the only one
    ``ModelRegistry`` accepts for a class it must import lazily.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, REGISTERED_TARGET)
    yield


@pytest.fixture(autouse=True)
def unpatched_gate():
    """Each test installs the patch itself, so the default is upstream's gate.

    Restored afterwards as well: the patch is process-wide, and a test that
    left it installed would make the next one pass for the wrong reason.
    """
    _uninstall_tt_async_spec_method_patch()
    yield
    _uninstall_tt_async_spec_method_patch()


@pytest.fixture
def checkpoint(tmp_path):
    """A directory shaped like a checkpoint, for the real ``ModelConfig``.

    Small on purpose: nothing here is loaded, and the fields are the ones
    ``ModelConfig`` reads to resolve an architecture and a context length.
    """
    return _write_checkpoint(tmp_path, ARCHITECTURE)


def _write_checkpoint(directory, architecture):
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": [architecture],
                "model_type": "llama",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "vocab_size": 512,
                "max_position_embeddings": 2048,
                "torch_dtype": "bfloat16",
            }
        )
    )
    return directory


def _register(checkpoint, architecture, class_name):
    """Point the checkpoint at another stand-in, registered under its own name.

    A capability declaration is a class attribute, so a test that wants a
    different declaration needs a different class, a different architecture
    name for the registry, and a checkpoint naming it.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    if architecture not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            architecture, f"tests.spec.test_spec_async_admission:{class_name}"
        )
    return _write_checkpoint(checkpoint, architecture)


def _model_config(checkpoint):
    from vllm.config import ModelConfig

    return ModelConfig(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        skip_tokenizer_init=True,
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
        max_model_len=2048,
        enforce_eager=True,
        # Stated rather than inferred. vLLM's registry classifies a stand-in it
        # cannot introspect as a pooling model, and the automatic
        # async-scheduling selection disables itself for pooling, so an
        # inferred runner would have this file testing that instead of the
        # speculative-method gate. A TT serving launch is a generate launch.
        runner="generate",
    )


def _build_config(
    checkpoint,
    *,
    async_scheduling,
    method=MODEL_OWNED_DRAFT_METHOD,
    model=MODEL_OWNED_DRAFT_SENTINEL,
    num_speculative_tokens=DRAFT_LEN,
    speculative=True,
    **speculative_overrides,
):
    """Construct a real ``VllmConfig``, running the real gate and the TT hook."""
    from vllm.config import CacheConfig, ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.config.speculative import SpeculativeConfig

    model_config = _model_config(checkpoint)
    speculative_config = None
    if speculative:
        speculative_config = SpeculativeConfig(
            target_model_config=model_config,
            target_parallel_config=ParallelConfig(),
            method=method,
            model=model,
            num_speculative_tokens=num_speculative_tokens,
            **speculative_overrides,
        )
    return VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=32, gpu_memory_utilization=0.9, cache_dtype="auto"
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_model_len=2048,
            is_encoder_decoder=False,
            async_scheduling=async_scheduling,
        ),
        speculative_config=speculative_config,
        additional_config={"tt": {}},
    )


# region The gate this patch exists for


def test_the_unpatched_gate_refuses_an_explicitly_async_speculating_launch(checkpoint):
    """Upstream's refusal, reproduced, so the tests below prove causation.

    Without this, every assertion that follows could be passing because the
    gate never objected in the first place.
    """
    with pytest.raises(Exception) as excinfo:
        _build_config(checkpoint, async_scheduling=True)

    assert "async scheduling is only supported" in str(excinfo.value)


def test_the_unpatched_gate_disables_async_for_the_automatic_selection(checkpoint):
    """The quieter half: no refusal, and asynchronous scheduling turned off."""
    config = _build_config(checkpoint, async_scheduling=None)

    assert config.scheduler_config.async_scheduling is False


# endregion The gate this patch exists for

# region The supported launch


def test_the_patched_launch_keeps_explicit_async_and_publishes_its_plan(checkpoint):
    """The whole point: the setting survives and the TT worker gets a plan.

    Three things are asserted together because any one of them alone would be
    a launch that looks right and serves something else: asynchronous
    scheduling still on, the speculative plan resolved and published for the
    runner to read, and the TT scheduler in place.
    """
    _install_tt_async_spec_method_patch()

    config = _build_config(checkpoint, async_scheduling=True)

    assert config.scheduler_config.async_scheduling is True
    plan = get_tt_spec_plan(config)
    assert plan is not None
    assert plan.effective_k == DRAFT_LEN
    assert config.scheduler_config.scheduler_cls == (
        "vllm_tt_plugin.scheduler.TTScheduler"
    )


def test_the_patched_launch_keeps_async_through_the_automatic_selection(checkpoint):
    """An operator who says nothing about async gets it, as upstream intends."""
    _install_tt_async_spec_method_patch()

    config = _build_config(checkpoint, async_scheduling=None)

    assert config.scheduler_config.async_scheduling is True
    assert get_tt_spec_plan(config) is not None


def test_the_method_and_the_sentinel_survive_construction(checkpoint):
    """Neither is rewritten on the way through.

    The patch admits the name; it must not change it. A launch whose method
    came out as something else would be driving a different proposer, and the
    sentinel is what says nothing is meant to be imported from that path.
    """
    _install_tt_async_spec_method_patch()

    config = _build_config(checkpoint, async_scheduling=True)

    assert config.speculative_config.method == MODEL_OWNED_DRAFT_METHOD
    assert config.speculative_config.model == MODEL_OWNED_DRAFT_SENTINEL


def test_explicitly_disabled_async_stays_disabled(checkpoint):
    """``--no-async-scheduling`` is the operator's, and the patch leaves it."""
    _install_tt_async_spec_method_patch()

    config = _build_config(checkpoint, async_scheduling=False)

    assert config.scheduler_config.async_scheduling is False
    assert get_tt_spec_plan(config) is not None


# endregion The supported launch

# region What the patch must not admit


def test_a_model_not_declaring_the_speculative_async_path_is_refused(checkpoint):
    """Past upstream's gate, and into the plugin's own refusal.

    This is the division of labour the patch creates: upstream stops deciding
    which TT model may serve the pairing, and the plugin decides it by
    capability, with a message naming the missing declaration.
    """
    _install_tt_async_spec_method_patch()
    checkpoint = _register(
        checkpoint, "TTAsyncAdmissionNoSpecAsync", "NoAsyncSpecDecodeModel"
    )

    with pytest.raises(Exception) as excinfo:
        _build_config(checkpoint, async_scheduling=True)

    assert "supports_async_spec_decode" in str(excinfo.value)


def test_a_model_not_declaring_the_ordinary_async_path_decodes_synchronously(
    checkpoint,
):
    """The other declaration, and it is not interchangeable with the first.

    ``supports_async_decode`` is about the ordinary decode's split submission
    and readback. Absent, the plugin clears asynchronous scheduling for the
    launch rather than refusing it, exactly as it does for a model that never
    speculates, and the speculative plan is still resolved: the launch serves,
    synchronously.
    """
    _install_tt_async_spec_method_patch()
    checkpoint = _register(
        checkpoint, "TTAsyncAdmissionNoAsyncDecode", "NoAsyncDecodeModel"
    )

    config = _build_config(checkpoint, async_scheduling=None)

    assert config.scheduler_config.async_scheduling is False
    assert get_tt_spec_plan(config) is not None


def test_a_launch_with_a_foreign_proposer_path_is_still_refused(checkpoint):
    """The sentinel is what identifies the TT model-owned drafter.

    The rebind cannot tell one ``custom_class`` launch from another, and does
    not have to: a launch naming any other proposer path reaches the plugin's
    admission and is refused there, by name.
    """
    _install_tt_async_spec_method_patch()

    with pytest.raises(Exception) as excinfo:
        _build_config(
            checkpoint,
            async_scheduling=True,
            model="some.other.Proposer",
        )

    message = str(excinfo.value)
    assert MODEL_OWNED_DRAFT_SENTINEL in message


def test_an_unrelated_speculative_method_keeps_upstream_behaviour(checkpoint):
    """The patch widens one name by one value, and that value only.

    ``ngram`` is a host proposer upstream also excludes from asynchronous
    scheduling, and it must still be excluded: a patch that widened the gate
    generally would show up here.
    """
    _install_tt_async_spec_method_patch()

    with pytest.raises(Exception) as excinfo:
        _build_config(
            checkpoint,
            async_scheduling=True,
            method="ngram",
            model=None,
            prompt_lookup_max=4,
        )

    assert "async scheduling is only supported" in str(excinfo.value)


def test_the_padded_drafter_refusal_still_applies(checkpoint):
    """Upstream's other async-and-speculation condition is untouched."""
    _install_tt_async_spec_method_patch()

    with pytest.raises(Exception) as excinfo:
        _build_config(
            checkpoint,
            async_scheduling=True,
            disable_padded_drafter_batch=True,
        )

    assert "disable_padded_drafter_batch" in str(excinfo.value)


# endregion What the patch must not admit

# region The patch itself


def test_installing_twice_changes_nothing(checkpoint):
    """Called from two hooks and on every re-entry into configuration."""
    import vllm.config.vllm as vllm_config_module

    _install_tt_async_spec_method_patch()
    once = get_args(vllm_config_module.EagleModelTypes)
    _install_tt_async_spec_method_patch()

    assert get_args(vllm_config_module.EagleModelTypes) == once
    assert once.count(MODEL_OWNED_DRAFT_METHOD) == 1


def test_the_patch_widens_the_gate_and_nothing_else():
    """One value added, in one module, and the original kept for restoration."""
    import vllm.config.speculative as upstream
    import vllm.config.vllm as vllm_config_module

    before = get_args(vllm_config_module.EagleModelTypes)
    _install_tt_async_spec_method_patch()
    after = get_args(vllm_config_module.EagleModelTypes)

    assert set(after) - set(before) == {MODEL_OWNED_DRAFT_METHOD}
    # The type every other consumer imports is untouched: they bind it in their
    # own namespaces, which is why rebinding one module's name is enough.
    assert MODEL_OWNED_DRAFT_METHOD not in get_args(upstream.EagleModelTypes)

    _uninstall_tt_async_spec_method_patch()
    assert get_args(vllm_config_module.EagleModelTypes) == before


def test_a_restructured_upstream_gate_fails_the_launch(monkeypatch):
    """The failure this patch must never have is the silent one.

    If upstream stops routing the decision through the name the patch rebinds,
    the rebind still applies and asynchronous scheduling goes quietly back to
    disabled. So installation checks the shape of the gate and refuses, naming
    what it expected.
    """
    from vllm_tt_plugin import platform as platform_module

    monkeypatch.setattr(platform_module, "_ASYNC_SPEC_GATE_COMPARISONS", 99)

    with pytest.raises(RuntimeError) as excinfo:
        _install_tt_async_spec_method_patch()

    message = str(excinfo.value)
    assert "restructured" in message
    assert "_install_tt_async_spec_method_patch" in message


# endregion The patch itself
