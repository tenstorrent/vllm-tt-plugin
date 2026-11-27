# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``supports_spec_device_penalties``: the declaration check, the
device-sampling decision of a decode step that carries a penalty, and the
certifiability of a penalised request by the id walk.

``tests/test_mamba_spec_async.py`` drives penalised requests through the
scheduler, the runner and a device model end to end, chained verifies
included; ``tests/spec/test_spec_inputs.py`` builds their verify rows.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.spec_admission import (
    spec_device_penalties,
    validate_spec_device_penalties,
)

PENALISED = "penalised"


@pytest.mark.parametrize(
    ("capabilities", "match"),
    [
        ({"supports_spec_device_penalties": 1}, "is a bool"),
        ({"supports_spec_device_penalties": True}, "without spec_ids_are_sampled"),
        (
            {"supports_spec_device_penalties": True, "spec_ids_are_sampled": False},
            "without spec_ids_are_sampled",
        ),
    ],
)
def test_a_declaration_needs_the_sampled_ids(capabilities, match):
    with pytest.raises(ValueError, match=match):
        validate_spec_device_penalties(capabilities, "M")


@pytest.mark.parametrize(
    "capabilities",
    [
        None,
        {},
        {"supports_spec_device_penalties": False},
        {"supports_spec_device_penalties": True, "spec_ids_are_sampled": True},
    ],
)
def test_declarations_the_plugin_accepts(capabilities):
    validate_spec_device_penalties(capabilities, "M")
    assert spec_device_penalties(
        SimpleNamespace(model_capabilities=capabilities)
    ) is bool(capabilities and capabilities.get("supports_spec_device_penalties"))


def _sampling_runner(capabilities, has_penalties=True, num_speculative_tokens=3):
    """A runner whose batch carries a penalty, on a launch that samples on the
    device (``check_perform_device_sampling`` and ``_decode_device_sampled``
    read only these fields)."""
    runner = SimpleNamespace(
        sample_on_device_mode="all",
        seeded_sampling_policy="auto",
        num_devices=4,
        tt_data_parallel_size=1,
        _num_speculative_tokens=num_speculative_tokens,
        requests={},
        model=SimpleNamespace(model_capabilities=capabilities),
        model_config=SimpleNamespace(logits_processors=[]),
        input_batch=SimpleNamespace(
            no_penalties=not has_penalties,
            no_allowed_token_ids=True,
            max_num_logprobs=None,
            sampling=SimpleNamespace(
                bad_words_token_ids={},
                logprob_token_ids={},
                has_active_logitsprocs=lambda: False,
            ),
        ),
    )
    runner.check_perform_device_sampling = (
        TTModelRunner.check_perform_device_sampling.__get__(runner)
    )
    return runner


MTP_CAPABILITIES = {
    "supports_device_penalties": False,
    "spec_ids_are_sampled": True,
    "supports_spec_device_penalties": True,
}


@pytest.mark.parametrize(
    ("capabilities", "num_speculative_tokens", "expected"),
    [
        pytest.param(MTP_CAPABILITIES, 3, True, id="declared"),
        pytest.param(
            {**MTP_CAPABILITIES, "supports_spec_device_penalties": False},
            3,
            False,
            id="not-declared",
        ),
        pytest.param(MTP_CAPABILITIES, 0, False, id="not-speculating"),
        pytest.param({}, 3, True, id="supports_device_penalties-absent"),
    ],
)
def test_a_decode_step_with_a_penalty_samples_on_the_device_when_the_verify_applies_it(
    capabilities, num_speculative_tokens, expected
):
    """On a speculating launch whose model declares
    ``supports_spec_device_penalties``, a decode step with a penalty samples on
    the device although the model declares ``supports_device_penalties``
    False; otherwise that declaration keeps the step on the host. A prefill
    keeps the ordinary rule."""
    runner = _sampling_runner(
        capabilities, num_speculative_tokens=num_speculative_tokens
    )
    decided = TTModelRunner._decode_device_sampled(
        runner, SchedulerOutput.make_empty(), None, [0]
    )
    assert decided is expected
    prefill = runner.check_perform_device_sampling(
        is_decode=False, has_structured_outputs=False
    )
    assert prefill is capabilities.get("supports_device_penalties", True)


def _certifying_runner(capabilities, req_ids):
    return SimpleNamespace(
        model=SimpleNamespace(model_capabilities=capabilities),
        input_batch=SimpleNamespace(
            random_reqs={"sampled_penalised"},
            presence_penalties_reqs={PENALISED, "sampled_penalised"},
            frequency_penalties_reqs=set(),
            repetition_penalties_reqs={"repeat"},
        ),
        requests={},
    )


@pytest.mark.parametrize("req_id", [PENALISED, "sampled_penalised", "repeat"])
def test_a_penalised_request_is_certifiable_on_a_declaring_models_device_step(
    req_id,
):
    """The verify of a declaring model applies the penalties to every column it
    picks, so the id walk certifies a penalised request, greedy or sampled, on
    a device-sampled step. A host-sampled step, or a model without the
    declaration, keeps it uncertifiable."""
    certify = TTModelRunner._request_is_argmax_certifiable
    runner = _certifying_runner(MTP_CAPABILITIES, [req_id])
    assert certify(runner, req_id, device_sampled=True) is True
    assert certify(runner, req_id, device_sampled=False) is False
    runner = _certifying_runner(
        {**MTP_CAPABILITIES, "supports_spec_device_penalties": False}, [req_id]
    )
    assert certify(runner, req_id, device_sampled=True) is False
