# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The automatic-prefix-caching contract a model class opts into.

A model class that declares ``model_capabilities["supports_prefix_caching"]``
gets vLLM automatic prefix caching (APC) and, on every prefill, the scheduler's
cached-prefix length:

* the platform keeps ``enable_prefix_caching`` on for it and switches it off for
  every other model;
* each prefill row receives ``start_pos = num_computed_tokens`` (the cached
  prefix ``H``) with ``tokens`` holding the full prompt row and ``prompt_lens``
  the full prompt length;
* a full-prompt hit, which vLLM caps at ``H = P - 1`` so the final prompt row
  is always computed, is still a prefill with ``start_pos = P - 1``. It does not
  go through decode, so the model's prefill-side sampler setup runs for it.

These are the inputs a resumed-prefill model implementation is written against,
for example tt-quetzalcoatlus ``serving/prefix_resume.py`` (tt-quetzalcoatlus
#702), so they are locked here.
"""

import pytest

from tests.test_block_request_validation import (
    ARModel,
    TTPlatform,
    _ar_config,
    _patch_model_resolution,
)
from tests.test_stage_inputs import (
    BLOCK_SIZE,
    _new_request,
    _run_prefill,
    _runner,
    _step,
)


class _PrefixCachingARModel(ARModel):
    model_capabilities = {
        **ARModel.model_capabilities,
        "supports_prefix_caching": True,
    }


# region Platform: capability-gated APC


def test_a_model_declaring_prefix_caching_keeps_apc_enabled(monkeypatch):
    config = _ar_config()
    config.cache_config.enable_prefix_caching = True
    _patch_model_resolution(monkeypatch, _PrefixCachingARModel)

    TTPlatform.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is True


def test_a_model_without_the_declaration_loses_apc(monkeypatch):
    config = _ar_config()
    config.cache_config.enable_prefix_caching = True
    _patch_model_resolution(monkeypatch, ARModel)

    TTPlatform.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is False


def test_the_declaration_does_not_turn_apc_on_by_itself(monkeypatch):
    # The operator's --no-enable-prefix-caching stays authoritative.
    config = _ar_config()
    config.cache_config.enable_prefix_caching = False
    _patch_model_resolution(monkeypatch, _PrefixCachingARModel)

    TTPlatform.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is False


# endregion

# region Runner: the cached prefix reaches prefill as start_pos


def _cached_new_request(req_id, prompt, num_computed_tokens):
    """A new request the scheduler found ``num_computed_tokens`` of in the cache."""
    request = _new_request(req_id)
    request.prompt_token_ids = list(prompt)
    request.num_computed_tokens = num_computed_tokens
    blocks = -(-len(prompt) // BLOCK_SIZE)
    request.block_ids = (list(range(1, blocks + 1)),)
    return request


def _prompt(length, base=3):
    return [base + i for i in range(length)]


def _prefill(runner, *rows):
    """Schedule ``(req_id, prompt, cached)`` rows as new requests in one step."""
    new = [_cached_new_request(r, p, c) for r, p, c in rows]
    scheduled = {r: len(p) - c for r, p, c in rows}
    return _run_prefill(runner, _step(new=new, scheduled=scheduled))


def test_a_cold_prefill_gets_start_pos_zero():
    prompt = _prompt(20)
    kwargs = _prefill(_runner(supports_stage_inputs=False), ("a", prompt, 0))

    assert kwargs["start_pos"].tolist() == [0]
    assert kwargs["prompt_lens"].tolist() == [20]
    assert kwargs["tokens"][0, :20].tolist() == prompt


def test_a_cached_prefix_reaches_prefill_as_start_pos_with_the_full_prompt():
    # vLLM hits whole blocks only: H is a multiple of the block size.
    prompt = _prompt(40)
    hit = 2 * BLOCK_SIZE
    kwargs = _prefill(_runner(supports_stage_inputs=False), ("a", prompt, hit))

    assert kwargs["start_pos"].tolist() == [hit]
    assert kwargs["prompt_lens"].tolist() == [40], "full length, not the suffix"
    assert kwargs["tokens"][0, :40].tolist() == prompt, "full row, not the suffix"


def test_rows_with_different_hits_get_their_own_start_pos_in_row_order():
    first, second, third = _prompt(40), _prompt(24, base=50), _prompt(33, base=90)
    kwargs = _prefill(
        _runner(supports_stage_inputs=False),
        ("a", first, 2 * BLOCK_SIZE),
        ("b", second, 0),
        ("c", third, BLOCK_SIZE),
    )

    assert kwargs["start_pos"].tolist() == [2 * BLOCK_SIZE, 0, BLOCK_SIZE]
    assert kwargs["prompt_lens"].tolist() == [40, 24, 33]
    assert kwargs["tokens"][0, :40].tolist() == first
    assert kwargs["tokens"][1, :24].tolist() == second
    assert kwargs["tokens"][2, :33].tolist() == third


def test_a_full_prompt_hit_is_a_prefill_of_the_last_row():
    # P = 2 blocks, fully cached: vLLM caps the hit at P - 1 and schedules one
    # token. That token is prompt work and must reach prefill_forward, which
    # owns the request's sampler setup, rather than decode.
    prompt = _prompt(2 * BLOCK_SIZE)
    runner = _runner(supports_stage_inputs=False)
    kwargs = _prefill(runner, ("a", prompt, len(prompt) - 1))

    assert len(runner.model.prefills) == 1
    assert kwargs["start_pos"].tolist() == [len(prompt) - 1]
    assert kwargs["prompt_lens"].tolist() == [len(prompt)]
    assert kwargs["tokens"][0, : len(prompt)].tolist() == prompt


@pytest.mark.parametrize("hit", [BLOCK_SIZE, 2 * BLOCK_SIZE - 1])
def test_a_hit_is_never_marked_as_an_intermediate_chunk(hit):
    # A resumed row ends at the end of its prompt: its last row is sampled.
    runner = _runner(supports_stage_inputs=False)
    step = _step(
        new=[_cached_new_request("a", _prompt(2 * BLOCK_SIZE), hit)],
        scheduled={"a": 2 * BLOCK_SIZE - hit},
    )
    runner._update_states(step)
    model_input = runner._prepare_model_inputs(step, None)

    assert model_input.prompt_lens.tolist() == [2 * BLOCK_SIZE]
    assert model_input.intermediate_prefill_mask.tolist() == [False]


# endregion
