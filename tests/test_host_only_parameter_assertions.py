# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host regressions for the live allowed-token test's token-level oracle."""

from types import SimpleNamespace

import pytest

from tests.tt import test_host_only_params as live_tests


@pytest.mark.parametrize(
    "fault, expected_error",
    [
        (None, None),
        ("disallowed", "outside its allowed set"),
        ("usage_count", "token count differs from usage"),
        ("early_length", "stopped at length before its token cap"),
        ("missing_ids", "must return exact integer token IDs"),
    ],
)
def test_allowed_token_oracle_checks_ids_not_printable_text(
    monkeypatch, fault, expected_error
):
    calls = []

    def concurrent_batch(server, model, configs, *, return_full_response):
        calls.append(configs)
        assert return_full_response is True
        assert len(configs) == 5
        assert [config.allowed_token_ids for config in configs] == [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
        ]
        responses = []
        for i, config in enumerate(configs):
            assert config.prompt == "Allowed: "
            assert config.max_tokens == 10
            assert config.return_tokens_as_token_ids is True
            assert config.logprobs == 0
            ids = (config.allowed_token_ids * 4)[: config.max_tokens]
            usage_count = len(ids)
            if i == 0:
                if fault == "disallowed":
                    ids[-1] = 15
                elif fault == "usage_count":
                    usage_count -= 1
                elif fault == "early_length":
                    ids.pop()
                    usage_count = len(ids)
            responses.append(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            # A byte sequence can emit ten IDs and no text.
                            text="" if i < 4 else '!"#!"#!"#!',
                            finish_reason="length",
                            logprobs=(
                                None
                                if i == 0 and fault == "missing_ids"
                                else SimpleNamespace(
                                    tokens=[f"token_id:{token}" for token in ids]
                                )
                            ),
                        )
                    ],
                    usage=SimpleNamespace(completion_tokens=usage_count),
                )
            )
        return responses

    monkeypatch.setattr(live_tests, "run_concurrent_batch", concurrent_batch)
    test = live_tests.TestHostOnlyParameters()
    if expected_error is None:
        test.test_allowed_token_ids(object(), "model", 32)
    else:
        with pytest.raises(AssertionError, match=expected_error):
            test.test_allowed_token_ids(object(), "model", 32)
    assert len(calls) == 1
