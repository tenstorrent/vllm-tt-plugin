# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
import math

import openai
import pytest

from tests.tt.utils import RequestConfig, run_concurrent_batch


# 0 means just the sampled token's logprob
class TestLogprobs:
    @pytest.mark.parametrize("batch_fraction", [0, 0.5, 1, 1.5])
    @pytest.mark.parametrize("num_logprobs", [0, 1, 3, 5, 10])
    def test_logprobs(
        self, tt_server, tt_model_name, max_batch_size, batch_fraction, num_logprobs
    ):
        """Test logprobs parameter returns actual logprobs data.

        batch_fraction: 1.5 = one and a half batches, 1 = full batch,
            0.5 = half batch, 0 = single request
        num_logprobs: number of top logprobs alternatives to return
        """
        if batch_fraction == 0:
            num_requests = 1
        else:
            num_requests = max(1, int(max_batch_size * batch_fraction))

        # Use return_tokens_as_token_ids to ensure unique keys in top_logprobs
        # dict. Without this, different token IDs that decode to the same
        # string would collide, reducing the count below num_logprobs.
        configs = [
            RequestConfig(
                prompt=f"Count from {i}: ",
                max_tokens=10,
                logprobs=num_logprobs,
                return_tokens_as_token_ids=True,
            )
            for i in range(num_requests)
        ]
        results = run_concurrent_batch(
            tt_server, tt_model_name, configs, return_full_response=True
        )
        assert len(results) == len(configs)

        # Verify ALL responses have valid logprobs (catches DP rank issues)
        for i, response in enumerate(results):
            choice = response.choices[0]

            # Verify logprobs are returned for every request
            assert choice.logprobs is not None, (
                f"logprobs should be returned for request {i}"
            )
            assert choice.logprobs.tokens is not None, (
                f"logprobs.tokens should exist for request {i}"
            )
            assert len(choice.logprobs.tokens) > 0, (
                f"should have at least one token for request {i}"
            )

            # Verify top_logprobs contains the requested number of alternatives
            assert choice.logprobs.top_logprobs is not None, (
                f"top_logprobs should be returned for request {i}"
            )
            assert len(choice.logprobs.top_logprobs) > 0, (
                f"should have top_logprobs for tokens for request {i}"
            )
            # Each position should have up to num_logprobs+1 alternatives
            for j, top_lp in enumerate(choice.logprobs.top_logprobs):
                assert len(top_lp) >= num_logprobs, (
                    f"request {i}, token {j}: should have at least "
                    f"{num_logprobs} alternatives per token"
                )
                assert len(top_lp) <= num_logprobs + 1, (
                    f"request {i}, token {j}: should have at most "
                    f"{num_logprobs + 1} alternatives per token"
                )

            # Verify actual logprob values exist (not just structure)
            assert choice.logprobs.token_logprobs is not None, (
                f"request {i}: token_logprobs is None"
            )
            for j, lp in enumerate(choice.logprobs.token_logprobs):
                assert lp is not None, f"request {i}, token {j}: logprob value is None"

    def test_chat_logprobs_all_vocab(self, tt_server, tt_model_name):
        """Test chat completions with top_logprobs=-1 (all vocab logprobs).

        The legacy completions API rejects logprobs=-1, so this must go
        through the chat API where top_logprobs=-1 is allowed.
        Servers with a low max_logprobs cap will reject this request;
        the test is skipped in that case.
        """
        configs = [
            RequestConfig(
                prompt="Say hello.",
                max_tokens=5,
                logprobs=-1,
            )
        ]
        try:
            results = run_concurrent_batch(
                tt_server,
                tt_model_name,
                configs,
                use_chat=True,
                return_full_response=True,
            )
        except openai.BadRequestError as e:
            if "max allowed" in str(e).lower() and "logprobs" in str(e).lower():
                pytest.skip(f"Server does not support all-vocab logprobs: {e}")
            raise
        assert len(results) == 1

        choice = results[0].choices[0]
        assert choice.logprobs is not None, "Logprobs missing"
        assert choice.logprobs.content is not None, "Logprobs content missing"
        assert len(choice.logprobs.content) > 0, "Logprobs content empty"
        for i, token_lp in enumerate(choice.logprobs.content):
            assert token_lp.top_logprobs is not None, f"token {i}: top_logprobs missing"
            assert len(token_lp.top_logprobs) > 10, (
                f"token {i}: expected many top_logprobs for all-vocab, "
                f"got {len(token_lp.top_logprobs)}"
            )


def test_greedy_logprobs_are_a_distribution_led_by_the_token(
    tt_server, tt_model_name, max_batch_size
):
    """Greedy requests asking for top-5 logprobs, a full batch at once: each
    generated token is its own top-1 alternative with the same logprob, the
    alternatives are sorted and at most 0, and their probabilities sum to at
    most 1. On a speculating launch whose model returns the logprobs with an
    argmax_ids verify (supports_spec_device_logprobs) these come from that
    payload; a payload with a wrong log-sum-exp or wrong top ids fails here."""
    configs = [
        RequestConfig(
            prompt=f"Write the numbers from {i} to {i + 30}: ",
            max_tokens=24,
            temperature=0.0,
            logprobs=5,
            return_tokens_as_token_ids=True,
        )
        for i in range(max(1, max_batch_size))
    ]
    results = run_concurrent_batch(
        tt_server, tt_model_name, configs, return_full_response=True
    )
    for i, response in enumerate(results):
        logprobs = response.choices[0].logprobs
        assert logprobs is not None and logprobs.top_logprobs, f"request {i}"
        for j, (token, value, top) in enumerate(
            zip(logprobs.tokens, logprobs.token_logprobs, logprobs.top_logprobs)
        ):
            where = f"request {i}, token {j}"
            ranked = sorted(top.items(), key=lambda kv: kv[1], reverse=True)
            assert ranked[0][1] == pytest.approx(value, abs=1e-3), where
            assert top.get(token) == pytest.approx(value, abs=1e-3), where
            assert all(v <= 1e-3 for v in top.values()), where
            assert sum(math.exp(v) for v in top.values()) <= 1.0 + 1e-3, where
