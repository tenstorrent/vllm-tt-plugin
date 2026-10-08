# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

from __future__ import annotations

import pytest

from tests.tt.spec.spec_client import (
    Acceptance,
    Completion,
    assert_full_length_completion,
    assert_multirow_decode_was_scheduled,
    assert_rejections_happened,
)


def _completion(token_count: int) -> Completion:
    return Completion(
        request={"max_tokens": 64, "ignore_eos": True},
        status=200,
        body={
            "choices": [
                {
                    "token_ids": list(range(token_count)),
                    "text": "",
                    "finish_reason": "length",
                }
            ],
            "usage": {"completion_tokens": token_count},
        },
    )


def test_full_length_assertion_rejects_a_correct_truncated_prefix():
    with pytest.raises(AssertionError):
        assert_full_length_completion(_completion(59), 64)

    assert_full_length_completion(_completion(64), 64)


def test_multirow_assertion_rejects_strictly_sequential_decode(tmp_path):
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "TT scheduler: widest decode batch reached 1 request row(s)\n"
    )

    with pytest.raises(AssertionError, match="no decode batch"):
        assert_multirow_decode_was_scheduled(server_log)

    server_log.write_text(
        "TT scheduler: widest decode batch reached 1 request row(s)\n"
        "TT scheduler: widest decode batch reached 3 request row(s)\n"
    )
    assert assert_multirow_decode_was_scheduled(server_log) == 3


def _acceptance(draft_tokens: int, per_position: list[int]) -> Acceptance:
    return Acceptance(
        drafts=draft_tokens // len(per_position),
        draft_tokens=draft_tokens,
        accepted=sum(per_position),
        per_position=per_position,
        steps=draft_tokens,
        preemptions=0,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_rejection_assertion_accepts_one_draft_position(asynchronous):
    assert_rejections_happened(_acceptance(100, [40]), asynchronous)
    assert_rejections_happened(_acceptance(200, [40, 20]), asynchronous)


def test_rejection_assertion_rejects_runs_without_a_rejection():
    with pytest.raises(AssertionError, match="every draft was accepted"):
        assert_rejections_happened(_acceptance(100, [100]), asynchronous=False)
    with pytest.raises(AssertionError, match="no step stopped"):
        assert_rejections_happened(_acceptance(200, [50, 50]), asynchronous=True)
    with pytest.raises(AssertionError, match="no draft was accepted"):
        assert_rejections_happened(_acceptance(100, [0]), asynchronous=True)
