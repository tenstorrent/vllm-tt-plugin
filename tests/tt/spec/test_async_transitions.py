# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A speculating server that actually schedules asynchronously.

Every other device test here runs against `--no-async-scheduling`, because
until now a speculating launch could not keep asynchronous scheduling: vLLM
refuses it for the `custom_class` method before the TT platform hook, and the
dummy declared no `supports_async_decode`. Both are fixed, so this file drives
the launch that was previously unreachable and asserts the four things such a
server has to show.

Three of them cannot be read from an HTTP response, and two cannot be read from
vLLM's metrics either:

- **the resolved scheduling mode**, which vLLM logs once as "Asynchronous
  scheduling is enabled"; a launch flag is a request, not an outcome, and
  upstream may still disable it.
- **that ordinary steps overlapped**, which the plugin logs as its count of
  submissions that found a step already outstanding. No metric reports overlap,
  and per-step wall clock cannot separate it from a faster model.
- **that no verify overlapped anything**, the second number on that same line,
  which must be zero for the life of the server: a verify's candidate block
  starts from each row's last committed token, and an outstanding step is
  holding one.

The fourth, that speculation still happens and still resumes, is the acceptance
counters, and the fifth, that the output is the one an unspeculated run emits,
is the closed-form rule the `fixed` target follows.

These tests read the server log, so they need the driver's
`--tt-spec-server-log`, and they skip without it rather than assert something
weaker.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.tt.spec.conftest import DUMMY_VOCAB_SIZE
from tests.tt.spec.spec_client import acceptance_delta

MAX_TOKENS = 64
OVERLAP_LINE = re.compile(
    r"TT async decode: (\d+) submission\(s\) overlapped an outstanding step, "
    r"(\d+) of them a step that was not overlap-safe"
)


@pytest.fixture(autouse=True)
def only_the_asynchronous_launch(request):
    if str(request.config.getoption("--tt-spec-async-scheduling")).lower() != "true":
        pytest.skip(
            "this server was launched with --no-async-scheduling: pass "
            "--tt-spec-async-scheduling=true for the launch that keeps it"
        )


@pytest.fixture
def server_log(request):
    path = request.config.getoption("--tt-spec-server-log")
    if not path or not Path(path).exists():
        pytest.skip("pass --tt-spec-server-log: these claims live in the log")
    return Path(path)


def _overlap_counts(server_log):
    """The plugin's last overlap report, or None if it never logged one.

    Logged every power of two, so the last line is the largest count the run
    reached. None means no submission ever found a step outstanding, which is
    a server that never overlapped.
    """
    last = None
    for line in server_log.read_text(errors="replace").splitlines():
        found = OVERLAP_LINE.search(line)
        if found:
            last = (int(found.group(1)), int(found.group(2)))
    return last


def test_the_resolved_server_schedules_asynchronously(server_log, record):
    """The outcome, not the request.

    vLLM decides asynchronous scheduling inside configuration and logs what it
    decided. It disables the setting for a speculative method it does not
    know, so a launch that merely asked for it proves nothing; this reads what
    the engine resolved.
    """
    text = server_log.read_text(errors="replace")
    record(log_says_enabled="Asynchronous scheduling is enabled." in text)

    assert "Asynchronous scheduling is enabled." in text, (
        "the engine resolved to synchronous scheduling, so every other claim "
        "in this file would be about the wrong server"
    )
    assert "Asynchronous scheduling is disabled." not in text


def test_ordinary_steps_overlap_and_no_verify_does(
    spec_server, spec_config, ascending_prompt, max_batch_size, server_log, record
):
    """The performance claim and the safety claim, from the same counter.

    A batch is served by ordinary decode steps under the adaptive draft
    policy, and those are the steps that may overlap: the plugin counts a
    submission that found a step already outstanding. The same line counts how
    many of those were not overlap-safe, which is what a verify is, and that
    number must be zero.
    """
    rows = min(4, max_batch_size)
    if rows < 2:
        pytest.skip("this server serves one row at a time")
    before_counts = _overlap_counts(server_log)
    before = spec_server.metrics()

    def send(index):
        return spec_server.complete(
            ascending_prompt(64, start=100 * (index + 1)), max_tokens=MAX_TOKENS
        )

    with ThreadPoolExecutor(max_workers=rows) as pool:
        results = list(pool.map(send, range(rows)))
    after = spec_server.metrics()
    # The counters are logged as the submissions happen, and the response
    # returns before the log line is flushed in the worst case.
    time.sleep(1.0)
    counts = _overlap_counts(server_log)
    record(
        requests=[result.request for result in results],
        acceptance=acceptance_delta(before, after, spec_config.k).as_dict(),
        overlap_before=before_counts,
        overlap_after=counts,
    )

    for result in results:
        assert result.completion_tokens >= MAX_TOKENS - spec_config.k

    assert counts is not None, (
        "no submission ever overlapped an outstanding step, so this server "
        "serialized every step it ran"
    )
    overlapped, unsafe = counts
    if before_counts is not None:
        assert overlapped > before_counts[0], (
            "the overlap count did not move while this batch was decoding"
        )
    assert unsafe == 0, (
        f"{unsafe} submission(s) overlapped an outstanding step while not "
        "overlap-safe; a verify built over an outstanding step starts its "
        "candidate block one token behind"
    )


def test_speculation_happens_and_resumes_on_the_asynchronous_launch(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """Async must not cost the speculation itself.

    Upstream stops routing drafts through the scheduler when asynchronous
    scheduling is on, so a launch that got this wrong would serve correctly and
    never speculate. A lone request has to be drafted for, and has to be
    drafted for again after a peer comes and goes.
    """
    if max_batch_size < 2:
        pytest.skip("this server serves one row at a time")
    solo_before = spec_server.metrics()
    solo = spec_server.complete(ascending_prompt(64), max_tokens=MAX_TOKENS)
    solo_after = spec_server.metrics()
    solo_delta = acceptance_delta(solo_before, solo_after, spec_config.k)

    # Long enough to outlive the peer even while speculating, which makes it
    # the faster of the two per step: it commits up to 1+K where the peer
    # commits one.
    long_tokens = MAX_TOKENS * 16
    with ThreadPoolExecutor(max_workers=2) as pool:
        long_future = pool.submit(
            spec_server.complete,
            ascending_prompt(64, start=300),
            max_tokens=long_tokens,
        )
        time.sleep(0.4)
        peer = pool.submit(
            spec_server.complete, ascending_prompt(64, start=900), max_tokens=MAX_TOKENS
        )
        peer_result = peer.result()
        after_peer = spec_server.metrics()
        long_result = long_future.result()
    tail = spec_server.metrics()
    resumed = acceptance_delta(after_peer, tail, spec_config.k)
    record(
        requests=[solo.request, long_result.request, peer_result.request],
        solo=solo_delta.as_dict(),
        after_the_peer_left=resumed.as_dict(),
    )

    # Acceptance, not the draft counters. ``AsyncScheduler`` gives every
    # scheduled request ``[-1] * num_spec_tokens`` as its lookahead
    # reservation, and vLLM counts those as drafts, so on an asynchronous
    # launch ``spec_decode_num_drafts_total`` moves whether or not a real
    # draft was ever verified. What cannot move without real speculation is
    # the accepted count.
    assert solo_delta.accepted > 0, "a lone request never had a draft accepted"
    assert solo_delta.mean_acceptance_length > 1.0
    assert resumed.steps > 0, "the long request finished before its peer"
    assert resumed.accepted > 0, "speculation never resumed after the peer left"


def test_the_asynchronous_output_is_the_rule_s_own_sequence(
    spec_server, spec_config, ascending_prompt, record
):
    """Correctness, against something that does not depend on the run.

    The ``fixed`` target chooses by ``(token * 31 + position * 7 + 11) % vocab``
    for an ordinary decode and for a verify alike, so its output is a property
    of the prompt tail rather than of what was drafted, of how much was
    accepted, or of which steps overlapped. A deferred commit that wrote a
    token twice, dropped one, or committed one from the wrong row would show
    here.
    """
    if spec_config.target != "fixed":
        pytest.skip("this claim needs the fixed target: pass --tt-spec-target=fixed")
    prompt = ascending_prompt(64, start=100)

    result = spec_server.complete(prompt, max_tokens=MAX_TOKENS)
    record(requests=[result.request])

    ids = result.token_ids
    assert len(ids) >= MAX_TOKENS - spec_config.k
    # The first emitted token is the prefill's, and this model answers a
    # prefill the way its base class does, with zero logits whose argmax is
    # token 0. Every token after it is the rule applied to the one before,
    # starting at the position that token occupies.
    expected = [0]
    token, position = 0, len(prompt)
    while len(expected) < len(ids):
        token = (token * 31 + position * 7 + 11) % DUMMY_VOCAB_SIZE
        position += 1
        expected.append(token)
    assert ids == expected
