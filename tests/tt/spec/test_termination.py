# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A request stops where it should, inside a committed prefix.

Speculation makes stopping hard in one specific way: a step commits several
tokens at once, so the token that ends the request usually arrives in the
middle of a block rather than at its end. Everything after it in that block has
to be discarded, and the same boundary has to hold whether the client is
streaming or not.

The three ways a request ends are each put inside a multi-token prefix here:

``max_tokens``
    A limit that is not a multiple of the committed width, so the last step
    commits more tokens than the request is allowed to keep.
``stop_token_ids``
    A token the target will emit in the middle of a step's prefix.
``the end-of-sequence id``
    The dummy's ``config.json`` sets only ``model_type``, so ``eos_token_id``
    comes from HF's ``LlamaConfig`` default of 2, and this model counts
    straight through it: with an ascending run, token 2 lands inside the first
    step's committed prefix.

Cancellation is here too, because an abandoned request has to free its row: the
test that follows a cancellation with another request is the one that shows the
row came back.
"""

from __future__ import annotations

import pytest

from tests.tt.spec.conftest import DUMMY_EOS_TOKEN_ID
from tests.tt.spec.dummy_arithmetic import (
    a_token_strictly_inside_a_prefix,
    discarded_after,
    ids_up_to_and_including,
)
from tests.tt.spec.spec_client import acceptance_delta


def test_max_tokens_cuts_inside_a_committed_prefix(
    spec_server, spec_config, ascending_prompt, record
):
    """A limit that falls mid-block, so the last step overshoots it.

    The width is ``1+depth``, and the limit is chosen to leave a remainder, so
    the final step commits tokens the response must not contain.
    """
    width = spec_config.committed_per_step
    if width < 2:
        pytest.skip("this configuration commits one token per step")
    limit = width * 3 + 1
    prompt = ascending_prompt(64, start=11)

    before = spec_server.metrics()
    completion = spec_server.complete(prompt, max_tokens=limit)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        request=completion.request, response=completion.body, acceptance=delta.as_dict()
    )

    assert completion.status == 200
    assert completion.completion_tokens == limit, (
        f"a limit of {limit} with a committed width of {width} has to cut the "
        "last step's prefix, and the response has to stop at the limit"
    )
    assert len(completion.token_ids) == limit
    assert completion.finish_reason == "length"


def test_a_stop_token_inside_a_prefix_ends_the_request_there(
    spec_server, spec_config, ascending_prompt, record
):
    """Everything the step committed after the stop token is discarded."""
    if spec_config.target != "depth":
        pytest.skip("the stop token is picked from this target's own arithmetic")
    stop_id = a_token_strictly_inside_a_prefix(spec_config.k, spec_config.accept_depth)
    if stop_id is None:
        pytest.skip("this configuration commits one token per step")
    prompt = ascending_prompt(64, start=0)

    before = spec_server.metrics()
    completion = spec_server.complete(
        prompt, max_tokens=96, stop_token_ids=[stop_id], ignore_eos=True
    )
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        request=completion.request,
        response=completion.body,
        acceptance=delta.as_dict(),
        stop_id=stop_id,
    )

    assert completion.status == 200
    assert completion.finish_reason == "stop"
    assert completion.stop_reason == stop_id
    ids = completion.token_ids
    # The stopping token is the last id the response carries: it appears in the
    # raw ids and not in the text, which is vLLM's convention here.
    assert ids[-1] == stop_id
    assert stop_id not in ids[:-1]
    # And the rest of the step that committed it is gone. The whole sequence is
    # predictable from the accept depth, so anything the step committed after
    # the stop token would show up as an extra id.
    expected = ids_up_to_and_including(spec_config.k, spec_config.accept_depth, stop_id)
    discarded = discarded_after(spec_config.k, spec_config.accept_depth, stop_id)
    assert discarded > 0, (
        "this configuration ends the step on the stop token, so nothing would "
        "be discarded and the test proves nothing"
    )
    assert ids == expected, (
        f"the step that committed {stop_id} committed {discarded} token(s) "
        "after it, and all of them have to be discarded"
    )
    assert len(ids) < 96


def test_the_end_of_sequence_id_inside_a_prefix_ends_the_request(
    spec_server, spec_config, ascending_prompt, record
):
    """EOS committed in the middle of a block, with ``ignore_eos`` off.

    This model counts through the id HF defaults ``eos_token_id`` to, so an
    accept-all run commits 0, 1, 2, ... and the end of sequence lands inside
    the very first step's prefix. A server that only looked at the last token
    of a block would run past it.
    """
    if spec_config.target != "depth":
        pytest.skip("the target's arithmetic is what places EOS inside a step")
    discarded = discarded_after(
        spec_config.k, spec_config.accept_depth, DUMMY_EOS_TOKEN_ID
    )
    if discarded == 0:
        pytest.skip(
            "this configuration commits the end-of-sequence id at the end of "
            "its step, so ending there discards nothing"
        )
    prompt = ascending_prompt(64, start=0)

    before = spec_server.metrics()
    completion = spec_server.complete(prompt, max_tokens=96, ignore_eos=False)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        request=completion.request, response=completion.body, acceptance=delta.as_dict()
    )

    assert completion.status == 200
    assert completion.finish_reason == "stop"
    ids = completion.token_ids
    # The end-of-sequence id is the last id the response carries, and the
    # tokens its own step committed after it are discarded.
    expected = ids_up_to_and_including(
        spec_config.k, spec_config.accept_depth, DUMMY_EOS_TOKEN_ID
    )
    assert ids == expected, (
        "the request has to stop at the end-of-sequence id that this step "
        f"committed mid-prefix, discarding the {discarded} token(s) after it: "
        f"saw {ids}"
    )
    # The text stops one token earlier, because the stopping token is not part
    # of it.
    assert completion.body["choices"][0]["text"] != ""


def test_streaming_and_non_streaming_stop_at_the_same_boundary(
    spec_server, spec_config, ascending_prompt, record
):
    """One request, two transports, one boundary.

    A streamed response is assembled from the chunks a step produced, so a
    boundary applied to the whole block but not to the chunks, or the other way
    round, shows up here and nowhere else.
    """
    if spec_config.target != "depth":
        pytest.skip("the stop token is picked from this target's own arithmetic")
    stop_id = a_token_strictly_inside_a_prefix(spec_config.k, spec_config.accept_depth)
    if stop_id is None:
        pytest.skip("this configuration commits one token per step")
    prompt = ascending_prompt(64, start=0)

    plain = spec_server.complete(
        prompt, max_tokens=96, stop_token_ids=[stop_id], ignore_eos=True
    )
    streamed = spec_server.stream(
        prompt, max_tokens=96, stop_token_ids=[stop_id], ignore_eos=True
    )
    record(
        request=plain.request,
        non_streaming=plain.body,
        streamed_chunks=len(streamed.chunks),
        streamed_text=streamed.text,
        streamed_ids=streamed.token_ids,
    )

    assert plain.finish_reason == "stop"
    assert streamed.finish_reason == "stop"
    assert streamed.text == plain.text
    if streamed.token_ids:
        assert streamed.token_ids == plain.token_ids


def test_streaming_and_non_streaming_agree_on_a_length_boundary(
    spec_server, spec_config, ascending_prompt, record
):
    """The same, for a limit that cuts a block."""
    width = spec_config.committed_per_step
    limit = width * 3 + 1
    prompt = ascending_prompt(64, start=11)

    plain = spec_server.complete(prompt, max_tokens=limit)
    streamed = spec_server.stream(prompt, max_tokens=limit)
    record(
        request=plain.request,
        non_streaming=plain.body,
        streamed_chunks=len(streamed.chunks),
        streamed_text=streamed.text,
    )

    assert plain.completion_tokens == limit
    assert streamed.text == plain.text
    assert streamed.finish_reason == "length"


def test_a_cancelled_request_frees_its_row_for_the_next_one(
    spec_server, spec_config, ascending_prompt, record
):
    """Abandon a stream, then serve another request correctly.

    A cancelled request that kept its row would leak a slot, and a speculative
    launch has one accepted count and one pending proposal per row to clean up
    with it. The request that follows is what shows the row came back: it has
    to complete, with its own length, and with drafting still happening.
    """
    prompt = ascending_prompt(64, start=901)

    chunks = spec_server.stream_and_abandon(prompt, after=2, max_tokens=512)

    before = spec_server.metrics()
    completion = spec_server.complete(ascending_prompt(64, start=13), max_tokens=48)
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        abandoned_after_chunks=chunks,
        request=completion.request,
        response=completion.body,
        acceptance=delta.as_dict(),
    )

    assert chunks >= 1, "the stream produced nothing to abandon"
    assert completion.status == 200
    assert completion.completion_tokens == 48
    assert delta.drafts > 0, (
        "the request after the cancellation did not speculate, so the "
        "cancelled request may have left its speculative state behind"
    )


def test_many_cancellations_do_not_exhaust_the_rows(
    spec_server, spec_config, ascending_prompt, max_batch_size, record
):
    """Cancel as many requests as there are rows, then use them all.

    One leaked row is invisible on a server with several. Cancelling
    ``max_num_seqs`` requests and then filling the batch is what makes a leak
    fail rather than pass.
    """
    rows = min(4, max_batch_size)
    abandoned = []
    for index in range(rows):
        abandoned.append(
            spec_server.stream_and_abandon(
                ascending_prompt(64, start=1000 + 50 * index), after=2, max_tokens=512
            )
        )

    before = spec_server.metrics()
    results = [
        spec_server.complete(
            ascending_prompt(64, start=17 * (index + 1)), max_tokens=32
        )
        for index in range(rows)
    ]
    after = spec_server.metrics()
    delta = acceptance_delta(before, after, spec_config.k)
    record(
        abandoned=abandoned,
        responses=[result.body for result in results],
        acceptance=delta.as_dict(),
    )

    assert all(result.status == 200 for result in results)
    assert all(result.completion_tokens == 32 for result in results)
    assert delta.drafts > 0
