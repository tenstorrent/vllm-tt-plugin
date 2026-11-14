# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import pytest

from vllm_tt_plugin.deepseek_v41_dsml import (
    END,
    EOS,
    START,
    parse_calls,
    safe_content_prefix,
)

CALL = """<｜DSML｜ invoke name="bank::lookup">
<｜DSML｜ parameter name="id" string="true">001</｜DSML｜ parameter>
<｜DSML｜ parameter name="amount" string="false">12.5</｜DSML｜ parameter>
<｜DSML｜ parameter name="active" string="false">true</｜DSML｜ parameter>
<｜DSML｜ parameter name="items" string="false">[1, null, "é"]</｜DSML｜ parameter>
</｜DSML｜ invoke>"""


def test_typed_arguments_and_parallel_calls():
    content, calls = parse_calls(
        "checking\n\n" + START + "\n" + CALL + "\n" + CALL + "\n" + END + EOS
    )
    assert content == "checking\n\n"
    assert len(calls) == 2
    assert calls[0] == (
        "bank::lookup",
        {"id": "001", "amount": 12.5, "active": True, "items": [1, None, "é"]},
    )


def test_parameterless_call():
    _, calls = parse_calls(
        START + '<｜DSML｜ invoke name="done">\n</｜DSML｜ invoke>' + END
    )
    assert calls == [("done", {})]


@pytest.mark.parametrize(
    "output",
    [
        START,
        START + END,
        START + CALL,
        START + CALL.replace(">12.5<", ">oops<") + END,
        START + CALL.replace('name="amount"', 'name="id"') + END,
        START + "junk" + CALL + END,
        START + CALL + END + "unexpected",
    ],
)
def test_malformed_calls_do_not_fabricate_arguments(output):
    with pytest.raises(ValueError):
        parse_calls(output)


def test_plain_text_and_partial_tag_prefixes():
    assert parse_calls("hello" + EOS) == ("hello", [])
    for n in range(1, len(START)):
        assert safe_content_prefix("hello" + START[:n]) == "hello"
    assert safe_content_prefix("hello" + START + CALL) == "hello"


def test_partial_eos_is_accepted_only_for_streaming():
    complete = START + CALL + END
    expected = parse_calls(complete)
    for length in range(1, len(EOS)):
        partial = complete + EOS[:length]
        with pytest.raises(ValueError, match="Unexpected content"):
            parse_calls(partial)
        assert parse_calls(partial, allow_partial_eos=True) == expected
    with pytest.raises(ValueError, match="Unexpected content"):
        parse_calls(complete + "unexpected", allow_partial_eos=True)
