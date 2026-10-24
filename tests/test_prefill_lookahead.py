"""Model capability ``prefill_lookahead``: stored by the platform next to the other prefill capabilities, read by the model runner (``get_tt_prefill_lookahead``), overridden by ``TT_PREFILL_LOOKAHEAD``."""

import types

import pytest

from vllm_tt_plugin.config import get_tt_prefill_lookahead, store_tt_prefill_lookahead


def _cfg():
    return types.SimpleNamespace(additional_config={})


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("TT_PREFILL_LOOKAHEAD", raising=False)


def test_off_by_default():
    assert get_tt_prefill_lookahead(_cfg()) is False


def test_stored_capability_is_read():
    c = _cfg()
    store_tt_prefill_lookahead(c, True)
    assert get_tt_prefill_lookahead(c) is True
    store_tt_prefill_lookahead(c, False)
    assert get_tt_prefill_lookahead(c) is False


def test_environment_wins(monkeypatch):
    c = _cfg()
    store_tt_prefill_lookahead(c, True)
    monkeypatch.setenv("TT_PREFILL_LOOKAHEAD", "0")
    assert get_tt_prefill_lookahead(c) is False
    store_tt_prefill_lookahead(c, False)
    monkeypatch.setenv("TT_PREFILL_LOOKAHEAD", "1")
    assert get_tt_prefill_lookahead(c) is True


def test_config_without_additional_config():
    assert get_tt_prefill_lookahead(types.SimpleNamespace()) is False
    c = types.SimpleNamespace()
    store_tt_prefill_lookahead(c, True)
    assert get_tt_prefill_lookahead(c) is True
