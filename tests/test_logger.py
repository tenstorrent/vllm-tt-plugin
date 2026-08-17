# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import logging

import pytest

from vllm_tt_plugin.logger import _log_once, init_tt_logger


@pytest.fixture(autouse=True)
def clear_log_once_cache():
    _log_once.cache_clear()
    yield
    _log_once.cache_clear()


def test_tt_logger_provides_warning_once():
    """The warning_once idiom vLLM patches onto its loggers must also work on
    TT loggers, regardless of whether transformers (which patches warning_once
    onto logging.Logger) has been imported yet."""
    logger = init_tt_logger("vllm_tt_plugin.logger_probe")

    logger.warning_once("probe warning %s", "arg")


def test_existing_class_level_warning_once_is_kept(monkeypatch):
    """A process where another library already patched warning_once onto
    logging.Logger must keep that method rather than get an instance shadow."""
    class_level_calls = []

    def class_warning_once(self, msg, *args, **kwargs):
        class_level_calls.append(msg)

    monkeypatch.setattr(
        logging.Logger, "warning_once", class_warning_once, raising=False
    )

    logger = init_tt_logger("vllm_tt_plugin.logger_probe_partial")

    assert "warning_once" not in vars(logger)
    logger.warning_once("class-level warning")
    assert class_level_calls == ["class-level warning"]


def test_warning_once_deduplicates_identical_calls(caplog):
    logger = init_tt_logger("vllm_tt_plugin.logger_probe_dedup")

    # vLLM configures its logger tree with propagate=False, so records never
    # reach the root logger where caplog listens; attach its handler directly.
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="vllm.tt.logger_probe_dedup"):
            logger.warning_once("dedup probe %s", "arg")
            logger.warning_once("dedup probe %s", "arg")
    finally:
        logger.removeHandler(caplog.handler)

    records = [r for r in caplog.records if "dedup probe" in r.getMessage()]
    assert len(records) == 1
    assert records[0].getMessage() == "dedup probe arg"
