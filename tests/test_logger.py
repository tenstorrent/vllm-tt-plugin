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

    # Dedup by record identity: the assertion is about EMISSIONS, and pytest 9
    # attaches its own capture handler to the non-propagating logger, so the
    # one emitted record can appear in caplog twice (once per handler).
    records = {id(r): r for r in caplog.records if "dedup probe" in r.getMessage()}
    assert len(records) == 1
    assert next(iter(records.values())).getMessage() == "dedup probe arg"


def test_warning_once_deduplicates_unhashable_arguments(caplog):
    logger = init_tt_logger("vllm_tt_plugin.logger_probe_unhashable")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="vllm.tt.logger_probe_unhashable"):
            logger.warning_once("config=%s", {"mode": "all"})
            logger.warning_once("config=%s", {"mode": "all"})
    finally:
        logger.removeHandler(caplog.handler)

    records = {id(r): r for r in caplog.records if "config=" in r.getMessage()}
    assert len(records) == 1
    assert next(iter(records.values())).getMessage() == "config={'mode': 'all'}"


def test_warning_once_does_not_format_disabled_message():
    logger = init_tt_logger("vllm_tt_plugin.logger_probe_disabled")
    logger.setLevel(logging.ERROR)
    cache_size_before = _log_once.cache_info().currsize

    class _RecordingArg:
        rendered = False

        def __str__(self):
            type(self).rendered = True
            return "rendered"

    logger.warning_once("bad value=%s", _RecordingArg())

    # The isEnabledFor guard must return before rendering the dedup key or
    # touching the once-cache; both would observably happen otherwise.
    assert _RecordingArg.rendered is False
    assert _log_once.cache_info().currsize == cache_size_before
