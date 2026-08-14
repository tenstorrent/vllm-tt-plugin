# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import logging

from vllm_tt_plugin.logger import init_tt_logger


def test_tt_logger_provides_all_once_methods():
    """Every *_once idiom vLLM patches onto its loggers must also work on TT
    loggers, regardless of whether transformers (which patches warning_once
    onto logging.Logger) has been imported yet."""
    logger = init_tt_logger("vllm_tt_plugin.logger_probe")

    logger.warning_once("probe warning %s", "arg")
    logger.info_once("probe info")
    logger.debug_once("probe debug")


def test_once_methods_are_installed_independently(monkeypatch):
    """A process where another library already patched a subset of the *_once
    methods onto logging.Logger must keep that method and still receive the
    missing ones."""
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

    logger.info_once("instance info")
    logger.debug_once("instance debug")


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
