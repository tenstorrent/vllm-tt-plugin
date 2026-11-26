# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Activates TT model, parser and KV connector registration and TT platform discovery.

vLLM invokes ``register`` and ``platform_plugin`` through package entry points.
Model and parser registration imports are deferred until activation so importing
this module does not initialize vLLM or load model-output parsers.
"""

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


def register() -> None:
    """Registers TT models, missing model-output parsers and TT KV connectors."""
    from vllm_tt_plugin.model_registry import register_tt_models_from_plugin
    from vllm_tt_plugin.nixl_staging import register_kv_connectors
    from vllm_tt_plugin.parsers.registration import register_parsers

    register_tt_models_from_plugin()
    register_parsers()
    register_kv_connectors()


def platform_plugin() -> str | None:
    """Return the TT platform class when TT runtime libraries are present."""
    try:
        import ttnn  # noqa: F401
    except Exception as exc:
        logger.debug("TT plugin platform is not available because: %s", exc)
        return None

    logger.debug("Confirmed TT plugin platform is available because ttnn is found.")
    return "vllm_tt_plugin.platform.TTPlatform"
