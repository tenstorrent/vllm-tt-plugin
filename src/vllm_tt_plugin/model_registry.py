# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

from vllm_tt_plugin.platform import (
    _install_diffusion_gemma_architecture_patch,
    _should_pre_register_tt_test_models_from_cli,
    register_tt_models,
    register_tt_test_models,
)

__all__ = [
    "register_tt_models",
    "register_tt_models_from_plugin",
    "register_tt_test_models",
]


def register_tt_models_from_plugin() -> None:
    """Entry point used by ``vllm.general_plugins``."""
    register_tt_models(
        register_test_models=_should_pre_register_tt_test_models_from_cli()
    )
    # The rewrite must travel with registration: every process that can build
    # a ModelConfig (engine core, workers, the registry subprocess) also loads
    # general plugins, and without the rewrite upstream's DiffusionGemma
    # MODELS_CONFIG_MAP hook fires there before any platform hook runs.
    _install_diffusion_gemma_architecture_patch()
