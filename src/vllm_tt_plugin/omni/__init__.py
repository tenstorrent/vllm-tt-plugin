# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""TT glue for vLLM-Omni: the Omni platform, stage schedulers, workers and runners.

Everything here imports ``vllm_omni``, which the core plugin does not depend on, so
nothing outside this package imports it.
"""
