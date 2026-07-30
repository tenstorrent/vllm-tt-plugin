# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

VLLM_TARGET_DEVICE=empty uv pip install --no-binary vllm \
    --override docs/vllm-overrides.txt vllm==0.24.0   # see that file for why
uv pip uninstall torchaudio   # CUDA wheel, unloadable next to CPU torch; transformers>=5.12 imports it if merely installed
uv pip install -e .
