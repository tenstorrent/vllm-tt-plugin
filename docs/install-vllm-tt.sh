# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

VLLM_TARGET_DEVICE=empty uv pip install --no-binary vllm vllm==0.24.0
uv pip install -e .
