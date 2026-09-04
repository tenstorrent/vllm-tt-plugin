# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

# vLLM's PyPI metadata is generated on a CUDA machine, and uv takes resolver
# metadata from the published wheel even when the package is built from source.
# A plain `uv pip install vllm` therefore resolves requirements/cuda.txt (torch,
# torchaudio, numba, flashinfer, tilelang, tokenspeed-mla, nvidia-*) no matter
# what VLLM_TARGET_DEVICE says. The empty target declares requirements/common.txt
# instead, so fetch that list and install it explicitly, then give vLLM itself
# --no-deps so the CUDA list is never consulted. This script owns the dependency
# set as a result: re-read common.txt when bumping the version below.
# In a container, set UV_NO_CACHE=1 so the sdist and the wheel built from it do
# not stay behind in the image layer.
VLLM_COMMON_REQUIREMENTS=$(mktemp)
curl -fsSL \
    https://raw.githubusercontent.com/vllm-project/vllm/v0.28.0/requirements/common.txt \
    -o "$VLLM_COMMON_REQUIREMENTS" ||
    # `return`, not `exit`: this script is sourced into the caller's shell.
    { echo "install-vllm-tt: cannot fetch vLLM common.txt"; return 1; }
uv pip install --override docs/vllm-overrides.txt \
    -r "$VLLM_COMMON_REQUIREMENTS"   # see that file for why. Must read when bumping vllm version!
rm -f "$VLLM_COMMON_REQUIREMENTS"
# torchvision is absent from common.txt (requirements/cuda.txt carries it for
# the CUDA target), yet several vLLM model and processor modules import it
# unconditionally. Registry inspection imports the module for the resolved
# architecture, so without torchvision a serve command dies before any TT code
# runs; transformers' Gemma4 processor reaches it for google/gemma-4-*.
# --no-deps and the CPU index leave torch alone: this is the torchvision half
# of the torch pair the tt-metal env fixes, and the default PyPI wheel is the
# CUDA build.
uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.26.0   # keep in sync with tt-metal requirements-dev.txt
# --no-binary vllm: the published wheel is the CUDA build, kernels included, so
# vLLM has to come from source. vLLM ends up declaring no torch dependency, which
# is intended; torch belongs to the tt-metal env this plugin runs inside.
VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-binary vllm vllm==0.28.0
uv pip install -e .
