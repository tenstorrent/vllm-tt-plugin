# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# vLLM-Omni on top of docs/install-vllm-tt.sh, for the Qwen3-TTS PoC. Source it
# after that script, from the repository root, in a Python 3.12 env: vLLM-Omni
# 0.30.0's CLI does not import on 3.10. poc/qwen3-tts-omni/INSTALL.md has the
# reasons behind each pin.
python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' ||
    # `return`, not `exit`: this script is sourced into the caller's shell.
    { echo "install-vllm-omni: vLLM-Omni 0.30.0 needs Python 3.12"; return 1; }
# --no-deps: Omni's metadata asks for a newer transformers than tt-metal pins.
uv pip install --no-deps "vllm-omni==0.30.0" "transformers==5.12.1"
# The PyPI torchaudio is the CUDA build; this is the CPU half of tt-metal's torch pair.
uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
    "torchaudio==2.11.0+cpu"
uv pip install --overrides docs/vllm-overrides.txt \
    "aenum==3.1.16" "omegaconf==2.4.0" "prettytable==3.18.0" "janus==2.0.0" \
    "soundfile==0.14.0" "diffusers==0.38.0" "av>=14.0.0" "onnxruntime>=1.23.2"
