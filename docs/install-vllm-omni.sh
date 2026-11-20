# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# vLLM-Omni on top of docs/install-vllm-tt.sh, for the Omni PoCs. Source it after
# that script, from the repository root. poc/qwen3-tts-omni/INSTALL.md has the
# reasons behind each pin.
# --no-deps: Omni's other requirements are CUDA-side or only needed by models the
# PoCs do not serve; transformers comes from docs/vllm-overrides.txt, whose range
# is inside Omni's.
uv pip install --no-deps "vllm-omni==0.32.0"
# The PyPI torchaudio is the CUDA build; this is the CPU half of tt-metal's torch pair.
uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
    "torchaudio==2.11.0+cpu"
uv pip install --overrides docs/vllm-overrides.txt \
    "aenum==3.1.16" "omegaconf==2.4.0" "prettytable==3.18.0" "janus==2.0.0" \
    "soundfile==0.14.0" "diffusers==0.38.0" "av>=14.0.0" "onnxruntime>=1.23.2"
