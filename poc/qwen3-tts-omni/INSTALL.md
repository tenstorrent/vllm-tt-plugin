# vLLM-Omni install for the Qwen3-TTS PoC

PoC-branch recipe for tenstorrent/vllm-tt-plugin#190. It moves to `docs/`
only when the Omni backend lands.

## Pinned pair

| Package | Version |
|---|---|
| vLLM | 0.32.0 (`docs/install-vllm-tt.sh`) |
| vLLM-Omni | 0.32.0 |
| Python | 3.11 or newer (tested on 3.12) |

### Python 3.11 or newer is required

vLLM 0.32.0 and vLLM-Omni 0.32.0 both require Python 3.11 or newer.

tt-metal's `create_venv.sh` picks 3.10 on Ubuntu 22.04. Its `ttnn` bindings
are built against the `python3` on `PATH`, without the stable ABI, so a 3.12
env needs a tt-metal build made with 3.12 as `python3`:

```bash
# python3 on PATH must be 3.12 for both steps
bash build_metal.sh
bash create_venv.sh --python-version 3.12
```

## Install

Run from the repository root, inside the target env (`$VIRTUAL_ENV` set).

```bash
# 1. Plugin and vLLM, as for any plugin env.
source docs/install-vllm-tt.sh

# 2. vLLM-Omni without its declared dependencies. transformers stays as
#    docs/vllm-overrides.txt set it in step 1 (>=5.16.1,<5.17.0), which is
#    inside Omni's >=5.16.1,<5.20.0.
uv pip install --no-deps "vllm-omni==0.32.0"

# 3. torchaudio from the CPU index, paired with tt-metal's torch 2.11.0+cpu.
#    The PyPI wheel is the CUDA build and fails to load (libcudart.so.13).
uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
  "torchaudio==2.11.0+cpu"

# 4. The other dependencies Qwen3-TTS serving needs.
uv pip install --overrides docs/vllm-overrides.txt \
  "aenum==3.1.16" "omegaconf==2.4.0" "prettytable==3.18.0" "janus==2.0.0" \
  "soundfile==0.14.0" "diffusers==0.38.0" "av>=14.0.0" "onnxruntime>=1.23.2"
```

Check:

```bash
python -c "import vllm, vllm_omni; print(vllm.__version__, vllm_omni.__version__)"
# 0.32.0 0.32.0
```

## Dependencies

### Added

| Package | Needed by |
|---|---|
| `aenum`, `omegaconf`, `prettytable` | `import vllm_omni` |
| `soundfile`, `diffusers`, `janus`, `torchaudio`, `av` | `vllm-omni serve` |
| `onnxruntime` | `qwen3_tts.qwen3_tts_tokenizer` |

Against a tt-metal `python_env`, steps 3 and 4 only add packages. They
upgrade or downgrade nothing that tt-metal installed.

### Held below Omni's metadata

| Package | Omni declares | Installed | Why |
|---|---|---|---|
| `diffusers` | `==0.40.0` | 0.38.0 | tt-metal ships 0.38.0 |

### Not installed

Omni declares these, and Qwen3-TTS does not import them:
`cosmos-guardrail`, `kernels`, `cache-dit`, `accelerate`, `openai-whisper`,
`x-transformers`, `torchsde`, `s3tokenizer`, `gguf`, `imageio`, `soxr`,
`scipy`. `uv pip check` lists them as missing; that is expected.

## What `vllm-omni serve --omni` does today

With `tests/host-stubs` on `PYTHONPATH` and the plugin installed:

| | Resolves to |
|---|---|
| vLLM platform, ttnn stub present | `TTPlatform` |
| vLLM platform, no ttnn | `UnspecifiedPlatform` (plugin inert) |
| Omni platform | `UnspecifiedOmniPlatform` (`device_type` `cpu`) |

`vllm-omni serve <Qwen3-TTS snapshot> --omni` (and `vllm serve --omni`) builds
the two-stage pipeline, then stops:

```text
vllm_omni/platforms/interface.py, in get_omni_ar_worker_cls
NotImplementedError
```

`TTOmniPlatform` (#200) supplies that worker class (#202).
