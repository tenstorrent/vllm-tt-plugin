# vLLM-Omni install for the Qwen3-TTS PoC

PoC-branch recipe for tenstorrent/vllm-tt-plugin#190. It moves to `docs/`
only when the Omni backend lands.

## Pinned pair

| Package | Version |
|---|---|
| vLLM | 0.30.0 (`docs/install-vllm-tt.sh`) |
| vLLM-Omni | 0.30.0 |
| Python | **3.12** |

### Python 3.12 is required

vLLM-Omni 0.30.0 declares `>=3.10`, but its CLI does not import on 3.10:

```text
vllm_omni/engine/duplex/commands.py", line 84
TypeError: multiple bases have instance lay-out conflict
```

The fix (vllm-project/vllm-omni#7985) is in `v0.31.0rc1`, not in `v0.30.0`.
See vllm-project/vllm-omni#7475.

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

# 2. vLLM-Omni without its declared dependencies, and transformers held at the
#    tt-metal pin. Do this in a --no-deps step: docs/vllm-overrides.txt
#    overrides every transformers requirement, an explicit ==5.12.1 included.
uv pip install --no-deps "vllm-omni==0.30.0" "transformers==5.12.1"

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
# 0.30.0 0.30.0
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
| `transformers` | `>=5.13,<5.15` | 5.12.1 | tt-metal pins 5.12.1 |
| `diffusers` | `==0.40.0` | 0.38.0 | tt-metal ships 0.38.0 |

`import vllm_omni`, all `qwen3_tts` model modules, and the two-stage plan
work with transformers 5.12.1. Model execution on 5.12.1 is not yet verified.

### Not installed

Omni declares these, and Qwen3-TTS does not import them: `fa3-fwd`,
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

A TT Omni platform supplies that worker class (#193).
