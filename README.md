# vLLM TT Plugin

Tenstorrent backend plugin for vLLM.

`vllm-tt-plugin` integrates Tenstorrent hardware into vLLM using the standard
plugin mechanism. Install it alongside vLLM and, when `ttnn` is importable
([see TT-Metal](https://github.com/tenstorrent/tt-metal/tree/main)),
TT hardware is automatically available as a vLLM platform.

The plugin is self-contained: model registration, platform detection, request
validation, scheduling, worker execution, model loading, async decode, single-
process standard and multi-lane execution, data-parallel execution all live
here. Nothing TT-specific needs to touch vLLM core.

## Package Layout

```text
.
+-- src/vllm_tt_plugin/
|   +-- entrypoints.py       # vLLM plugin entry points
|   +-- platform.py          # TTPlatform and config validation
|   +-- model_registry.py    # TT model architecture registration
|   +-- worker.py            # TT worker implementation
|   +-- model_runner.py      # TT model execution bridge
|   +-- scheduler.py         # TT scheduling policy
|   +-- lane_scheduler.py    # Single-process multi-lane (lane-DP) coordinator
|   +-- launcher.py          # retained tt-run / MPI launcher (not hooked by vLLM 0.24)
|   +-- loader.py            # TT model loader
|   +-- input_batch.py       # TT input-batch representation
|   +-- async_decode.py      # Decode overlap helpers
|   +-- config.py            # TT plugin config access
|   +-- utils/               # Common helpers such as device discovery tools for DP
+-- docs/                    # TT runtime notes
+-- examples/                # Offline and OpenAI-server examples
+-- tests/tt/                # Server-facing TT plugin tests
```

## Requirements

If testing a specific model, check the
[TT-Metal LLMs table](https://github.com/tenstorrent/tt-metal?tab=readme-ov-file#llms)
for the appropriate tt-metal and vLLM commits.

vLLM requires Python `>=3.10,<3.14`. Python 3.10.12 is the default `python3` on
Ubuntu 22.04.

The installation script builds vLLM `0.24.0` from source with
`VLLM_TARGET_DEVICE=empty`. Other vLLM versions are not tested.

## Environment Setup

Install tt-metal first by following
[INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
If installing tt-metal from source, build it, create the virtual environment,
and set the environment variables needed for tt-metal tests.

Activate the environment where tt-metal is installed, then install vLLM
and the TT plugin. Run it from the repository root, which the relative paths
inside assume:

```bash
source docs/install-vllm-tt.sh
```

The script installs vLLM with
`VLLM_TARGET_DEVICE=empty` because `tt` platform is provided by this plugin
at runtime. It then installs the plugin with a few dependencies.
Most dependencies come from the active tt-metal env.

The script also installs vLLM's dependency list itself, fetched from
`requirements/common.txt` at the pinned vLLM tag, and installs vLLM with
`--no-deps`. Resolving them the usual way pulls vLLM's CUDA dependency set
(torch pinned, `flashinfer`, `tilelang`, `nvidia-*`), which would fight the
tt-metal env over `torch` and add several GB. This means the script needs network
access to `raw.githubusercontent.com` beyond the package index. When installing
inside a container, also set `UV_NO_CACHE=1` to keep the uv cache out of the
image layer.

To install or refresh only the plugin package:

```bash
uv pip install -e .
```

To run the offline Qwen-VL example, also install its extra:

```bash
uv pip install -e ".[examples]"
```

After the first setup, activate the same environment before running vLLM:

```bash
source "$PYTHON_ENV_DIR/bin/activate"
```

`VLLM_TARGET_DEVICE` is a build-time variable only and does not need to be set at
runtime. The TT platform is detected automatically when `ttnn` is importable.

Install pre-commit hooks once in your active development environment. After
installation, these checks run automatically on every `git commit`:

```bash
uv pip install pre-commit
pre-commit install
```

## Verify Plugin Discovery

The editable install registers two vLLM entry points:

| Entry point group | Name | Target |
| --- | --- | --- |
| `vllm.general_plugins` | `tt_model_registry` | `vllm_tt_plugin.entrypoints:register` |
| `vllm.platform_plugins` | `tt` | `vllm_tt_plugin.entrypoints:platform_plugin` |

`platform_plugin()` returns `vllm_tt_plugin.platform.TTPlatform` only when
`ttnn` is importable. This keeps ordinary vLLM environments from accidentally
selecting the TT platform.

Quick checks:

```bash
python -c "import vllm_tt_plugin; print(vllm_tt_plugin.__file__)"
python -c "import ttnn; print('ttnn available')"
```

If `VLLM_PLUGINS` is set, it must allow both TT entry point names:

```bash
export VLLM_PLUGINS=tt,tt_model_registry
```

## Hugging Face Access

To run Meta Llama 3.1 or 3.2 models, request access on Hugging Face:

- [Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-70B)
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Llama 3.2 Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

After access is approved, create an access token in Hugging Face settings and
log in from Python:

```python
from huggingface_hub import login

login()
```

## Preparing TT-Metal Models

For the target model, follow any setup instructions in the corresponding
tt-metal demo. For Llama 3.1, Llama 3.2, and Qwen 2.5 models, follow the
[tt-transformers demo instructions](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)
for weights and environment variables.

## Running The Offline Inference Example

Run offline generation with the default Llama 3.1 70B model:

```bash
MESH_DEVICE=T3K python examples/offline_inference_tt.py
```

Measure offline performance for one batch of prompts:

```bash
MESH_DEVICE=T3K \
python examples/offline_inference_tt.py --measure_perf
```

To run a different text model, set `MESH_DEVICE` to `N150`, `N300`, `T3K`, `TG`,
`BH-Galaxy`, or a mesh shape such as `"(4,8)"`, then pass `--model`:

- Llama 3.1 8B: `--model "meta-llama/Llama-3.1-8B"`
- Llama 3.2 1B: `--model "meta-llama/Llama-3.2-1B"`
- Llama 3.2 3B: `--model "meta-llama/Llama-3.2-3B"`
- Qwen 2.5 7B: `--model "Qwen/Qwen2.5-7B"`
- Qwen 2.5 72B: `--model "Qwen/Qwen2.5-72B"`
- DeepSeek R1 Distill Llama 70B: `--model "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"`
- GPT-OSS 20B: `--model "openai/gpt-oss-20b"`
- GPT-OSS 120B: `--model "openai/gpt-oss-120b"`

For Llama 3.1 8B on N150, set `--max_model_len 32768`; see the tt-metal model
demo for context-length details.

To run Llama 70B on Galaxy:

```bash
MESH_DEVICE=TG \
LLAMA_DIR=<path-to-weights> \
TT_LLAMA_TEXT_VER=llama3_70b_galaxy \
python examples/offline_inference_tt.py \
  --model "meta-llama/Llama-3.1-70B-Instruct" \
  --additional-config '{"tt": {"dispatch_core_axis": "col", "sample_on_device_mode": "all", "fabric_config": "FABRIC_1D_RING", "worker_l1_size": 1344544, "trace_region_size": 216580672}}'
```

To run GPT-OSS 20B on Galaxy:

```bash
MESH_DEVICE="(4,8)" \
python examples/offline_inference_tt.py \
  --model "openai/gpt-oss-20b" \
  --max_seqs_in_batch 1 \
  --additional-config '{"tt": {"fabric_config": "FABRIC_1D_RING"}}'
```

Run Llama 3.2 Vision on N300:

```bash
MESH_DEVICE=N300 \
python examples/offline_inference_tt.py \
  --model "meta-llama/Llama-3.2-11B-Vision-Instruct" \
  --multi_modal \
  --max_seqs_in_batch 16 \
  --num_repeat_prompts 8
```

Useful vision-model variants:

- Llama 3.2 11B Vision on QuietBox: set `MESH_DEVICE=T3K` and `--max_seqs_in_batch 32`.
- Llama 3.2 90B Vision: set `MESH_DEVICE=T3K`, `--model "meta-llama/Llama-3.2-90B-Vision-Instruct"`, and `--max_seqs_in_batch 4`.
- Qwen 2.5-VL 32B: set `MESH_DEVICE=T3K`, `--model "Qwen/Qwen2.5-VL-32B"`, and `--max_seqs_in_batch 32`.
- Qwen 2.5-VL 72B: set `MESH_DEVICE=T3K`, `--model "Qwen/Qwen2.5-VL-72B"`, `--max_seqs_in_batch 32`, `--max_model_len 2048`, and `--additional-config '{"tt": {"trace_region_size": 28467200}}'`.
- Gemma 3 27B: set `MESH_DEVICE=T3K`, `--model "google/gemma-3-27b-it"`, `--max_seqs_in_batch 32`, `--additional-config '{"tt": {"l1_small_size": 768, "fabric_config": "FABRIC_1D"}}'`, `--multi_modal`, `--multi_image`, and `--mm_processor_kwargs '{"use_fast": true, "do_convert_rgb": true}'`.

For debugging V1, set `VLLM_ENABLE_V1_MULTIPROCESSING=0` to disable
multiprocessing. This is useful for stepping through code or making scheduling
deterministic, but it is not compatible with DP models.

## Running The Server Example

Start the OpenAI-compatible server:

```bash
VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=T3K \
python examples/server_example_tt.py
```

DiffusionGemma uses a 256-token block-serving contract with stricter launch and
request constraints. See [DiffusionGemma block serving](docs/diffusion-gemma.md)
for the exact vLLM 0.24 command, request validation, and block metrics.

Send a completion request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "prompt": "San Francisco is a",
    "max_tokens": 32,
    "temperature": 1,
    "top_p": 0.9,
    "top_k": 10
  }'
```

Requests that cannot use TT on-device sampling automatically fall back to vLLM’s host-side sampling path. This fallback is selected per batch and requires no user configuration.

For vision models, start the server with the correct `--model`, then send a chat
completion request with image content. Qwen 2.5-VL models can use either a
base64 `data:image/...` URL or a real URL such as
`https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg`.

## Configuration

TT options are passed through vLLM's generic additional config namespace:

```bash
--additional-config '{"tt": {"sample_on_device_mode": "all"}}'
```

Plugin code reads this through `vllm_tt_plugin.config.get_tt_config()`, which
returns `vllm_config.additional_config["tt"]`.

Common options:

| Key | Purpose |
| --- | --- |
| `sample_on_device_mode` | Select on-device sampling mode, currently `all` or `decode_only` when supported by the model. |
| `trace_mode` | Control TT tracing: `all`, `decode_only`, or `none`. Default: `all`. |
| `enable_model_warmup` | Warm up the model before the server reports healthy. Default: `true`. |
| `trace_region_size` | Trace region size for TT runtime tracing. |
| `worker_l1_size` | Worker L1 size override. |
| `l1_small_size` | Small L1 size override. |
| `fabric_config` | Fabric config such as `DISABLED`, `FABRIC_1D`, `FABRIC_2D`, `FABRIC_1D_RING`, or `CUSTOM`. |
| `fabric_reliability_mode` | Fabric reliability mode, such as `STRICT_INIT` or `RELAXED_INIT`. |
| `dispatch_core_axis` | Dispatch core axis, `row` or `col`. |
| `always_compat_sampling` | Use vLLM's LogitProcessor and sampler path even when not required by the batch. Default: `false`. |
| `optimizations` | Select model/runtime optimization profile, such as `accuracy` or `performance`. |
| `register_test_models` | Register non-production TT test models for infrastructure tests. Default: `false`. |

### `max_model_len` And KV Cache Capacity

TT does not profile device memory. `get_num_available_blocks_tt()` instead
derives a block count from the model's `max_tokens_all_users` — a per-model,
per-device token budget declared by the model class in tt-metal
(`get_max_tokens_all_users`, default 131072, sometimes exposed as an environment
override such as `GEMMA4_MAX_TOKENS_ALL_USERS`), plus headroom for vLLM's
worst-case block allocation and for sliding-window groups on hybrid models.
`TTWorker.determine_available_memory()` publishes that count both as
`cache_config.num_gpu_blocks_override` and as an equivalent byte budget — the
latter is what reaches the engine-side KV planner, which in standard DP runs in
a different process from the worker that set the override.

That pool is the budget shared by *all* concurrent requests, and it is
independent of `max_model_len`, the *per-request* context limit. Since vLLM
sizes and validates the KV cache against it, the two must be chosen together:

| `--max-model-len` | Behavior |
| --- | --- |
| numeric, fits the pool | Used as given. Maximum concurrency is roughly `pool / max_model_len`. |
| numeric, exceeds the pool | Startup fails in `get_kv_cache_configs` with the estimated maximum servable length. Block-output models fail earlier, in the TT worker: they must fit `max_model_len` plus one output canvas. Lower `--max-model-len` or raise `max_tokens_all_users`. |
| `-1` | vLLM auto-fits `max_model_len` to the pool and syncs the result to the workers and the API-server process. |
| omitted | vLLM uses the HF-derived value. Startup fails if that value exceeds the pool. |

Explicit `-1` auto-fit picks the largest length that fits **one** request, so it
lands at roughly the whole pool with maximum concurrency `1.00x`: a single long
request can occupy the entire KV cache and stall the rest. Prefer an explicit
positive `--max-model-len` for serving, sized around
`max_tokens_all_users / max_num_seqs`, and treat auto-fit as a convenience for
one-off runs.

## Runtime Architecture

`TTPlatform.check_and_update_config()` is the main handoff from vLLM into the TT
runtime. It validates configuration, registers TT model architectures, and
selects the TT-owned runtime classes through vLLM's extension points:

| vLLM config field | TT implementation |
| --- | --- |
| `parallel_config.worker_cls` | `vllm_tt_plugin.worker.TTWorker` |
| `scheduler_config.scheduler_cls` | `vllm_tt_plugin.scheduler.TTScheduler` or `vllm_tt_plugin.lane_scheduler.TTLaneCoordinator` |

The execution model matches TT hardware characteristics:

- A TT step is either prefill-only or decode-only.
- Token-chunked prefill is available for Gemma 4: a long prompt is split across
  prefill steps and only the chunk that completes the prompt emits a token.
  Every other model type keeps prefill unsplit.
- Async scheduling overlaps decode submission with host-side scheduling when
  the model declares support.
- For Galaxy-generator models (Llama3 70B, Qwen3-32B) and GPT-OSS,
  `--data_parallel_size N` runs as `N` in-process TT lanes scheduled by
  `TTLaneCoordinator` (one engine, one device mesh); see
  [Single-Process Galaxy Serving](#single-process-galaxy-serving).
- For other models, `--data_parallel_size N` uses standard multi-process DP:
  each DP rank runs an independent engine core with its own TT submesh,
  scheduler, and KV cache. Device groups are discovered at startup and assigned
  via `TT_VISIBLE_DEVICES`. A per-rank assignment is accepted only when it
  exactly matches the discovered group; a conflicting GPU-style `--device-ids`
  assignment fails before mesh creation. This is upstream vLLM's standard DP
  mechanism (no gather/scatter; ranks are fully independent).

For a deeper walk-through of the scheduling and execution model, read
`docs/SCHEDULING.md`.

## Single-Process Galaxy Serving

Galaxy text models served by the single-execute Galaxy generator
(Llama 3.3 70B via `TT_LLAMA_TEXT_VER=llama3_70b_galaxy`, Qwen3-32B via
`TT_QWEN3_TEXT_VER=qwen3_32b_galaxy`) run on a single Galaxy device mesh, so
they use single-process TT lanes: one vLLM engine process with internal TT
lanes.

Serve them with the familiar `--data_parallel_size N --max_num_seqs M` flags;
the TT backend transparently maps them to `N` in-process lanes. No config
changes are needed:

```bash
MESH_DEVICE=TG \
TT_LLAMA_TEXT_VER=llama3_70b_galaxy \
VLLM_RPC_TIMEOUT=900000 \
python examples/server_example_tt.py \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --data_parallel_size 4 \
  --max_num_seqs 8 \
  --async-scheduling \
  --additional-config '{"tt": {"dispatch_core_axis": "col", "sample_on_device_mode": "all", "fabric_config": "FABRIC_1D_RING", "worker_l1_size": 1344544, "trace_region_size": 220000000}}'
```

`--data_parallel_size 4 --max_num_seqs 8` runs `4` TT lanes of `8` requests
each (`32` concurrent total); `--max_num_seqs` is the per-lane capacity. This
conversion is specific to the Galaxy generators and GPT-OSS; other model
families still run `--data_parallel_size` as multi-process DP.
At startup the backend logs that it is running single-process lane-DP.

## Supported Model Families

The plugin registers TT-prefixed model architectures backed by tt-metal model
implementations. Current families:

- Llama 3.1 / 3.2 / 3.3 text models (`TTLlamaForCausalLM`)
- Llama 3.2 vision models (`TTMllamaForConditionalGeneration`)
- Qwen 2.5 and Qwen 3 text models (`TTQwen2ForCausalLM`, `TTQwen3ForCausalLM`)
- Qwen 3.5 text models on Blackhole (`TTQwen3_5ForConditionalGeneration`)
- Qwen 2.5-VL and Qwen 3-VL vision-language models
- Mistral and Mistral 3 multimodal models
- Gemma 3 multimodal models
- Gemma 4 text-only models (`TTGemma4ForCausalLM`,
  `TTGemma4ForConditionalGeneration`,
  `TTGemma4UnifiedForConditionalGeneration`)
- DiffusionGemma block-output models (`TTDiffusionGemmaForBlockDiffusion`,
  `TTDiffusionGemmaForCausalLM`)
- DeepSeek V3 (`TTDeepseekV3ForCausalLM`)
- GPT-OSS 20B / 120B (`TTGptOssForCausalLM`)

Model availability, supported device shapes, max sequence limits, and required
environment variables are documented in the corresponding tt-metal model demos.

### Registering models dynamically (`EXTRA_MODELS_DIR`)

Instead of adding a hard-coded line to `platform.py`, a model can be registered at
startup by dropping a **bundle folder** under a directory named by the
`EXTRA_MODELS_DIR` environment variable. Each subfolder holds a `vllm_metadata.json`
and the adapter class (plus its dependencies):

```text
$EXTRA_MODELS_DIR/
  my-model/
    vllm_metadata.json      # {"arch": "<HFArch>", "main_class": "module:Class", ...}
    <adapter class + deps>
```

At import time the plugin scans `EXTRA_MODELS_DIR`, appends each folder to `sys.path`
(so an installed package of the same name is never shadowed), and registers `arch`
under the plugin's `TT`-prefixed convention (`TT<HFArch>`) pointing at `main_class`.
This lets a distribution tool (e.g. `tt-kernel`) deliver a ready-to-serve model with no
source edit to the plugin. The built-in map above stays enabled by default; set
`TT_VLLM_BUILTIN_MODELS=0` to rely solely on `EXTRA_MODELS_DIR`.

## Operational Constraints

`TTPlatform` rejects or adjusts unsupported feature combinations early, giving a
clear error before anything reaches the device:

- Tensor parallel and pipeline parallel execution are not supported.
- Speculative decoding is not currently supported.
- LoRA is not currently supported.
- Chunked prefill is disabled for every model type except Gemma 4, and
  `max_num_batched_tokens` is bumped to `max_model_len` when it is disabled.
- Where chunked prefill is active, multimodal inputs are never split across a
  chunk boundary.
- Prompt logprobs are rejected at request validation time.
- Prefix caching is enabled only for models that declare TT support for it.
- Async decode overlap is enabled only for models that declare the capability.

These are TT runtime characteristics, not vLLM plugin API limitations.

## Benchmarking

Offline benchmarking is done by passing `--measure_perf` to
`offline_inference_tt.py`:

```bash
MESH_DEVICE=T3K \
python examples/offline_inference_tt.py \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --measure_perf
```

Client-server benchmarking can be done with `vllm bench serve` after starting
the server:

```bash
vllm bench serve --model meta-llama/Llama-3.1-70B-Instruct \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 128 \
  --num-prompts 32 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el
```

For prefix-cache experiments, use prompts with shared prefixes:

```bash
python examples/offline_inference_tt.py \
  --prompts_json examples/prompts_overlapping.json
```

You can also pass `--random-prefix-len <N>` to `vllm bench serve`.

## Testing

The plugin ships server-facing tests under `tests/tt`. Start a vLLM server with
a TT model, then run:

```bash
pytest tests/tt -v \
  --tt-server-url=http://localhost:8000 \
  --tt-model-name=meta-llama/Llama-3.1-8B-Instruct
```

Tests cover request isolation, sampling behavior, penalties, logprobs,
host-only parameter handling, and TT utility helpers.

Plugin-local unit tests that do not require a running server live directly
under `tests/`, for example:

```bash
pytest tests/test_lane_scheduler.py
```

## Hybrid Attention Models

Hybrid attention models have mixed sliding-window and full-attention layers
such as Gemma 3, Gemma 4, and GPT-OSS. They opt in to upstream vLLM's hybrid KV
cache manager through a per-model spec hook on the registered TT model class.

The hybrid manager packs sliding and full layers into separate
`KVCacheGroupSpec`s, sized by upstream's
[Hybrid KV Cache Manager design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/).
Sliding-window layers then occupy only `sliding_window` worth of KV state per
request instead of `max_seq_len`. On Gemma 4 31B at 256k context this is roughly
a 6x reduction in KV cache memory.

To enable hybrid KV cache support for a TT model:

1. Inherit from `models.tt_transformers.tt.generator_vllm.HybridAttentionForCausalLM`
   instead of `Generator`. The base class provides a default
   `get_kv_cache_spec` classmethod that builds per-layer specs from
   `hf_config.text_config.layer_types`.
2. Implement `prefill_forward` and `decode_forward` to consume the
   `page_tables_per_group` kwarg and route each layer to the right group's page
   table.
3. Implement `allocate_kv_cache_per_layer(per_layer_specs)`. The base class
   default delegates to `allocate_vllm_kv_cache_per_layer`.

Models that do not opt in stay on the legacy `Generator` path: uniform
single-group KV cache, one page table, and no behavioral change. The plugin only
sends `page_tables_per_group` to model classes that expose `get_kv_cache_spec`.

Hybrid models with `data_parallel_size > 1` have not been validated on
hardware. Both DP modes carry the full per-group block tables (a standard-DP
rank is an independent DP=1 engine, and lane-DP builds per-group tables for the
merged batch), so there is no known blocker, but the combination is untested.


## Development Notes

- Normal Python changes under `src/vllm_tt_plugin/` take effect after restarting
  the Python or vLLM process.
- Reinstall the plugin when package metadata or entry points change, such as
  edits to `pyproject.toml`.
- Model capability declarations (`model_capabilities` dict on the model class)
  are the preferred way to gate features like async decode and prefix caching,
  rather than hard-coded model-name checks.

## Contributing

Contributions are welcome! Bug reports and feature requests should be filed via
[GitHub Issues](https://github.com/tenstorrent/vllm-tt-plugin/issues). Bug
fixes and new functionality are submitted via pull requests. Pull requests are
reviewed weekly. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

- [LICENSE](LICENSE) — Overall license for this project (Apache 2.0), except
  where specified
- [LICENSE_understanding.txt](LICENSE_understanding.txt) — Tenstorrent's
  clarification of how the Apache 2.0 license applies to this repository
