# DiffusionGemma Block Serving

DiffusionGemma is a block-output model. One model invocation commits a complete
256-token canvas, not one autoregressive token. Its tt-metal adapter declares:

- `output_tokens_per_step=256`
- `supports_sample_on_device=True`
- `supports_async_decode=False`
- `supports_prefix_caching=False`

The adapter owns the Gumbel sampler and temperature schedule, prompt and canvas
KV state, denoise loop, and persistent Metal captures. vLLM schedules physical
canvases and trims the final canvas to the request's logical `max_tokens`.

## Installation

Follow [Environment Setup in the main README](../README.md#environment-setup):
activate the tt-metal environment, then run `source docs/install-vllm-tt.sh`
to install the pinned vLLM 0.25.1 build and this plugin.

## Launch

Run from the paired tt-metal checkout so the registered model target under
`models.experimental.diffusion_gemma` is importable.

```bash
export PYTHONPATH=/path/to/tt-metal
export MESH_DEVICE=P150x4
export DG_UPFRONT_COARSE_PREFILL_BUCKETS=1
export DG_UPFRONT_LAZY_PREFILL_RECAPTURE=1
export DG_PREFILL_FIXED_CHUNKS=1
export DG_PREFILL_RAGGED_CHUNK=1024
export DG_UPFRONT_PREFILL_WARMUP_LENS=32,64,96
export DG_TRACE_REGION_SIZE=3758096384

python -m vllm.entrypoints.openai.api_server \
  --model google/diffusiongemma-26B-A4B-it \
  --served-model-name diffusiongemma-26B-A4B-it \
  --generation-config vllm \
  --max-model-len 262144 \
  --max-num-batched-tokens 262144 \
  --max-num-seqs 1 \
  --block-size 64 \
  --no-enable-chunked-prefill \
  --no-async-scheduling \
  --reasoning-parser gemma4 \
  --additional-config '{"tt":{"sample_on_device_mode":"all","enable_model_warmup":true,"trace_mode":"all","trace_region_size":3758096384}}'
```

DiffusionGemma uses the `gemma4` reasoning and tool parsers shipped by vLLM
0.25.1. The plugin does not override or alias upstream parser names. For tool
calling add `--enable-auto-tool-choice --tool-call-parser gemma4`.

`--max-num-batched-tokens` concerns scheduler prompt admission. It does not
disable DiffusionGemma's model-internal ragged and chunked prompt processing.
The `DG_TRACE_REGION_SIZE` environment value must match
`tt.trace_region_size`. The values above are the QB2 (`MESH_DEVICE=P150x4`)
release configuration validated by tt-shield; variables whose validated value
matches the tt-metal default are omitted, as is
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, which the api_server never reads (its
AsyncLLM always runs the engine in a background process; the variable only
affects the offline `LLM` entrypoint). The
`DG_UPFRONT_COARSE_PREFILL_BUCKETS`, `DG_UPFRONT_LAZY_PREFILL_RECAPTURE`, and
`DG_PREFILL_FIXED_CHUNKS` gates (all default off) are required to serve
arbitrary prompt lengths; without them only prompts whose execution length is
in `DG_UPFRONT_PREFILL_WARMUP_LENS` are served, and any other prompt returns
an empty completion. If the expected aligned prompt lengths change, update
`DG_UPFRONT_PREFILL_WARMUP_LENS` accordingly. With
`DG_UPFRONT_COARSE_PREFILL_BUCKETS=1`, `--max-model-len` must be a power of
two no larger than 262144, as in the release configuration: the adapter
resolves each prefill to a power-of-two bucket capped at `max_model_len`, so
under a non-power-of-two limit an admitted prompt aligned past the largest
bucket has no servable execution shape and the failure is engine-fatal.

## Request Contract

Current support is synchronous, DP=1, one active sequence, on-device sampling
for all model calls, and no scheduler chunked prefill; vLLM automatic prefix
caching is disabled by the platform. Async scheduling, host sampling, custom
logits processors, and non-uniproc executors are rejected at launch;
`prompt_embeds` inputs and streaming-input (resumable) sessions are rejected
per request; other unsupported per-request controls are listed below.

HTTP sampling controls are accepted for OpenAI-client compatibility but ignored:

- `temperature`, `top_p`, `top_k`, and `min_p`
- `seed`
- presence, frequency, and repetition penalties

The model-owned denoise loop always uses its internal temperature schedule and
Gumbel sampler, so these fields do not alter generation. The plugin ignores
them and logs a one-time warning. Controls that would change the response
contract, such as `n>1`, logprobs, structured outputs, bad words, logit bias,
allowed token IDs, nonzero minimum tokens, and custom sampling `extra_args`,
are rejected.

Physical admission rounds the prompt to a TT tile, rounds the configured limit
down to a TT tile, and rounds logical output up to complete canvases:

```text
ceil(prompt_tokens / 32) * 32
  + ceil(max_tokens / 256) * 256
  <= floor(max_model_len / 32) * 32
```

The historical `/v1/completions` default `max_tokens=16` remains valid. It
runs one physical canvas and returns at most 16 logical tokens. Over the
OpenAI endpoints any output limit — omitted or explicit — is capped to the
largest whole-canvas capacity that fits; the formula rejects an oversized
`max_tokens` only for offline `SamplingParams` callers.

## Metrics

vLLM request metrics count logical output tokens after EOS, stop-token, and
`max_tokens` trimming. A stop *string* truncates only the returned text: the
full committed canvas still counts toward token metrics, up to 255 tokens
past the match. DiffusionGemma model telemetry is physical: one block is 256
committed tokens. Use the adapter's `prefill_block0`, `decode_block`,
`block_ids`, `committed_tokens`, `denoise_steps`, and block-latency events
for device performance. Report block throughput as:

```text
physical tokens per block second = 256 / block_latency_seconds
```

Do not report autoregressive TPOT for this model.

## Current Limitations

- The serving constraints in [Request Contract](#request-contract) apply;
  speculative decoding and preemption overlap are additionally unsupported.
- A forced prefix-cache reset aborts running block requests instead of
  resuming them: a half-generated canvas cannot resume, so the engine — a
  background `EngineCoreProc` for the api_server and the default offline
  `LLM` alike — finishes those requests with an abort before resetting. Only
  an in-process engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) has no client
  stream to notify and refuses the reset with an error instead.
  `pause_generation(mode="keep", clear_cache=True)` is refused while a block
  request is live; pausing without `clear_cache` works.
- Device server validation requires the paired tt-metal checkout and model
  artifacts. Host plugin tests do not prove model correctness or performance.
