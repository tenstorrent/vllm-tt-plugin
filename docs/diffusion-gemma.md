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
to install the pinned vLLM 0.24.0 build and this plugin.

## Launch

Run from the paired tt-metal checkout so the registered model target under
`models.experimental.diffusion_gemma` is importable.

```bash
export PYTHONPATH=/path/to/tt-metal
export MESH_DEVICE=P150x4
export VLLM_RPC_TIMEOUT=1800000
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
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --no-async-scheduling \
  --reasoning-parser gemma4 \
  --additional-config '{"tt":{"sample_on_device_mode":"all","enable_model_warmup":true,"trace_mode":"all","trace_region_size":3758096384}}'
```

DiffusionGemma uses the `gemma4` reasoning and tool parsers shipped by vLLM
0.24. The plugin does not override or alias upstream parser names. For tool
calling add `--enable-auto-tool-choice --tool-call-parser gemma4`.

`--max-num-batched-tokens` concerns scheduler prompt admission. It does not
disable DiffusionGemma's model-internal ragged and chunked prompt processing.
The `DG_TRACE_REGION_SIZE` environment value must match
`tt.trace_region_size`; the values above are the QB2 (`MESH_DEVICE=P150x4`)
release configuration validated by tt-shield. If the expected aligned prompt
lengths change, update `DG_UPFRONT_PREFILL_WARMUP_LENS` accordingly.

## Request Contract

Current support is synchronous, DP=1, one active sequence, on-device sampling
for all model calls, and no vLLM automatic prefix caching or scheduler chunked
prefill. Async scheduling, host sampling, and custom logits processors are
rejected at launch; unsupported per-request controls are listed below.

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
runs one physical canvas and returns at most 16 logical tokens. An omitted
output limit is clamped to the largest whole-canvas capacity that fits.

## Metrics

vLLM request metrics count client-visible logical output tokens after EOS,
stop, and `max_tokens` trimming. DiffusionGemma model telemetry is physical:
one block is 256 committed tokens. Use the adapter's `prefill_block0`,
`decode_block`, `block_ids`, `committed_tokens`, `denoise_steps`, and
block-latency events for
device performance. Report block throughput as:

```text
physical tokens per block second = 256 / block_latency_seconds
```

Do not report autoregressive TPOT for this model.

## Current Limitations

- The serving constraints in [Request Contract](#request-contract) apply;
  speculative decoding and preemption overlap are additionally unsupported.
- A forced prefix-cache reset aborts running block requests instead of
  resuming them: a half-generated canvas cannot resume, so the engine finishes
  those requests with an abort before resetting. A deferred
  `pause_generation(mode="keep")` reset returns `False` without touching the
  retained requests.
- Device server validation requires the paired tt-metal checkout and model
  artifacts. Host plugin tests do not prove model correctness or performance.
