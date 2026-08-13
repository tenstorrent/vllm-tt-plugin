# DiffusionGemma Block Serving

DiffusionGemma is a block-output model. One model invocation commits a complete
256-token canvas, not one autoregressive token. The paired implementation is
`tenstorrent/tt-metal` commit
`e37613fd2973d969c362a127b2d0c401a5e145d6`. Its adapter declares:

- `output_tokens_per_step=256`
- `supports_sample_on_device=True`
- `supports_async_decode=False`
- `supports_prefix_caching=False`

The adapter owns the Gumbel sampler and temperature schedule, prompt and canvas
KV state, denoise loop, and persistent Metal captures. vLLM schedules physical
canvases and trims the final canvas to the request's logical `max_tokens`.

## Installation

Install vLLM 0.24.0 into an activated tt-metal environment, then install this
plugin. Do not install a CUDA vLLM wheel over that environment.

```bash
VLLM_TARGET_DEVICE=empty uv pip install --no-binary vllm vllm==0.24.0
uv pip install -e .
```

## Launch

Run from the paired tt-metal checkout so the registered model target under
`models.experimental.diffusion_gemma` is importable.

```bash
export PYTHONPATH=/path/to/tt-metal
export MESH_DEVICE=P150x4

python -m vllm.entrypoints.openai.api_server \
  --model <diffusion-gemma-checkpoint> \
  --served-model-name diffusiongemma-26B-A4B-it \
  --generation-config vllm \
  --max-model-len <served-context-limit> \
  --max-num-batched-tokens <served-context-limit> \
  --max-num-seqs 1 \
  --block-size 64 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --no-async-scheduling \
  --reasoning-parser gemma4 \
  --additional-config '{"tt":{"sample_on_device_mode":"all","enable_model_warmup":true,"trace_mode":"all"}}'
```

Reasoning and tool-call parsing use the `gemma4` parsers that ship with vLLM
0.24 (the same ones upstream's DiffusionGemma recipe uses); for tool calling
add `--enable-auto-tool-choice --tool-call-parser gemma4`. The plugin
registers no parsers of its own.

The plugin pins `VLLM_USE_V2_MODEL_RUNNER=0` because vLLM 0.24 otherwise
classifies any model with `canvas_length` as a Triton-only V2 diffusion runner.
The TT worker implements the V1 runner.

`--max-num-batched-tokens` concerns scheduler prompt admission. It does not
disable DiffusionGemma's model-internal ragged and chunked prompt processing.
Set the trace-region and DiffusionGemma capture environment variables required
by the paired tt-metal model for the target context and mesh.

## Request Contract

Current support is synchronous, DP=1, one active sequence, on-device sampling
for all model calls, and no vLLM automatic prefix caching or scheduler chunked
prefill. Async scheduling, structured output, vLLM logprobs, host sampling,
custom logits processors, and multiple responses are rejected.

HTTP sampling controls are accepted for OpenAI-client compatibility but ignored:

- `temperature`, `top_p`, `top_k`, and `min_p`
- `seed`
- presence, frequency, and repetition penalties

The model-owned denoise loop always uses its internal temperature schedule and
Gumbel sampler, so these fields do not alter generation. Validation neutralizes
them in place before vLLM clones `SamplingParams`; offline callers must not
reuse the same object for a later autoregressive request. Response-contract
controls such as `n>1`, logprobs, structured outputs, bad words, logit bias,
allowed token IDs, nonzero minimum tokens, and custom sampling `extra_args`
remain rejected before EngineCore.

Physical admission rounds logical output up to complete canvases:

```text
prompt_tokens + ceil(max_tokens / 256) * 256 <= max_model_len
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

- One request at a time (`max_num_seqs=1`) and DP=1.
- No generic async block scheduling or preemption overlap.
- No vLLM APC, scheduler chunked prefill, speculative decoding, or host-side
  sampling controls.
- A running block request prevents forced prefix-cache reset; finish or abort
  it first.
- Device server validation requires the paired tt-metal checkout and model
  artifacts. Host plugin tests do not prove model correctness or performance.
