# Model execution

`TTModelRunner` translates scheduled vLLM requests into model adapter calls.
The model adapter executes prefill or decode and returns logits or sampled
tokens in the submitted row order. The plugin selects sampling for the whole
submitted batch and applies the resulting tokens to vLLM request state.
Start with the [interface guide](MODEL_INTERFACE.md) for setup and lifecycle.
The plugin rejects encoder-decoder models and `prompt_embeds` input.

## Rows, positions, and page tables

`B` below means the number of submitted tensor rows, including decode padding.
`V` means vocabulary width. These are notation for this page, not API names.

- Ordinary execution packs active request rows at the front. Decode may pad
  to the configured capacity or the nearest `tt_supported_decode_batch_sizes`
  value that fits. Publish only batch sizes prepared on the loaded instance.
- Lane-DP prefill packs only scheduled requests. Lane-DP decode uses merged
  stable slots, so inactive rows can occur between active rows. Standard
  multi-process DP gives each worker an independent model and batch.
- `start_pos=-1` marks an inactive decode row. The model adapter must not
  interpret an inactive row as a live request or modify another request's state.
- `page_table` is the first KV group's integer logical-to-physical block
  table. If the resolved cache has multiple groups, the plugin also sends
  `page_tables_per_layer`, ordered by attention-layer index. The model adapter
  must use the page table for each layer. See [KV setup](MODEL_SETUP.md).

`TTModelInput` is an internal plugin payload. The model adapter receives the
keyword arguments below, not a `TTModelInput` object. Source:
[model_input.py](../src/vllm_tt_plugin/model_input.py),
`TTModelRunner._prepare_model_inputs` in
[model_runner.py](../src/vllm_tt_plugin/model_runner.py), and
`TTLaneInputBatch` in [input_batch.py](../src/vllm_tt_plugin/input_batch.py).

## Prefill

`TTModelRunner.submit_prefill` calls `prefill_forward` with these keywords:

| Keyword | Meaning |
|---|---|
| `tokens` | CPU integer tensor `[B, T]`, containing token history from position 0 through the largest scheduled chunk end. |
| `start_pos` | One absolute chunk start per row. Front-packed prefill supplies a NumPy array; lane-DP supplies a torch tensor. |
| `prompt_lens` | One exclusive chunk end per row, supplied as a NumPy array. `prompt_lens[i]` is not the chunk length. |
| `page_table`, `kv_cache` | Block addresses and the object returned by the adapter's KV allocator. |
| `enable_trace` | True when `tt.trace_mode="all"`. |
| `empty_slots` | When supplied, one destination device-state slot per submitted row. Preserve state belonging to other slots. |
| `page_tables_per_layer` | Supplied for multiple KV groups, as described above. |
| `sampling_params` | Supplied only when this prefill uses device sampling. See the sampling section. |
| `stage_inputs` | Supplied only when the model declares `supports_stage_inputs`. One entry per row: the request's stage payload as a plain dict, or `None`. The same payload arrives with every prefill of a request, so a chunked or resumed prefill sees it again. Decode receives none; see [stage inputs](MODEL_LIFECYCLE.md#stage-inputs). |

For row `i`, `prefill_forward` computes
`tokens[i, start_pos[i]:prompt_lens[i]]`. For example, `start_pos[i]=128` and
`prompt_lens[i]=192` request 64 tokens while retaining absolute positions.
Prefix-cache hits and resumed requests can also have nonzero `start_pos`.
Resume prefill can replay generated tokens to rebuild KV state.

The model adapter returns host logits `[B, 1, V]` for host sampling, or sampled
token IDs for device sampling. The plugin has no separate prefill readback
hook. `TTModelRunner` suppresses output for intermediate prefill chunks and
selects host sampling when a submitted prefill contains intermediate chunks,
so intermediate chunks must not advance a device sampler.

### Image inputs and request-specific RoPE

For multimodal prefill, `_gather_multi_modal_inputs` adds `pixel_values` and
`image_grid_thw`. Each value is a list in submitted request order. A text-only
request has `None`; an image request has a list per image, with `None` for
missing payloads. The plugin currently accepts image features only.

When `model_config.uses_mrope` enables request-specific RoPE, `prefill_forward`
returns `(output, rope_deltas)`, with one delta per request. On decode,
`TTAsyncDecodeController` supplies `rope_deltas_all_users` after a layout
change or a new request, and `None` otherwise. Lane-DP rejects
request-specific RoPE.

## Decode

`TTAsyncDecodeController.submit_decode` calls `decode_forward` with:

| Keyword | Meaning |
|---|---|
| `tokens` | CPU integer tensor `[B, 1]` for ordinary decode. |
| `start_pos` | CPU integer tensor `[B]` of absolute positions; inactive rows contain `-1`. |
| `page_table`, `kv_cache` | Block addresses and adapter-owned KV storage. |
| `enable_trace` | True for `tt.trace_mode="all"` or `"decode_only"`. |
| `read_from_device` | `False` for front-packed and asynchronous decode. Synchronous lane-DP requests `True`, except for deferred grammar sampling or selective readback. The adapter must support the readback mode its execution path requests. |
| `page_tables_per_layer` | Supplied for multiple KV groups. |
| `slot_remap` | When present, destination slot `i` reads previous slot `slot_remap[i]`. Supplied for both host and device sampling. See [state ownership](MODEL_LIFECYCLE.md#request-state). |
| `sampling_params` | Supplied for immediate device sampling; absent for host sampling and deferred grammar sampling. |
| `prompt_tokens`, `output_tokens` | Supplied together when device sampling needs token history for penalties. Histories use `-1` padding. |

The [decode reload contract](DECODE_RELOAD_CONTRACT.md) specifies
`decode_input_update_contract`, `reload_inputs`, `reload_page_table`,
`reload_sampling_params`, and `reset_sampling_state`, plus the version-0
`reset_batch` call. The model adapter must execute the negotiated update
commands even when host token tensors lag resident device state.

The [speculative contract](SPEC_DECODE_CONTRACT.md) defines wide verification
inputs, `num_valid_drafts`, `accepted_counts`, `spec_mode`, `VerifyOutput`, and
`propose_draft_tokens`. An ordinary decode receives no `spec_mode` and must
not return `VerifyOutput`.

### Stage outputs

A model that declares `supports_stage_outputs` implements
`pop_stage_outputs()`. `submit_decode` calls it right after every
`decode_forward` and keeps the result with that submission.

- Return one entry per decode row: the stage frame this forward completed
  for the row's request, or `None`. Padding rows must be `None`.
- The call comes before any device synchronization, and the plugin never
  synchronizes or reads a stage frame's `data`. `data` must be host-resident,
  picklable, and own its storage until the consumer reads it. Reading from
  the device inside `pop_stage_outputs()` is allowed; it blocks the step.
- The runner tags each stage frame `(request ID, frame_index)`.
  `frame_index` is the row's decode position minus the request's prompt
  length: the step that consumes output token `k` completes stage frame `k`.
- The stage frames are published on the step's `ModelRunnerOutput`. Read
  them with `stage_outputs.get_tt_stage_outputs`.
- Scheduling is synchronous for these models: they cannot declare
  `supports_async_decode`, without which the platform turns async
  scheduling off.
- Prefill completes no stage frame, and the runner never calls the hook for
  it.

A stage model's request cannot be preempted and resumed: a resume re-prefills
from the token history, which does not hold the stage frames already emitted.
The platform therefore requires `max_num_seqs=1`, refuses a forced
prefix-cache reset while such a request runs (or aborts it first when the
engine can tell its client), and `TTModelRunner` raises if a preemption
reaches it anyway.

## Decode readback and output conversion

1. `TTAsyncDecodeController` calls `decode_forward`. The model adapter may
   return a host torch tensor or an adapter-specific output object.
2. For asynchronous readback, `TTAsyncDecodeController` calls
   `read_decode_output(output, async_read=True)` when the adapter exposes
   `read_decode_output`. The adapter returns `(output_handles, read_events)`.
   Without `read_decode_output`, `decode_forward` must already return host
   torch output. Keep asynchronous output buffers valid until consumption.
3. `TTAsyncDecodeController.finalize_decode` synchronizes each TTNN read event.
   For output that is not already host torch output, the controller calls
   `process_decode_output_host(output_handles, is_tokens=perform_device_sampling)`.
   The adapter returns host logits or sampled IDs. Synchronous decode uses
   this conversion too when `decode_forward` returns adapter-specific output.
4. `TTModelRunner` uses the last position in host logits `[B, S, V]`, commonly
   `[B, 1, V]`, for sampling, or consumes device-sampled IDs `[B, 1]`.
   For `output_tokens_per_step > 1`, the adapter returns
   `[B, output_tokens_per_step]`; adaptive block
   execution also permits one-token prefill and batched decode outputs.

Here `S` is the returned sequence dimension; the front-packed sampler uses
only position `S - 1`. Lane-DP host extraction also accepts `[B, V]` logits.

For device logprobs, the adapter returns `(token_ids, logprobs)` through the
same prefill or decode output boundary. `build_device_logprobs` accepts sampled
logprobs or `(topk_logprobs, topk_indices)` according to the selected model's
support. Exact payload handling is in
[logprobs.py](../src/vllm_tt_plugin/logprobs.py).

The optional `supports_compact_host_logits` and
`supports_selective_host_readback` declarations allow scheduled-row host
output in lane-DP. The [selective-read contract](DECODE_RELOAD_CONTRACT.md#host-sampling-output-and-selective-reads)
defines `sample_rows`, output order, buffer ownership, and the two readback
callbacks. Full-row output remains required when the plugin omits `sample_rows`.

## Sampling and device grammar

`sampling_params` is a `TTSamplingParams` object whose device-facing fields
contain per-row Python lists: `temperature`, `top_k`, `top_p`,
`presence_penalty`, `frequency_penalty`, `repetition_penalty`, `seed`,
`num_logprobs`, and `enable_log_probs`. The plugin converts unspecified seeds
to `None` and pads decode sampling rows with neutral defaults.

`supports_sample_on_device` permits device sampling; the declaration does not
force device sampling. `TTModelRunner.check_perform_device_sampling` selects
one sampling path for the submitted batch. Ordinary adapters must return
logits when the plugin selects host sampling, including for unsupported
device controls. The [capability reference](MODEL_CAPABILITIES.md#sampling-and-async-capabilities)
lists admission and fallback rules. Block-output requests have separate
restrictions because host sampling cannot construct a multi-token block.

For an eligible structured decode:

1. `TTModelRunner.load_model` calls optional `enable_device_grammar()` and
   requires `device_grammar_enabled=True` plus callable `sample_deferred_decode`.
2. `TTAsyncDecodeController` calls `decode_forward(defer_device_sampling=True,
   read_from_device=False, ...)` without `sampling_params`. The adapter retains
   device logits without advancing the sampler.
3. After vLLM supplies the grammar, `TTAsyncDecodeController` calls
   `sample_deferred_decode(output, sampling_params=..., grammar_bitmask=...)`.
   The adapter applies the packed int32 grammar mask and returns sampled
   output through the ordinary decode conversion path.

Each bit in `grammar_bitmask` names one vocabulary token; a one bit permits
that token. The model adapter must apply the submitted row order.

Structured prefill uses host sampling. Deferred grammar sampling requires
synchronous scheduling and effective runtime support. A missing mask or a
failure after deferred submission invalidates the runner's grammar path;
the plugin raises instead of resampling an already advanced step on the host.

Sources: `submit_prefill`, `check_perform_device_sampling`, and
`_get_output_tokens` in [model_runner.py](../src/vllm_tt_plugin/model_runner.py);
`submit_decode`, `finalize_decode`, and `complete_deferred_device_sampling` in
[async_decode.py](../src/vllm_tt_plugin/async_decode.py).

## Adapter sketches: DP=1 and lane-DP

The sketches below are pseudocode. They show how an adapter maps submitted
rows to KV storage and device state in the two single-process layouts. They
omit tracing, device sampling, readback options, multimodal inputs, and
`page_tables_per_layer`. Helper names such as `run_prefill`, `run_decode`, and
`split_mesh` are adapter-internal, not plugin API.

### DP=1 (front-packed)

`tt_data_parallel` is 1. One KV cache serves all rows. Prefill row `i` uses
`page_table[i]` and writes its device state to `empty_slots[i]`, which need
not equal `i`. Decode padding rows have `start_pos=-1`.

```python
class MyModel:
    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size,
                              max_seq_len, tt_data_parallel=1,
                              optimizations=None):
        assert tt_data_parallel == 1
        return cls(hf_config, mesh_device, max_batch_size, max_seq_len)

    def allocate_kv_cache(self, shape, dtype, num_layers):
        # shape = (num_blocks, heads_per_device, block_size, head_size)
        return [alloc_kv(self.mesh, shape, dtype) for _ in range(num_layers)]

    def prefill_forward(self, tokens, start_pos, prompt_lens, page_table,
                        kv_cache, empty_slots, **kwargs):
        logits = []
        for i in range(tokens.shape[0]):
            chunk = tokens[i, start_pos[i]:prompt_lens[i]]
            logits.append(run_prefill(chunk, start_pos[i], page_table[i],
                                      kv_cache, state_slot=empty_slots[i]))
        return stack(logits)  # [B, 1, V] in submitted row order

    def decode_forward(self, tokens, start_pos, page_table, kv_cache,
                       slot_remap=None, **kwargs):
        if slot_remap is not None:
            permute_device_state(slot_remap)  # slot i reads old slot_remap[i]
        active = start_pos >= 0
        return run_decode(tokens, start_pos, page_table, kv_cache, active)
```

### Lane-DP (`tt_data_parallel = L > 1`)

`max_batch_size` is `L * per_lane`. Lane `l` owns slots
`[l * per_lane, (l + 1) * per_lane)`. Each lane scheduler allocates block IDs
from its own lane-local space, so each lane needs separate KV storage. Prefill
rows are front-packed scheduled requests from any lane; `empty_slots[i]` is
the stable slot of row `i`, so its lane is `empty_slots[i] // per_lane`.
Decode submits all `L * per_lane` rows in slot order. Gap rows have
`start_pos=-1` and zeroed page-table rows, and `slot_remap` is the identity.

```python
class MyLaneModel:
    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size,
                              max_seq_len, tt_data_parallel=1,
                              optimizations=None):
        return cls(hf_config, mesh_device, max_batch_size, max_seq_len,
                   lanes=tt_data_parallel)

    def __init__(self, hf_config, mesh_device, max_batch_size, max_seq_len,
                 lanes):
        self.lanes = lanes
        self.per_lane = max_batch_size // lanes
        self.submeshes = split_mesh(mesh_device, lanes)
        self.models = [build_model(m, self.per_lane) for m in self.submeshes]

    def allocate_kv_cache(self, shape, dtype, num_layers):
        # One replica per lane. shape is per lane: heads_per_device uses
        # devices_per_lane, and num_blocks is the lane's block budget.
        return [[alloc_kv(m, shape, dtype) for _ in range(num_layers)]
                for m in self.submeshes]

    def prefill_forward(self, tokens, start_pos, prompt_lens, page_table,
                        kv_cache, empty_slots, **kwargs):
        logits = []
        for i in range(tokens.shape[0]):
            lane = empty_slots[i] // self.per_lane
            chunk = tokens[i, start_pos[i]:prompt_lens[i]]
            logits.append(self.models[lane].run_prefill(
                chunk, start_pos[i], page_table[i], kv_cache[lane],
                state_slot=empty_slots[i] % self.per_lane))
        return stack(logits)  # [B, 1, V] in submitted row order

    def decode_forward(self, tokens, start_pos, page_table, kv_cache,
                       **kwargs):
        outs = []
        for lane in range(self.lanes):
            rows = slice(lane * self.per_lane, (lane + 1) * self.per_lane)
            outs.append(self.models[lane].run_decode(
                tokens[rows], start_pos[rows], page_table[rows],
                kv_cache[lane], active=start_pos[rows] >= 0))
        return concat(outs)  # [L * per_lane, 1, V] in slot order
```

Lane-DP rejects request-specific RoPE and speculative decode. Lane-DP output
handling is in
[Decode readback and output conversion](#decode-readback-and-output-conversion).

The tt-metal `tt_transformers` adapters follow the lane-DP sketch: the shared
`Generator` keeps one model and one KV cache list per sub-mesh, maps a prefill
slot to `user_id // max_batch_size_per_model`, and splits decode inputs with
`torch.chunk`. The Llama 3 70B Galaxy adapter divides `tt_data_parallel` by
the mesh width, builds fewer models, and allocates one KV cache list
replicated across its mesh. Its own generator maps users to devices; this
page does not describe that mapping. Any layout is valid if each lane's
lane-local block IDs address storage that no other lane uses.

Sources: `TTLaneInputBatch` in [input_batch.py](../src/vllm_tt_plugin/input_batch.py);
in [tt-metal](https://github.com/tenstorrent/tt-metal),
`models/tt_transformers/tt/generator.py`,
`models/tt_transformers/tt/generator_vllm.py`, and
`models/demos/llama3_70b_galaxy/tt/generator_vllm.py`.
