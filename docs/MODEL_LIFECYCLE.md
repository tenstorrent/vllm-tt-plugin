# Model lifecycle

The plugin schedules requests while a model adapter retains TT programs,
traces, KV storage, and possibly per-request device state. The model adapter
must prepare execution before serving, preserve request ownership when batch
rows move, and release persistent captures while the mesh remains open.
See the [interface guide](MODEL_INTERFACE.md) for setup and execution calls.

## Warmup and trace capture

`TTWorker.compile_or_warm_up_model` runs `TTModelRunner.warmup_model` when
`tt.enable_model_warmup` is true, which is the default. The adapter supplies:

```python
warmup_model_prefill(kv_cache=..., can_sample_on_device=..., enable_trace=...)
warmup_model_decode(kv_cache=..., max_batch_size=..., num_blocks=...,
                    can_sample_on_device=..., enable_trace=...)
```

`max_batch_size` is the merged TT batch capacity. `num_blocks` is the profiled
maximum block-table width per request. When effective device grammar support
is enabled, decode warmup also receives `can_sample_device_grammar=True`.

1. `TTModelRunner.warmup_model` calls `warmup_model_prefill` and then
   `warmup_model_decode` with `enable_trace=False`.
2. `TTModelRunner.warmup_model` resets `already_warmed_up_prefill=False` if
   the adapter exposes `already_warmed_up_prefill`.
3. For `tt.trace_mode="all"`, `TTModelRunner.warmup_model` calls
   `warmup_model_prefill(enable_trace=True)`.
4. For `tt.trace_mode="all"` or `"decode_only"`,
   `TTModelRunner.warmup_model` calls `warmup_model_decode(enable_trace=True)`.

The adapter must prepare the operators and persistent buffers that traced
execution uses. Eager warmup must cover traced operators, and prefill warmup
must cover supported sequence lengths. Compilation or incompatible buffer
allocation after trace capture can invalidate retained trace storage.
`tt.trace_mode="none"` skips trace capture but still runs enabled eager warmup.

Source: `TTModelRunner.warmup_model` in
[model_runner.py](../src/vllm_tt_plugin/model_runner.py).

## Request state

Device-state slots identify adapter-owned request state. A prefill row is not
necessarily the slot that owns that request. KV page addresses and state slots
serve different purposes; the adapter must honor both.

| Interface | Adapter responsibility |
|---|---|
| `prefill_forward(empty_slots=[...])` | Write each prefill row's state into the corresponding destination slot; preserve live state in other slots. Stateless adapters can ignore `empty_slots`. |
| `decode_forward(slot_remap=...)` | Gather old state so destination slot `i` reads old slot `slot_remap[i]` before decode uses that state. `None` means identity. Apply the whole permutation without overwriting state still needed by another destination. |
| `note_state_slots_moved(moves)` | Optional front-packed execution callback receiving `{old_slot: new_slot}` after the plugin commits a movement. Update adapter-owned session identities; do not repeat a device gather already performed by `decode_forward`. |
| `release_request(slot)` | Optional front-packed execution callback that releases the state of the request currently owning `slot`. The plugin passes the current slot, not necessarily the original prefill slot. |

`TTModelRunner._decode_state_slot_remap` covers the full state-slot capacity,
which can exceed a bucketed decode's token row count. The adapter must preserve
off-batch state moved by that permutation. `TTModelRunner` commits the new
ownership only after `decode_forward` accepts the submission, then calls
`note_state_slots_moved` when applicable.

In front-packed execution, `TTModelRunner._update_states` calls
`release_request` for finished and preempted requests while slot ownership is
available. A resumed request rebuilds state through prefill.
A request omitted from a prefill-only step can still own live decode state;
absence from one submitted batch does not authorize the adapter to release
that request's state.

Lane-DP uses scheduler-owned stable slots. `TTLaneInputBatch.apply_step_plan`
removes finished and preempted requests without calling `release_request`;
the lane path also does not issue `note_state_slots_moved` callbacks. A model
adapter that requires those callbacks cannot assume lane-DP supplies them.

Block-output models must implement callable `release_request` and
`release_persistent_capture`. Adaptive block-output models with
`max_num_seqs > 1` must also implement `note_state_slots_moved`.
Stateful speculative adapters have additional ownership duties in the
[speculative contract](SPEC_DECODE_CONTRACT.md).

### Stage inputs

A model that declares `supports_stage_inputs` receives each request's payload
on prefill only, as a read-only snapshot without the request ID. So:

- Decode-time state derived from the payload is the model's to keep, in its
  slot-owned state, written by the request's last prefill chunk.
- Work derived from the payload, such as a speaker embedding, cannot be cached
  across prefill chunks by slot: a request's row and slot can change between
  chunks.
- `warmup_model_prefill` receives no `stage_inputs`. A model whose prefill is
  built from the payload supplies its own warmup payload.

Sources: `_alloc_prefill_state_slots`, `_decode_state_slot_remap`,
`_release_model_request`, and `_update_states` in
[model_runner.py](../src/vllm_tt_plugin/model_runner.py);
`TTLaneInputBatch.apply_step_plan` in
[input_batch.py](../src/vllm_tt_plugin/input_batch.py).

## Shutdown

1. `TTWorker.shutdown` calls `TTModelRunner.shutdown` while the mesh is open.
2. `TTModelRunner.shutdown` calls optional `release_persistent_capture()` once
   to release model-lifetime captures. The adapter must release device
   resources that require an open mesh here. The runner logs exceptions and
   marks capture release complete rather than retrying capture release.
3. `TTWorker.shutdown` closes the mesh.

The plugin does not call a generic model `shutdown()` hook or guarantee model
destruction before mesh close. Keep device-dependent capture cleanup in
`release_persistent_capture`.

Sources: `TTModelRunner.shutdown` in
[model_runner.py](../src/vllm_tt_plugin/model_runner.py) and
`TTWorker.shutdown` in [worker.py](../src/vllm_tt_plugin/worker.py).
