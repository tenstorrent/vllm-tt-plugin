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
| `note_state_slots_moved(moves)` | Optional front-packed execution callback receiving `{old_slot: new_slot}` after the plugin commits a movement. Update adapter-owned session identities; do not repeat a device gather already performed by `decode_forward`. Lane-DP never moves a slot, so it never calls this. |
| `release_request(slot)` | Optional callback that releases the state of the request currently owning `slot`. The plugin passes the current slot: in front-packed execution not necessarily the original prefill slot, in lane-DP the request's stable row. |

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

An adapter with `MambaSpec` groups keeps each request's recurrent state in vLLM blocks, not in device-state slots, and `TTModelRunner` addresses those blocks per row in every step through `mamba_state` (see [recurrent state blocks](MODEL_EXECUTION.md#recurrent-state-blocks)). So:

- Row movement needs no state gather: a moved row's `mamba_state` names the same blocks as before. `slot_remap` and `note_state_slots_moved` still apply to any slot-owned state the adapter keeps.
- Completion and cancellation need no release: vLLM frees the request's blocks, and the next owner of a block starts from zero state, from state it writes itself, or, in `mamba_cache_mode` `align`, from a block vLLM kept in the prefix cache.
- A preempted request loses its blocks. vLLM schedules its resume as a prefill over its prompt and generated tokens: from `start_pos` 0 without a prefix-cache hit, so the adapter rebuilds the state from zero state, or from the state of a prefix-cache hit.

In `mamba_cache_mode` `align` (prefix caching on), vLLM's prefix cache also covers recurrent state. When a request's computation reaches a position whose state vLLM keeps, vLLM registers the block that holds the state after exactly that many tokens. With `prefix_cache_retention_interval` 0, the only value TT supports, vLLM keeps what its `CacheConfig` documentation calls semantic checkpoints, including the replay boundary of a prompt (its last cacheable position) and shared-prefix junctions. A later request whose tokens start with the same tokens gets a **prefix-cache hit**: vLLM schedules it from `start_pos` equal to the hit length and puts the kept block into the request's block-table column for that length, so `mamba_state.start_blocks` names it. The kept block can be the start block of several requests at once and must stay unchanged; vLLM reuses it only after no request holds it and it is evicted from the prefix cache. A hit is shorter than the request's tokens, and it ends at a block boundary or, with `prefix_match_unit` below the block size, at a multiple of `prefix_match_unit`.

Lane-DP uses scheduler-owned stable slots, and a request's row is its slot
for its whole life. `TTLaneInputBatch.apply_step_plan` calls `release_request`
with that row for every request whose row the step frees -- a finished
request, a preempted one, and a resumed one that still held a row -- while
the request still holds it, before the row can be reused. A request that
holds no row when it finishes (aborted while queued, or preempted and aborted
before its resume) releases nothing; a preempted request that holds no row is
a bookkeeping fault and raises. Lane-DP never moves a slot (`condense` is a
no-op and `slot_remap` the identity), so it never issues
`note_state_slots_moved`.

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

## KV cache block operations

vLLM can ask the worker to change KV cache blocks before a step's forward. The adapter owns the device buffers, so `TTModelRunner._apply_kv_cache_block_operations` hands each operation to an adapter hook. It runs in `TTModelRunner.build_model_input`, after `_update_states` and before `TTModelRunner` builds the step's inputs, and calls the zeroing hook before the copy hook, as the upstream GPU runner orders them. Block ids are vLLM block ids; with `MambaSpec` groups, one block id addresses one page in every layer slot (see [recurrent state](MODEL_SETUP.md#recurrent-state-mambaspec)).

| Hook | Adapter responsibility |
|---|---|
| `zero_kv_cache_blocks(kv_cache=..., block_ids=[...])` | Zero each listed block in every KV buffer. vLLM lists the blocks it newly allocated to attention groups this step (`SchedulerOutput.new_block_ids_to_zero`). It sends them when `KVCacheConfig.needs_kv_cache_zeroing` is true, which holds with `MambaSpec` layers: a reused block can hold recurrent-state bytes that decode to NaN or Inf. `TTModelRunner` passes the whole list. |
| `copy_kv_cache_blocks(kv_cache=..., copies=[(src, dst), ...])` | For each `(src, dst)` pair, make block `dst` hold the content of block `src` in every KV buffer: the attention keys and values of an attention layer, and the full state of a recurrent layer (recurrent state and convolution history, including any part the adapter has not yet written into block `src` itself). The pairs are `SchedulerOutput.kv_cache_block_copies`, as plain `int` pairs. vLLM queues them in `mamba_cache_mode` `align` when a prefix-cache hit ends inside a block, so the hitting request writes a private copy (copy-on-write) and the cached block stays unchanged. Block `src` must stay unchanged. |

`kv_cache` is the object the adapter's KV allocator returned. Enqueue each operation on the device before the step's forward, so the forward reads its result. `TTModelRunner` does not wait for the operations to complete: a hook can return after it enqueues the work, and the adapter must order it before every later device read or write of the same blocks, including the step's forward.

`copy_kv_cache_blocks` applies the pairs in list order, after the step's zeroing, so a destination that vLLM also lists for zeroing ends up with the copied content. No block is the destination of two pairs, and no block is both a source and a destination, because `TTModelRunner` raises before it calls either hook if vLLM's list breaks that rule. So the adapter can also apply all pairs of a call at once.

An adapter with `MambaSpec` groups must implement `zero_kv_cache_blocks`, and in `mamba_cache_mode` `align` also `copy_kv_cache_blocks`; `TTModelRunner.initialize_kv_cache` raises otherwise. In mode `none`, prefix caching is off and vLLM queues no copies, so `TTModelRunner` raises if a copy arrives. For an adapter without `MambaSpec` groups, `TTModelRunner` calls neither hook.

Sources: `_apply_kv_cache_block_operations` and `initialize_kv_cache` in [model_runner.py](../src/vllm_tt_plugin/model_runner.py); `MambaCacheGroups.block_copies` in [mamba_state.py](../src/vllm_tt_plugin/mamba_state.py).

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
