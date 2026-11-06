# Model setup and KV storage

The TT plugin selects a model class, opens a device mesh, constructs a model
instance, and asks the model instance to allocate KV storage. This guide calls
the model class implementation an **adapter**. The adapter supplies TT model
execution; the plugin supplies vLLM scheduling and translates vLLM cache plans.
Start with the [interface overview](MODEL_INTERFACE.md) for the full lifecycle.

## Registration and class declarations

`--model` identifies the checkpoint. The checkpoint's `hf_config.architectures`
selects the adapter through vLLM's model registry. Registration uses a lazy
`module.path:ClassName` target. `TTPlatform.check_and_update_config` adds a `TT`
prefix to checkpoint architecture names before resolving the TT adapter.
The adapter module must be importable in each vLLM process that resolves it.

Registration precedence is `TT_MODEL_CLASS_OVERRIDES`, then `EXTRA_MODELS_DIR`,
then built-in registrations. Use the README examples for
[bundle registration](../README.md#registering-models-dynamically-extra_models_dir)
and [class selection](../README.md#selecting-a-serving-class-tt_model_class_overrides).
`TT_VLLM_BUILTIN_MODELS=0` disables ordinary built-in registrations;
DiffusionGemma aliases remain registered.

Declare `model_capabilities` on the adapter class so
`TTPlatform.check_and_update_config` can read support before construction.
Keep class and instance declarations consistent because runtime consumers also
read `model_capabilities`. See [Model capabilities](MODEL_CAPABILITIES.md) for
every supported key, default, and validation rule. Operators request supported
behavior through vLLM arguments and `additional_config["tt"]`.

Sources: `register_tt_models`, `_register_models_from_extra_dir`, and
`TTPlatform.check_and_update_config` in [platform.py](../src/vllm_tt_plugin/platform.py).

## Construction and mesh ownership

1. `TTWorker.init_device` binds visible devices and repeats platform validation
   in the worker process.
2. `TTWorker.init_device` opens a `ttnn.MeshDevice`. `set_fabric` applies hardware
   defaults, `model_capabilities["fabric_config"]`, then explicit launch
   overrides. `set_fabric` does nothing for a single-device mesh.
3. `TTWorker.init_device` computes the KV block budget and fits block-output
   `max_model_len` before model construction.
4. `TTModelLoader.load_model` calls the required class-callable factory:

```python
adapter = model_class.initialize_vllm_model(
    model_config.hf_config,
    device_config.device,
    max_batch_size,
    max_seq_len=model_config.max_model_len,
    tt_data_parallel=tt_data_parallel,
    optimizations=optimizations,
)
```

`initialize_vllm_model` returns the instance that implements the execution,
allocation, and lifecycle hooks. The adapter owns model and weight preparation;
`TTModelLoader.download_model` and `TTModelLoader.load_weights` are not implemented.

| Factory input | Meaning |
|---|---|
| `model_config.hf_config` | Hugging Face checkpoint configuration. |
| `device_config.device` | Already-open `ttnn.MeshDevice`; `TTWorker` owns mesh closure. |
| `max_batch_size` | Engine-local capacity, including all lanes in lane-DP. |
| `max_seq_len` | Resolved `model_config.max_model_len`. |
| `tt_data_parallel` | In-process lane count; 1 for each standard multi-process DP rank. |
| `optimizations` | `None`, `"performance"`, or `"accuracy"`, from `additional_config["tt"]`. |

`TTWorker.shutdown` releases runner resources before closing the mesh. See
[Model lifecycle](MODEL_LIFECYCLE.md) for adapter cleanup hooks.

Sources: [loader.py](../src/vllm_tt_plugin/loader.py), `TTWorker.init_device` and
`set_fabric` in [worker.py](../src/vllm_tt_plugin/worker.py), and
`get_tt_max_batch_size` / `get_tt_data_parallel_size` in [config.py](../src/vllm_tt_plugin/config.py).

## KV budget and specifications

The optional class-callable `get_max_tokens_all_users` supplies the KV token
capacity for one lane or one standard-DP rank, before plugin-added headroom:

```python
model_class.get_max_tokens_all_users(
    model_name=model_config.model,
    num_devices=num_devices,
    tt_data_parallel=tt_data_parallel,
    max_model_len=model_config.max_model_len,
    max_num_seqs=per_lane_max_num_seqs,
)
```

`num_devices` counts physical devices in the worker's open mesh.
`max_num_seqs` is the per-lane or per-rank capacity, unlike the merged capacity
passed to `initialize_vllm_model`. The plugin uses `131_072` when the sizing
call raises `AttributeError`, including when `get_max_tokens_all_users` is absent.
The plugin adds output reservation and applicable sliding-window headroom.
The plugin does not measure free device memory to choose this KV budget.
A model with `MambaSpec` layers gets a different block count; see
[Recurrent state](#recurrent-state-mambaspec).

The optional class-callable `get_kv_cache_spec(vllm_config)` returns
`dict[str, KVCacheSpec] | None`. `TTWorker.get_kv_cache_spec` uses a homogeneous
specification named `"foo"` when `get_kv_cache_spec` is absent or returns `None`.
For per-layer specifications, use names containing `layers.<index>`, such as
`model.layers.5.self_attn`, and cover every attention-layer index.

`TTModelRunner` accepts `AttentionSpec` groups, `UniformTypeKVCacheSpecs`
groups, and `MambaSpec` groups next to attention groups.
`UniformTypeKVCacheSpecs` can carry different per-layer shapes within one
group. The worker advertises only the vLLM `LBNHC` cache layout.

A stage with no token KV cache, such as an audio decoder, uses a `TTWorker`
subclass that sets `uses_token_kv_cache = False`. That worker sizes no KV pool
and reports an empty KV specification. vLLM then treats the stage as
attention-free: it skips `determine_available_memory`, which returns `0`, and
passes the runner a KV cache config with no groups. `TTModelRunner` refuses
that config, so the subclass also names a runner, in `model_runner_cls`, that
accepts it.

Sources: `get_num_available_blocks_tt`, `TTWorker.get_kv_cache_spec`, and
`TTWorker.determine_available_memory` in [worker.py](../src/vllm_tt_plugin/worker.py).

## Allocation and page-table routing

1. `TTModelRunner._build_per_layer_specs` translates the vLLM cache plan into
   `(shape, dtype, tensor_idx)` triples, ordered by attention-layer index.
   With `MambaSpec` groups, the list covers every layer; see
   [Recurrent state](#recurrent-state-mambaspec).
2. `TTModelRunner._allocate_kv_caches` calls the instance's
   `allocate_kv_cache_per_layer(per_layer_specs)` when the adapter exposes it.
   Otherwise, `TTModelRunner._allocate_kv_caches` calls
   `allocate_kv_cache(shape, dtype, num_layers)` and requires identical per-layer
   shapes and dtypes.
3. `TTModelRunner` stores the allocator result as `kv_caches` and passes the
   allocator result to warmup and execution as `kv_cache`.

`shape` is `(num_blocks, heads_per_device, block_size, head_size)`, where
`heads_per_device = spec.num_kv_heads // min(devices_per_lane, spec.num_kv_heads)`
and `devices_per_lane = num_devices // tt_data_parallel`.
`tensor_idx` identifies shared storage: layers with the same `tensor_idx` share
a buffer. A single-group cache gives every attention layer a separate
`tensor_idx`. The plugin treats the allocator result as opaque model storage.

When vLLM creates multiple KV groups, `TTModelRunner` expands group page tables
into `page_tables_per_layer`, ordered by attention-layer index. Prefill and
decode receive `page_tables_per_layer` in addition to group 0's `page_table`
(the first attention group's with `MambaSpec` groups).
Each per-layer tensor has shape
`[tt_max_batch_size, max_num_blocks_per_req]`, with unused rows and columns
zero-padded. The per-layer row capacity can exceed `tokens.shape[0]` during
prefill or bucketed decode.
A single-group cache omits `page_tables_per_layer`, including a single group
with heterogeneous per-layer shapes. See [Model execution](MODEL_EXECUTION.md).

For sliding-window groups, the class attribute
`_HYBRID_KV_CACHE_GROUPS_ENABLED=True` enables additional sizing headroom when
the checkpoint supplies `sliding_window`. The adapter must implement the
addressing required by its cache specifications. For a speculative drafter
that shares bounded target storage, the adapter must also preserve the target
and drafter addressing requirements in [the speculative contract](SPEC_DECODE_CONTRACT.md).

Sources: `_kv_cache_shape`, `_build_per_layer_specs`, `_block_tables_per_layer`,
and `submit_prefill` in [model_runner.py](../src/vllm_tt_plugin/model_runner.py);
`submit_decode` in [async_decode.py](../src/vllm_tt_plugin/async_decode.py).

## Recurrent state (MambaSpec)

A model with recurrent layers, such as gated delta network (GDN) layers,
declares their state to vLLM: `get_kv_cache_spec` returns a `MambaSpec` for
each recurrent layer and an attention specification for each attention layer.
vLLM then owns the block ids of the recurrent state as it owns attention block
ids. vLLM's hybrid KV cache manager puts the layers into KV cache groups of
equal size, for example one attention group and three `MambaSpec` groups of
16 layers each for 16 attention and 48 GDN layers. Every group takes block ids
from one pool, and vLLM overlays the groups on the same memory: block id `b`
addresses the same bytes in every group, and one group owns a block id at a
time. This section calls the memory that the layers at one position of each
group share a **layer slot**. The plugin reports it as `tensor_idx`.

Declaration rules:

- `MambaSpec.shapes` and `MambaSpec.dtypes` describe the state of one layer on
  one device, convolution history included. The plugin passes them to the
  adapter unchanged.
- vLLM requires one page size, in bytes per block per layer, across all groups.
  `unify_kv_cache_spec_page_size` pads a smaller `MambaSpec` page to the
  largest page. For a smaller attention page, it multiplies the attention
  `block_size` when the attention page divides the largest page, and pads the
  attention page otherwise. vLLM's `Platform.update_block_size_for_backend`
  returns before `_align_hybrid_block_size` on TT, because a TT model registers
  no vLLM attention layer for `_find_non_ssm_backend` to find, so the adapter
  chooses the attention `block_size` itself: choose one that TT
  paged attention supports and whose page is at least the recurrent state.
- `MambaSpec.mamba_cache_mode` must equal the resolved
  `cache_config.mamba_cache_mode`. vLLM's `MambaModelConfig` selects `none`
  when prefix caching is off and `align` when it is on; prefix caching stays
  on only for a model that declares `supports_prefix_caching`. In mode `none`
  a request holds one state block per group, provided `MambaSpec.block_size`
  is at least `max_model_len`, as upstream resolves `mamba_block_size`;
  `MambaManager` otherwise allocates one block per `block_size` tokens in
  every `MambaSpec` group. In mode `align`, block-table column `p` of a group
  holds the state after `(p + 1) * block_size` tokens, and `MambaSpec.block_size`
  must equal the attention `block_size`: the scheduler ends prefill chunks on
  multiples of the smallest group `block_size`. `TTScheduler` schedules a
  multimodal prompt that fits one step as one chunk. Where a chunk end of a
  longer one falls inside a multimodal item, `TTScheduler` moves it out of
  the item; a block boundary the chunk then crosses keeps no state for prefix
  caching. A multimodal request never reads the prefix cache (see
  [Chunked prefill](SCHEDULING.md#chunked-prefill)). A
  `cache_config.prefix_match_unit` must be a multiple of
  `additional_config['tt']['prefill_chunk_alignment']`, because with partial
  hash hits vLLM also ends chunks on its multiples.
- `TTModelRunner` supports modes `none` and `align`, and one `block_size`,
  one `num_speculative_blocks` and one `num_prefill_checkpoint_blocks` for all
  `MambaSpec` groups.
- `MambaSpec.num_prefill_checkpoint_blocks` is 0, or 1 in mode `align` with a
  `MambaSpec.prefill_checkpoint_alignment`. With 1, vLLM no longer ends a
  prefill chunk at the prompt's last cacheable block boundary: a prompt that
  fits one chunk is one prefill step instead of two, and vLLM reserves a
  block inside the chunk for the state at its checkpoint position
  (`get_mamba_prefill_checkpoint_position`). The prefill must then write
  the states named by `state_checkpoint_ids_per_layer`; see
  [recurrent state blocks](MODEL_EXECUTION.md#recurrent-state-blocks). vLLM
  places a checkpoint only where its offset from the chunk start is a
  multiple of `prefill_checkpoint_alignment`, so declare the granularity at
  which the prefill can stop to write a state.
- `MambaSpec.num_speculative_blocks` must equal the launch's draft length K
  (0 without speculative decoding), as upstream sets it from
  `num_speculative_tokens`. A request then holds `1 + K` state blocks per
  group in mode `none` and up to `2 + K` in mode `align`; each holds the full
  state of one position. The block-table row of a `MambaSpec` group holds
  `MambaSpec.max_num_blocks_per_req` blocks: `1 + K` in mode `none`,
  `ceil(max_model_len / block_size) + K` in mode `align`.
- The adapter must implement `zero_kv_cache_blocks`, and in mode `align` also
  `copy_kv_cache_blocks`. See [block operations](MODEL_LIFECYCLE.md#kv-cache-block-operations).
  Prefill and decode receive `state_block_ids_per_layer`; see
  [recurrent state blocks](MODEL_EXECUTION.md#recurrent-state-blocks).
- Lane-DP and block-output models do not support `MambaSpec` groups.

Allocation:

- `per_layer_specs` has one entry per layer that the groups name, in
  decoder-layer index order, so the list covers the recurrent layers between
  the attention layers. Layer names must contain `layers.<index>` with indices
  `0` to `len(per_layer_specs) - 1`.
- An attention entry is `(shape, dtype, tensor_idx)`, as above. A `MambaSpec`
  entry is `(MambaSpec.shapes, MambaSpec.dtypes, tensor_idx)`.
- The layer at position `j` of each group gets `tensor_idx` `j`. The adapter
  allocates the buffers of each `tensor_idx` from its attention entries and
  stores the state of each recurrent layer with that `tensor_idx` inside them,
  at the block ids the layer's group owns. `TTModelRunner` raises if a
  recurrent layer's `tensor_idx` has no attention layer.
- The adapter must implement `allocate_kv_cache_per_layer`.

Block count: `get_num_available_blocks_tt` converts `max_tokens_all_users` to
bytes per layer slot with the attention layers' bytes per token, then divides
by the unified page size. A token count divided by `block_size` would ignore
page padding. It then adds one block per running request in each attention
group, for output, and the state blocks each running request holds in each
`MambaSpec` group: `1 + K` per group in mode `none` and `2 + K` in mode
`align`, as `MambaSpec.max_memory_usage_bytes` counts them. Upstream's admission floor
applies as for sliding-window models. Mode `align` also keeps the state blocks
of earlier block boundaries as prefix-cache entries; as with cached attention
blocks, vLLM evicts them when it needs free blocks.

Page tables: `page_tables_per_layer` has one entry per layer, and the entry of
a recurrent layer is its group's table. vLLM orders the groups by
specification, so group 0 can be a `MambaSpec` group. With `MambaSpec` groups,
`page_table` is therefore the table of the first attention group, not of
group 0.
The table width `max_num_blocks_per_req` is the largest
block-table row width over the groups, at most `num_blocks`, rounded up to a
multiple of 8. TT chunked SDPA requires a page-table row of a
multiple of 32 bytes. `cache_config.block_size` does not set this width.

Sources: `mamba_cache_groups` and `MambaCacheGroups` in
[mamba_state.py](../src/vllm_tt_plugin/mamba_state.py);
`TTModelRunner.initialize_kv_cache` and `_build_per_layer_specs` in
[model_runner.py](../src/vllm_tt_plugin/model_runner.py);
`_mamba_hybrid_num_blocks` in [worker.py](../src/vllm_tt_plugin/worker.py).
