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

The optional class-callable `get_kv_cache_spec(vllm_config)` returns
`dict[str, KVCacheSpec] | None`. `TTWorker.get_kv_cache_spec` uses a homogeneous
specification named `"foo"` when `get_kv_cache_spec` is absent or returns `None`.
For per-layer specifications, use names containing `layers.<index>`, such as
`model.layers.5.self_attn`, and cover every attention-layer index.

`TTModelRunner` accepts `AttentionSpec` groups and `UniformTypeKVCacheSpecs`
groups. `UniformTypeKVCacheSpecs` can carry different per-layer shapes within
one group. The worker advertises only the vLLM `LBNHC` cache layout.

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
decode receive `page_tables_per_layer` in addition to group 0's `page_table`.
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
