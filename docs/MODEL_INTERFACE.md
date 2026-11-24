# Model-plugin interface

`vllm-tt-plugin` connects vLLM scheduling and sampling to Tenstorrent model
execution. A registered TT model class supplies construction, KV allocation,
prefill, and decode operations. This guide calls that implementation the
**model adapter**; the plugin does not require a shared adapter base class.
The guide describes the interface the plugin calls and the behavior each
adapter must provide.

## Reading guide

| Task | Reference |
|---|---|
| Register an adapter, construct the model, size and allocate KV storage | [Model setup](MODEL_SETUP.md) |
| Implement prefill, decode, sampling, and output conversion | [Model execution](MODEL_EXECUTION.md) |
| Prepare traces and manage request state and shutdown | [Model lifecycle](MODEL_LIFECYCLE.md) |
| Declare optional behavior and understand admission limits | [Model capabilities](MODEL_CAPABILITIES.md) |
| Implement resident decode input updates and selective host readback | [Decode reload contract](DECODE_RELOAD_CONTRACT.md) |
| Implement speculative verification and model-owned proposals | [Speculative decode contract](SPEC_DECODE_CONTRACT.md) |

The setup, execution, and lifecycle pages describe the common interface.
The capability, reload, and speculative references define their respective
extensions. [Scheduling](SCHEDULING.md) explains plugin orchestration;
the [README](../README.md) provides installation and launch instructions.

## Ownership and call order

vLLM owns requests, scheduling, and logical KV allocation. The plugin converts
scheduled work into model calls, selects host or device sampling, and returns
committed tokens to vLLM. The model adapter owns TT programs, physical KV
storage, device execution, and any model-specific state.

1. `TTPlatform.check_and_update_config` resolves the TT model class and checks
   class declarations against the requested launch.
2. `TTWorker.init_device` opens the device mesh and determines the KV budget.
3. `TTModelLoader.load_model` calls `initialize_vllm_model` to construct the
   model adapter on the open mesh.
4. `TTModelRunner.initialize_kv_cache` asks the model adapter to allocate KV
   storage. `TTWorker.compile_or_warm_up_model` then runs enabled warmup.
5. `TTModelRunner` submits either prefill or decode work. The model adapter
   returns logits, sampled tokens, or the negotiated speculative output.
6. In front-packed execution, `TTModelRunner` reports request-state moves and
   releases to optional model hooks. `TTWorker.shutdown` releases persistent
   captures before mesh close.

## KV connectors

With `--kv-transfer-config`, the single-process path without data parallelism
runs vLLM's KV connector lifecycle; lane-DP and multi-process DP reject the
option. `TTModelRunner.initialize_kv_cache` calls the worker-side connector's
`register_kv_caches` once, after the model adapter allocates KV storage. The
argument is a dictionary keyed by the layer names in `kv_cache_config`. Each
value is a torch tensor that only describes that layer's KV cache and holds no
KV. The KV stays in the ttnn tensors the adapter allocated. A layer is
described as bytes, `(num_blocks, num_kv_heads, block_size, n)` `uint8`, sized
so that a block is exactly the packed tiles of its K and V on every chip. The
copy below moves those tiles unchanged, so startup raises unless every layer's
cache is `bfloat8_b`, and both servers of a transfer need the same total KV
heads, head size and block size. Their chip counts may differ.

A model class without a `get_kv_cache_spec` hook gets one layer name per
attention layer, `model.layers.<idx>.self_attn`, all with the default spec.

The runner then calls `set_host_xfer_buffer_ops` with the function that copies
KV blocks between the adapter's ttnn tensors and a CPU buffer shaped like
those descriptions. A connector that stages KV through such a buffer, like
`NixlConnector` with `"kv_buffer_device": "cpu"`, moves KV with it. Startup
raises unless the allocator returned one `[k, v]` pair per layer name, each
name following the `...layers.<idx>...` convention. On a mesh of several chips
the copy takes each chip's tensor to hold an equal share of the KV heads, in
chip order.

## Minimum implementation

An ordinary text adapter needs `initialize_vllm_model`, `prefill_forward`,
`decode_forward`, and either `allocate_kv_cache_per_layer` or
`allocate_kv_cache`. Default-enabled warmup also needs
`warmup_model_prefill` and `warmup_model_decode`. Supply `model_capabilities`
on the class and instance, even when the dictionary is empty: platform
admission tolerates absence, but runtime sampling reads the dictionary directly.

The execution page specifies when `read_decode_output` and
`process_decode_output_host` are needed. Add extension hooks only with the
matching declarations and behavior. In particular, a capability declaration
does not implement a hook or establish device correctness.

Source entry points: [loader.py](../src/vllm_tt_plugin/loader.py),
[worker.py](../src/vllm_tt_plugin/worker.py),
[model_runner.py](../src/vllm_tt_plugin/model_runner.py), and
[async_decode.py](../src/vllm_tt_plugin/async_decode.py).
