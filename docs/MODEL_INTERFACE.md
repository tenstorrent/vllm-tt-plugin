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

KV connector support on TT is experimental. `--kv-transfer-config` is
supported without data parallelism, for a model that declares
`supports_prefix_caching`: a request that loads KV from a connector resumes its
prefill after that KV, as after a prefix-cache hit, and a model without that
resume would compute on KV or state that never arrived. With a connector,
`TTModelRunner.initialize_kv_cache` calls the worker-side connector's
`register_kv_caches` once, after the model adapter allocates KV storage, and
then `set_host_xfer_buffer_ops` with the function that copies KV blocks between
the adapter's ttnn tensors and the connector's CPU buffer. vLLM's base
connector ignores that function, and `NixlConnector` keeps it only with a CPU
buffer.

The copy moves each chip's share of each block as one region with
`ttnn.copy_device_to_host_regions` and `ttnn.copy_host_to_device_regions`,
straight between the chip and its place in the buffer, one call per layer part
with the layer's buffer and byte offsets into it.
Both come with
[tt-metal#67057](https://github.com/tenstorrent/tt-metal/pull/67057). A launch
whose connector has a CPU buffer (`"kv_buffer_device": "cpu"`) is refused at
startup when the installed ttnn lacks them.

### NixlConnector

vLLM's `NixlConnector` moves KV between servers, for example from a prefill
server to a decode server, with the NIXL transfer library. NIXL cannot register
TT device memory, so on TT the connector runs only with a CPU buffer. The
prefill server is the producer and the decode server the consumer (vLLM 0.33
deprecates `kv_both` for this connector):

```bash
# prefill server
--kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_producer", "kv_buffer_device": "cpu"}'
# decode server
--kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_consumer", "kv_buffer_device": "cpu"}'
```

For it, `TTPlatform` names `TTAttentionBackend` as the attention backend (only
its name, `TT`, is used: two servers compare it), and the config hook tells
the NIXL worker that it is tensor-parallel rank 0 of 1. The install script does
not install the `nixl` package; vLLM's `requirements/kv_connectors.txt` pins
the version its connector needs.

### KV transfer layout

`register_kv_caches` and the copy both come from the model's KV transfer
layout: per layer name in `kv_cache_config`, the paged device tensors (parts)
that hold a block of that layer, the bytes of one block of a part on one chip,
and which chips hold which shard of it. Block `b` of a part is bytes
`[b * block_bytes, (b + 1) * block_bytes)` of each chip's buffer. A shard held
by several chips is the same bytes on each of them: the copy reads it from the
first and writes it to all of them. A transferred block of a layer is its
parts' blocks in order, each part shard after shard.

A model declares its layout with an optional method:

```python
def get_kv_transfer_layout(self, kv_caches, kv_cache_config):
    # layer name -> [(device tensor, block_bytes, shards), ...]
    # shards: per shard, the mesh device indices (row-major) holding it
    return {
        "model.layers.0.self_attn": [(k0, k_bytes, [[0], [1]]), (v0, v_bytes, [[0], [1]])],
        ...
    }
```

It must describe every layer name in `kv_cache_config` and may use any part
structure: K and V, one MLA latent, heads split or copied over any mesh shape.
Without the method, the plugin derives the layout of a `[k, v]` pair per
attention layer (`kv_caches` in layer-index order, layer
`model.layers.<idx>...` is entry `idx`), each cache
`[num_blocks, heads, block_size, head_size]` per chip, with `num_blocks` from
`kv_cache_config` and `block_size` and `head_size` from the layer's spec, in
TILE layout of bfloat16, float32, bfloat8_b or bfloat4_b, with the heads either
split evenly over the chips or copied to every chip. A split is taken to put
heads `[i * heads, (i + 1) * heads)` on chip `i` in row-major mesh order: the
cache does not record which heads a chip holds, so a model that splits them in
another order needs the method. Anything else raises at startup and needs the
method.

The connector's description of a layer is bytes, `(num_blocks, num_kv_heads,
block_size, n)` `uint8`, sized so that a block is one transferred block. It
holds no KV; the KV stays in the ttnn tensors. Two servers can exchange KV only
when every layer's transferred block has the same bytes on both: same dtype,
head size, block size and total heads. Their chip counts may differ when both
split heads in chip order.

A model class without a `get_kv_cache_spec` hook gets one layer name per
attention layer, `model.layers.<idx>.self_attn`, all with the default spec.

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
