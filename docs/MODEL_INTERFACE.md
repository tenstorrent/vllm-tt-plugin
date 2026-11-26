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
`register_kv_caches` once, after the model adapter allocates KV storage, with
the KV's DRAM regions (see [KV in DRAM banks](#kv-in-dram-banks)).

### NixlConnector

`NixlConnector` (pull mode) moves KV directly between the DRAM of two servers'
cards. It needs NIXL's Tenstorrent build, and runs only with
`"kv_buffer_device": "tt"` (the default on TT); any other buffer device is
refused at launch:

```bash
# prefill server
--kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_producer", "kv_buffer_device": "tt"}'
# decode server
--kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_consumer", "kv_buffer_device": "tt"}'
```

A server with fewer chips than the other sets `"tt_kv_parts"` in
`kv_connector_extra_config` to the other server's chip count. The config hook
makes the NIXL worker tensor-parallel rank 0 of 1 and lets it register the
DRAM regions: each DRAM bank once, and one strided descriptor per region, on
that region's card.

### KV transfer layout

A connector that moves KV between servers needs to know where each layer's KV
is on the chips. The model's KV transfer layout says it: per layer name in
`kv_cache_config`, the paged device tensors (parts) that hold a block of that
layer, the bytes of one block of a part on one chip, and which chips hold which
shard of it. Block `b` of a part is bytes `[b * block_bytes, (b + 1) *
block_bytes)` of each chip's buffer. A shard held by several chips is the same
bytes on each of them, so a transfer reads it from the first and writes it to
all of them. A transferred block of a layer is its parts' blocks in order, each
part shard after shard.

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
another order needs the method. `default_kv_layout` refuses anything else;
such a model needs the method.

A model class without a `get_kv_cache_spec` hook gets one layer name per
attention layer, `model.layers.<idx>.self_attn`, all with the default spec.

#### KV in DRAM banks

NIXL's Tenstorrent build moves KV between the DRAM of two servers' cards. It
addresses a card by N of `/dev/tenstorrent/N` and a byte by DRAM bank and
offset in the bank. `kv_dram_layout` (`kv_dram.py`) turns the KV transfer
layout into such addresses for `NixlConnector`. ttnn deals the pages of a DRAM
interleaved tensor round-robin over a chip's 8 DRAM banks, so when a part's
block is a multiple of 8 pages on a chip, block `b` is one byte range in each
bank. Per layer name, the result lists one
region per bank for each shard of each part, in that order, and every bank that
holds one. It refuses a tensor that is not DRAM interleaved, a shard copied to
several chips (a pull reads one remote range into one local range), a chip that
does not use all 8 banks (P100a and Bin6 Galaxy chips use 7), a chip that is
not a PCIe card of the host, and blocks that are not whole pages in each bank
or do not fit a bank. A server with fewer chips than the other server of a
transfer passes `parts` (`"tt_kv_parts"`) equal to the other's shard count:
with `parts=2`, a 1-chip server lists each range as the two halves that the
chips of a 2-chip server hold.

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
