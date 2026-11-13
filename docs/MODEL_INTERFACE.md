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
| Declare recurrent state through `MambaSpec` and address its blocks | [Recurrent state](MODEL_SETUP.md#recurrent-state-mambaspec), [state blocks](MODEL_EXECUTION.md#recurrent-state-blocks), [block operations](MODEL_LIFECYCLE.md#kv-cache-block-operations) |
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

## Minimum implementation

An ordinary text adapter needs `initialize_vllm_model`, `prefill_forward`,
`decode_forward`, and either `allocate_kv_cache_per_layer` or
`allocate_kv_cache`. Default-enabled warmup also needs
`warmup_model_prefill` and `warmup_model_decode`. Supply `model_capabilities`
on the class and instance, even when the dictionary is empty: platform
admission tolerates absence, but runtime sampling reads the dictionary directly.

An adapter that declares recurrent state through `MambaSpec` also needs `allocate_kv_cache_per_layer` and `zero_kv_cache_blocks`. Its prefill and decode receive `mamba_state`, one `MambaGroupState` per `MambaSpec` group (see [state blocks](MODEL_EXECUTION.md#recurrent-state-blocks)).

The execution page specifies when `read_decode_output` and
`process_decode_output_host` are needed. Add extension hooks only with the
matching declarations and behavior. In particular, a capability declaration
does not implement a hook or establish device correctness.

Source entry points: [loader.py](../src/vllm_tt_plugin/loader.py),
[worker.py](../src/vllm_tt_plugin/worker.py),
[model_runner.py](../src/vllm_tt_plugin/model_runner.py), and
[async_decode.py](../src/vllm_tt_plugin/async_decode.py).

## Chained speculative verification

A model that declares `supports_chained_spec_verify` (see
[model capabilities](MODEL_CAPABILITIES.md)) lets `TTModelRunner` submit an
`argmax_ids` verify over the verify before it, before the runner has read that
one. The [speculative decode contract](SPEC_DECODE_CONTRACT.md) states the
ordinary verify; a chained verify differs as follows.

| Item | Chained verify |
|---|---|
| `spec_predecessor` | Keyword argument of `decode_forward`: the `VerifyOutput` the model returned for the verify submitted just before, still unread. Absent on every other call. |
| Live rows | Every live row of the call is a live row of the predecessor at the same row index. A predecessor row whose request finished is a padding row of the call. |
| `tokens`, `start_pos` | Placeholders on live rows: `PLACEHOLDER_TOKEN_ID` in every column, and the positions of a predecessor that accepted all its drafts (its column-0 position plus `1 + K + j` in column `j`). Padding rows as on any verify. |
| `num_valid_drafts` | On live rows, the most drafts the row may verify (`SpecPlan.draft_cap` of the step's live rows). |
| `accepted_counts` | 1 on every row. |
| `mamba_state` | Each `MambaGroupState` also carries `start_blocks_by_count` `[rows, 1 + K]`: column `c - 1` is the block holding the row's start state if the predecessor committed `c` tokens for it. `start_blocks` holds column K. |

The model builds each live row from the predecessor's device result. With `c`
the number of tokens the predecessor committed for the row by the model's own
acceptance of its `argmax_ids`, the row is the predecessor's committed token
`c - 1` in column 0, then `n = min(num_valid_drafts, offered)` drafts that the
model's drafter made from that commit (`offered` is 0 when the drafter
declines the row and K otherwise), at the predecessor's column-0 position plus
`c`, starting from `start_blocks_by_count[row, c - 1]`. The return is the
`VerifyOutput` of that block, as on any verify.

`TTModelRunner` then reads the predecessor, walks it, commits it and calls
`propose_draft_tokens` for it. That call must return the drafts the chained
verify ran (`draft_token_ids[row, :n]` with `num_valid` the drafter's offer),
and must raise when the runner's commit differs from the model's own
acceptance: the chained verify continued the model's count. The runner fills
the chained verify's rows from that proposal and walks the chained verify's
`argmax_ids` against them. It chains only greedy rows whose request takes the
scheduler's full `1 + K` tokens, with room for two more steps below
`max_model_len` and, in `mamba_cache_mode` `align`, the same window column for
every count (`MambaCacheGroups.chain_window_safe`); a prefill, a new or resumed
request, a preemption, a forced prefix-cache reset, structured output, a
moved row or a second unread verify drains the predecessor first. Sources:
[spec_chain.py](../src/vllm_tt_plugin/spec_chain.py) and
[mamba_state.py](../src/vllm_tt_plugin/mamba_state.py).
