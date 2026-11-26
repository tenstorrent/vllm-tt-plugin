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
| Live rows | Every live row `i` of the call is the live row `slot_remap[i]` of the predecessor (row `i` when `slot_remap` is absent). A predecessor row whose request finished, or that the step does not schedule, is not a live row of the call. |
| `slot_remap` | As on every decode: the gather permutation of the step (row `i` reads state slot `slot_remap[i]`). After a decode `TTModelRunner` keeps each request's state slot at its row, so on a chained verify `slot_remap[i]` is also the predecessor row that row `i` continues. A row moves when a predecessor row leaves and `InputBatch.condense` moves the last row into its place. |
| `tokens`, `start_pos` | Placeholders on live rows: `PLACEHOLDER_TOKEN_ID` in every column, and the positions of a predecessor that accepted all its drafts (its column-0 position plus `1 + K + j` in column `j`). Padding rows as on any verify. |
| `num_valid_drafts` | On live rows, the most drafts the row may verify (`SpecPlan.draft_cap` of the step's live rows). |
| `accepted_counts` | 1 on every row. |
| `mamba_state` | Each `MambaGroupState` also carries `start_blocks_by_count` `[rows, 1 + K]`: column `c - 1` is the block holding the row's start state if the predecessor committed `c` tokens for it. `start_blocks` holds column K. `windows` is the window vLLM allocated by the scheduled count, one for every count, as on a drained verify of the step. |

The model builds each live row from the predecessor's device result. With `c`
the number of tokens the predecessor committed for the row by the model's own
acceptance of its `argmax_ids`, the row is the predecessor's committed token
`c - 1` in column 0, then `n = min(num_valid_drafts, offered)` drafts that the
model's drafter made from that commit (`offered` is 0 when the drafter
declines the row and K otherwise), at the predecessor's column-0 position plus
`c`, starting from `start_blocks_by_count[row, c - 1]`. Here `c`, the
committed tokens, the drafts and the column-0 position are those of the
predecessor row `slot_remap[row]`; `start_blocks_by_count`, `windows`,
`num_valid_drafts` and the sampling parameters are the call's own row. The
return is the `VerifyOutput` of that block, as on any verify. A model that
does not support a moved row raises on a `slot_remap` that is not the
identity.

`TTModelRunner` then reads the predecessor, walks it, commits it and calls
`propose_draft_tokens` for it. That call must return the drafts the chained
verify ran (`draft_token_ids[row, :n]` with `num_valid` the drafter's offer),
and must raise when the runner's commit differs from the model's own
acceptance: the chained verify continued the model's count. The runner fills
the chained verify's rows from that proposal and walks the chained verify's
`argmax_ids` against them. It chains only rows the id walk certifies (greedy
rows, and sampled rows as below) whose request takes the scheduler's full
`1 + K` tokens, with room for two more steps below
`max_model_len`; a prefill, a new or resumed request, a preemption, a forced
prefix-cache reset, structured output, a request that is not a row of the
predecessor or a second unread verify drains the predecessor first. A row
that moved because a predecessor row left does not: `slot_remap` names its
predecessor row. In `mamba_cache_mode` `align` a chained row can
be next to a block boundary: its `windows` can then begin one column past the
column that holds, by position, its last query token at a low count, and that
count's start block is an earlier window column than `c - 1`, or the block
before the window. The model reads and writes these blocks as on any verify
(see [speculative verification](MODEL_EXECUTION.md#speculative-verification)),
and the step names, for the count the model computes, the blocks that a
drained verify of the same step names.

A model that also declares `spec_ids_are_sampled` (see
[sampled ids](SPEC_DECODE_CONTRACT.md#4i-sampled-ids-spec_ids_are_sampled))
also chains device-sampled rows. The runner chains a sampled row only when the
predecessor sampled on the device (`TTModelInput.perform_device_sampling`) and
the step being built samples on the device too; `plan_chain` reads that
decision before `_update_states`, and the build raises when its own decision
does not certify every chained row. The call then carries the step's
`sampling_params`, as any device-sampled verify does, and the model must:

| Duty | Chained verify of a sampled row |
|---|---|
| Count | `c` is the model's acceptance over the predecessor's draws, which is the walk the runner runs over the same `argmax_ids`. |
| Draws | Each column of the row is a draw under the row's own temperature, top-k, top-p and seed, as on an unchained device-sampled verify. |
| Seeds | A seeded row's seed for each column follows from the position the device computed (the predecessor's column-0 position plus `c`, plus the column), so the draw at a position equals the draw an unchained verify of the request makes at that position. An unseeded row takes fresh randomness on every step. |

A host-sampled step, and a row the draws do not certify (logprobs, structured
output, a token filter, a pending `min_tokens`, and a penalty unless the model
declares `supports_spec_device_penalties`), drain the predecessor first. A
model declaring `supports_spec_device_penalties` also chains penalised rows:
the step's `output_tokens` lack the predecessor's commit, and the model adds
that commit to the row's penalty history itself
([penalties on the verify](SPEC_DECODE_CONTRACT.md#4j-penalties-on-the-verify-supports_spec_device_penalties)). Sources:
[spec_chain.py](../src/vllm_tt_plugin/spec_chain.py) and
[mamba_state.py](../src/vllm_tt_plugin/mamba_state.py).

## Prefill behind an unread verify

A model that declares `supports_prefill_over_pending_spec` (see
[model capabilities](MODEL_CAPABILITIES.md)) lets `TTModelRunner` call
`prefill_forward` while one `argmax_ids` verify it returned unread has not
been read, committed or proposed. Without it the runner reads that verify,
commits it and calls `propose_draft_tokens` for it before it builds the
prefill.

In order, with verify n + 1 unread, request R finished at verify n (so R is
a live row of n + 1) and request J waiting:

1. `TTScheduler` returns J's prefill step, with R in `finished_req_ids`, and
   no release step before it (`TTScheduler._prefill_runs_behind_verify`).
2. `TTModelRunner.execute_model` does not read verify n + 1
   (`spec_chain.prefill_over_pending`): it is the only pending step, an
   unread `argmax_ids` verify, and the step is prefill work only, with no
   resumed or preempted request, no multimodal input and no forced
   prefix-cache reset (`scheduler.prefill_qualifies_behind_verify`, the rule
   the scheduler applied).
3. `_update_states` calls `release_request` for R's state slot, then the
   build hands J a free slot, often R's (`empty_slots`), and calls
   `prefill_forward`.
4. After the prefill the engine reads verify n + 1; the next build commits
   it and calls `propose_draft_tokens` for its rows, R's row included, with
   the walk's commit.
5. The verify after that is built from host state: it does not chain on
   verify n + 1.

The model must:

| Duty | Prefill behind an unread verify |
|---|---|
| Order | Enqueue the prefill's device work after the verify and the reads it returned, on the same in-order queue. |
| Results | Keep what `read_decode_output` and `propose_draft_tokens` of that verify use valid across the prefill. |
| Slot reuse | Accept a prefill row in the state slot of a live row of that verify whose request finished. That verify's `propose_draft_tokens` reads none of the new request's per-slot state, and leaves the new request's per-slot state as a propose before the prefill would. |
| Propose | Return the drafts and raise on a commit that differs from the model's own acceptance, as on any verify. |

Tokens do not change. The verify's tokens are published after the prefill
instead of before it.
