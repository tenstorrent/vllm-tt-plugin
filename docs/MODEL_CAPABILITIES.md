# Model capabilities

The TT plugin selects scheduling, sampling, device setup, and speculative
decoding behavior from the model class that tt-metal provides. Model authors
declare supported behavior in the class-level `model_capabilities` dictionary.
`TTPlatform.check_and_update_config` validates launch requests against
`model_capabilities`, and runtime consumers select the supported execution
path. This reference describes every `model_capabilities` key the plugin reads
and the separate `SpecPlan` that a speculative model returns.

Capability declarations describe an interface promise. A declaration does not
establish numerical correctness, device execution, or performance for every
model, mesh, prompt length, and batch size. Use the selected tt-metal model
class and the corresponding model guide to determine model-specific support.
See [README.md](../README.md) for launch instructions and
[SPEC_DECODE_CONTRACT.md](SPEC_DECODE_CONTRACT.md) for speculative execution,
request restrictions, and lifecycle requirements.

## Ownership and resolution

`model_capabilities` belongs to the model class, not to
`--additional-config`. Operators request behavior through vLLM arguments and
`--additional-config '{"tt": ...}'`; operators cannot enable unsupported model
behavior by supplying internal `_tt_*` values.

1. `TTPlatform.check_and_update_config` obtains the selected model class and
   reads `model_capabilities` before model construction.
2. `TTPlatform.check_and_update_config` updates scheduler and cache settings,
   validates incompatible combinations, and stores resolved TT values on
   `VllmConfig.additional_config` through `store_tt_*` helpers.
3. `TTWorker.init_device` repeats `TTPlatform.check_and_update_config` in each
   worker process and reads `model_capabilities["fabric_config"]` before
   `open_mesh_device` opens the device mesh.
4. `TTModelRunner` and `TTAsyncDecodeController` also read runtime declarations for
   sampling, hidden-state handoff, and resident decode. Model authors must
   keep class and instance declarations consistent.

The plugin uses defaults when a declaration is absent. Most Boolean
declarations use truth-value checks, not a common strict schema. Model authors
must supply actual Boolean values and correctly typed numeric values. Unknown
dictionary keys have no effect unless a plugin consumer reads them.

Sources: [platform.py](../src/vllm_tt_plugin/platform.py),
[config.py](../src/vllm_tt_plugin/config.py),
[worker.py](../src/vllm_tt_plugin/worker.py),
[model_runner.py](../src/vllm_tt_plugin/model_runner.py), and
[async_decode.py](../src/vllm_tt_plugin/async_decode.py).

## Scheduling and output capabilities

| `model_capabilities` key | Default when absent | Meaning and current validation |
|---|---|---|
| `supports_chunked_prefill` | `False` | Permits scheduler-driven chunked prefill when `scheduler_config.enable_chunked_prefill` is also true. The platform disables chunked prefill for `output_tokens_per_step > 1` regardless of this declaration. When chunked prefill remains enabled, the platform sets `disable_chunked_mm_input=True`. |
| `supports_prefix_caching` | `False` | Permits vLLM automatic prefix caching. The platform disables requested prefix caching when this declaration is false. The platform rejects `supports_prefix_caching=True` together with `output_tokens_per_step > 1`. |
| `output_tokens_per_step` | `1` | Declares the committed output width. The platform requires an integer at least 1 and rejects Boolean values. A value above 1 selects block-output execution. Contract-based speculation instead uses `output_tokens_per_step=1` and an admitted `SpecPlan`. |
| `tt_adaptive_block_output` | `False` | Allows an eligible solo decode to commit a block and requires batched decode to use the ordinary one-token path. The platform rejects this declaration unless `output_tokens_per_step > 1`. Adaptive block output relaxes the plain block-output concurrency, data-parallel, executor, async-scheduling, and device-sampling restrictions described below. |
| `tt_adaptive_block_max_prompt_tokens` | `0` | Limits prompts eligible for an adaptive block session. `0` means no declared limit. Requests with longer prompts use one-token decoding throughout their lifetime. The platform converts the value with `int`, rejects negative values, and rejects a nonzero value without `tt_adaptive_block_output`. The model must apply the same prompt-length limit as the scheduler. |
| `tt_block_kv_extent_tokens` | `0` | Declares the total KV positions one block-output decode may touch, including verification and carried tokens beyond the committed output. `0` means undeclared. The platform converts the value with `int`, rejects negative values, and requires a nonzero value to cover at least `output_tokens_per_step`. `TTScheduler` reserves the maximum of existing lookahead, twice `output_tokens_per_step`, and `tt_block_kv_extent_tokens`. This declaration does not describe generic speculative proposal writes. |

Plain block-output execution requires `max_num_seqs=1`, data-parallel size 1,
`distributed_executor_backend` unset or `uni`, synchronous scheduling, and
`sample_on_device_mode="all"`. Adaptive block-output execution permits multiple
requests, data parallelism, the `mp` executor, async scheduling when the model
declares `supports_async_decode`, and `sample_on_device_mode="decode_only"`.

Both block-output variants reject `speculative_config`, prefix caching,
`max_device_top_k`, explicit `diffusion_config`, custom `logits_processors`,
and the Rust frontend. Both variants require callable `release_request` and
`release_persistent_capture` hooks. Adaptive block-output execution additionally
requires `note_state_slots_moved` when `max_num_seqs > 1`.

Sources: [platform.py](../src/vllm_tt_plugin/platform.py),
[scheduler.py](../src/vllm_tt_plugin/scheduler.py), and
[SCHEDULING.md](SCHEDULING.md).

## Sampling and async capabilities

| `model_capabilities` key | Default when absent | Meaning and current validation |
|---|---|---|
| `supports_sample_on_device` | `False` | Permits the requested `sample_on_device_mode`. The platform rejects any non-`None` sampling mode when this declaration is false. Runtime sampling requirements can still select host sampling. |
| `max_device_top_k` | `None` | Bounds stochastic device sampling. `TTModelRunner.check_perform_device_sampling` selects host sampling when any selected sampling row has nonzero temperature and `top_k < 1` or `top_k > max_device_top_k`. Greedy rows do not trigger this restriction. The platform rejects any non-`None` value on a block-output model. The plugin does not otherwise validate the numeric type or range of `max_device_top_k`; model authors must supply the actual supported bound. |
| `supports_device_penalties` | `True` | Permits device sampling with active penalties. When this declaration is false and `InputBatch.no_penalties` is false, `TTModelRunner.check_perform_device_sampling` selects host sampling. The default preserves the existing behavior of model implementations that predate this declaration. |
| `supports_async_decode` | `False` | Declares split decode submission/readback and the resident-input behavior specified by `DECODE_RELOAD_CONTRACT.md`. The platform warns and disables requested async scheduling when this declaration is false. `TTAsyncDecodeController.plan_decode_reload` also reads this declaration when deciding whether ordinary device-sampled traced decode may reuse resident inputs. |
| `supports_async_spec_decode` | `False` | Declares that `read_decode_output` can read a wide speculative verify and preserve the verify hidden handle until proposal. If async scheduling remains enabled with `speculative_config`, the platform rejects a model without this declaration. `supports_async_decode=True` alone does not satisfy this requirement. |

The async capability combination has a specific order: the platform first
disables async scheduling for a model without `supports_async_decode`; the
platform then checks `supports_async_spec_decode` only if speculative decoding
and async scheduling are both enabled. Neither declaration forces async
scheduling to remain enabled when an upstream executor or launch restriction
requires synchronous execution.

Device sampling also falls back to the host for supported host-only sampling
controls, structured output, and unsupported logprob requests. These runtime
restrictions remain in force when `supports_sample_on_device=True`. Generic
speculative request restrictions remain separate from ordinary sampling:
the current speculative acceptance path requires greedy requests. See
[SPEC_DECODE_CONTRACT.md](SPEC_DECODE_CONTRACT.md).

Sources: [platform.py](../src/vllm_tt_plugin/platform.py),
[model_runner.py](../src/vllm_tt_plugin/model_runner.py),
[async_decode.py](../src/vllm_tt_plugin/async_decode.py), and
[DECODE_RELOAD_CONTRACT.md](DECODE_RELOAD_CONTRACT.md).

## Device setup capability

| `model_capabilities` key | Default when absent | Meaning and current validation |
|---|---|---|
| `fabric_config` | `None` | Supplies a dictionary of keyword arguments to `ttnn.set_fabric_config`, using TTNN enum and configuration objects. `set_fabric` merges hardware defaults, then model defaults, then explicit `tt.fabric_config` and `tt.fabric_reliability_mode` launch overrides. `set_fabric` does nothing for a single-device mesh. The plugin forwards the merged dictionary to TTNN; the model declaration is not the string accepted by the operator's `tt.fabric_config` setting. |

Sources: [worker.py](../src/vllm_tt_plugin/worker.py) and the
[README fabric configuration example](../README.md#model-fabric-configuration).

## Speculative capabilities and admission

| `model_capabilities` key | Default when absent | Meaning and current validation |
|---|---|---|
| `supports_spec_decode` | `False` | Allows speculative admission to continue. With `speculative_config`, the platform requires this declaration and a callable model-class `spec_plan`. The declaration does not enable speculation by itself. |
| `spec_requirements` | Empty tuple | Lists supported contract features: `device_propose`, `hidden_feed`, `drafter_scores`, and `paged_drafter_cache`. The plugin translates a vLLM method into required features and rejects missing features. `normalize_declared_values` rejects a single string, unknown values, and duplicate values. |
| `spec_hidden_handoff` | Empty tuple | Lists `on_device`, `roundtrip`, or both. `resolve_speculative_plan` validates a nonempty declaration whenever the requested method requires `hidden_feed` or the model declares `hidden_feed`. `normalize_declared_values` applies the same name and duplicate checks. Without a hidden feed, speculative admission does not validate this declaration. |

`resolve_speculative_plan` returns `None` without reading speculative
declarations when the launch has no `speculative_config`.

| Requested vLLM method | Required `spec_requirements` values | Current plugin execution support |
|---|---|---|
| `ngram` | None | Host n-gram proposal and greedy verification, subject to the model's `spec_plan`. |
| `custom_class` | `device_propose` | Model-owned `propose_draft_tokens`. The `model` setting must be exactly `vllm_tt_plugin.model_owned_drafter`; this value is a dispatch marker, not a class the plugin imports. A model that consumes target hidden state must also declare `hidden_feed` and `spec_hidden_handoff`. |
| `suffix` | None | Recognized, but rejected because the runner has no suffix proposer. |
| `draft_model` | `device_propose`, `paged_drafter_cache` | Rejected because the plugin does not allocate a scheduler-owned drafter cache. |
| `medusa`, `mlp_speculator`, and vLLM `EagleModelTypes` values | `device_propose`, `hidden_feed` | Recognized, but rejected because the runner has no proposer for these method names. A model-owned implementation uses `custom_class` instead. |

`TTPlatform.check_and_update_config` rejects generic speculative decoding with
block-output execution or lane-DP. Standard multi-process DP does not use the
lane-DP rejection, but the selected model must admit the concurrency of each
engine. Admission is not device validation for a particular DP deployment.

`TTModelRunner._narrow_steps_serve_the_drafter` reads `spec_requirements` and
`spec_hidden_handoff` after model loading. If a model-owned drafter declares
`hidden_feed` and includes `roundtrip`, the runner disables narrow draftless
decode so every decode supplies a verify hidden handle. This applies even if
the model also declares `on_device`.

Sources: [spec_admission.py](../src/vllm_tt_plugin/spec_admission.py),
[spec_decode.py](../src/vllm_tt_plugin/spec_decode.py), and
[model_runner.py](../src/vllm_tt_plugin/model_runner.py).

## `SpecPlan` fields and limits

The model class implements
`spec_plan(vllm_config, max_num_seqs, requested_k)` and returns a `SpecPlan` or
`SpecReject`. The platform passes the resolved engine-local concurrency as
`max_num_seqs`. `spec_plan` must not rely on resolved `get_tt_*` state because
speculative admission does not guarantee that state exists yet.

| `SpecPlan` field | Default | Meaning and current validation |
|---|---|---|
| `effective_k` | Required | Draft length, at least 1. Admission rejects `effective_k > requested_k`. The platform writes a reduced `effective_k` back to `speculative_config.num_speculative_tokens` so vLLM uses the admitted lookahead width. |
| `lanes_per_request` | Required | Model decode rows consumed by one speculative request, at least 1. This value does not mean lane-DP lanes. The plugin validates the lower bound but does not enforce a row budget. |
| `extra_bytes_per_seq` | Required | Nonnegative fixed device bytes per speculative request. The plugin does not reserve or enforce this byte budget. |
| `extra_bytes_per_token` | Required | Nonnegative device bytes per KV token beyond target KV storage. The plugin does not reserve or enforce this byte budget. |
| `accept_modes` | Required | Nonempty sequence using `logits`, `argmax_ids`, or `fused_sample`. `SpecPlan` normalizes the sequence to a tuple and rejects strings, unknown names, and duplicates. Admission currently requires `argmax_ids` among the offered modes because the runner implements greedy acceptance only. |
| `drafter_state` | Required | One of `internal`, `paged`, or `shared_with_target`. Admission rejects `paged` because the plugin has no scheduler-owned drafter-cache allocation. |
| `drafter_target_cache_requires` | `()` | Sequence using `named_layer_caches` and `absolute_positions`. `SpecPlan` rejects strings, unknown names, duplicates, and a nonempty sequence unless `drafter_state="shared_with_target"`. The plugin does not verify that the target cache satisfies these requirements. |
| `supports_narrow_decode` | `False` | Declares ordinary `[B, 1]` decode support alongside wide `[B, 1+K]` verification. The runner uses this permission for draftless steps when the drafter's hidden-state requirements allow narrow decode. This value is a `SpecPlan` field, not a `model_capabilities` key. |

`SpecPlan.block_width` is `1 + effective_k` for wide verification.
`SpecPlan.accepted_counts_range` is the inclusive range
`(1, 1 + effective_k)`. Neither property describes the tensor shape of an
ordinary narrow decode.

For `drafter_state="shared_with_target"`, model authors describe constraints
on target caches through `drafter_target_cache_requires`. The contract expects
zero additional drafter-cache byte costs; `SpecPlan.__post_init__` does not
enforce zero byte values. A `named_layer_caches` requirement means the model
must keep the named target-layer caches addressable for the request lifetime.
An `absolute_positions` requirement means the model must preserve the
drafter's addressing semantics, including matching bounded-ring wrap and
window behavior where applicable.

`SpecReject.reason` must be nonempty. `SpecReject.supported_k` defaults to an
empty tuple; when present, `supported_k` must contain distinct draft lengths
of at least 1 for the same `max_num_seqs`. Admission reports the refusal and
supported alternatives to the operator.

`store_tt_spec_plan` stores a plain dictionary on
`VllmConfig.additional_config` for serialization and hashing.
`get_tt_spec_plan` reconstructs `SpecPlan`, which repeats `SpecPlan` validation.
Model authors must still reject infeasible resource combinations in
`spec_plan`; a structurally valid `SpecPlan` is not proof that device memory,
row capacity, or shared target-cache constraints are satisfied.

Sources: [spec_decode.py](../src/vllm_tt_plugin/spec_decode.py),
[spec_admission.py](../src/vllm_tt_plugin/spec_admission.py), and
[config.py](../src/vllm_tt_plugin/config.py).

## Hooks that are not capability keys

`get_kv_cache_spec` opts a model into hybrid KV-cache layouts.
`release_request`, `release_persistent_capture`, and
`note_state_slots_moved` define model state ownership and cleanup.
`spec_plan` declares speculative feasibility, and `propose_draft_tokens`
implements model-owned proposal. Adding a similarly named dictionary key
does not implement any of these hooks.

Model authors should add new feature gates through explicit capability
declarations and matching plugin consumers, with linked changes when tt-metal
and the plugin must land together. Existing identity checks for Galaxy/GPT-OSS
lane conversion and GPT-OSS top-K logprobs are not a pattern for new feature
gates.
