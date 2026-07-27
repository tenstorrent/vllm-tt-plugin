# Preserve Explicit `max_model_len` Values

## Goal

Make TT follow vLLM's standard `max_model_len` policy:

- A numeric value is preserved and validated against available KV cache capacity.
- `-1` requests automatic fitting to the largest value that fits.
- TT does not silently convert numeric values or an omitted value into auto-fit mode.

## Current Behavior

`TTPlatform.check_and_update_config` unconditionally sets
`model_config.original_max_model_len` to `-1`. This makes vLLM auto-fit every
configuration, including an explicitly requested numeric value.

TT also retains `_validate_tt_kv_cache_capacity` in `worker.py`. This local
validation predates the override-aware capacity validation added upstream in
vLLM PR 41069. With vLLM 0.24, upstream already validates
`num_gpu_blocks_override` before worker cache initialization.

## Design

Remove the unconditional assignment to `original_max_model_len`. The value
produced by vLLM's configuration parser remains authoritative:

- `None` means the user omitted the option. vLLM uses the model-derived length
  and raises a capacity error if it cannot fit.
- A positive integer is an explicit limit. vLLM preserves it and raises a
  capacity error if it cannot fit.
- `-1` is an explicit auto-fit request. vLLM reduces `max_model_len` to the
  largest value supported by the configured KV cache.

Keep `TTWorker.update_max_model_len`. The vLLM engine calls this hook after an
explicit auto-fit operation to synchronize the effective value to workers.
Update its comment so it no longer claims that TT always opts into auto-fit.

Remove `_validate_tt_kv_cache_capacity`, its now-unused upstream helper import,
and its call from `initialize_from_config`. Upstream performs the same check
earlier while constructing the KV cache configuration and provides an estimated
supported model length in its error.

## Error Handling

No TT-specific fallback or silent correction is added. An oversized numeric
value fails during upstream KV cache configuration with guidance to increase
cache capacity or reduce `max_model_len`. An explicit `-1` remains the supported
way to request automatic fitting.

## Tests

Add host tests around `TTPlatform.check_and_update_config` that verify it
preserves `original_max_model_len` when the value is:

- `None`
- a positive integer
- `-1`

Add a focused worker test that calls `TTWorker.update_max_model_len` on a
lightweight instance and verifies that the shared model configuration is
updated. This protects the explicit auto-fit synchronization path.

Run the focused tests and the repository pre-commit checks before opening the
pull request.

## Scope

This change does not alter TT KV cache sizing, scheduler limits, model defaults,
or upstream vLLM behavior. It only stops TT from forcing all configurations
into auto-fit mode and removes the superseded local capacity check.
