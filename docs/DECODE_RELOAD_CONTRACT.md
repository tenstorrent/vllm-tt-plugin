# TT Decode Reload Contract

Async device sampling deliberately allows the plugin's host token state to
trail the TT device by one decode step. Only the plugin runner knows whether
the host tensors are authoritative. A contract-aware tt-metal adapter declares
`decode_input_update_contract = 1` and receives four independent boolean
commands on every decode:

## Commands

| Command | Required action |
| --- | --- |
| `reload_inputs` | Copy every forward input: token, position, RoPE inputs, and page tables. |
| `reload_page_table` | Copy only page-table inputs. Ignore when `reload_inputs` is true. |
| `reload_sampling_params` | Upload temperature, top-k/top-p, penalties, seeds, and logprob configuration. |
| `reset_sampling_state` | Rebuild mutable penalty and RNG state for the current request layout. |

`reload_inputs` already includes page tables, so the legal forward-input modes
are "everything", "page tables only", and "nothing". The plugin asserts that
`reload_inputs` and `reload_page_table` are never both true. It also asserts
that `reset_sampling_state` implies `reload_inputs`, because a sampler may align
seed counters from host positions only on a step that makes those positions
authoritative.

These are commands, not hints. A version-1 adapter must not replace them with
model-local sampling-mode checks, tensor comparisons, or prior-call
heuristics.

### Command defaults

The plugin sends all four commands explicitly on every version-1 decode. If an
adapter exposes defaults for direct callers, only the host-authoritative
defaults are safe: `reload_inputs=True` and the other three commands `False`.
An adapter that accepts arbitrary `**kwargs` must still reject the legacy
`reset_batch` keyword after declaring version 1, so an older plugin fails
loudly instead of silently taking an unintended reload path.

`decode_layout_changed` is internal plugin lifecycle state. The runner
translates it into the four commands for version 1 and into the historical
`reset_batch` keyword for a legacy device-sampling call. It is not forwarded to
a version-1 adapter.

`slot_remap` is data rather than reload policy. `remap[i] = j` means row `i`
reads the state currently in slot `j`. It is an absolute permutation, not a
delta, and applying it twice is incorrect. The standalone plugin sends it for
both host- and device-sampling decode, including to legacy adapters, because
model-owned recurrent, convolution, or RoPE state remains slot-indexed even
when sampling runs on the host. Every slot-owning subsystem, including a
temporarily dormant device sampler, must apply the remap exactly once before it
reads that state. A full forward-input reload does not implicitly remap model
or sampling state.

The indices belong to the submission that carries them. For ordinary and
standard-DP execution they are local to that independent runner's slot space.
For single-process lane-DP they index the one merged lane batch. There is no
cross-rank/global standard-DP remap and no rebasing between ranks.

Two superficially reasonable implementations are wrong:

1. **Deliver only during device sampling.** Host sampling can still compact or
   reassign rows. Withholding the remap leaves model-owned recurrent,
   convolution, or cached RoPE state attached to the previous request.
2. **Apply only inside the active device sampler.** Delivering the remap during
   host sampling is insufficient if dormant seed/RNG/penalty state ignores it.
   A later return to device sampling then resumes another request's state.

Every slot-addressable subsystem must therefore consume each supplied remap
exactly once on the accepted decode that carries it. State that is not
addressable by vLLM slot is exempt because it cannot be moved. The known
example is unseeded on-device RNG: its
state lives in per-core hardware PRNG registers with no slot-to-slot gather
primitive. The adapter must document such state and reset/reinitialize it on a
commanded sampling-state transition; the exemption never permits silently
ignoring state that does have a move primitive.

The plugin derives a remap while building the input but advances its state-slot
ownership map only after `decode_forward` accepts the submission. Advancing it
earlier would make a raised call claim a gather the device never ran and would
derive every later remap from a false layout. The sticky layout-change signal is
retired at the same accepted-submission boundary.

Prefill placement is explicit through `empty_slots`: it initializes a new or
resumed request's state in its scheduler-owned destination slot. Lane-DP keeps
live decode rows stable, so merely omitting a live request from a prefill step
does not constitute a layout change.

A remap cannot express slot reuse. A newly admitted request has no predecessor
state to gather, so reuse is reported by `decode_layout_changed`; the resulting
sampling-state reset and model-side authoritative rebuild initialize that slot.

## Modes

- **Host sampling:** tt-metal returns logits and vLLM selects the token. Host
  token and position tensors are authoritative, so every decode fully reloads
  forward inputs.
- **Device sampling:** tt-metal selects the token. A supporting adapter writes
  the selected token into the persistent token buffer for the next decode and
  advances its persistent position in the forward trace.
- **Transition decode:** the first decode; the first decode after prefill; a
  request-layout, sampling-mode, or resume transition. Pending work is drained
  and applied first, then host-authoritative inputs and required sampling state
  are reloaded.
- **Chunked-prefill continuation:** a later prompt chunk for a request already
  present in the persistent batch. Membership can remain unchanged, so the
  plugin classifies it using the scheduler's context-phase marker rather than
  a token-count heuristic. A final prompt chunk can contain one token, while a
  speculative decode may schedule several.
- **Steady device decode:** request layout and sampling mode remain unchanged
  after a valid device-sampling decode. Token and position stay resident and a
  full input reload is omitted.
- **Page-table-only refresh:** a steady device decode whose allocated KV block
  mapping changed. Only page-table trace inputs are copied; token, position,
  and RoPE state remain untouched.

## Transition table

| Transition | Inputs | Page only | Sampling params | Sampling state |
| --- | ---: | ---: | ---: | ---: |
| First decode or prefill → decode | reload | no | reload on device | reset on device |
| Request add/remove/reuse/condense, preemption, or resume | reload | no | reload on device | reset on device |
| Chunked-prefill continuation | reload | no | reload on device | reset on device |
| Host → device sampling | reload | no | reload | reset |
| Steady host sampling | reload every step | no | n/a | n/a |
| Steady device sampling | keep resident | only if allocation changed | keep | keep |
| Decode tracing disabled | reload every step | no | on transition | on transition |
| Adapter without `supports_async_decode` | reload every step | no | on transition | on transition |

Any transition requiring full inputs or sampling-state mutation drains pending
decode work first. The plugin applies the completed token and advances the host
position before building authoritative inputs. It rejects an older result for
a request the current scheduler output finished, preempted, or resumed.
Page-table-only refresh remains overlap-safe because page tables come from the
current scheduler allocation even when token and position tensors are stale.

## Host input authority

The `tokens` and `start_pos` arguments are authoritative only when
`reload_inputs` is true. When it is false they are deliberately one step
behind. An adapter must derive nothing from them—not forward inputs, sampling
state, or RNG counters.

For example, an adapter that ties seeded sampling to the absolute decode
position must align its counter from `start_pos` on a reloading step and then
advance its resident counter exactly once per sampled token. Re-reading stale
`start_pos` on a steady step makes a seeded stream depend on async readback
timing and batch composition.

## Correctness invariant

Immediately after device-sampling step `k`, the resident token slot contains
sampled token `t_k` and the resident position is the position at which `t_k`
must be consumed by step `k+1`.

The base case is a full reload. Before issuing it, the runner finalizes the
older step, rejects rows invalidated by current scheduler lifecycle events,
and applies accepted tokens to host request state. It then copies the
authoritative token, position, layout, and requested sampling state.

For the induction step, a steady device decode issues no full or
sampling-state reload. The trace consumes resident `t_k`, writes exactly one KV
position, advances position once, and sampling writes `t_(k+1)` back to the
persistent token slot. Readback is observational and may finish later. A
page-table-only copy can change KV addressing but cannot overwrite resident
token or position state.

Host sampling always returns to the base case. Prefill and request-layout,
sampling-mode, preemption, and resume transitions also break the steady
invariant and re-establish it through a drain and full reload.

Upstream vLLM replaces each external request id with an internal id carrying a
random suffix before scheduling, so an ordinary abort and resubmit does not
reuse a runner id. Rejection is keyed on scheduler lifecycle events rather than
request-object snapshots: finished, preempted, and resumed ids invalidate the
outstanding result. Invalid output rows are emptied before the scheduler
consumes them, preventing both a host-state append and an extra emitted token.

## Requirements for `decode_input_update_contract = 1`

Every version-1 adapter must satisfy these requirements, whether or not it
advertises async decode:

1. **Exact command handling:** honor all four commands independently and reject
   unsupported combinations loudly instead of adding a heuristic fallback.
2. **Sampling-state ordering:** apply slot remaps before parameter/state reset;
   reset RNG and penalty state only when requested; advance seed state once per
   sampled token; initialize and upload it even when both the requested and
   cached seed are `None`. Explicitly seeded, slot-addressable counters follow
   the request through a remap. Unseeded per-core PRNG state cannot move and is
   instead reinitialized on a commanded reset.
3. **Complete slot remapping:** apply each remap exactly once to all persistent
   state indexed by the vLLM slot, including model recurrent/conv state and a
   dormant device sampler during host sampling.
4. **Stable-buffer lifetime:** keep persistent forward and sampling buffers
   valid through readback and until an explicit command replaces them.

### Partial adapters

An adapter that structurally requires `reload_inputs=True` can still implement
version 1 if it leaves `supports_async_decode` false and rejects any unsupported
resident combination. Silently treating `reload_inputs=False` as true is not a
compatible fallback.

## Additional requirements for `supports_async_decode`

For a version-1 adapter, set
`model_capabilities["supports_async_decode"] = True` only when all of the
following are true:

1. **Split submission and readback:** `decode_forward(...,
   read_from_device=False)` submits without synchronizing, and
   `read_decode_output(..., async_read=True)` can read that exact submission
   later. Readback must not sample, advance position, or mutate decode state.
2. **Persistent token feedback:** device sampling writes the selected token
   into the same persistent token buffer consumed by the next decode.
3. **One position advance:** each successful decode forward advances the
   resident position exactly once. Sampling and readback do not advance it.
4. **Independent page-table refresh:** changed page tables can be uploaded
   without copying or rebinding token, position, or RoPE inputs.
5. **Host input authority:** treat `tokens`, `start_pos`, and RoPE host inputs
   as stale whenever `reload_inputs=False`; derive no forward or sampling state
   from them on that step.

The capability is fail-closed. Without it, the plugin disables async scheduling
for that model and requests a full forward-input reload on every version-1
decode.

## Negotiation and rollout

Adapters opt in with `decode_input_update_contract = 1`. Missing or zero means
legacy mode: the plugin omits the four new keywords, translates the internal
layout signal to `reset_batch` on device-sampling calls, preserves the
adapter's previous overlap behavior, and emits a one-time warning. Existing
standalone-plugin `slot_remap` delivery in both sampling modes is unchanged.
The exact v0 drain points are also preserved: new v1 residency, sampling-mode,
and context-phase checks do not make a legacy adapter drain more often.
Standard-DP v0 runners remain on their prior synchronous path; independent
per-rank overlap is enabled only after the loaded adapter negotiates version 1.

| Plugin | tt-metal adapter | Result |
| --- | --- | --- |
| Old | Legacy / version 0 | Existing behavior |
| New | Legacy / version 0 | Compatibility behavior plus warning |
| New | Version 1+ | Explicit commands and eligible resident overlap |
| Old | Strict version 1 | Unsupported: required commands are absent |

This permits the plugin change to merge before tt-metal adapters are migrated.
`supports_async_decode` remains independent of protocol version: it certifies
resident decode behavior, while `decode_input_update_contract` selects the call
interface.

### Marker placement and inheritance

The marker belongs on the vLLM-facing generator implementation that executes
the commands. Subclasses inherit both the implementation and marker. Moving a
marker onto a shared base changes the opt-in set, so every existing subclass
must be audited first. A subclass that overrides `decode_forward` no longer
inherits the behavior the marker attests to: the override must execute every
command itself or set `decode_input_update_contract = 0` and remain legacy.
Merely re-declaring the marker does not make an override conformant.

Versions greater than 1 must remain backward-compatible supersets of version
1. A later adapter cannot require a new keyword from this plugin because there
is no reverse handshake. Any added command must be keyword-only with a default
that reproduces version-1 behavior; a breaking interface needs a distinct
negotiation key or supported-version range.

Standard multi-process DP needs no TT-specific negotiation. Upstream vLLM runs
one independent scheduler, plugin runner, model, TT mesh, and async submission
stream per DP rank, so each rank plans and applies its own reload commands. No
rank combines inputs or votes on overlap. Single-process lane-DP is separate:
it submits one merged stable-slot device batch and uses the lane coordinator's
exact `TTStepPlan` layout transition. It likewise has no process gather/scatter
or cross-rank vote.

The paired tt-metal implementation initializes decode-only seed state
unconditionally when `reset_sampling_state=True`, including `seed=None`. It
implements the same correctness fix as tt-metal#51556 but does not depend on
that PR.
