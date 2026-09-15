# Speculative decoding: what a tt-metal model must implement

This is the model-side contract the plugin admits against. A tt-metal model
class that wants speculative decoding through this plugin implements everything
below. The plugin side that reads it is
`src/vllm_tt_plugin/spec_admission.py`, and the value types it exchanges are
`src/vllm_tt_plugin/spec_decode.py`.

The design this implements is
https://github.com/tenstorrent/vllm-tt-plugin/issues/110.

## Status

A model implementing this contract serves speculative decoding, within one
boundary. What runs:

- the **n-gram** method, and no other. The runner proposes for `ngram` only, so
  every other method vLLM knows is refused at configuration time rather than
  admitted to draft nothing.
- the **`argmax_ids`** accept mode, and no other. A plan offering only `logits`
  is refused, because the runner requests `argmax_ids` on every step.
- **greedy requests**, and no others. A request carrying a temperature,
  logprobs, structured output, a token filter or a penalty is refused per
  request: the accept walk compares token ids and never sees logits, so it
  cannot arbitrate any of those, and answering greedily anyway would change
  what was asked for without saying so.
- the **synchronous** decode tail. A launch combining speculation with
  asynchronous scheduling is refused, because the accept walk lives in the
  synchronous path.
- **front-packed** execution. Lane mode is refused: it builds its device input
  from `TTLaneInputBatch`, which has no candidate-block builder.

Every one of those is a refusal that raises with the offending values, never a
silent fallback. A device drafter, the sampled accept walk, structured output
over drafts, and `fused_sample` each need their own execution path before the
matching refusal can go.

## 1. Capability declarations

Four `model_capabilities` entries, read only when the launch carries a
`speculative_config`. Absent keys default as shown, following the plugin's
existing default-if-absent convention.

| Key | Default | Meaning |
| --- | --- | --- |
| `supports_spec_decode` | `False` | Master gate. Absent means the model cannot serve any speculative configuration. |
| `spec_requirements` | `[]` | What the drafter can serve: `device_propose`, `hidden_feed`, `drafter_scores`, `paged_drafter_cache`. |
| `spec_hidden_handoff` | `[]` | How the target hidden state reaches a device drafter: `on_device`, `roundtrip`. Required when the method needs `hidden_feed`. |
| `output_tokens_per_step` | `1` | Must stay `1`. A value above 1 selects the block-output rail, which cannot be combined with speculation. |

A model never names a vLLM speculative method. The plugin owns the mapping from
a method name to the requirements that method places on the model, so a new
upstream method is a plugin change rather than a model release.

## 2. The `spec_plan` classmethod

```python
@classmethod
def spec_plan(cls, vllm_config, max_num_seqs: int, requested_k: int) -> SpecPlan | SpecReject
```

Called once per process at configuration time, before any instance exists. The
plugin calls it positionally, so parameter names are not part of the contract.

**Arguments.** `max_num_seqs` is the concurrency the plan is dimensioned
against, and it is the value that is final for the launch, after the platform
has folded data parallelism into lanes. `requested_k` is the operator's
`num_speculative_tokens`.

**`vllm_config` carries no Tenstorrent platform state.** A `spec_plan`
implementation must not call `get_tt_output_tokens_per_step`,
`get_tt_data_parallel_size` or any other `get_tt_*` helper on it: the platform
has not stored them at that point, so two of those return a default of 1
silently and `require_tt_output_tokens_per_step` raises.

**Returns** a `SpecPlan` when the model can serve that point, or a `SpecReject`
when it cannot. It must not raise, and it must not return anything else.

**Rules the plugin enforces.** `effective_k` must not exceed `requested_k`; a
model that can serve less returns a lower value and a model that can serve
nothing returns `SpecReject`. `drafter_state` must not be `paged`, which needs
a scheduler-owned drafter cache the plugin does not yet allocate.
`accept_modes` must offer at least one of `logits` or `argmax_ids`; offering
`fused_sample` in addition is fine and does not make the model less admissible.

**`SpecReject.supported_k`** carries the draft lengths that would have worked at
that concurrency, and the plugin quotes it to the operator. Populate it.

## 3. `SpecPlan` fields

See `src/vllm_tt_plugin/spec_decode.py` for the authoritative definitions and
their validation. In summary: `effective_k` is the draft length the model will
actually serve; `lanes_per_request` is the decode rows one speculating request
occupies, which is not a lane-DP lane; `extra_bytes_per_seq` and
`extra_bytes_per_token` are the device bytes a speculating request costs, fixed
and per KV token respectively; `accept_modes` names what the verify call
returns; `drafter_state` says where the drafter's own state lives; and
`drafter_target_cache_requires` carries what a drafter sharing the target's
caches needs to stay true of them.

`lanes_per_request`, `extra_bytes_per_seq` and `extra_bytes_per_token` are
declared but not yet budgeted against. The byte fields need a bytes-per-KV-token
conversion the block-count function does not have, and the row check needs the
runner. Declare them accurately anyway: they are what the budgeting will read.

## 4. What the runner sends on a decode step

Once a launch carries a resolved `SpecPlan`, `TTModelRunner` widens every
decode step to the candidate block, whether or not any request has drafts
pending. `TTModelInput` carries it as four values.

| Value | Shape | Meaning |
| --- | --- | --- |
| `input_tokens` | `[B, 1+K]` int32 | Column 0 is the row's last committed token; columns 1..K are its pending drafts |
| `input_positions` | `[B, 1+K]` int32 | Column j is that row's position for column j's token |
| `num_valid_drafts` | `[B]` int32 | How many of a row's K draft columns are real, in `[0, K]` |
| `accepted_counts` | `[B]` int32 | How many tokens that row's previous step committed, in `[1, 1+K]` |

`B` is the padded decode batch, the same capacity a plain decode is padded to,
so the block carries rows for requests that do not exist. A padding row holds
token 0 and position -1 in every column, `num_valid_drafts` 0, and
`accepted_counts` 1.

A row with fewer than K valid drafts pads its own tail columns, and the cap is
per row: one request whose drafter ran short never shortens another request's
speculation. A padded column carries `PLACEHOLDER_TOKEN_ID` for the token and
-1 for the position, which is the same no-position marker a padding row
carries. A padded column must not be verified, and `num_valid_drafts` is what
says where a row's real columns stop.

`accepted_counts` is a count and not an index. It is never 0, it is 1 after a
prefill and after a non-speculating step, and a model selecting a per-candidate
state slot selects slot `accepted_counts - 1`. A prefill drops a row's pending
drafts and resets its count to 1, because a request resumed from preemption
replays its own history and its drafts no longer sit at the positions they were
drafted for.

### The two call shapes

A model that does not declare `supports_narrow_decode` only ever sees the wide
call. One that does sees the narrow one on a step where no row carries a draft.

| | wide call | narrow call |
| --- | --- | --- |
| `tokens` | `[B, 1+K]` int32 | `[B, 1]` int32 |
| `start_pos` | `[B, 1+K]` int32 | `[B]` int32, 1-D |
| `draft_token_ids` | `[B, K]` int32 | `[B, K]` int32, every entry padding |
| `num_valid_drafts` | `[B]` int32 | `[B]` int32, every entry 0 |
| `accepted_counts` | `[B]` int32 | `[B]` int32 |
| `spec_mode` | present | present |
| return | `VerifyOutput`, `argmax_ids` `[B, 1+K]` | `VerifyOutput`, `argmax_ids` `[B, 1]` |

The narrow call is the ordinary decode call: its `tokens` and `start_pos` are
exactly the shapes a non-speculating decode sends, so a model that declares it
implements no third shape. Both `[B]` side tensors still come with it, because
`accepted_counts` is how a model picks the candidate state slot its previous
step committed from whatever this step's width is.

Its return carries one column, which is that step's committed token, and the
runner reads it with an accepted count of 1. `SpecPlan.block_width` describes
the wide call only.

## 4a. The verify call

Verify is not a new entry point. It is the model's existing `decode_forward`,
whose `tokens` and `start_pos` arrive `1+K` wide, plus three added keyword
arguments:

```python
def decode_forward(
    self,
    tokens,            # [B, 1+K]
    start_pos,         # [B, 1+K]
    *,
    num_valid_drafts,  # [B] int32
    accepted_counts,   # [B] int32
    spec_mode: str,    # the accept mode the runner wants
    **kwargs,          # page_table, kv_cache, sampling_params, reload commands
) -> VerifyOutput
```

Everything a decode already receives it still receives, under the name it
already has. A model implementing this grows three keyword arguments and a
wider block; it does not grow a second call.

## 5. What a verify returns, column by column

The input block's column `j` carries the token at candidate position `j`:
column 0 the row's last committed token, columns 1..K its drafts. A verify's
return is indexed by **drafted position**, not by input column:

| Return column | What it is |
| --- | --- |
| `j` for `j < K` | The model's choice at candidate position `j`, which is the token draft `j` has to match |
| `K` | The bonus, the choice that follows a fully accepted row |

So for `argmax_ids` the committed block is the return truncated at the accepted
count, with no column spent echoing an input the runner already holds, and for
`logits` column `j` is the distribution the accept test for draft `j` reads.
A row with `num_valid_drafts` of `n` finds its bonus at column `n`.

This is upstream's layout: its greedy kernel compares `target_argmax[pos]`
against `draft_token_ids[pos]` and writes the committed token at `pos`, so a
kernel or a test ported from upstream needs no index adjustment. It is spelled
out because it is invisible from the shape: `[B, 1+K]` in and `[B, 1+K]` out
admits an off-by-one that only shows up as wrong output text.

## 6. What the plugin does with a refusal

Every refusal raises `ValueError` at configuration time, naming the offending
values and the command-line flag that changes them. Speculation is never
disabled silently, because a server that accepts the flags and serves no
speculation reports a speedup it did not achieve.
