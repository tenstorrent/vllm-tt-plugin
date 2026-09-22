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

- two drafting methods: **`ngram`**, which runs on the host and asks the model
  for nothing, and **`custom_class`**, which is the model's own drafter
  proposing on device through `propose_draft_tokens`. Every other method vLLM
  knows is refused at configuration time rather than admitted to draft
  nothing.
- the **`argmax_ids`** accept mode, and no other. A plan offering only `logits`
  is refused, because the runner requests `argmax_ids` on every step.
- **greedy requests**, and no others. A request carrying a temperature,
  logprobs, structured output, a token filter or a penalty is refused per
  request: the accept walk compares token ids and never sees logits, so it
  cannot arbitrate any of those, and answering greedily anyway would change
  what was asked for without saying so.
- **ordinary decode steps inside a speculating launch**, for a model
  declaring `supports_narrow_decode`: a step with nothing to verify is sent as
  that model's own decode call and can overlap, so configuring speculation does
  not cost a server its asynchronous batched decoding. Section 4d.
- both decode tails. The **synchronous** tail accepts and commits inside the
  step. The **asynchronous** tail defers: acceptance is walked where the
  readback completes, and the commit and the next proposal run on the engine
  thread at the top of the following step. A launch combining speculation with
  asynchronous scheduling is admitted only for a model declaring
  `supports_async_spec_decode`, which is about the model's readback and hidden
  handle rather than about step order: a speculative step never overlaps the
  next one, so verify, accept and propose stay ordered per request.
- **front-packed** execution. Lane mode is refused: it builds its device input
  from `TTLaneInputBatch`, which has no candidate-block builder.

Every one of those is a refusal that raises with the offending values, never a
silent fallback. The sampled accept walk, structured output over drafts,
`fused_sample`, `drafter_scores` and a scheduler-owned paged drafter cache each
need their own execution path before the matching refusal can go.

## 1. Capability declarations

Speculative decoding uses the entries below together with the ordinary
`supports_async_decode` capability. Absent keys default as shown.
`supports_async_decode` is checked for every asynchronous launch, including
launches without `speculative_config`.

| Key | Default | Meaning |
| --- | --- | --- |
| `supports_spec_decode` | `False` | Master gate. Absent means the model cannot serve any speculative configuration. |
| `spec_requirements` | `[]` | What the drafter can serve: `device_propose`, `hidden_feed`, `drafter_scores`, `paged_drafter_cache`. |
| `spec_hidden_handoff` | `[]` | How the target hidden state reaches a device drafter: `on_device`, `roundtrip`. Required when `speculative_config.method` needs `hidden_feed`. |
| `output_tokens_per_step` | `1` | Must stay `1`. A value above 1 selects the block-output rail, which cannot be combined with speculation. |
| `supports_async_decode` | `False` | Ordinary decode supports split submission/readback and the applicable decode reload contract. The platform disables async scheduling when this capability is absent. |
| `supports_async_spec_decode` | `False` | Verification output and any hidden handle remain valid through deferred readback and proposal: see section 4c. Required when speculation is configured and async scheduling remains enabled after the ordinary capability check. |

A model never names a vLLM speculative method. The plugin owns the mapping from
a method name to the requirements that method places on the model, so a new
upstream method is a plugin change rather than a model release.

## 1a. What each static declaration promises, and what it does not

Five declarations decide, before any request arrives, which call shapes a
launch can use at all. The model states each one once, at configuration time,
and the plugin does not revise it while the server runs. None of the five
names a batch size, and none of them turns asynchronous scheduling on. A model
author can predict which input shape the plugin sends from these five plus the
per-step `DraftOutput.num_valid` counts of section 4b, without reading any
other model's adapter.

**`SpecPlan.supports_narrow_decode`** is a field of the `SpecPlan` the model's
`spec_plan` classmethod returns, not a `model_capabilities` key.
`supports_narrow_decode=True` promises two things: the model's
`decode_forward` also serves the ordinary `[B, 1]` decode call of section 4
inside a launch that is otherwise speculating, and the adapter keeps or
rebuilds whatever drafter state a later `propose_draft_tokens` call needs
across such a step. `supports_narrow_decode` selects no batch size at which
the plugin changes shape, enables no asynchronous scheduling, and does not
promise that any step will actually be narrow:
`vllm_tt_plugin.model_runner._step_verifies` decides that per step, from the
runtime values of section 4d.
`TTModelRunner.load_model` revokes `supports_narrow_decode` for exactly one
declared pairing, named under `spec_hidden_handoff` below.

**`supports_async_decode`** is a `model_capabilities` key, and it is not
speculative. It promises the split submission and readback of an ordinary
decode plus the resident-input behavior of
[DECODE_RELOAD_CONTRACT.md](DECODE_RELOAD_CONTRACT.md). `TTPlatform` reads it
for every launch: when the operator asked for asynchronous scheduling and the
model does not declare `supports_async_decode`, `TTPlatform` logs a warning
and sets `scheduler_config.async_scheduling=False`. Declaring it does not
switch asynchronous scheduling on either. The operator still has to ask for
it, and the per-step overlap decision still belongs to
`TTAsyncDecodeController`.

**`supports_async_spec_decode`** is a `model_capabilities` key. It promises
that `read_decode_output` serves a `[B, 1+K]` verify block whose committed
length the host decides after the forward returns, and that the
`VerifyOutput.hidden` handle stays valid until the next step's
`propose_draft_tokens` consumes it. `TTPlatform` requires it only when
`speculative_config` is set and `scheduler_config.async_scheduling` is still
true after the `supports_async_decode` check. It does not authorize one verify
to overlap another, and it promises no speedup. Section 4c gives the check
order.

**`spec_requirements`** is a `model_capabilities` key naming what the model's
drafter can serve: `device_propose`, `hidden_feed`, `drafter_scores`,
`paged_drafter_cache`. `resolve_speculative_plan` compares it against what the
requested speculative method requires and raises naming the missing entries.
It describes the drafter's needs; it does not say when the drafter will offer
drafts.

**`spec_hidden_handoff`** is a `model_capabilities` key naming how the target
hidden state reaches the drafter: `on_device`, `roundtrip`.
`resolve_speculative_plan` requires it to be non-empty whenever `hidden_feed`
is required by `speculative_config.method` or declared by the model.
`TTModelRunner._narrow_steps_serve_the_drafter`, which
`TTModelRunner.load_model` runs once, reads `spec_requirements` and
`spec_hidden_handoff` together: a model declaring `hidden_feed` and
`roundtrip` loses `supports_narrow_decode` for the whole launch, because an
ordinary decode returns no `VerifyOutput` and so hands that drafter no hidden
handle. A model declaring `on_device`, or declaring no `hidden_feed`, keeps
`supports_narrow_decode`.

None of the five is evidence. Each is a statement by the model author that the
adapter behaves that way. The plugin enforces shapes, dtypes and value ranges
at run time; it enforces nothing about whether the adapter's drafter state
actually survives a step it promised to survive.

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
`accept_modes` must include `argmax_ids`, the return format the current runner
executes. A model may also declare `logits` or `fused_sample`, but neither
additional declaration enables an execution path for that return format.

**`SpecReject.supported_k`** carries the draft lengths that would have worked at
that concurrency, and the plugin quotes it to the operator. Populate it.

## 3. `SpecPlan` fields

See `src/vllm_tt_plugin/spec_decode.py` for the authoritative definitions and
their validation. In summary: `effective_k` is the draft length the model will
actually serve; `lanes_per_request` is the decode rows one speculating request
occupies, which is not a lane-DP lane; `extra_bytes_per_seq` and
`extra_bytes_per_token` are the device bytes a speculating request costs, fixed
and per KV token respectively; `accept_modes` names what the verify call
returns; `drafter_state` says where the drafter's own state lives;
`drafter_target_cache_requires` carries what a drafter sharing the target's
caches needs to stay true of them; and `supports_narrow_decode` is the
second-call-shape promise of section 1a.

`lanes_per_request`, `extra_bytes_per_seq` and `extra_bytes_per_token` are
declared but not yet budgeted against. The byte fields need a bytes-per-KV-token
conversion the block-count function does not have, and the row check needs the
runner. Declare them accurately anyway: they are what the budgeting will read.

## 4. What the runner sends on a decode step

For a verification step, `TTModelRunner` builds the candidate block described
below. Section 4d defines when a speculative launch instead uses ordinary
decode. `TTModelInput` carries the verification inputs as four values.

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

`accepted_counts` counts every token the previous step committed for that row,
which includes the correction or bonus token the verify appended after the
matching drafts, not only the drafts that matched. A row whose first draft was
rejected records 1, because the only token committed was the model's own
correction. A row whose K drafts were all accepted records `1 + K`, the K
drafts plus the bonus.

### The two call shapes

A model that does not declare `supports_narrow_decode` sees the verify call on
every decode step of a speculating launch, including a step where no row
carries a draft: that step's `num_valid_drafts` is 0 on every row and its
verify commits one token per row.

A model that declares `supports_narrow_decode` also serves its **own ordinary
decode call** inside a speculating launch, and a step with nothing to verify is
sent as exactly that: no `spec_mode`, neither side tensor, and the sampling
path a non-speculating launch uses. So such a model implements two calls and no
third shape, and section 4d explains which steps take which.

| | verify call | ordinary decode call |
| --- | --- | --- |
| `tokens` | `[B, 1+K]` int32 | `[B, 1]` int32 |
| `start_pos` | `[B, 1+K]` int32 | `[B]` int32, 1-D |
| `num_valid_drafts` | `[B]` int32 | absent |
| `accepted_counts` | `[B]` int32 | absent |
| `spec_mode` | present | absent |
| `sampling_params` | present when the launch samples on device | present when the launch samples on device |
| return | `VerifyOutput`, `argmax_ids` `[B, 1+K]` | whatever this model's decode already returns |

`TTModelInput.draft_token_ids` is a separate `[B, K]` tensor retained by the
runner for host acceptance. `submit_decode` does not pass
`draft_token_ids` to the model; the verify call already carries draft IDs in
`tokens[:, 1:]`.

The three speculative arguments arrive together or not at all, so their
absence is what makes a call the ordinary one. `SpecPlan.block_width`
describes the verify call only.

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

The `VerifyOutput` return is required, not conventional. A step that sends
`spec_mode` is refused at the submission boundary unless the model answers
with one, so a model that declares `supports_spec_decode` and whose
`decode_forward` serves no verify fails by name on its first speculative step.
Nothing further down can catch it instead: a plain decode's `[B, 1]` id tensor
has the two dimensions the accept walk expects, and the walk reads it as a
verify that claimed one token on every row, so the server commits one token
per step for its whole life and reports no error. The reverse is refused too:
a `VerifyOutput` returned from a step that sent no `spec_mode` has no accepted
count to be read against.

## 4b. The propose call, for a model that drafts

A launch whose method requires `device_propose` calls the model after every
commit:

```python
def propose_draft_tokens(
    self,
    num_drafts,            # K
    committed_tokens,      # [B, 1+K] int32, this step's committed block
    committed_positions,   # [B, 1+K] int32, where those tokens sit
    accepted_counts,       # [B] int32 in [1, 1+K], how much of the block is real
    hidden=None,           # the HiddenHandle this step's verify returned
) -> DraftOutput
```

The rows are the verify's rows, padding included, because a drafter's state is
indexed by row and a device graph has one shape. Which entry of the committed
block is a row's last token is `accepted_counts - 1`, the same arithmetic the
verify uses to select a candidate state slot; reading a fixed column instead
continues every row from the same place. `DraftOutput.draft_token_ids` is
`[B, K]` int32, every offered id inside the vocabulary: the runner checks the
dtype and the range before the scheduler stores them, because a stored draft is
verified next step and committed if the model agrees with it, and a fractional
value would be truncated on the way in.

`DraftOutput.num_valid` is `[B]` int32 and optional, how many of each row's `K`
drafts the drafter is offering. A count of 0 declines drafting for that row.
A fully draftless batch can use ordinary decode only when narrow decoding is
available and no live row has an unresolved multi-token commit (section 4d).
Another row with drafts still requires a verification call. `None` means every row offers
all `K`, which is what a drafter that always drafts returns, so a drafter
written before this field keeps working. A drafter with nothing for a row still
returns ids in that row, because a device graph has one shape; those ids are
not read, and not range-checked either, so the row may be padded with
`PLACEHOLDER_TOKEN_ID`. Never encode an empty proposal as a dummy token id:
the runner cannot tell that from a real draft and would verify it. Each count
is checked for dtype, shape and the range `[0, K]` before any of it is used.

**What a `DraftOutput` answers for.** `DraftOutput` answers for the step that
just completed, and for the rows that step ran on. `DraftOutput.num_valid` is
per row and independent per row: row 3 returning 0 does not shorten row 4's
proposal, and row 4 returning 5 does not oblige row 3 to offer anything. The
token storage of a row whose `DraftOutput.num_valid` entry is 0 is ignored
completely, so an adapter with nothing to offer leaves that row's ids as
whatever its device graph produced.

This per-row count is the whole of the runtime half of the contract. An
adapter that wants a policy such as "draft while one request is live, decline
otherwise" implements it by returning `num_valid=[5]` after a step with one
row and `num_valid=[0, 0]` after a step with two rows. The plugin holds no
matching rule: there is no `should_speculate` callback, no batch-size
threshold and no model-class test anywhere in the selection of section 4d. The
two inputs to the plugin's choice are the static declarations of section 1a
and these counts.

Because `DraftOutput` describes the completed step, it does not bind the next
one. The scheduler can add or remove requests between the proposal and the
step that verifies it. Section 4e states what the plugin does with a proposal
that is already outstanding when that happens.

The call has one shape. A model that also declares `supports_narrow_decode`
still receives `[B, 1+K]` here after an ordinary decode step, with the columns
past each row's `accepted_counts` padded, exactly as a row that accepted less
than the full width looks after a verify. After such a step `hidden` is `None`,
which is why that path is closed to a drafter needing a fed hidden state: see
section 4d.

`hidden` is whatever this step's own `VerifyOutput.hidden` carried, handed back
without being interpreted. A model that needs none returns none and receives
none. Selecting this drafter requires vLLM's `custom_class` method, whose
`model` key must be exactly `vllm_tt_plugin.model_owned_drafter`: vLLM demands
a dotted proposer path for its custom-proposer extension. Upstream
`create_custom_proposer` imports and constructs that class. The TT runner
instead recognizes the fixed marker and calls the loaded model adapter; the TT
runner does not import `vllm_tt_plugin.model_owned_drafter` or load a separate
draft checkpoint from that value.

## 4c. The verify call under asynchronous scheduling

`supports_async_decode` and `supports_async_spec_decode` are separate model
promises. Neither capability enables scheduling by itself, and neither means
the drafter and target may execute concurrently for the same request.

- **`supports_async_decode` covers ordinary decode.** The model supports
  `decode_forward(..., read_from_device=False)` followed by
  `read_decode_output(..., async_read=True)`. When ordinary device sampling
  permits overlap, the model also preserves resident forward inputs, token
  feedback, positions, and reload behavior described in
  [DECODE_RELOAD_CONTRACT.md](DECODE_RELOAD_CONTRACT.md).
- **`supports_async_spec_decode` adds deferred verification support.** The same
  readback mechanism must return the verification block, whose accepted length
  the plugin decides later. The model must preserve verification output and
  any hidden state needed by the subsequent proposal until their consumers
  finish. Ordinary single-token decode support does not establish these
  multi-token and lifetime requirements.

For a launch requesting speculation and async scheduling, `TTPlatform` applies
these checks in this order:

1. `TTPlatform` checks `supports_async_decode`. If false or absent,
   `TTPlatform` logs a warning and sets `async_scheduling=False`. The launch
   can proceed synchronously if the remaining speculative checks pass, even
   if `supports_async_spec_decode=True`.
2. If async scheduling remains enabled, `TTPlatform` requires
   `supports_async_spec_decode=True`. A missing or false declaration raises
   `ValueError`; the operator can select `--no-async-scheduling`.
3. `TTPlatform` also requires `supports_spec_decode` and an admissible
   `SpecPlan`. Both async declarations are necessary for the combined path,
   but do not replace the other admission checks.

The TT compatibility patch admits `custom_class` through the upstream async
method check before platform validation. The patch preserves the marker and
capability checks. `--no-async-scheduling` still selects synchronous execution.

Asynchronous scheduling changes when the runner applies a step, not what it
sends. `execute_model` submits and returns nothing; the engine collects the
output later when the deferred result is resolved; and the runner applies the
result to request state at the top of the following step. Output resolution
may run on the caller thread, depending on the executor; asynchronous
scheduling does not require a separate output thread. For a speculative step
the accept walk runs where the readback completes, and the commit of the
accepted prefix plus the next proposal run on the engine thread when the next
step drains this one.

Ordering first, because it bounds everything below. A speculative step is
registered as not overlap-safe, so the runner drains it before it builds the
next step, and it applies the drained result before it builds. A verify is
therefore never submitted while the previous verify's acceptance is still
unapplied, and verify, accept and propose stay serialized per request. What
this path defers is the readback and the commit, not the order of the steps.

Two demands nevertheless reach the model, and they are what
`supports_async_spec_decode` declares:

1. **`read_decode_output` serves a verify.** On a synchronous speculative step
   the hook is never called: the runner asks for the output with the
   submission. On this path the runner calls
   `read_decode_output(tt_out, async_read=True)` with the tensor unwrapped from
   the `VerifyOutput`, so the hook is handed the mode's `[B, 1+K]` block rather
   than a decode's single column, and the number of tokens per row that will
   be committed out of it is decided by the host after the forward returns.
2. **The hidden handle outlives the step's submission.** The model returns it
   from `decode_forward`, the runner carries it across the readback and a
   queue, and `propose_draft_tokens` receives it at the next step's drain. A
   model declaring `spec_hidden_handoff: ["on_device"]` is promising that the
   device state behind the handle is still valid then.

Neither is covered by `supports_async_decode`, whose requirements in
[`DECODE_RELOAD_CONTRACT.md`](DECODE_RELOAD_CONTRACT.md) are written for a
decode that commits one token per forward: they speak of a persistent token
buffer holding the selected token and of one resident position advance per
forward. A model can satisfy every one of them for an ordinary decode and be
wrong for a deferred verify, which is why this is a separate declaration
rather than the conjunction of the two existing ones. Deriving it would
enlarge what a model that already declares `supports_async_decode` promised.

## 4d. Which steps verify, and which are ordinary decodes

`vllm_tt_plugin.model_runner._step_verifies` makes this choice once per decode
step, from three values the input builder hands it: the effective
`supports_narrow_decode`, the step's `num_valid_drafts` row vector, and the
step's `accepted_counts` row vector.
`vllm_tt_plugin.model_runner._step_verifies` returns true, and the step is a
verify, when any one of the following holds.

1. **Narrow decode is unavailable.** The `SpecPlan` did not set
   `supports_narrow_decode`, or `TTModelRunner.load_model` revoked it for the
   `hidden_feed` plus `roundtrip` drafter described below. Such a model
   implements one input shape, so every decode step of the launch is a verify,
   including a step whose `num_valid_drafts` is 0 on every row.
2. **Some row carries a draft.** `num_valid_drafts` is nonzero somewhere, so
   there is something to verify. One row is enough: a batch in which one row
   offers five drafts and every other row offers none is still sent as the
   `[B, 1+K]` candidate block, with the draftless rows padded.
3. **Some live row's previous step committed more than one token.**
   `accepted_counts` is how a model finds which candidate state slot that
   commit landed on, so the step after such a commit carries the count even
   when it drafts nothing. `vllm_tt_plugin.model_runner._step_verifies` reads
   only the live rows: a padding row sits at the post-prefill default of 1 and owns no
   request. One step resolves it, because that step commits a single token and
   records a count of 1, so leaving speculation costs exactly one verify.

When none of the three holds, the step is an ordinary decode.
`TTModelRunner._prepare_model_inputs` then discards the candidate block it
would have built and sends the plain `[B, 1]` tokens and 1-D `start_pos`, with
no `spec_mode`, neither side tensor, and the sampling path a non-speculating
launch uses.

**Overlap follows from that choice, and only an ordinary decode is eligible.**
`TTAsyncDecodeController.submit_async_decode` reads `TTModelInput.spec_mode`:
a step carrying one is registered through
`TTAsyncDecodeController.register_pending_async_step` with `overlap_ok=False`,
because the next candidate block is built from this step's committed tokens.
Nothing else marks a verify unsafe. `check_perform_device_sampling` does not
look at speculation, so on a launch that samples on device a verify otherwise
passes every condition of
`TTAsyncDecodeController.can_use_steady_decode_fast_path`. A step with no
`spec_mode` is registered with whatever verdict those ordinary checks
produced, so an ordinary decode inside a speculating launch is eligible for
overlap on exactly the terms an ordinary decode in a non-speculating launch
is. Those terms are not restated here. They are
`TTAsyncDecodeController.can_use_steady_decode_fast_path` and the resident
input, reload command and slot remap rules of
[DECODE_RELOAD_CONTRACT.md](DECODE_RELOAD_CONTRACT.md). Eligible is not the
same as overlapped: the controller still refuses overlap on a layout change,
a host-sampling step, a structured-output step and the rest of the conditions
that document lists.

Two obligations come with the declaration.

**The drafter is still asked, after an ordinary decode too.** The proposer is
called once per step, so a launch that skipped it on these steps would never
draft again whatever the batch did afterwards.
`TTModelRunner.propose_after_plain_step` is the call site. The committed block
is one column wide, that column being the token the step committed, every
`accepted_counts` is 1, and `TTModelRunner._propose_model_drafts` pads the
block to the uniform `1+K` before the call, so the drafter sees one shape. On
an asynchronous launch this runs at the next step's drain, because the drafts
continue a token that had to be read back first, so a drafter that starts
offering again is acted on one step later.

**The drafter must not need a fed hidden state.** An ordinary decode returns no
`VerifyOutput`, so it produces no hidden handle, and `propose_draft_tokens`
receives `None` after one. A model requiring `hidden_feed` and declaring
`spec_hidden_handoff: ["roundtrip"]` is therefore kept off this path entirely:
`TTModelRunner._narrow_steps_serve_the_drafter` logs the reason when
`TTModelRunner.load_model` runs and keeps every step a verify. A drafter that
keeps its state on device, or needs none, is unaffected.

## 4e. Batch changes while a proposal is outstanding

A proposal and the verify that consumes it are separated by a scheduler
decision, so the set of live requests can change in between. The plugin does
not read that change as cancelling the proposal. A model that admits more than
one concurrent request has to be written for the transitions below.

**A proposal belongs to a request, not to a row or to a batch size.**
`TTModelRunner._propose_model_drafts` records each row's offered ids under the
owning request id in `TTModelRunner._proposed_draft_token_ids`.
`TTModelRunner._req_accepted_counts` is keyed by request id for the same
reason: a running request that one step does not schedule leaves the
persistent batch and comes back, and only an explicit preemption releases the
candidate state its count selects. `TTModelRunner._drafts_to_verify` and
`TTModelRunner._spec_row_state` look both up by request id for whatever rows
the new step has.

So when request A holds five outstanding drafts and the scheduler adds request
B before the next decode, `TTModelRunner._spec_row_state` places A's five
drafts on A's row and zero on B's row,
`vllm_tt_plugin.model_runner._step_verifies` sees a nonzero `num_valid_drafts`
and selects a verify, and the adapter receives a `[B, 1+K]` block in which A's row carries
real drafts while B's row carries its own committed token in column 0 and
`PLACEHOLDER_TOKEN_ID` with position -1 in columns 1..K. B joining does not
turn A's existing drafts into an ordinary decode. The adapter must serve that
mixed block and must respect per-row `num_valid_drafts`.

**Declining the next proposal is always available.** In that same step the
adapter can return `DraftOutput.num_valid=[0, 0]`, offering nothing for either
row. The following step is then an ordinary decode, unless A's commit was
multi-token, in which case `vllm_tt_plugin.model_runner._step_verifies` sends
one more verify to resolve A's `accepted_counts` and the ordinary decode follows it.
The plugin always resolves a multi-token acceptance before it selects an
ordinary decode.

**Returning to drafting is a model obligation.** The plugin keeps asking:
`TTModelRunner.propose_after_plain_step` calls `propose_draft_tokens` after an
ordinary step too, so the adapter is offered the chance on every step for the
life of the server. Whether `DraftOutput.num_valid` can become positive again
for a request that survived a batched stretch depends on whether the adapter
still holds, or can rebuild, that request's drafter state. Nothing in the
plugin restores it, and nothing in the plugin notices that an adapter has
silently stopped drafting. An adapter that releases its speculative session
when it declines a batched proposal does not resume for the surviving request.
A fresh request prefilled after the batch shrinks is a different request and
exercises none of this.

**Rows move, and device state follows the request.**
`TTModelRunner._decode_state_slot_remap` builds the gather permutation
`TTModelInput.slot_remap` from `TTModelRunner._req_state_slot`, where
`slot_remap[i] = j` means decode row `i` reads the state currently in slot
`j`. The adapter applies `slot_remap` exactly once per submission, as
[DECODE_RELOAD_CONTRACT.md](DECODE_RELOAD_CONTRACT.md) specifies, and that
duty covers drafter state as much as sampler or recurrent state. After the
adapter accepts the decode, `TTModelRunner.note_decode_state_slots_settled`
commits `TTModelRunner._req_state_slot` and calls `_notify_model_slot_moves`
when slots moved. `_notify_model_slot_moves` calls the optional model callback
`note_state_slots_moved(old_to_new)` with the old-slot to new-slot mapping.
The adapter uses `note_state_slots_moved` to update session ownership metadata;
the callback does not replace the device gather performed for `slot_remap`.
A refused submission produces no ownership update or movement notification.

**The plugin notifies the adapter before releasing a request's slot.**
`TTModelRunner._update_states` calls `TTModelRunner._release_model_request`
for every id in `SchedulerOutput.finished_req_ids` and
`SchedulerOutput.preempted_req_ids`. If the request has a current slot and
the adapter implements `release_request`,
`TTModelRunner._release_model_request` calls `release_request(current_slot)`.
The adapter releases the model-owned state associated with that current slot.
`TTModelRunner._release_dead_state_slots` then removes those requests from
`TTModelRunner._req_state_slot`. A preempted request re-prefills its history
when the scheduler resumes it.
`TTModelRunner._prepare_model_inputs` drops a request's
`TTModelRunner._req_accepted_counts` entry once the request is gone from
`TTModelRunner.requests`, and drops it for every request of a prefill step, so
a resumed request restarts at the post-prefill count of 1 with no candidate
state selected. `TTModelRunner._apply_committed_spec_tokens_to_state` records
no accepted count for a request whose slot a preemption already released.

## 4f. KV reservation and the drafter's write extent

`TTScheduler.__init__` raises `self.num_lookahead_tokens` to at least
`spec_lookahead_tokens(get_tt_spec_plan(vllm_config), self.num_spec_tokens,
speculative_config.method)`, and `spec_lookahead_tokens` returns
`num_spec_tokens + 1` when a `SpecPlan` was admitted, `num_spec_tokens` is
positive, and `speculative_config.method` is `custom_class`. An
`ngram` launch also carries an admitted plan, but its drafts come from the
plugin and its target verifies them inside the step's own allocation, so
`spec_lookahead_tokens` returns zero for `ngram`. `TTScheduler.__init__` uses
`max` to preserve any upstream reservation while adding the model-owned
drafter's requirement. Lookahead slots reserve KV block space past the tokens
the step itself computes.

The reason is where a model-owned drafter writes. Such a drafter proposes for
the next step from inside the current one: after the accept walk commits, the
adapter's fused body runs over the committed anchor plus each of the `K`
drafts and writes K/V for those `K + 1` positions, none of which the current
step's own allocation covers. Upstream vLLM reserves lookahead only for the
drafter methods it knows, and `custom_class` is not one of them, so without
this reservation the rows that cross into the next block land in the null
block and the first token that reads them diverges from an ordinary decode.

`K + 1` is the demonstrated bound for the drafter implementation this was
measured against. It is not a proof about every adapter. An adapter whose
`propose_draft_tokens` writes K/V past the anchor plus `K` drafted positions,
for instance one running extra target layers or a deeper candidate tree, will
write outside the reservation. The plugin offers no way to declare that: there
is no capability key for a proposal write extent and `SpecPlan` has no field
for one. An adapter in that position raises it with the plugin owner before it
is admitted, because nothing detects the overrun at run time.

## 5. One request, from prefill to release

The sequence below follows a single request A on a launch that declares
`supports_spec_decode`, declares `device_propose` in `spec_requirements`,
returns a `SpecPlan` with `effective_k` 5 and `supports_narrow_decode=True`,
uses the `custom_class` method, and runs with asynchronous scheduling. Each
step names who acts.

1. **The scheduler** admits A and schedules a prefill.
   **`TTModelRunner._alloc_prefill_state_slots`** assigns A a device state slot
   and records it in `TTModelRunner._req_state_slot`.
   **`TTModelRunner._prepare_model_inputs`** removes A's
   `TTModelRunner._req_accepted_counts` entry, so A's first decode step reads
   the post-prefill default of 1. **The adapter** runs the prefill and commits
   one token for A.
2. First decode step. **`TTModelRunner._drafts_to_verify`** finds no draft for
   A, and **`TTModelRunner._spec_row_state`** produces `num_valid_drafts` 0 and
   `accepted_counts` 1 on A's row. **`vllm_tt_plugin.model_runner._step_verifies`** returns
   false, because `supports_narrow_decode` holds and neither of the other two
   conditions of section 4d does, so **`TTModelRunner._prepare_model_inputs`**
   sends the ordinary `[B, 1]` decode call. A model that had not declared
   `supports_narrow_decode` would receive the `[B, 1+K]` verify here instead,
   with `num_valid_drafts` 0 on every row.
3. Bootstrap. **The adapter** must initialize whatever drafter state it needs
   from whichever of those two calls it received. An adapter that initializes
   its drafter only inside the verify call never drafts on a launch whose first
   decode step is the ordinary one, and the plugin reports nothing, because
   declining to draft is a legal answer.
4. Proposal. **`TTAsyncDecodeController`** drains the ordinary step and calls
   **`TTModelRunner.propose_after_plain_step`**, which calls
   **`TTModelRunner._propose_model_drafts`**, which pads the one-column
   committed block to `1+K` and calls **the adapter's**
   `propose_draft_tokens` with `hidden=None`. **The adapter** returns a
   `DraftOutput` whose `num_valid` entry for A's row is 5.
   **`TTModelRunner._propose_model_drafts`** validates dtype, shape and range
   and stores A's five ids in `TTModelRunner._proposed_draft_token_ids`.
5. Verification. On the next decode step **`TTModelRunner._drafts_to_verify`**
   hands those five ids back inside the scheduler's lookahead reservation,
   **`vllm_tt_plugin.model_runner._step_verifies`** returns true, and
   **`TTModelRunner._spec_candidate_block`** widens A's row to `[1 + 5]`
   columns. **`TTAsyncDecodeController.submit_async_decode`** sees
   `TTModelInput.spec_mode` and registers the step with `overlap_ok=False`.
   **The adapter** returns a `VerifyOutput` carrying `argmax_ids` of shape
   `[B, 6]` and, if its drafter needs one, a hidden handle.
6. Acceptance. **`TTModelRunner.walk_spec_acceptance`** calls
   **`accept_greedy_drafts`**, which compares the adapter's choice at each
   candidate position against the draft at that position and stops at the
   first mismatch, producing A's committed block and its count.
7. Commit. **`TTModelRunner.commit_spec_acceptance`**, on the engine thread,
   calls **`TTModelRunner._apply_committed_spec_tokens_to_state`**, which
   appends A's accepted prefix to A's output tokens and writes
   `TTModelRunner._req_accepted_counts[A]` to the length of that prefix.
8. Next proposal. The same **`TTModelRunner.commit_spec_acceptance`** call
   then runs **`TTModelRunner._propose_model_drafts`** again, this time with
   the `[B, 1+K]` committed block, the real `accepted_counts`, and the hidden
   handle **the adapter** returned in step 5. **The adapter** either offers
   drafts again, returning to step 5, or returns
   `DraftOutput.num_valid` 0 for A's row. After a zero, the next step is a
   verify if A's count from step 7 exceeds 1, and an ordinary decode
   afterwards.
9. Release. When `SchedulerOutput.finished_req_ids` contains A,
   **`TTModelRunner._update_states`** calls
   **`TTModelRunner._release_model_request`**, which calls **the adapter's**
   optional `release_request(current_slot)` while A's current slot is known.
   **The adapter** releases A's model-owned drafter state.
   **`TTModelRunner._release_dead_state_slots`** then drops A from
   `TTModelRunner._req_state_slot`, and
   **`TTModelRunner._prepare_model_inputs`** drops A from
   `TTModelRunner._req_accepted_counts` once A has left
   `TTModelRunner.requests`. **`TTModelRunner._update_states`** also sends
   the release notification when `SchedulerOutput.preempted_req_ids`
   contains A.

## 6. What a verify returns, column by column

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

## 7. What the plugin does with a refusal

`resolve_speculative_plan` rejects unsupported launch settings during
configuration. `TTPlatform.validate_request` rejects unsupported sampling
controls during request admission. `submit_decode` validates `VerifyOutput`
at execution time and raises the corresponding type or mode error. These
checks report the offending values rather than silently replacing the requested
speculative behavior. The ordinary async capability check separately permits
synchronous fallback with a warning, as specified in section 4c.
