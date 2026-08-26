# TT Scheduling and Execution Flow

This note summarizes how TT execution works in the current TT vLLM integration, how scheduling and queueing behave today, and how that differs from upstream vLLM.

The emphasis here is on the concepts:

- sync vs async
- single-process non-DP vs single-process lane-DP vs standard multi-process DP
- what is coordinated within one process vs what remains independent per rank

Code pointers are intentionally minimal. The main entry points are
`src/vllm_tt_plugin/platform.py`, `src/vllm_tt_plugin/scheduler.py`,
`src/vllm_tt_plugin/lane_scheduler.py`, `src/vllm_tt_plugin/worker.py`,
`src/vllm_tt_plugin/model_runner.py`, `src/vllm_tt_plugin/async_decode.py`, and
`src/vllm_tt_plugin/utils/dp_discovery.py`.

## Short Version

The current TT path is more specialized than upstream vLLM:

- A TT step is treated as either all-prefill or all-decode.
- TT does not support mixed prefill+decode batches.
- Token-chunked prefill is supported, but only for models whose tt-metal class
  declares that its generator can resume a prefill (never for block-output
  models); a chunk continuation is prefill work and only ever runs in a prefill
  step.
- CPU-device work overlap is a decode optimization.
- Standard multi-process DP runs independent per-rank engines.
- Single-process lane-DP coordinates lanes within one process and executes one
  merged TT batch.

Upstream vLLM is more general:

- The scheduler is token-budget based, not phase-based.
- A single step can naturally include both "prefill-like" and "decode-like" progress.
- Chunked prefill is part of the normal scheduler model.
- Async queueing is a generic executor pipeline mechanism, not a TT-specific decode overlap mechanism.

## Mental Model

There are three important scheduler-side collections:

- `waiting`: requests not yet admitted into active execution
- `skipped_waiting`: pending prefill requests temporarily blocked, for example
  while their structured-output grammar is compiled
- `running`: requests already admitted and holding active scheduler state

TT treats `waiting` and `skipped_waiting` together when checking for pending
prefill work. A partly prefilled request sits in `running` with
`is_prefill_chunk` set; TT counts it as pending prefill work too, because only
a prefill step can advance it. What TT changes is not the existence of these
collections, but the rules for choosing what kind of work a step may contain
and how that scheduled work is executed afterward.

For TT, it is useful to think in terms of two batch types:

- prefill batch: admits ready work from the pending prefill queues and advances
  partial-prefill continuations from `running`
- decode batch: advances already-running work from `running`, with partial
  prefills hidden

That is a simplification compared with upstream vLLM, but it matches how the TT path is intentionally organized today.

TT still performs continuous batching in the broad sense: requests arrive in
`waiting`, may be held in `skipped_waiting`, are admitted while other requests
remain active, can be preempted back to `waiting`, and leave independently as
they finish. The restriction is within each device step: TT does not mix
prefill and decode work in the same batch.

## Current TT Execution Flow

### 1. Request admission

New requests enter the scheduler's `waiting` queue.

The TT scheduler prefers to admit waiting work first, because TT wants to form a clean prefill step when possible. If no prefill can be admitted, and there are already-running decode requests, the scheduler can fall back to a decode-only step so progress continues and KV pressure can relax.

### 2. Scheduler decision

The current TT scheduler enforces one important rule: no mixed prefill+decode
batch. A chunked prefill stays on the prefill side of that split, so its
continuations are scheduled by prefill steps only.

So each TT scheduling step picks one of:

- prefill-only
- decode-only
- empty step

This is the main conceptual difference from upstream.

### 3. Engine step selection

The scheduler runs in one of three topologies:

1. Single-process non-DP: one engine, one scheduler, and one TT worker.
2. Single-process lane-DP: one engine coordinates multiple lane schedulers and
   submits one merged TT batch.
3. Standard multi-process DP: each rank has an independent engine, scheduler,
   KV cache, TT worker, and TT submesh.

Each engine process can execute synchronously or use async decode overlap. The
choice depends on:

- whether the effective `async_scheduling` setting is enabled
- whether the selected model declares async decode support
- whether the current batch satisfies the steady-decode fast-path conditions

vLLM normally enables async scheduling when the configuration is compatible.
The TT platform turns it off when the selected model does not declare
`supports_async_decode`. Block-output models are stricter: the platform
refuses to start unless `--no-async-scheduling` is passed.

With standard DP there is no global prefill/decode decision. One rank can run
prefill while another runs decode.

### 4. Worker/model-runner execution

The TT worker forwards scheduled work into the TT model runner.

For non-DP, lane-DP, and standard-DP ranks, execution is split into forward and
sample phases:

1. `execute_model()` prepares and submits the TT forward.
2. `sample_tokens()` applies sample-time grammar state and produces tokens.

Prefill remains effectively synchronous. Decode may return a deferred output
whose host readback and sampling are finalized asynchronously.

Even when a call crosses an async-looking executor boundary, TT prefill still behaves like a synchronous step in practice. The meaningful overlap optimization is decode.

### 5. State update

When model output is available, the scheduler updates request state, moves requests between `waiting` and `running` as needed, and emits outputs back to the engine.

In async decode mode, there can be a controlled one-step lag between:

- submitting decode work to the device
- applying the completed result back to scheduler/host state

That lag is intentional and is what creates host/device overlap.

## How TT Scheduling Works Now

### Local scheduler policy

The TT scheduler behaves like this:

- if `waiting` or `skipped_waiting` contains pending prefill work, or any
  running request is a partial-prefill continuation, run the prefill scheduling
  pass so newly ready requests can be promoted and admitted and continuations
  can advance.
- otherwise, advance decode work
- if pending prefill work cannot be admitted, fall back to decode to free
  capacity. Partial prefills do not make this fallback viable: a decode step
  cannot advance them, so only genuine running decodes count.

The reason for this policy is simple: TT wants a homogeneous batch type per
step.

At configuration time the platform turns requested chunked prefill off for
every model that does not declare
`model_capabilities['supports_chunked_prefill']` — and for every block-output
model, which cannot resume a split prompt — and zeroes
`long_prefill_token_threshold` (the base scheduler applies that cap before it
consults `enable_chunked_prefill`, so leaving it set would still split a
prefill). When chunked prefill is off and `max_num_batched_tokens` is smaller
than `max_model_len`, it raises that token budget to `max_model_len` so a full
prompt can be admitted instead of leaving an unschedulable request in
`waiting`.

### Chunked prefill

For a model that declares support, the base scheduler may give a long prompt
only part of the token budget. The request then moves to `running` with
`is_prefill_chunk` set, and later prefill steps schedule the next chunk until
the prompt is fully computed.

Two things follow for the TT execution path:

- The value handed to the generator as `prompt_lens` is the *end position of
  the scheduled chunk* (`num_computed_tokens + num_scheduled_tokens`), not the
  sequence length. The generator processes `tokens[start_pos:prompt_lens]`.
- A chunk that does not finish the prompt is *intermediate*: its forward writes
  KV state but emits no token. Those rows are marked in
  `TTModelInput.intermediate_prefill_mask`, force the step onto host sampling
  (device sampling would advance device RNG state for them), get a generator
  clone so the request's RNG does not drift, and report `[]` in the
  `ModelRunnerOutput` instead of a sampled token.

### Block-output reservation

Output placeholders (see "Why TT uses an async-style scheduler even in
TT-specific flows" below) have a second user: a synchronous block-output model
reserves one complete output block through the same accounting, even though
there is no decode lookahead.

When `output_tokens_per_step > 1`, the base scheduler reserves its normal
sampled-token placeholder and `TTScheduler` reserves the remaining physical
block width. All placeholders are consumed when that block result is applied;
client-visible output is still trimmed at EOS, stop tokens, and `max_tokens`.
See [DiffusionGemma block serving](diffusion-gemma.md) for the current
256-token block contract.

### Why TT uses an async-style scheduler even in TT-specific flows

TT uses an async-capable scheduler base because decode overlap needs output placeholders: a request can be scheduled one step ahead before the previous step's output has been fully applied on the host.

This does not mean the whole TT path is fully asynchronous.

It means:

- the scheduler can safely "reserve" the next decode token
- the engine can submit the next decode step before the prior result is fully retired

That mechanism matters most for steady-state decode.

## Queueing Within One Engine Process

### What is queued

When async scheduling is enabled, an engine keeps a host-side queue of
in-flight scheduled steps. This applies to a non-DP engine, the one engine in
lane-DP, and each independent engine rank in standard DP. Conceptually, each
queue entry contains:

- the scheduled batch description
- a future or future-like handle for its output

The queue is used with a "fill before blocking" policy:

- if more work can be scheduled, submit it first
- only block on the oldest in-flight result when needed

### What async really means here

For TT, async primarily means:

- decode submission can run ahead of host-side result application
- device readback/finalization can complete later
- the host can spend that time scheduling the next decode step

It does not mean:

- arbitrary prefill/decode mixing
- unlimited queue depth of useful overlapped TT work
- every model step is non-blocking in the same way

In practice:

- prefill remains a synchronous-style step
- decode can use the steady async path when invariants hold
- at most two scheduled steps can be outstanding; this allows overlap but does
  not guarantee it for every batch

### Threading and waiting

There are three different mechanisms involved, and they serve different purposes.

#### 1. Engine batch queue

The engine keeps a small queue of in-flight scheduled steps. With async
scheduling enabled, the queue depth is `2`.

This queue defines how many scheduled steps can be outstanding. It is an engine/executor queue, not a TT-model-internal queue.

#### 2. Executor-side future resolution

When the engine submits `execute_model(..., non_block=True)`, it gets back a future or future-like object.

What happens next depends on the executor mode:

- uniprocess mode: if the TT runner returns an async output wrapper, a single background `ThreadPoolExecutor` thread calls `get_output()`
- multiprocess mode: the worker process itself calls `get_output()` before sending the response back to the engine

In uniprocess mode there is one background output thread. In multiprocess mode the waiting happens inside the worker process.

#### 3. TT decode completion signaling

Inside the TT runner, async decode completion is tracked with deferred output
objects, `threading.Event` objects, and lock-protected deques.

These structures do not execute the device decode. They ensure that deferred
host readback and finalization run exactly once, even if the executor output
thread and the engine thread reach the same result concurrently.

The important pieces are:

- `_pending_async_steps`: submitted decode steps that may still need finalization
- `_pending_async_overlap_ok`: whether each pending step can remain overlapped
- `_completed_decode_steps`: decode results that have completed readback/finalization but have not yet been applied back to runner state
- `_steady_decode_lock`: protects those deques

The completion event is set after one-time finalization completes, whether
finalization was triggered by `get_output()` or by an explicit runner drain. It
means "this decode result is ready to consume", not merely "submission
happened".

#### 3a. Basic TT async mechanism

Under the hood, TT async decode is built on asynchronous host readback, not on a separate device-side execution thread managed by vLLM.

The basic sequence is:

1. Submit decode work with `decode_forward(..., read_from_device=False)`.
2. Ask the model to start host readback with `read_decode_output(..., async_read=True)`.
3. Keep the returned read events with the submission record.
4. Later, during finalization, wait on those read events with `ttnn.event_synchronize(...)`.
5. Only after those events complete, convert the decode output into normal host tensors and sampling results.

So the low-level meaning of "non-blocking" here is:

- do not immediately read the decode output back to the host
- issue host read requests asynchronously
- defer the blocking wait until finalization time

This is why the higher-level future/event bookkeeping exists. It is tracking when those asynchronous readbacks have become safe to consume.

#### 4. Where the code actually waits

There are two important wait points:

- engine-level wait: when the batch queue cannot be filled further, the engine blocks on the oldest queued future
- runner-level drain: before leaving the steady decode fast path, TT finalizes
  all pending deferred decode outputs and applies the completed steps

The runner drain calls the idempotent `ensure_finalized()` operation. It does
not merely wait for another thread to set an event, because the engine may need
to drain before the executor has resolved the older future.

So the host does not continuously poll. It blocks or finalizes at explicit
boundaries:

- `future.result()` at the engine/executor boundary
- `ensure_finalized()` when the runner must drain pending async decode work

### When steady async decode is allowed

Async scheduling is first gated by model capability. If a model does not
declare `supports_async_decode`, the platform disables async scheduling. The
steady path also requires TT tracing, so `trace_mode="none"` disables overlap.

TT only keeps decode overlapped when the batch is "steady" enough. In plain
terms, overlap is allowed only when the batch shape and sampling path are
stable and sampling can run on device.

Overlap is disabled and pending async work is drained when correctness would otherwise become ambiguous, for example when there is:

- prompt activity or resumed prefill work
- layout change in the decode batch
- structured output bookkeeping
- penalties, allowed-token masks, bad-word filtering, or logits processors
- any other requirement for host-side sampling
- logprobs, including an explicit `logprobs=0`, or other features that force a
  more synchronous path

So the TT async path is best understood as a fast path for steady decode, not as a universal async execution model.

The effective `async_scheduling` setting controls this engine and device
overlap. It can be disabled explicitly with `--no-async-scheduling`.
`--async_engine` in the offline example controls use of vLLM's asynchronous
client API and is a separate option.

## Queueing in Single-Process Lane TT

Single-process lane-DP uses one vLLM engine process and one TT worker.
`TTLaneCoordinator` owns one independent `TTScheduler` per lane. Each lane has
its own `waiting` and `running` queues, admission decisions, KV cache manager,
and lane-local block ID space.

New requests are assigned to the least-loaded lane and remain bound to that
lane. Because the device executes all lanes together, the coordinator selects
one shared mode for each step:

- if any lane can admit prefill, all lanes run a prefill step
- otherwise, all lanes run a decode step
- a lane with no work for the selected mode contributes an empty part of the
  merged batch

If a forced prefill step admits zero tokens while any lane has running decode
requests, the coordinator retries the step in decode mode. This prevents KV
pressure from causing a no-progress loop.

The coordinator merges the per-lane `SchedulerOutput` objects, the worker
builds one merged TT input, and the runner splits the output back by lane in
process. No process-level collectives are involved.

Lane mode reuses the non-DP TT async queue depth, but each queued item is a
merged multi-lane step; decode overlap uses the same steady-state checks, and
if any lane breaks them pending async decode work is drained before the next
step.

For single-execute models, `--data_parallel_size N` is transparently folded
into `N` in-process lanes. This covers Galaxy generators selected with
`TT_LLAMA_TEXT_VER=llama3_70b_galaxy` or
`TT_QWEN3_TEXT_VER=qwen3_32b_galaxy`, and GPT-OSS selected by its model
architecture. The user-provided `--max_num_seqs M` remains the capacity per
lane, so the engine's global capacity becomes `N * M`.

Lane-DP does not currently support request-specific RoPE state, including mRoPE
vision models.

## Standard Multi-Process DP

Models that do not fold into lane-DP use upstream vLLM's standard
multi-process DP when started with `--data_parallel_size N`.

Each DP rank has:

- an independent engine core process
- its own `TTScheduler`, `waiting` queue, and `running` set
- its own KV cache
- its own TT worker and TT submesh

The ranks do not negotiate a shared batch mode. Each rank independently chooses
prefill or decode, executes a local TT batch, and updates only its own request
state. From the scheduling perspective, this is equivalent to running `N`
independent TT servers behind vLLM's DP request routing.

On a single host, the plugin discovers TT device groups at startup and assigns
one group to each rank through `TT_VISIBLE_DEVICES`. Request routing binds each
request to one rank; requests do not migrate between ranks.

For standard DP, `--max_num_seqs M` is the capacity of each rank. The plugin
does not multiply the rank-local model or KV-cache capacity by the number of
ranks.

Async decode is also rank-local. Each rank has its own queue and applies the
same steady-decode checks described above. One rank draining or switching to
prefill does not force the other ranks to do the same.

Standard DP does not currently support MoE models. Single-execute models that
need internal data-parallel lanes, such as GPT-OSS, are folded into lane-DP
instead.

## Non-DP vs Standard DP

| Topic | Non-DP TT | Standard multi-process DP |
| --- | --- | --- |
| Scheduler state | One local scheduler | One independent scheduler per rank |
| Batch type per step | Prefill-only or decode-only | Same, selected independently per rank |
| Queueing shape | One host-side in-flight queue | One host-side queue per rank |
| Prefill behavior | Synchronous-style | Synchronous-style per rank |
| Decode overlap | Yes, when steady | Yes, independently per rank |
| Shared batch-mode or model-data coordination | None | None |
| Execution payload | One local TT model input | One local TT model input per submesh |

## TT vs Upstream vLLM

### 1. Scheduling model

Upstream uses a generic token-budget scheduler.

The important property is that upstream scheduling is not organized around an explicit "prefill batch vs decode batch" split. A request simply has computed tokens and target tokens, and the scheduler assigns more token work subject to budgets and constraints.

That naturally supports:

- mixed progress across different requests in one step
- chunked prefill
- a more uniform scheduling model across backends

The current TT path is more constrained:

- it treats prefill and decode as separate batch modes
- it avoids mixed prefill+decode batches
- it allows chunked prefill only for models that declare support for it, and
  always on the prefill side of the split

### 2. Async queueing model

Upstream has a generic batch-queue path that lets the engine keep the executor fed before blocking on the oldest completed result.

That queueing model is backend-agnostic. It is mainly about executor pipelining.

TT inherits that idea, but changes the meaning of "async":

- in each TT engine process, useful overlap is limited to steady-state decode
- in lane-DP, each queued item is one merged multi-lane step
- in standard DP, every rank owns an independent queue

### 2a. Waiting model

Upstream generic async queueing is mostly expressed as futures around executor work.

The TT path adds more explicit decode-completion bookkeeping:

- a background output thread in uniprocess mode, or blocking output handling inside the worker process in multiprocess mode
- TT decode completion events
- a lock-protected queue of completed decode steps that are applied later

So compared with upstream, TT has more explicit "submit now, finalize later, apply later" logic around decode completion.

### 3. Sampling boundary

In upstream vLLM, execution and token sampling are more naturally separable.

In the TT path, more of the decode and sampling behavior is bundled into TT-specific execution handling because the device/host readback path and device-sampling path are part of the TT execution contract.

This is one reason TT needs extra execution helpers instead of using only the generic upstream engine path.

### 4. DP behavior

Standard TT DP uses upstream vLLM's independent per-rank engine model. TT adds
device discovery and submesh assignment, but it does not add per-step
cross-rank scheduling or execution collectives.

Single-process lane-DP is TT-specific because one engine coordinates several
device lanes. It adds:

- static request-to-lane assignment
- one prefill/decode mode shared by all in-process lanes
- merging lane scheduler outputs into one TT input
- splitting the merged result back into lane-local scheduler state
- conservative all-lane checks before allowing steady decode overlap

## Practical Takeaways

If you want the most accurate mental model for the current TT stack, use this:

- The scheduler is local, but TT execution rules are specialized.
- TT prefill and decode are treated as different batch modes.
- Async mainly means decode overlap, not fully async end-to-end execution.
- Each engine process uses a local queue of in-flight steps.
- Standard DP ranks schedule and execute independently.
- Lane-DP coordinates multiple schedulers only within one process.
- Upstream vLLM scheduling is more general and less phase-constrained than the
  TT path.

That is why TT keeps a specialized scheduler and model-runner path while still
using upstream's engine and standard DP structure where possible.
