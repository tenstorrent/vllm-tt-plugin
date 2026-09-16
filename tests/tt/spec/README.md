# Speculative decoding on a device: the regression suite

These tests drive a running vLLM server that serves tt-metal's
`DummySpecDecodeModel`, which implements the speculative-decoding contract in host arithmetic and does no device work. They exist because the host suite deliberately stops at the runner: the engine core's draft handshake, the real scheduler's budget and preemption, the persistent batch's rows, the worker, and HTTP are all on the far side of it, and a device is needed to reach them because `TTWorker.init_device` opens a mesh whatever the model does.

A reserved Tenstorrent device is therefore required, and one Wormhole chip is enough. `tests/tt` is excluded from the plugin's continuous integration, so this is a manual job; registering it as a device job is a separate step.

## What each test establishes

| File | Establishes |
| --- | --- |
| `test_acceptance_metrics.py` | drafting happened at all, and acceptance matched the configured depth position by position |
| `test_lossless_device.py` | the speculated output equals the unspeculated output, token for token, through the real stack |
| `test_concurrency.py` | several rows speculate in the same engine steps, with the batch-size claim taken from the scheduler's histogram |
| `test_termination.py` | `max_tokens`, a stop token and the end-of-sequence id each end a request inside a multi-token committed prefix, identically streaming and not, and a cancelled request frees its row |
| `test_capacity.py` | preemption and replay occurred, and every preempted request still emitted its whole output |

No test treats a completed response as proof of anything. Each one reads vLLM's own counters across the work, and the capacity tests skip rather than pass when `vllm:num_preemptions_total` did not move.

## How the tests know what the server is

A speculative server's behaviour depends on how it was launched in ways no endpoint reports. The launch is therefore declared on the command line, and a test skips unless the declared configuration is the one it needs:

| Option | Meaning |
| --- | --- |
| `--tt-spec-k` | the launched `num_speculative_tokens`; 0 skips the whole suite |
| `--tt-spec-accept-depth` | the launched `TT_SPEC_ACCEPT_DEPTH`, or `all` |
| `--tt-spec-target` | the launched `TT_SPEC_TARGET`: `depth` or `fixed` |
| `--tt-spec-drafter` | `model` for the model's own drafter, `ngram` for the host one |
| `--tt-reference-url` | a second, unspeculated server, for the losslessness comparison |
| `--tt-spec-artifacts` | where the run manifest is written |
| `--tt-spec-server-log` | a server log to copy into the manifest directory |
| `--tt-spec-launch-args` | the exact server command, recorded in the manifest |
| `--tt-metal-home` | the tt-metal checkout whose commit and dirty diff go in the manifest |

Declaring the configuration wrongly makes tests fail rather than pass, because each asserts counter deltas that hold only for its own configuration.

## The two dummy targets, and why both exist

`TT_SPEC_ACCEPT_DEPTH` makes acceptance a knob by having the verify return each draft unchanged up to that depth. That is what every acceptance-accounting assertion wants, and it is why `test_acceptance_metrics.py` can predict the per-position counters exactly. It cannot answer whether speculation is lossless: the output of that target is a function of what was drafted, so a speculated run and a plain run are meant to differ.

`TT_SPEC_TARGET=fixed` chooses the next token by a rule the drafts never enter, `(token * 31 + position * 7 + 11) % vocab`, computed identically for an ordinary decode and for a verify. Its output is a property of the rule rather than of the drafting, so the same prompt served by a speculating server and by an unspeculated one has to produce the same tokens. In that mode the drafter walks the rule forward and bends the drafts past the accept depth, so wrongness lives in the drafter, which is where a real drafter's wrongness lives.

## Launching

Common to every configuration:

```bash
source /path/to/env.sh                  # TT_METAL_HOME, the venv, HF cache
export MESH_DEVICE="(1, 1)"             # this model does no device work
PLUGIN=/path/to/vllm-tt-plugin
SPEC='{"method":"custom_class","model":"vllm_tt_plugin.model_owned_drafter","num_speculative_tokens":5}'
COMMON="--model models/vllm_test_utils/spec_test
        --tokenizer meta-llama/Llama-3.1-8B-Instruct
        --additional-config {\"tt\":{\"register_test_models\":true}}
        --no-async-scheduling"
```

Two things about invoking pytest here, both of which fail loudly rather than subtly.

Pass every option in `--name=value` form. These options are registered in `tests/tt/spec/conftest.py`, which pytest loads only after its first pass over the command line, so on that pass `--tt-metal-home /path/to/tt-metal` leaves the path standing as a positional argument: pytest collects that directory, walks it for `conftest.py` files, and dies importing the tt-metal checkout's root conftest before any test runs.

Put the plugin checkout first on `PYTHONPATH`, because tt-metal has a top-level `tests` package of its own and a tt-metal environment puts tt-metal ahead:

```bash
export PYTHONPATH="$PLUGIN:$PYTHONPATH"
```

`--no-async-scheduling` is not optional: the plugin refuses a launch that combines speculation with asynchronous scheduling rather than taking a path that would skip acceptance. Requests must be greedy, which every test here sends.

**1. Acceptance accounting, full acceptance.**

```bash
TT_SPEC_ACCEPT_DEPTH=-1 python $PLUGIN/examples/server_example_tt.py $COMMON \
    --max_num_seqs 8 --max_model_len 2048 --port 8100 --speculative-config "$SPEC"

pytest tests/tt/spec/test_acceptance_metrics.py tests/tt/spec/test_concurrency.py \
    tests/tt/spec/test_termination.py \
    --tt-server-url=http://localhost:8100 --tt-model-name=models/vllm_test_utils/spec_test \
    --tt-max-num-seqs=8 --tt-spec-k=5 --tt-spec-accept-depth=all \
    --tt-spec-artifacts=/tmp/spec-run/accept-all
```

**2. Acceptance accounting, partial and none.** The same, with `TT_SPEC_ACCEPT_DEPTH=2` and `--tt-spec-accept-depth=2`, then `TT_SPEC_ACCEPT_DEPTH=0` and `--tt-spec-accept-depth=0`.

**3. Losslessness.** Two servers, one speculating and one not, both with the fixed target:

```bash
TT_SPEC_TARGET=fixed TT_SPEC_ACCEPT_DEPTH=2 python $PLUGIN/examples/server_example_tt.py \
    $COMMON --max_num_seqs 8 --max_model_len 2048 --port 8100 --speculative-config "$SPEC"
TT_SPEC_TARGET=fixed python $PLUGIN/examples/server_example_tt.py \
    $COMMON --max_num_seqs 8 --max_model_len 2048 --port 8101     # no --speculative-config

pytest tests/tt/spec/test_lossless_device.py \
    --tt-server-url=http://localhost:8100 --tt-reference-url=http://localhost:8101 \
    --tt-model-name=models/vllm_test_utils/spec_test --tt-max-num-seqs=8 \
    --tt-spec-k=5 --tt-spec-accept-depth=2 --tt-spec-target=fixed \
    --tt-spec-artifacts=/tmp/spec-run/lossless
```

The two servers need two chips: give each one its own through `TT_VISIBLE_DEVICES`, which is what the driver does (`SPEC_CHIP` and `REFERENCE_CHIP`, defaulting to 0 and 1).

**4. Constrained capacity, for preemption.** A small context and long requests, so the blocks do not all fit:

```bash
TT_SPEC_ACCEPT_DEPTH=-1 python $PLUGIN/examples/server_example_tt.py $COMMON \
    --max_num_seqs 8 --max_model_len 512 --port 8100 --speculative-config "$SPEC"

pytest tests/tt/spec/test_capacity.py \
    --tt-server-url=http://localhost:8100 --tt-model-name=models/vllm_test_utils/spec_test \
    --tt-max-num-seqs=8 --tt-spec-k=5 --tt-spec-accept-depth=all \
    --tt-spec-artifacts=/tmp/spec-run/capacity
```

If the capacity tests skip, the configuration did not reach preemption: lower `--max_model_len` further, or raise the requested `max_tokens`. They are written to skip rather than pass, so a skip is a real result and not a silent one.

**5. The host n-gram drafter.** Replace the speculative config with
`{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":2,"prompt_lookup_max":4}`
and pass `--tt-spec-drafter=ngram`. The n-gram drafter only drafts where the request's own text repeats, and this model's output is an ascending run that repeats no n-gram, so the prompt has to be that same run; the suite's prompts already are.

## The driver

`run_spec_regression.sh` runs every configuration in order: it launches each server, waits for it, runs the matching selection, stops the server, and writes one manifest directory per configuration. It refuses to start while another engine is alive, because a server left behind by an earlier run answers on its port and invalidates everything that follows.

```bash
tests/tt/spec/run_spec_regression.sh /tmp/spec-run                  # every configuration
tests/tt/spec/run_spec_regression.sh /tmp/spec-run capacity         # or one of them
```

The configurations are `accept-all`, `accept-2`, `accept-0`, `capacity` and `lossless`. The last one needs two chips and is the only one that launches two servers.

## The manifest

Each configuration's artifacts directory holds:

- `manifest.json`: the declared configuration, the exact launch command, the plugin and tt-metal commits with their branch, `git describe`, dirty flag, full `git diff` and untracked files, the vLLM version, the Python version, the relevant environment variables, and one record per test with the request bodies, the raw responses and the counter deltas.
- `metrics-final.txt`: the last `/metrics` scrape of the session.
- `server.log`: the server's own output, when `--tt-spec-server-log` was passed.

A device result without that is not reproducible, because these runs are taken against working trees with changes applied by hand, and the diff is part of the identity of the run.
