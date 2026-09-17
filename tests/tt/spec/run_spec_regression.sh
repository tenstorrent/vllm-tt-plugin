#!/usr/bin/env bash
# Run the speculative-decoding device suite over its configurations.
#
# One server per configuration, because the accept depth, the target and the
# capacity are all fixed when the engine starts. Each configuration writes its
# own manifest directory, and the run refuses to start while another engine is
# alive: a server left behind by an earlier run answers on its port and
# invalidates everything that follows it.
#
# Usage: run_spec_regression.sh <artifacts-dir> [config ...]
#   with no config names, runs: accept-all accept-2 accept-0 capacity lossless
#
# Needs a reserved Tenstorrent device, TT_METAL_HOME set, and the plugin's
# virtual environment active. The losslessness configuration needs two chips,
# because it runs a speculating server and an unspeculated one side by side;
# it takes them through TT_VISIBLE_DEVICES, overridable with SPEC_CHIP and
# REFERENCE_CHIP. README.md beside this file carries every recipe.
set -uo pipefail

ARTIFACTS="${1:?usage: run_spec_regression.sh <artifacts-dir> [config ...]}"
shift || true
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(accept-all accept-2 accept-0 adaptive async capacity lossless)
fi

: "${TT_METAL_HOME:?TT_METAL_HOME must point at the tt-metal checkout}"
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="models/vllm_test_utils/spec_test"
TOKENIZER="${SPEC_TOKENIZER:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${SPEC_PORT:-8100}"
REFERENCE_PORT="${SPEC_REFERENCE_PORT:-8101}"
K="${SPEC_K:-5}"
export MESH_DEVICE="${MESH_DEVICE:-(1, 1)}"

mkdir -p "$ARTIFACTS"

engine_running() {
    # The process name is truncated to 15 characters by the kernel, so this
    # has to match the full command line rather than the name.
    pgrep -f "VLLM::EngineCore" >/dev/null 2>&1
}

stop_server() {
    local launcher="$1" port="$2"
    kill "$launcher" 2>/dev/null
    for _ in $(seq 1 45); do
        engine_running || return 0
        sleep 2
    done
    echo "WARNING: an engine is still running after stopping $launcher" >&2
}

start_server() {
    # $1 label, $2 log, then the server arguments.
    local label="$1" log="$2"
    shift 2
    if engine_running; then
        echo "REFUSING $label: an engine is already running" >&2
        pgrep -af "server_example_tt|VLLM::EngineCore" | grep -v pgrep >&2
        return 1
    fi
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        echo "REFUSING $label: port $PORT is held" >&2
        return 1
    fi
    # ``exec`` so the recorded pid is the server itself: without it ``$!`` is
    # the subshell, and stopping that leaves the server holding the device.
    ( cd "$TT_METAL_HOME" && exec python "$PLUGIN_DIR/examples/server_example_tt.py" "$@" ) \
        >"$log" 2>&1 &
    LAUNCHER=$!
    for _ in $(seq 1 150); do
        sleep 2
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$LAUNCHER" 2>/dev/null || break
    done
    echo "SERVER FAILED for $label, see $log" >&2
    tail -30 "$log" >&2
    stop_server "$LAUNCHER" "$PORT"
    return 1
}

run_config() {
    local label="$1" depth="$2" declared_depth="$3" max_model_len="$4" max_num_seqs="$5"
    local blocks="$6" policy="${7:-always}" async="${8:-false}" target="${9:-depth}"
    shift 9
    local selection=("$@")
    local dir="$ARTIFACTS/$label"
    local log="$dir/server.log"
    mkdir -p "$dir"

    local spec='{"method":"custom_class","model":"vllm_tt_plugin.model_owned_drafter","num_speculative_tokens":'"$K"'}'
    # Device sampling is what any overlapped decode needs: the plugin's
    # steady-decode fast path refuses a host-sampled step whatever else is
    # true of it, because the token the next step reads has to be the one the
    # device wrote. The asynchronous configuration therefore asks for it, and
    # the others leave it alone so their measurements stay comparable.
    local tt_config='{"tt": {"register_test_models": true}}'
    if [ "$async" = "true" ]; then
        tt_config='{"tt": {"register_test_models": true, "sample_on_device_mode": "decode_only"}}'
    fi
    local args=(
        --model "$MODEL"
        --tokenizer "$TOKENIZER"
        --max_num_seqs "$max_num_seqs"
        --max_model_len "$max_model_len"
        --port "$PORT"
        --additional-config "$tt_config"
        --speculative-config "$spec"
    )
    # Asynchronous scheduling is upstream's default when nothing objects, and
    # for a speculating launch the plugin's own patch is what stops it from
    # objecting. So the asynchronous configuration passes no flag at all and
    # checks the server log for what the engine resolved; every other
    # configuration asks for synchronous explicitly.
    if [ "$async" != "true" ]; then
        args+=(--no-async-scheduling)
    fi
    # The KV budget, in tokens, which the model declares and the plugin turns
    # into the block pool. ``--num-gpu-blocks-override`` cannot be used for
    # this: the plugin writes that field itself from the model's declaration,
    # so an operator's value is replaced.

    echo "=== $label: accept_depth=$depth max_model_len=$max_model_len max_num_seqs=$max_num_seqs draft_policy=$policy"
    export TT_SPEC_ACCEPT_DEPTH="$depth"
    export TT_SPEC_DRAFT_POLICY="$policy"
    export TT_SPEC_TARGET="$target"
    if [ "$blocks" != "-" ]; then
        export TT_SPEC_MAX_TOKENS_ALL_USERS="$blocks"
    else
        unset TT_SPEC_MAX_TOKENS_ALL_USERS
    fi
    start_server "$label" "$log" "${args[@]}" || return 1

    local launch_args="TT_SPEC_ACCEPT_DEPTH=$depth TT_SPEC_DRAFT_POLICY=$policy TT_SPEC_TARGET=$target TT_SPEC_MAX_TOKENS_ALL_USERS=${TT_SPEC_MAX_TOKENS_ALL_USERS:-unset} MESH_DEVICE='$MESH_DEVICE' python examples/server_example_tt.py ${args[*]}"
    # Every option in ``--name=value`` form, not ``--name value``. These
    # options are registered in ``tests/tt/spec/conftest.py``, which pytest
    # loads after its first pass over argv, so on that pass an unknown
    # ``--tt-metal-home /path`` leaves the path standing as a positional
    # argument: pytest then collects that directory, walks it for conftest
    # files, and dies importing the tt-metal checkout's root conftest.
    #
    # The plugin also goes first on PYTHONPATH, because tt-metal has a
    # top-level ``tests`` package of its own and a tt-metal environment puts
    # tt-metal ahead.
    ( cd "$PLUGIN_DIR" && PYTHONPATH="$PLUGIN_DIR:${PYTHONPATH:-}" python -m pytest "${selection[@]}" -v \
        --tt-server-url="http://localhost:$PORT" \
        --tt-model-name="$MODEL" \
        --tt-max-num-seqs="$max_num_seqs" \
        --tt-spec-k="$K" \
        --tt-spec-accept-depth="$declared_depth" \
        --tt-spec-target="$target" \
        --tt-spec-drafter=model \
        --tt-spec-draft-policy="$policy" \
        --tt-spec-async-scheduling="$async" \
        --tt-spec-artifacts="$dir" \
        --tt-spec-server-log="$log" \
        --tt-spec-launch-args="$launch_args" \
        --tt-metal-home="$TT_METAL_HOME" ) 2>&1 | tee "$dir/pytest.log"
    local status=${PIPESTATUS[0]}
    stop_server "$LAUNCHER" "$PORT"
    echo "=== $label finished with status $status"
    return "$status"
}

run_lossless() {
    # Two servers, one speculating and one not, both with the fixed target,
    # because whether a launch speculates is fixed when the engine starts. They
    # take separate chips through TT_VISIBLE_DEVICES, so this configuration
    # needs a box with at least two.
    local dir="$ARTIFACTS/lossless"
    mkdir -p "$dir"
    # The capacity configuration exports a small KV budget, and these servers
    # run at the full context: inheriting it leaves the engine refusing to
    # start, because the pool cannot hold one request.
    unset TT_SPEC_MAX_TOKENS_ALL_USERS
    unset TT_SPEC_ACCEPT_DEPTH
    # And the draft policy, for the same reason: an earlier configuration
    # exported it, and these two servers must be launched from what this
    # function states rather than from what ran before them.
    unset TT_SPEC_DRAFT_POLICY
    unset TT_SPEC_TARGET
    if engine_running; then
        echo "REFUSING lossless: an engine is already running" >&2
        return 1
    fi
    local common=(
        --model "$MODEL" --tokenizer "$TOKENIZER"
        --max_num_seqs 8 --max_model_len 2048
        --additional-config '{"tt": {"register_test_models": true}}'
        --no-async-scheduling
    )
    local spec='{"method":"custom_class","model":"vllm_tt_plugin.model_owned_drafter","num_speculative_tokens":'"$K"'}'
    echo "=== lossless: fixed target, accept_depth=2, one speculating server and one not"
    ( cd "$TT_METAL_HOME" && TT_SPEC_TARGET=fixed TT_SPEC_ACCEPT_DEPTH=2 \
        TT_VISIBLE_DEVICES="${SPEC_CHIP:-0}" exec python \
        "$PLUGIN_DIR/examples/server_example_tt.py" "${common[@]}" --port "$PORT" \
        --speculative-config "$spec" ) >"$dir/spec.log" 2>&1 &
    local spec_pid=$!
    ( cd "$TT_METAL_HOME" && TT_SPEC_TARGET=fixed \
        TT_VISIBLE_DEVICES="${REFERENCE_CHIP:-1}" exec python \
        "$PLUGIN_DIR/examples/server_example_tt.py" "${common[@]}" \
        --port "$REFERENCE_PORT" ) >"$dir/plain.log" 2>&1 &
    local plain_pid=$!

    local healthy=1
    for port in "$PORT" "$REFERENCE_PORT"; do
        for _ in $(seq 1 150); do
            sleep 2
            curl -sf "http://localhost:$port/health" >/dev/null 2>&1 && break
        done
        curl -sf "http://localhost:$port/health" >/dev/null 2>&1 || healthy=0
    done
    if [ "$healthy" -eq 0 ]; then
        echo "SERVER FAILED for lossless, see $dir" >&2
        tail -20 "$dir"/spec.log "$dir"/plain.log >&2
        kill "$spec_pid" "$plain_pid" 2>/dev/null
        return 1
    fi

    ( cd "$PLUGIN_DIR" && PYTHONPATH="$PLUGIN_DIR:${PYTHONPATH:-}" python -m pytest \
        tests/tt/spec/test_lossless_device.py -v \
        --tt-server-url="http://localhost:$PORT" \
        --tt-reference-url="http://localhost:$REFERENCE_PORT" \
        --tt-model-name="$MODEL" \
        --tt-max-num-seqs=8 \
        --tt-spec-k="$K" \
        --tt-spec-accept-depth=2 \
        --tt-spec-target=fixed \
        --tt-spec-drafter=model \
        --tt-spec-artifacts="$dir" \
        --tt-spec-server-log="$dir/spec.log" \
        --tt-spec-launch-args="TT_SPEC_TARGET=fixed TT_SPEC_ACCEPT_DEPTH=2 python examples/server_example_tt.py ${common[*]} --speculative-config $spec" \
        --tt-metal-home="$TT_METAL_HOME" ) 2>&1 | tee "$dir/pytest.log"
    local status=${PIPESTATUS[0]}
    kill "$spec_pid" "$plain_pid" 2>/dev/null
    for _ in $(seq 1 45); do engine_running || break; sleep 2; done
    echo "=== lossless finished with status $status"
    return "$status"
}

BEHAVIOUR=(tests/tt/spec/test_acceptance_metrics.py tests/tt/spec/test_concurrency.py tests/tt/spec/test_termination.py)
OVERALL=0
for config in "${CONFIGS[@]}"; do
    case "$config" in
        accept-all) run_config accept-all -1 all 2048 8 - always false depth "${BEHAVIOUR[@]}" ;;
        accept-2)   run_config accept-2 2 2 2048 8 - always false depth "${BEHAVIOUR[@]}" ;;
        accept-0)   run_config accept-0 0 0 2048 8 - always false depth "${BEHAVIOUR[@]}" ;;
        # The adaptive drafter: it offers the full draft length while one
        # request is live and nothing while more are, so the batched steps of
        # this configuration are ordinary decodes rather than verifies. Its
        # tests assert the ratio between the two, which no other configuration
        # can produce.
        adaptive)   run_config adaptive -1 all 2048 8 - solo false depth tests/tt/spec/test_adaptive_policy.py ;;
        # The launch that was unreachable until the plugin admitted the
        # model-owned drafter to asynchronous scheduling: no
        # --no-async-scheduling, the adaptive drafter so the batched steps are
        # ordinary decodes that can overlap, and the fixed target so the output
        # can be checked against the rule rather than against another server.
        async)      run_config async -1 all 2048 8 - solo true fixed tests/tt/spec/test_async_transitions.py tests/tt/spec/test_adaptive_policy.py ;;
        # A KV budget of 1024 tokens against eight requests that each want
        # 96 of prompt and 192 of output: all eight are admitted on their
        # prompts and then grow past the pool, so the scheduler has to preempt.
        # Sizing by max_model_len alone does not, because this model allocates
        # no real cache and its declared budget defaults to 131072 tokens.
        capacity)   run_config capacity -1 all 512 8 1024 always false depth tests/tt/spec/test_capacity.py ;;
        lossless)   run_lossless ;;
        *) echo "unknown configuration: $config" >&2; OVERALL=1; continue ;;
    esac
    status=$?
    [ "$status" -eq 0 ] || OVERALL="$status"
done

echo "=== manifests under $ARTIFACTS"
exit "$OVERALL"
