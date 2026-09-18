# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Measure the host overhead a speculative launch adds.

Two questions, and the second only makes sense once the first is answered.

**What does configuring speculation cost a batch that is not speculating?**
An adaptive drafter declines to draft while several requests are live, so the
batched steps of such a launch are ordinary decodes. The comparison is that
launch against one with no ``speculative_config`` at all, over the same
prompts, the same concurrency, the same output lengths, the same sampling mode
and the same logging. Whatever separates them is what the speculative
machinery costs when it is doing nothing: the wider input build, the
scheduler's lookahead reservation, the proposal call on every commit, and the
drain the runner performs before a verify.

**What does asynchronous scheduling cost or buy the contract path?** The same
launch synchronously and asynchronously, at zero, partial and full acceptance,
which is where the deferred completion, the drain and the deferred commit are
exercised.

What it measures, and how:

``committed tokens per second``
    ``vllm:generation_tokens_total`` differenced across the measured interval,
    over the interval's wall clock. The aggregate over the batch, not a
    per-request rate.

``request latency``
    Each request's own wall clock, client-side, reported as a distribution.
    Summing these would not give CPU time and is not used for one.

``CPU seconds per committed token``
    ``utime + stime`` of the server's whole process tree, read from ``/proc``
    before and after the interval, over the tokens committed in it. This is
    the only figure here that is a cost rather than a rate, and it is why the
    benchmark has to run on the machine under test.

``ordinary decode and verify submissions``
    The runner's own counters, from its log. Not the speculative metrics:
    those count the scheduler's lookahead reservation, which under
    asynchronous scheduling exists for every scheduled request whether or not
    a draft was ever verified.

What it deliberately does not do: sleep anywhere inside a measured interval,
read the submission and completion timestamps as a per-step cost (they omit
the state application and the proposal that follow them), or report the first
interval after a launch. Each configuration runs a discarded warmup interval
and then several measured ones, and the startup cost is reported on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
SUBMISSIONS = re.compile(r"TT submissions: (\d+) ordinary decode, (\d+) verify")
GENERATION_TOKENS = "vllm:generation_tokens_total"
DUMMY_VOCAB_SIZE = 128256


def ascending_prompt(length: int, start: int) -> list[int]:
    return [(start + index) % DUMMY_VOCAB_SIZE for index in range(length)]


# region The server under measurement


def process_tree(pid: int) -> list[int]:
    """Every live descendant of ``pid``, plus ``pid``.

    The work is not in the process the launcher runs: the engine core is a
    child, and the CPU time that matters is spent there.
    """
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The comm field is parenthesised and may contain spaces, so the
        # fields after it are found from the last close parenthesis.
        tail = stat[stat.rindex(")") + 2 :].split()
        parent = int(tail[1])
        children.setdefault(parent, []).append(int(entry.name))
    tree = [pid]
    frontier = [pid]
    while frontier:
        current = frontier.pop()
        for child in children.get(current, ()):
            tree.append(child)
            frontier.append(child)
    return tree


def cpu_seconds(pid: int) -> float:
    """User plus system CPU time of the whole tree, in seconds."""
    total = 0
    for process in process_tree(pid):
        try:
            stat = Path(f"/proc/{process}/stat").read_text()
        except OSError:
            continue
        tail = stat[stat.rindex(")") + 2 :].split()
        total += int(tail[11]) + int(tail[12])
    return total / CLOCK_TICKS


def generation_tokens(base_url: str) -> float:
    response = httpx.get(f"{base_url}/metrics", timeout=30.0)
    response.raise_for_status()
    for line in response.text.splitlines():
        if line.startswith(GENERATION_TOKENS + "{"):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"{GENERATION_TOKENS} is not in this server's metrics")


def submission_counts(log: Path) -> tuple[int, int]:
    """The runner's last reported ``(ordinary, verify)`` submission counts."""
    last = (0, 0)
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return last
    for line in text.splitlines():
        if found := SUBMISSIONS.search(line):
            last = (int(found.group(1)), int(found.group(2)))
    return last


def launch(command: list[str], env: dict[str, str], log: Path, cwd: str):
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env={**os.environ, **env},
        cwd=cwd,
    )
    return process, handle


def wait_until_healthy(base_url: str, process, timeout: float = 600.0) -> float:
    """Seconds from launch to a healthy server, which is the startup cost."""
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"the server exited with {process.returncode}")
        try:
            if httpx.get(f"{base_url}/health", timeout=5.0).status_code == 200:
                return time.perf_counter() - started
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError("the server never became healthy")


def stop(process, handle) -> None:
    """Stop the launcher and wait for the engine core it owns to go.

    Never a signal to the engine core itself: killing that process while it
    holds the device wedges the mesh.
    """
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=180)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=60)
    handle.close()
    for _ in range(90):
        if not engine_core_is_alive():
            return
        time.sleep(2.0)
    raise RuntimeError("an engine core is still holding the device")


def engine_core_is_alive() -> bool:
    """Whether any engine core is still running, by its own command line."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if "VLLM::EngineCore" in cmdline:
            return True
    return False


# endregion The server under measurement

# region One measured interval


def run_interval(
    base_url: str,
    model: str,
    log: Path,
    server_pid: int,
    *,
    concurrency: int,
    prompt_length: int,
    max_tokens: int,
) -> dict:
    """One batched interval: N requests together, measured end to end.

    Nothing sleeps in here. The requests are posted at once, so the interval is
    the batch's own wall clock, and the counters are read immediately either
    side of it.
    """
    prompts = [
        ascending_prompt(prompt_length, start=1000 * (index + 1))
        for index in range(concurrency)
    ]

    def send(prompt: list[int]) -> tuple[float, int, int]:
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        started = time.perf_counter()
        response = httpx.post(f"{base_url}/v1/completions", json=body, timeout=600.0)
        latency = time.perf_counter() - started
        response.raise_for_status()
        payload = response.json()
        completed = int(payload["usage"]["completion_tokens"])
        return latency, completed, response.status_code

    tokens_before = generation_tokens(base_url)
    cpu_before = cpu_seconds(server_pid)
    submissions_before = submission_counts(log)
    wall_started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(send, prompts))

    wall = time.perf_counter() - wall_started
    cpu_after = cpu_seconds(server_pid)
    tokens_after = generation_tokens(base_url)
    submissions_after = submission_counts(log)

    latencies = sorted(latency for latency, _, _ in outcomes)
    committed = tokens_after - tokens_before
    requested = sum(completed for _, completed, _ in outcomes)
    cpu = cpu_after - cpu_before
    return {
        "wall_seconds": wall,
        "committed_tokens": committed,
        "completion_tokens_reported": requested,
        "tokens_per_second": committed / wall if wall else 0.0,
        "cpu_seconds": cpu,
        "cpu_ms_per_token": (cpu / committed * 1000.0) if committed else None,
        "latency_seconds": {
            "min": latencies[0],
            "median": statistics.median(latencies),
            "max": latencies[-1],
            "mean": statistics.fmean(latencies),
            "all": latencies,
        },
        "ordinary_decode_submissions": submissions_after[0] - submissions_before[0],
        "verify_submissions": submissions_after[1] - submissions_before[1],
    }


# endregion One measured interval

# region Configurations


def configuration(name: str, *, plugin: str, model: str, port: int) -> dict:
    """The launches this benchmark compares, as command and environment.

    Every one of them shares the sampling mode, the context, the concurrency
    the caller passes, and the logging. What differs is named in the key: the
    presence of a ``speculative_config``, the execution mode, the draft policy
    and the accept depth.
    """
    common = [
        sys.executable,
        f"{plugin}/examples/server_example_tt.py",
        "--model",
        model,
        "--tokenizer",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--max_num_seqs",
        "8",
        "--max_model_len",
        "2048",
        "--port",
        str(port),
        "--additional-config",
        json.dumps(
            {
                "tt": {
                    "register_test_models": True,
                    "sample_on_device_mode": "decode_only",
                }
            }
        ),
    ]
    spec = json.dumps(
        {
            "method": "custom_class",
            "model": "vllm_tt_plugin.model_owned_drafter",
            "num_speculative_tokens": 5,
        }
    )
    base_env = {"TT_SPEC_TARGET": "fixed", "MESH_DEVICE": "T3K"}

    if name == "ordinary-async":
        return {
            "command": common,
            "env": base_env,
            "describe": "no speculative_config, asynchronous",
        }
    if name == "adaptive-async":
        return {
            "command": [*common, "--speculative-config", spec],
            "env": {
                **base_env,
                "TT_SPEC_DRAFT_POLICY": "solo",
                "TT_SPEC_ACCEPT_DEPTH": "-1",
            },
            "describe": "adaptive drafter that declines for a batch, asynchronous",
        }
    for depth, label in ((0, "accept-0"), (2, "accept-2"), (-1, "accept-all")):
        for mode in ("sync", "async"):
            if name != f"contract-{mode}-{label}":
                continue
            command = [*common, "--speculative-config", spec]
            if mode == "sync":
                command.append("--no-async-scheduling")
            return {
                "command": command,
                "env": {
                    **base_env,
                    "TT_SPEC_DRAFT_POLICY": "always",
                    "TT_SPEC_ACCEPT_DEPTH": str(depth),
                },
                "describe": f"contract {mode}, accept depth {label}",
            }
    raise SystemExit(f"unknown configuration {name!r}")


# endregion Configurations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--model", default="models/vllm_test_utils/spec_test")
    parser.add_argument("--tt-metal-home", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("configurations", nargs="+")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    base_url = f"http://localhost:{args.port}"
    results = {}

    for name in args.configurations:
        plan = configuration(name, plugin=args.plugin, model=args.model, port=args.port)
        log = artifacts / name / "server.log"
        print(f"=== {name}: {plan['describe']}", flush=True)
        print("    " + " ".join(shlex.quote(part) for part in plan["command"]))
        process, handle = launch(
            plan["command"], plan["env"], log, cwd=args.tt_metal_home
        )
        try:
            startup = wait_until_healthy(base_url, process)
            print(f"    startup: {startup:.1f}s", flush=True)
            # Discarded: the first interval pays for lazy imports, the first
            # trace capture and the first block allocation, none of which a
            # steady-state cost should carry.
            warmup = run_interval(
                base_url,
                args.model,
                log,
                process.pid,
                concurrency=args.concurrency,
                prompt_length=args.prompt_length,
                max_tokens=args.max_tokens,
            )
            intervals = [
                run_interval(
                    base_url,
                    args.model,
                    log,
                    process.pid,
                    concurrency=args.concurrency,
                    prompt_length=args.prompt_length,
                    max_tokens=args.max_tokens,
                )
                for _ in range(args.repeats)
            ]
        finally:
            stop(process, handle)

        rates = [interval["tokens_per_second"] for interval in intervals]
        costs = [
            interval["cpu_ms_per_token"]
            for interval in intervals
            if interval["cpu_ms_per_token"] is not None
        ]
        results[name] = {
            "describe": plan["describe"],
            "startup_seconds": startup,
            "warmup_discarded": warmup,
            "intervals": intervals,
            "median_tokens_per_second": statistics.median(rates),
            "median_cpu_ms_per_token": statistics.median(costs) if costs else None,
            "median_latency_seconds": statistics.median(
                [interval["latency_seconds"]["median"] for interval in intervals]
            ),
        }
        print(
            f"    tokens/s {statistics.median(rates):.1f}"
            f"  cpu ms/token {statistics.median(costs):.3f}"
            f"  verify submissions {intervals[-1]['verify_submissions']}"
            f"  ordinary {intervals[-1]['ordinary_decode_submissions']}",
            flush=True,
        )

    summary = {
        "settings": {
            "concurrency": args.concurrency,
            "prompt_length": args.prompt_length,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "model": args.model,
        },
        "results": results,
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary: {artifacts / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
