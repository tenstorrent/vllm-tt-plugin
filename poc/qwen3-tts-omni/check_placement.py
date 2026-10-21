# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which Tenstorrent cards the Omni orchestrator and each stage process hold.

A process holds a card while it has ``/dev/tenstorrent/<n>`` open; ``<n>`` is the
physical card, as UMD's "Opening local chip ids/PCIe ids" line names it. The
check reads those handles from ``/proc`` at three points: before the stages
start, while they serve one request, and after ``Omni.close()``.

    python poc/qwen3-tts-omni/check_placement.py \
        --model <Qwen3-TTS-12Hz-1.7B-Base snapshot> \
        --deploy-config poc/qwen3-tts-omni/deploy/tt_qwen3_tts.yaml \
        --ref-audio <clip.wav>

With ``--expect-startup-failure`` the stages are expected not to start; the check
then requires that no process is left holding a card.
"""

import argparse
import os
import sys
import time

from smoke_talker import prompt_len, tts_params
from vllm_omni.entrypoints.omni import Omni

_DEVICE_DIR = "/dev/tenstorrent/"


def _cards(pid):
    """The cards ``pid`` has open, by physical number."""
    cards = set()
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return cards
    for fd in fds:
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if target.startswith(_DEVICE_DIR):
            cards.add(int(target[len(_DEVICE_DIR) :]))
    return cards


def _descendants(root):
    children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as stat:
                # The name field may hold spaces; the parent pid follows its ")".
                parent = int(stat.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(parent, []).append(int(entry))
    found, stack = [], [root]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


def _name(pid):
    try:
        with open(f"/proc/{pid}/comm") as comm:
            return comm.read().strip()
    except OSError:
        return "?"


def report(label):
    """Print every process holding a card; return {pid: cards}."""
    holders = {}
    for pid in [os.getpid(), *_descendants(os.getpid())]:
        cards = _cards(pid)
        if cards:
            holders[pid] = cards
            print(
                f"PLACEMENT {label} pid={pid} name={_name(pid)} "
                f"self={pid == os.getpid()} cards={sorted(cards)}",
                flush=True,
            )
    if not holders:
        print(f"PLACEMENT {label} no process holds a card", flush=True)
    return holders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--text", default="Good morning. Today is a beautiful day.")
    parser.add_argument("--language", default="English")
    parser.add_argument("--expect-startup-failure", action="store_true")
    args = parser.parse_args()

    info = tts_params(args)
    length = prompt_len(args, info)
    failures = []

    if report("before_start"):
        failures.append("a card is held before the stages start")

    try:
        omni = Omni(
            model=args.model, deploy_config=args.deploy_config, trust_remote_code=True
        )
    except Exception as error:
        print(f"PLACEMENT startup_failed {type(error).__name__}: {error}", flush=True)
        if not args.expect_startup_failure:
            raise
        # Give exiting stage processes a moment to release their cards.
        time.sleep(5)
        if report("after_failed_start"):
            failures.append("a card is still held after the failed start")
    else:
        if args.expect_startup_failure:
            failures.append("the stages started")
        serving = report("serving")
        if os.getpid() in serving:
            failures.append("the orchestrator holds a card")
        if len(serving) != 2 or any(len(cards) != 1 for cards in serving.values()):
            failures.append("expected two stage processes holding one card each")
        omni.generate(
            [{"prompt_token_ids": [1] * length, "additional_information": info}]
        )
        omni.close()
        time.sleep(5)
        if report("after_close"):
            failures.append("a card is still held after close")

    for failure in failures:
        print(f"PLACEMENT FAIL {failure}", flush=True)
    print(f"PLACEMENT {'FAIL' if failures else 'OK'}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
