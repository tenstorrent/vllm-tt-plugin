#!/usr/bin/env python3
"""Read a server log, keep the last "TT specials:" line per worker pid, sum the ranks, print a table.

usage: specials_parse.py LOG [LOG ...]

The line is cumulative since worker start, so the last line of a pid is its total.
Keys (see vllm_tt_plugin/spec_specials.py): reqs, steps, req_<feature>,
path_<path>, <feature>@<path>, why_<reason>.
A worker pid is taken from "(Worker pid=N)"; a line without one counts as pid 0.
A pid that restarted (same pid, counters fall) is not detected: use one run per log.
"""

from __future__ import annotations

import re
import sys
from collections import Counter

PREFIX = "TT specials:"
_PID = re.compile(r"\(\w+ pid=(\d+)\)")
_KV = re.compile(r"(\S+?)=(\d+)(?=\s|$)")


def parse_line(line: str) -> tuple[int, dict[str, int]] | None:
    """(pid, counts) of one log line, or None when it is no specials line."""
    at = line.find(PREFIX)
    if at < 0:
        return None
    match = _PID.search(line[:at])
    pid = int(match.group(1)) if match else 0
    return pid, {k: int(v) for k, v in _KV.findall(line[at + len(PREFIX) :])}


def last_per_pid(lines) -> dict[int, dict[str, int]]:
    last: dict[int, dict[str, int]] = {}
    for line in lines:
        parsed = parse_line(line)
        if parsed is not None:
            last[parsed[0]] = parsed[1]
    return last


def total(per_pid: dict[int, dict[str, int]]) -> Counter:
    out: Counter = Counter()
    for counts in per_pid.values():
        out.update(counts)
    return out


def table(per_pid: dict[int, dict[str, int]]) -> str:
    if not per_pid:
        return "no TT specials lines"
    sums = total(per_pid)
    reqs, steps = sums["reqs"], sums["steps"]
    rows = [f"workers={len(per_pid)} requests={reqs} steps={steps}"]
    rows.append(f"{'key':28} {'count':>10} {'share':>7}")

    def add(title: str, keys: list[str], base: int) -> None:
        rows.append(f"-- {title}")
        for key in sorted(keys, key=lambda k: -sums[k]):
            share = f"{100.0 * sums[key] / base:6.2f}%" if base else "      -"
            rows.append(f"{key:28} {sums[key]:>10} {share}")

    add(
        "requests with feature (share of requests)",
        [k for k in sums if k.startswith("req_")],
        reqs,
    )
    add(
        "steps by path (share of steps)",
        [k for k in sums if k.startswith("path_")],
        steps,
    )
    add(
        "steps carrying feature, by path (share of steps)",
        [k for k in sums if "@" in k],
        steps,
    )
    add(
        "logits steps by reason (share of steps)",
        [k for k in sums if k.startswith("why_")],
        steps,
    )
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    last: dict[int, dict[str, int]] = {}
    for path in argv:
        with open(path, errors="replace") as handle:
            last.update(last_per_pid(handle))
    print(table(last))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
