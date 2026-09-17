# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixtures for the speculative-decoding device suite.

These tests run against a server that is already running, which is the
convention the rest of ``tests/tt`` follows. A speculative server's behaviour
depends on how it was launched in ways no endpoint reports: the draft length,
which drafter is proposing, and which target the dummy model stands in for. So
the launch is declared on the command line and every test skips unless the
declared configuration is the one it needs. ``README.md`` beside this file
carries a recipe per configuration, and ``run_spec_regression.sh`` runs all of
them in order.

A configuration the tests cannot read is a configuration they must be told
about. Declaring it wrongly makes tests fail rather than pass, because each one
asserts counter deltas that only hold for its own configuration.

Every test records what it sent, what came back, and the counter movement into
a run manifest, written at the end of the session together with the commits,
the dirty diffs, the vLLM version and the launch arguments. A device result
without that is not reproducible.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tests.tt.spec.spec_client import Metrics, SpecServer

# The dummy model's vocabulary, from its own config.json, and the token id HF's
# LlamaConfig defaults to for end of sequence. The model counts straight
# through that id, which is what makes it reachable inside a committed prefix.
DUMMY_VOCAB_SIZE = 128256
DUMMY_EOS_TOKEN_ID = 2


def pytest_addoption(parser):
    group = parser.getgroup("tt-spec")
    group.addoption(
        "--tt-spec-k",
        type=int,
        default=0,
        help=(
            "num_speculative_tokens the server was launched with. 0 means the "
            "server is not speculating, and every test here skips."
        ),
    )
    group.addoption(
        "--tt-spec-accept-depth",
        default="all",
        help=(
            "TT_SPEC_ACCEPT_DEPTH the server was launched with: an integer, or "
            "'all' for the default that accepts every draft."
        ),
    )
    group.addoption(
        "--tt-spec-target",
        default="depth",
        choices=("depth", "fixed"),
        help=(
            "TT_SPEC_TARGET the server was launched with. 'depth' returns each "
            "draft unchanged up to the accept depth, so acceptance is a knob. "
            "'fixed' chooses by a rule the drafts never enter, which is what a "
            "losslessness comparison needs."
        ),
    )
    group.addoption(
        "--tt-spec-drafter",
        default="model",
        choices=("model", "ngram"),
        help="Which drafter proposes: the model's own, or the host n-gram one.",
    )
    group.addoption(
        "--tt-spec-artifacts",
        default=None,
        help="Directory for the run manifest. Skipped when unset.",
    )
    group.addoption(
        "--tt-spec-server-log",
        default=None,
        help="Server log to copy into the manifest directory.",
    )
    group.addoption(
        "--tt-spec-launch-args",
        default="",
        help="The exact server command, recorded in the manifest.",
    )
    group.addoption(
        "--tt-reference-url",
        default=None,
        help=(
            "A second server, launched from the same model with no "
            "--speculative-config, for the losslessness comparison. Whether a "
            "launch speculates is fixed when the engine starts, so the two "
            "arms cannot be one server."
        ),
    )
    group.addoption(
        "--tt-spec-draft-policy",
        default="always",
        help=(
            "The launched TT_SPEC_DRAFT_POLICY: 'always' drafts on every step, "
            "'solo' drafts only while one request is live. The adaptive tests "
            "skip unless this is 'solo', because a launch that always drafts "
            "cannot produce what they assert."
        ),
    )
    group.addoption(
        "--tt-metal-home",
        default=None,
        help="tt-metal checkout whose commit and dirty diff go in the manifest.",
    )


@dataclass(frozen=True)
class SpecConfig:
    """How the server under test was launched, as declared."""

    k: int
    accept_depth: int | None
    target: str
    drafter: str
    draft_policy: str = "always"

    @property
    def speculating(self) -> bool:
        return self.k > 0

    @property
    def accepts_every_draft(self) -> bool:
        return self.accept_depth is None or self.accept_depth >= self.k

    @property
    def accepts_nothing(self) -> bool:
        return self.accept_depth == 0

    @property
    def committed_per_step(self) -> int:
        """Tokens one row commits on a step whose drafts are all in flight.

        The accepted drafts plus the one token that always commits: either the
        bonus after a full acceptance, or the target's own choice at the
        position that rejected.
        """
        depth = self.k if self.accept_depth is None else min(self.accept_depth, self.k)
        return depth + 1

    def describe(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "accept_depth": "all" if self.accept_depth is None else self.accept_depth,
            "target": self.target,
            "drafter": self.drafter,
            "draft_policy": self.draft_policy,
        }


@pytest.fixture(scope="session")
def spec_config(request) -> SpecConfig:
    raw_depth = str(request.config.getoption("--tt-spec-accept-depth")).strip()
    depth = None if raw_depth in ("all", "-1", "") else int(raw_depth)
    return SpecConfig(
        k=int(request.config.getoption("--tt-spec-k")),
        accept_depth=depth,
        target=str(request.config.getoption("--tt-spec-target")),
        drafter=str(request.config.getoption("--tt-spec-drafter")),
        draft_policy=str(request.config.getoption("--tt-spec-draft-policy")),
    )


@pytest.fixture(scope="session")
def spec_server(tt_server_url, tt_model_name, spec_config) -> SpecServer:
    if not spec_config.speculating:
        pytest.skip(
            "the server is not speculating: pass --tt-spec-k with the launched "
            "num_speculative_tokens"
        )
    return SpecServer(tt_server_url, tt_model_name)


@pytest.fixture(scope="session")
def ascending_prompt():
    """The prompt the n-gram drafter needs, as a factory.

    The dummy's own output is an ascending run, which repeats no n-gram, so the
    host proposer drafts nothing unless the prompt is the same run. The model's
    own drafter needs no such prompt, and the fixed target needs none either,
    but one prompt shape for every configuration keeps the manifests
    comparable.
    """

    def build(length: int = 400, start: int = 0) -> list[int]:
        return [(start + index) % DUMMY_VOCAB_SIZE for index in range(length)]

    return build


@dataclass
class RunManifest:
    """What the session did, in enough detail to repeat it."""

    directory: Path
    launch_args: str
    config: dict[str, Any]
    environment: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, **payload: Any) -> None:
        self.records.append(
            {
                "test": name,
                "at": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
        )

    def write(self, metrics: Metrics | None) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        if metrics is not None:
            (self.directory / "metrics-final.txt").write_text(metrics.raw)
        manifest = {
            "written_at": datetime.now(timezone.utc).isoformat(),
            "declared_config": self.config,
            "launch_args": self.launch_args,
            "environment": self.environment,
            "records": self.records,
        }
        path = self.directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=False))
        return path


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def _repo_state(repo: Path | None) -> dict[str, Any]:
    """The commit and whatever is not committed, which is what a rerun needs.

    A device run is almost always taken against a working tree with something
    applied by hand, so the diff is part of the identity of the run and not an
    afterthought.
    """
    if repo is None or not repo.exists():
        return {"path": str(repo), "available": False}
    diff = _git(repo, "diff")
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard")
    return {
        "path": str(repo),
        "available": True,
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git(repo, "describe", "--always", "--dirty"),
        "dirty": bool(diff) or bool(untracked),
        "diff": diff,
        "untracked": untracked.splitlines(),
    }


@pytest.fixture(scope="session")
def run_manifest(request, spec_config, tt_server_url, tt_model_name):
    """Collects the session's evidence and writes it once, at the end."""
    directory = request.config.getoption("--tt-spec-artifacts")
    if not directory:
        pytest.skip("no --tt-spec-artifacts directory, so nothing can be recorded")
    manifest = RunManifest(
        directory=Path(directory),
        launch_args=str(request.config.getoption("--tt-spec-launch-args")),
        config=spec_config.describe(),
    )

    plugin_repo = Path(__file__).resolve().parents[3]
    metal_home = request.config.getoption("--tt-metal-home") or os.environ.get(
        "TT_METAL_HOME"
    )
    try:
        import vllm

        vllm_version = vllm.__version__
    except Exception as exc:  # pragma: no cover
        vllm_version = f"<unavailable: {exc}>"
    manifest.environment = {
        "server_url": tt_server_url,
        "model": tt_model_name,
        "vllm_version": vllm_version,
        "python": sys.version,
        "plugin": _repo_state(plugin_repo),
        "tt_metal": _repo_state(Path(metal_home) if metal_home else None),
        "spec_env": {
            name: os.environ.get(name)
            for name in (
                "TT_SPEC_ACCEPT_DEPTH",
                "TT_SPEC_TARGET",
                "MESH_DEVICE",
                "PYTHONPATH",
            )
        },
    }

    yield manifest

    server_log = request.config.getoption("--tt-spec-server-log")
    final: Metrics | None = None
    try:
        final = Metrics.scrape(tt_server_url)
    except Exception:  # pragma: no cover  (a server that died mid-session)
        final = None
    path = manifest.write(final)
    if server_log and Path(server_log).exists():
        destination = manifest.directory / "server.log"
        # The driver already writes the server log into this directory, so the
        # copy would be onto itself.
        if not destination.exists() or not Path(server_log).samefile(destination):
            shutil.copy2(server_log, destination)
    print(f"\nrun manifest: {path}")


@pytest.fixture
def record(run_manifest, request):
    """Records one test's evidence into the manifest."""

    def _record(**payload: Any) -> None:
        run_manifest.record(request.node.name, **payload)

    return _record
