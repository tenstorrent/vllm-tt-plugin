# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""HTTP and metrics helpers for the speculative-decoding device suite.

These tests talk to the server directly rather than through the OpenAI client,
for two reasons. They send token-id prompts and the fields the speculative
dummy needs (``ignore_eos``, ``stop_token_ids``, ``return_token_ids``), which
the client does not all expose. And every assertion about whether speculation
happened is a delta on vLLM's own Prometheus counters, which means scraping
``/metrics`` around each request rather than reading a response body.

The counters that matter, all per engine:

``vllm:spec_decode_num_drafts_total``
    Speculative steps. Zero means no step ever carried a draft, whatever the
    responses look like.
``vllm:spec_decode_num_draft_tokens_total``
    Drafts offered across those steps.
``vllm:spec_decode_num_accepted_tokens_total``
    Drafts accepted. The ratio to the line above is the acceptance rate the
    server actually achieved.
``vllm:spec_decode_num_accepted_tokens_per_pos_total{position}``
    Acceptances per draft position, which is what distinguishes "accepted two
    of five on every step" from "accepted five of five on two steps in five".
``vllm:num_preemptions_total``
    Preemptions. A capacity test that does not check this has not established
    that preemption happened.
The server log reports ``TT scheduler: widest decode batch reached N request
row(s)`` from the scheduler's actual decode batch. The concurrency regression
uses that direct signal because ``vllm:iteration_tokens_total`` also counts
prefill tokens and cannot establish decode batch width.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# A generous timeout: a speculative step is milliseconds, but a capacity test
# queues many requests behind each other and the whole point is that they wait.
TIMEOUT = httpx.Timeout(600.0, connect=10.0)

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][^{\s]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)$"
)
_WIDEST_DECODE_BATCH = re.compile(
    r"TT scheduler: widest decode batch reached (\d+) request row\(s\)"
)


def _labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
        # A label value can contain a comma only inside quotes, and none of the
        # labels read here do, so a plain split is enough.
    out = {}
    for piece in raw.split(","):
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        out[key.strip()] = value.strip().strip('"')
    return out


@dataclass(frozen=True)
class Metrics:
    """One ``/metrics`` scrape, parsed into a name-and-labels lookup."""

    raw: str
    samples: dict[tuple[str, tuple[tuple[str, str], ...]], float]

    @classmethod
    def scrape(cls, base_url: str) -> Metrics:
        response = httpx.get(f"{base_url.rstrip('/')}/metrics", timeout=TIMEOUT)
        response.raise_for_status()
        samples: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        for line in response.text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = _SAMPLE.match(line)
            if not match:
                continue
            labels = _labels(match.group("labels"))
            # The engine and model labels are constant for one server and only
            # get in the way of a lookup, so drop them and keep the rest.
            keep = tuple(
                sorted(
                    (key, value)
                    for key, value in labels.items()
                    if key not in ("engine", "model_name")
                )
            )
            try:
                samples[(match.group("name"), keep)] = float(match.group("value"))
            except ValueError:
                continue
        return cls(raw=response.text, samples=samples)

    def get(self, name: str, **labels: str) -> float:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        return self.samples.get(key, 0.0)

    def per_position(self, name: str, positions: int) -> list[float]:
        return [self.get(name, position=str(index)) for index in range(positions)]


@dataclass
class Acceptance:
    """What the counters say happened between two scrapes."""

    drafts: int
    draft_tokens: int
    accepted: int
    per_position: list[int]
    steps: int
    preemptions: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.draft_tokens if self.draft_tokens else 0.0

    @property
    def mean_acceptance_length(self) -> float:
        """vLLM's own definition: one committed token plus the accepted drafts."""
        return 1 + self.accepted / self.drafts if self.drafts else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "drafts": self.drafts,
            "draft_tokens": self.draft_tokens,
            "accepted": self.accepted,
            "per_position": self.per_position,
            "steps": self.steps,
            "preemptions": self.preemptions,
            "acceptance_rate": round(self.acceptance_rate, 6),
            "mean_acceptance_length": round(self.mean_acceptance_length, 6),
        }


def acceptance_delta(before: Metrics, after: Metrics, positions: int) -> Acceptance:
    """The counter movement across one request, or one batch of them."""
    return Acceptance(
        drafts=int(
            after.get("vllm:spec_decode_num_drafts_total")
            - before.get("vllm:spec_decode_num_drafts_total")
        ),
        draft_tokens=int(
            after.get("vllm:spec_decode_num_draft_tokens_total")
            - before.get("vllm:spec_decode_num_draft_tokens_total")
        ),
        accepted=int(
            after.get("vllm:spec_decode_num_accepted_tokens_total")
            - before.get("vllm:spec_decode_num_accepted_tokens_total")
        ),
        per_position=[
            int(a - b)
            for a, b in zip(
                after.per_position(
                    "vllm:spec_decode_num_accepted_tokens_per_pos_total", positions
                ),
                before.per_position(
                    "vllm:spec_decode_num_accepted_tokens_per_pos_total", positions
                ),
            )
        ],
        steps=int(
            after.get("vllm:iteration_tokens_total_count")
            - before.get("vllm:iteration_tokens_total_count")
        ),
        preemptions=int(
            after.get("vllm:num_preemptions_total")
            - before.get("vllm:num_preemptions_total")
        ),
    )


def widest_decode_batch_size(server_log: Path) -> int:
    """Return the largest decode row count reported by ``TTScheduler``."""
    widest = 0
    for line in server_log.read_text(errors="replace").splitlines():
        if found := _WIDEST_DECODE_BATCH.search(line):
            widest = max(widest, int(found.group(1)))
    return widest


def assert_multirow_decode_was_scheduled(server_log: Path) -> int:
    """Require direct evidence that at least two rows shared a decode step."""
    widest = widest_decode_batch_size(server_log)
    assert widest >= 2, (
        "the TT scheduler reported no decode batch with at least two request "
        f"rows; the widest reported decode batch contained {widest} row(s)"
    )
    return widest


@dataclass
class Completion:
    """One non-streaming ``/v1/completions`` response, with its request."""

    request: dict[str, Any]
    status: int
    body: dict[str, Any]

    @property
    def token_ids(self) -> list[int]:
        return list(self.body["choices"][0].get("token_ids") or [])

    @property
    def text(self) -> str:
        return self.body["choices"][0]["text"]

    @property
    def finish_reason(self) -> str:
        return self.body["choices"][0]["finish_reason"]

    @property
    def stop_reason(self):
        return self.body["choices"][0].get("stop_reason")

    @property
    def completion_tokens(self) -> int:
        return int(self.body["usage"]["completion_tokens"])


def assert_full_length_completion(completion: Completion, requested: int) -> None:
    """Require the full ignored-EOS completion and its length termination."""
    assert completion.status == 200
    assert completion.completion_tokens == requested
    assert len(completion.token_ids) == requested
    assert completion.finish_reason == "length"


@dataclass
class StreamedCompletion:
    """One streaming response, reassembled."""

    request: dict[str, Any]
    chunks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(chunk["choices"][0].get("text") or "" for chunk in self.chunks)

    @property
    def token_ids(self) -> list[int]:
        ids: list[int] = []
        for chunk in self.chunks:
            ids.extend(chunk["choices"][0].get("token_ids") or [])
        return ids

    @property
    def finish_reason(self):
        for chunk in reversed(self.chunks):
            reason = chunk["choices"][0].get("finish_reason")
            if reason:
                return reason
        return None


class SpecServer:
    """The server under test, as this suite uses it."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def request_body(self, prompt: list[int], **overrides: Any) -> dict[str, Any]:
        """A greedy completion request, which is all speculation serves.

        Token ids rather than text: the prompt has to be exact for the n-gram
        drafter to reach anything, and a tokenizer would put its own tokens in
        the way.
        """
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        body.update(overrides)
        return body

    def complete(self, prompt: list[int], **overrides: Any) -> Completion:
        body = self.request_body(prompt, **overrides)
        response = httpx.post(
            f"{self.base_url}/v1/completions", json=body, timeout=TIMEOUT
        )
        return Completion(
            request=body,
            status=response.status_code,
            body=response.json() if response.content else {},
        )

    def stream(self, prompt: list[int], **overrides: Any) -> StreamedCompletion:
        body = self.request_body(prompt, stream=True, **overrides)
        streamed = StreamedCompletion(request=body)
        with httpx.stream(
            "POST", f"{self.base_url}/v1/completions", json=body, timeout=TIMEOUT
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                streamed.chunks.append(json.loads(payload))
        return streamed

    def stream_and_abandon(
        self, prompt: list[int], after: int, **overrides: Any
    ) -> int:
        """Start a stream, read ``after`` chunks, then disconnect.

        Returns how many chunks were read. Closing the response mid-stream is
        what a cancelling client does, and the server has to notice and free
        the request's row rather than keep decoding it forever.
        """
        body = self.request_body(prompt, stream=True, **overrides)
        read = 0
        with httpx.stream(
            "POST", f"{self.base_url}/v1/completions", json=body, timeout=TIMEOUT
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                read += 1
                if read >= after:
                    break
        return read

    def context_length(self) -> int:
        """The context this server resolved, from its own model listing.

        Asked rather than passed in: a test that needs to stay inside the
        context should use the number the engine is enforcing, and the
        capacity configurations run a deliberately small one.
        """
        response = httpx.get(f"{self.base_url}/v1/models", timeout=TIMEOUT)
        response.raise_for_status()
        for card in response.json().get("data", []):
            length = card.get("max_model_len")
            if length:
                return int(length)
        raise AssertionError("the server's model listing carries no max_model_len")

    def reset_prefix_cache(self, *, reset_running_requests: bool = True) -> bool:
        """Ask the engine for a wholesale prefix-cache reset.

        The path the plugin's forced-reset accounting exists for: with
        ``reset_running_requests`` the engine preempts every running request
        and frees its blocks, and each request's in-flight output frames become
        stale. Returns what the engine reports, which is ``False`` while blocks
        are still held, so a caller retries rather than assuming.

        Behind ``VLLM_SERVER_DEV_MODE``, which the launch that uses this sets.
        A server without it answers 404 and the test skips.
        """
        response = httpx.post(
            f"{self.base_url}/reset_prefix_cache",
            params={"reset_running_requests": str(reset_running_requests).lower()},
            timeout=TIMEOUT,
        )
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return bool(response.json().get("success", False))

    def prefix_cache_reset_is_available(self) -> bool:
        """Whether this launch exposes the reset endpoint at all."""
        response = httpx.post(
            f"{self.base_url}/reset_prefix_cache",
            params={"reset_running_requests": "false"},
            timeout=TIMEOUT,
        )
        return response.status_code != 404

    def metrics(self) -> Metrics:
        return Metrics.scrape(self.base_url)
