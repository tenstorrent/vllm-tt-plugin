# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stop a request that has fallen into a single-token loop.

A model can fall into a degenerate loop that emits one token for ever (a Python
traceback underline of ``^`` characters is the case seen on Gemma 4 31B). Nothing
ends it before ``max_tokens``: with a 32K limit that is minutes of device time for
a reply an agent will reject anyway, and under load the loop holds KV blocks and a
decode row.

The guard counts the consecutive identical committed token ids of each request.
When a run of ``MAX_IDENTICAL_TOKEN_RUN`` is followed by one more identical token,
that token is replaced by the request's own stop token id, so the engine finishes
the request with ``finish_reason=stop``; the repeats already produced are delivered
unchanged.

The limit is far above anything legitimate: over 8.1M real generated tokens of
SWE-bench and Terminal-Bench agent runs (25K generations) the longest identical
token run was 62 outside one 13,833-token loop. Output that never reaches the limit
is bit-identical with or without the guard. A request with ``ignore_eos``
(benchmark and forced-length runs) or without any stop token id is never touched.
"""

from __future__ import annotations

import threading

MAX_IDENTICAL_TOKEN_RUN = 1024


class RepetitionGuard:
    """Per-request run length of the last committed token."""

    def __init__(self, max_run: int = MAX_IDENTICAL_TOKEN_RUN) -> None:
        self.max_run = max_run
        # req_id -> [last token, run length]
        self._runs: dict[str, list[int]] = {}
        self._lock = threading.Lock()

    def apply(
        self, req_id: str, tokens: list[int], stop_token_id: int | None
    ) -> list[int]:
        """Account for ``tokens`` committed in order and return what to publish.

        Returns ``tokens`` itself unless the run limit is exceeded inside it: then
        the tokens before the offending one, followed by ``stop_token_id``. Without
        a stop token id (``None``) nothing is counted or changed.
        """
        if stop_token_id is None or not tokens:
            return tokens
        with self._lock:
            state = self._runs.get(req_id)
            last, run = (state[0], state[1]) if state else (None, 0)
            for i, token in enumerate(tokens):
                if token == last:
                    if run >= self.max_run:
                        self._runs.pop(req_id, None)
                        return tokens[:i] + [stop_token_id]
                    run += 1
                else:
                    last, run = token, 1
            self._runs[req_id] = [last, run]
        return tokens

    def drop(self, req_id: str) -> None:
        with self._lock:
            self._runs.pop(req_id, None)
