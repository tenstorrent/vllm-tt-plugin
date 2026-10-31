# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Model-owned prefix cache: the scheduler asks the model for the hit of a new request (``prefix_cache_hit_policy = "model"``).

A model that keeps its own block-paged prefix cache on the device (DeepSeek-V4.1-Flash, ``DSV41_APC_PAGED``) decides what a request can reuse: only it knows which 512-token blocks and
halo records are resident, on which mesh row, and whether a user is free there. The scheduler and the model runner live in one process (uniproc executor), so a module-level registry is
the hand-over, like ``prefill_rows``. The model registers an object with

    plan_hit(request_id, prompt_token_ids, num_tokens, max_hit=None) -> (hit_tokens, row)    # pins the cached blocks and reserves a user on ``row``
    cancel(request_id)                                                                       # drop a reservation that was never bound (abort)

``hit_tokens`` is a multiple of the model's grid and at most ``max_hit`` (vLLM's own hit: the blocks vLLM can hand out). The model recognises the reservation when ``prefill_forward``
receives the request: the runner passes ``request_ids=[...]`` (one per prefill row) and ``start_pos`` = the hit. No hook registered (another model, the model's switch off, a worker in
another process): the hit is 0 and the request is computed from position 0. Stdlib only: importable without vLLM.
"""

from __future__ import annotations

import threading

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

_lock = threading.Lock()
_hook = None
_errors = 0


def set_prefix_hook(hook) -> None:
    """Register (or, with None, remove) the model's hook object."""
    global _hook
    with _lock:
        _hook = hook


def get_prefix_hook():
    with _lock:
        return _hook


def plan_model_hit(request_id, prompt_token_ids, num_tokens: int, max_hit: int | None = None) -> int:
    """The model's prefix hit for a new request, in tokens (0 = none, no hook, or a failing hook: a broken model hook must not stop the scheduler)."""
    global _errors
    hook = get_prefix_hook()
    if hook is None or num_tokens <= 0 or (max_hit is not None and max_hit <= 0):
        return 0
    try:
        hit, _row = hook.plan_hit(request_id, prompt_token_ids, int(num_tokens), max_hit=max_hit)
    except Exception:  # noqa: BLE001
        _errors += 1
        if _errors <= 3:
            logger.exception("model prefix hook failed for %s: treating as a miss", request_id)
        return 0
    hit = int(hit or 0)
    if max_hit is not None:
        hit = min(hit, int(max_hit))
    return max(hit, 0)


def cancel_model_hit(request_ids) -> None:
    """Tell the model that these requests will never be bound (aborted while waiting): it drops their reservations (a bound request is released by the model itself)."""
    hook = get_prefix_hook()
    cancel = getattr(hook, "cancel", None)
    if cancel is None:
        return
    for rid in [request_ids] if isinstance(request_ids, str) else list(request_ids):
        try:
            cancel(rid)
        except Exception:  # noqa: BLE001
            logger.exception("model prefix hook cancel failed for %s", rid)
