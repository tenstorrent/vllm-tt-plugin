"""Row-aware admission of prompts into prefill (default off: ``additional_config['tt']['prefill_row_aware']`` or ``TT_PREFILL_ROW_AWARE=1``).

A model whose prefill state lives in a per-mesh-row slot (DeepSeek-V4.1-Flash with DSV41_PREFILL_UP < users per row) loses the carried state of a prompt when another prompt is
prefilled in the same mesh row. Which row a request sits in is the model's business (its slot table), so the model publishes how many more prompts its rows can take into prefill
(``prefill_row_capacity()``: rows without an in-progress prompt that still have a free user) and the scheduler admits at most that many waiting prompts per step; the rest wait.

Scheduler and model runner live in one process (uniproc executor), so a module-level registry is the hand-over. No provider registered (another model, the model's own switch off, a
worker in another process) -> ``prefill_row_capacity()`` is None and the scheduler's static cap is all there is: nothing changes.
"""

from __future__ import annotations

from collections.abc import Callable

_provider: Callable[[], int | None] | None = None


def set_prefill_capacity_provider(fn: Callable[[], int | None] | None) -> None:
    global _provider
    _provider = fn


def prefill_row_capacity() -> int | None:
    """Prompts that can start prefill now, or None (unknown / no limit)."""
    fn = _provider
    if fn is None:
        return None
    try:
        value = fn()
    except Exception:  # a broken provider must not stop the scheduler
        return None
    return None if value is None else max(0, int(value))
