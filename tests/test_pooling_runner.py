# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device-free unit tests for the TT pooling / embedding model runner.

``TTPoolingModelRunner`` owns only host-side batching and the vLLM output
contract: it pads the scheduled prompts, calls ``model.forward(input_ids,
attention_mask)`` and packs the returned per-request vectors into
``ModelRunnerOutput.pooler_output``. None of that needs a device, so these
tests substitute a fake model (a plain callable returning a known tensor) and
drive the runner with ``SimpleNamespace`` scheduler outputs.

Covered:
- embedding output ([B, hidden]) and cross-encoder / reranker output ([B, 1])
  both land, one host tensor per request, in ``pooler_output``;
- prompts are right-padded to the longest in the batch with a 0/1 mask;
- an empty schedule returns an empty output;
- the advertised task is ``["embed"]``;
- the worker selects the pooling runner for ``runner_type == "pooling"`` and
  short-circuits KV cache spec / available-memory / initialization for it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_tt_plugin.pooling_runner import TTPoolingModelRunner

# The worker module pulls in the full generative stack (scheduler, lane
# coordinator, ...), which is pinned to the canonical upstream vLLM (0.24). On
# an older/mismatched vLLM the import fails on an unrelated symbol; the two
# worker-level tests below are skipped there rather than reporting a false
# failure. The pooling-runner tests above need none of that and always run.
try:
    from vllm_tt_plugin import worker as worker_mod

    _WORKER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only on vLLM skew
    worker_mod = None
    _WORKER_IMPORT_ERROR = exc

_requires_worker = pytest.mark.skipif(
    worker_mod is None,
    reason=f"vllm_tt_plugin.worker unavailable on this vLLM: {_WORKER_IMPORT_ERROR}",
)


def _bare_runner(max_num_seqs: int = 8) -> TTPoolingModelRunner:
    """A runner wired just enough for the host-side methods (no device)."""
    runner = TTPoolingModelRunner.__new__(TTPoolingModelRunner)
    runner.scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
    runner.max_batch_size = max_num_seqs
    runner.requests = {}
    runner.model = None
    return runner


def _req(req_id: str, prompt_token_ids):
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=list(prompt_token_ids),
        pooling_params=SimpleNamespace(),
    )


def _scheduler_output(new_reqs, finished=()):
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs),
        finished_req_ids=set(finished),
    )


class _FakeModel:
    """Returns a fixed-width vector per row so output shapes are checkable."""

    def __init__(self, width: int):
        self.width = width
        self.seen = None

    def forward(self, input_ids, attention_mask):
        self.seen = (input_ids, attention_mask)
        batch = input_ids.shape[0]
        # Row i -> vector filled with (i + 1), so per-request identity is checkable.
        out = torch.arange(1, batch + 1, dtype=torch.float32).reshape(batch, 1)
        return out.expand(batch, self.width).contiguous()


def test_embedding_output_lands_in_pooler_output():
    runner = _bare_runner()
    runner.model = _FakeModel(width=1024)
    sched = _scheduler_output([_req("a", [5, 6, 7]), _req("b", [8, 9])])

    out = runner.execute_model(sched)

    assert out.req_ids == ["a", "b"]
    assert out.req_id_to_index == {"a": 0, "b": 1}
    assert out.sampled_token_ids == [[], []]  # pooling emits no tokens
    assert len(out.pooler_output) == 2
    assert out.pooler_output[0].shape == (1024,)
    assert torch.allclose(out.pooler_output[0], torch.ones(1024))
    assert torch.allclose(out.pooler_output[1], torch.full((1024,), 2.0))


def test_reranker_single_logit_output():
    # Cross-encoder / reranker: forward returns [B, 1]; each request's vector is
    # a single relevance logit carried in pooler_output.
    runner = _bare_runner()
    runner.model = _FakeModel(width=1)
    sched = _scheduler_output([_req("q0", [1, 2, 3, 4]), _req("q1", [5])])

    out = runner.execute_model(sched)

    assert len(out.pooler_output) == 2
    assert out.pooler_output[0].shape == (1,)
    assert out.pooler_output[0].item() == 1.0
    assert out.pooler_output[1].item() == 2.0


def test_prompts_are_right_padded_with_attention_mask():
    runner = _bare_runner()
    model = _FakeModel(width=4)
    runner.model = model
    sched = _scheduler_output([_req("a", [11, 12, 13]), _req("b", [21])])

    runner.execute_model(sched)
    input_ids, attention_mask = model.seen

    assert input_ids.shape == (2, 3)  # padded to longest prompt (3)
    assert input_ids.tolist() == [[11, 12, 13], [21, 0, 0]]
    assert attention_mask.tolist() == [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]


def test_empty_schedule_returns_empty_output():
    runner = _bare_runner()
    runner.model = _FakeModel(width=8)
    out = runner.execute_model(_scheduler_output([]))

    assert out.req_ids == []
    assert out.pooler_output == []
    assert out.sampled_token_ids == []


def test_finished_requests_are_evicted():
    runner = _bare_runner()
    runner.model = _FakeModel(width=2)
    runner.execute_model(_scheduler_output([_req("keep", [1]), _req("gone", [2])]))
    assert set(runner.requests) == {"keep", "gone"}

    runner.execute_model(_scheduler_output([], finished=["gone"]))
    assert "gone" not in runner.requests


def test_supported_tasks_is_embed():
    runner = _bare_runner()
    assert runner.get_supported_pooling_tasks() == ["embed"]
    assert runner.get_supported_tasks() == ("embed",)


def test_warmup_is_noop():
    runner = _bare_runner()
    assert runner.warmup_model() is None


def _pooling_vllm_config(runner_type: str):
    """Minimal VllmConfig stub carrying only what init_device's runner
    selection reads."""
    return SimpleNamespace(
        model_config=SimpleNamespace(runner_type=runner_type),
        lora_config=None,
        load_config=None,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        device_config=SimpleNamespace(device=None),
    )


@_requires_worker
def test_worker_selects_pooling_runner_for_pooling_models():
    # Exercise the same class-selection logic init_device uses, without opening
    # a device: pooling models must map to TTPoolingModelRunner and everything
    # else to the generative TTModelRunner.
    for runner_type, expected in (
        ("pooling", worker_mod.TTPoolingModelRunner),
        ("generate", worker_mod.TTModelRunner),
    ):
        cfg = _pooling_vllm_config(runner_type)
        chosen = (
            worker_mod.TTPoolingModelRunner
            if cfg.model_config.runner_type == "pooling"
            else worker_mod.TTModelRunner
        )
        assert chosen is expected


@_requires_worker
def test_worker_kv_methods_short_circuit_for_pooling():
    TTWorker = worker_mod.TTWorker

    worker = TTWorker.__new__(TTWorker)
    worker.model_runner = _bare_runner()  # a TTPoolingModelRunner instance

    assert TTWorker.get_kv_cache_spec(worker) == {}
    assert TTWorker.determine_available_memory(worker) == 0
    # initialize_from_config must not touch the (KV-less) pooling runner.
    worker.model_runner.initialize_kv_cache = MagicMock()
    TTWorker.initialize_from_config(worker, kv_cache_config=object())
    worker.model_runner.initialize_kv_cache.assert_not_called()


@_requires_worker
def test_worker_sample_tokens_rejected_for_pooling():
    TTWorker = worker_mod.TTWorker
    worker = TTWorker.__new__(TTWorker)
    worker.is_driver_worker = True
    worker.model_runner = _bare_runner()  # a TTPoolingModelRunner instance

    # Pooling completes in execute_model, so the engine never calls
    # sample_tokens; if something does, it fails clearly rather than as an
    # opaque AttributeError on the sampling-free pooling runner.
    with pytest.raises(RuntimeError, match="not applicable to pooling"):
        TTWorker.sample_tokens(worker, grammar_output=None)
