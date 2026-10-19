# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The TT Omni generation worker and runner: streamed chunks decode as one stream."""

import os
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm_omni")

import torch  # noqa: E402
import vllm.config  # noqa: E402,F401  # finish vLLM init before the plugin package
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheConfig  # noqa: E402
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT  # noqa: E402
from vllm_omni.core.sched.omni_generation_scheduler import (  # noqa: E402
    OmniGenerationScheduler,
)
from vllm_omni.core.sched.output import (  # noqa: E402
    OmniCachedRequestData,
    OmniNewRequestData,
)
from vllm_omni.outputs import OmniModelRunnerOutput  # noqa: E402
from vllm_omni.worker.omni_connector_validation import (  # noqa: E402
    validate_worker_omni_connector,
)

from vllm_tt_plugin.config import resolve_tt_scheduler_cls  # noqa: E402
from vllm_tt_plugin.omni import generation_model_runner  # noqa: E402
from vllm_tt_plugin.omni.generation_model_runner import (  # noqa: E402
    TTOmniGenerationModelRunner,
)
from vllm_tt_plugin.omni.worker import TTOmniGenerationWorker  # noqa: E402

NUM_QUANTIZERS = 2
SAMPLES_PER_FRAME = 3
SAMPLE_RATE = 24000


class _StatefulDecoder:
    """Each frame's samples depend on the frame before it, as a causal decoder's do.

    A chunk decoded without its request's state, or after skipping the wrong number
    of context frames, comes out different from one-shot decoding.
    """

    def __init__(self):
        self.last_frame: dict[str, int] = {}
        self.finished: list[set[str]] = []

    def on_requests_finished(self, req_ids):
        self.finished.append(set(req_ids))
        for req_id in req_ids:
            self.last_frame.pop(req_id, None)

    def __call__(
        self,
        input_ids,
        positions,
        runtime_additional_information,
        model_intermediate_buffer,
        seq_token_counts,
        request_ids,
    ):
        assert runtime_additional_information is model_intermediate_buffer
        audio, offset = [], 0
        for req_id, count, info in zip(
            request_ids, seq_token_counts, runtime_additional_information, strict=True
        ):
            codes = input_ids[offset : offset + count].reshape(NUM_QUANTIZERS, -1)
            offset += count
            context = info["meta"]["left_context_size"]
            samples, previous = decode(codes, self.last_frame.get(req_id, 0))
            self.last_frame[req_id] = previous
            audio.append(samples[context * SAMPLES_PER_FRAME :])
        return SimpleNamespace(
            multimodal_outputs={
                "model_outputs": audio,
                "sr": [torch.tensor(SAMPLE_RATE, dtype=torch.int32)] * len(audio),
            }
        )


def decode(codes: torch.Tensor, previous: int) -> tuple[torch.Tensor, int]:
    """Quantizer-major codes [Q, F] to samples, given the last frame decoded before."""
    samples = []
    for frame in codes.sum(dim=0).tolist():
        samples += [float(frame + 10 * previous)] * SAMPLES_PER_FRAME
        previous = frame
    return torch.tensor(samples, dtype=torch.float32), previous


def _frames(first: int, count: int) -> torch.Tensor:
    return torch.arange(first, first + NUM_QUANTIZERS * count).reshape(
        count, NUM_QUANTIZERS
    )


def _flat(frames: torch.Tensor) -> list[int]:
    """Frames [F, Q] as Omni ships them: quantizer-major, flat."""
    return frames.T.reshape(-1).tolist()


def _new(req_id: str, frames: torch.Tensor, context: int) -> OmniNewRequestData:
    return OmniNewRequestData(
        req_id=req_id,
        prompt_token_ids=_flat(frames),
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
        additional_information={
            "meta": {"left_context_size": context, "request_id": f"ext-{req_id}"},
            "speaker": {"name": "vivian"},
        },
    )


def _step(new=(), cached=None, finished=()) -> SchedulerOutput:
    """One scheduler step. ``cached`` maps a running request to its next chunk."""
    cached = cached or {}
    num_scheduled = {req.req_id: len(req.prompt_token_ids) for req in new}
    num_scheduled.update({req_id: len(_flat(f)) for req_id, f in cached.items()})
    return SchedulerOutput(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=OmniCachedRequestData(
            req_ids=list(cached),
            resumed_req_ids=set(),
            new_token_ids=[[] for _ in cached],
            all_token_ids={},
            new_block_ids=[None for _ in cached],
            num_computed_tokens=[0 for _ in cached],
            num_output_tokens=[0 for _ in cached],
            prompt_token_ids={req_id: _flat(f) for req_id, f in cached.items()},
            # Omni's scheduler sends the request's metadata merged so far; a later
            # chunk carries no reference context.
            additional_information={
                req_id: {"meta": {"left_context_size": 0}} for req_id in cached
            },
        ),
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
    )


def _runner(async_chunk=True, async_scheduling=False, use_v2_model_runner=False):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            async_chunk=async_chunk, use_v2_model_runner=use_v2_model_runner
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    runner = TTOmniGenerationModelRunner(vllm_config, None, "none", False, 1)
    runner.model = _StatefulDecoder()
    return runner


def _run(runner, step) -> OmniModelRunnerOutput:
    assert runner.execute_model(step) is None
    output = runner.sample_tokens(None)
    assert isinstance(output, OmniModelRunnerOutput)
    assert output.sampled_token_ids == []
    return output


def _audio(output, req_id) -> torch.Tensor:
    row = output.multimodal_outputs[output.req_id_to_index[req_id]]
    assert int(row["sr"]) == SAMPLE_RATE
    return row["model_outputs"]


def test_omni_accepts_the_generation_worker():
    validate_worker_omni_connector(TTOmniGenerationWorker, True)
    assert TTOmniGenerationWorker.uses_token_kv_cache is False


def test_omnis_generation_scheduler_runs_the_stage_unchanged():
    resolved = resolve_tt_scheduler_cls(
        OmniGenerationScheduler, TTOmniGenerationWorker, False
    )
    assert resolved is OmniGenerationScheduler


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ({"use_v2_model_runner": True}, "model_runner: v1"),
        ({"async_chunk": False}, "async_chunk: true"),
        ({"async_scheduling": True}, "async_scheduling: false"),
    ],
)
def test_the_runner_refuses_an_unsupported_stage_config(setting, message):
    with pytest.raises(ValueError, match=message):
        _runner(**setting)


def test_the_runner_takes_the_empty_kv_config_and_refuses_kv_groups():
    runner = _runner()
    runner.initialize_kv_cache(
        KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[])
    )

    with pytest.raises(ValueError, match="no token KV cache"):
        runner.initialize_kv_cache(
            KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=["g"])
        )


def test_streamed_chunks_decode_as_one_stream():
    """First chunk with a reference prefix, ramp, steady and final partial chunks,
    then an empty final chunk; two requests share steps at different offsets."""
    ref_a, ref_b = 3, 0
    a = _frames(0, ref_a + 1 + 2 + 4 + 5 + 5 + 2)
    b = _frames(500, 1 + 2 + 3)
    runner = _runner()
    streamed = {"a": [], "b": []}

    def collect(output):
        for req_id in output.req_ids:
            streamed[req_id].append(_audio(output, req_id))

    collect(_run(runner, _step(new=[_new("a", a[: ref_a + 1], ref_a)])))
    collect(_run(runner, _step(new=[_new("b", b[:1], ref_b)], cached={"a": a[4:6]})))
    collect(_run(runner, _step(cached={"a": a[6:10], "b": b[1:3]})))
    collect(_run(runner, _step(cached={"a": a[10:15], "b": b[3:6]})))
    # b's empty final chunk: the scheduler finishes it with nothing to run.
    assert runner.execute_model(_step(finished={"b"})) is EMPTY_MODEL_RUNNER_OUTPUT
    collect(_run(runner, _step(cached={"a": a[15:20]})))
    collect(_run(runner, _step(cached={"a": a[20:22]})))
    assert runner.execute_model(_step(finished={"a"})) is EMPTY_MODEL_RUNNER_OUTPUT

    for req_id, frames, ref in (("a", a, ref_a), ("b", b, ref_b)):
        one_shot, _ = decode(frames.T, 0)
        assert torch.equal(
            torch.cat(streamed[req_id]), one_shot[ref * SAMPLES_PER_FRAME :]
        )
    assert runner.model.finished == [{"b"}, {"a"}]
    assert runner.model.last_frame == {}
    assert runner.infos == runner.prompt_token_ids == runner.num_computed_tokens == {}


def test_a_new_requests_metadata_reaches_the_model_merged_with_later_chunks():
    seen = []

    class _Recording(_StatefulDecoder):
        def __call__(self, **kwargs):
            infos = kwargs["model_intermediate_buffer"]
            seen.append([dict(info["meta"]) for info in infos])
            return super().__call__(**kwargs)

    runner = _runner()
    runner.model = _Recording()
    _run(runner, _step(new=[_new("a", _frames(0, 2), 1)]))
    _run(runner, _step(cached={"a": _frames(4, 1)}))

    assert seen == [
        [{"left_context_size": 1, "request_id": "ext-a"}],
        [{"left_context_size": 0, "request_id": "ext-a"}],
    ]
    assert runner.infos["a"]["speaker"] == {"name": "vivian"}


def test_the_model_is_built_on_the_host_from_the_stage_architecture(
    monkeypatch: pytest.MonkeyPatch,
):
    built = {}

    class _Model(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix):
            super().__init__()
            built["device"] = vllm_config.device_config.device

    class _Loader:
        def load_weights(self, model, model_config):
            built["loaded"] = model

    monkeypatch.setattr(
        generation_model_runner,
        "get_model_architecture",
        lambda _cfg: (_Model, "TTQwen3TTSCode2Wav"),
    )
    monkeypatch.setattr(
        generation_model_runner, "get_model_loader", lambda _load_config: _Loader()
    )
    monkeypatch.setattr(
        generation_model_runner,
        "set_current_vllm_config",
        lambda _cfg: torch.no_grad(),
    )
    mesh_device = object()
    runner = _runner()
    runner.vllm_config.device_config = SimpleNamespace(device=mesh_device)
    runner.vllm_config.load_config = None

    runner.load_model()

    assert built["device"] == torch.device("cpu")
    assert built["loaded"] is runner.model
    assert not runner.model.training
    assert runner.vllm_config.device_config.device is mesh_device


QWEN3_TTS_MODEL = os.environ.get("QWEN3_TTS_MODEL")


def _code2wav(async_chunk: bool):
    """vLLM-Omni's Code2Wav on the host, with the chunk schedule the talker sends."""
    from vllm.config import LoadConfig
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import (
        Qwen3TTSCode2Wav,
    )

    extra = {
        "codec_chunk_frames": 25,
        "codec_left_context_frames": 72,
        "initial_codec_chunk_frames": 1,
        "codec_chunk_ramp": "1,2,4,25",
    }
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=QWEN3_TTS_MODEL,
            async_chunk=async_chunk,
            revision=None,
            dtype=torch.float32,
            stage_connector_config={"extra": extra},
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        load_config=LoadConfig(),
    )
    model = Qwen3TTSCode2Wav(vllm_config=vllm_config)
    model.load_weights(iter(()))
    return vllm_config, model.eval()


@pytest.mark.skipif(
    not QWEN3_TTS_MODEL, reason="QWEN3_TTS_MODEL names no Qwen3-TTS checkpoint"
)
def test_the_code2wav_reference_streams_as_it_decodes_one_shot(monkeypatch):
    """First chunk with a reference prefix, ramp, steady, final partial and empty
    final chunks; two requests at different offsets share steps, so the decoder pads
    the shorter one. Every chunk join is checked."""
    monkeypatch.setattr(sys.modules[__name__], "NUM_QUANTIZERS", 16)
    generator = torch.Generator().manual_seed(0)
    ref = {"a": 8, "b": 4}
    chunks = {"a": [8 + 1, 2, 4, 25, 25, 7], "b": [4 + 1, 2, 4, 25, 3]}
    frames = {
        req_id: torch.randint(0, 2048, (sum(sizes), 16), generator=generator)
        for req_id, sizes in chunks.items()
    }
    vllm_config, model = _code2wav(async_chunk=True)
    runner = TTOmniGenerationModelRunner(vllm_config, None, "none", False, 1)
    runner.model = model

    def first(req_id):
        new = _new(req_id, frames[req_id][: chunks[req_id][0]], ref[req_id])
        new.additional_information["meta"].update(
            ref_context_size=ref[req_id],
            ref_context_request_id=f"ref-{req_id}",
            ref_context_included=True,
        )
        return new

    edges = {req_id: [0] for req_id in chunks}
    for req_id, sizes in chunks.items():
        for size in sizes:
            edges[req_id].append(edges[req_id][-1] + size)
    streamed = {req_id: [] for req_id in chunks}

    def collect(output):
        for req_id in output.req_ids:
            streamed[req_id].append(_audio(output, req_id))

    def window(req_id, index):
        return frames[req_id][edges[req_id][index] : edges[req_id][index + 1]]

    collect(_run(runner, _step(new=[first("a")])))
    collect(_run(runner, _step(new=[first("b")], cached={"a": window("a", 1)})))
    for index in range(2, 5):
        cached = {"a": window("a", index), "b": window("b", index - 1)}
        collect(_run(runner, _step(cached=cached)))
    collect(_run(runner, _step(cached={"a": window("a", 5), "b": window("b", 4)})))
    # Both empty final chunks: the scheduler finishes the requests with nothing to run.
    assert runner.execute_model(_step(finished={"a", "b"})) is (
        EMPTY_MODEL_RUNNER_OUTPUT
    )
    assert model._decoder_state_cache == {}

    _, one_shot_model = _code2wav(async_chunk=False)
    for req_id in chunks:
        one_shot = one_shot_model(
            input_ids=torch.tensor(_flat(frames[req_id])),
            runtime_additional_information=[
                {"meta": {"left_context_size": ref[req_id]}}
            ],
            seq_token_counts=[frames[req_id].numel()],
            request_ids=[req_id],
        ).multimodal_outputs["model_outputs"][0]
        pieces = streamed[req_id]
        audio = torch.cat(pieces)
        assert audio.shape == one_shot.shape
        assert [piece.numel() // 1920 for piece in pieces] == (
            [chunks[req_id][0] - ref[req_id]] + chunks[req_id][1:]
        )
        # Decoding in a padded batch rounds differently: 5e-4 at worst, measured.
        join = 0
        for piece in pieces[:-1]:
            join += piece.numel()
            around = slice(join - 1920, join + 1920)
            assert torch.allclose(audio[around], one_shot[around], atol=1e-3), (
                req_id,
                join // 1920,
            )
        assert torch.allclose(audio, one_shot, atol=1e-3)
        assert torch.corrcoef(torch.stack([audio, one_shot]))[0, 1] > 0.99999
