"""PoC: ``response_format=diarized_json`` for the ``qwen3_asr`` pipeline when its
forced-aligner stage is the TT ``Qwen3ForcedAlignerDiarizer``
(tenstorrent/vllm-tt-plugin#326).

vLLM-Omni has no diarization stage or per-request merge yet
(vllm-project/vllm-omni#11075), so these runtime patches of its transcription path
stand in for it:

- serving: a diarized request runs as ``verbose_json`` with word timestamps; each
  chunk's aligner request also carries the next 10 s of audio, so diarization
  windows can cross chunk boundaries; the chunks' speaker data are clustered once
  per request and the words grouped by speaker;
- ``asr2aligner``: appends that context to the chunk audio and asks the stage-1
  processor to diarize;
- the stage-1 pooling-output decoder and ``extract_word_timestamps``: accept the
  dict output ``{"logits" | "words_ms", "diar.*"}`` that carries the speaker data
  next to the word timestamps.
"""

from __future__ import annotations

import asyncio
import contextvars
import math
from types import SimpleNamespace
from typing import Any

import numpy as np

CONTEXT_SECONDS = 10.0
# Stage-0 prompt key holding the audio after the chunk; asr2aligner reads it.
CONTEXT_KEY = "tt_diarize_context"
CONTEXT_SAMPLES_KWARG = "diarize_context_samples"
DIARIZER_ARCHITECTURES = ("Qwen3ForcedAlignerDiarizer", "TTQwen3ForcedAlignerDiarizer")

# Per diarized request: the chunk prompts in order and the aligner result of each.
_request_state: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "tt_diarize_request", default=None
)


def _flat(data: Any) -> dict:
    """Pooling output dict, nested or dotted, as dotted keys."""
    out = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out.update({f"{key}.{k}": v for k, v in _flat(value).items()})
        else:
            out[key] = value
    return out


def _pooling_data(result: Any) -> Any:
    outs = getattr(result, "outputs", None)
    out = outs[0] if isinstance(outs, (list, tuple)) else outs
    return getattr(out, "data", None)


def _install_decoder() -> None:
    from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

    decode = OmniSchedulerMixin._maybe_decode_pooling_output
    if getattr(decode, "_tt_diarization", False):
        return

    def _maybe_decode_pooling_output(self, request, pooler_output):
        if isinstance(pooler_output, dict) and "logits" in pooler_output:
            rest = {k: v for k, v in pooler_output.items() if k != "logits"}
            return {"words_ms": decode(self, request, pooler_output["logits"]), **rest}
        return decode(self, request, pooler_output)

    _maybe_decode_pooling_output._tt_diarization = True
    OmniSchedulerMixin._maybe_decode_pooling_output = _maybe_decode_pooling_output


def _install_word_timestamps() -> None:
    import vllm_omni.entrypoints.openai.serving_transcription as serving
    import vllm_omni.utils.forced_aligner as fa_utils

    extract = fa_utils.extract_word_timestamps
    if getattr(extract, "_tt_diarization", False):
        return

    def extract_word_timestamps(res, text, language=None):
        data = _pooling_data(res)
        if isinstance(data, dict):
            words_ms = _flat(data).get("words_ms")
            if words_ms is None:
                raise ValueError(
                    f"aligner dict output has no words_ms: {sorted(_flat(data))}"
                )
            res = SimpleNamespace(outputs=[SimpleNamespace(data=words_ms)])
        return extract(res, text, language)

    extract_word_timestamps._tt_diarization = True
    fa_utils.extract_word_timestamps = extract_word_timestamps
    serving.extract_word_timestamps = extract_word_timestamps


def _install_asr2aligner() -> None:
    import vllm_omni.model_executor.stage_input_processors.forced_aligner as fa_proc

    asr2aligner = fa_proc.asr2aligner
    if getattr(asr2aligner, "_tt_diarization", False):
        return

    def asr2aligner_with_context(
        source_outputs,
        prompt=None,
        _requires_multimodal_data=False,
        _streaming_context=None,
    ):
        items = asr2aligner(
            source_outputs, prompt, _requires_multimodal_data, _streaming_context
        )
        finished = [
            i for i, src in enumerate(source_outputs) if getattr(src, "finished", False)
        ]
        if len(finished) != len(items):
            raise RuntimeError(
                f"asr2aligner returned {len(items)} prompts for {len(finished)} "
                "finished outputs"
            )
        for item, i in zip(items, finished):
            source = prompt[i] if isinstance(prompt, list) else prompt
            context = source.get(CONTEXT_KEY) if isinstance(source, dict) else None
            if context is None:
                continue
            waveform, sample_rate = item["multi_modal_data"]["audio"]
            item["multi_modal_data"] = {
                "audio": (
                    np.concatenate([waveform, context]).astype(np.float32),
                    sample_rate,
                )
            }
            # A constant value: vLLM builds and caches an HF processor per kwargs value.
            item["mm_processor_kwargs"] = {CONTEXT_SAMPLES_KWARG: int(len(context))}
        return items

    asr2aligner_with_context._tt_diarization = True
    fa_proc.asr2aligner = asr2aligner_with_context


def _wrap_generate(engine_client: Any) -> None:
    generate = engine_client.generate
    if getattr(generate, "_tt_diarization", False):
        return

    def diarizing_generate(prompt, params, request_id, *args, **kwargs):
        state = _request_state.get()
        if state is None:
            return generate(prompt, params, request_id, *args, **kwargs)
        index = len(state["prompts"])
        state["prompts"].append(prompt)

        async def chunk_results():
            # Iteration starts once every chunk's generator exists, so the next
            # chunks' audio is known.
            prompts, rate = state["prompts"], state["sample_rate"]
            following = [
                p["multi_modal_data"]["audio"][0] for p in prompts[index + 1 :]
            ]
            # Exactly 10 s, zero-padded. The last chunk gets none, which still marks it
            # for diarization.
            size = int(CONTEXT_SECONDS * rate) if following else 0
            context = np.zeros(size, dtype=np.float32)
            if following:
                audio = np.concatenate(following)[:size]
                context[: len(audio)] = audio
            prompt[CONTEXT_KEY] = context
            async for result in generate(prompt, params, request_id, *args, **kwargs):
                if getattr(result, "final_output_type", None) == "timestamps":
                    state["aligner"][index] = result
                yield result

        return chunk_results()

    diarizing_generate._tt_diarization = True
    engine_client.generate = diarizing_generate


def _speaker_segments(words, turns, separator: str) -> list[dict]:
    """Each word to the speaker whose turns overlap it most (else the nearest turn),
    then runs of one speaker's words to segments. Speakers are named in order of first
    appearance."""
    names: dict[int, str] = {}
    segments: list[dict] = []
    for word in words:
        best, best_overlap, best_gap = None, 0.0, math.inf
        for start, end, speaker in turns:
            overlap = min(end, word.end) - max(start, word.start)
            gap = max(start - word.end, word.start - end, 0.0)
            if overlap > best_overlap or (best_overlap <= 0 and gap < best_gap):
                best, best_overlap, best_gap = speaker, max(overlap, 0.0), gap
        name = names.setdefault(best, f"SPEAKER_{len(names):02d}")
        if segments and segments[-1]["speaker"] == name:
            segments[-1]["end"] = word.end
            segments[-1]["words"].append(word.word)
        else:
            segments.append(
                {
                    "speaker": name,
                    "start": word.start,
                    "end": word.end,
                    "words": [word.word],
                }
            )
    return [
        {
            "id": f"seg_{i}",
            "start": s["start"],
            "end": s["end"],
            "text": separator.join(s["words"]),
            "speaker": s["speaker"],
        }
        for i, s in enumerate(segments)
    ]


def _install_serving() -> None:
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionResponseDiarized,
        TranscriptionResponseVerbose,
    )
    from vllm_omni.entrypoints.openai.serving_transcription import (
        OmniOpenAIServingTranscription,
        _resolve_language_code,
        asr_inter_chunk_separator,
    )

    create = OmniOpenAIServingTranscription.create_transcription
    if getattr(create, "_tt_diarization", False):
        return

    async def create_transcription(self, audio_data, request, raw_request=None):
        if request.response_format != "diarized_json":
            return await create(self, audio_data, request, raw_request)
        if request.stream:
            return self.create_error_response(
                "diarized_json does not support streaming."
            )
        if not self.has_asr_aligner:
            return self.create_error_response(
                "diarized_json requires an ASR stage connected to a forced aligner."
            )
        # Other stage-1 models cannot take the context audio added below.
        aligner = self.engine_client.engine.stage_vllm_configs[1].model_config
        if aligner.architecture not in DIARIZER_ARCHITECTURES:
            return self.create_error_response(
                "diarized_json requires a diarizing forced-aligner stage "
                f"({', '.join(DIARIZER_ARCHITECTURES)}); stage 1 is "
                f"{aligner.architecture}."
            )
        from models.experimental.pyannote.reference.diarization import diarize

        _wrap_generate(self.engine_client)
        rate = int(self.asr_config.sample_rate)
        state = {"prompts": [], "aligner": {}, "sample_rate": rate}
        verbose_request = request.model_copy(
            update={
                "response_format": "verbose_json",
                "timestamp_granularities": ["word"],
            }
        )
        token = _request_state.set(state)
        try:
            response = await create(self, audio_data, verbose_request, raw_request)
        finally:
            _request_state.reset(token)
        if not isinstance(response, TranscriptionResponseVerbose):
            return response

        segmentations, starts, embeddings, offset = [], [], [], 0.0
        for index, prompt in enumerate(state["prompts"]):
            data = _pooling_data(state["aligner"].get(index))
            data = _flat(data) if isinstance(data, dict) else {}
            if "diar.segmentation" not in data:
                return self.create_error_response(
                    f"chunk {index}: the aligner stage returned no speaker data; "
                    "diarized_json needs the Qwen3ForcedAlignerDiarizer stage-1 "
                    "architecture."
                )
            segmentations.append(
                np.asarray(data["diar.segmentation"], dtype=np.float32)
            )
            embeddings.append(np.asarray(data["diar.embeddings"], dtype=np.float32))
            starts.append(np.asarray(data["diar.starts"], dtype=np.float64) + offset)
            offset += len(prompt["multi_modal_data"]["audio"][0]) / rate
        turns = await asyncio.to_thread(
            diarize,
            np.concatenate(segmentations),
            np.concatenate(starts),
            np.concatenate(embeddings),
        )
        code = _resolve_language_code(
            response.language, self.model_cls.supported_languages
        )
        separator = asr_inter_chunk_separator(code, self.model_cls.no_space_languages)
        return TranscriptionResponseDiarized(
            duration=response.duration,
            text=response.text,
            segments=_speaker_segments(response.words or [], turns, separator),
            usage={"type": "duration", "seconds": math.ceil(response.duration)},
        )

    create_transcription._tt_diarization = True
    OmniOpenAIServingTranscription.create_transcription = create_transcription


def install() -> None:
    _install_decoder()
    _install_word_timestamps()
    _install_asr2aligner()
    _install_serving()
