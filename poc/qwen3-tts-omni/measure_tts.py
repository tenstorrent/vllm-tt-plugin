# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Time to first audio and real-time ratio of Qwen3-TTS on TT through vLLM-Omni.

Runs one warmup request, then each prompt once, one at a time, streaming the
output. Per request:

  TTFA   submit to the first output that carries audio
  start  the earliest playback start that never runs out of audio: the largest
         gap between a chunk's arrival and the audio already received before it
  RTR    audio seconds / seconds from submit to the last output

``--log-stats`` adds Omni's per-stage tables (stage 0 TTFT and TPOT: the talker's
per-frame step time on the host clock).

    python poc/qwen3-tts-omni/measure_tts.py \
        --model <Qwen3-TTS-12Hz-1.7B-Base snapshot> \
        --deploy-config poc/qwen3-tts-omni/deploy/tt_qwen3_tts.yaml \
        --ref-audio <clip.wav>
"""

import argparse
import copy
import time

from smoke_talker import prompt_len, tts_params
from vllm.sampling_params import RequestOutputKind
from vllm_omni.entrypoints.omni import Omni

PROMPTS = {
    "short": "Good morning. Today is a beautiful day.",
    "medium": (
        "The quick brown fox jumps over the lazy dog. She sells sea shells by the "
        "sea shore, and the shells she sells are surely sea shells."
    ),
    "long": (
        "It was a bright cold day in April, and the clocks were striking thirteen. "
        "The hallway smelt of boiled cabbage and old rag mats. At one end of it a "
        "coloured poster, too large for indoor display, had been tacked to the wall. "
        "It depicted simply an enormous face, more than a metre wide: the face of a "
        "man of about forty-five, with a heavy black moustache and ruggedly handsome "
        "features. The flat was seven flights up, and the lift was seldom working. "
        "On each landing, opposite the lift shaft, the poster with the enormous face "
        "gazed from the wall. It was one of those pictures which are so contrived "
        "that the eyes follow you about when you move."
    ),
}


def _samples(output):
    """Audio samples carried by one streamed output, and its sample rate."""
    multimodal = output.outputs[0].multimodal_output or {}
    audio = multimodal.get("audio")
    if audio is None:
        return 0, None
    chunks = audio if isinstance(audio, list) else [audio]
    rate = multimodal.get("sr")
    rate = rate[-1] if isinstance(rate, list) else rate
    rate = int(rate.item() if hasattr(rate, "item") else rate)
    return sum(chunk.numel() for chunk in chunks), rate


def run(omni, params, args, text):
    args.text = text
    info = tts_params(args)
    prompt = {"prompt_token_ids": [1] * prompt_len(args, info)}
    prompt["additional_information"] = info
    submitted = time.perf_counter()
    first = last = None
    samples, rate, outputs, start = 0, None, 0, 0.0
    for output in omni.generate([prompt], params, py_generator=True, use_tqdm=False):
        count, output_rate = _samples(output)
        if not count:
            continue
        now = time.perf_counter()
        first = first if first is not None else now
        last = now
        outputs += 1
        rate = output_rate
        start = max(start, now - submitted - samples / rate)
        samples += count
    return first - submitted, start, last - submitted, samples / rate, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--log-stats", action="store_true")
    args = parser.parse_args()

    omni = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        trust_remote_code=True,
        log_stats=args.log_stats,
    )
    # Offline Omni returns only the finished request unless a stage asks for
    # deltas; the last stage streams each decoded chunk as it lands.
    params = copy.deepcopy(omni.default_sampling_params_list)
    params[-1].output_kind = RequestOutputKind.DELTA
    run(omni, params, args, PROMPTS["short"])
    for name, text in PROMPTS.items():
        ttfa, start, total, audio_s, outputs = run(omni, params, args, text)
        print(
            f"MEASURE prompt={name} chars={len(text)} ttfa_s={ttfa:.3f} "
            f"start_s={start:.3f} "
            f"total_s={total:.2f} audio_s={audio_s:.2f} rtr={audio_s / total:.3f} "
            f"audio_outputs={outputs}",
            flush=True,
        )
    omni.close()


if __name__ == "__main__":
    main()
