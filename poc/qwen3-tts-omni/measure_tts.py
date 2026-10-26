# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Time to first audio and real-time ratio of Qwen3-TTS on TT through vLLM-Omni.

Runs one warmup request, then each prompt once, one at a time, streaming the
output. Per request:

  TTFA   submit to the first output that carries audio
  start  the earliest playback start that never runs out of audio: the largest
         gap between a chunk's arrival and the audio already received before it
  RTR    audio seconds / seconds from submit to the last output

``--concurrency N`` submits N copies of each prompt at once instead, and reports the
worst TTFA and start across them and the audio all N made per second of wall time.

``--log-stats`` adds Omni's per-stage tables (stage 0 TTFT and TPOT: the talker's
per-frame step time on the host clock).

    python poc/qwen3-tts-omni/measure_tts.py \
        --model <Qwen3-TTS-12Hz-1.7B-Base snapshot> \
        --deploy-config poc/qwen3-tts-omni/deploy/tt_qwen3_tts.yaml \
        --ref-audio <clip.wav>
"""

import argparse
import copy
import os
import time

import soundfile
import torch
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


def _audio(output):
    """The audio chunks one streamed output carries, and their sample rate."""
    multimodal = output.outputs[0].multimodal_output or {}
    audio = multimodal.get("audio")
    if audio is None:
        return [], None
    rate = multimodal.get("sr")
    rate = rate[-1] if isinstance(rate, list) else rate
    rate = int(rate.item() if hasattr(rate, "item") else rate)
    return (audio if isinstance(audio, list) else [audio]), rate


def run(omni, params, args, text, out=None):
    args.text = text
    info = tts_params(args)
    prompt = {"prompt_token_ids": [1] * prompt_len(args, info)}
    prompt["additional_information"] = info
    submitted = time.perf_counter()
    first = last = None
    samples, rate, outputs, start = 0, None, 0, 0.0
    received = []
    for output in omni.generate([prompt], params, py_generator=True, use_tqdm=False):
        chunks, output_rate = _audio(output)
        count = sum(chunk.numel() for chunk in chunks)
        if not count:
            continue
        received += chunks
        now = time.perf_counter()
        first = first if first is not None else now
        last = now
        outputs += 1
        rate = output_rate
        start = max(start, now - submitted - samples / rate)
        samples += count
    if out:
        audio = torch.cat([chunk.reshape(-1) for chunk in received]).float()
        soundfile.write(out, audio.cpu().numpy(), samplerate=rate, format="WAV")
    return first - submitted, start, last - submitted, samples / rate, outputs


def run_together(omni, params, args, texts):
    """Submit every text at once. Returns (TTFA, start, audio seconds) per text, in
    order, and the seconds until the last one finished."""
    prompts = []
    for text in texts:
        args.text = text
        info = tts_params(args)
        prompts.append(
            {
                "prompt_token_ids": [1] * prompt_len(args, info),
                "additional_information": info,
            }
        )
    submitted = time.perf_counter()
    first, start, samples, rate = {}, {}, {}, None
    for output in omni.generate(prompts, params, py_generator=True, use_tqdm=False):
        chunks, output_rate = _audio(output)
        count = sum(chunk.numel() for chunk in chunks)
        if not count:
            continue
        # Omni names the i-th prompt's request "i_<uuid>".
        index = int(str(output.request_id).split("_", 1)[0])
        now = time.perf_counter() - submitted
        rate = output_rate
        first.setdefault(index, now)
        start[index] = max(start.get(index, 0.0), now - samples.get(index, 0) / rate)
        samples[index] = samples.get(index, 0) + count
    total = time.perf_counter() - submitted
    return [(first[i], start[i], samples[i] / rate) for i in range(len(texts))], total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--log-stats", action="store_true")
    parser.add_argument("--out-dir", help="write each prompt's streamed audio here")
    parser.add_argument("--concurrency", type=int, default=1)
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
    if args.concurrency > 1:
        for name, text in PROMPTS.items():
            rows, total = run_together(omni, params, args, [text] * args.concurrency)
            audio_s = sum(row[2] for row in rows)
            print(
                f"MEASURE prompt={name} concurrency={args.concurrency} "
                f"ttfa_s={max(row[0] for row in rows):.3f} "
                f"start_s={max(row[1] for row in rows):.3f} total_s={total:.2f} "
                f"audio_s={audio_s:.2f} rtr_total={audio_s / total:.3f}",
                flush=True,
            )
        omni.close()
        return
    for name, text in PROMPTS.items():
        out = args.out_dir and os.path.join(args.out_dir, f"{name}.wav")
        ttfa, start, total, audio_s, outputs = run(omni, params, args, text, out)
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
