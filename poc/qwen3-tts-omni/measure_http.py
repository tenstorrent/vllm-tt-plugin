# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""TTFA and RTR of Qwen3-TTS on TT through `vllm serve --omni`, at /v1/audio/speech.

Same prompts and voice-only cloning as `measure_tts.py`, so the numbers compare
with its offline ones. Per prompt, one request at a time:

  stream=True   raw pcm (stream_format="audio"): TTFA is the first byte of
                audio, RTR is audio seconds over the time to the last byte
  stream=False  one wav: TTFA is the whole response, RTR as above

    python poc/qwen3-tts-omni/measure_http.py --url http://127.0.0.1:8091 \
        --model <served name> --ref-audio <clip.wav>
"""

import argparse
import base64
import io
import time

import httpx
import soundfile
from measure_tts import PROMPTS

PCM_RATE = 24000
PCM_BYTES = 2


def body(args, text, stream):
    with open(args.ref_audio, "rb") as f:
        clip = "data:audio/wav;base64," + base64.b64encode(f.read()).decode()
    request = {
        "model": args.model,
        "input": text,
        "task_type": "Base",
        "language": "English",
        "ref_audio": clip,
        "x_vector_only_mode": True,
    }
    if stream:
        request.update(stream=True, stream_format="audio", response_format="pcm")
    else:
        request.update(stream=False, response_format="wav")
    return request


def run(client, args, text, stream):
    url = f"{args.url}/v1/audio/speech"
    start = time.perf_counter()
    if stream:
        first, size = None, 0
        with client.stream("POST", url, json=body(args, text, True)) as response:
            response.raise_for_status()
            for piece in response.iter_bytes():
                if piece and first is None:
                    first = time.perf_counter() - start
                size += len(piece)
        total = time.perf_counter() - start
        return first, total, size / PCM_BYTES / PCM_RATE
    response = client.post(url, json=body(args, text, False))
    response.raise_for_status()
    total = time.perf_counter() - start
    audio, rate = soundfile.read(io.BytesIO(response.content))
    return total, total, len(audio) / rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    with httpx.Client(timeout=600) as client:
        run(client, args, PROMPTS["short"], True)
        for stream in (True, False):
            for name, text in PROMPTS.items():
                for repeat in range(args.repeats):
                    ttfa, total, audio = run(client, args, text, stream)
                    print(
                        f"HTTP stream={stream} prompt={name} chars={len(text)} "
                        f"repeat={repeat} ttfa_s={ttfa:.3f} total_s={total:.2f} "
                        f"audio_s={audio:.2f} rtr={audio / total:.3f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
