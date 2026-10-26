# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""How much a request arriving mid-stream stalls the requests already speaking.

Starts `--requests` copies of one prompt `--gap` seconds apart through vLLM-Omni's
async engine, streaming each. Per request, with playback starting at its first audio:

  TTFA    submit to the first audio
  stall   total time the player would wait for audio after that
  worst   the longest single wait
  start   the earliest playback start that never waits (as in `measure_tts.py`)

    python poc/qwen3-tts-omni/stagger_tts.py \
        --model <Qwen3-TTS-12Hz-1.7B-Base snapshot> \
        --deploy-config <deploy yaml with max_num_seqs >= --requests> \
        --ref-audio <clip.wav> [--requests 4] [--gap 1.5]
"""

import argparse
import asyncio
import copy
import time

from measure_tts import PROMPTS, _audio
from smoke_talker import prompt_len, tts_params
from vllm.sampling_params import RequestOutputKind
from vllm_omni.entrypoints.async_omni import AsyncOmni


async def speak(omni, params, prompt, request_id, delay, rate_out):
    await asyncio.sleep(delay)
    submitted = time.perf_counter()
    arrivals = []  # (seconds since submit, samples)
    async for output in omni.generate(
        prompt, request_id=request_id, sampling_params_list=params
    ):
        chunks, rate = _audio(output)
        count = sum(chunk.numel() for chunk in chunks)
        if count:
            arrivals.append((time.perf_counter() - submitted, count))
            rate_out[0] = rate
    return arrivals


def playback(arrivals, rate):
    """(TTFA, total stall, worst stall, stall-free start) for one request's chunks."""
    ttfa = arrivals[0][0]
    clock, stall, worst, start, received = ttfa, 0.0, 0.0, 0.0, 0
    for arrived, count in arrivals:
        wait = max(0.0, arrived - clock)
        stall, worst = stall + wait, max(worst, wait)
        clock = max(clock, arrived) + count / rate
        start = max(start, arrived - received / rate)
        received += count
    return ttfa, stall, worst, start


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--prompt", default="long", choices=sorted(PROMPTS))
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--gap", type=float, default=1.5)
    args = parser.parse_args()

    omni = AsyncOmni(
        model=args.model, deploy_config=args.deploy_config, trust_remote_code=True
    )
    params = copy.deepcopy(omni.default_sampling_params_list)
    params[-1].output_kind = RequestOutputKind.DELTA
    args.text = PROMPTS[args.prompt]
    info = tts_params(args)
    prompt = {
        "prompt_token_ids": [1] * prompt_len(args, info),
        "additional_information": info,
    }
    rate = [None]
    await speak(omni, params, prompt, "warmup", 0.0, rate)
    results = await asyncio.gather(
        *(
            speak(omni, params, prompt, f"r{i}", i * args.gap, rate)
            for i in range(args.requests)
        )
    )
    for index, arrivals in enumerate(results):
        ttfa, stall, worst, start = playback(arrivals, rate[0])
        audio = sum(count for _, count in arrivals) / rate[0]
        print(
            f"STAGGER request={index} arrives_s={index * args.gap:.1f} "
            f"ttfa_s={ttfa:.3f} stall_s={stall:.3f} worst_s={worst:.3f} "
            f"start_s={start:.3f} audio_s={audio:.2f} done_s={arrivals[-1][0]:.2f}",
            flush=True,
        )
    omni.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
