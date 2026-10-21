# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Offline smoke run of Qwen3-TTS on TT through vLLM-Omni: talker, then Code2Wav.

One Base (voice clone, x-vector only) request through vLLM-Omni's `qwen3_tts`
pipeline; the audio is written to --out.

    python poc/qwen3-tts-omni/smoke_tts.py \
        --model <Qwen3-TTS-12Hz-1.7B-Base snapshot> \
        --deploy-config poc/qwen3-tts-omni/deploy/tt_qwen3_tts.yaml \
        --ref-audio <clip.wav> --text "..." --out out.wav
"""

import argparse
import time

import soundfile
import torch
from smoke_talker import prompt_len, tts_params
from vllm_omni.entrypoints.omni import Omni


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--text", default="Good morning. Today is a beautiful day.")
    parser.add_argument("--language", default="English")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    info = tts_params(args)
    length = prompt_len(args, info)
    print(f"SMOKE prompt_len={length}", flush=True)

    started = time.time()
    omni = Omni(
        model=args.model, deploy_config=args.deploy_config, trust_remote_code=True
    )
    print(f"SMOKE engine_up_s={time.time() - started:.1f}", flush=True)
    talker = omni.default_sampling_params_list[0]
    print(
        f"SMOKE talker_sampling temperature={talker.temperature} top_k={talker.top_k} "
        f"repetition_penalty={talker.repetition_penalty} "
        f"min_tokens={talker.min_tokens} max_tokens={talker.max_tokens}",
        flush=True,
    )
    started = time.time()
    outputs = omni.generate(
        [{"prompt_token_ids": [1] * length, "additional_information": info}]
    )
    elapsed = time.time() - started
    for output in outputs:
        multimodal = output.outputs[0].multimodal_output
        audio = multimodal["audio"]
        audio = torch.cat(audio, dim=-1) if isinstance(audio, list) else audio
        rate = multimodal["sr"]
        rate = rate[-1] if isinstance(rate, list) else rate
        rate = int(rate.item() if hasattr(rate, "item") else rate)
        samples = audio.float().cpu().numpy().reshape(-1)
        soundfile.write(args.out, samples, samplerate=rate, format="WAV")
        print(
            f"SMOKE stage={output.stage_id} samples={samples.size} sr={rate} "
            f"audio_s={samples.size / rate:.2f} gen_s={elapsed:.1f} "
            f"peak={abs(samples).max():.3f}",
            flush=True,
        )
    omni.close()


if __name__ == "__main__":
    main()
