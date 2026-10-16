# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Offline smoke run of the Qwen3-TTS talker on TT through vLLM-Omni.

One Base (voice clone, x-vector only) request through the `tt_qwen3_tts_talker`
pipeline: codebook-0 ids come back and stop at end-of-speech (2150), and the 16-code
frames come back as `codes.audio` [frames, 16].

    python poc/qwen3-tts-omni/smoke_talker.py --model <Qwen3-TTS-12Hz-1.7B-Base snapshot> \
        --deploy-config poc/qwen3-tts-omni/deploy/tt_qwen3_tts_talker.yaml \
        --ref-audio <clip.wav> --text "..."
"""

import argparse
import json
import os
import time

import soundfile
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    Qwen3TTSPromptEmbedsBuilder,
)

from vllm_tt_plugin.omni.pipelines import register_tt_pipelines

EOS = 2150


def tts_params(args):
    audio, rate = soundfile.read(args.ref_audio, dtype="float32", always_2d=True)
    return {
        "task_type": ["Base"],
        "text": [args.text],
        "language": [args.language],
        "ref_audio": [[audio.mean(axis=1).tolist(), int(rate)]],
        "x_vector_only_mode": [True],
    }


def prompt_len(args, info):
    """The placeholder length vLLM-Omni's server gives this request."""
    with open(os.path.join(args.model, "config.json")) as f:
        talker = json.load(f)["talker_config"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        info,
        task_type="Base",
        tokenize_prompt=tokenizer.encode,
        codec_language_id=talker.get("codec_language_id"),
        spk_is_dialect=talker.get("spk_is_dialect"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--text", default="Good morning. Today is a beautiful day.")
    parser.add_argument("--language", default="English")
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    register_tt_pipelines()
    info = tts_params(args)
    length = prompt_len(args, info)
    print(f"SMOKE prompt_len={length}", flush=True)

    started = time.time()
    omni = Omni(
        model=args.model, deploy_config=args.deploy_config, trust_remote_code=True
    )
    print(f"SMOKE engine_up_s={time.time() - started:.1f}", flush=True)
    sampling = SamplingParams(
        temperature=0.9,
        top_k=50,
        max_tokens=args.max_tokens,
        min_tokens=2,
        repetition_penalty=1.05,
        stop_token_ids=[EOS],
        detokenize=False,
    )
    started = time.time()
    outputs = omni.generate(
        [{"prompt_token_ids": [1] * length, "additional_information": info}], [sampling]
    )
    elapsed = time.time() - started
    for output in outputs:
        completion = output.outputs[0]
        ids = list(completion.token_ids)
        multimodal = getattr(output, "multimodal_output", None) or getattr(
            completion, "multimodal_output", None
        )
        codes = (multimodal or {}).get("codes", {})
        audio = codes.get("audio") if isinstance(codes, dict) else None
        shape = None if audio is None else tuple(audio.shape)
        print(
            f"SMOKE tokens={len(ids)} last={ids[-1] if ids else None} "
            f"finish={completion.finish_reason} codes_audio={shape} gen_s={elapsed:.1f}",
            flush=True,
        )
        print(f"SMOKE first_ids={ids[:12]}", flush=True)
    omni.close()


if __name__ == "__main__":
    main()
