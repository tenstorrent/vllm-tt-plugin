# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speak tt-inference-server's TTS eval inputs on TT through vLLM-Omni, for scoring.

`--inputs` holds what `wer.json` and `clips.json` name: the WER texts, each spoken in
the `--ref-audio` voice, and one clip per speaker for speaker similarity. Speaker i says
speaker i+1's sentence twice, once cloned from its own clip and once cloned from speaker
i-1's: the cross-speaker baseline the eval scores a clone against. Clones are x-vector
only, the eval's `voice_only` mode. `score_tts.py` scores the wavs.

    python poc/qwen3-tts-omni/quality_tts.py \
        --model <Qwen3-TTS-12Hz-1.7B-Base snapshot> \
        --deploy-config poc/qwen3-tts-omni/deploy/tt_qwen3_tts.yaml \
        --ref-audio <clip.wav> --inputs <dir> --out-dir <dir>
"""

import argparse
import copy
import json
import os

from measure_tts import run
from vllm.sampling_params import RequestOutputKind
from vllm_omni.entrypoints.omni import Omni


def jobs(inputs, ref_audio):
    """(output name, text, reference clip) for every utterance the eval scores."""
    with open(os.path.join(inputs, "wer.json")) as f:
        for sample in json.load(f):
            yield f"wer/{sample['id']}.wav", sample["text"], ref_audio
    with open(os.path.join(inputs, "clips.json")) as f:
        clips = json.load(f)
    for index, clip in enumerate(clips):
        text = clips[(index + 1) % len(clips)]["text"]
        yield f"secs/{clip['speaker']}_clone.wav", text, clip["wav"]
        yield f"secs/{clip['speaker']}_cross.wav", text, clips[index - 1]["wav"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument(
        "--ref-audio", required=True, help="the voice the WER texts are spoken in"
    )
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--language", default="English")
    args = parser.parse_args()

    omni = Omni(
        model=args.model, deploy_config=args.deploy_config, trust_remote_code=True
    )
    # Streamed, as `measure_tts.py` reads it; the chunks are joined into one wav.
    params = copy.deepcopy(omni.default_sampling_params_list)
    params[-1].output_kind = RequestOutputKind.DELTA
    for subdir in ("wer", "secs"):
        os.makedirs(os.path.join(args.out_dir, subdir), exist_ok=True)
    for name, text, ref_audio in jobs(args.inputs, args.ref_audio):
        args.ref_audio = ref_audio
        _, _, total, audio_s, _ = run(
            omni, params, args, text, os.path.join(args.out_dir, name)
        )
        print(
            f"QUALITY {name} chars={len(text)} audio_s={audio_s:.2f} "
            f"total_s={total:.2f}",
            flush=True,
        )
    omni.close()


if __name__ == "__main__":
    main()
