# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Build `quality_tts.py`'s inputs the way tt-inference-server's TTS evals pick them.

WER texts: the first `--texts` rows of LibriTTS-R test.clean, `text_original`
(`tts_quality_test.py`). Speaker clips: one per speaker, first come first served,
for `--speakers` distinct speakers, `text_normalized`
(`_libritts.iter_speaker_clips` at tt-inference-server 2f2eb145).

    python poc/qwen3-tts-omni/build_quality_inputs.py --out <dir>
"""

import argparse
import itertools
import json
import os

from datasets import Audio, load_dataset


def stream():
    dataset = load_dataset(
        "blabble-io/libritts_r", "clean", split="test.clean", streaming=True
    )
    return dataset.cast_column("audio", Audio(decode=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--texts", type=int, default=10)
    parser.add_argument("--speakers", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)

    wer = []
    for idx, sample in enumerate(itertools.islice(stream(), args.texts)):
        wer.append(
            {
                "id": sample.get("id", f"libritts_{idx:05d}"),
                "text": sample.get("text_original", ""),
            }
        )
    with open(os.path.join(args.out, "wer.json"), "w") as f:
        json.dump(wer, f, indent=2)

    clips, seen = [], set()
    for sample in stream():
        speaker = str(sample.get("speaker_id"))
        wav = (sample.get("audio") or {}).get("bytes")
        if speaker in seen or not wav:
            continue
        seen.add(speaker)
        path = os.path.join(args.out, "clips", f"{speaker}.wav")
        with open(path, "wb") as f:
            f.write(wav)
        text = sample.get("text_normalized") or sample.get("text_original", "")
        clips.append({"speaker": speaker, "wav": path, "text": text})
        if len(clips) >= args.speakers:
            break
    with open(os.path.join(args.out, "clips.json"), "w") as f:
        json.dump(clips, f, indent=2)
    print(f"{len(wer)} WER texts, {len(clips)} speaker clips in {args.out}")


if __name__ == "__main__":
    main()
