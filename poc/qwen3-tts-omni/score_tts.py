# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Score `quality_tts.py`'s wavs the way tt-inference-server's TTS evals do.

WER (`tts_quality_test.py`): whisper-base transcribes each `wer/` wav; text is
lowercased and stripped of punctuation, and the word-level edit distance is divided by
the reference's word count, capped at 1. SECS (`tts_speaker_similarity_test.py`):
UniSpeech-SAT x-vectors, the cosine of each speaker's clip with its clone and with
the cross-speaker clone, and their margin. The eval passes at mean SECS >= 0.65 and
mean margin >= 0.05.

    python poc/qwen3-tts-omni/score_tts.py --inputs <dir> --wavs <dir> [--wavs <dir>]
"""

import argparse
import json
import os
import re

import librosa
import soundfile
import torch
from transformers import (
    AutoFeatureExtractor,
    UniSpeechSatForXVector,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

WHISPER = "openai/whisper-base"
SPEAKER_ENCODER = "microsoft/unispeech-sat-base-plus-sv"
RATE = 16000


def _words(text):
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split()).split()


def wer(reference, hypothesis):
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    row = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(hyp, 1):
            previous, row[j] = (
                row[j],
                min(row[j] + 1, row[j - 1] + 1, previous + (word != other)),
            )
    return min(row[-1] / len(ref), 1.0)


def _mono(path):
    audio, rate = soundfile.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=RATE)
    return audio


class Scorer:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained(WHISPER)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER).eval()
        self.extractor = AutoFeatureExtractor.from_pretrained(SPEAKER_ENCODER)
        self.encoder = UniSpeechSatForXVector.from_pretrained(SPEAKER_ENCODER).eval()

    @torch.no_grad()
    def transcribe(self, path):
        features = self.processor(
            _mono(path), sampling_rate=RATE, return_tensors="pt"
        ).input_features
        ids = self.whisper.generate(features)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    @torch.no_grad()
    def embed(self, path):
        inputs = self.extractor(_mono(path), sampling_rate=RATE, return_tensors="pt")
        return self.encoder(**inputs).embeddings[0]

    def secs(self, reference, path):
        cosine = torch.nn.functional.cosine_similarity(
            reference, self.embed(path), dim=-1
        )
        return float(cosine.clamp(-1.0, 1.0))


def score(scorer, inputs, wavs):
    with open(os.path.join(inputs, "wer.json")) as f:
        samples = json.load(f)
    errors = []
    for sample in samples:
        heard = scorer.transcribe(os.path.join(wavs, "wer", f"{sample['id']}.wav"))
        errors.append(wer(sample["text"], heard))
        print(f"  WER {errors[-1]:.3f} {sample['id']}: {heard}")
    with open(os.path.join(inputs, "clips.json")) as f:
        clips = json.load(f)
    rows = []
    for clip in clips:
        reference = scorer.embed(clip["wav"])
        base = os.path.join(wavs, "secs", clip["speaker"])
        rows.append(
            (
                scorer.secs(reference, f"{base}_clone.wav"),
                scorer.secs(reference, f"{base}_cross.wav"),
            )
        )
        print(f"  SECS {rows[-1][0]:.3f} cross {rows[-1][1]:.3f} {clip['speaker']}")
    mean = lambda values: sum(values) / len(values)  # noqa: E731
    secs, cross = mean([r[0] for r in rows]), mean([r[1] for r in rows])
    print(
        f"SCORE {wavs}: wer={mean(errors):.4f} max_wer={max(errors):.3f} "
        f"secs={secs:.3f} cross={cross:.3f} margin={secs - cross:+.3f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--wavs", required=True, action="append")
    args = parser.parse_args()
    scorer = Scorer()
    for wavs in args.wavs:
        score(scorer, args.inputs, wavs)


if __name__ == "__main__":
    main()
