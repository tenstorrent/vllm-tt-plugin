# Qwen3-ASR under `vllm serve --omni` on TT (P5, stage 0)

Stage 0 runs the TT Qwen3-ASR class on one N150 through the upstream `qwen3_asr`
pipeline. `/v1/audio/transcriptions` needs vllm-project/vllm-omni#10609 (or #5753):
without it vLLM-Omni never reports the `transcription` task.

Until that merges, run vLLM-Omni 0.32.0 with the two #10609 commits
(`efd3ab1b`, `7ad6eff9`) cherry-picked.

    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=200 VLLM_MAX_AUDIO_DECODE_DURATION_S=4000 \
    vllm serve Qwen/Qwen3-ASR-1.7B --omni --deploy-config poc/qwen3-asr-omni/deploy/tt_qwen3_asr.yaml

The two limits only matter for files over 25 MB or 600 s.
