# Qwen3-ASR under `vllm serve --omni` on TT

Qwen3-ASR runs as stage 0 of the upstream `qwen3_asr` pipeline, and
Qwen3-ForcedAligner-0.6B as stage 1 for word timestamps. Both need
vllm-project/vllm-omni#10609. Until it is in a release on this vLLM,
`docs/install-vllm-omni.sh` installs 0.32.0 with #10609 and three pooling-output
fixes from Omni main (`cleonidouTT/vllm-omni@qwen3-asr-poc-0.32`).

Transcriptions, one N150:

    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=200 VLLM_MAX_AUDIO_DECODE_DURATION_S=4000 \
    vllm serve Qwen/Qwen3-ASR-1.7B --omni --deploy-config poc/qwen3-asr-omni/deploy/tt_qwen3_asr.yaml

Word timestamps (`response_format=verbose_json`, `timestamp_granularities[]=word`),
ASR on device 0 and the aligner on device 1:

    vllm serve Qwen/Qwen3-ASR-1.7B --omni \
      --deploy-config poc/qwen3-asr-omni/deploy/tt_qwen3_asr_aligner.yaml \
      --forced-aligner-config poc/qwen3-asr-omni/deploy/tt_qwen3_forced_aligner.yaml \
      --forced-aligner-device 1

The two limits only matter for files over 25 MB or 600 s.
