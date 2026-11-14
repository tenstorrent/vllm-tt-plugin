# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Use the pinned checkpoint's own text/tool encoder with its HF tokenizer.

The checkpoint encoder is self-contained Python distributed with the model.
Verify its bytes before loading it; never substitute the V4 prompt format.
"""

import copy
import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path

CHECKPOINT = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
ENCODER_SHA256 = "502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1"


@lru_cache(maxsize=4)
def load_encoder(path):
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != ENCODER_SHA256:
        raise ValueError(f"DeepSeek-V4.1 checkpoint encoder hash mismatch: {path}")
    spec = importlib.util.spec_from_file_location(
        "deepseek_v41_checkpoint_encoder", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load checkpoint encoder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_chat(messages, *, encoder_path, tools=None, **kwargs):
    """Render with OpenAI tool schemas attached to the first system message."""
    conversation = copy.deepcopy(kwargs.get("conversation", messages))
    if tools:
        if not conversation or conversation[0].get("role") != "system":
            conversation.insert(0, {"role": "system", "content": ""})
        conversation[0]["tools"] = copy.deepcopy(tools)
    effort = kwargs.get("reasoning_effort")
    thinking = kwargs.get("thinking", False) or kwargs.get("enable_thinking", False)
    if effort == "none":
        thinking, effort = False, None
    return load_encoder(str(encoder_path)).encode_messages(
        conversation,
        thinking_mode="thinking" if thinking else "chat",
        drop_thinking=kwargs.get("drop_thinking", True),
        reasoning_effort=effort,
    )


def get_deepseek_v41_tokenizer(tokenizer, encoder_path):
    """Wrap only chat rendering; retain the checkpoint's token IDs and decoder."""
    load_encoder(str(encoder_path))
    wrapped = copy.copy(tokenizer)

    class _DeepseekV41Tokenizer(tokenizer.__class__):
        def apply_chat_template(self, messages=None, tools=None, **kwargs):
            prompt = encode_chat(
                messages, encoder_path=encoder_path, tools=tools, **kwargs
            )
            if not kwargs.get("tokenize", True):
                return prompt
            encode_kwargs = {
                key: kwargs[key]
                for key in ("truncation", "max_length")
                if key in kwargs
            }
            return self.encode(prompt, add_special_tokens=False, **encode_kwargs)

        def __reduce__(self):
            return get_deepseek_v41_tokenizer, (tokenizer, encoder_path)

    _DeepseekV41Tokenizer.__name__ = f"DSV41{tokenizer.__class__.__name__}"
    wrapped.__class__ = _DeepseekV41Tokenizer
    return wrapped


class DeepseekV41Tokenizer:
    @classmethod
    def from_pretrained(cls, name, *args, **kwargs):
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        from vllm.tokenizers.hf import get_cached_tokenizer

        local = Path(name) / "encoding/encoding.py"
        if local.is_file():
            encoder_path = local.resolve()
        else:
            if str(name) != CHECKPOINT:
                raise ValueError(
                    f"V4.1 tokenizer requires {CHECKPOINT} or its snapshot"
                )
            encoder_path = Path(
                hf_hub_download(
                    CHECKPOINT,
                    "encoding/encoding.py",
                    revision=REVISION,
                    cache_dir=kwargs.get("cache_dir"),
                    local_files_only=True,
                )
            )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(name, *args, **kwargs)
        return get_cached_tokenizer(
            get_deepseek_v41_tokenizer(tokenizer, str(encoder_path))
        )
