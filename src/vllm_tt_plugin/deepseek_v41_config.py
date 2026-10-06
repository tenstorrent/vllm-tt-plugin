# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""HF config of DeepSeek-V4.1 for the pinned vLLM, which predates the model.

The checkpoint's ``config.json`` (``model_type: deepseek_v41``, architecture
``DeepseekV41ForCausalLM``) nests the text model under ``text_config`` and the
vision tower under ``vision_config``; neither the pinned transformers nor the
pinned vLLM knows the type. This flattens the text fields onto the top level, as
upstream vLLM's ``transformers_utils/configs/deepseek_v41.py`` does (v0.30), so
``ModelConfig`` finds ``num_hidden_layers``, ``num_attention_heads``,
``num_key_value_heads``, ``head_dim``, ``vocab_size`` and
``max_position_embeddings`` where it looks for them. ``register_deepseek_v41_config``
puts it in vLLM's config registry ahead of ``AutoConfig``.
"""

from typing import Any

from transformers import PretrainedConfig


class DeepseekV41Config(PretrainedConfig):
    model_type = "deepseek_v41"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        text_config = dict(text_config or {})
        vision_config = dict(vision_config or {})
        # rope_scaling is a property in transformers v5 (backed by rope_parameters);
        # keep it aside and restore after super().__init__ re-standardizes rope params
        rope_scaling = text_config.pop("rope_scaling", None)
        for key, value in text_config.items():
            if key == "model_type":
                continue
            if isinstance(getattr(type(self), key, None), property):
                continue
            setattr(self, key, value)
        super().__init__(**kwargs)
        if rope_scaling is not None:
            self.rope_parameters = rope_scaling
        quant_cfg = getattr(self, "quantization_config", None) or {}
        if not hasattr(self, "expert_dtype") and "expert_dtype" in quant_cfg:
            self.expert_dtype = quant_cfg["expert_dtype"]
        self.vision_n_layers = vision_config.get("num_hidden_layers", 0)
        self.vision_dim = vision_config.get("hidden_size", 1024)
        self.vision_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_inter_dim = vision_config.get("intermediate_size", 2816)
        self.vision_patch_size = vision_config.get("patch_size", 14)
        self.vision_rope_theta = vision_config.get("rope_theta", 10000.0)
        self.vision_downsample_ratio = vision_config.get("downsample_ratio", 3)
        self.vision_max_n_token = vision_config.get("max_image_tokens", 1024)
        self.vision_min_pixels = vision_config.get("min_pixels", 295936)
        self.vision_max_wh_ratio = vision_config.get("max_wh_ratio")


def register_deepseek_v41_config() -> None:
    """Make vLLM's config parser use ``DeepseekV41Config`` for ``model_type``
    ``deepseek_v41``."""
    from vllm.transformers_utils import config as vllm_config

    registry = vllm_config._CONFIG_REGISTRY
    if "deepseek_v41" not in registry:
        registry["deepseek_v41"] = DeepseekV41Config
