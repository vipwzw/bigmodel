"""省内存配置测试（不加载真实模型、不需要 GPU/bitsandbytes）。

覆盖：
- SFTConfig 的新增省内存字段与默认值（保持旧行为：dtype=None → fp32）；
- attn_implementation 缺 flash-attn 时自动回落 sdpa；
- 量化配置在缺少 bitsandbytes 时给出明确错误，非法量化值直接 ValueError。
"""

from __future__ import annotations

import importlib.util

import pytest

from bigmodel_lab.finetune.sft import (
    SFTConfig,
    _bnb_quantization_config,
    _resolve_attn_implementation,
    _torch_dtype,
)


def _config(**overrides) -> SFTConfig:
    base = {"model_dir": "/tmp/model", "output_dir": "/tmp/out"}
    base.update(overrides)
    return SFTConfig(**base)


def test_default_memory_options_preserve_old_behavior():
    config = _config()
    assert config.dtype is None
    assert config.base_dtype_name() == "float32"
    assert config.quantization is None
    assert config.gradient_checkpointing is False
    assert config.attn_implementation is None


def test_bf16_qlora_profile_options():
    config = _config(
        dtype="bfloat16",
        quantization="4bit",
        gradient_checkpointing=True,
        attn_implementation="sdpa",
    )
    assert config.base_dtype_name() == "bfloat16"
    assert config.quantization == "4bit"
    assert config.gradient_checkpointing is True


def test_resolve_attn_implementation_fallback():
    assert _resolve_attn_implementation(None) is None
    assert _resolve_attn_implementation("sdpa") == "sdpa"
    resolved = _resolve_attn_implementation("flash_attention_2")
    assert resolved in {"flash_attention_2", "sdpa"}
    if importlib.util.find_spec("flash_attn") is None:
        assert resolved == "sdpa"


def test_invalid_quantization_rejected():
    with pytest.raises(ValueError, match="4bit / 8bit"):
        _bnb_quantization_config("3bit", "bfloat16")


@pytest.mark.skipif(
    importlib.util.find_spec("bitsandbytes") is not None,
    reason="本机已安装 bitsandbytes",
)
def test_missing_bitsandbytes_gives_clear_error():
    with pytest.raises(RuntimeError, match="bitsandbytes"):
        _bnb_quantization_config("4bit", "bfloat16")


def test_torch_dtype_mapping():
    import torch

    assert _torch_dtype("bfloat16") is torch.bfloat16
    assert _torch_dtype("float16") is torch.float16
    assert _torch_dtype("float32") is torch.float32
