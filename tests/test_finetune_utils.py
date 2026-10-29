"""微调工具函数测试（不加载真实模型）。"""

from __future__ import annotations

import torch.nn as nn

from bigmodel_lab.finetune.sft import (
    LoraConfigSpec,
    parameter_report,
    probe_qlora,
    write_generation_comparison,
)


def test_parameter_report():
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    report = parameter_report(model)
    assert report["total_params"] == 4 * 8 + 8 + 8 * 2 + 2
    assert report["trainable_params"] == report["total_params"]
    assert report["trainable_ratio"] == 1.0

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen = parameter_report(model)
    assert frozen["trainable_params"] == 0


def test_lora_config_defaults():
    spec = LoraConfigSpec()
    assert spec.r == 16
    assert spec.alpha == 32
    assert "q_proj" in spec.target_modules


def test_qlora_probe_has_fallback():
    report = probe_qlora()
    assert "available" in report
    if not report["available"]:
        assert report["fallback"]


def test_write_generation_comparison(tmp_path):
    path = tmp_path / "compare.md"
    write_generation_comparison(
        base_responses=["基座回答一", "基座回答二"],
        tuned_responses=["微调回答一", "微调回答二"],
        prompts=["问题一", "问题二"],
        path=path,
    )
    text = path.read_text(encoding="utf-8")
    assert "基座回答一" in text
    assert "微调回答二" in text
    assert text.count("|") > 10
