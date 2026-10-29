"""DPO 损失与稳定性监控测试。"""

from __future__ import annotations

import math

import pytest
import torch

from bigmodel_lab.finetune.align import (
    check_alignment_health,
    dpo_loss,
    sequence_logprobs,
    write_alignment_comparison,
)


def test_dpo_loss_zero_preference_gap_when_policy_equals_reference():
    chosen = torch.tensor([-1.0, -2.0])
    rejected = torch.tensor([-3.0, -4.0])
    result = dpo_loss(chosen, rejected, chosen.clone(), rejected.clone(), beta=0.1)
    assert result["loss"].item() == pytest.approx(math.log(2), rel=1e-5)
    assert result["reward_margin"].item() == pytest.approx(0.0)
    assert result["accuracy"].item() == pytest.approx(0.0)
    assert result["approx_kl"].item() == pytest.approx(0.0)


def test_dpo_loss_rewards_policy_preference():
    policy_chosen = torch.tensor([0.0])
    policy_rejected = torch.tensor([-5.0])
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    result = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    expected = math.log(1 + math.exp(-0.5))
    assert result["loss"].item() == pytest.approx(expected, rel=1e-4)
    assert result["accuracy"].item() == pytest.approx(1.0)


def test_health_ok():
    report = check_alignment_health([
        {"reward_margin": 0.2, "approx_kl": 0.05, "quality": 0.8},
        {"reward_margin": 0.3, "approx_kl": 0.06, "quality": 0.78},
    ])
    assert report["status"] == "ok"
    assert report["warnings"] == []


def test_health_warns_on_large_margin():
    report = check_alignment_health([{"reward_margin": 2.0, "approx_kl": 0.2, "quality": 0.8}])
    assert report["status"] == "warn"
    assert report["warnings"]


def test_health_stops_on_huge_margin_or_kl():
    assert check_alignment_health([{"reward_margin": 4.0, "approx_kl": 0.2}])["status"] == "stop"
    assert check_alignment_health([{"reward_margin": 0.1, "approx_kl": 3.0}])["status"] == "stop"


def test_health_stops_on_quality_drop():
    report = check_alignment_health([
        {"reward_margin": 0.5, "approx_kl": 0.2, "quality": 0.8},
        {"reward_margin": 0.8, "approx_kl": 0.4, "quality": 0.4},
    ])
    assert report["status"] == "stop"
    assert any("质量" in warning for warning in report["warnings"])


def test_sequence_logprobs_respects_response_mask():
    class DummyOutput:
        def __init__(self, logits):
            self.logits = logits

    class DummyModel:
        def __call__(self, input_ids, attention_mask=None):
            batch, length = input_ids.shape
            return DummyOutput(torch.zeros(batch, length, 10))

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "response_mask": torch.tensor([[False, False, True, True]]),
    }
    logprobs = sequence_logprobs(DummyModel(), batch, "cpu")
    # 位置 2、3 的 token 来自前一个位置的 logits，每个 logprob = -log(10)
    assert logprobs.item() == pytest.approx(2 * (-math.log(10)), rel=1e-5)


def test_write_alignment_comparison(tmp_path):
    path = tmp_path / "side_by_side.md"
    write_alignment_comparison(
        sft_responses=["SFT 回答一", "SFT 回答二"],
        dpo_responses=["DPO 回答一", "DPO 回答二"],
        prompts=["问题一", "问题二"],
        path=path,
    )
    text = path.read_text(encoding="utf-8")
    assert "SFT 回答一" in text and "DPO 回答二" in text
    assert "人工偏好" in text
