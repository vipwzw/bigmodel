"""预训练训练循环测试：loss 下降、检查点续训、超时保存。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bigmodel_lab.budget import Budget, BudgetExceeded
from bigmodel_lab.data.packing import save_packed
from bigmodel_lab.train.pretrain import PretrainConfig, Pretrainer

BLOCK = 64
VOCAB = 64


def _make_packed(tmp_path: Path, sequences: int = 16) -> Path:
    patterns = np.stack([(np.arange(BLOCK) + index) % VOCAB for index in range(sequences)]).astype(np.int32)
    loss_mask = np.ones_like(patterns, dtype=np.uint8)
    segment_ids = np.zeros_like(patterns, dtype=np.int32)
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    for split in ("train", "val"):
        save_packed(
            packed_dir,
            split,
            {"tokens": patterns, "loss_mask": loss_mask, "segment_ids": segment_ids},
            {"split": split, "block_size": BLOCK},
        )
    return packed_dir


class _FakeCtx:
    """最小 RunContext 替身：只记录指标。"""

    def __init__(self, tmp_path: Path):
        self.checkpoints_dir = tmp_path / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = tmp_path / "artifacts"
        self.artifacts.mkdir(exist_ok=True)
        self.metrics: list[dict] = []

    def log_metrics(self, **kwargs):
        self.metrics.append(kwargs)

    def artifact_path(self, name: str) -> Path:
        return self.artifacts / name

    def add_artifact(self, path):
        return path


def _config(**overrides) -> PretrainConfig:
    base = dict(
        preset="s",
        batch_size=1,
        grad_accum=1,
        lr=3e-3,
        warmup_steps=2,
        max_tokens=64 * 8,
        log_every=1,
        checkpoint_every=4,
        keep_last=1,
        eval_every=4,
        eval_batches=2,
        device="cpu",
        profile="sanity",
    )
    base.update(overrides)
    return PretrainConfig(**base)


def test_training_reduces_loss(tmp_path):
    packed = _make_packed(tmp_path)
    ctx = _FakeCtx(tmp_path)
    trainer = Pretrainer(_config(), ctx=ctx, budget=Budget("standard"), packed_dir=packed)
    summary = trainer.train()
    losses = [record["loss"] for record in summary["metrics"]]
    assert summary["tokens_seen"] == 64 * 8
    assert summary["steps"] == 8
    assert losses[-1] < losses[0], f"loss 未下降: {losses}"
    assert (ctx.checkpoints_dir / "s" / "best.pt").exists()


def test_checkpoint_resume_continuity(tmp_path):
    packed = _make_packed(tmp_path)
    ctx1 = _FakeCtx(tmp_path / "run1")
    config1 = _config(max_tokens=64 * 6, eval_every=None, keep_last=10)
    summary1 = Pretrainer(config1, ctx=ctx1, budget=Budget("standard"), packed_dir=packed).train()
    # 从周期检查点续训（而不是最终检查点，否则没有剩余步数可训）
    candidates = sorted((ctx1.checkpoints_dir / "s").glob("ckpt_step*.pt"))
    assert candidates, "应生成周期检查点"
    # 选一个未训练到总步数的中间检查点
    mid = [path for path in candidates if torch_load_step(path) < 6]
    assert mid, "应存在中间检查点"
    checkpoint = mid[-1]
    assert checkpoint.exists()

    resumed_step = int(torch_load_step(checkpoint))
    assert resumed_step > 0

    ctx2 = _FakeCtx(tmp_path / "run2")
    config2 = _config(max_tokens=64 * 6, eval_every=None, resume_from=str(checkpoint))
    summary2 = Pretrainer(config2, ctx=ctx2, budget=Budget("standard"), packed_dir=packed).train()

    # 续训从检查点记录的步数开始，且总步数一致
    first_record = summary2["metrics"][0]
    assert first_record["step"] >= resumed_step
    assert summary2["steps"] == summary1["steps"]
    assert summary2["tokens_seen"] == summary1["tokens_seen"]
    # 续训后第一段 loss 与原始训练同 step 的 loss 接近（RNG 与数据位置均已恢复）
    original = {record["step"]: record["loss"] for record in summary1["metrics"]}
    common_steps = [step for step in original if step in {record["step"] for record in summary2["metrics"]}]
    assert common_steps, "续训与原始训练应有重叠步数用于比对"
    step = max(common_steps)
    resumed_loss = next(record["loss"] for record in summary2["metrics"] if record["step"] == step)
    assert resumed_loss == pytest.approx(original[step], rel=0.05, abs=0.05)


def torch_load_step(path: Path) -> int:
    import torch  # noqa: PLC0415

    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state["step"])


def test_budget_exceeded_saves_checkpoint(tmp_path):
    packed = _make_packed(tmp_path)
    ctx = _FakeCtx(tmp_path)
    trainer = Pretrainer(_config(max_tokens=64 * 1000), ctx=ctx, budget=Budget("standard", max_minutes=0.0001), packed_dir=packed)
    with pytest.raises(BudgetExceeded):
        trainer.train()
    checkpoint_dir = ctx.checkpoints_dir / "s"
    assert list(checkpoint_dir.glob("ckpt_step*.pt")), "超时后应保留检查点"


def test_resume_missing_checkpoint_raises(tmp_path):
    packed = _make_packed(tmp_path)
    ctx = _FakeCtx(tmp_path)
    config = _config(resume_from=str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError):
        Pretrainer(config, ctx=ctx, budget=Budget("standard"), packed_dir=packed).train()


def test_metrics_written_with_required_fields(tmp_path):
    packed = _make_packed(tmp_path)
    ctx = _FakeCtx(tmp_path)
    summary = Pretrainer(_config(max_tokens=64 * 3, eval_every=None), ctx=ctx, budget=Budget("standard"), packed_dir=packed).train()
    record = summary["metrics"][0]
    for key in ("step", "loss", "lr", "grad_norm", "tokens_seen", "tokens_per_second", "peak_memory_gb"):
        assert key in record
    assert summary["final_val_loss"] > 0
    # 训练曲线与指标文件已生成
    assert any(path.name.endswith("_curves.png") for path in ctx.artifacts.iterdir())
    assert any(path.name.endswith("_metrics.jsonl") for path in ctx.artifacts.iterdir())
