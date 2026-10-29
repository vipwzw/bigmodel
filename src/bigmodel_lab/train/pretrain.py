"""从零预训练训练循环：无状态批采样、混合精度、梯度累积/检查点、断点续训、监控与曲线。"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..budget import Budget, BudgetExceeded
from ..data.packing import load_packed
from ..manifest import RunContext
from ..models.gpt import GPT, PRESETS, GPTConfig
from .precision import (
    autocast_context,
    default_device,
    detect_autocast_dtype,
    driver_memory_gb,
    empty_cache,
    mps_available,
    sync,
)

logger = logging.getLogger(__name__)


@dataclass
class PretrainConfig:
    preset: str = "s"
    data_key: str = "pretrain_zh_bpe16384"
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    max_tokens: int = 2_000_000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    seed: int = 42
    grad_checkpoint: bool = False
    log_every: int = 10
    checkpoint_every: int = 200
    keep_last: int = 2
    eval_every: int = 200
    eval_batches: int = 8
    device: str = field(default_factory=default_device)
    profile: str = "sanity"
    resume_from: str | None = None

    def tokens_per_step(self, seq_len: int) -> int:
        return self.batch_size * seq_len * self.grad_accum


class PackedBatcher:
    """无状态批采样器：批次由全局 batch 序号唯一确定，天然支持续训对齐。"""

    def __init__(
        self,
        packed_dir: Path,
        split: str,
        *,
        seq_len: int,
        batch_size: int,
        seed: int,
        device: str = "mps",
    ):
        arrays = load_packed(Path(packed_dir) / f"{split}.npz")
        self.tokens = torch.from_numpy(arrays["tokens"].astype(np.int64))
        self.loss_mask = torch.from_numpy(arrays["loss_mask"].astype(np.float32))
        self.block = int(self.tokens.shape[1])
        self.seq_len = min(seq_len, self.block)
        self.chunks_per_row = max(1, self.block // self.seq_len)
        self.num_chunks = int(self.tokens.shape[0]) * self.chunks_per_row
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self._epoch_cache: tuple[int | None, torch.Tensor | None] = (None, None)

    def __len__(self) -> int:
        return self.num_chunks

    @property
    def total_input_tokens(self) -> int:
        return self.num_chunks * max(self.seq_len - 1, 1)

    def _permutation(self, epoch: int) -> torch.Tensor:
        cached_epoch, cached = self._epoch_cache
        if cached_epoch == epoch and cached is not None:
            return cached
        generator = torch.Generator().manual_seed(self.seed + epoch)
        permutation = torch.randperm(self.num_chunks, generator=generator)
        self._epoch_cache = (epoch, permutation)
        return permutation

    def batch(self, batch_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = batch_index * self.batch_size
        epoch = start // self.num_chunks
        offset = start % self.num_chunks
        indices = self._permutation(epoch)[offset : offset + self.batch_size]
        if indices.numel() < self.batch_size:
            extra = self._permutation(epoch + 1)[: self.batch_size - indices.numel()]
            indices = torch.cat([indices, extra])
        rows = torch.div(indices, self.chunks_per_row, rounding_mode="floor").tolist()
        columns = (indices % self.chunks_per_row).tolist()
        sequences = torch.stack(
            [
                self.tokens[row, column * self.seq_len : (column + 1) * self.seq_len]
                for row, column in zip(rows, columns, strict=True)
            ]
        )
        masks = torch.stack(
            [
                self.loss_mask[row, column * self.seq_len : (column + 1) * self.seq_len]
                for row, column in zip(rows, columns, strict=True)
            ]
        )
        inputs = sequences[:, :-1].to(self.device)
        targets = sequences[:, 1:].to(self.device)
        loss_mask = masks[:, 1:].to(self.device)
        return inputs, targets, loss_mask


class CheckpointManager:
    def __init__(self, directory: Path, keep_last: int = 2):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(self, *, step: int, model: GPT, optimizer, scheduler, config: PretrainConfig, tokens_seen: int, best_loss: float) -> Path:
        path = self.directory / f"ckpt_step{step:07d}.pt"
        state = {
            "step": step,
            "tokens_seen": tokens_seen,
            "best_loss": best_loss,
            "config": asdict(config),
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "mps": torch.mps.get_rng_state() if mps_available() else None,
            },
        }
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        self._prune(keep=path)
        return path

    def _prune(self, keep: Path) -> None:
        checkpoints = sorted(self.directory.glob("ckpt_step*.pt"))
        best = self.best()
        removable = [path for path in checkpoints if path != keep and path != best]
        while len([path for path in checkpoints if path.exists()]) > self.keep_last + (1 if best else 0):
            candidate = removable.pop(0) if removable else None
            if candidate is None:
                break
            candidate.unlink(missing_ok=True)
            checkpoints = [path for path in checkpoints if path.exists()]

    def best(self) -> Path | None:
        candidates = sorted(self.directory.glob("best.pt"))
        return candidates[0] if candidates else None

    def latest(self) -> Path | None:
        candidates = sorted(self.directory.glob("ckpt_step*.pt"))
        return candidates[-1] if candidates else None

    def save_best(self, *, step: int, model: GPT, optimizer, scheduler, config: PretrainConfig, tokens_seen: int, best_loss: float) -> Path:
        path = self.directory / "best.pt"
        state = {
            "step": step,
            "tokens_seen": tokens_seen,
            "best_loss": best_loss,
            "config": asdict(config),
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "mps": torch.mps.get_rng_state() if mps_available() else None,
            },
        }
        torch.save(state, path)
        return path


def _load_checkpoint(path: Path, model: GPT, optimizer, scheduler) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng = state.get("rng", {})
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("mps") is not None and mps_available():
        torch.mps.set_rng_state(rng["mps"])
    return state


def build_scheduler(optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def evaluate(model: GPT, batcher: PackedBatcher, *, batches: int, autocast_dtype, device: str) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index in range(batches):
            inputs, targets, loss_mask = batcher.batch(batch_index)
            with autocast_context(device, autocast_dtype):
                loss, _ = model(inputs, targets, loss_mask)
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


class Pretrainer:
    def __init__(
        self,
        config: PretrainConfig,
        *,
        ctx: RunContext,
        budget: Budget,
        packed_dir: Path,
    ):
        self.config = config
        self.ctx = ctx
        self.budget = budget
        self.packed_dir = packed_dir
        self.device = config.device if mps_available() else "cpu"
        self.model_config: GPTConfig = PRESETS[config.preset]
        self.seq_len = min(self.model_config.block_size, 1024)
        self.autocast_dtype, self.autocast_name = detect_autocast_dtype(self.device)
        self.metrics: list[dict[str, Any]] = []

    # ---------- 初始化 ----------
    def _build(self) -> tuple[GPT, torch.optim.Optimizer]:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        model = GPT(self.model_config).to(self.device)
        if self.config.grad_checkpoint:
            model.set_gradient_checkpointing(True)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )
        return model, optimizer

    # ---------- 训练 ----------
    def train(self) -> dict[str, Any]:
        config = self.config
        model, optimizer = self._build()
        train_batcher = PackedBatcher(
            self.packed_dir, "train", seq_len=self.seq_len, batch_size=config.batch_size, seed=config.seed, device=self.device
        )
        val_batcher = PackedBatcher(
            self.packed_dir, "val", seq_len=self.seq_len, batch_size=config.batch_size, seed=config.seed + 1, device=self.device
        )
        self.seq_len = train_batcher.seq_len
        max_steps = max(1, math.ceil(config.max_tokens / config.tokens_per_step(self.seq_len)))
        # 短程实验按总步数自适应缩短 warmup，避免全程都在预热
        effective_warmup = min(config.warmup_steps, max(1, max_steps // 10))
        scheduler = build_scheduler(optimizer, warmup_steps=effective_warmup, total_steps=max(max_steps, 1), min_lr_ratio=config.min_lr_ratio)
        checkpoints = CheckpointManager(self.ctx.checkpoints_dir / config.preset, keep_last=config.keep_last)

        start_step = 0
        tokens_seen = 0
        best_loss = float("inf")
        if config.resume_from:
            state = _load_checkpoint(Path(config.resume_from), model, optimizer, scheduler)
            start_step = int(state["step"])
            tokens_seen = int(state["tokens_seen"])
            best_loss = float(state.get("best_loss", float("inf")))
            logger.info("从检查点续训: %s（step=%d, tokens_seen=%d）", config.resume_from, start_step, tokens_seen)

        training_started = time.time()
        peak_memory = driver_memory_gb(self.device)
        last_loss = float("nan")
        for step in range(start_step, max_steps):
            try:
                self.budget.check()
            except BudgetExceeded:
                checkpoints.save(
                    step=step, model=model, optimizer=optimizer, scheduler=scheduler,
                    config=config, tokens_seen=tokens_seen, best_loss=best_loss,
                )
                raise
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for micro_step in range(config.grad_accum):
                batch_index = step * config.grad_accum + micro_step
                inputs, targets, loss_mask = train_batcher.batch(batch_index)
                try:
                    with autocast_context(self.device, self.autocast_dtype):
                        loss, _ = model(inputs, targets, loss_mask)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise RuntimeError(
                            "显存不足：可尝试减小 batch_size、启用 grad_checkpoint、或增大 grad_accum。"
                            f"当前 batch_size={config.batch_size}, grad_accum={config.grad_accum}, seq_len={self.seq_len}"
                        ) from exc
                    raise
                loss = loss / config.grad_accum
                loss.backward()
                accumulated_loss += float(loss.item())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            last_loss = accumulated_loss
            tokens_seen += config.tokens_per_step(self.seq_len)
            sync(self.device)
            step_seconds = time.perf_counter() - step_started
            peak_memory = max(peak_memory, driver_memory_gb(self.device))

            if not math.isfinite(last_loss):
                raise RuntimeError(f"训练损失非有限值（step={step}, loss={last_loss}）")

            if config.log_every and step % config.log_every == 0:
                record = {
                    "preset": config.preset,
                    "step": step,
                    "loss": round(last_loss, 6),
                    "lr": round(scheduler.get_last_lr()[0], 8),
                    "grad_norm": round(float(grad_norm), 6),
                    "tokens_seen": tokens_seen,
                    "tokens_per_second": round(config.tokens_per_step(self.seq_len) / max(step_seconds, 1e-9), 1),
                    "step_seconds": round(step_seconds, 4),
                    "peak_memory_gb": peak_memory,
                }
                self.metrics.append(record)
                self.ctx.log_metrics(**record)
                logger.info(
                    "[%s] step %d/%d loss %.4f lr %.2e tok/s %.0f",
                    config.preset, step, max_steps, last_loss, record["lr"], record["tokens_per_second"],
                )

            if config.eval_every and step > 0 and step % config.eval_every == 0:
                val_loss = evaluate(model, val_batcher, batches=config.eval_batches, autocast_dtype=self.autocast_dtype, device=self.device)
                self.ctx.log_metrics(preset=config.preset, kind="eval", step=step, val_loss=round(val_loss, 6))
                if val_loss < best_loss:
                    best_loss = val_loss
                    checkpoints.save_best(
                        step=step, model=model, optimizer=optimizer, scheduler=scheduler,
                        config=config, tokens_seen=tokens_seen, best_loss=best_loss,
                    )

            if config.checkpoint_every and step > 0 and step % config.checkpoint_every == 0:
                saved = checkpoints.save(
                    step=step + 1, model=model, optimizer=optimizer, scheduler=scheduler,
                    config=config, tokens_seen=tokens_seen, best_loss=best_loss,
                )
                logger.info("[%s] 检查点已保存: %s", config.preset, saved.name)

        final_val_loss = evaluate(model, val_batcher, batches=config.eval_batches, autocast_dtype=self.autocast_dtype, device=self.device)
        wall_seconds = time.time() - training_started
        final_checkpoint = checkpoints.save(
            step=max_steps, model=model, optimizer=optimizer, scheduler=scheduler,
            config=config, tokens_seen=tokens_seen, best_loss=min(best_loss, final_val_loss),
        )
        summary = {
            "preset": config.preset,
            "data_key": config.data_key,
            "device": self.device,
            "autocast": self.autocast_name,
            "seq_len": self.seq_len,
            "batch_size": config.batch_size,
            "grad_accum": config.grad_accum,
            "grad_checkpoint": config.grad_checkpoint,
            "steps": max_steps,
            "warmup_steps": effective_warmup,
            "tokens_seen": tokens_seen,
            "final_train_loss": round(last_loss, 6),
            "final_val_loss": round(final_val_loss, 6),
            "best_val_loss": round(min(best_loss, final_val_loss), 6),
            "wall_seconds": round(wall_seconds, 1),
            "tokens_per_second": round(tokens_seen / max(wall_seconds, 1e-9), 1),
            "peak_memory_gb": peak_memory,
            "params": self.model_config.num_params(),
            "flops_per_token": self.model_config.flops_per_token(),
            "checkpoint": str(final_checkpoint),
            "metrics": self.metrics,
        }
        empty_cache(self.device)
        self._save_curve(summary)
        return summary

    # ---------- 产物 ----------
    def _save_curve(self, summary: dict[str, Any]) -> None:
        metrics = summary["metrics"]
        if not metrics:
            return
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        steps = [record["step"] for record in metrics]
        losses = [record["loss"] for record in metrics]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(steps, losses, label="train loss")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].set_title(f"preset {summary['preset']} loss")
        axes[0].legend()
        axes[1].plot(steps, [record["lr"] for record in metrics], color="tab:orange", label="lr")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("lr")
        axes[1].set_title("learning rate schedule")
        axes[1].legend()
        fig.tight_layout()
        path = self.ctx.artifact_path(f"pretrain_{summary['preset']}_curves.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        self.ctx.add_artifact(path)
        metrics_path = self.ctx.artifact_path(f"pretrain_{summary['preset']}_metrics.jsonl")
        metrics_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in metrics) + "\n", encoding="utf-8"
        )
        self.ctx.add_artifact(metrics_path)
