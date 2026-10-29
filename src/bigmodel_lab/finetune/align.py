"""DPO 偏好优化：参考模型、偏好损失、稳定性监控与对齐对比。"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..budget import Budget, BudgetExceeded
from ..data.chat import build_dpo_example, collate_dpo
from ..manifest import RunContext
from ..train.precision import default_device, driver_memory_gb, empty_cache, mps_available, sync
from .sft import LoraConfigSpec, apply_lora, generate_responses, load_model_and_tokenizer, parameter_report

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    model_dir: Path  # Stage 5 的 SFT 模型（合并后的模型目录）
    output_dir: Path
    beta: float = 0.1
    max_steps: int = 20
    batch_size: int = 2
    grad_accum: int = 2
    max_seq_len: int = 384
    learning_rate: float = 5e-6
    warmup_steps: int = 2
    log_every: int = 5
    eval_every: int = 10
    eval_batches: int = 2
    save_every: int = 10
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = field(default_factory=default_device)
    profile: str = "sanity"
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    quality_prompts: list[str] = field(default_factory=list)
    quality_max_new_tokens: int = 40
    stop_on_unhealthy: bool = False


def sequence_logprobs(model, batch: dict[str, torch.Tensor], device: str) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    response_mask = batch["response_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = response_mask[:, 1:].float()
    return (token_logprobs * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    *,
    beta: float,
) -> dict[str, torch.Tensor]:
    policy_delta = policy_chosen - policy_rejected
    ref_delta = ref_chosen - ref_rejected
    implicit_reward = beta * (policy_delta - ref_delta)
    loss = -F.logsigmoid(implicit_reward).mean()
    approx_kl = ((policy_chosen - ref_chosen) + (policy_rejected - ref_rejected)).mean() / 2
    return {
        "loss": loss,
        "reward_margin": implicit_reward.mean().detach(),
        "accuracy": (implicit_reward > 0).float().mean().detach(),
        "approx_kl": approx_kl.detach(),
    }


def check_alignment_health(
    history: list[dict[str, Any]],
    *,
    margin_warn: float = 1.5,
    margin_hard: float = 3.0,
    quality_drop_ratio: float = 0.3,
) -> dict[str, Any]:
    """根据奖励间隔/近似 KL/生成质量历史检测过度优化迹象。"""
    warnings: list[str] = []
    status = "ok"
    margins = [item["reward_margin"] for item in history if item.get("reward_margin") is not None]
    kls = [abs(item["approx_kl"]) for item in history if item.get("approx_kl") is not None]
    qualities = [item["quality"] for item in history if item.get("quality") is not None]
    if margins:
        latest = margins[-1]
        if latest > margin_hard:
            status = "stop"
            warnings.append(f"奖励间隔 {latest:.3f} 超过硬阈值 {margin_hard}，可能过度优化")
        elif latest > margin_warn:
            status = "warn" if status == "ok" else status
            warnings.append(f"奖励间隔 {latest:.3f} 超过预警阈值 {margin_warn}")
    if kls and kls[-1] > 2.0:
        status = "stop" if status == "ok" else status
        warnings.append(f"近似 KL {kls[-1]:.3f} 过大，与参考模型偏离明显")
    if len(qualities) >= 2 and qualities[0] > 0:
        drop = (qualities[0] - qualities[-1]) / qualities[0]
        if drop > quality_drop_ratio:
            status = "stop"
            warnings.append(f"生成质量相对基线下降 {drop:.1%}（超过 {quality_drop_ratio:.0%}）")
    return {"status": status, "warnings": warnings, "margin": margins[-1] if margins else None, "approx_kl": kls[-1] if kls else None}


@torch.no_grad()
def generation_quality(model, tokenizer, prompts: list[str], *, device: str, max_new_tokens: int = 40) -> float:
    """启发式质量分：distinct-1（越高越不重复），用于监控退化。"""
    if not prompts:
        return 1.0
    responses = generate_responses(model, tokenizer, prompts, device=device, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)
    scores: list[float] = []
    for response in responses:
        if not response:
            scores.append(0.0)
            continue
        scores.append(len(set(response)) / len(response))
    return sum(scores) / len(scores)


class DPOTrainer:
    def __init__(
        self,
        config: DPOConfig,
        *,
        ctx: RunContext,
        budget: Budget,
        train_records: list[dict[str, Any]],
        val_records: list[dict[str, Any]],
    ):
        self.config = config
        self.ctx = ctx
        self.budget = budget
        self.device = config.device if mps_available() else "cpu"
        self.train_records = train_records
        self.val_records = val_records
        self.policy = None
        self.reference = None
        self.tokenizer = None
        self.metrics: list[dict[str, Any]] = []
        self.health_history: list[dict[str, Any]] = []
        self.quality_baseline: float | None = None

    def _build_datasets(self) -> tuple[DataLoader, DataLoader, dict[str, int]]:
        stats = {"records": len(self.train_records), "valid": 0, "dropped": {}}

        def prepare(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            examples: list[dict[str, Any]] = []
            for record in records:
                example, reason = build_dpo_example(self.tokenizer, record, max_seq_len=self.config.max_seq_len)
                if example is None:
                    stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1
                    continue
                examples.append(example)
                stats["valid"] += 1
            return examples

        train_examples = prepare(self.train_records)
        val_examples = prepare(self.val_records[: max(len(self.val_records), 32)]) if self.val_records else train_examples[:8]
        collator = lambda batch: collate_dpo(batch, pad_token_id=self.tokenizer.pad_token_id)  # noqa: E731
        generator = torch.Generator().manual_seed(self.config.seed)
        train_loader = DataLoader(train_examples, batch_size=self.config.batch_size, shuffle=True, generator=generator, collate_fn=collator)
        val_loader = DataLoader(val_examples, batch_size=self.config.batch_size, shuffle=False, collate_fn=collator)
        return train_loader, val_loader, stats

    def _prepare(self):
        policy_base, tokenizer = load_model_and_tokenizer(self.config.model_dir, device=self.device, dtype_name="float32")
        self.tokenizer = tokenizer
        self.policy = apply_lora(policy_base, self.config.lora).to(self.device)
        reference, _ = load_model_and_tokenizer(self.config.model_dir, device=self.device, dtype_name="float32")
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self.reference = reference
        return self._build_datasets()

    def _save(self, directory: Path, step: int, optimizer, scheduler) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(str(directory))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(directory))
        torch.save(
            {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": {"torch": torch.get_rng_state()}},
            directory / "trainer_state.pt",
        )
        return directory

    def _load_resume(self, path: Path, optimizer, scheduler) -> int:
        from peft import PeftModel  # noqa: PLC0415

        base = self.policy.get_base_model()
        self.policy = PeftModel.from_pretrained(base, str(path)).to(self.device)
        state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("rng", {}).get("torch") is not None:
            torch.set_rng_state(state["rng"]["torch"])
        return int(state["step"])

    def _evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        self.policy.eval()
        totals: dict[str, float] = {"loss": 0.0, "reward_margin": 0.0, "accuracy": 0.0, "approx_kl": 0.0}
        count = 0
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if index >= self.config.eval_batches:
                    break
                policy_chosen = sequence_logprobs(self.policy, batch["chosen"], self.device)
                policy_rejected = sequence_logprobs(self.policy, batch["rejected"], self.device)
                ref_chosen = sequence_logprobs(self.reference, batch["chosen"], self.device)
                ref_rejected = sequence_logprobs(self.reference, batch["rejected"], self.device)
                loss_dict = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=self.config.beta)
                for key in totals:
                    totals[key] += float(loss_dict[key].item())
                count += 1
        self.policy.train()
        return {key: value / max(count, 1) for key, value in totals.items()}

    def train(self, *, resume_from: str | None = None) -> dict[str, Any]:
        config = self.config
        train_loader, val_loader, dataset_stats = self._prepare()
        trainable = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
        effective_warmup = min(config.warmup_steps, max(1, config.max_steps // 10))

        def factor(step: int) -> float:
            if step < effective_warmup:
                return step / max(effective_warmup, 1)
            progress = min(max((step - effective_warmup) / max(config.max_steps - effective_warmup, 1), 0.0), 1.0)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
        start_step = self._load_resume(Path(resume_from), optimizer, scheduler) if resume_from else 0
        self.policy.train()
        if config.quality_prompts:
            self.quality_baseline = generation_quality(self.policy, self.tokenizer, config.quality_prompts, device=self.device, max_new_tokens=config.quality_max_new_tokens)
            logger.info("DPO 初始生成质量基线: %.4f", self.quality_baseline)
        peak_memory = driver_memory_gb(self.device)
        started = time.time()
        step = start_step
        tokens_seen = 0
        last_loss = float("nan")
        batch_iterator = iter(train_loader)

        def next_batch():
            nonlocal batch_iterator
            try:
                return next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(train_loader)
                return next(batch_iterator)

        while step < config.max_steps:
            try:
                self.budget.check()
            except BudgetExceeded:
                self._save(self.ctx.checkpoints_dir / "dpo", step, optimizer, scheduler)
                raise
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            step_tokens = 0
            for _ in range(config.grad_accum):
                batch = next_batch()
                policy_chosen = sequence_logprobs(self.policy, batch["chosen"], self.device)
                policy_rejected = sequence_logprobs(self.policy, batch["rejected"], self.device)
                with torch.no_grad():
                    ref_chosen = sequence_logprobs(self.reference, batch["chosen"], self.device)
                    ref_rejected = sequence_logprobs(self.reference, batch["rejected"], self.device)
                loss_dict = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=config.beta)
                (loss_dict["loss"] / config.grad_accum).backward()
                step_tokens += int(batch["chosen"]["attention_mask"].sum().item()) + int(batch["rejected"]["attention_mask"].sum().item())
                last_loss = float(loss_dict["loss"].item())
                last_margin = float(loss_dict["reward_margin"].item())
                last_accuracy = float(loss_dict["accuracy"].item())
                last_kl = float(loss_dict["approx_kl"].item())
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            sync(self.device)
            step_seconds = time.perf_counter() - step_started
            tokens_seen += step_tokens
            peak_memory = max(peak_memory, driver_memory_gb(self.device))
            if not math.isfinite(last_loss):
                raise RuntimeError(f"DPO 损失非有限值（step={step}, loss={last_loss}）")
            if config.log_every and step % config.log_every == 0:
                record = {
                    "stage": "dpo",
                    "step": step,
                    "loss": round(last_loss, 6),
                    "reward_margin": round(last_margin, 6),
                    "accuracy": round(last_accuracy, 4),
                    "approx_kl": round(last_kl, 6),
                    "lr": round(scheduler.get_last_lr()[0], 10),
                    "tokens_seen": tokens_seen,
                    "tokens_per_second": round(step_tokens / max(step_seconds, 1e-9), 1),
                    "peak_memory_gb": peak_memory,
                }
                self.metrics.append(record)
                self.ctx.log_metrics(**record)
                logger.info("[DPO] step %d/%d loss %.4f margin %.3f acc %.2f kl %.3f", step, config.max_steps, last_loss, last_margin, last_accuracy, last_kl)
            if config.eval_every and step > 0 and step % config.eval_every == 0:
                eval_metrics = self._evaluate(val_loader)
                quality = None
                if config.quality_prompts:
                    quality = generation_quality(self.policy, self.tokenizer, config.quality_prompts, device=self.device, max_new_tokens=config.quality_max_new_tokens)
                self.health_history.append({"step": step, **eval_metrics, "quality": quality})
                self.ctx.log_metrics(stage="dpo", kind="eval", step=step, **{key: round(value, 6) for key, value in eval_metrics.items()}, quality=quality)
                health = check_alignment_health(
                    [{"reward_margin": item["reward_margin"], "approx_kl": item["approx_kl"], "quality": item.get("quality")} for item in self.health_history]
                )
                self.ctx.log_metrics(stage="dpo", kind="monitor", step=step, monitor_status=health["status"], warnings=len(health["warnings"]))
                if health["status"] != "ok":
                    logger.warning("DPO 稳定性告警: %s", health)
                if health["status"] == "stop" and config.stop_on_unhealthy:
                    self._save(self.ctx.checkpoints_dir / "dpo", step, optimizer, scheduler)
                    raise RuntimeError(f"DPO 稳定性监控触发中止: {health['warnings']}")
            if config.save_every and step > 0 and step % config.save_every == 0:
                self._save(self.ctx.checkpoints_dir / "dpo" / f"step_{step}", step, optimizer, scheduler)
            step += 1

        final_eval = self._evaluate(val_loader)
        adapter_dir = self._save(self.ctx.artifact_path("dpo_adapter"), step, optimizer, scheduler)
        self.ctx.add_artifact(adapter_dir)
        health = check_alignment_health(
            [{"reward_margin": item["reward_margin"], "approx_kl": item["approx_kl"], "quality": item.get("quality")} for item in self.health_history]
        )
        if not self.health_history:
            health = check_alignment_health([{"reward_margin": last_margin, "approx_kl": last_kl, "quality": None}])
        report = parameter_report(self.policy)
        summary = {
            "profile": config.profile,
            "model_dir": str(config.model_dir),
            "beta": config.beta,
            "lora": {"r": config.lora.r, "alpha": config.lora.alpha},
            "steps": step,
            "dataset_stats": dataset_stats,
            "final_train_loss": round(last_loss, 6),
            "final_eval": {key: round(value, 6) for key, value in final_eval.items()},
            "tokens_seen": tokens_seen,
            "wall_seconds": round(time.time() - started, 1),
            "tokens_per_second": round(tokens_seen / max(time.time() - started, 1e-9), 1),
            "peak_memory_gb": peak_memory,
            "device": self.device,
            **report,
            "adapter": str(adapter_dir),
            "health": health,
            "health_history": self.health_history,
            "quality_baseline": self.quality_baseline,
            "metrics": self.metrics,
        }
        empty_cache(self.device)
        return summary


def write_alignment_comparison(
    *,
    sft_responses: list[str],
    dpo_responses: list[str],
    prompts: list[str],
    path: Path,
) -> None:
    lines = [
        "# 对齐效果对比（SFT vs DPO）",
        "",
        "| # | 提示 | SFT 输出 | DPO 输出 | 人工偏好（A/B/平） | 备注 |",
        "|---|---|---|---|---|---|",
    ]
    for index, (prompt, sft, dpo) in enumerate(zip(prompts, sft_responses, dpo_responses, strict=True)):
        lines.append(
            f"| {index + 1} | {prompt} | {sft.replace(chr(10), ' ')[:160]} | {dpo.replace(chr(10), ' ')[:160]} |  |  |"
        )
    lines.append("")
    lines.append("> 采样参数固定（temperature=0.7, top_p=0.9, 固定种子）；人工偏好列供填写。")
    path.write_text("\n".join(lines), encoding="utf-8")
