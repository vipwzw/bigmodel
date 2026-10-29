"""SFT 微调：Qwen3 小模型 + LoRA，含掩码自检、检查点续训、生成对比与合并导出。"""

from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..budget import Budget, BudgetExceeded
from ..data.chat import IGNORE_INDEX, SFTDataset, collate_sft
from ..manifest import RunContext
from ..train.precision import default_device, driver_memory_gb, empty_cache, mps_available, sync

logger = logging.getLogger(__name__)

IGNORE = IGNORE_INDEX


@dataclass
class LoraConfigSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class SFTConfig:
    model_dir: Path
    output_dir: Path
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    max_steps: int = 40
    batch_size: int = 2
    grad_accum: int = 4
    max_seq_len: int = 512
    learning_rate: float = 1e-4
    warmup_steps: int = 5
    log_every: int = 5
    eval_every: int = 20
    eval_batches: int = 4
    save_every: int = 0
    keep_last: int = 2
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    seed: int = 42
    device: str = field(default_factory=default_device)
    profile: str = "sanity"
    resume_from: str | None = None
    dtype: str = "float32"
    autocast: bool = True
    max_samples: int | None = None


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_model_and_tokenizer(
    model_dir: Path,
    *,
    device: str,
    dtype_name: str = "bfloat16",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"本地模型目录不存在: {model_dir}；请先运行 lab run 5 自动下载，或手动执行 "
            f"modelscope download --repo-type model <model_id> --local-dir {model_dir}"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=_torch_dtype(dtype_name),
    )
    model.config.use_cache = False
    model.to(device)
    return model, tokenizer


def parameter_report(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_params": total, "trainable_params": trainable, "trainable_ratio": round(trainable / total, 6) if total else 0.0}


def apply_lora(model, spec: LoraConfigSpec):
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415

    config = LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=list(spec.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def probe_qlora() -> dict[str, Any]:
    try:
        import bitsandbytes as bnb  # noqa: PLC0415

        return {
            "available": True,
            "version": getattr(bnb, "__version__", "unknown"),
            "note": "bitsandbytes 已安装；MPS 上的 4bit 训练仍属实验性，需实测验证",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"未安装 bitsandbytes（{type(exc).__name__}: {exc}）",
            "fallback": "主路径使用 bf16 LoRA；模型规模在 48GB 统一内存内无需 4bit 量化即可训练"
                        "（Qwen3-0.6B/1.7B LoRA 均可直接跑）",
        }


def _build_loader(
    records: list[dict[str, Any]],
    tokenizer,
    *,
    config: SFTConfig,
    shuffle: bool,
) -> tuple[DataLoader, SFTDataset]:
    selected = records[: config.max_samples] if config.max_samples else records
    dataset = SFTDataset(selected, tokenizer, max_seq_len=config.max_seq_len)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda batch: collate_sft(batch, pad_token_id=tokenizer.pad_token_id),
        drop_last=False,
    )
    return loader, dataset


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, *, batches: int, device: str, autocast: bool = True) -> float:
    model.eval()
    losses: list[float] = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        context = torch.autocast(device_type="mps", dtype=torch.bfloat16) if autocast and device == "mps" else nullcontext()
        with context:
            outputs = model(**batch)
        losses.append(float(outputs.loss.item()))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


class SFTTrainer:
    def __init__(
        self,
        config: SFTConfig,
        *,
        ctx: RunContext,
        budget: Budget,
        train_records: list[dict[str, Any]],
        val_records: list[dict[str, Any]],
        model=None,
        tokenizer=None,
    ):
        self.config = config
        self.ctx = ctx
        self.budget = budget
        self.device = config.device if mps_available() else "cpu"
        self.train_records = train_records
        self.val_records = val_records
        self.model = model
        self.tokenizer = tokenizer
        self.metrics: list[dict[str, Any]] = []

    def _prepare(self) -> tuple[DataLoader, DataLoader]:
        if self.model is None or self.tokenizer is None:
            base_model, tokenizer = load_model_and_tokenizer(
                self.config.model_dir, device=self.device, dtype_name=self.config.dtype
            )
            self.model = base_model
            self.tokenizer = tokenizer
        self.model = apply_lora(self.model, self.config.lora).to(self.device)
        train_loader, self.train_dataset = _build_loader(self.train_records, self.tokenizer, config=self.config, shuffle=True)
        val_loader, self.val_dataset = _build_loader(self.val_records, self.tokenizer, config=self.config, shuffle=False)
        return train_loader, val_loader

    def _forward_context(self):
        if self.config.autocast and self.device == "mps":
            return torch.autocast(device_type="mps", dtype=torch.bfloat16)
        return nullcontext()

    def _save_adapter(self, directory: Path, step: int, optimizer, scheduler) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(directory))
        self.tokenizer.save_pretrained(str(directory))
        state = {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": {"torch": torch.get_rng_state()},
        }
        torch.save(state, directory / "trainer_state.pt")
        return directory

    def _load_resume(self, path: Path, optimizer, scheduler) -> int:
        from peft import PeftModel  # noqa: PLC0415

        if not path.exists():
            raise FileNotFoundError(f"续训目录不存在: {path}")
        peft_model = PeftModel.from_pretrained(self.model.get_base_model(), str(path))
        self.model = peft_model.to(self.device)
        state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("rng", {}).get("torch") is not None:
            torch.set_rng_state(state["rng"]["torch"])
        return int(state["step"])

    def train(self) -> dict[str, Any]:
        config = self.config
        train_loader, val_loader = self._prepare()
        logger.info(
            "SFT 数据: train %d / val %d（原始 records %d / %d），tokens 统计: %s",
            len(self.train_dataset), len(self.val_dataset), len(self.train_records), len(self.val_records),
            self.train_dataset.stats,
        )
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
        effective_warmup = min(config.warmup_steps, max(1, config.max_steps // 10))

        def factor(step: int) -> float:
            if step < effective_warmup:
                return step / max(effective_warmup, 1)
            progress = min(max((step - effective_warmup) / max(config.max_steps - effective_warmup, 1), 0.0), 1.0)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
        baseline_val_loss = evaluate_loss(self.model, val_loader, batches=config.eval_batches, device=self.device, autocast=config.autocast)
        self.ctx.log_metrics(stage="sft", kind="eval", step=0, val_loss=round(baseline_val_loss, 6))
        logger.info("SFT 初始验证损失: %.4f", baseline_val_loss)
        start_step = 0
        if config.resume_from:
            start_step = self._load_resume(Path(config.resume_from), optimizer, scheduler)
            logger.info("SFT 从 %s 续训，起始 step=%d", config.resume_from, start_step)

        self.model.train()
        tokens_seen = 0
        peak_memory = driver_memory_gb(self.device)
        started = time.time()
        step = start_step
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
                self._save_adapter(self.ctx.checkpoints_dir / "sft", step, optimizer, scheduler)
                raise
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            step_tokens = 0
            step_loss_sum = 0.0
            for _ in range(config.grad_accum):
                batch = {key: value.to(self.device) for key, value in next_batch().items()}
                with self._forward_context():
                    outputs = self.model(**batch)
                loss = outputs.loss
                (loss / config.grad_accum).backward()
                step_loss_sum += float(loss.item())
                step_tokens += int(batch["attention_mask"].sum().item())
            last_loss = step_loss_sum / config.grad_accum
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            sync(self.device)
            step_seconds = time.perf_counter() - step_started
            tokens_seen += step_tokens
            peak_memory = max(peak_memory, driver_memory_gb(self.device))
            if not math.isfinite(last_loss):
                raise RuntimeError(f"SFT 损失非有限值（step={step}, loss={last_loss}）")
            if config.log_every and step % config.log_every == 0:
                record = {
                    "stage": "sft",
                    "preset": f"r{config.lora.r}",
                    "step": step,
                    "loss": round(last_loss, 6),
                    "grad_norm": round(float(grad_norm), 6),
                    "lr": round(scheduler.get_last_lr()[0], 8),
                    "tokens_seen": tokens_seen,
                    "tokens_per_second": round(step_tokens / max(step_seconds, 1e-9), 1),
                    "peak_memory_gb": peak_memory,
                }
                self.metrics.append(record)
                self.ctx.log_metrics(**record)
                logger.info("[SFT r=%d] step %d/%d loss %.4f tok/s %.0f", config.lora.r, step, config.max_steps, last_loss, record["tokens_per_second"])
            if config.eval_every and step > 0 and step % config.eval_every == 0:
                val_loss = evaluate_loss(self.model, val_loader, batches=config.eval_batches, device=self.device, autocast=config.autocast)
                self.ctx.log_metrics(stage="sft", kind="eval", step=step, val_loss=round(val_loss, 6))
            if config.save_every and step > 0 and step % config.save_every == 0:
                self._save_adapter(self.ctx.checkpoints_dir / "sft" / f"step_{step}", step, optimizer, scheduler)
            step += 1

        final_val_loss = evaluate_loss(self.model, val_loader, batches=config.eval_batches, device=self.device, autocast=config.autocast)
        adapter_dir = self._save_adapter(self.ctx.artifact_path("adapter"), step, optimizer, scheduler)
        self.ctx.add_artifact(adapter_dir)
        wall_seconds = time.time() - started
        report = parameter_report(self.model)
        summary = {
            "profile": config.profile,
            "base_model": str(config.model_dir),
            "lora": {"r": config.lora.r, "alpha": config.lora.alpha, "dropout": config.lora.dropout, "target_modules": list(config.lora.target_modules)},
            "steps": step,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "dataset_stats": self.train_dataset.stats,
            "final_train_loss": round(last_loss, 6),
            "initial_val_loss": round(baseline_val_loss, 6),
            "final_val_loss": round(final_val_loss, 6),
            "tokens_seen": tokens_seen,
            "wall_seconds": round(wall_seconds, 1),
            "tokens_per_second": round(tokens_seen / max(wall_seconds, 1e-9), 1),
            "peak_memory_gb": peak_memory,
            "device": self.device,
            **report,
            "adapter": str(adapter_dir),
            "metrics": self.metrics,
        }
        empty_cache(self.device)
        return summary


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    *,
    device: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
) -> list[str]:
    torch.manual_seed(seed)
    model.eval()
    responses: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = output[0][inputs["input_ids"].shape[1] :]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return responses


def write_generation_comparison(
    *,
    base_responses: list[str],
    tuned_responses: list[str],
    prompts: list[str],
    path: Path,
) -> None:
    lines = ["# 微调前后生成对比", "", "| # | 提示 | 基座模型 | LoRA 微调后 |", "|---|---|---|---|"]
    for index, (prompt, base, tuned) in enumerate(zip(prompts, base_responses, tuned_responses, strict=True)):
        base_clean = base.replace("\n", " ")[:160]
        tuned_clean = tuned.replace("\n", " ")[:160]
        lines.append(f"| {index + 1} | {prompt} | {base_clean} | {tuned_clean} |")
    lines.append("")
    lines.append("> 说明：生成使用相同采样参数（temperature=0.7, top_p=0.9, 种子固定）。")
    path.write_text("\n".join(lines), encoding="utf-8")


def merge_and_export(*, adapter_dir: Path, base_model_dir: Path, output_dir: Path, device: str) -> dict[str, Any]:
    """把 LoRA 适配器合并进基座模型并落盘，验证可重新加载生成。"""
    from peft import PeftModel  # noqa: PLC0415

    base_model, tokenizer = load_model_and_tokenizer(base_model_dir, device="cpu", dtype_name="float32")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    load_probe, probe_tokenizer = load_model_and_tokenizer(output_dir, device=device)
    sample = generate_responses(load_probe, probe_tokenizer, ["用一句话介绍你自己。"], device=device, max_new_tokens=32)
    del load_probe, model, merged, base_model
    empty_cache(device)
    return {"output_dir": str(output_dir), "loadable": True, "sample": sample[0] if sample else ""}
