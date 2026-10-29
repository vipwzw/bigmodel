"""Stage 6：DPO 偏好优化（偏好对构造 → DPO 训练 → 稳定性监控 → 对齐对比）。"""

from __future__ import annotations

import json
import logging
from functools import partial
from pathlib import Path
from typing import Any

from ..budget import Budget
from ..config import PROJECT_ROOT, LabConfig
from ..data.preference import make_selfplay_pairs
from ..finetune.align import DPOConfig, DPOTrainer, write_alignment_comparison
from ..finetune.sft import LoraConfigSpec, generate_responses, load_model_and_tokenizer
from ..manifest import RunContext
from ..train.precision import default_device, empty_cache

logger = logging.getLogger(__name__)


def _stage_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("stage6") or {}


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def _generate_once(
    prompt: str,
    sample_index: int,
    *,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
) -> str:
    responses = generate_responses(
        model, tokenizer, [prompt], device=device,
        max_new_tokens=max_new_tokens, temperature=0.9, top_p=0.95, seed=sample_index + 1,
    )
    return responses[0] if responses else ""


def _lora_spec(payload: dict[str, Any]) -> LoraConfigSpec:
    return LoraConfigSpec(
        r=int(payload.get("r", 8)),
        alpha=int(payload.get("alpha", 16)),
        dropout=float(payload.get("dropout", 0.05)),
        target_modules=list(payload.get("target_modules") or []),
    )


def _build_selfplay(
    *,
    stage_config: dict[str, Any],
    sft_model_dir: Path,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = stage_config.get("self_play") or {}
    if not payload.get("enabled"):
        return [], {"enabled": False}
    prompts = list(payload.get("prompts") or [])
    max_pairs = int(payload.get("max_pairs", 20))
    max_new_tokens = int(payload.get("max_new_tokens", 40))
    model, tokenizer = load_model_and_tokenizer(sft_model_dir, device=device)
    generate = partial(
        _generate_once,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    pairs = make_selfplay_pairs(prompts, generate, max_pairs=max_pairs)
    del model
    empty_cache(device)
    report = {
        "enabled": True,
        "prompts": len(prompts),
        "generated_pairs": len(pairs),
        "max_pairs": max_pairs,
        "preview": [{"prompt": pair["prompt"], "chosen": pair["chosen"][:80], "rejected": pair["rejected"][:80]} for pair in pairs[:3]],
    }
    return pairs, report


def run(*, ctx: RunContext, config: dict[str, Any], budget: Budget) -> dict[str, Any]:
    LabConfig.model_validate(config)
    stage_config = _stage_config(config)
    device = default_device()
    sft_model_dir = PROJECT_ROOT / str(stage_config.get("sft_model_dir", "models/exports/qwen3-0.6b-sft-lora-merged"))
    if not sft_model_dir.exists():
        raise RuntimeError(f"缺少 Stage 5 的 SFT 模型 {sft_model_dir}，请先运行 lab run 5")
    preference_dir = PROJECT_ROOT / str(stage_config.get("preference_dir", "data/processed/stage3/preference/pref_en"))
    train_path = preference_dir / "train.jsonl"
    val_path = preference_dir / "val.jsonl"
    if not train_path.exists():
        raise RuntimeError(f"缺少 Stage 3 偏好数据 {train_path}，请先运行 lab run 3")

    public_train = _load_jsonl(train_path)
    val_records = _load_jsonl(val_path) if val_path.exists() else public_train[:16]
    selfplay_pairs, selfplay_report = _build_selfplay(stage_config=stage_config, sft_model_dir=sft_model_dir, device=device)
    ctx.write_json_artifact("selfplay_report.json", selfplay_report)
    train_records = public_train + selfplay_pairs
    logger.info("DPO 数据：公开偏好 %d 条 + self-play %d 条，验证集 %d 条", len(public_train), len(selfplay_pairs), len(val_records))
    if not train_records:
        raise RuntimeError("偏好数据为空，无法进行 DPO 训练")

    dpo_payload = stage_config.get("dpo") or {}
    settings = dpo_payload.get(budget.profile) or dpo_payload.get("sanity") or {}
    monitor_cfg = stage_config.get("monitor") or {}
    dpo_config = DPOConfig(
        model_dir=sft_model_dir,
        output_dir=ctx.artifact_path("dpo_adapter"),
        beta=float(dpo_payload.get("beta", 0.1)),
        max_steps=int(settings.get("max_steps", 20)),
        batch_size=int(settings.get("batch_size", 2)),
        grad_accum=int(settings.get("grad_accum", 2)),
        max_seq_len=int(settings.get("max_seq_len", 384)),
        learning_rate=float(settings.get("learning_rate", 5e-6)),
        warmup_steps=int(settings.get("warmup_steps", 2)),
        log_every=int(settings.get("log_every", 5)),
        eval_every=int(settings.get("eval_every", 10)),
        eval_batches=int(settings.get("eval_batches", 2)),
        save_every=int(settings.get("save_every", 10)),
        seed=int(config.get("seed", 42)),
        device=device,
        profile=budget.profile,
        lora=_lora_spec(dpo_payload.get("lora") or stage_config.get("lora") or {}),
        quality_prompts=list(stage_config.get("quality_prompts") or []),
        quality_max_new_tokens=int(settings.get("quality_max_new_tokens", 32)),
        stop_on_unhealthy=bool(monitor_cfg.get("stop_on_unhealthy", False)),
    )
    summary = DPOTrainer(
        dpo_config, ctx=ctx, budget=budget, train_records=train_records, val_records=val_records
    ).train()
    summary["selfplay"] = selfplay_report
    summary["data"] = {"public_pairs": len(public_train), "selfplay_pairs": len(selfplay_pairs), "val_pairs": len(val_records)}
    path = ctx.artifact_path("dpo_summary.json")
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    ctx.add_artifact(path)

    health = summary.get("health", {})
    ctx.write_json_artifact("monitor.json", {"health": health, "history": summary.get("health_history", []), "thresholds": monitor_cfg})

    generation_cfg = stage_config.get("generation") or {}
    prompts = list(generation_cfg.get("prompts") or [])
    if prompts:
        logger.info("对齐效果对比：SFT vs DPO（%d 条提示）", len(prompts))
        sft_model, tokenizer = load_model_and_tokenizer(sft_model_dir, device=device)
        sft_responses = generate_responses(
            sft_model, tokenizer, prompts, device=device,
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 60)),
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
        )
        del sft_model
        empty_cache(device)
        dpo_model, dpo_tokenizer = load_model_and_tokenizer(sft_model_dir, device=device)
        from peft import PeftModel  # noqa: PLC0415

        dpo_model = PeftModel.from_pretrained(dpo_model, summary["adapter"])
        dpo_responses = generate_responses(
            dpo_model, dpo_tokenizer, prompts, device=device,
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 60)),
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
        )
        del dpo_model
        empty_cache(device)
        comparison_path = ctx.artifact_path("side_by_side.md")
        write_alignment_comparison(
            sft_responses=sft_responses, dpo_responses=dpo_responses, prompts=prompts, path=comparison_path
        )
        ctx.add_artifact(comparison_path)

    final = {
        "profile": budget.profile,
        "sft_model": str(sft_model_dir),
        "data": summary["data"],
        "beta": summary["beta"],
        "steps": summary["steps"],
        "final_train_loss": summary["final_train_loss"],
        "final_eval": summary["final_eval"],
        "reward_margin": summary.get("health", {}).get("margin"),
        "approx_kl": summary.get("health", {}).get("approx_kl"),
        "monitor_status": summary.get("health", {}).get("status"),
        "warnings": summary.get("health", {}).get("warnings"),
        "tokens_per_second": summary["tokens_per_second"],
        "peak_memory_gb": summary["peak_memory_gb"],
        "adapter": summary["adapter"],
    }
    ctx.write_json_artifact("stage_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2, default=str))
    return final
