"""Stage 5：Qwen3 小模型 LoRA 指令微调（掩码自检、超参对比、前后生成对比、合并导出）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..budget import Budget
from ..config import PROJECT_ROOT, LabConfig
from ..data.chat import load_sft_records, mask_sanity_check
from ..data.specs import load_resources
from ..data.sources import fetch_resource
from ..finetune.sft import (
    LoraConfigSpec,
    SFTConfig,
    SFTTrainer,
    generate_responses,
    load_model_and_tokenizer,
    merge_and_export,
    parameter_report,
    probe_qlora,
    write_generation_comparison,
)
from ..manifest import RunContext
from ..train.precision import default_device, driver_memory_gb, empty_cache

logger = logging.getLogger(__name__)


def _stage_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("stage5") or {}


def _resolve_model_dir(spec_payload: dict[str, Any], fallback_key: str, resources) -> Path:
    spec = resources.get(fallback_key)
    if spec and spec.dest:
        return PROJECT_ROOT / spec.dest
    raise RuntimeError(f"配置缺少 data.resources.{fallback_key} 或未设置 dest")


def _train_settings(stage_config: dict[str, Any], profile: str) -> dict[str, Any]:
    train = stage_config.get("train") or {}
    settings = train.get(profile) or train.get("sanity") or {}
    return dict(settings)


def _lora_spec(stage_config: dict[str, Any], variant: dict[str, Any] | None = None) -> LoraConfigSpec:
    base = stage_config.get("lora") or {}
    payload = dict(base)
    if variant:
        if "lora_r" in variant:
            payload["r"] = variant["lora_r"]
        if "lora_alpha" in variant:
            payload["alpha"] = variant["lora_alpha"]
    return LoraConfigSpec(
        r=int(payload.get("r", 16)),
        alpha=int(payload.get("alpha", 32)),
        dropout=float(payload.get("dropout", 0.05)),
        target_modules=list(payload.get("target_modules") or []),
    )


def _model_inventory(resources, stage_config: dict[str, Any], device: str) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    keys = [stage_config.get("base_model_key", "qwen3_06b"), stage_config.get("advance_model_key", "qwen3_17b")]
    for key in keys:
        spec = resources.get(key)
        if spec is None or not spec.dest:
            continue
        path = PROJECT_ROOT / spec.dest
        if not path.exists():
            inventory[key] = {"available": False, "path": str(path), "hint": f"可执行 lab run 5（联网）自动下载，或 modelscope download --repo-type model {spec.repo_id} --local-dir {path}"}
            continue
        try:
            model, tokenizer = load_model_and_tokenizer(path, device=device)
            inventory[key] = {
                "available": True,
                "path": str(path),
                "repo_id": spec.repo_id,
                **parameter_report(model),
                "memory_gb": driver_memory_gb(device),
                "tokenizer_vocab": tokenizer.vocab_size,
            }
            del model, tokenizer
            empty_cache(device)
        except Exception as exc:  # noqa: BLE001 - 记录但不阻断主流程
            inventory[key] = {"available": False, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return inventory


def run(*, ctx: RunContext, config: dict[str, Any], budget: Budget) -> dict[str, Any]:
    lab_config = LabConfig.model_validate(config)
    stage_config = _stage_config(config)
    resources = load_resources(config)
    device = default_device()
    base_key = str(stage_config.get("base_model_key", "qwen3_06b"))
    spec = resources.get(base_key)
    if spec is None:
        raise RuntimeError(f"配置缺少 data.resources.{base_key}")
    entry = fetch_resource(
        spec, data_dir=lab_config.resolved_dir("data_dir"), network=lab_config.network,
        offline=lab_config.offline, budget=budget,
    )
    model_dir = _resolve_model_dir(stage_config, base_key, resources)
    logger.info("基座模型: %s（%s）", model_dir, entry["status"])

    instruction_dir = PROJECT_ROOT / str(stage_config.get("instruction_dir", "data/processed/stage3/instruction/sft_zh"))
    train_path = instruction_dir / "train.jsonl"
    val_path = instruction_dir / "val.jsonl"
    if not train_path.exists():
        raise RuntimeError(f"缺少 Stage 3 指令数据 {train_path}，请先运行 lab run 3")
    settings = _train_settings(stage_config, budget.profile)

    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sample_records = load_sft_records([train_path], max_records=5)
    mask_report = mask_sanity_check(tokenizer, sample_records[0]["messages"], max_seq_len=int(settings.get("max_seq_len", 384)))
    mask_report["sample_preview"] = sample_records[0]["messages"]
    ctx.write_json_artifact("mask_check.json", mask_report)
    logger.info("掩码自检: ok=%s, prompt_tokens=%s, response_tokens=%s", mask_report.get("ok"), mask_report.get("prompt_tokens"), mask_report.get("response_tokens"))
    if not mask_report.get("ok"):
        raise RuntimeError(f"对话模板/损失掩码自检失败: {mask_report}")
    del tokenizer

    qlora = probe_qlora()
    ctx.write_json_artifact("qlora_probe.json", qlora)
    inventory = _model_inventory(resources, stage_config, device)
    ctx.write_json_artifact("model_inventory.json", inventory)
    logger.info("模型清单: %s", {key: value.get("available") for key, value in inventory.items()})

    train_records = load_sft_records([train_path], max_records=int(settings.get("max_samples", 400)))
    val_records = load_sft_records([val_path], max_records=int(settings.get("val_samples", 40))) if val_path.exists() else train_records[:20]
    if not train_records:
        raise RuntimeError(f"指令数据为空: {train_path}")

    variants = stage_config.get("compare") or [{"name": "default"}]
    results: list[dict[str, Any]] = []
    for variant in variants:
        name = str(variant.get("name", f"variant_{len(results)}"))
        lora = _lora_spec(stage_config, variant)
        sft_config = SFTConfig(
            model_dir=model_dir,
            output_dir=ctx.artifact_path(f"adapter_{name}"),
            lora=lora,
            max_steps=int(variant.get("max_steps", settings.get("max_steps", 40))),
            batch_size=int(settings.get("batch_size", 2)),
            grad_accum=int(settings.get("grad_accum", 4)),
            max_seq_len=int(settings.get("max_seq_len", 384)),
            learning_rate=float(variant.get("learning_rate", settings.get("learning_rate", 1e-4))),
            warmup_steps=int(settings.get("warmup_steps", 3)),
            log_every=int(settings.get("log_every", 5)),
            eval_every=int(settings.get("eval_every", 20)),
            eval_batches=int(settings.get("eval_batches", 2)),
            save_every=int(settings.get("save_every", 0)),
            seed=int(config.get("seed", 42)),
            device=device,
            profile=budget.profile,
            max_samples=None,
        )
        logger.info("开始 SFT 变体 %s：lora_r=%d，steps=%d", name, lora.r, sft_config.max_steps)
        summary = SFTTrainer(
            sft_config, ctx=ctx, budget=budget, train_records=train_records, val_records=val_records
        ).train()
        path = ctx.artifact_path(f"sft_summary_{name}.json")
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        ctx.add_artifact(path)
        results.append({"name": name, "lora_r": lora.r, "learning_rate": sft_config.learning_rate, "summary": summary})

    comparison_lines = ["# SFT 超参对比", "", "| 变体 | LoRA rank | 可训练参数 | steps | final train loss | val loss | tokens/s | 峰值显存(GB) | 耗时(s) |", "|---|---|---|---|---|---|---|---|---|"]
    for result in results:
        summary = result["summary"]
        comparison_lines.append(
            f"| {result['name']} | {result['lora_r']} | {summary['trainable_params']:,} | {summary['steps']} | "
            f"{summary['final_train_loss']} | {summary['final_val_loss']} | {summary['tokens_per_second']} | "
            f"{summary['peak_memory_gb']} | {summary['wall_seconds']} |"
        )
    comparison_lines.append("")
    comparison_lines.append("> 相同数据、种子与步数下比较 LoRA rank 对损失、显存与吞吐的影响。")
    ctx.write_artifact("sft_comparison.md", "\n".join(comparison_lines))

    generation_cfg = stage_config.get("generation") or {}
    prompts = list(generation_cfg.get("prompts") or [])
    primary = results[0]["summary"]
    if prompts:
        logger.info("生成对比：基座 vs 微调（%d 条提示）", len(prompts))
        base_model, base_tokenizer = load_model_and_tokenizer(model_dir, device=device)
        base_responses = generate_responses(
            base_model, base_tokenizer, prompts, device=device,
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 60)),
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
        )
        del base_model
        empty_cache(device)
        tuned_model, tuned_tokenizer = load_model_and_tokenizer(model_dir, device=device)
        from peft import PeftModel  # noqa: PLC0415

        tuned_model = PeftModel.from_pretrained(tuned_model, primary["adapter"])
        tuned_responses = generate_responses(
            tuned_model, tuned_tokenizer, prompts, device=device,
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 60)),
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
        )
        del tuned_model
        empty_cache(device)
        comparison_path = ctx.artifact_path("before_after.md")
        write_generation_comparison(
            base_responses=base_responses, tuned_responses=tuned_responses, prompts=prompts, path=comparison_path
        )
        ctx.add_artifact(comparison_path)

    export_cfg = stage_config.get("export") or {}
    export_dir = PROJECT_ROOT / str(export_cfg.get("dir", "models/exports/qwen3-0.6b-sft-lora-merged"))
    logger.info("合并 LoRA 并导出到 %s", export_dir)
    export_report = merge_and_export(
        adapter_dir=Path(primary["adapter"]), base_model_dir=model_dir, output_dir=export_dir, device=device
    )
    ctx.write_json_artifact("export_report.json", export_report)

    advance_cfg = stage_config.get("advance") or {}
    advance_report: dict[str, Any] = {"enabled": bool(advance_cfg.get("enabled", False))}
    if advance_cfg.get("enabled"):
        advance_key = str(stage_config.get("advance_model_key", "qwen3_17b"))
        advance_spec = resources.get(advance_key)
        if advance_spec is None:
            raise RuntimeError(f"配置缺少 data.resources.{advance_key}")
        advance_entry = fetch_resource(
            advance_spec, data_dir=lab_config.resolved_dir("data_dir"), network=lab_config.network,
            offline=lab_config.offline, budget=budget,
        )
        advance_model_dir = _resolve_model_dir(stage_config, advance_key, resources)
        advance_config = SFTConfig(
            model_dir=advance_model_dir,
            output_dir=ctx.artifact_path("adapter_1.7b"),
            lora=_lora_spec(stage_config),
            max_steps=int(advance_cfg.get("max_steps", 20)),
            batch_size=int(settings.get("batch_size", 2)),
            grad_accum=int(settings.get("grad_accum", 4)),
            max_seq_len=int(settings.get("max_seq_len", 384)),
            learning_rate=float(settings.get("learning_rate", 1e-4)),
            seed=int(config.get("seed", 42)),
            device=device,
            profile=budget.profile,
        )
        advance_summary = SFTTrainer(
            advance_config, ctx=ctx, budget=budget, train_records=train_records, val_records=val_records
        ).train()
        advance_report.update({"model": str(advance_model_dir), "source": advance_entry["status"], "summary": advance_summary})

    summary = {
        "profile": budget.profile,
        "base_model": str(model_dir),
        "base_model_source": entry["status"],
        "qlora": qlora,
        "inventory": inventory,
        "mask_check": {key: mask_report.get(key) for key in ("ok", "prompt_tokens", "response_tokens", "prompt_masked")},
        "variants": [
            {
                "name": result["name"],
                "lora_r": result["lora_r"],
                **{
                    key: result["summary"][key]
                    for key in ("steps", "trainable_params", "trainable_ratio", "final_train_loss", "final_val_loss", "tokens_seen", "tokens_per_second", "peak_memory_gb", "wall_seconds")
                },
            }
            for result in results
        ],
        "export": export_report,
        "advance": {key: advance_report[key] for key in ("enabled", "source") if key in advance_report},
    }
    ctx.write_json_artifact("stage_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary
