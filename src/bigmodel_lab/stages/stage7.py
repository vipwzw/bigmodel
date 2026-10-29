"""Stage 7：评测与实验追踪（困惑度、选择题、生成规则评测、跨实验对比）。"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

from ..budget import Budget
from ..config import PROJECT_ROOT, LabConfig
from ..data.tokenizer import load_hf_tokenizer
from ..evaluation.runner import (
    build_compare_report,
    collect_cross_stage_artifacts,
    generation_rule_eval,
    load_tiny_checkpoint,
    multiple_choice_eval,
    perplexity_eval,
)
from ..finetune.sft import load_model_and_tokenizer
from ..manifest import RunContext
from ..train.precision import default_device, empty_cache

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stage_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("stage7") or {}


def _latest_successful_run(stage_slug: str) -> Path:
    runs_root = PROJECT_ROOT / "runs" / stage_slug
    if not runs_root.exists():
        raise RuntimeError(f"未找到 {stage_slug} 的运行记录，请先运行前序阶段")
    for run_dir in sorted(runs_root.iterdir(), reverse=True):
        manifest = run_dir / "manifest.json"
        if not manifest.exists():
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") == "success":
            return run_dir
    raise RuntimeError(f"{stage_slug} 没有成功的运行记录")


def _pick_checkpoint(stage4_run: Path, preset: str) -> Path:
    directory = stage4_run / "checkpoints" / preset
    if not directory.exists():
        raise RuntimeError(f"Stage 4 运行缺少 {preset} 档检查点: {directory}")
    best = directory / "best.pt"
    if best.exists():
        return best
    candidates = sorted(directory.glob("ckpt_step*.pt"))
    if not candidates:
        raise RuntimeError(f"Stage 4 运行缺少 {preset} 档检查点: {directory}")
    return candidates[-1]


def run(*, ctx: RunContext, config: dict[str, Any], budget: Budget) -> dict[str, Any]:
    lab_config = LabConfig.model_validate(config)
    stage_config = _stage_config(config)
    device = default_device()
    data_dir = lab_config.resolved_dir("data_dir")
    mc_items = _load_jsonl(PROJECT_ROOT / str(stage_config.get("mc_benchmark", "data/benchmarks/zh_mc.jsonl")))
    instruction_items = _load_jsonl(PROJECT_ROOT / str(stage_config.get("instruction_benchmark", "data/benchmarks/instructions.jsonl")))
    stage4_run = _latest_successful_run("stage-4")
    logger.info("使用 Stage 4 运行: %s", stage4_run)

    tiny_models: dict[str, Any] = {}
    for preset in stage_config.get("presets", ["s", "m", "l"]):
        budget.check()
        checkpoint = _pick_checkpoint(stage4_run, preset)
        model, meta = load_tiny_checkpoint(checkpoint, device=device)
        tokenizer_path = PROJECT_ROOT / str(stage_config["tokenizers"][preset])
        tokenizer = load_hf_tokenizer(tokenizer_path)
        packed_dir = data_dir / "processed" / "stage3" / "packed" / str(stage_config["data_keys"][preset])
        ppl = perplexity_eval(model, packed_dir=packed_dir, split="val", seq_len=model.config.block_size, device=device, max_batches=int(stage_config.get("max_batches", 24)))
        ppl_repeat = perplexity_eval(model, packed_dir=packed_dir, split="val", seq_len=model.config.block_size, device=device, max_batches=int(stage_config.get("max_batches", 24)))
        multiple_choice = multiple_choice_eval(model, tokenizer, mc_items, device=device)
        del model
        empty_cache(device)
        tiny_models[preset] = {
            "checkpoint": str(checkpoint),
            "meta": meta,
            "perplexity": ppl,
            "reproducibility": {"ppl_diff": abs(ppl["perplexity"] - ppl_repeat["perplexity"])},
            "multiple_choice": {key: multiple_choice[key] for key in ("total", "correct", "accuracy", "random_baseline")},
        }
        ctx.log_metrics(stage="eval", preset=preset, kind="perplexity", perplexity=ppl["perplexity"], tokens=ppl["tokens_evaluated"], overlap=ppl["train_val_row_overlap_ratio"])
        ctx.log_metrics(stage="eval", preset=preset, kind="multiple_choice", accuracy=multiple_choice["accuracy"], total=multiple_choice["total"])
        logger.info("[%s] ppl=%.3f, MC accuracy=%.2f%%", preset, ppl["perplexity"], multiple_choice["accuracy"] * 100)

    generation: dict[str, Any] = {}
    generation_dir = PROJECT_ROOT / str(stage_config.get("generation_model_dir", "models/exports/qwen3-0.6b-sft-lora-merged"))
    if stage_config.get("generation_eval", True) and generation_dir.exists():
        logger.info("指令遵循生成评测: %s", generation_dir)
        model, tokenizer = load_model_and_tokenizer(generation_dir, device=device)
        generation = generation_rule_eval(
            model, tokenizer, instruction_items, device=device,
            max_new_tokens=int(stage_config.get("generation_max_new_tokens", 64)),
        )
        del model
        empty_cache(device)
        ctx.log_metrics(stage="eval", kind="instruction_generation", pass_rate=generation["pass_rate"], total=generation["total"])
    else:
        logger.warning("跳过生成评测：模型目录不存在（%s）", generation_dir)

    ctx.write_json_artifact("ppl_report.json", {preset: payload["perplexity"] for preset, payload in tiny_models.items()})
    ctx.write_json_artifact(
        "task_report.json",
        {
            "multiple_choice": {preset: payload["multiple_choice"] for preset, payload in tiny_models.items()},
            "generation": generation,
            "benchmarks": {"mc_items": len(mc_items), "instruction_items": len(instruction_items)},
        },
    )

    report_lines = ["# 生成评测报告（指令遵循）", ""]
    if generation:
        report_lines.append(f"- 通过率：{generation['pass_rate']}（{generation['passed']}/{generation['total']}）")
        report_lines.append("")
        report_lines.append("| # | 指令 | 规则 | 通过 | 输出摘要 |")
        report_lines.append("|---|---|---|---|---|")
        for index, item in enumerate(generation["results"]):
            response = item["response"].replace("\n", " ")[:90]
            report_lines.append(f"| {index + 1} | {item['instruction']} | {item['rule']} | {'✅' if item['passed'] else '❌'} | {response} |")
    else:
        report_lines.append("（未运行生成评测：缺少 SFT 模型或已关闭）")
    ctx.write_artifact("generation_report.md", "\n".join(report_lines))

    cross = collect_cross_stage_artifacts()
    compare_report = build_compare_report(current={"tiny_models": tiny_models, "generation": generation}, cross=cross)
    ctx.write_artifact("compare_report.md", compare_report)

    tolerance = stage_config.get("reproducibility") or {}
    ppl_diffs = {preset: payload["reproducibility"]["ppl_diff"] for preset, payload in tiny_models.items()}
    reproducibility = {
        "ppl_tolerance": float(tolerance.get("ppl_tolerance", 0.001)),
        "ppl_diffs": ppl_diffs,
        "passed": all(value <= float(tolerance.get("ppl_tolerance", 0.001)) for value in ppl_diffs.values()),
    }
    ctx.write_json_artifact("reproducibility.json", reproducibility)
    tracking = {
        "metrics_jsonl": True,
        "tensorboard": importlib.util.find_spec("tensorboard") is not None,
        "wandb": importlib.util.find_spec("wandb") is not None,
        "offline_mode": bool(lab_config.offline),
        "note": "评测与报告均本地生成，不依赖外部追踪服务",
    }
    ctx.write_json_artifact("tracking.json", tracking)

    summary = {
        "profile": budget.profile,
        "stage4_run": str(stage4_run),
        "tiny_models": {preset: {"perplexity": payload["perplexity"]["perplexity"], "mc_accuracy": payload["multiple_choice"]["accuracy"], "overlap_ratio": payload["perplexity"]["train_val_row_overlap_ratio"]} for preset, payload in tiny_models.items()},
        "generation": {"pass_rate": generation.get("pass_rate"), "total": generation.get("total")} if generation else None,
        "reproducibility": reproducibility,
        "tracking": tracking,
    }
    ctx.write_json_artifact("stage_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary
