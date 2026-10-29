"""Stage 1：环境自检 + MPS/MLX 性能基线 + token 预算校准。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..benchmark import run_benchmark, save_report
from ..budget import Budget
from ..config import LabConfig
from ..doctor import format_report, overall_status, report_json, run_checks
from ..manifest import RunContext

logger = logging.getLogger(__name__)


def _doctor_markdown(results) -> str:  # noqa: ANN001 - CheckResult 列表
    lines = ["# Stage 1 环境自检报告", "", "| 检查项 | 状态 | 详情 |", "|---|---|---|"]
    for result in results:
        lines.append(f"| {result.name} | {result.status} | {result.detail} |")
    lines.append("")
    status = overall_status(results)
    lines.append(f"**总体结论：{status}**")
    fixes = [item for item in results if item.fix and item.status != "PASS"]
    if fixes:
        lines.append("")
        lines.append("## 修复建议")
        lines.append("")
        lines.extend(f"- `{item.name}`: {item.fix}" for item in fixes)
    return "\n".join(lines)


def run(*, ctx: RunContext, config: dict[str, Any], budget: Budget) -> dict[str, Any]:
    lab_config = LabConfig.model_validate(config)
    offline = bool(config.get("offline", False))

    logger.info("运行环境自检（offline=%s）", offline)
    results = run_checks(lab_config, offline=offline)
    report_text = format_report(results)
    print(report_text)
    ctx.write_artifact("doctor.json", report_json(results))
    ctx.write_artifact("doctor.md", _doctor_markdown(results))
    status = overall_status(results)
    ctx.log_metrics(
        check_pass=sum(1 for item in results if item.status == "PASS"),
        check_warn=sum(1 for item in results if item.status == "WARN"),
        check_fail=sum(1 for item in results if item.status == "FAIL"),
        overall=status,
    )
    if status == "FAIL":
        raise RuntimeError("环境自检存在 FAIL 项，请先按报告修复后再运行后续阶段")

    budget.check()
    logger.info("运行 MPS/MLX 性能基线（profile=%s）", budget.profile)
    benchmark = run_benchmark(budget)
    save_report(
        benchmark,
        json_path=ctx.artifact_path("benchmark.json"),
        markdown_path=ctx.artifact_path("benchmark.md"),
    )
    ctx.add_artifact(ctx.artifact_path("benchmark.json"))
    ctx.add_artifact(ctx.artifact_path("benchmark.md"))
    for preset, entry in benchmark["pretrain_step"].items():
        if entry.get("available"):
            ctx.log_metrics(preset=preset, **{k: entry[k] for k in ("tokens_per_second", "peak_driver_gb", "autocast_dtype", "params")})
    for preset, entry in benchmark["calibration"].items():
        if entry.get("available"):
            ctx.log_metrics(preset=preset, estimated_hours=entry["estimated_hours"], kind="calibration")
    summary = {
        "overall": status,
        "matmul_best_gflops": benchmark["matmul_best_gflops"],
        "calibration": benchmark["calibration"],
        "profile": budget.profile,
    }
    ctx.write_json_artifact("stage_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
