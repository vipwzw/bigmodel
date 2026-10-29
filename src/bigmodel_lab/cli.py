"""`lab` 命令行入口。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from .budget import Budget, BudgetExceeded
from .config import PROJECT_ROOT, ConfigError, LabConfig, load_config
from .doctor import format_report, overall_status, report_json, run_checks
from .manifest import RunContext
from .stages import STAGES, format_stage_list, get_stage, run_stage

logger = logging.getLogger("bigmodel_lab")

_PROFILE_CHOICES = ("sanity", "standard")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab", description="单机 Apple Silicon 大模型训练实验课程")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    sub = parser.add_subparsers(dest="command", required=True)

    stages_parser = sub.add_parser("stages", help="列出 8 个课程阶段")
    stages_parser.add_argument("--json", action="store_true", help="以 JSON 输出")

    run_parser = sub.add_parser("run", help="运行指定阶段")
    run_parser.add_argument("stage", help="阶段编号（1-8）")
    run_parser.add_argument("--config", action="append", default=[], metavar="PATH", help="额外配置文件（可多次）")
    run_parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖配置项（支持点号路径）")
    run_parser.add_argument("--profile", choices=_PROFILE_CHOICES, help="运行档位")
    run_parser.add_argument("--max-minutes", type=float, help="最大运行分钟数")
    run_parser.add_argument("--dry-run", action="store_true", help="只打印将执行的计划，不创建实验目录")
    run_parser.add_argument("--offline", action="store_true", help="离线模式（跳过网络探测/下载）")

    doctor_parser = sub.add_parser("doctor", help="环境与来源自检")
    doctor_parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    doctor_parser.add_argument("--offline", action="store_true", help="跳过网络探测")
    doctor_parser.add_argument("--config", action="append", default=[], metavar="PATH")
    doctor_parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    return parser


def parse_set_values(values: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ConfigError(f"--set 需要 KEY=VALUE 形式，收到: {item!r}")
        key, _, raw_value = item.partition("=")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        overrides[key.strip()] = value
    return overrides


def _resolve_config(args: argparse.Namespace, stage_config: Path | None = None) -> tuple[dict[str, Any], LabConfig]:
    overrides = parse_set_values(getattr(args, "set", []))
    if getattr(args, "profile", None):
        overrides["profile"] = args.profile
    if getattr(args, "max_minutes", None) is not None:
        overrides["max_minutes"] = args.max_minutes
    if getattr(args, "dry_run", False):
        overrides["dry_run"] = True
    if getattr(args, "offline", False):
        overrides["offline"] = True
    paths = [Path(item) for item in getattr(args, "config", [])]
    if stage_config is not None and stage_config.exists():
        paths.append(stage_config)
    return load_config(*paths, overrides=overrides, use_default=True)


def _cmd_stages(args: argparse.Namespace) -> int:
    if args.json:
        payload = [
            {
                "number": stage.number,
                "name": stage.name,
                "requires": list(stage.requires),
                "artifacts": list(stage.artifacts),
                "summary": stage.summary,
            }
            for stage in STAGES
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_stage_list())
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    raw, config = _resolve_config(args)
    offline = bool(getattr(args, "offline", False) or raw.get("offline", False))
    results = run_checks(config, offline=offline)
    if args.json:
        print(report_json(results))
    else:
        print(format_report(results))
    return 0 if overall_status(results) != "FAIL" else 1


def _attach_file_logging(log_path: Path) -> logging.Handler:
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _cmd_run(args: argparse.Namespace) -> int:
    stage_text = str(args.stage).strip()
    if not stage_text.isdigit():
        print(f"错误：阶段参数必须是编号 1-8，收到 {stage_text!r}", file=sys.stderr)
        return 2
    stage = get_stage(int(stage_text))
    if stage is None:
        available = ", ".join(str(item.number) for item in STAGES)
        print(f"错误：未知阶段 {stage_text}；可用阶段: {available}", file=sys.stderr)
        return 2

    stage_config = PROJECT_ROOT / "configs" / f"stage{stage.number}.yaml"
    raw, config = _resolve_config(args, stage_config=stage_config)
    budget = Budget(profile=config.profile, max_minutes=config.max_minutes, dry_run=config.dry_run)

    if budget.dry_run:
        print(f"[dry-run] 阶段 {stage.number} {stage.name}（不会创建实验目录）")
        print(f"依赖: {stage.requires or '无'}")
        print(f"产物: {', '.join(stage.artifacts)}")
        print(f"档位: {budget.profile}（max_minutes={budget.max_minutes}）")
        print("--- 合并后的配置 ---")
        print(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
        return 0

    ctx = RunContext.create(stage.slug, raw, runs_dir=config.resolved_dir("runs_dir"))
    handler = _attach_file_logging(ctx.log_path)
    logger.info("开始阶段 %s（%s），实验目录: %s", stage.number, stage.name, ctx.run_dir)
    try:
        run_stage(stage, ctx, raw, budget)
        ctx.finish("success", 0)
        logger.info("阶段 %s 完成，耗时 %.1fs", stage.number, ctx._manifest["duration_seconds"])
        print(f"\n✓ 阶段 {stage.number} 完成，实验目录: {ctx.run_dir}")
        return 0
    except BudgetExceeded as exc:
        ctx.finish("timeout", 124, str(exc))
        logger.error("阶段 %s 超时: %s", stage.number, exc)
        print(f"\n✗ 阶段 {stage.number} 超时：{exc}（已保留产物: {ctx.run_dir}）", file=sys.stderr)
        return 124
    except KeyboardInterrupt:
        ctx.finish("interrupted", 130, "用户中断")
        logger.error("阶段 %s 被用户中断", stage.number)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 边界统一转成退出码
        ctx.finish("failed", 1, f"{type(exc).__name__}: {exc}")
        logger.exception("阶段 %s 失败", stage.number)
        print(f"\n✗ 阶段 {stage.number} 失败：{exc}（产物: {ctx.run_dir}）", file=sys.stderr)
        return 1
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if args.command == "stages":
            return _cmd_stages(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "run":
            return _cmd_run(args)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    parser.error(f"未知命令: {args.command}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
