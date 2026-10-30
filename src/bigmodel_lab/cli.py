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


    generate_parser = sub.add_parser("generate", help="本地生成（CLI 接口，支持流式与多轮）")
    generate_parser.add_argument("--model-path", required=True, help="本地模型目录")
    generate_parser.add_argument("--backend", choices=["mlx", "transformers"], default="mlx", help="推理后端")
    generate_parser.add_argument("--prompt", action="append", required=True, help="用户消息（可重复，模拟多轮）")
    generate_parser.add_argument("--system-prompt", default="你是一个乐于助人的中文助手。")
    generate_parser.add_argument("--max-new-tokens", type=int, default=64)
    generate_parser.add_argument("--temperature", type=float, default=0.7)
    generate_parser.add_argument("--top-p", type=float, default=0.9)
    generate_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    generate_parser.add_argument("--stream", action="store_true", help="流式输出")
    generate_parser.add_argument("--dry-run", action="store_true", help="只校验参数并打印计划")

    serve_parser = sub.add_parser("serve", help="启动 OpenAI 兼容本地推理服务")
    serve_parser.add_argument("--model-path", required=True)
    serve_parser.add_argument("--backend", choices=["mlx", "transformers"], default="mlx")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--model-name", default="bigmodel-lab")
    serve_parser.add_argument("--dry-run", action="store_true", help="只打印计划，不启动服务")

    clean_parser = sub.add_parser("clean", help="清理历史实验目录（默认 dry-run）")
    clean_parser.add_argument("--keep", type=int, default=3, help="每个阶段保留最近的 N 次运行")
    clean_parser.add_argument("--runs-dir", default=None, help="实验目录（默认项目 runs/）")
    clean_parser.add_argument("--apply", action="store_true", help="真正执行删除（默认仅打印计划）")

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
    # 加载顺序：默认配置 → 阶段默认配置 → 用户 --config（后者覆盖前者）
    paths: list[Path] = []
    if stage_config is not None and stage_config.exists():
        paths.append(stage_config)
    paths.extend(Path(item) for item in getattr(args, "config", []))
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
    if stage_text.lower() == "all":
        return _run_all_stages(args)
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



def _cmd_generate(args: argparse.Namespace) -> int:
    from .serve.generate import SamplingParams, build_generator  # noqa: PLC0415

    try:
        params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        params.validate()
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2
    messages = [{"role": "system", "content": args.system_prompt}]
    messages.extend({"role": "user", "content": prompt} for prompt in args.prompt)
    if args.dry_run:
        print(f"[dry-run] backend={args.backend}, model={args.model_path}, messages={len(messages)}")
        print(f"[dry-run] 采样参数: max_tokens={params.max_tokens}, temperature={params.temperature}, top_p={params.top_p}")
        return 0
    generator = build_generator(args.backend, args.model_path)
    if args.stream:
        for chunk in generator.stream(messages, params):
            print(chunk, end="", flush=True)
        print()
        return 0
    result = generator.generate(messages, params)
    print(result["text"])
    print(f"\n[{result['backend']}] {result['tokens_per_second']} tokens/s，TTFT {result['ttft_seconds']}s")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .serve.generate import build_generator  # noqa: PLC0415
    from .serve.openai_server import create_app, start_server, stop_server  # noqa: PLC0415

    if args.dry_run:
        print(f"[dry-run] 将在 http://{args.host}:{args.port}/v1 启动 OpenAI 兼容服务（model={args.model_path}）")
        return 0
    generator = build_generator(args.backend, args.model_path)
    app = create_app(generator, model_name=args.model_name)
    server, thread = start_server(app, args.port, host=args.host)
    print(f"OpenAI 兼容服务已启动: http://{args.host}:{args.port}/v1/chat/completions（Ctrl-C 退出）")
    try:
        while thread.is_alive():
            thread.join(timeout=1)
    except KeyboardInterrupt:
        print("\n正在停止服务…")
    finally:
        stop_server(server, thread)
    return 0




def _run_all_stages(args: argparse.Namespace) -> int:
    """顺序运行 1-8 全部阶段（每阶段独立实验目录与预算）。"""
    results: list[tuple[int, str, int]] = []
    for stage in STAGES:
        stage_config = PROJECT_ROOT / "configs" / f"stage{stage.number}.yaml"
        raw, config = _resolve_config(args, stage_config=stage_config)
        budget = Budget(profile=config.profile, max_minutes=config.max_minutes, dry_run=config.dry_run)
        if budget.dry_run:
            print(f"[dry-run] 阶段 {stage.number} {stage.name}（依赖 {stage.requires or '无'}）")
            continue
        ctx = RunContext.create(stage.slug, raw, runs_dir=config.resolved_dir("runs_dir"))
        handler = _attach_file_logging(ctx.log_path)
        try:
            logger.info("开始阶段 %s（%s），实验目录: %s", stage.number, stage.name, ctx.run_dir)
            run_stage(stage, ctx, raw, budget)
            ctx.finish("success", 0)
            print(f"✓ 阶段 {stage.number} {stage.name} 完成 → {ctx.run_dir}")
            results.append((stage.number, "success", 0))
        except BudgetExceeded as exc:
            ctx.finish("timeout", 124, str(exc))
            print(f"✗ 阶段 {stage.number} 超时：{exc}", file=sys.stderr)
            results.append((stage.number, "timeout", 124))
            break
        except Exception as exc:  # noqa: BLE001 - CLI 边界统一处理
            ctx.finish("failed", 1, f"{type(exc).__name__}: {exc}")
            logger.exception("阶段 %s 失败", stage.number)
            print(f"✗ 阶段 {stage.number} 失败：{exc}", file=sys.stderr)
            results.append((stage.number, "failed", 1))
            break
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
    print("\n端到端运行结果：")
    for number, status, _ in results:
        print(f"  阶段 {number}: {status}")
    failed = [item for item in results if item[1] != "success"]
    return 124 if any(item[2] == 124 for item in failed) else (1 if failed else 0)


def _clean_runs(runs_dir: Path, *, keep: int, apply: bool) -> list[str]:
    removed: list[str] = []
    if not runs_dir.exists():
        return removed
    for stage_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        run_dirs = sorted((path for path in stage_dir.iterdir() if path.is_dir()), key=lambda path: path.name)
        for run_dir in run_dirs[:-keep] if keep > 0 else run_dirs:
            removed.append(str(run_dir))
            if apply:
                import shutil  # noqa: PLC0415

                shutil.rmtree(run_dir)
    return removed


def _cmd_clean(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir) if args.runs_dir else PROJECT_ROOT / "runs"
    removed = _clean_runs(runs_dir, keep=max(0, args.keep), apply=args.apply)
    if not removed:
        print(f"没有可清理的实验目录（{runs_dir}，每阶段保留 {args.keep} 次）")
        return 0
    print(f"{'已删除' if args.apply else '将删除'} {len(removed)} 个历史实验目录：")
    for path in removed:
        print(f"  - {path}")
    if not args.apply:
        print("（当前为 dry-run，加 --apply 真正删除）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if args.command == "generate":
            return _cmd_generate(args)
        if args.command == "serve":
            return _cmd_serve(args)
        if args.command == "clean":
            return _cmd_clean(args)
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
