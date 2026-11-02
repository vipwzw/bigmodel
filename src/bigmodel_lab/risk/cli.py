"""`lab risk` 子命令族：交易签名风险审查应用实验。"""

from __future__ import annotations

import argparse
import sys

from .config import RiskConfigError


def _add_config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="PATH",
        help="额外风险配置文件（可多次；后者覆盖前者）",
    )


def register_risk_subparser(parser: argparse.ArgumentParser) -> None:
    """在 `lab` 主解析器下注册 `risk` 子命令族。"""
    parser.set_defaults(command="risk")
    sub = parser.add_subparsers(dest="risk_command", required=True, metavar="{data,review,train,eval}")

    # ---------- data ----------
    data_parser = sub.add_parser("data", help="生成、校验与查看交易审查数据集")
    data_parser.set_defaults(func=_run_data)
    data_sub = data_parser.add_subparsers(dest="data_command", required=True, metavar="{generate,validate,show}")

    generate = data_sub.add_parser("generate", help="生成确定性场景数据集")
    _add_config_option(generate)
    generate.add_argument("--out", default=None, help="输出目录（默认取配置 dataset.output_dir）")
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--samples-per-family", type=int, default=None)
    generate.add_argument("--json", action="store_true", help="以 JSON 输出摘要")
    generate.set_defaults(data_command="generate", func=_run_data)

    validate = data_sub.add_parser("validate", help="校验数据集与 schema")
    _add_config_option(validate)
    validate.add_argument("--dataset", required=True, help="数据集目录或 JSONL 文件")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(data_command="validate", func=_run_data)

    show = data_sub.add_parser("show", help="展示数据集统计与数据卡")
    _add_config_option(show)
    show.add_argument("--dataset", required=True)
    show.add_argument("--json", action="store_true")
    show.set_defaults(data_command="show", func=_run_data)

    # ---------- review ----------
    review = sub.add_parser("review", help="审查单笔或批量签名请求")
    _add_config_option(review)
    source = review.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", help="单笔请求 JSON 文件")
    source.add_argument("--batch", help="批量请求 JSONL 文件")
    review.add_argument("--mode", choices=["rules_only", "rules_plus_model"], default=None)
    review.add_argument("--out", default=None, help="报告输出目录（默认取配置 review.output_dir）")
    review.add_argument("--json", action="store_true", help="以 JSON 输出报告")
    review.set_defaults(func=_run_review)

    # ---------- train ----------
    train = sub.add_parser("train", help="训练交易审查模型（复用 SFT/DPO 链路）")
    _add_config_option(train)
    train.add_argument("--profile", choices=["sanity", "standard"], default=None)
    train.add_argument("--dataset", default=None, help="训练数据集目录（默认取 runs/risk/dataset）")
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--export", action="store_true", help="训练后合并导出")
    train.add_argument("--quantize", action="store_true", help="导出后生成 MLX 4bit 量化模型")
    train.add_argument("--json", action="store_true")
    train.set_defaults(func=_run_train)

    # ---------- eval ----------
    evaluate = sub.add_parser("eval", help="评测审查引擎/模型并执行币安回归门禁")
    _add_config_option(evaluate)
    evaluate.add_argument("--dataset", required=True, help="评测集目录")
    evaluate.add_argument("--mode", choices=["rules_only", "rules_plus_model"], default="rules_only")
    evaluate.add_argument("--out", default=None, help="报告输出目录（默认取配置 eval.output_dir）")
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(func=_run_eval)


# 各子命令的惰性分发：真正实现位于对应模块，避免 `lab --help` 时导入训练栈。
def _run_data(args: argparse.Namespace) -> int:
    from .dataset import run_data_command  # noqa: PLC0415

    return run_data_command(args)


def _run_review(args: argparse.Namespace) -> int:
    from .pipeline import run_review_command  # noqa: PLC0415

    return run_review_command(args)


def _run_train(args: argparse.Namespace) -> int:
    from .train import run_train_command  # noqa: PLC0415

    return run_train_command(args)


def _run_eval(args: argparse.Namespace) -> int:
    from .evaluate import run_eval_command  # noqa: PLC0415

    return run_eval_command(args)


def run_risk_command(args: argparse.Namespace) -> int:
    """统一分发并处理配置/文件错误；退出码由各子命令返回。"""
    try:
        return int(args.func(args))
    except RiskConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"文件不存在: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2


__all__ = ["register_risk_subparser", "run_risk_command"]
