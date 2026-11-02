"""`lab risk` 子命令注册与分发测试。"""

from __future__ import annotations

import pytest

from bigmodel_lab.cli import build_parser, main
from bigmodel_lab.risk.cli import run_risk_command
from bigmodel_lab.risk.config import RiskConfigError


def test_lab_stages_still_lists_eight_stages(capsys):
    assert main(["stages"]) == 0
    output = capsys.readouterr().out
    assert "环境与实验基建" in output
    assert "验证" not in output or "环境" in output


def test_risk_help_lists_four_subcommands(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["risk", "--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for command in ("data", "review", "train", "eval"):
        assert command in output


def test_parser_accepts_risk_data_generate():
    args = build_parser().parse_args(["risk", "data", "generate", "--seed", "7"])
    assert args.command == "risk"
    assert args.risk_command == "data"
    assert args.data_command == "generate"
    assert args.seed == 7


def test_parser_accepts_risk_review():
    args = build_parser().parse_args(["risk", "review", "--request", "req.json", "--mode", "rules_only"])
    assert args.request == "req.json"
    assert args.mode == "rules_only"


def test_parser_requires_data_source_for_review():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["risk", "review"])


def test_parser_accepts_risk_eval_and_train():
    eval_args = build_parser().parse_args(["risk", "eval", "--dataset", "runs/risk/dataset/x"])
    assert eval_args.mode == "rules_only"
    train_args = build_parser().parse_args(["risk", "train", "--profile", "sanity", "--export"])
    assert train_args.profile == "sanity"
    assert train_args.export is True


def _args_with_handler(handler):
    class Args:
        func = staticmethod(handler)

    return Args()


def test_run_risk_command_maps_config_error_to_exit_code(capsys):
    def handler(_args):
        raise RiskConfigError("bad config")

    assert run_risk_command(_args_with_handler(handler)) == 2
    assert "配置错误" in capsys.readouterr().err


def test_run_risk_command_maps_file_error_to_exit_code(capsys):
    def handler(_args):
        raise FileNotFoundError("missing.json")

    assert run_risk_command(_args_with_handler(handler)) == 2
    assert "文件不存在" in capsys.readouterr().err


def test_run_risk_command_passes_through_exit_code():
    def handler(_args):
        return 2

    assert run_risk_command(_args_with_handler(handler)) == 2
