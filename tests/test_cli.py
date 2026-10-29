"""CLI 行为测试。"""

from __future__ import annotations

import json

from bigmodel_lab.cli import main


def test_stages_lists_eight_stages(capsys):
    assert main(["stages"]) == 0
    output = capsys.readouterr().out
    assert "环境与实验基建" in output
    assert "量化推理与服务" in output
    assert "1 环境与实验基建" not in output  # 表格使用编号与名称分列
    assert output.count("\n") >= 8


def test_stages_json_output(capsys):
    assert main(["stages", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 8
    assert payload[0]["number"] == 1
    assert payload[-1]["number"] == 8


def test_unknown_stage_returns_error_and_creates_no_run(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    exit_code = main(["run", "99", "--set", f"runs_dir={runs_dir}"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "未知阶段" in captured.err
    assert not list(runs_dir.rglob("manifest.json")) if runs_dir.exists() else True


def test_invalid_stage_argument(tmp_path, capsys):
    exit_code = main(["run", "abc", "--set", f"runs_dir={tmp_path / 'runs'}"])
    assert exit_code == 2
    assert "必须是编号" in capsys.readouterr().err


def test_dry_run_prints_plan_without_artifacts(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    exit_code = main(["run", "1", "--dry-run", "--set", f"runs_dir={runs_dir}"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[dry-run]" in output
    assert "环境自检" in output or "benchmark" in output
    assert not runs_dir.exists()


def test_invalid_config_override_reports_config_error(tmp_path, capsys):
    exit_code = main(["run", "1", "--dry-run", "--set", "profile=turbo", "--set", f"runs_dir={tmp_path}"])
    assert exit_code == 2
    assert "配置错误" in capsys.readouterr().err


def test_doctor_offline_mode(capsys):
    exit_code = main(["doctor", "--offline", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code in {0, 1}
    assert payload["status"] in {"PASS", "WARN", "FAIL"}
    names = {check["name"] for check in payload["checks"]}
    assert "python" in names
    assert "network" in names
