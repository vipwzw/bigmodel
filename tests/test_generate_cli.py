"""CLI 生成/服务参数校验测试。"""

from __future__ import annotations

from bigmodel_lab.cli import main


def test_generate_rejects_invalid_temperature(capsys):
    exit_code = main(["generate", "--model-path", "/tmp/does-not-exist", "--prompt", "你好", "--temperature", "3"])
    assert exit_code == 2
    assert "temperature" in capsys.readouterr().err


def test_generate_rejects_invalid_top_p(capsys):
    exit_code = main(["generate", "--model-path", "/tmp/does-not-exist", "--prompt", "你好", "--top-p", "0"])
    assert exit_code == 2
    assert "top_p" in capsys.readouterr().err


def test_generate_dry_run_plan(capsys):
    exit_code = main(["generate", "--model-path", "x", "--prompt", "你好", "--prompt", "继续", "--dry-run"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[dry-run]" in output
    assert "messages=3" in output


def test_serve_dry_run_plan(capsys):
    exit_code = main(["serve", "--model-path", "x", "--dry-run"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "127.0.0.1:8000" in output
