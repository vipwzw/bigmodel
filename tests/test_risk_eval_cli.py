"""`lab risk eval` CLI 测试：JSON/Markdown 报告、manifest、重复评测一致性与模型降级。"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from bigmodel_lab.cli import main
from bigmodel_lab.risk import evaluate as evaluate_module
from bigmodel_lab.risk.dataset import (
    BINANCE_CASES_FILENAME,
    SAMPLES_FILENAME,
    SPLITS_FILENAME,
    STATS_FILENAME,
    build_splits,
    dataset_stats,
    generate_dataset,
    load_binance_samples,
)
from bigmodel_lab.risk.evaluate import compare_eval_reports
from bigmodel_lab.risk.schema import ModelOpinion

SEED = 5


def _write_eval_dataset(base: Path) -> Path:
    """写入一个真实 schema 的临时数据集 run 目录（22 条，含 4 条币安用例）。"""
    samples = [
        *generate_dataset(seed=SEED, samples_per_family=2),
        *load_binance_samples(),
    ]
    splits = build_splits(samples, seed=SEED)
    stats = dataset_stats(samples, splits, seed=SEED, samples_per_family=2)
    artifacts = base / "dataset-run" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / SAMPLES_FILENAME).write_text(
        "".join(f"{sample.model_dump_json()}\n" for sample in samples), encoding="utf-8"
    )
    (artifacts / SPLITS_FILENAME).write_text(json.dumps(splits, ensure_ascii=False), encoding="utf-8")
    (artifacts / STATS_FILENAME).write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    (artifacts / BINANCE_CASES_FILENAME).write_text(
        "".join(f"{sample.model_dump_json()}\n" for sample in load_binance_samples()),
        encoding="utf-8",
    )
    return base / "dataset-run"


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _eval_run(out_root: Path) -> Path:
    runs = sorted((out_root / "risk" / "eval").iterdir())
    assert len(runs) == 1, f"期望一次评测运行，实际 {runs}"
    return runs[0]


def _load_report(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "artifacts" / "eval_report.json").read_text(encoding="utf-8"))


class _FakeReviewer:
    """满足 Reviewer 协议的假模型：只返回确定性的第二意见。"""

    version = "fake-reviewer-v1"

    def review(self, request):  # noqa: ANN001, ANN201 - CLI 测试用 fake
        return ModelOpinion(
            available=True,
            risk_level="medium",
            recommended_action="STEP_UP",
            capability_delta="value_transfer",
            confidence=0.6,
            risk_factors=["fake-reviewer"],
            evidence=["fake:evidence"],
        )


@pytest.fixture()
def dataset_run(tmp_path: Path) -> Path:
    return _write_eval_dataset(tmp_path)


# ---------------------------------------------------------------------------
# rules_only：JSON + Markdown 报告与 manifest
# ---------------------------------------------------------------------------


def test_cli_rules_only_writes_json_and_markdown(dataset_run: Path, tmp_path: Path):
    out_root = tmp_path / "runs"
    code, stdout, stderr = _run_cli(
        [
            "risk",
            "eval",
            "--dataset",
            str(dataset_run),
            "--mode",
            "rules_only",
            "--out",
            str(out_root),
        ]
    )

    run_dir = _eval_run(out_root)
    report = _load_report(run_dir)
    assert code == report["exit_code"]
    assert code in {0, 2}
    assert report["mode"] == "rules_only"
    assert report["passed"] is (code == 0)
    assert report["dataset"]["split"] == "test"
    assert report["dataset"]["leaks"] == []
    assert report["dataset"]["binance_cases"] == 4
    assert report["versions"]["rules"]
    assert report["versions"]["dataset"] == "risk-dataset-v1"
    assert report["versions"]["code"]
    assert "风险评测完成" in stdout
    assert "风险评测完成" in stdout and "报告:" in stdout

    markdown = (run_dir / "artifacts" / "eval_report.md").read_text(encoding="utf-8")
    assert markdown.startswith("# 交易签名风险审查评测报告")
    for section in ("## 门禁结论", "## 核心指标", "## 币安回归门禁", "## 版本与可复现性"):
        assert section in markdown
    if code == 0:
        assert "4/4" in markdown

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "risk/eval"
    assert manifest["status"] == ("success" if code == 0 else "failed_gate")
    assert manifest["exit_code"] == code
    artifact_names = set(manifest["artifacts"])
    assert "artifacts/eval_report.json" in artifact_names
    assert "artifacts/eval_report.md" in artifact_names
    assert stderr == "" or "产物目录" in stderr


def test_cli_json_stdout_is_pure_json_and_repeatable(dataset_run: Path, tmp_path: Path):
    argv = [
        "risk",
        "eval",
        "--dataset",
        str(dataset_run),
        "--mode",
        "rules_only",
        "--json",
    ]
    code_first, stdout_first, _ = _run_cli([*argv, "--out", str(tmp_path / "runs-1")])
    report_first = json.loads(stdout_first)
    code_second, stdout_second, _ = _run_cli([*argv, "--out", str(tmp_path / "runs-2")])
    report_second = json.loads(stdout_second)

    assert code_first == report_first["exit_code"]
    assert code_second == report_second["exit_code"]
    assert set(report_first["metrics"]) >= {
        "overall",
        "by_family",
        "asset_weighted_recall",
        "latency",
        "model",
        "rules",
    }
    assert report_first["reproducibility"]["tolerances"]["recall"] == 0.0
    comparison = compare_eval_reports(report_first, report_second)
    assert comparison["within_tolerance"] is True
    assert comparison["passed"] is True
    assert comparison["regressions"] == []
    assert comparison["decision_distribution"]["changed"] is False
    # 确定性指标必须逐位一致，延迟只要求落在声明容差内
    for metric in ("recall", "fpr", "expected_match_rate"):
        assert report_first["metrics"]["overall"][metric] == report_second["metrics"]["overall"][metric]
    assert (
        report_first["metrics"]["asset_weighted_recall"]["value"]
        == report_second["metrics"]["asset_weighted_recall"]["value"]
    )
    assert (
        report_first["metrics"]["model"]["invalid_output_rate"]
        == report_second["metrics"]["model"]["invalid_output_rate"]
    )
    assert comparison["metric_differences"]
    latency_diffs = [
        item for item in comparison["metric_differences"] if item["metric"].startswith("latency")
    ]
    assert all(item["passed"] for item in latency_diffs)


# ---------------------------------------------------------------------------
# rules_plus_model：模型降级与消融
# ---------------------------------------------------------------------------


def test_cli_rules_plus_model_falls_back_when_reviewer_unavailable(
    dataset_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def failing_loader() -> Any:
        raise RuntimeError("模型依赖缺失")

    monkeypatch.setattr(evaluate_module, "_load_build_reviewer", failing_loader)
    code, stdout, stderr = _run_cli(
        [
            "risk",
            "eval",
            "--dataset",
            str(dataset_run),
            "--mode",
            "rules_plus_model",
            "--out",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    report = json.loads(stdout)
    assert code == report["exit_code"]
    assert report["mode"] == "rules_only"
    assert report["requested_mode"] == "rules_plus_model"
    assert report["degraded"] is True
    assert "模型依赖缺失" in report["degraded_reason"]
    assert any("降级为 rules_only" in warning for warning in report["warnings"])
    assert "ablation" not in report
    assert "警告" in stderr


def test_cli_rules_plus_model_runs_ablation_with_fake_reviewer(
    dataset_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        evaluate_module, "_load_build_reviewer", lambda: (lambda config: _FakeReviewer())
    )
    out_root = tmp_path / "runs"
    code, stdout, stderr = _run_cli(
        [
            "risk",
            "eval",
            "--dataset",
            str(dataset_run),
            "--mode",
            "rules_plus_model",
            "--out",
            str(out_root),
            "--json",
        ]
    )

    report = json.loads(stdout)
    assert code == report["exit_code"]
    assert report["mode"] == "rules_plus_model"
    assert report["degraded"] is False
    assert report["versions"]["model"] == "fake-reviewer-v1"
    assert report["metrics"]["model"]["enabled"] is True
    assert report["metrics"]["model"]["invalid_output_rate"] == 0.0
    ablation = report["ablation"]
    assert ablation["available"] is True
    assert "rules_only" in ablation and "rules_plus_model" in ablation
    assert ablation["by_family"]
    assert "recall_delta" in ablation["model_net_gain"]
    assert "newly_blocked" in ablation["extra_fp_cost"]
    assert stderr == "" or "产物目录" in stderr

    markdown = (_eval_run(out_root) / "artifacts" / "eval_report.md").read_text(encoding="utf-8")
    assert "消融对比" in markdown


def test_run_context_retries_when_run_id_collides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """同一秒内重复评测时 run_id 冲突应自动追加序号，而不是崩溃。"""
    from datetime import UTC, datetime

    from bigmodel_lab import manifest as manifest_module
    from bigmodel_lab.manifest import config_fingerprint
    from bigmodel_lab.risk.config import load_risk_config
    from bigmodel_lab.risk.evaluate import _create_run_context

    frozen = datetime(2026, 11, 2, 17, 0, 0, tzinfo=UTC)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001, ANN206 - 固定时间便于断言
            return frozen.replace(tzinfo=None) if tz is None else frozen.astimezone(tz)

    monkeypatch.setattr(manifest_module, "datetime", _FrozenDatetime)
    monkeypatch.setattr(evaluate_module, "datetime", _FrozenDatetime)
    config = load_risk_config()
    runs_dir = tmp_path / "runs"
    base = f"{frozen.strftime('%Y%m%d-%H%M%S')}-{config_fingerprint(config.raw)[:8]}"
    (runs_dir / "risk" / "eval" / base).mkdir(parents=True)

    first = _create_run_context(config, runs_dir)
    second = _create_run_context(config, runs_dir)

    assert first.run_dir.name == f"{base}-1"
    assert second.run_dir.name == f"{base}-2"
    assert first.run_dir != second.run_dir


# ---------------------------------------------------------------------------
# 参数与数据错误
# ---------------------------------------------------------------------------


def test_cli_missing_dataset_returns_two(tmp_path: Path):
    code, stdout, stderr = _run_cli(
        [
            "risk",
            "eval",
            "--dataset",
            str(tmp_path / "does-not-exist"),
            "--out",
            str(tmp_path / "runs"),
        ]
    )
    assert code == 2
    assert "文件不存在" in stderr
    assert stdout == ""


def test_cli_missing_dataset_argument_returns_two(tmp_path: Path):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        with pytest.raises(SystemExit) as exc_info:
            main(["risk", "eval", "--out", str(tmp_path / "runs")])
    assert exc_info.value.code == 2
    assert "--dataset" in stderr.getvalue()
