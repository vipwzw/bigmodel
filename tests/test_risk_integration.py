"""交易签名风险审查应用实验的端到端集成测试。

链路：生成数据集 → 规则基线评测（币安门禁）→ 单笔审查（币安 7702 用例）。
只跑离线、确定性的规则路径，不加载真实模型，保证在 CI/单机上秒级完成。
"""

from __future__ import annotations

import json
from pathlib import Path

from bigmodel_lab.cli import main


def _latest_dataset_run(runs_dir: Path) -> Path:
    dataset_root = runs_dir / "risk" / "dataset"
    candidates = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    assert candidates, "数据集生成失败"
    return candidates[-1]


def test_end_to_end_data_eval_review(tmp_path, capsys):
    runs_dir = tmp_path / "runs"

    # 1) 生成数据集（合成 9 族 + 币安回归）
    assert main(["risk", "data", "generate", "--out", str(runs_dir)]) == 0
    dataset_run = _latest_dataset_run(runs_dir)
    artifacts = dataset_run / "artifacts"
    for name in ("samples.jsonl", "splits.json", "data_card.md", "stats.json", "binance_cases.jsonl"):
        assert (artifacts / name).exists(), name

    # 2) 规则基线评测：币安回归必须全部 BLOCK，门禁通过
    eval_out = tmp_path / "eval-runs"
    assert main(["risk", "eval", "--dataset", str(dataset_run), "--mode", "rules_only", "--out", str(eval_out)]) == 0
    eval_run = sorted((eval_out / "risk" / "eval").iterdir())[-1]
    report = json.loads((eval_run / "artifacts" / "eval_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["binance_gate"]["total"] == 4
    assert report["binance_gate"]["blocked"] == 4
    assert report["metrics"]["overall"]["recall"] >= 0.95

    # 3) 单笔审查币安原始用例：必须 BLOCK 且退出码 2
    cases = [
        json.loads(line)
        for line in (artifacts / "binance_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cases) >= 4
    request_path = tmp_path / "binance_request.json"
    request_path.write_text(json.dumps(cases[0]["request"], ensure_ascii=False), encoding="utf-8")
    review_out = tmp_path / "review-runs"
    capsys.readouterr()  # 清空前面 data/eval 命令的输出
    exit_code = main(
        [
            "risk",
            "review",
            "--request",
            str(request_path),
            "--mode",
            "rules_only",
            "--out",
            str(review_out),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    payload = json.loads(captured.out.strip().splitlines()[0])
    assert payload["decision"] == "BLOCK"
    assert payload["exit_code"] == 2
    assert payload["simulation"]["exposure_ratio"] == 1.0
    assert payload["simulation"]["capability_delta"] == "full_control"
    assert any(hit["rule_id"] == "R-001" for hit in payload["rule_hits"])


def test_batch_review_aggregates_exit_code(tmp_path):
    runs_dir = tmp_path / "runs"
    assert main(["risk", "data", "generate", "--out", str(runs_dir)]) == 0
    dataset_run = _latest_dataset_run(runs_dir)
    samples = dataset_run / "artifacts" / "samples.jsonl"
    review_out = tmp_path / "review-runs"
    assert (
        main(
            [
                "risk",
                "review",
                "--batch",
                str(samples),
                "--mode",
                "rules_only",
                "--out",
                str(review_out),
                "--json",
            ]
        )
        == 2
    )
    review_run = sorted((review_out / "risk" / "review").iterdir())[-1]
    summary = json.loads((review_run / "artifacts" / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 58
    assert summary["decisions"]["BLOCK"] >= 4
    assert summary["exit_code"] == 2
