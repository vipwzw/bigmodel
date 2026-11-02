"""`risk.evaluate` 门禁与消融测试：币安回归必须全 BLOCK，失败时非零退出。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_risk_model_data import make_sample

from bigmodel_lab.cli import main
from bigmodel_lab.risk import evaluate as evaluate_module
from bigmodel_lab.risk.config import EvalSettings, RiskConfig, load_risk_config
from bigmodel_lab.risk.dataset import (
    BINANCE_CASES_FILENAME,
    SAMPLES_FILENAME,
    SPLITS_FILENAME,
    STATS_FILENAME,
    dataset_stats,
    load_binance_samples,
)
from bigmodel_lab.risk.evaluate import EvalDataset, evaluate, run_ablation
from bigmodel_lab.risk.schema import (
    ModelOpinion,
    ReviewReport,
    RuleHit,
    SimulationResult,
    exit_code_for,
)

BINANCE_SAMPLES = load_binance_samples()
BINANCE_IDS = {sample.sample_id: sample.request.request_id for sample in BINANCE_SAMPLES}


def _rid(sample_id: str) -> str:
    return BINANCE_IDS[sample_id]


class _FakeReviewer:
    version = "fake-reviewer-v1"

    def review(self, request):  # noqa: ANN001, ANN201 - 消融测试用 fake
        return ModelOpinion(available=True, recommended_action="BLOCK", confidence=0.7)


def _rule() -> RuleHit:
    return RuleHit(
        rule_id="R-001",
        name="不可逆操作",
        severity="critical",
        evidence=["币安式 EIP-7702 委托"],
        rule_version="risk-rules-v1",
        suggested_decision="BLOCK",
    )


def _report(request: Any, decision: str, *, rule_hits: list[RuleHit] | None = None) -> ReviewReport:
    return ReviewReport(
        report_id=f"rep-{request.request_id}",
        request_id=request.request_id,
        generated_at=request.timestamp,
        account=request.account,
        chain=request.chain,
        simulation=SimulationResult(exposure_usd=1_000.0),
        rule_hits=list(rule_hits or []),
        decision=decision,  # type: ignore[arg-type]
        exit_code=exit_code_for(decision),
        duration_ms=0.5,
    )


def _install_fake_review(
    monkeypatch: pytest.MonkeyPatch, decisions: dict[str, str]
) -> None:
    """按 request_id 返回固定决策；币安用例默认携带 BLOCK 规则命中作为决策依据。"""

    def fake_review(request, *, rules, reviewer=None):  # noqa: ANN001 - 与被替换函数同签名
        hits = [_rule()] if "binance" in request.request_id else []
        return _report(request, decisions[request.request_id], rule_hits=hits)

    monkeypatch.setattr(evaluate_module, "review_request", fake_review)


def _samples() -> list:
    return [
        make_sample(
            sample_id="pos-big",
            scenario_family="native_transfer",
            expected_decision="BLOCK",
            risk_level="critical",
            capability_delta="value_transfer",
            exposure_ratio=0.5,
        ),
        make_sample(
            sample_id="pos-small",
            scenario_family="approve_permit",
            expected_decision="STEP_UP",
            risk_level="medium",
            capability_delta="allowance_grant",
            exposure_ratio=0.2,
        ),
        make_sample(
            sample_id="neg-normal",
            scenario_family="normal_ops",
            expected_decision="ALLOW",
            risk_level="low",
            capability_delta="value_transfer",
            exposure_ratio=0.001,
        ),
    ]


def _dataset(binance: list | None = None) -> EvalDataset:
    samples = _samples()
    return EvalDataset(
        samples=samples,
        binance_samples=list(BINANCE_SAMPLES) if binance is None else list(binance),
        source="<memory>",
        stats=dataset_stats(samples),
    )


def _config(**overrides: Any) -> RiskConfig:
    config = load_risk_config()
    settings: dict[str, Any] = {
        "min_recall": 0.5,
        "max_fpr": 0.5,
        "min_asset_weighted_recall": 0.5,
        "binance_gate": True,
        "latency_p95_budget_ms": 1000.0,
    }
    settings.update(overrides)
    config.settings.eval = EvalSettings(**settings)
    return config


def _all_block_decisions() -> dict[str, str]:
    return {_rid(sample_id): "BLOCK" for sample_id in BINANCE_IDS}


def _good_decisions(**overrides: str) -> dict[str, str]:
    decisions = {
        "req-pos-big": "BLOCK",
        "req-pos-small": "STEP_UP",
        "req-neg-normal": "ALLOW",
        **_all_block_decisions(),
    }
    decisions.update(overrides)
    return decisions


# ---------------------------------------------------------------------------
# 币安回归门禁
# ---------------------------------------------------------------------------


def test_binance_gate_passes_when_all_cases_block(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(monkeypatch, _good_decisions())
    report = evaluate(_dataset(), config=_config())

    assert report["passed"] is True
    assert report["exit_code"] == 0
    assert report["failures"] == []
    gate = report["binance_gate"]
    assert gate["enabled"] is True
    assert gate["total"] == len(BINANCE_SAMPLES) == 4
    assert gate["blocked"] == 4
    assert all(case["decision"] == "BLOCK" for case in gate["cases"])
    assert report["gates"]["binance_gate"]["passed"] is True


def test_binance_gate_failure_blocks_evaluation(monkeypatch: pytest.MonkeyPatch):
    failing_id = "binance-7702-variant-2"
    decisions = _good_decisions(**{_rid(failing_id): "ALLOW"})
    _install_fake_review(monkeypatch, decisions)

    report = evaluate(_dataset(), config=_config())

    assert report["passed"] is False
    assert report["exit_code"] == 2
    assert report["failures"] == ["binance_gate"]
    gate = report["binance_gate"]
    assert gate["passed"] is False
    assert gate["blocked"] == 3
    assert len(gate["failures"]) == 1
    failure = gate["failures"][0]
    assert failure["sample_id"] == failing_id
    assert failure["decision"] == "ALLOW"
    assert failure["expected_decision"] == "BLOCK"
    assert failure["blocking_rule_ids"] == ["R-001"]
    assert failure["decision_basis"]["rules_floor"] == "BLOCK"
    assert failure["decision_basis"]["final"] == "ALLOW"
    # 失败用例的证据必须进入报告（可供排查）
    assert failure["decision_basis"]["rule_hits"][0]["rule_id"] == "R-001"


def test_binance_gate_disabled_skips_gate(monkeypatch: pytest.MonkeyPatch):
    failing_id = "binance-7702-variant-3"
    _install_fake_review(monkeypatch, _good_decisions(**{_rid(failing_id): "ALLOW"}))

    report = evaluate(_dataset(), config=_config(binance_gate=False))

    assert report["passed"] is True
    assert report["failures"] == []
    assert report["binance_gate"]["enabled"] is False
    assert report["gates"]["binance_gate"]["passed"] is True
    assert "未启用" in report["binance_gate"]["note"]


def test_binance_gate_without_cases_warns(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(monkeypatch, _good_decisions())
    report = evaluate(_dataset(binance=[]), config=_config())

    assert report["passed"] is True
    assert report["binance_gate"]["total"] == 0
    assert any("币安回归用例" in warning for warning in report["warnings"])


def test_recall_gate_failure_lists_failed_item(monkeypatch: pytest.MonkeyPatch):
    # pos-small 被漏报（ALLOW）→ 召回 0.5，未达默认 0.95 门限
    decisions = _good_decisions(**{"req-pos-small": "ALLOW"})
    _install_fake_review(monkeypatch, decisions)
    config = load_risk_config()  # 默认门限：recall>=0.95 / fpr<=0.10 / asset>=0.90

    report = evaluate(_dataset(), config=config)

    assert report["passed"] is False
    assert report["exit_code"] == 2
    assert "min_recall" in report["failures"]
    assert "binance_gate" not in report["failures"]
    assert report["gates"]["binance_gate"]["passed"] is True
    assert report["gates"]["min_recall"]["value"] == pytest.approx(0.5)
    assert report["metrics"]["overall"]["recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# rules_only vs rules_plus_model 消融
# ---------------------------------------------------------------------------


def _ablation_dataset() -> EvalDataset:
    samples = [
        make_sample(
            sample_id="hard-1",
            scenario_family="native_transfer",
            expected_decision="BLOCK",
            risk_level="critical",
            exposure_ratio=0.8,
        ),
        make_sample(
            sample_id="hard-2",
            scenario_family="approve_permit",
            expected_decision="STEP_UP",
            risk_level="medium",
            exposure_ratio=0.2,
        ),
        make_sample(
            sample_id="easy-neg",
            scenario_family="normal_ops",
            expected_decision="ALLOW",
            risk_level="low",
            exposure_ratio=0.001,
        ),
        make_sample(
            sample_id="false-neg",
            scenario_family="normal_ops",
            expected_decision="ALLOW",
            risk_level="low",
            exposure_ratio=0.002,
        ),
    ]
    return EvalDataset(
        samples=samples,
        binance_samples=[],
        source="<memory>",
        stats=dataset_stats(samples),
    )


def _install_ablation_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """规则基线全部 ALLOW；模型可用后提升两条例正样本并误杀一条负样本。"""

    def fake_review(request, *, rules, reviewer=None):  # noqa: ANN001 - 与被替换函数同签名
        if reviewer is None:
            decision = "ALLOW"
        else:
            decision = {
                "req-hard-1": "BLOCK",
                "req-hard-2": "STEP_UP",
                "req-easy-neg": "ALLOW",
                "req-false-neg": "BLOCK",
            }[request.request_id]
        return _report(request, decision)

    monkeypatch.setattr(evaluate_module, "review_request", fake_review)


def test_ablation_reports_net_gain_and_false_positive_cost(monkeypatch: pytest.MonkeyPatch):
    _install_ablation_review(monkeypatch)
    config = _config(max_fpr=0.6)

    ablation = run_ablation(
        _ablation_dataset(),
        config=config,
        reviewer=_FakeReviewer(),  # type: ignore[arg-type]
    )

    assert ablation["available"] is True
    assert ablation["rules_only"]["recall"] == pytest.approx(0.0)
    assert ablation["rules_plus_model"]["recall"] == pytest.approx(1.0)
    assert ablation["delta"]["recall"] == pytest.approx(1.0)
    assert ablation["delta"]["asset_weighted_recall"] == pytest.approx(1.0)
    assert ablation["delta"]["fpr"] == pytest.approx(0.5)
    assert ablation["model_net_gain"]["newly_detected"] == ["hard-1", "hard-2"]
    assert ablation["model_net_gain"]["newly_missed"] == []
    assert ablation["extra_fp_cost"]["newly_blocked"] == ["false-neg"]
    assert ablation["extra_fp_cost"]["fpr_delta"] == pytest.approx(0.5)
    assert ablation["by_family"]["native_transfer"]["recall_delta"] == pytest.approx(1.0)
    assert ablation["by_family"]["normal_ops"]["fpr_delta"] == pytest.approx(0.5)
    assert "召回" in ablation["summary"]
    assert ablation["passed"] is True


def test_ablation_without_reviewer_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    _install_ablation_review(monkeypatch)
    ablation = run_ablation(_ablation_dataset(), config=_config())

    assert ablation["available"] is False
    assert ablation["rules_plus_model"] is None
    assert "reviewer" in ablation["reason"]
    assert ablation["rules_only"]["recall"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CLI 门禁失败 → 非零退出
# ---------------------------------------------------------------------------


def _write_cli_dataset(base: Path) -> Path:
    """写入可直接被 `lab risk eval` 加载的临时数据集（全部样本位于 test 划分）。"""
    samples = _samples()
    artifacts = base / "dataset" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / SAMPLES_FILENAME).write_text(
        "".join(f"{sample.model_dump_json()}\n" for sample in samples), encoding="utf-8"
    )
    splits = {
        "splits": {
            "train": [],
            "val": [],
            "test": [sample.sample_id for sample in samples],
        }
    }
    (artifacts / SPLITS_FILENAME).write_text(json.dumps(splits, ensure_ascii=False), encoding="utf-8")
    (artifacts / BINANCE_CASES_FILENAME).write_text(
        "".join(f"{sample.model_dump_json()}\n" for sample in BINANCE_SAMPLES), encoding="utf-8"
    )
    stats = dataset_stats(samples, splits)
    (artifacts / STATS_FILENAME).write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    return base / "dataset"


def test_cli_exit_code_nonzero_when_binance_gate_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    dataset_dir = _write_cli_dataset(tmp_path)
    decisions = _good_decisions(**{_rid("binance-7702-variant-1"): "ALLOW"})
    _install_fake_review(monkeypatch, decisions)

    code = main(
        [
            "risk",
            "eval",
            "--dataset",
            str(dataset_dir),
            "--mode",
            "rules_only",
            "--out",
            str(tmp_path / "runs"),
        ]
    )

    assert code == 2
    runs = list((tmp_path / "runs" / "risk" / "eval").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    report = json.loads((run_dir / "artifacts" / "eval_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["exit_code"] == 2
    assert report["failures"] == ["binance_gate"]
    assert report["binance_gate"]["failures"][0]["decision"] == "ALLOW"
    assert report["binance_gate"]["failures"][0]["decision_basis"]["notes"] == []
    assert (run_dir / "artifacts" / "eval_report.md").is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "risk/eval"
    assert manifest["status"] == "failed_gate"
    assert manifest["exit_code"] == 2
