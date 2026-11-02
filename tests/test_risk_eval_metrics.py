"""`risk.evaluate` 核心指标测试：用固定输入手工验证召回/FPR/资产加权/延迟/非法输出率。"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from test_risk_model_data import make_sample

from bigmodel_lab.risk import evaluate as evaluate_module
from bigmodel_lab.risk.config import EvalSettings, RiskConfig, load_risk_config
from bigmodel_lab.risk.dataset import dataset_stats
from bigmodel_lab.risk.evaluate import (
    EvalDataset,
    compare_eval_reports,
    evaluate,
    percentile,
)
from bigmodel_lab.risk.schema import (
    ModelOpinion,
    ReviewReport,
    RuleHit,
    SimulationResult,
    exit_code_for,
)

# 固定输入：3 正 + 2 负；决策映射与耗时完全手工指定，不依赖真实模型质量。
DECISIONS = {
    "req-pos-block": "BLOCK",
    "req-pos-stepup": "STEP_UP",
    "req-pos-miss": "ALLOW",
    "req-neg-block": "BLOCK",
    "req-neg-allow": "ALLOW",
}
DURATIONS = {
    "req-pos-block": 1.0,
    "req-pos-stepup": 2.0,
    "req-pos-miss": 5.0,
    "req-neg-block": 4.0,
    "req-neg-allow": 3.0,
}


def _make_samples() -> list:
    return [
        make_sample(
            sample_id="pos-block",
            scenario_family="native_transfer",
            expected_decision="BLOCK",
            risk_level="critical",
            capability_delta="value_transfer",
            exposure_ratio=0.5,
        ),
        make_sample(
            sample_id="pos-stepup",
            scenario_family="approve_permit",
            expected_decision="STEP_UP",
            risk_level="medium",
            capability_delta="allowance_grant",
            exposure_ratio=0.1,
        ),
        make_sample(
            sample_id="pos-miss",
            scenario_family="native_transfer",
            expected_decision="BLOCK",
            risk_level="critical",
            capability_delta="value_transfer",
            exposure_ratio=1.0,
        ),
        make_sample(
            sample_id="neg-block",
            scenario_family="normal_ops",
            expected_decision="ALLOW",
            risk_level="low",
            capability_delta="value_transfer",
            exposure_ratio=0.001,
        ),
        make_sample(
            sample_id="neg-allow",
            scenario_family="normal_ops",
            expected_decision="ALLOW",
            risk_level="low",
            capability_delta="value_transfer",
            exposure_ratio=0.002,
        ),
    ]


def _eval_dataset(samples: list, binance: list | None = None) -> EvalDataset:
    return EvalDataset(
        samples=list(samples),
        binance_samples=list(binance or []),
        source="<memory>",
        stats=dataset_stats(list(samples)),
    )


def _rule(
    rule_id: str = "R-001",
    *,
    name: str = "不可逆操作",
    severity: str = "critical",
    suggested: str = "BLOCK",
) -> RuleHit:
    return RuleHit(
        rule_id=rule_id,
        name=name,
        severity=severity,  # type: ignore[arg-type]
        evidence=["测试证据"],
        rule_version="risk-rules-v1",
        suggested_decision=suggested,  # type: ignore[arg-type]
    )


def _report(
    request: Any,
    decision: str,
    *,
    duration_ms: float = 1.0,
    rule_hits: list[RuleHit] | None = None,
    opinion: ModelOpinion | None = None,
    exposure_usd: float = 0.0,
) -> ReviewReport:
    return ReviewReport(
        report_id=f"rep-{request.request_id}",
        request_id=request.request_id,
        generated_at=request.timestamp,
        account=request.account,
        chain=request.chain,
        simulation=SimulationResult(exposure_usd=exposure_usd),
        rule_hits=list(rule_hits or []),
        model_opinion=opinion,
        decision=decision,  # type: ignore[arg-type]
        exit_code=exit_code_for(decision),
        duration_ms=duration_ms,
    )


def _install_fake_review(
    monkeypatch: pytest.MonkeyPatch,
    *,
    decisions: dict[str, str],
    durations: dict[str, float] | None = None,
    rule_hits: dict[str, list[RuleHit]] | None = None,
    opinions: dict[str, ModelOpinion] | None = None,
    exposure_usd: dict[str, float] | None = None,
) -> None:
    """按 request_id 映射固定决策，替换真实审查流水线。"""

    def fake_review(request, *, rules, reviewer=None):  # noqa: ANN001 - 与被替换函数同签名
        decision = decisions[request.request_id]
        opinion = opinions.get(request.request_id) if (opinions and reviewer is not None) else None
        return _report(
            request,
            decision,
            duration_ms=(durations or {}).get(request.request_id, 1.0),
            rule_hits=(rule_hits or {}).get(request.request_id),
            opinion=opinion,
            exposure_usd=(exposure_usd or {}).get(request.request_id, 0.0),
        )

    monkeypatch.setattr(evaluate_module, "review_request", fake_review)


def _config(**overrides: Any) -> RiskConfig:
    config = load_risk_config()
    settings: dict[str, Any] = {
        "min_recall": 0.95,
        "max_fpr": 0.10,
        "min_asset_weighted_recall": 0.90,
        "binance_gate": True,
        "latency_p95_budget_ms": 5000.0,
    }
    settings.update(overrides)
    config.settings.eval = EvalSettings(**settings)
    return config


# ---------------------------------------------------------------------------
# 召回 / FPR / 分场景族
# ---------------------------------------------------------------------------


def test_handcrafted_recall_fpr_and_family_metrics(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(
        monkeypatch,
        decisions=DECISIONS,
        durations=DURATIONS,
        rule_hits={
            "req-pos-block": [_rule()],
            "req-neg-block": [_rule("R-007", name="金额阈值", severity="medium", suggested="STEP_UP")],
        },
    )
    report = evaluate(_eval_dataset(_make_samples()), config=_config())
    overall = report["metrics"]["overall"]

    assert overall["total"] == 5
    assert overall["positive"] == 3
    assert overall["negative"] == 2
    assert overall["detected"] == 2
    assert overall["recall"] == pytest.approx(2 / 3, abs=1e-6)
    assert overall["fpr"] == pytest.approx(0.5)
    assert overall["false_positives"] == 1
    assert overall["false_negatives"] == 1
    assert overall["expected_match_rate"] == pytest.approx(3 / 5)
    assert overall["under_reaction"] == 1
    assert overall["over_reaction"] == 1

    by_family = report["metrics"]["by_family"]
    assert by_family["native_transfer"]["total"] == 2
    assert by_family["native_transfer"]["recall"] == pytest.approx(0.5)
    assert by_family["approve_permit"]["recall"] == pytest.approx(1.0)
    assert by_family["normal_ops"]["negative"] == 2
    assert by_family["normal_ops"]["fpr"] == pytest.approx(0.5)


def test_decision_distribution_and_rule_hit_distribution(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(
        monkeypatch,
        decisions=DECISIONS,
        rule_hits={
            "req-pos-block": [_rule("R-001")],
            "req-pos-miss": [_rule("R-001")],
            "req-neg-block": [_rule("R-007", suggested="STEP_UP", severity="medium")],
        },
    )
    report = evaluate(_eval_dataset(_make_samples()), config=_config())

    assert report["metrics"]["decision_distribution"] == {
        "ALLOW": 2,
        "STEP_UP": 1,
        "BLOCK": 2,
        "ESCALATE": 0,
    }
    assert report["metrics"]["expected_decision_distribution"] == {
        "ALLOW": 2,
        "STEP_UP": 1,
        "BLOCK": 2,
        "ESCALATE": 0,
    }
    rule_metrics = report["metrics"]["rules"]
    assert rule_metrics["total_hits"] == 3
    assert rule_metrics["distinct_rules"] == 2
    distribution = {item["rule_id"]: item["count"] for item in rule_metrics["hit_distribution"]}
    assert distribution == {"R-001": 2, "R-007": 1}
    assert report["per_sample"][0]["rule_hits"][0]["rule_id"] == "R-001"


# ---------------------------------------------------------------------------
# 资产加权召回
# ---------------------------------------------------------------------------


def test_asset_weighted_recall_uses_label_exposure_ratio(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(monkeypatch, decisions=DECISIONS)
    report = evaluate(_eval_dataset(_make_samples()), config=_config())
    asset = report["metrics"]["asset_weighted_recall"]

    assert asset["basis"] == "label.exposure_ratio"
    # 检出权重 0.5 + 0.1，总权重 0.5 + 0.1 + 1.0
    assert asset["value"] == pytest.approx(0.6 / 1.6)
    assert asset["total_weight"] == pytest.approx(1.6)
    assert asset["detected_weight"] == pytest.approx(0.6)


def test_asset_weighted_recall_falls_back_to_simulation_exposure(monkeypatch: pytest.MonkeyPatch):
    samples = [
        make_sample(sample_id="pos-a", expected_decision="BLOCK", exposure_ratio=0.0),
        make_sample(sample_id="pos-b", expected_decision="BLOCK", exposure_ratio=0.0),
    ]
    _install_fake_review(
        monkeypatch,
        decisions={"req-pos-a": "BLOCK", "req-pos-b": "ALLOW"},
        exposure_usd={"req-pos-a": 100.0, "req-pos-b": 50.0},
    )
    asset = evaluate(_eval_dataset(samples), config=_config())["metrics"]["asset_weighted_recall"]

    assert asset["basis"] == "simulation.exposure_usd"
    assert asset["value"] == pytest.approx(100.0 / 150.0)


def test_asset_weighted_recall_falls_back_to_count(monkeypatch: pytest.MonkeyPatch):
    samples = [
        make_sample(sample_id="pos-a", expected_decision="BLOCK", exposure_ratio=0.0),
        make_sample(sample_id="pos-b", expected_decision="STEP_UP", exposure_ratio=0.0),
    ]
    _install_fake_review(
        monkeypatch,
        decisions={"req-pos-a": "STEP_UP", "req-pos-b": "ALLOW"},
    )
    asset = evaluate(_eval_dataset(samples), config=_config())["metrics"]["asset_weighted_recall"]

    assert asset["basis"] == "count"
    assert asset["value"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 延迟统计
# ---------------------------------------------------------------------------


def test_percentile_linear_interpolation():
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95) == pytest.approx(4.8)
    assert percentile([7.0], 95) == pytest.approx(7.0)
    assert percentile([], 50) == 0.0


def test_latency_p50_p95_and_slowest_samples(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(monkeypatch, decisions=DECISIONS, durations=DURATIONS)
    latency = evaluate(_eval_dataset(_make_samples()), config=_config())["metrics"]["latency"]

    assert latency["count"] == 5
    assert latency["p50_ms"] == pytest.approx(3.0)
    assert latency["p95_ms"] == pytest.approx(4.8)
    assert latency["max_ms"] == pytest.approx(5.0)
    assert latency["budget_ms"] == pytest.approx(5000.0)
    assert latency["within_budget"] is True
    assert [item["sample_id"] for item in latency["slowest_samples"]][:2] == [
        "pos-miss",
        "neg-block",
    ]
    slowest = latency["slowest_samples"][0]
    assert slowest["duration_ms"] == pytest.approx(5.0)
    assert slowest["decision"] == "ALLOW"
    assert slowest["operations"] == 0  # fake 报告未携带 operations


# ---------------------------------------------------------------------------
# 模型非法输出率
# ---------------------------------------------------------------------------


class _FakeReviewer:
    version = "fake-model-v1"

    def review(self, request):  # noqa: ANN001, ANN201 - 评测测试用 fake
        return ModelOpinion(available=True, recommended_action="BLOCK")


def test_model_invalid_output_rate(monkeypatch: pytest.MonkeyPatch):
    samples = [
        make_sample(sample_id="m-1", expected_decision="BLOCK"),
        make_sample(sample_id="m-2", expected_decision="BLOCK"),
        make_sample(sample_id="m-3", expected_decision="BLOCK"),
    ]
    opinions = {
        "req-m-1": ModelOpinion(
            available=False, degraded=True, degraded_reason="timeout", invalid_output=True
        ),
        "req-m-2": ModelOpinion(available=False, degraded=True, degraded_reason="timeout"),
        "req-m-3": ModelOpinion(available=True, recommended_action="BLOCK", confidence=0.8),
    }
    _install_fake_review(
        monkeypatch,
        decisions={"req-m-1": "BLOCK", "req-m-2": "BLOCK", "req-m-3": "BLOCK"},
        opinions=opinions,
    )
    reviewer = _FakeReviewer()
    report = evaluate(
        _eval_dataset(samples),
        config=_config(),
        mode="rules_plus_model",
        reviewer=reviewer,  # type: ignore[arg-type]
    )
    model = report["metrics"]["model"]

    assert model["enabled"] is True
    assert model["opinions"] == 3
    assert model["available"] == 1
    assert model["degraded"] == 2
    assert model["invalid_output"] == 1
    assert model["invalid_output_rate"] == pytest.approx(2 / 3)
    assert model["recommendation_distribution"] == {"BLOCK": 1}
    assert model["degraded_reasons"] == {"timeout": 2}
    assert report["versions"]["model"] == "fake-model-v1"


def test_rules_only_mode_does_not_use_model(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(monkeypatch, decisions=DECISIONS)
    reviewer = _FakeReviewer()
    report = evaluate(
        _eval_dataset(_make_samples()),
        config=_config(),
        mode="rules_only",
        reviewer=reviewer,  # type: ignore[arg-type]
    )

    assert report["metrics"]["model"]["enabled"] is False
    assert report["metrics"]["model"]["invalid_output_rate"] == 0.0
    assert report["versions"]["model"] == "rules_only"
    assert any("忽略传入的 reviewer" in warning for warning in report["warnings"])


# ---------------------------------------------------------------------------
# 门限判定与报告对比
# ---------------------------------------------------------------------------


def test_gate_threshold_failures_are_reported(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(monkeypatch, decisions=DECISIONS, durations=DURATIONS)
    report = evaluate(_eval_dataset(_make_samples()), config=_config())

    assert report["passed"] is False
    assert report["exit_code"] == 2
    assert set(report["failures"]) == {"min_recall", "max_fpr", "min_asset_weighted_recall"}
    assert report["gates"]["min_recall"]["passed"] is False
    assert report["gates"]["max_fpr"]["passed"] is False
    assert report["gates"]["min_asset_weighted_recall"]["passed"] is False
    assert report["gates"]["latency_p95_budget_ms"]["passed"] is True


def test_gate_passes_with_relaxed_thresholds(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(monkeypatch, decisions=DECISIONS, durations=DURATIONS)
    report = evaluate(
        _eval_dataset(_make_samples()),
        config=_config(min_recall=0.5, max_fpr=0.6, min_asset_weighted_recall=0.2, binance_gate=False),
    )

    assert report["passed"] is True
    assert report["exit_code"] == 0
    assert report["failures"] == []
    assert report["gates"]["binance_gate"]["enabled"] is False


def test_latency_budget_gate_failure(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(monkeypatch, decisions=DECISIONS, durations=DURATIONS)
    report = evaluate(
        _eval_dataset(_make_samples()),
        config=_config(
            min_recall=0.5,
            max_fpr=0.6,
            min_asset_weighted_recall=0.2,
            binance_gate=False,
            latency_p95_budget_ms=1.0,
        ),
    )

    assert report["passed"] is False
    assert report["failures"] == ["latency_p95_budget_ms"]


def test_compare_eval_reports_detects_tolerance_and_gate_changes(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(monkeypatch, decisions=DECISIONS, durations=DURATIONS)
    base_config = _config(min_recall=0.5, max_fpr=0.6, min_asset_weighted_recall=0.2, binance_gate=False)
    report = evaluate(_eval_dataset(_make_samples()), config=base_config)

    same = compare_eval_reports(report, copy.deepcopy(report))
    assert same["within_tolerance"] is True
    assert same["passed"] is True
    assert same["regressions"] == []
    assert all(item["passed"] for item in same["metric_differences"])

    changed = copy.deepcopy(report)
    changed["metrics"]["overall"]["recall"] = 0.5
    changed["failures"] = ["max_fpr"]
    changed["passed"] = False
    comparison = compare_eval_reports(report, changed)

    assert comparison["within_tolerance"] is False
    assert comparison["decision_distribution"]["changed"] is False
    assert any("max_fpr" in item for item in comparison["regressions"])
    assert comparison["previous"]["passed"] is True
    assert comparison["current"]["passed"] is False


def test_evaluate_accepts_plain_sample_sequence(monkeypatch: pytest.MonkeyPatch):
    _install_fake_review(monkeypatch, decisions=DECISIONS)
    report = evaluate(_make_samples(), config=_config())
    assert report["dataset"]["source"] == "<memory>"
    assert report["metrics"]["overall"]["total"] == 5
    assert report["metrics"]["overall"]["recall"] == pytest.approx(2 / 3)
