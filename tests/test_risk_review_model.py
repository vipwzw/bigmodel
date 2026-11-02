"""规则 + 模型集成测试：`review_request(..., reviewer=...)` 的融合与降级。

对应 tasks 6.3：模型只能维持或提升决策；模型不可用/超时/非法输出时回退 rules_only
并在报告标记 degraded。引擎代码存在时跑完整集成，缺失时跳过以避免并行开发期间互相阻塞。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_risk_infer import FakeGenerator
from test_risk_model_data import ATTACKER, COUNTERPARTY, HOT_WALLET, make_sample

from bigmodel_lab.risk import infer
from bigmodel_lab.risk.config import load_risk_config
from bigmodel_lab.risk.model_data import build_gold_report
from bigmodel_lab.risk.schema import ModelOpinion, ReviewReport, canonical_json

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("bigmodel_lab.risk.pipeline") is None,
    reason="引擎风险流水线尚未就绪（依赖 task-2），跳过规则+模型集成测试",
)

if importlib.util.find_spec("bigmodel_lab.risk.pipeline") is not None:
    from bigmodel_lab.risk import pipeline
else:  # pragma: no cover - 上面的 skip 已生效
    pipeline = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 合成请求：一个规则地板 ALLOW、一个规则地板 BLOCK
# ---------------------------------------------------------------------------


def _allow_request() -> Any:
    sample = make_sample(sample_id="allow-0001", scenario_family="native_transfer")
    sample.request.context.known_counterparties = [COUNTERPARTY]
    return sample


def _block_request() -> Any:
    from bigmodel_lab.risk.schema import Operation

    operation = Operation(
        kind="eip7702_authorization",
        chain="ethereum",
        account=HOT_WALLET,
        target=ATTACKER,
        delegate=ATTACKER,
        nonce=50,
        nonce_span=50,
        batch_size=50,
    )
    sample = make_sample(
        sample_id="binance-7702",
        scenario_family="eip7702_delegation",
        operation=operation,
        expected_decision="BLOCK",
        risk_level="critical",
        capability_delta="full_control",
        exposure_ratio=1.0,
        action_type="eip7702_authorization",
    )
    sample.request.context.origin = "devops-container"
    return sample


def _opinion(**overrides: Any) -> ModelOpinion:
    values: dict[str, Any] = {
        "available": True,
        "risk_level": "high",
        "recommended_action": "STEP_UP",
        "capability_delta": "allowance_grant",
        "confidence": 0.8,
        "risk_factors": ["模型发现的新风险"],
        "evidence": ["model:test"],
        "raw_output": "{}",
    }
    values.update(overrides)
    return ModelOpinion(**values)


class FakeReviewer:
    """按需返回 ModelOpinion 或抛异常的假审查器（鸭子类型，满足 pipeline.Reviewer）。"""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[Any] = []

    def review(self, request):  # noqa: ANN001, ANN202
        self.calls.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


# ---------------------------------------------------------------------------
# 融合语义
# ---------------------------------------------------------------------------


def test_rules_only_has_no_model_opinion() -> None:
    cfg = load_risk_config()
    request = _allow_request()
    report = pipeline.review_request(request.request, rules=cfg.rules)
    assert report.model_opinion is None
    assert report.decision == "ALLOW"
    assert report.degraded is False
    assert isinstance(report, ReviewReport)


def test_model_can_raise_allow_to_block() -> None:
    cfg = load_risk_config()
    reviewer = FakeReviewer(_opinion(recommended_action="BLOCK", risk_level="critical"))
    report = pipeline.review_request(_allow_request().request, rules=cfg.rules, reviewer=reviewer)
    assert report.decision == "BLOCK"
    assert report.exit_code == 2
    assert report.model_opinion is not None and report.model_opinion.available is True
    assert report.degraded is False
    assert any("model:" in item for item in report.evidence)


def test_model_cannot_lower_rule_block() -> None:
    cfg = load_risk_config()
    for action in ("ALLOW", "STEP_UP", "ESCALATE"):
        reviewer = FakeReviewer(_opinion(recommended_action=action))
        report = pipeline.review_request(_block_request().request, rules=cfg.rules, reviewer=reviewer)
        assert report.decision == "BLOCK", action
        assert report.exit_code == 2
        assert not report.degraded


def test_model_available_degraded_flag_marks_report() -> None:
    cfg = load_risk_config()
    unavailable = ModelOpinion(available=False, degraded=True, degraded_reason="模型推理超时")
    report = pipeline.review_request(_block_request().request, rules=cfg.rules, reviewer=FakeReviewer(unavailable))
    assert report.degraded is True
    assert report.decision == "BLOCK"  # 规则地板不降低
    assert report.model_opinion is not None
    assert report.model_opinion.available is False
    assert any("超时" in item for item in report.notes)


def test_model_exception_falls_back_to_rules_only() -> None:
    cfg = load_risk_config()
    reviewer = FakeReviewer(RuntimeError("generator 崩溃"))
    report = pipeline.review_request(_block_request().request, rules=cfg.rules, reviewer=reviewer)
    assert report.degraded is True
    assert report.decision == "BLOCK"
    assert report.model_opinion is not None and report.model_opinion.available is False
    assert any("generator 崩溃" in item for item in report.notes)


def test_model_opinion_returned_as_dict_is_validated() -> None:
    cfg = load_risk_config()
    reviewer = FakeReviewer({"available": True, "recommended_action": "BLOCK", "risk_level": "high"})
    report = pipeline.review_request(_allow_request().request, rules=cfg.rules, reviewer=reviewer)
    assert report.decision == "BLOCK"
    assert report.model_opinion is not None and report.model_opinion.available is True


# ---------------------------------------------------------------------------
# ModelReviewer 端到端（fake generator，不加载真实模型）
# ---------------------------------------------------------------------------


def _reviewer_config(tmp_path: Path) -> Any:
    cfg = load_risk_config()
    cfg.settings.model.train.base_model_dir = str(tmp_path / "base-model")
    cfg.settings.model.train.export_dir = str(tmp_path / "export-model")
    (tmp_path / "base-model").mkdir()
    (tmp_path / "export-model").mkdir()
    return cfg


def test_model_reviewer_integration_with_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _reviewer_config(tmp_path)
    sample = _allow_request()
    gold = canonical_json(build_gold_report(sample))
    generator = FakeGenerator([gold])
    monkeypatch.setattr(infer, "build_generator", lambda backend, model_dir: generator)  # noqa: ARG005
    reviewer = infer.build_reviewer(cfg)
    report = pipeline.review_request(sample.request, rules=cfg.rules, reviewer=reviewer)

    assert report.degraded is False
    assert report.model_opinion is not None and report.model_opinion.available is True
    # gold 报告来自 label（ALLOW），规则地板 ALLOW → 最终不高于模型建议
    assert report.model_opinion.recommended_action == sample.label.expected_decision
    assert report.decision == "ALLOW"


def test_model_reviewer_invalid_output_degrades_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _reviewer_config(tmp_path)
    sample = _block_request()
    generator = FakeGenerator(["这不是 JSON"])
    monkeypatch.setattr(infer, "build_generator", lambda backend, model_dir: generator)  # noqa: ARG005
    reviewer = infer.build_reviewer(cfg)
    report = pipeline.review_request(sample.request, rules=cfg.rules, reviewer=reviewer)
    assert report.degraded is True
    assert report.model_opinion is not None
    assert report.model_opinion.invalid_output is True
    assert report.decision == "BLOCK"


# ---------------------------------------------------------------------------
# CLI 模式集成：rules_plus_model 与模型不可用回退
# ---------------------------------------------------------------------------


def _write_request(tmp_path: Path, request: Any) -> Path:
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
    return path


def _review_args(request_path: Path, out_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        request=str(request_path),
        batch=None,
        mode="rules_plus_model",
        out=str(out_dir),
        json=False,
        config=[],
    )


def test_review_cli_rules_plus_model_uses_reviewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = _block_request()
    request_path = _write_request(tmp_path, sample.request)
    reviewer = FakeReviewer(_opinion(recommended_action="ALLOW"))
    monkeypatch.setattr(pipeline, "_build_reviewer_from_config", lambda config: (reviewer, None))  # noqa: ARG005
    exit_code = pipeline.run_review_command(_review_args(request_path, tmp_path / "review"))
    assert exit_code == 2  # 规则 BLOCK 不被模型降低
    report_files = list((tmp_path / "review").rglob("reports/*.json"))
    assert report_files
    report = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert report["decision"] == "BLOCK"
    assert report["model_opinion"]["recommended_action"] == "ALLOW"
    assert reviewer.calls, "模型审查器应被调用"


def test_review_cli_model_unavailable_falls_back_to_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = _block_request()
    request_path = _write_request(tmp_path, sample.request)
    monkeypatch.setattr(
        pipeline,
        "_build_reviewer_from_config",
        lambda config: (None, "未找到审查模型"),  # noqa: ARG005
    )
    exit_code = pipeline.run_review_command(_review_args(request_path, tmp_path / "review"))
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "降级为 rules_only" in captured.err
    report_file = next((tmp_path / "review").rglob("reports/*.json"))
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["degraded"] is True
    assert report["decision"] == "BLOCK"
    assert any("未找到审查模型" in note for note in report["notes"])


def test_review_cli_rules_only_never_imports_model_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = _allow_request()
    request_path = _write_request(tmp_path, sample.request)
    args = _review_args(request_path, tmp_path / "review")
    args.mode = "rules_only"
    called = {"value": False}

    def fail_build(config):  # noqa: ANN001, ARG001
        called["value"] = True
        raise AssertionError("rules_only 不应构造模型审查器")

    monkeypatch.setattr(pipeline, "_build_reviewer_from_config", fail_build)
    exit_code = pipeline.run_review_command(args)
    assert exit_code == 0
    assert called["value"] is False
    report_file = next((tmp_path / "review").rglob("reports/*.json"))
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["model_opinion"] is None
