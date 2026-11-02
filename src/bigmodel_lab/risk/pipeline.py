"""审查流水线：解码 → 模拟 → 特征 → 规则 → 可选模型意见 → 决策 → 报告。

两个入口：
- `review_request(request, *, rules, reviewer=None)`：纯函数式的单笔审查，返回
  `ReviewReport`；不写文件、不依赖 CLI，供评测/训练数据/集成测试复用。
- `run_review_command(args)`：`lab risk review` 的实现，支持单笔 JSON 与 JSONL
  批量、`rules_only|rules_plus_model` 模式、`--out`/`--json`，产物写入
  `RunContext("risk/review")` 的 `artifacts/` 目录（reports.jsonl、summary.json、
  reports/<request_id>.json）。

安全边界：不接触私钥、不广播交易；解码/模拟/规则配置失败一律 fail-closed；
模型只是第二意见，缺失/超时/非法输出时自动回退规则基线并在报告标记 degraded。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Protocol

from ..manifest import RunContext
from .config import RiskConfig, RuleSettings, load_risk_config
from .decision import decision_exit_code, fuse_decision, rules_floor
from .decode import DecodeError, decode_request
from .features import enrich_features
from .rules import RuleConfigError, RuleEngine
from .schema import (
    DECISION_RANK,
    SCHEMA_VERSION,
    Decision,
    ModelOpinion,
    Operation,
    ReviewReport,
    ReviewRequest,
    RuleHit,
    SimulationResult,
    exit_code_for,
)
from .simulate import simulate_request

ENGINE_VERSION = "risk-engine-v1"
SIMULATOR_VERSION = "offline-v1"
_DECISION_COUNTS_KEYS: tuple[Decision, ...] = ("ALLOW", "STEP_UP", "BLOCK", "ESCALATE")


class Reviewer(Protocol):
    """模型审查器鸭子类型：`review(request) -> ModelOpinion`。"""

    def review(self, request: ReviewRequest) -> ModelOpinion:  # pragma: no cover - 协议定义
        ...


# ---------------------------------------------------------------------------
# 单笔审查
# ---------------------------------------------------------------------------


def review_request(
    request: ReviewRequest,
    *,
    rules: RuleSettings,
    reviewer: Reviewer | None = None,
) -> ReviewReport:
    """对单笔签名请求执行完整审查，返回可审计报告。

    Args:
        request: 已通过 schema 校验的审查请求。
        rules: 规则配置（`load_risk_config().rules`）。
        reviewer: 可选模型审查器；为 None 时只跑规则基线（不导入模型模块）。

    Returns:
        `ReviewReport`；报告内决策可追溯到规则命中/模型证据。

    Raises:
        RuleConfigError: 规则配置非法（拒绝启动审查，CLI 非零退出）。
    """
    started = time.perf_counter()
    notes: list[str] = []

    operations: list[Operation] = []
    try:
        operations = decode_request(request)
        simulation = simulate_request(request, operations)
    except DecodeError as exc:
        simulation = SimulationResult(failed=True, failure_reason=f"解码失败: {exc}")
        notes.append(f"解码失败，已 fail-closed: {exc}")
    simulation = enrich_features(simulation, request, rules)

    engine = RuleEngine(rules)
    rule_hits = engine.evaluate(request, operations, simulation)

    model_opinion, degraded, opinion_notes = _collect_opinion(reviewer, request)
    notes.extend(opinion_notes)

    decision = fuse_decision(rule_hits, model_opinion)
    if simulation.failed and DECISION_RANK[decision] < DECISION_RANK["ESCALATE"]:
        decision = "ESCALATE"
        notes.append("模拟/解码失败：按 fail-closed 规则提升为 ESCALATE")
    if any(operation.kind == "unknown" for operation in operations) and (
        DECISION_RANK[decision] < DECISION_RANK["ESCALATE"]
    ):
        decision = "ESCALATE"
        notes.append("存在未知操作类型：按 fail-closed 规则提升为 ESCALATE")

    suspicious_text = any(hit.rule_id == "R-009" for hit in rule_hits)
    evidence = _build_evidence(request, operations, simulation, rule_hits, model_opinion, decision)
    generated_at = request.timestamp or "1970-01-01T00:00:00+00:00"
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    return ReviewReport(
        report_id=f"rep-{request.request_id}",
        request_id=request.request_id,
        generated_at=generated_at,
        account=request.account,
        chain=request.chain,
        operations=operations,
        simulation=simulation,
        rule_hits=rule_hits,
        model_opinion=model_opinion,
        decision=decision,
        exit_code=exit_code_for(decision),
        evidence=evidence,
        versions={
            "engine": ENGINE_VERSION,
            "schema": str(SCHEMA_VERSION),
            "rules": rules.version,
            "simulator": SIMULATOR_VERSION,
        },
        duration_ms=duration_ms,
        suspicious_text=suspicious_text,
        degraded=degraded,
        notes=notes,
    )


def _collect_opinion(
    reviewer: Reviewer | None, request: ReviewRequest
) -> tuple[ModelOpinion | None, bool, list[str]]:
    """执行可选的模型审查；任何异常都降级为不可用意见（不打断审查）。"""
    if reviewer is None:
        return None, False, []
    notes: list[str] = []
    try:
        raw = reviewer.review(request)
    except KeyboardInterrupt:  # pragma: no cover - 用户中断应向上传播
        raise
    except Exception as exc:  # noqa: BLE001 - 模型不可用不得中断审查
        reason = f"reviewer 异常: {type(exc).__name__}: {exc}"
        notes.append(f"模型审查失败，已降级为规则基线: {reason}")
        return ModelOpinion(available=False, degraded=True, degraded_reason=reason), True, notes
    if raw is None:
        reason = "reviewer 返回 None"
        notes.append(f"模型审查失败，已降级为规则基线: {reason}")
        return ModelOpinion(available=False, degraded=True, degraded_reason=reason), True, notes
    if not isinstance(raw, ModelOpinion):
        try:
            raw = ModelOpinion.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 - 非法输出同样降级
            reason = f"模型输出不合法: {type(exc).__name__}: {exc}"
            notes.append(f"模型审查失败，已降级为规则基线: {reason}")
            return ModelOpinion(available=False, degraded=True, degraded_reason=reason), True, notes
    degraded = bool(raw.degraded or not raw.available)
    if degraded:
        reason = raw.degraded_reason or "模型不可用/降级"
        notes.append(f"模型意见不可用，规则基线继续生效: {reason}")
    return raw, degraded, notes


def _build_evidence(
    request: ReviewRequest,
    operations: list[Operation],
    simulation: SimulationResult,
    rule_hits: list[RuleHit],
    model_opinion: ModelOpinion | None,
    decision: Decision,
) -> list[str]:
    """汇总报告证据：模拟摘要、逐条规则命中理由、模型意见与最终决策依据。"""
    evidence = [
        f"request={request.request_id} account={request.account} chain={request.chain}",
        f"operations={len(operations)} kinds="
        + ",".join(sorted({operation.kind for operation in operations}))
        if operations
        else "operations=0",
    ]
    if simulation.failed:
        evidence.append(f"simulation_failed: {simulation.failure_reason}")
    else:
        evidence.append(
            "simulation: "
            f"capability_delta={simulation.capability_delta}, "
            f"exposure_usd={simulation.exposure_usd}, "
            f"exposure_ratio={simulation.exposure_ratio}, "
            f"nav_usd={simulation.nav_usd}, "
            f"rolling_exposure_ratio={simulation.rolling_exposure_ratio}, "
            f"cross_chain_exposure_ratio={simulation.cross_chain_exposure_ratio}"
        )
    for hit in rule_hits:
        evidence.append(
            f"{hit.rule_id} {hit.name}({hit.severity}) → {hit.suggested_decision}: "
            + "；".join(hit.evidence)
        )
    if model_opinion is not None:
        if model_opinion.available:
            evidence.append(
                "model: "
                f"risk_level={model_opinion.risk_level}, "
                f"action={model_opinion.recommended_action}, "
                f"confidence={model_opinion.confidence}, "
                f"factors={model_opinion.risk_factors}"
            )
        else:
            evidence.append(f"model_degraded: {model_opinion.degraded_reason or '不可用'}")
    evidence.append(f"rules_floor={rules_floor(rule_hits)} final={decision}")
    return evidence


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def run_review_command(args: argparse.Namespace) -> int:
    """实现 `lab risk review`：加载配置、审查单笔/批量、写报告与汇总。

    Args:
        args: argparse 命名空间，需含 request|batch、mode、out、json、config。

    Returns:
        退出码：0=无阻断；2=存在 BLOCK；3=ESCALATE/无法完成审查（批量取最严重）。
    """
    config_paths = list(getattr(args, "config", []) or [])
    config = load_risk_config(config_paths or None)
    mode = getattr(args, "mode", None) or config.settings.review.mode
    if mode not in {"rules_only", "rules_plus_model"}:
        raise ValueError(f"未知审查模式: {mode!r}，可选 rules_only|rules_plus_model")
    requests = _load_requests(args, max_batch=config.settings.review.max_batch)

    reviewer: Reviewer | None = None
    fallback_reason: str | None = None
    if mode == "rules_plus_model":
        reviewer, fallback_reason = _build_reviewer_from_config(config)
        if fallback_reason is not None:
            print(f"警告: 模型审查不可用，已降级为 rules_only（{fallback_reason}）", file=sys.stderr)

    out = getattr(args, "out", None)
    runs_dir = Path(out) if out else config.lab.resolved_dir("runs_dir")
    ctx = RunContext.create("risk/review", config.raw, runs_dir=runs_dir)
    try:
        reports = [
            review_request(request, rules=config.rules, reviewer=reviewer) for request in requests
        ]
        if fallback_reason is not None:
            note = f"模型不可用，已降级为 rules_only: {fallback_reason}"
            reports = [
                report.model_copy(update={"degraded": True, "notes": [*report.notes, note]})
                for report in reports
            ]
        exit_code = aggregate_exit_code(reports)
        summary = _build_summary(reports, config, mode, fallback_reason, exit_code)
        _write_artifacts(ctx, reports, summary)
        ctx.log_metrics(**summary)
        ctx.finish("success", exit_code)
    except (RuleConfigError, ValueError):
        ctx.finish("failed", 2, "配置/参数错误")
        raise
    except Exception as exc:  # noqa: BLE001 - 审查失败必须 fail-closed
        reason = f"{type(exc).__name__}: {exc}"
        ctx.finish("failed", 3, reason)
        print(f"审查失败（fail-closed）: {reason}", file=sys.stderr)
        return 3

    _emit_output(reports, summary, ctx, as_json=bool(getattr(args, "json", False)))
    return exit_code


def aggregate_exit_code(reports: list[ReviewReport]) -> int:
    """批量汇总退出码：任一 ESCALATE→3，否则任一 BLOCK→2，否则 0。"""
    if any(report.decision == "ESCALATE" for report in reports):
        return 3
    if any(report.decision == "BLOCK" for report in reports):
        return 2
    return 0


def _build_reviewer_from_config(config: RiskConfig) -> tuple[Reviewer | None, str | None]:
    """惰性构造模型审查器；任何导入/构造失败都返回降级原因而不是抛错。"""
    try:
        from .infer import build_reviewer  # noqa: PLC0415 - 规则模式不导入模型模块
    except Exception as exc:  # noqa: BLE001 - 模型模块缺失/依赖不全
        return None, f"导入 risk.infer 失败: {type(exc).__name__}: {exc}"
    try:
        reviewer = build_reviewer(config)
    except Exception as exc:  # noqa: BLE001 - 模型不可用必须降级
        return None, f"构造模型审查器失败: {type(exc).__name__}: {exc}"
    if reviewer is None:
        return None, "build_reviewer 返回 None"
    return reviewer, None  # type: ignore[return-value]


def _load_requests(args: argparse.Namespace, *, max_batch: int) -> list[ReviewRequest]:
    request_path = getattr(args, "request", None)
    batch_path = getattr(args, "batch", None)
    if request_path and batch_path:
        raise ValueError("--request 与 --batch 互斥，只能提供一个")
    if request_path:
        return [_load_single_request(Path(request_path))]
    if batch_path:
        return _load_batch_requests(Path(batch_path), max_batch=max_batch)
    raise ValueError("必须提供 --request（单笔 JSON）或 --batch（JSONL）")


def _load_single_request(path: Path) -> ReviewRequest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"请求文件不是合法 JSON: {path}: {exc}") from exc
    return _request_from_mapping(data, source=str(path))


def _load_batch_requests(path: Path, *, max_batch: int) -> list[ReviewRequest]:
    requests: list[ReviewRequest] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise
    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"批量文件第 {line_number} 行不是合法 JSON: {path}: {exc}") from exc
        requests.append(_request_from_mapping(data, source=f"{path}:{line_number}"))
        if len(requests) > max_batch:
            raise ValueError(f"批量请求数超过配置上限 review.max_batch={max_batch}")
    if not requests:
        raise ValueError(f"批量文件没有有效请求: {path}")
    return requests


def _request_from_mapping(data: Any, *, source: str) -> ReviewRequest:
    if isinstance(data, dict) and "request_id" not in data and isinstance(data.get("request"), dict):
        # 兼容 RiskSample 包装：{"sample_id":..., "request": {...}, "label": {...}}
        data = data["request"]
    if not isinstance(data, dict):
        raise ValueError(f"{source}: 请求必须是 JSON 对象")
    try:
        return ReviewRequest.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError 统一转参数错误
        raise ValueError(f"{source}: 请求不符合 schema: {exc}") from exc


def _build_summary(
    reports: list[ReviewReport],
    config: RiskConfig,
    mode: str,
    fallback_reason: str | None,
    exit_code: int,
) -> dict[str, Any]:
    decisions: dict[str, int] = {decision: 0 for decision in _DECISION_COUNTS_KEYS}
    for report in reports:
        decisions[report.decision] = decisions.get(report.decision, 0) + 1
    ratios = [float(report.simulation.exposure_ratio) for report in reports]
    summary: dict[str, Any] = {
        "mode": mode,
        "rules_version": config.rules.version,
        "engine_version": ENGINE_VERSION,
        "simulator": config.settings.simulator,
        "total": len(reports),
        "decisions": decisions,
        "exit_code": exit_code,
        "degraded": any(report.degraded for report in reports),
        "suspicious_text": any(report.suspicious_text for report in reports),
        "request_ids": [report.request_id for report in reports],
        "report_ids": [report.report_id for report in reports],
        "max_exposure_ratio": round(max(ratios), 6) if ratios else 0.0,
    }
    if fallback_reason is not None:
        summary["degraded_reason"] = fallback_reason
    return summary


def _write_artifacts(
    ctx: RunContext, reports: list[ReviewReport], summary: dict[str, Any]
) -> None:
    """写 reports.jsonl、summary.json 与逐笔 reports/<request_id>.json。"""
    lines = [
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False) for report in reports
    ]
    ctx.write_artifact("reports.jsonl", "\n".join(lines) + "\n")
    ctx.write_json_artifact("summary.json", summary)
    used_names: dict[str, int] = {}
    for report in reports:
        base = _safe_report_name(report.request_id)
        count = used_names.get(base, 0) + 1
        used_names[base] = count
        name = base if count == 1 else f"{base}-{count}"
        ctx.write_json_artifact(f"reports/{name}.json", report.model_dump(mode="json"))


def _safe_report_name(request_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(request_id)).strip("._")
    return (cleaned or "request")[:120]


def _emit_output(
    reports: list[ReviewReport],
    summary: dict[str, Any],
    ctx: RunContext,
    *,
    as_json: bool,
) -> None:
    """输出报告：--json 时机器可读，否则人类可读摘要。"""
    if as_json:
        if len(reports) == 1:
            payload: Any = reports[0].model_dump(mode="json")
        else:
            payload = {
                "summary": summary,
                "reports": [report.model_dump(mode="json") for report in reports],
            }
        print(json.dumps(payload, ensure_ascii=False))
        print(f"产物目录: {ctx.run_dir}", file=sys.stderr)
        return
    for report in reports:
        print(
            f"{report.report_id}: decision={report.decision} "
            f"exit={decision_exit_code(report.decision)} "
            f"exposure_ratio={report.simulation.exposure_ratio} "
            f"capability={report.simulation.capability_delta} "
            f"rules={len(report.rule_hits)}"
            + (" [degraded]" if report.degraded else "")
        )
        if report.suspicious_text:
            print("  ! 检测到不可信文本注入特征（suspicious_text=True）")
        for hit in report.rule_hits:
            if hit.suggested_decision in {"BLOCK", "ESCALATE"}:
                print(f"  - {hit.rule_id} {hit.name}: {hit.suggested_decision} / {hit.severity}")
    print(
        "汇总: "
        f"total={summary['total']} decisions={summary['decisions']} "
        f"exit_code={summary['exit_code']} degraded={summary['degraded']}"
    )
    print(f"产物目录: {ctx.run_dir}")


__all__ = [
    "ENGINE_VERSION",
    "Reviewer",
    "SIMULATOR_VERSION",
    "aggregate_exit_code",
    "review_request",
    "run_review_command",
]
