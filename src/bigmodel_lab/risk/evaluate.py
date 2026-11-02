"""交易签名风险审查的评测：指标、消融与币安回归门禁。

本模块实现 `specs/tx-risk-eval` 的评测能力（tasks 7.1–7.4）：

- `load_eval_dataset`：加载标注评测集并校验 schema / 标签 / 场景族分布；支持按
  场景族与时间窗口切片；用 `splits.json` 的 train 集合做训练集泄漏检测（严格模式
  直接抛错，由 CLI 转为非零退出）。
- `evaluate`：对每条样本调用 `pipeline.review_request`，统计分场景族召回、固定
  FPR、资产加权召回（被拦截暴露/总暴露）、决策分布、P50/P95 延迟、模型非法输出率、
  规则命中分布与币安回归门禁，并给出 `0/2` 退出码。
- `run_ablation`：对比 `rules_only` 与 `rules_plus_model` 两种模式，输出逐场景族
  差异、模型净收益与误杀代价。
- `compare_eval_reports`：按声明容差比较两次评测报告，输出指标差异与回归项，用于
  重复评测一致性与历史报告对比。
- `run_eval_command`：`lab risk eval` 入口，产物写入 `RunContext("risk/eval")` 的
  `artifacts/eval_report.json`、`artifacts/eval_report.md` 与 manifest。

安全与安全边界：评测只做离线审查，不接触私钥、不广播交易；模型不可用时自动降级为
规则基线（报告标记 degraded）。延迟是实测墙钟时间，其余指标在相同输入/配置下确定性
可复现；重复评测按 `REPEAT_TOLERANCES` 声明的容差比较。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..manifest import RunContext, code_fingerprint, config_fingerprint
from .config import RiskConfig, load_risk_config
from .dataset import (
    BINANCE_CASES_FILENAME,
    DATASET_VERSION,
    SAMPLES_FILENAME,
    SPLITS_FILENAME,
    STATS_FILENAME,
    dataset_stats,
    load_binance_cases,
    load_dataset,
    load_splits,
)
from .decision import rules_floor
from .pipeline import ENGINE_VERSION, SIMULATOR_VERSION, review_request
from .schema import (
    DECISION_RANK,
    SCENARIO_FAMILIES,
    SCHEMA_VERSION,
    ReviewReport,
    RiskSample,
)

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查需要
    from .pipeline import Reviewer

EVAL_VERSION = "risk-eval-v1"
REPORT_JSON = "eval_report.json"
REPORT_MD = "eval_report.md"

DECISIONS: tuple[str, ...] = ("ALLOW", "STEP_UP", "BLOCK", "ESCALATE")
# 处置口径：正样本期望非 ALLOW；只要最终决策不是 ALLOW 即视为“已拦截/已处置”。
DETECTED_DECISIONS: frozenset[str] = frozenset(DECISIONS[1:])
# 固定阈值 FPR 口径：负样本被 BLOCK/ESCALATE 即误报（STEP_UP 是分级处置，不计误杀）。
BLOCKING_DECISIONS: frozenset[str] = frozenset({"BLOCK", "ESCALATE"})
MODES: tuple[str, ...] = ("rules_only", "rules_plus_model")

# 重复评测的声明容差：确定性指标要求完全一致；延迟为实测时间，按绝对毫秒容差比较。
REPEAT_TOLERANCES: dict[str, float] = {
    "recall": 0.0,
    "fpr": 0.0,
    "asset_weighted_recall": 0.0,
    "invalid_output_rate": 0.0,
    "expected_match_rate": 0.0,
    "latency_p50_ms": 250.0,
    "latency_p95_ms": 500.0,
}
_SCALAR_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("recall", ("metrics", "overall", "recall")),
    ("fpr", ("metrics", "overall", "fpr")),
    ("expected_match_rate", ("metrics", "overall", "expected_match_rate")),
    ("asset_weighted_recall", ("metrics", "asset_weighted_recall", "value")),
    ("invalid_output_rate", ("metrics", "model", "invalid_output_rate")),
    ("latency_p50_ms", ("metrics", "latency", "p50_ms")),
    ("latency_p95_ms", ("metrics", "latency", "p95_ms")),
)


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _parse_timestamp(value: str) -> datetime:
    """解析 ISO-8601 时间戳（兼容 `Z` 后缀与无时区值）。"""
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"时间戳不是合法 ISO-8601: {value!r}") from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _safe_ratio(numerator: float, denominator: float, *, empty: float = 1.0) -> float:
    """安全比例：分母为 0 时返回约定的空值（召回默认 1.0，FPR 用 0.0）。"""
    if denominator <= 0:
        return float(empty)
    return round(float(numerator) / float(denominator), 6)


def percentile(values: Sequence[float], q: float) -> float:
    """线性插值百分位（与 numpy 默认 `linear` 口径一致），用于 P50/P95 延迟。"""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (float(q) / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def _instance_key(sample: RiskSample) -> str:
    payload = sample.request.payload or {}
    value = payload.get("instance_id")
    return str(value) if value else sample.sample_id


def _is_binance_sample(sample: RiskSample) -> bool:
    """币安回归用例：payload 标记 `regression_gate`/`binance_case`/`test_only`。"""
    payload = sample.request.payload or {}
    return bool(
        payload.get("regression_gate")
        or payload.get("binance_case")
        or str(payload.get("split_policy")) == "test_only"
    )


def _resolve_artifacts_dir(path: str | Path) -> Path:
    """定位数据集产物目录（支持 run 目录、artifacts 目录与直接 JSONL 文件）。"""
    target = Path(path).expanduser()
    candidates: list[Path] = [target.parent] if target.is_file() else [target / "artifacts", target]
    for candidate in candidates:
        if (candidate / SAMPLES_FILENAME).is_file():
            return candidate
    raise FileNotFoundError(
        f"未找到 {SAMPLES_FILENAME}: {path}（支持数据集 run 目录、artifacts 目录或 JSONL 文件）"
    )


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"元数据不是合法 JSON: {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# 评测集加载
# ---------------------------------------------------------------------------


class EvalDatasetError(ValueError):
    """评测集校验失败（严格模式下的泄漏/标签错误）。"""


@dataclass(frozen=True)
class EvalDataset:
    """一次评测使用的标注数据集快照。

    Attributes:
        samples: 本次评测的样本（已按场景族/时间/划分过滤）。
        binance_samples: 币安回归门禁用例（不受评测划分过滤影响，必须全部 BLOCK）。
        source: `samples.jsonl` 的路径。
        dataset_dir: 数据集产物目录；内存构造时为 None。
        split: 使用的 splits 划分名；None 表示全量。
        families: 场景族过滤条件；None 表示不限制。
        time_window: 时间窗口过滤条件。
        stats: 完整数据集统计（`dataset_stats` 输出）。
        leaks: 评测集与训练集的泄漏项。
        warnings: 非致命告警（缺失场景族、无币安用例等）。
        version: 数据集版本。
    """

    samples: list[RiskSample]
    binance_samples: list[RiskSample]
    source: str
    dataset_dir: str | None = None
    split: str | None = None
    families: tuple[str, ...] | None = None
    time_window: dict[str, str | None] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    leaks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    version: str = DATASET_VERSION

    def describe(self) -> dict[str, Any]:
        """返回报告中的数据集摘要（JSON 可序列化）。"""
        coverage = self.stats.get("coverage", {}) if isinstance(self.stats, Mapping) else {}
        return {
            "source": self.source,
            "dataset_dir": self.dataset_dir,
            "version": self.version,
            "split": self.split,
            "families_filter": list(self.families) if self.families else None,
            "time_window": dict(self.time_window),
            "total": len(self.samples),
            "positive": self.stats.get("positive"),
            "negative": self.stats.get("negative"),
            "binance_cases": len(self.binance_samples),
            "families": self.stats.get("families", {}),
            "coverage": coverage,
            "missing_families": list(coverage.get("missing", [])) if isinstance(coverage, Mapping) else [],
            "splits": self.stats.get("splits"),
            "leaks": list(self.leaks),
            "warnings": list(self.warnings),
            "stats": self.stats,
        }


def _validate_samples(
    samples: Sequence[RiskSample], configured_families: Sequence[str]
) -> list[str]:
    """校验样本标签完整性与场景族分布；返回场景族缺失告警。"""
    if not samples:
        raise EvalDatasetError("评测集为空")
    allowed_families = set(SCENARIO_FAMILIES) | set(configured_families)
    seen_samples: set[str] = set()
    seen_requests: set[str] = set()
    for sample in samples:
        sample_id = sample.sample_id
        if not sample_id:
            raise EvalDatasetError("样本 sample_id 为空")
        if sample_id in seen_samples:
            raise EvalDatasetError(f"样本 sample_id 重复: {sample_id}")
        seen_samples.add(sample_id)
        request_id = sample.request.request_id
        if request_id in seen_requests:
            raise EvalDatasetError(f"请求 request_id 重复: {request_id}")
        seen_requests.add(request_id)
        label = sample.label
        for field_name in ("action_type", "scenario_family"):
            if not str(getattr(label, field_name)).strip():
                raise EvalDatasetError(f"样本 {sample_id} 的标签字段 {field_name} 为空")
        if label.scenario_family not in allowed_families:
            raise EvalDatasetError(
                f"样本 {sample_id} 的场景族未知: {label.scenario_family!r}；"
                f"支持: {', '.join(sorted(allowed_families))}"
            )
        if not 0.0 <= float(label.exposure_ratio) <= 1.0:
            raise EvalDatasetError(
                f"样本 {sample_id} 的 exposure_ratio 越界: {label.exposure_ratio}"
            )
    present = {sample.label.scenario_family for sample in samples}
    missing = [family for family in configured_families if family not in present]
    return [f"评测集缺少场景族: {', '.join(missing)}"] if missing else []


def _detect_leaks(
    samples: Sequence[RiskSample],
    splits_payload: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """把评测集与 `splits.json` 的 train 集合对比，返回样本/实例级泄漏项。"""
    if not splits_payload:
        return []
    raw_splits = splits_payload.get("splits")
    if not isinstance(raw_splits, Mapping):
        return []
    train_raw = raw_splits.get("train", [])
    train_ids = {str(item) for item in train_raw} if isinstance(train_raw, (list, tuple)) else set()
    eval_ids = {sample.sample_id for sample in samples}
    leaks: list[dict[str, Any]] = [
        {"kind": "sample_id", "value": sample_id, "splits": ["eval", "train"]}
        for sample_id in sorted(eval_ids & train_ids)
    ]
    instances = splits_payload.get("instances")
    if not isinstance(instances, Mapping):
        return leaks
    train_instances: dict[str, set[str]] = {}
    for key, ids in instances.items():
        if not isinstance(ids, (list, tuple)):
            continue
        id_set = {str(item) for item in ids}
        if id_set & train_ids:
            train_instances[str(key)] = id_set
    eval_instances: dict[str, set[str]] = {}
    for sample in samples:
        eval_instances.setdefault(_instance_key(sample), set()).add(sample.sample_id)
    for key, instance_ids in sorted(eval_instances.items()):
        # 同一实例跨集但 sample_id 不同才单独报告，避免与 sample_id 泄漏重复。
        if key in train_instances and not (instance_ids & train_ids):
            leaks.append({"kind": "instance", "value": key, "splits": ["eval", "train"]})
    return leaks


def load_eval_dataset(
    path: str | Path,
    *,
    families: Sequence[str] | None = None,
    split: str | None = None,
    since: str | None = None,
    until: str | None = None,
    strict: bool = False,
) -> EvalDataset:
    """加载并校验评测集，支持场景族/时间/划分切片与训练集泄漏检测。

    Args:
        path: 数据集 run 目录、`artifacts/` 目录或 JSONL 文件。
        families: 只评测指定场景族（None 表示全部）。
        split: 只评测 `splits.json` 中指定划分（如 `test`）；无 splits 时忽略并告警。
        since: 起始时间（ISO-8601，含边界）。
        until: 结束时间（ISO-8601，含边界）。
        strict: 严格模式：检出与训练集的泄漏或标签错误时抛 `EvalDatasetError`。

    Returns:
        `EvalDataset`；可直接交给 `evaluate`。

    Raises:
        FileNotFoundError: 数据集或样本文件不存在。
        EvalDatasetError: 样本标签非法、切片为空或严格模式下存在泄漏。
        ValueError: JSONL/splits 内容非法。
    """
    samples_path = _resolve_artifacts_dir(path) / SAMPLES_FILENAME
    artifacts_dir = samples_path.parent
    samples = load_dataset(path)
    stats_payload = _read_json_if_exists(artifacts_dir / STATS_FILENAME) or {}
    configured_raw = stats_payload.get("configured_families")
    configured_families = (
        [str(item) for item in configured_raw if str(item) in SCENARIO_FAMILIES]
        if isinstance(configured_raw, list)
        else []
    ) or list(SCENARIO_FAMILIES)
    warnings = _validate_samples(samples, configured_families)

    splits_payload: Mapping[str, Any] | None = None
    if (artifacts_dir / SPLITS_FILENAME).is_file():
        splits_payload = load_splits(artifacts_dir)

    selected = list(samples)
    if split is not None:
        if splits_payload is None:
            warnings.append(f"数据集缺少 {SPLITS_FILENAME}，请求的 split={split!r} 已忽略（按全量评测）")
            split = None
        else:
            raw_splits = splits_payload.get("splits", {})
            available = sorted(str(name) for name in raw_splits) if isinstance(raw_splits, Mapping) else []
            split_ids = raw_splits.get(split) if isinstance(raw_splits, Mapping) else None
            if not isinstance(split_ids, (list, tuple)):
                raise EvalDatasetError(
                    f"未知评测划分: {split!r}；可选: {', '.join(available) or '无'}"
                )
            id_set = {str(item) for item in split_ids}
            selected = [sample for sample in selected if sample.sample_id in id_set]
            if not selected:
                raise EvalDatasetError(f"划分 {split!r} 没有可评测样本")

    if families:
        family_list = tuple(dict.fromkeys(str(item) for item in families))
        known = set(configured_families) | set(SCENARIO_FAMILIES)
        unknown = [family for family in family_list if family not in known]
        if unknown:
            raise EvalDatasetError(f"未知场景族: {', '.join(unknown)}")
        family_set = set(family_list)
        selected = [sample for sample in selected if sample.label.scenario_family in family_set]
        if not selected:
            raise EvalDatasetError(f"场景族过滤后没有样本: {', '.join(family_list)}")

    time_window: dict[str, str | None] = {"since": since, "until": until}
    if since is not None or until is not None:
        since_dt = _parse_timestamp(since) if since is not None else None
        until_dt = _parse_timestamp(until) if until is not None else None
        if since_dt is not None and until_dt is not None and since_dt > until_dt:
            raise EvalDatasetError(f"时间窗口非法: since={since!r} 晚于 until={until!r}")

        def _in_window(sample: RiskSample) -> bool:
            stamp = _parse_timestamp(sample.request.timestamp)
            return (since_dt is None or stamp >= since_dt) and (until_dt is None or stamp <= until_dt)

        selected = [sample for sample in selected if _in_window(sample)]
        if not selected:
            raise EvalDatasetError("时间窗口过滤后没有样本")

    leaks = _detect_leaks(selected, splits_payload)
    if leaks:
        preview = ", ".join(f"{item['kind']}={item['value']}" for item in leaks[:5])
        message = f"评测集与训练集存在泄漏 {len(leaks)} 项（{preview}）"
        if strict:
            raise EvalDatasetError(message)
        warnings.append(message)

    binance_path = artifacts_dir / BINANCE_CASES_FILENAME
    if binance_path.is_file():
        binance_samples = load_binance_cases(artifacts_dir)
    else:
        binance_samples = [sample for sample in samples if _is_binance_sample(sample)]
        if not binance_samples:
            warnings.append(f"缺少 {BINANCE_CASES_FILENAME} 且样本中没有币安回归标记，门禁无法验证")
    mismatch = [
        sample.sample_id
        for sample in binance_samples
        if sample.label.expected_decision != "BLOCK"
    ]
    if mismatch:
        warnings.append(f"币安回归用例标签不是 BLOCK: {', '.join(mismatch)}")

    stats = dataset_stats(selected, splits_payload, families=configured_families)
    time_values = [_parse_timestamp(sample.request.timestamp) for sample in selected]
    stats["time_range"] = {
        "start": min(time_values).isoformat() if time_values else None,
        "end": max(time_values).isoformat() if time_values else None,
    }
    return EvalDataset(
        samples=selected,
        binance_samples=binance_samples,
        source=str(samples_path),
        dataset_dir=str(artifacts_dir),
        split=split,
        families=tuple(families) if families else None,
        time_window=time_window,
        stats=stats,
        leaks=leaks,
        warnings=warnings,
    )


def _coerce_eval_dataset(dataset: EvalDataset | Sequence[RiskSample]) -> EvalDataset:
    """把样本序列包装成 `EvalDataset`（内存构造，供库调用与测试使用）。"""
    if isinstance(dataset, EvalDataset):
        return dataset
    samples = list(dataset)
    if not samples:
        raise EvalDatasetError("评测集为空")
    binance = [sample for sample in samples if _is_binance_sample(sample)]
    return EvalDataset(
        samples=samples,
        binance_samples=binance,
        source="<memory>",
        stats=dataset_stats(samples),
        warnings=[],
    )


# ---------------------------------------------------------------------------
# 单模式评测
# ---------------------------------------------------------------------------


def _build_record(sample: RiskSample, report: ReviewReport) -> dict[str, Any]:
    """把一条样本的标签与审查报告压缩成评测记录。"""
    expected = sample.label.expected_decision
    actual = report.decision
    expected_rank = DECISION_RANK.get(expected, 0)
    actual_rank = DECISION_RANK.get(actual, 0)
    opinion = report.model_opinion
    return {
        "sample_id": sample.sample_id,
        "request_id": report.request_id,
        "scenario_family": sample.label.scenario_family,
        "expected_decision": expected,
        "decision": actual,
        "detected": actual in DETECTED_DECISIONS,
        "expected_match": actual == expected,
        "under_reacted": actual_rank < expected_rank,
        "over_reacted": actual_rank > expected_rank,
        "exposure_ratio": _round(sample.label.exposure_ratio),
        "exposure_usd": _round(report.simulation.exposure_usd),
        "capability_delta": report.simulation.capability_delta,
        "duration_ms": _round(report.duration_ms, 3),
        "operations": len(report.operations),
        "rule_count": len(report.rule_hits),
        "rule_hits": [
            {
                "rule_id": hit.rule_id,
                "name": hit.name,
                "severity": hit.severity,
                "suggested_decision": hit.suggested_decision,
            }
            for hit in report.rule_hits
        ],
        "suspicious_text": bool(report.suspicious_text),
        "degraded": bool(report.degraded),
        "model": (
            {
                "available": bool(opinion.available),
                "degraded": bool(opinion.degraded),
                "invalid_output": bool(opinion.invalid_output),
                "degraded_reason": opinion.degraded_reason,
                "recommended_action": opinion.recommended_action,
            }
            if opinion is not None
            else None
        ),
    }


def _overall_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [record for record in records if record["expected_decision"] != "ALLOW"]
    negatives = [record for record in records if record["expected_decision"] == "ALLOW"]
    true_positives = sum(1 for record in positives if record["detected"])
    false_positives = sum(1 for record in records if record["expected_decision"] == "ALLOW" and record["decision"] in BLOCKING_DECISIONS)
    false_negatives = sum(1 for record in positives if not record["detected"])
    expected_matches = sum(1 for record in records if record["expected_match"])
    return {
        "total": len(records),
        "positive": len(positives),
        "negative": len(negatives),
        "detected": true_positives,
        "recall": _safe_ratio(true_positives, len(positives), empty=1.0),
        "fpr": _safe_ratio(false_positives, len(negatives), empty=0.0),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "expected_match": expected_matches,
        "expected_match_rate": _safe_ratio(expected_matches, len(records), empty=1.0),
        "under_reaction": sum(1 for record in records if record["under_reacted"]),
        "over_reaction": sum(1 for record in records if record["over_reacted"]),
    }


def _family_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = sorted({str(record["scenario_family"]) for record in records})
    order = [family for family in SCENARIO_FAMILIES if family in families]
    order.extend(family for family in families if family not in order)
    result: dict[str, Any] = {}
    for family in order:
        subset = [record for record in records if record["scenario_family"] == family]
        positives = [record for record in subset if record["expected_decision"] != "ALLOW"]
        negatives = [record for record in subset if record["expected_decision"] == "ALLOW"]
        detected = sum(1 for record in positives if record["detected"])
        false_positives = sum(1 for record in negatives if record["decision"] in BLOCKING_DECISIONS)
        result[family] = {
            "total": len(subset),
            "positive": len(positives),
            "negative": len(negatives),
            "detected": detected,
            "recall": _safe_ratio(detected, len(positives), empty=1.0),
            "fpr": _safe_ratio(false_positives, len(negatives), empty=0.0),
            "false_positives": false_positives,
        }
    return result


def _asset_weighted_recall(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """资产加权召回：被拦截正样本的暴露权重 / 全部正样本暴露权重。"""
    positives = [record for record in records if record["expected_decision"] != "ALLOW"]
    label_weights = [float(record["exposure_ratio"]) for record in positives]
    usd_weights = [float(record["exposure_usd"]) for record in positives]
    if sum(label_weights) > 0.0:
        basis = "label.exposure_ratio"
        weights = label_weights
    elif sum(usd_weights) > 0.0:
        basis = "simulation.exposure_usd"
        weights = usd_weights
    else:
        basis = "count"
        weights = [1.0 for _ in positives]
    total_weight = float(sum(weights))
    detected_weight = float(
        sum(weight for weight, record in zip(weights, positives, strict=True) if record["detected"])
    )
    return {
        "basis": basis,
        "value": _safe_ratio(detected_weight, total_weight, empty=1.0),
        "total_weight": _round(total_weight),
        "detected_weight": _round(detected_weight),
        "positive": len(positives),
    }


def _latency_metrics(records: Sequence[Mapping[str, Any]], budget_ms: float) -> dict[str, Any]:
    durations = [float(record["duration_ms"]) for record in records]
    p50 = percentile(durations, 50)
    p95 = percentile(durations, 95)
    slowest = sorted(records, key=lambda record: float(record["duration_ms"]), reverse=True)[:3]
    return {
        "count": len(durations),
        "p50_ms": _round(p50, 3),
        "p95_ms": _round(p95, 3),
        "mean_ms": _round(sum(durations) / len(durations), 3) if durations else 0.0,
        "max_ms": _round(max(durations), 3) if durations else 0.0,
        "budget_ms": float(budget_ms),
        "within_budget": p95 <= float(budget_ms),
        "slowest_samples": [
            {
                "sample_id": record["sample_id"],
                "duration_ms": record["duration_ms"],
                "decision": record["decision"],
                "operations": record["operations"],
                "rule_count": record["rule_count"],
                "model_available": bool(record["model"]["available"]) if record["model"] else False,
            }
            for record in slowest
        ],
    }


def _model_metrics(
    records: Sequence[Mapping[str, Any]], *, mode: str, reviewer: Reviewer | None
) -> dict[str, Any]:
    total = len(records)
    enabled = mode == "rules_plus_model" and reviewer is not None
    opinions = [record["model"] for record in records if record["model"] is not None]
    if not enabled:
        note = (
            "rules_only 模式不调用模型"
            if mode == "rules_only"
            else "rules_plus_model 模式未提供可用 reviewer，已退化为规则基线"
        )
        return {
            "enabled": False,
            "mode": mode,
            "total": total,
            "opinions": len(opinions),
            "available": 0,
            "degraded": 0,
            "invalid_output": 0,
            "invalid_output_rate": 0.0,
            "recommendation_distribution": {},
            "degraded_reasons": {},
            "note": note,
        }
    available = sum(1 for opinion in opinions if opinion["available"])
    degraded = sum(1 for opinion in opinions if opinion["degraded"])
    invalid_output = sum(1 for opinion in opinions if opinion["invalid_output"])
    flagged = sum(
        1 for opinion in opinions if opinion["invalid_output"] or opinion["degraded"] or not opinion["available"]
    )
    recommendation = Counter(
        str(opinion["recommended_action"])
        for opinion in opinions
        if opinion["available"] and opinion["recommended_action"]
    )
    degraded_reasons = Counter(
        str(opinion["degraded_reason"]) for opinion in opinions if opinion["degraded_reason"]
    )
    return {
        "enabled": True,
        "mode": mode,
        "total": total,
        "opinions": len(opinions),
        "available": available,
        "degraded": degraded,
        "invalid_output": invalid_output,
        "invalid_output_rate": _safe_ratio(flagged, total, empty=0.0),
        "recommendation_distribution": dict(sorted(recommendation.items())),
        "degraded_reasons": dict(sorted(degraded_reasons.items())),
        "note": "非法输出率 = (invalid_output 或 degraded 或不可用) / 评测样本数",
    }


def _rule_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    catalog: dict[str, dict[str, Any]] = {}
    counter: Counter[str] = Counter()
    for record in records:
        for hit in record["rule_hits"]:
            counter[hit["rule_id"]] += 1
            catalog.setdefault(
                hit["rule_id"],
                {
                    "rule_id": hit["rule_id"],
                    "name": hit["name"],
                    "severity": hit["severity"],
                    "suggested_decision": hit["suggested_decision"],
                },
            )
    distribution = [
        {**catalog[rule_id], "count": count}
        for rule_id, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "total_hits": sum(counter.values()),
        "distinct_rules": len(counter),
        "hit_distribution": distribution,
    }


def _decision_distribution(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(record[key]) for record in records)
    return {decision: counter.get(decision, 0) for decision in DECISIONS}


def _decision_basis(report: ReviewReport) -> dict[str, Any]:
    """失败用例的决策依据：规则地板、命中规则、模型建议、模拟暴露与备注。"""
    opinion = report.model_opinion
    return {
        "rules_floor": rules_floor(report.rule_hits),
        "final": report.decision,
        "rule_hits": [
            {
                "rule_id": hit.rule_id,
                "name": hit.name,
                "severity": hit.severity,
                "suggested_decision": hit.suggested_decision,
                "evidence": list(hit.evidence),
            }
            for hit in report.rule_hits
        ],
        "model_action": opinion.recommended_action if opinion is not None else None,
        "simulation": {
            "capability_delta": report.simulation.capability_delta,
            "exposure_ratio": report.simulation.exposure_ratio,
            "exposure_usd": report.simulation.exposure_usd,
            "rolling_exposure_ratio": report.simulation.rolling_exposure_ratio,
        },
        "notes": list(report.notes),
    }


def _binance_gate_cases(
    samples: Sequence[RiskSample],
    reports: Sequence[ReviewReport],
    *,
    enabled: bool,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for sample, report in zip(samples, reports, strict=True):
        passed = report.decision == "BLOCK"
        cases.append(
            {
                "sample_id": sample.sample_id,
                "request_id": report.request_id,
                "expected_decision": sample.label.expected_decision,
                "decision": report.decision,
                "exit_code": report.exit_code,
                "passed": passed,
                "blocking_rule_ids": [
                    hit.rule_id for hit in report.rule_hits if hit.suggested_decision == "BLOCK"
                ],
                "decision_basis": _decision_basis(report),
            }
        )
    failures = [case for case in cases if not case["passed"]]
    if not enabled:
        return {
            "enabled": False,
            "passed": True,
            "total": len(cases),
            "blocked": sum(1 for case in cases if case["passed"]),
            "failures": [],
            "cases": cases,
            "note": "配置 eval.binance_gate=false，币安回归门禁未启用",
        }
    return {
        "enabled": True,
        "passed": not failures,
        "total": len(cases),
        "blocked": sum(1 for case in cases if case["passed"]),
        "failures": [
            {
                "sample_id": case["sample_id"],
                "decision": case["decision"],
                "expected_decision": case["expected_decision"],
                "blocking_rule_ids": case["blocking_rule_ids"],
                "decision_basis": case["decision_basis"],
            }
            for case in failures
        ],
        "cases": cases,
        "note": (
            "没有可用的币安回归用例"
            if not cases
            else ("全部币安回归用例均被 BLOCK" if not failures else "存在未被 BLOCK 的币安回归用例，门禁失败")
        ),
    }


def _gate_checks(
    over_metrics: Mapping[str, Any],
    asset_metrics: Mapping[str, Any],
    latency_metrics: Mapping[str, Any],
    binance_gate: Mapping[str, Any],
    *,
    min_recall: float,
    max_fpr: float,
    min_asset_weighted_recall: float,
    latency_budget_ms: float,
    binance_enabled: bool,
) -> dict[str, Any]:
    recall = float(over_metrics["recall"])
    fpr = float(over_metrics["fpr"])
    asset_recall = float(asset_metrics["value"])
    p95 = float(latency_metrics["p95_ms"])
    checks: dict[str, Any] = {
        "min_recall": {
            "kind": ">=",
            "threshold": float(min_recall),
            "value": recall,
            "passed": recall >= float(min_recall),
        },
        "max_fpr": {
            "kind": "<=",
            "threshold": float(max_fpr),
            "value": fpr,
            "passed": fpr <= float(max_fpr),
        },
        "min_asset_weighted_recall": {
            "kind": ">=",
            "threshold": float(min_asset_weighted_recall),
            "value": asset_recall,
            "passed": asset_recall >= float(min_asset_weighted_recall),
        },
        "latency_p95_budget_ms": {
            "kind": "<=",
            "threshold": float(latency_budget_ms),
            "value": p95,
            "passed": p95 <= float(latency_budget_ms),
        },
        "binance_gate": {
            "kind": "all_block",
            "enabled": bool(binance_enabled),
            "threshold": "all_block",
            "value": int(binance_gate.get("blocked", 0)),
            "total": int(binance_gate.get("total", 0)),
            "passed": (not binance_enabled) or bool(binance_gate.get("passed", False)),
        },
    }
    return checks


def _model_version(mode: str, reviewer: Reviewer | None) -> str:
    if reviewer is None:
        return "rules_only" if mode == "rules_only" else "unavailable"
    version = getattr(reviewer, "version", None)
    if version:
        return str(version)
    model_dir = getattr(reviewer, "model_dir", None)
    if model_dir is not None:
        source = getattr(reviewer, "source", None) or "model"
        return f"{source}:{Path(str(model_dir)).name}"
    return type(reviewer).__name__


def evaluate(
    dataset: EvalDataset | Sequence[RiskSample],
    *,
    config: RiskConfig,
    mode: str = "rules_only",
    reviewer: Reviewer | None = None,
) -> dict[str, Any]:
    """对评测集执行一次审查评测，返回可复现的指标报告。

    Args:
        dataset: `EvalDataset`（含金币安门禁用例）或 `RiskSample` 序列。
        config: 风险配置（规则、门限、数据 seed）。
        mode: `rules_only` 或 `rules_plus_model`。
        reviewer: 模型审查器；`rules_plus_model` 下缺失时退化为规则基线并告警。

    Returns:
        评测报告字典；含 `passed`（门禁结论）、`exit_code`（0/2）、全部指标与版本。

    Raises:
        ValueError: 模式非法或评测集为空。
    """
    if mode not in MODES:
        raise ValueError(f"未知评测模式: {mode!r}，可选 {'|'.join(MODES)}")
    eval_dataset = _coerce_eval_dataset(dataset)
    warnings = list(eval_dataset.warnings)
    active_reviewer = reviewer if mode == "rules_plus_model" else None
    if mode == "rules_plus_model" and active_reviewer is None:
        warnings.append("rules_plus_model 模式未提供可用 reviewer，已退化为规则基线")
    if mode == "rules_only" and reviewer is not None:
        warnings.append("rules_only 模式忽略传入的 reviewer，不调用模型")

    records: list[dict[str, Any]] = []
    for sample in eval_dataset.samples:
        report = review_request(sample.request, rules=config.rules, reviewer=active_reviewer)
        records.append(_build_record(sample, report))

    gate_reports = [
        review_request(sample.request, rules=config.rules, reviewer=active_reviewer)
        for sample in eval_dataset.binance_samples
    ]

    settings = config.settings.eval
    over_metrics = _overall_metrics(records)
    family_metrics = _family_metrics(records)
    asset_metrics = _asset_weighted_recall(records)
    latency_metrics = _latency_metrics(records, settings.latency_p95_budget_ms)
    model_metrics = _model_metrics(records, mode=mode, reviewer=active_reviewer if mode == "rules_plus_model" else None)
    rule_metrics = _rule_metrics(records)
    binance_gate = _binance_gate_cases(
        eval_dataset.binance_samples, gate_reports, enabled=settings.binance_gate
    )
    if binance_gate["total"] == 0 and settings.binance_gate:
        warnings.append("没有可用的币安回归用例，门禁为空（结果不可完全验证）")
    checks = _gate_checks(
        over_metrics,
        asset_metrics,
        latency_metrics,
        binance_gate,
        min_recall=settings.min_recall,
        max_fpr=settings.max_fpr,
        min_asset_weighted_recall=settings.min_asset_weighted_recall,
        latency_budget_ms=settings.latency_p95_budget_ms,
        binance_enabled=settings.binance_gate,
    )
    failures = [name for name, check in checks.items() if not check["passed"]]
    passed = not failures
    versions = {
        "eval": EVAL_VERSION,
        "engine": ENGINE_VERSION,
        "schema": str(SCHEMA_VERSION),
        "simulator": SIMULATOR_VERSION,
        "rules": config.rules.version,
        "model": _model_version(mode, active_reviewer if mode == "rules_plus_model" else None),
        "dataset": eval_dataset.version,
        "code": None,
    }
    return {
        "schema_version": 1,
        "eval_version": EVAL_VERSION,
        "mode": mode,
        "passed": passed,
        "exit_code": 0 if passed else 2,
        "failures": failures,
        "warnings": warnings,
        "dataset": eval_dataset.describe(),
        "metrics": {
            "overall": over_metrics,
            "by_family": family_metrics,
            "asset_weighted_recall": asset_metrics,
            "decision_distribution": _decision_distribution(records, "decision"),
            "expected_decision_distribution": _decision_distribution(records, "expected_decision"),
            "latency": latency_metrics,
            "model": model_metrics,
            "rules": rule_metrics,
        },
        "binance_gate": binance_gate,
        "gates": checks,
        "versions": versions,
        "reproducibility": {
            "seed": config.settings.dataset.seed,
            "deterministic_metrics": [
                "decisions",
                "recall",
                "fpr",
                "asset_weighted_recall",
                "decision_distribution",
                "rule_hit_distribution",
                "binance_gate",
                "invalid_output_rate",
            ],
            "tolerances": dict(REPEAT_TOLERANCES),
            "note": "延迟为实测墙钟时间，重复评测需按 tolerances 中的绝对毫秒容差比较",
        },
        "per_sample": records,
    }


# ---------------------------------------------------------------------------
# 重复评测 / 历史报告对比
# ---------------------------------------------------------------------------


def _metric_value(report: Mapping[str, Any], path: Sequence[str]) -> float | None:
    node: Any = report
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def _report_brief(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eval_version": report.get("eval_version"),
        "mode": report.get("mode"),
        "passed": report.get("passed"),
        "failures": list(report.get("failures", [])),
        "versions": dict(report.get("versions", {})),
    }


def compare_eval_reports(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    tolerances: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """比较两次评测报告，输出指标差异与回归项（用于重复一致性与历史对比）。

    Args:
        previous: 历史评测报告（`eval_report.json` 内容）。
        current: 本次评测报告。
        tolerances: 覆盖声明容差；未提供的指标使用 `REPEAT_TOLERANCES`。

    Returns:
        含 `within_tolerance`、`metric_differences`、`regressions`、`passed` 的对比结果。
    """
    merged = dict(REPEAT_TOLERANCES)
    if tolerances:
        merged.update({str(key): float(value) for key, value in tolerances.items()})
    differences: list[dict[str, Any]] = []
    within_tolerance = True
    for name, path in _SCALAR_METRICS:
        before = _metric_value(previous, path)
        after = _metric_value(current, path)
        if before is None or after is None:
            continue
        tolerance = float(merged.get(name, 0.0))
        delta = after - before
        passed = abs(delta) <= tolerance + 1e-12
        within_tolerance = within_tolerance and passed
        differences.append(
            {
                "metric": name,
                "previous": _round(before),
                "current": _round(after),
                "delta": _round(delta),
                "tolerance": tolerance,
                "passed": passed,
            }
        )
    previous_failures = set(previous.get("failures", []))
    current_failures = set(current.get("failures", []))
    previous_decisions = dict(previous.get("metrics", {}).get("decision_distribution", {}))
    current_decisions = dict(current.get("metrics", {}).get("decision_distribution", {}))
    decision_changed = previous_decisions != current_decisions
    new_failures = sorted(current_failures - previous_failures)
    fixed = sorted(previous_failures - current_failures)
    regressions: list[str] = [f"门禁新增失败项: {name}" for name in new_failures]
    if decision_changed:
        regressions.append("决策分布发生变化（确定性指标不应变化）")
    if previous.get("passed") and not current.get("passed"):
        regressions.append("历史评测通过、本次评测未通过")
    improvements = [f"门禁修复项: {name}" for name in fixed]
    if not previous.get("passed") and current.get("passed"):
        improvements.append("本次评测通过门禁")
    return {
        "within_tolerance": within_tolerance,
        "metric_differences": differences,
        "decision_distribution": {
            "previous": previous_decisions,
            "current": current_decisions,
            "changed": decision_changed,
        },
        "regressions": regressions,
        "improvements": improvements,
        "passed": within_tolerance and not regressions,
        "previous": _report_brief(previous),
        "current": _report_brief(current),
    }


# ---------------------------------------------------------------------------
# rules_only vs rules_plus_model 消融
# ---------------------------------------------------------------------------


def _ablation_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    overall = metrics.get("overall", {})
    asset = metrics.get("asset_weighted_recall", {})
    latency = metrics.get("latency", {})
    model = metrics.get("model", {})
    return {
        "passed": bool(report.get("passed")),
        "recall": float(overall.get("recall", 0.0)),
        "fpr": float(overall.get("fpr", 0.0)),
        "expected_match_rate": float(overall.get("expected_match_rate", 0.0)),
        "asset_weighted_recall": float(asset.get("value", 0.0)),
        "invalid_output_rate": float(model.get("invalid_output_rate", 0.0)),
        "p50_ms": float(latency.get("p50_ms", 0.0)),
        "p95_ms": float(latency.get("p95_ms", 0.0)),
        "failures": list(report.get("failures", [])),
    }


def run_ablation(
    dataset: EvalDataset | Sequence[RiskSample],
    *,
    config: RiskConfig,
    reviewer: Reviewer | None = None,
    model_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """对比 `rules_only` 与 `rules_plus_model`，输出逐场景族差异与收益/代价。

    Args:
        dataset: 与 `evaluate` 相同的评测集。
        config: 风险配置。
        reviewer: 模型审查器；缺失且未提供 `model_report` 时返回 available=False。
        model_report: 已完成的 rules_plus_model 报告，避免重复跑模型推理。

    Returns:
        含 `rules_only`/`rules_plus_model` 指标、逐族差异、模型净收益与误杀代价的字典。
    """
    baseline = evaluate(dataset, config=config, mode="rules_only")
    if model_report is None:
        if reviewer is None:
            return {
                "available": False,
                "reason": "未提供模型 reviewer，无法执行 rules_plus_model 消融",
                "rules_only": _ablation_metrics(baseline),
                "rules_plus_model": None,
                "delta": {},
                "by_family": {},
                "model_net_gain": {},
                "extra_fp_cost": {},
                "passed": bool(baseline.get("passed")),
            }
        model_report = evaluate(dataset, config=config, mode="rules_plus_model", reviewer=reviewer)

    base_metrics = _ablation_metrics(baseline)
    model_metrics = _ablation_metrics(model_report)
    delta = {
        key: _round(model_metrics[key] - base_metrics[key])
        for key in (
            "recall",
            "fpr",
            "expected_match_rate",
            "asset_weighted_recall",
            "invalid_output_rate",
            "p50_ms",
            "p95_ms",
        )
    }
    base_family = baseline.get("metrics", {}).get("by_family", {})
    model_family = model_report.get("metrics", {}).get("by_family", {})
    families = sorted(set(base_family) | set(model_family))
    by_family: dict[str, Any] = {}
    for family in families:
        before = base_family.get(family, {})
        after = model_family.get(family, {})
        before_recall = float(before.get("recall", 0.0))
        after_recall = float(after.get("recall", 0.0))
        before_fpr = float(before.get("fpr", 0.0))
        after_fpr = float(after.get("fpr", 0.0))
        by_family[family] = {
            "total": int(after.get("total", before.get("total", 0))),
            "rules_recall": _round(before_recall),
            "model_recall": _round(after_recall),
            "recall_delta": _round(after_recall - before_recall),
            "rules_fpr": _round(before_fpr),
            "model_fpr": _round(after_fpr),
            "fpr_delta": _round(after_fpr - before_fpr),
        }

    base_decisions = {
        str(record.get("sample_id")): str(record.get("decision"))
        for record in baseline.get("per_sample", [])
    }
    model_decisions = {
        str(record.get("sample_id")): str(record.get("decision"))
        for record in model_report.get("per_sample", [])
    }
    expected = {
        str(record.get("sample_id")): str(record.get("expected_decision"))
        for record in baseline.get("per_sample", [])
    }
    newly_detected = sorted(
        sample_id
        for sample_id, decision in model_decisions.items()
        if expected.get(sample_id) != "ALLOW"
        and decision != "ALLOW"
        and base_decisions.get(sample_id) == "ALLOW"
    )
    newly_missed = sorted(
        sample_id
        for sample_id, decision in model_decisions.items()
        if expected.get(sample_id) != "ALLOW"
        and decision == "ALLOW"
        and base_decisions.get(sample_id) not in (None, "ALLOW")
    )
    newly_blocked = sorted(
        sample_id
        for sample_id, decision in model_decisions.items()
        if expected.get(sample_id) == "ALLOW"
        and decision in BLOCKING_DECISIONS
        and base_decisions.get(sample_id) not in BLOCKING_DECISIONS
    )
    newly_escalated = sorted(
        sample_id
        for sample_id, decision in model_decisions.items()
        if expected.get(sample_id) == "ALLOW"
        and decision == "ESCALATE"
        and base_decisions.get(sample_id) != "ESCALATE"
    )
    model_net_gain = {
        "recall_delta": delta["recall"],
        "asset_weighted_recall_delta": delta["asset_weighted_recall"],
        "newly_detected": newly_detected,
        "newly_missed": newly_missed,
    }
    extra_fp_cost = {
        "fpr_delta": delta["fpr"],
        "newly_blocked": newly_blocked,
        "newly_escalated": newly_escalated,
    }
    summary = (
        f"rules_plus_model 相比 rules_only：召回 {delta['recall']:+.4f}，"
        f"FPR {delta['fpr']:+.4f}，资产加权召回 {delta['asset_weighted_recall']:+.4f}；"
        f"新增检出 {len(newly_detected)} 条，新增误杀 {len(newly_blocked) + len(newly_escalated)} 条，"
        f"模型非法输出率 {model_metrics['invalid_output_rate']:.4f}"
    )
    return {
        "available": True,
        "rules_only": base_metrics,
        "rules_plus_model": model_metrics,
        "delta": delta,
        "by_family": by_family,
        "model_net_gain": model_net_gain,
        "extra_fp_cost": extra_fp_cost,
        "passed": bool(model_report.get("passed")),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Markdown 报告与人类可读输出
# ---------------------------------------------------------------------------


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}"


def _format_gate_value(value: Any) -> str:
    if isinstance(value, float):
        return _format_float(value, 4)
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    """把评测报告渲染成 Markdown（指标、门禁、消融与版本）。"""
    dataset = report.get("dataset", {})
    metrics = report.get("metrics", {})
    overall = metrics.get("overall", {})
    asset = metrics.get("asset_weighted_recall", {})
    latency = metrics.get("latency", {})
    model = metrics.get("model", {})
    rules = metrics.get("rules", {})
    gates = report.get("gates", {})
    binance = report.get("binance_gate", {})
    versions = report.get("versions", {})
    reproducibility = report.get("reproducibility", {})
    passed = bool(report.get("passed"))
    lines: list[str] = [
        "# 交易签名风险审查评测报告",
        "",
        f"- 评测版本：`{report.get('eval_version')}`（报告 schema `{report.get('schema_version')}`）",
        f"- 评测模式：`{report.get('mode')}`"
        + (f"（请求模式 `{report['requested_mode']}`）" if report.get("requested_mode") else ""),
        f"- 结论：{'**通过**' if passed else '**失败**'}（退出码 `{report.get('exit_code')}`）",
        f"- 数据集：`{dataset.get('source')}`（split={dataset.get('split')}，"
        f"样本 {dataset.get('total')}，正/负 {dataset.get('positive')}/{dataset.get('negative')}）",
        f"- 币安回归用例：{dataset.get('binance_cases')} 条",
        "",
        "## 门禁结论",
        "",
    ]
    gate_rows: list[list[Any]] = []
    for name, check in gates.items():
        if name == "binance_gate":
            status = "通过" if check.get("passed") else "失败"
            note = "未启用" if not check.get("enabled", True) else status
            gate_rows.append(["binance_gate", "全部 BLOCK", f"{check.get('value')}/{check.get('total')}", note])
            continue
        gate_rows.append(
            [
                name,
                f"{check.get('kind')} {_format_gate_value(check.get('threshold'))}",
                _format_gate_value(check.get("value")),
                "通过" if check.get("passed") else "失败",
            ]
        )
    lines.extend(_markdown_table(["检查项", "阈值", "实测", "结果"], gate_rows))
    failures = list(report.get("failures", []))
    lines.extend(["", f"- 失败项：{', '.join(failures) if failures else '无'}"])
    warnings = list(report.get("warnings", []))
    if warnings:
        lines.append(f"- 告警：{'；'.join(warnings)}")
    lines.extend(["", "## 核心指标", ""])
    lines.extend(
        _markdown_table(
            ["指标", "数值"],
            [
                ["召回（正样本非 ALLOW）", _format_float(overall.get("recall"))],
                ["固定 FPR（负样本 BLOCK/ESCALATE）", _format_float(overall.get("fpr"))],
                ["资产加权召回（被拦暴露/总暴露）", _format_float(asset.get("value"))],
                ["资产加权口径", asset.get("basis", "-")],
                ["期望决策完全匹配率", _format_float(overall.get("expected_match_rate"))],
                ["P50 / P95 延迟（ms）", f"{_format_float(latency.get('p50_ms'), 3)} / {_format_float(latency.get('p95_ms'), 3)}"],
                ["延迟预算（ms）", _format_float(latency.get("budget_ms"), 1)],
                ["模型非法/降级输出率", _format_float(model.get("invalid_output_rate"))],
                ["正/负样本数", f"{overall.get('positive')}/{overall.get('negative')}"],
                ["误漏报：FP/FN", f"{overall.get('false_positives')}/{overall.get('false_negatives')}"],
            ],
        )
    )
    lines.extend(["", "### 决策分布", ""])
    decision_rows = [
        [decision, count, metrics.get("expected_decision_distribution", {}).get(decision, 0)]
        for decision, count in metrics.get("decision_distribution", {}).items()
    ]
    lines.extend(_markdown_table(["最终决策", "样本数", "标签期望"], decision_rows))
    family_rows = [
        [
            family,
            data.get("total"),
            data.get("positive"),
            data.get("negative"),
            _format_float(data.get("recall")),
            _format_float(data.get("fpr")),
        ]
        for family, data in metrics.get("by_family", {}).items()
    ]
    lines.extend(["", "## 分场景族指标", ""])
    lines.extend(_markdown_table(["场景族", "样本", "正/负", "召回", "FPR"], family_rows))
    lines.extend(["", "## 延迟", ""])
    lines.extend(
        _markdown_table(
            ["指标", "数值（ms）"],
            [
                ["P50", _format_float(latency.get("p50_ms"), 3)],
                ["P95", _format_float(latency.get("p95_ms"), 3)],
                ["均值", _format_float(latency.get("mean_ms"), 3)],
                ["最大", _format_float(latency.get("max_ms"), 3)],
            ],
        )
    )
    slowest = latency.get("slowest_samples", [])
    if slowest:
        lines.extend(["", "最慢样本耗时分解："])
        for item in slowest:
            lines.append(
                f"- `{item.get('sample_id')}`：{item.get('duration_ms')} ms，"
                f"decision={item.get('decision')}，operations={item.get('operations')}，"
                f"rules={item.get('rule_count')}，model={item.get('model_available')}"
            )
    lines.extend(["", "## 模型与规则", ""])
    lines.extend(
        _markdown_table(
            ["项目", "数值"],
            [
                ["模型模式", model.get("mode")],
                ["模型意见数/可用数", f"{model.get('opinions')}/{model.get('available')}"],
                ["降级数 / 非法输出数", f"{model.get('degraded')}/{model.get('invalid_output')}"],
                ["模型建议分布", model.get("recommendation_distribution")],
                ["规则命中总数 / 不同规则数", f"{rules.get('total_hits')}/{rules.get('distinct_rules')}"],
            ],
        )
    )
    hit_distribution = rules.get("hit_distribution", [])
    if hit_distribution:
        lines.extend(["", "### 规则命中分布", ""])
        lines.extend(
            _markdown_table(
                ["规则", "名称", "严重度", "建议决策", "命中数"],
                [
                    [
                        item.get("rule_id"),
                        item.get("name"),
                        item.get("severity"),
                        item.get("suggested_decision"),
                        item.get("count"),
                    ]
                    for item in hit_distribution
                ],
            )
        )
    lines.extend(["", "## 币安回归门禁", ""])
    lines.extend(
        _markdown_table(
            ["用例", "期望", "最终决策", "结果", "BLOCK 依据规则"],
            [
                [
                    case.get("sample_id"),
                    case.get("expected_decision"),
                    case.get("decision"),
                    "通过" if case.get("passed") else "失败",
                    ", ".join(case.get("blocking_rule_ids", [])) or "-",
                ]
                for case in binance.get("cases", [])
            ],
        )
    )
    if binance.get("failures"):
        lines.extend(["", "失败用例决策依据："])
        for failure in binance["failures"]:
            basis = failure.get("decision_basis", {})
            hit_ids = [
                f"{hit.get('rule_id')}→{hit.get('suggested_decision')}"
                for hit in basis.get("rule_hits", [])
            ]
            lines.append(
                f"- `{failure.get('sample_id')}`：decision={failure.get('decision')}，"
                f"rules_floor={basis.get('rules_floor')}，"
                f"规则命中=[{', '.join(hit_ids) or '无'}]，"
                f"模型建议={basis.get('model_action')}；"
                f"notes={'；'.join(basis.get('notes', [])) or '无'}"
            )
    ablation = report.get("ablation")
    if isinstance(ablation, Mapping) and ablation.get("available"):
        lines.extend(["", "## 消融对比（rules_only vs rules_plus_model）", ""])
        lines.append(f"- {ablation.get('summary')}")
        lines.append(f"- 模型净收益：{ablation.get('model_net_gain')}")
        lines.append(f"- 误杀代价：{ablation.get('extra_fp_cost')}")
        lines.extend(["", "### 分场景族差异", ""])
        lines.extend(
            _markdown_table(
                ["场景族", "rules 召回", "model 召回", "召回 Δ", "rules FPR", "model FPR", "FPR Δ"],
                [
                    [
                        family,
                        _format_float(item.get("rules_recall")),
                        _format_float(item.get("model_recall")),
                        _format_float(item.get("recall_delta")),
                        _format_float(item.get("rules_fpr")),
                        _format_float(item.get("model_fpr")),
                        _format_float(item.get("fpr_delta")),
                    ]
                    for family, item in ablation.get("by_family", {}).items()
                ],
            )
        )
    lines.extend(["", "## 版本与可复现性", ""])
    lines.extend(
        _markdown_table(
            ["组件", "版本"],
            [
                ["eval", versions.get("eval")],
                ["engine", versions.get("engine")],
                ["schema", versions.get("schema")],
                ["simulator", versions.get("simulator")],
                ["rules", versions.get("rules")],
                ["model", versions.get("model")],
                ["dataset", versions.get("dataset")],
                ["code", versions.get("code")],
            ],
        )
    )
    lines.extend(
        [
            "",
            f"- 数据 seed：`{reproducibility.get('seed')}`；确定性指标：{', '.join(reproducibility.get('deterministic_metrics', []))}",
            f"- 重复评测容差：{reproducibility.get('tolerances')}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_summary(report: Mapping[str, Any]) -> str:
    """人类可读的评测摘要（非 --json 输出）。"""
    dataset = report.get("dataset", {})
    metrics = report.get("metrics", {})
    overall = metrics.get("overall", {})
    asset = metrics.get("asset_weighted_recall", {})
    latency = metrics.get("latency", {})
    binance = report.get("binance_gate", {})
    lines = [
        f"风险评测完成: mode={report.get('mode')} samples={dataset.get('total')} "
        f"split={dataset.get('split')}",
        f"  召回={_format_float(overall.get('recall'))} "
        f"FPR={_format_float(overall.get('fpr'))} "
        f"资产加权召回={_format_float(asset.get('value'))}（{asset.get('basis')}）",
        f"  延迟: P50={_format_float(latency.get('p50_ms'), 3)}ms "
        f"P95={_format_float(latency.get('p95_ms'), 3)}ms"
        f"（预算 {_format_float(latency.get('budget_ms'), 1)}ms，"
        f"{'达标' if latency.get('within_budget') else '超预算'}）",
        f"  决策分布: {metrics.get('decision_distribution')}",
        f"  币安回归门禁: {binance.get('blocked')}/{binance.get('total')} BLOCK，"
        f"{'通过' if binance.get('passed') else '失败'}",
    ]
    ablation = report.get("ablation")
    if isinstance(ablation, Mapping) and ablation.get("available"):
        lines.append(f"  消融: {ablation.get('summary')}")
    failures = list(report.get("failures", []))
    lines.append(
        f"  门禁结论: {'通过' if report.get('passed') else '失败'}"
        + (f"（失败项: {', '.join(failures)}）" if failures else "")
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def _create_run_context(config: RiskConfig, runs_dir: Path) -> RunContext:
    """创建 `risk/eval` 运行上下文；同一秒内重复评测时追加序号避免 run_id 冲突。"""
    try:
        return RunContext.create("risk/eval", config.raw, runs_dir=runs_dir)
    except FileExistsError:
        base = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{config_fingerprint(config.raw)[:8]}"
        for suffix in range(1, 100):
            try:
                return RunContext.create(
                    "risk/eval", config.raw, runs_dir=runs_dir, run_id=f"{base}-{suffix}"
                )
            except FileExistsError:
                continue
        raise


def _load_build_reviewer() -> Any:
    """惰性导入模型审查器构造器；测试可替换此函数以避免加载真实模型。"""
    from .infer import build_reviewer  # noqa: PLC0415 - 规则模式不导入模型模块

    return build_reviewer


def _build_reviewer_from_config(config: RiskConfig) -> tuple[Reviewer | None, str | None]:
    """构造模型审查器；导入/构造失败时返回降级原因而不是抛错。"""
    try:
        builder = _load_build_reviewer()
    except Exception as exc:  # noqa: BLE001 - 模型模块缺失/依赖不全必须降级
        return None, f"导入 risk.infer 失败: {type(exc).__name__}: {exc}"
    try:
        reviewer = builder(config)
    except Exception as exc:  # noqa: BLE001 - 模型不可用必须降级
        return None, f"构造模型审查器失败: {type(exc).__name__}: {exc}"
    if reviewer is None:
        return None, "build_reviewer 返回 None"
    return reviewer, None


def run_eval_command(args: argparse.Namespace) -> int:
    """实现 `lab risk eval`：加载评测集、执行评测、写 JSON/Markdown 报告与 manifest。

    Args:
        args: argparse 命名空间，需含 dataset、mode、out、json、config。

    Returns:
        退出码：门禁通过 0；门禁失败或评测无法完成 2。
    """
    config_paths = list(getattr(args, "config", []) or [])
    config = load_risk_config(config_paths or None)
    requested_mode = getattr(args, "mode", None) or "rules_only"
    if requested_mode not in MODES:
        raise ValueError(f"未知评测模式: {requested_mode!r}，可选 {'|'.join(MODES)}")
    dataset_path = getattr(args, "dataset", None)
    if not dataset_path:
        raise ValueError("eval 需要 --dataset（数据集 run 目录或 artifacts 目录）")
    # CLI 默认评测 holdout test 划分（不共享训练实例）；严格模式：泄漏直接失败。
    dataset = load_eval_dataset(dataset_path, split="test", strict=True)

    reviewer: Reviewer | None = None
    fallback_reason: str | None = None
    active_mode = requested_mode
    if requested_mode == "rules_plus_model":
        reviewer, fallback_reason = _build_reviewer_from_config(config)
        if fallback_reason is not None:
            active_mode = "rules_only"
            print(
                f"警告: 模型评测不可用，已降级为 rules_only（{fallback_reason}）",
                file=sys.stderr,
            )

    out = getattr(args, "out", None)
    runs_dir = Path(out).expanduser() if out else config.lab.resolved_dir("runs_dir")
    ctx = _create_run_context(config, runs_dir)
    try:
        report = evaluate(dataset, config=config, mode=active_mode, reviewer=reviewer)
        report["requested_mode"] = requested_mode
        report["degraded"] = fallback_reason is not None
        if fallback_reason is not None:
            report["degraded_reason"] = fallback_reason
            report["warnings"].append(f"模型不可用，已降级为 rules_only: {fallback_reason}")
        if active_mode == "rules_plus_model" and reviewer is not None:
            report["ablation"] = run_ablation(
                dataset, config=config, reviewer=reviewer, model_report=report
            )
        report["versions"]["code"] = code_fingerprint()[:16]
        report["run"] = {
            "run_id": ctx.run_dir.name,
            "run_dir": str(ctx.run_dir.resolve()),
            "artifacts_dir": str(ctx.artifacts_dir.resolve()),
            "report_json": f"artifacts/{REPORT_JSON}",
            "report_md": f"artifacts/{REPORT_MD}",
        }
        ctx.write_json_artifact(REPORT_JSON, report)
        ctx.write_artifact(REPORT_MD, render_markdown(report))
        metrics = report["metrics"]
        ctx.log_metrics(
            1,
            passed=report["passed"],
            recall=metrics["overall"]["recall"],
            fpr=metrics["overall"]["fpr"],
            asset_weighted_recall=metrics["asset_weighted_recall"]["value"],
            latency_p50_ms=metrics["latency"]["p50_ms"],
            latency_p95_ms=metrics["latency"]["p95_ms"],
            binance_blocked=report["binance_gate"]["blocked"],
            binance_total=report["binance_gate"]["total"],
            failures=report["failures"],
        )
        ctx.finish("success" if report["passed"] else "failed_gate", int(report["exit_code"]))
    except Exception as exc:  # noqa: BLE001 - 评测失败必须 fail-closed 且保留产物
        reason = f"{type(exc).__name__}: {exc}"
        ctx.finish("failed", 2, reason)
        print(f"评测失败（fail-closed）: {reason}", file=sys.stderr)
        return 2

    _emit_report_output(report, ctx, as_json=bool(getattr(args, "json", False)))
    return int(report["exit_code"])


def _emit_report_output(report: Mapping[str, Any], ctx: RunContext, *, as_json: bool) -> None:
    """输出评测结果：`--json` 时 stdout 只输出 JSON，其余信息走 stderr。"""
    if as_json:
        print(json.dumps(report, ensure_ascii=False))
        print(f"产物目录: {ctx.run_dir}", file=sys.stderr)
        return
    print(_render_summary(report))
    run = report.get("run", {})
    print(f"  产物目录: {run.get('run_dir', ctx.run_dir)}")
    print(f"  报告: {run.get('report_json')}, {run.get('report_md')}")


__all__ = [
    "EVAL_VERSION",
    "REPORT_JSON",
    "REPORT_MD",
    "REPEAT_TOLERANCES",
    "EvalDataset",
    "EvalDatasetError",
    "compare_eval_reports",
    "evaluate",
    "load_eval_dataset",
    "percentile",
    "render_markdown",
    "run_ablation",
    "run_eval_command",
]
