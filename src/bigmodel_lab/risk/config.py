"""交易签名风险审查的配置加载与校验。

配置来源：
- `configs/risk.yaml`：流水线、数据集、模型训练与评测参数（顶层 `risk:`）。
- `configs/risk-rules.yaml`：规则/阈值/白名单，带独立 `version`，由 `rules_file` 指定。

加载复用 `bigmodel_lab.config.load_config` 的 YAML 深合并与命令行覆盖能力，
但风险配置有独立 pydantic 模型，避免污染 8 阶段课程的通用配置。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..config import PROJECT_ROOT, ConfigError, LabConfig, load_config
from .schema import SCENARIO_FAMILIES

RISK_CONFIG_PATH = PROJECT_ROOT / "configs" / "risk.yaml"
RISK_RULES_PATH = PROJECT_ROOT / "configs" / "risk-rules.yaml"


class RiskConfigError(RuntimeError):
    """风险配置加载或校验失败。"""


class _RiskBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetSettings(_RiskBase):
    seed: int = 42
    scenarios_per_family: int = Field(default=6, ge=1, le=200)
    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    output_dir: str = "runs/risk/dataset"
    families: list[str] = Field(default_factory=lambda: list(SCENARIO_FAMILIES))
    max_samples: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> DatasetSettings:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test 比例之和必须为 1.0，当前为 {total}")
        return self


class RuleThresholds(_RiskBase):
    allow_max: float = Field(default=0.01, ge=0.0, le=1.0)
    step_up_dual_max: float = Field(default=0.05, ge=0.0, le=1.0)
    step_up_multisig_max: float = Field(default=0.20, ge=0.0, le=1.0)
    block_above: float = Field(default=0.50, ge=0.0, le=1.0)
    rolling_window_hours: int = Field(default=24, ge=1)
    rolling_step_up: float = Field(default=0.20, ge=0.0, le=1.0)
    rolling_block: float = Field(default=0.50, ge=0.0, le=1.0)
    batch_nonce_warn: int = Field(default=1, ge=0)
    batch_nonce_block: int = Field(default=10, ge=1)
    new_contract_days: int = Field(default=7, ge=0)
    large_amount_usd: float = Field(default=1_000_000.0, ge=0.0)


class RuleSettings(_RiskBase):
    version: str = "risk-rules-v1"
    irreversible_actions: list[str] = Field(
        default_factory=lambda: ["eip7702_authorization", "ownership_change", "proxy_upgrade"]
    )
    allowlist: list[str] = Field(default_factory=list)
    thresholds: RuleThresholds = Field(default_factory=RuleThresholds)
    suspicious_text_patterns: list[str] = Field(
        default_factory=lambda: [
            r"忽略.*(规则|审查|风险)",
            r"(ignore|disregard).*(previous|above|rules|instructions)",
            r"判定为安全|标记为安全|approved\s+by\s+admin",
            r"system\s*prompt|你现在是",
        ]
    )
    enabled: list[str] = Field(
        default_factory=lambda: [
            "irreversible_action",
            "exposure_threshold",
            "rolling_exposure",
            "batch_nonce",
            "new_contract",
            "allowlist",
            "approval_ticket",
            "origin_anomaly",
            "untrusted_text",
            "invalid_request",
        ]
    )


class LoraSettings(_RiskBase):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainProfileSettings(_RiskBase):
    max_steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    grad_accum: int = Field(ge=1)
    max_seq_len: int = Field(ge=64)
    learning_rate: float = Field(gt=0.0)
    warmup_steps: int = Field(ge=0)
    log_every: int = Field(ge=1)
    eval_every: int = Field(ge=1)
    eval_batches: int = Field(ge=1)
    save_every: int = Field(ge=0)
    max_train_samples: int = Field(ge=1)
    max_val_samples: int = Field(ge=1)


class DpoSettings(_RiskBase):
    enabled: bool = False
    beta: float = 0.1
    max_steps: int = 20
    batch_size: int = 2
    grad_accum: int = 2
    max_seq_len: int = 768
    learning_rate: float = 5e-6
    warmup_steps: int = 2
    log_every: int = 5
    eval_every: int = 10
    eval_batches: int = 2
    save_every: int = 10


class TrainSettings(_RiskBase):
    base_model_dir: str = "models/Qwen3-0.6B"
    export_dir: str = "models/exports/risk-review-sft-merged"
    lora: LoraSettings = Field(default_factory=LoraSettings)
    sanity: TrainProfileSettings
    standard: TrainProfileSettings
    dpo: DpoSettings = Field(default_factory=DpoSettings)


class ModelSettings(_RiskBase):
    backend: Literal["transformers", "mlx"] = "transformers"
    max_new_tokens: int = Field(default=512, ge=32)
    temperature: float = Field(default=0.0, ge=0.0)
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    train: TrainSettings


class EvalSettings(_RiskBase):
    output_dir: str = "runs/risk/eval"
    min_recall: float = Field(default=0.95, ge=0.0, le=1.0)
    max_fpr: float = Field(default=0.10, ge=0.0, le=1.0)
    min_asset_weighted_recall: float = Field(default=0.90, ge=0.0, le=1.0)
    binance_gate: bool = True
    latency_p95_budget_ms: float = Field(default=5000.0, gt=0.0)


class ReviewSettings(_RiskBase):
    mode: Literal["rules_only", "rules_plus_model"] = "rules_only"
    output_dir: str = "runs/risk/review"
    fail_closed: bool = True
    max_batch: int = Field(default=1000, ge=1)


class RiskSettings(_RiskBase):
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    rules_file: str = "configs/risk-rules.yaml"
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    model: ModelSettings
    eval: EvalSettings = Field(default_factory=EvalSettings)
    simulator: str = "offline-v1"


class RiskConfig(_RiskBase):
    """一次加载得到的完整风险配置：流水线设置 + 规则设置 + 通用实验配置。"""

    settings: RiskSettings
    rules: RuleSettings
    lab: LabConfig
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)
    path: str | None = None


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_rule_settings(path: str | Path) -> RuleSettings:
    rule_path = resolve_project_path(path)
    if not rule_path.exists():
        raise RiskConfigError(f"规则文件不存在: {rule_path}")
    try:
        payload = yaml.safe_load(rule_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - 依赖 PyYAML 的错误类型
        raise RiskConfigError(f"规则文件 YAML 解析失败: {rule_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RiskConfigError(f"规则文件顶层必须是映射: {rule_path}")
    block = payload.get("rules", payload)
    try:
        return RuleSettings.model_validate(block)
    except ValidationError as exc:
        raise RiskConfigError(f"规则配置校验失败: {_format_validation_error(exc)}") from exc


def load_risk_config(
    paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
) -> RiskConfig:
    """加载 risk 配置；支持多份 YAML 依次深合并，默认 `configs/risk.yaml`。"""
    if paths is None:
        config_paths: list[Path] = [RISK_CONFIG_PATH]
    elif isinstance(paths, (list, tuple)):
        config_paths = [Path(item) for item in paths] or [RISK_CONFIG_PATH]
    else:
        config_paths = [Path(paths)]
    try:
        raw, lab = load_config(*config_paths, use_default=True)
    except ConfigError as exc:
        raise RiskConfigError(str(exc)) from exc
    risk_raw = raw.get("risk")
    if not isinstance(risk_raw, dict):
        raise RiskConfigError(f"配置文件缺少顶层 `risk:` 段: {config_paths[0]}")
    try:
        settings = RiskSettings.model_validate(risk_raw)
    except ValidationError as exc:
        raise RiskConfigError(f"risk 配置校验失败: {_format_validation_error(exc)}") from exc
    rules = load_rule_settings(settings.rules_file)
    return RiskConfig(settings=settings, rules=rules, lab=lab, raw=raw, path=str(config_paths[0]))


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"{location}: {error['msg']}")
    return "; ".join(lines)


__all__ = [
    "RISK_CONFIG_PATH",
    "RISK_RULES_PATH",
    "DatasetSettings",
    "DpoSettings",
    "EvalSettings",
    "LoraSettings",
    "ModelSettings",
    "ReviewSettings",
    "RiskConfig",
    "RiskConfigError",
    "RiskSettings",
    "RuleSettings",
    "RuleThresholds",
    "TrainProfileSettings",
    "TrainSettings",
    "load_risk_config",
    "load_rule_settings",
    "resolve_project_path",
]
