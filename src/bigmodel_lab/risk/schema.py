"""交易签名风险审查的数据模型。

所有输入/输出均通过 pydantic 校验；`schema_version` 不匹配时拒绝。
模型本身不持有私钥、不发起签名，只描述审查所需的事实与结论。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1

Decision = Literal["ALLOW", "STEP_UP", "BLOCK", "ESCALATE"]
RiskLevel = Literal["low", "medium", "high", "critical"]
Severity = Literal["info", "low", "medium", "high", "critical"]
CapabilityDelta = Literal[
    "none",
    "value_transfer",
    "allowance_grant",
    "control_grant",
    "full_control",
]
OperationKind = Literal[
    "native_transfer",
    "erc20_transfer",
    "approve",
    "permit",
    "eip7702_authorization",
    "ownership_change",
    "proxy_upgrade",
    "batch_withdrawal",
    "cross_chain_transfer",
    "unknown",
]
RequestKind = Literal["structured", "evm_tx", "eip712", "eip7702_authorization"]
AccountRole = Literal["hot_wallet", "cold_wallet", "treasury", "ops", "personal"]

SCENARIO_FAMILIES: tuple[str, ...] = (
    "normal_ops",
    "native_transfer",
    "batch_withdrawal",
    "approve_permit",
    "eip7702_delegation",
    "ownership_change",
    "proxy_upgrade",
    "batch_nonce",
    "cross_chain_combo",
)

DECISION_RANK: dict[str, int] = {"ALLOW": 0, "STEP_UP": 1, "ESCALATE": 2, "BLOCK": 3}
RISK_LEVEL_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SEVERITY_RANK: dict[str, int] = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

# 退出码约定：0=审查完成且无阻断；2=存在 BLOCK 或门禁失败；3=ESCALATE/无法完成审查
EXIT_CONTRACT = {"allow": 0, "step_up": 0, "block": 2, "escalate": 3}


def exit_code_for(decision: str) -> int:
    return EXIT_CONTRACT.get(str(decision).lower(), 3)


def max_decision(*decisions: str | None) -> Decision:
    """返回给定决策中风险级别最高者。"""
    best: Decision = "ALLOW"
    for decision in decisions:
        if decision and DECISION_RANK.get(decision, 0) > DECISION_RANK[best]:
            best = decision  # type: ignore[assignment]
    return best


def canonical_json(value: Any) -> str:
    """稳定序列化：用于证据与报告的可复现输出。"""
    import json  # noqa: PLC0415

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


class RiskModel(BaseModel):
    """所有审查模型的基类：未知字段一律拒绝，避免误配被静默忽略。"""

    model_config = ConfigDict(extra="forbid")


class Allowance(RiskModel):
    token: str
    spender: str
    amount: float | None = None
    unlimited: bool = False


class Authority(RiskModel):
    owner: str | None = None
    pending_owner: str | None = None
    admin: str | None = None
    delegate: str | None = None
    implementation: str | None = None
    modules: list[str] = Field(default_factory=list)
    guardians: list[str] = Field(default_factory=list)


class ChainState(RiskModel):
    chain: str
    balances: dict[str, float] = Field(default_factory=dict)
    prices_usd: dict[str, float] = Field(default_factory=dict)
    allowances: list[Allowance] = Field(default_factory=list)
    authority: Authority = Field(default_factory=Authority)
    code_hashes: dict[str, str] = Field(default_factory=dict)
    contract_age_days: dict[str, int] = Field(default_factory=dict)
    nonce: int = 0

    def nav_usd(self) -> float:
        total = 0.0
        for token, amount in self.balances.items():
            total += float(amount) * float(self.prices_usd.get(token, 0.0))
        return round(total, 6)

    def token_usd(self, token: str) -> float:
        amount = float(self.balances.get(token, 0.0))
        return round(amount * float(self.prices_usd.get(token, 0.0)), 6)


class WorldState(RiskModel):
    chains: dict[str, ChainState] = Field(default_factory=dict)

    def nav_usd(self, chain: str | None = None) -> float:
        if chain is not None:
            state = self.chains.get(chain)
            return state.nav_usd() if state else 0.0
        return round(sum(state.nav_usd() for state in self.chains.values()), 6)

    def chain(self, name: str) -> ChainState | None:
        return self.chains.get(name)


class ApprovalTicket(RiskModel):
    ticket_id: str
    approved_by: list[str] = Field(default_factory=list)
    amount_usd: float | None = None
    recipient: str | None = None
    token: str | None = None
    allowed_operations: list[str] = Field(default_factory=list)
    irreversible_ok: bool = False
    expires_at: str | None = None
    action_signature: str | None = None


class RequestContext(RiskModel):
    origin: str
    timestamp: str
    operator: str | None = None
    channel: str = "api"
    ip: str | None = None
    account_role: AccountRole = "hot_wallet"
    approval: ApprovalTicket | None = None
    notes: str | None = None
    ui_text: str | None = None
    allowlist: list[str] = Field(default_factory=list)
    known_counterparties: list[str] = Field(default_factory=list)
    preapproved_contracts: list[str] = Field(default_factory=list)


class HistoryEntry(RiskModel):
    timestamp: str
    action_type: str
    target: str | None = None
    amount_usd: float = 0.0
    exposure_ratio: float = 0.0
    origin: str | None = None
    decision: Decision | None = None


class Operation(RiskModel):
    kind: OperationKind
    chain: str
    account: str
    target: str | None = None
    token: str | None = None
    amount: float | None = None
    amount_usd: float | None = None
    allowance: float | None = None
    unlimited_allowance: bool = False
    new_owner: str | None = None
    implementation: str | None = None
    delegate: str | None = None
    nonce: int | None = None
    nonce_span: int = 0
    batch_size: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewRequest(RiskModel):
    schema_version: int = SCHEMA_VERSION
    request_id: str
    timestamp: str
    account: str
    chain: str
    kind: RequestKind = "structured"
    payload: dict[str, Any] = Field(default_factory=dict)
    operations: list[Operation] = Field(default_factory=list)
    context: RequestContext
    world: WorldState
    history: list[HistoryEntry] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(f"不支持的 schema_version={value}，当前支持 {SCHEMA_VERSION}")
        return value


class Label(RiskModel):
    action_type: str
    capability_delta: CapabilityDelta
    exposure_ratio: float = Field(ge=0.0)
    risk_level: RiskLevel
    expected_decision: Decision
    scenario_family: str
    risk_factors: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class RiskSample(RiskModel):
    schema_version: int = SCHEMA_VERSION
    sample_id: str
    request: ReviewRequest
    label: Label

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(f"不支持的 schema_version={value}，当前支持 {SCHEMA_VERSION}")
        return value


class RuleHit(RiskModel):
    rule_id: str
    name: str
    severity: Severity
    evidence: list[str] = Field(default_factory=list)
    rule_version: str
    suggested_decision: Decision


class ModelOpinion(RiskModel):
    available: bool = False
    degraded: bool = False
    degraded_reason: str | None = None
    invalid_output: bool = False
    risk_level: RiskLevel | None = None
    recommended_action: Decision | None = None
    capability_delta: CapabilityDelta | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_factors: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    raw_output: str | None = None


class ChainDelta(RiskModel):
    chain: str
    balance_delta: dict[str, float] = Field(default_factory=dict)
    allowance_delta: list[Allowance] = Field(default_factory=list)
    authority_delta: dict[str, str | None] = Field(default_factory=dict)
    code_delta: dict[str, str] = Field(default_factory=dict)
    usd_at_risk: float = 0.0
    nonce_span: int = 0
    notes: list[str] = Field(default_factory=list)


class SimulationResult(RiskModel):
    capability_delta: CapabilityDelta = "none"
    exposure_usd: float = 0.0
    exposure_ratio: float = 0.0
    nav_usd: float = 0.0
    worst_loss_usd: float = 0.0
    rolling_exposure_ratio: float = 0.0
    cross_chain_exposure_ratio: float = 0.0
    window_hours: int = 24
    per_chain: list[ChainDelta] = Field(default_factory=list)
    failed: bool = False
    failure_reason: str | None = None


class ReviewReport(RiskModel):
    schema_version: int = SCHEMA_VERSION
    report_id: str
    request_id: str
    generated_at: str
    account: str
    chain: str
    operations: list[Operation] = Field(default_factory=list)
    simulation: SimulationResult
    rule_hits: list[RuleHit] = Field(default_factory=list)
    model_opinion: ModelOpinion | None = None
    decision: Decision
    exit_code: int
    evidence: list[str] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
    duration_ms: float = 0.0
    suspicious_text: bool = False
    degraded: bool = False
    notes: list[str] = Field(default_factory=list)


__all__ = [
    "SCHEMA_VERSION",
    "SCENARIO_FAMILIES",
    "DECISION_RANK",
    "RISK_LEVEL_RANK",
    "SEVERITY_RANK",
    "EXIT_CONTRACT",
    "AccountRole",
    "Allowance",
    "ApprovalTicket",
    "Authority",
    "CapabilityDelta",
    "ChainDelta",
    "ChainState",
    "Decision",
    "HistoryEntry",
    "Label",
    "ModelOpinion",
    "Operation",
    "OperationKind",
    "RequestContext",
    "RequestKind",
    "ReviewReport",
    "ReviewRequest",
    "RiskLevel",
    "RiskModel",
    "RiskSample",
    "RuleHit",
    "Severity",
    "SimulationResult",
    "WorldState",
    "canonical_json",
    "exit_code_for",
    "max_decision",
]
