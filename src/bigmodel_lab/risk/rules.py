"""版本化规则/不变量引擎：把模拟结果与请求上下文映射为可审计的规则命中。

规则输入是 `RiskConfig.rules`（`configs/risk-rules.yaml`），输出 `RuleHit` 列表；
每条命中携带规则 ID、名称、严重度、证据与规则版本，`suggested_decision` 是
决策融合的「规则地板」。引擎只给建议，不做最终放行（见 `decision.py`）。

内置规则：
- R-001 不可逆操作（EIP-7702 委托 / owner 变更 / 代理升级）
- R-002 暴露比例阈值（四档）
- R-003 滚动窗口累计暴露
- R-004 批量 nonce 跨度
- R-005 新合约/新对手方
- R-006 大额转出目标不在 allowlist
- R-007 审批票据（irreversible_ok / 金额 / 收款方 / 过期 / 允许操作）
- R-008 来源异常（service account 且无审批）
- R-009 不可信文本注入特征（只标记可疑，不直接 BLOCK）
- R-010 请求不可审查（解码/模拟失败、未知操作类型）
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from .config import RuleSettings
from .schema import (
    Decision,
    Operation,
    ReviewRequest,
    RuleHit,
    Severity,
    SimulationResult,
)
from .simulate import irreversible_operation_changes_state

# enabled 名称 -> (规则 ID, 规则名称)
RULE_DEFS: dict[str, tuple[str, str]] = {
    "irreversible_action": ("R-001", "不可逆操作"),
    "exposure_threshold": ("R-002", "暴露比例阈值"),
    "rolling_exposure": ("R-003", "滚动窗口累计暴露"),
    "batch_nonce": ("R-004", "批量 nonce 跨度"),
    "new_contract": ("R-005", "新合约/新对手方"),
    "allowlist": ("R-006", "大额转出白名单"),
    "approval_ticket": ("R-007", "审批票据"),
    "origin_anomaly": ("R-008", "来源异常"),
    "untrusted_text": ("R-009", "不可信文本"),
    "invalid_request": ("R-010", "请求不可审查"),
}

_OPERATION_KINDS = frozenset(
    {
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
    }
)
_TRANSFER_KINDS = frozenset({"native_transfer", "erc20_transfer"})
_OUTBOUND_KINDS = frozenset(
    {"native_transfer", "erc20_transfer", "batch_withdrawal", "cross_chain_transfer"}
)
_SERVICE_ORIGIN_MARKERS = ("service", "svc", "container", "devops", "runner", "pod", "cron", "bot")
# 证据里逐条列出操作/nonce 的上限，避免批量签发场景报告体积爆炸
_MAX_EVIDENCE_ITEMS = 5


class RuleConfigError(ValueError):
    """规则配置非法：引擎拒绝启动审查（CLI 以非零退出码报告）。"""


def validate_rule_settings(settings: RuleSettings) -> None:
    """校验规则配置；非法时抛出 `RuleConfigError`（fail-closed）。

    检查项：版本非空、enabled 无未知项/重复项、阈值单调、不可逆操作类型合法、
    可疑文本正则可编译。
    """
    problems: list[str] = []
    if not str(settings.version or "").strip():
        problems.append("规则版本 version 不能为空")
    unknown = [name for name in settings.enabled if name not in RULE_DEFS]
    if unknown:
        problems.append(f"未知规则启用项: {unknown}")
    duplicates = sorted({name for name in settings.enabled if settings.enabled.count(name) > 1})
    if duplicates:
        problems.append(f"规则 ID 重复（enabled 重复项）: {duplicates}")
    thresholds = settings.thresholds
    if not (
        thresholds.allow_max
        <= thresholds.step_up_dual_max
        <= thresholds.step_up_multisig_max
        <= thresholds.block_above
    ):
        problems.append(
            "暴露阈值必须单调递增: "
            f"allow_max={thresholds.allow_max}, "
            f"step_up_dual_max={thresholds.step_up_dual_max}, "
            f"step_up_multisig_max={thresholds.step_up_multisig_max}, "
            f"block_above={thresholds.block_above}"
        )
    if thresholds.rolling_step_up > thresholds.rolling_block:
        problems.append(
            f"滚动窗口阈值非法: rolling_step_up={thresholds.rolling_step_up} "
            f"> rolling_block={thresholds.rolling_block}"
        )
    if thresholds.batch_nonce_warn >= thresholds.batch_nonce_block:
        problems.append(
            f"批量 nonce 阈值非法: batch_nonce_warn={thresholds.batch_nonce_warn} "
            f">= batch_nonce_block={thresholds.batch_nonce_block}"
        )
    invalid_actions = [item for item in settings.irreversible_actions if item not in _OPERATION_KINDS]
    if invalid_actions:
        problems.append(f"不可逆操作包含未知类型: {invalid_actions}")
    for pattern in settings.suspicious_text_patterns:
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            problems.append(f"可疑文本正则非法 {pattern!r}: {exc}")
    if problems:
        raise RuleConfigError("规则配置校验失败: " + "；".join(problems))


class RuleEngine:
    """按配置启用规则并产出 `RuleHit` 列表。"""

    def __init__(self, settings: RuleSettings):
        validate_rule_settings(settings)
        self.settings = settings
        self.thresholds = settings.thresholds
        self._enabled = set(settings.enabled)
        self._patterns: list[tuple[str, re.Pattern[str]]] = [
            (pattern, re.compile(pattern, re.IGNORECASE))
            for pattern in settings.suspicious_text_patterns
        ]

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def evaluate(
        self,
        request: ReviewRequest,
        operations: list[Operation],
        simulation: SimulationResult,
    ) -> list[RuleHit]:
        """执行全部启用规则，返回按规则 ID 稳定排序的命中列表。"""
        hits: list[RuleHit] = []
        if "irreversible_action" in self._enabled:
            hits.extend(self._rule_irreversible(request, operations, simulation))
        if "exposure_threshold" in self._enabled:
            hits.extend(self._rule_exposure(request, operations, simulation))
        if "rolling_exposure" in self._enabled:
            hits.extend(self._rule_rolling(request, operations, simulation))
        if "batch_nonce" in self._enabled:
            hits.extend(self._rule_batch_nonce(request, operations, simulation))
        if "new_contract" in self._enabled:
            hits.extend(self._rule_new_contract(request, operations, simulation))
        if "allowlist" in self._enabled:
            hits.extend(self._rule_allowlist(request, operations, simulation))
        if "approval_ticket" in self._enabled:
            hits.extend(self._rule_approval_ticket(request, operations, simulation))
        if "origin_anomaly" in self._enabled:
            hits.extend(self._rule_origin_anomaly(request, operations, simulation))
        if "untrusted_text" in self._enabled:
            hits.extend(self._rule_untrusted_text(request, operations, simulation))
        if "invalid_request" in self._enabled:
            hits.extend(self._rule_invalid_request(request, operations, simulation))
        return sorted(hits, key=lambda hit: (hit.rule_id, "；".join(hit.evidence)))

    # ------------------------------------------------------------------
    # 规则实现
    # ------------------------------------------------------------------
    def _rule_irreversible(
        self,
        request: ReviewRequest,
        operations: list[Operation],
        simulation: SimulationResult,
    ) -> list[RuleHit]:
        irreversible = set(self.settings.irreversible_actions)
        relevant = [
            operation
            for operation in operations
            if operation.kind in irreversible
            and irreversible_operation_changes_state(request, operation)
        ]
        if not relevant:
            return []
        evidence = [
            "命中不可逆操作类型: " + ", ".join(sorted({operation.kind for operation in relevant})),
            "涉及链: " + ", ".join(sorted({operation.chain for operation in relevant})),
            f"账户 {request.account} 请求 {request.request_id}",
        ]
        for operation in relevant[:_MAX_EVIDENCE_ITEMS]:
            target = operation.target or operation.delegate or operation.new_owner or "-"
            detail = f"{operation.kind}@{operation.chain} target={target}"
            if operation.delegate:
                detail += f" delegate={operation.delegate}"
            evidence.append(detail)
        if len(relevant) > _MAX_EVIDENCE_ITEMS:
            evidence.append(f"…共 {len(relevant)} 条不可逆操作命中（其余明细省略）")
        ticket = request.context.approval
        approved = ticket is not None and bool(ticket.irreversible_ok)
        if approved:
            evidence.append("存在 irreversible_ok 预登记审批，仍需双人/多签带外确认")
            return [
                self._hit(
                    "irreversible_action",
                    severity="high",
                    suggested_decision="STEP_UP",
                    evidence=evidence,
                )
            ]
        evidence.append("缺少 irreversible_ok 预登记审批：不可逆操作禁止自动签名")
        return [
            self._hit(
                "irreversible_action",
                severity="critical",
                suggested_decision="BLOCK",
                evidence=evidence,
            )
        ]

    def _rule_exposure(
        self,
        request: ReviewRequest,
        operations: list[Operation],
        simulation: SimulationResult,
    ) -> list[RuleHit]:
        if simulation.failed:
            return []
        ratio = float(simulation.exposure_ratio)
        thresholds = self.thresholds
        evidence = [
            f"exposure_ratio={ratio}",
            f"exposure_usd={simulation.exposure_usd}",
            f"nav_usd={simulation.nav_usd}",
            (
                f"阈值 allow_max={thresholds.allow_max}, "
                f"step_up_dual_max={thresholds.step_up_dual_max}, "
                f"step_up_multisig_max={thresholds.step_up_multisig_max}, "
                f"block_above={thresholds.block_above}"
            ),
        ]
        if ratio <= thresholds.allow_max:
            evidence.append("暴露比例处于自动放行区间")
            return [self._hit("exposure_threshold", "info", "ALLOW", evidence)]
        if ratio <= thresholds.step_up_dual_max:
            evidence.append("暴露比例处于双人审批区间")
            return [self._hit("exposure_threshold", "medium", "STEP_UP", evidence)]
        if ratio <= thresholds.step_up_multisig_max:
            evidence.append("暴露比例处于多签审批区间")
            return [self._hit("exposure_threshold", "high", "STEP_UP", evidence)]
        if ratio < thresholds.block_above:
            evidence.append("暴露比例超过多签阈值但未到阻断线，需带外确认")
            return [self._hit("exposure_threshold", "high", "STEP_UP", evidence)]
        evidence.append("暴露比例达到/超过阻断阈值")
        return [self._hit("exposure_threshold", "critical", "BLOCK", evidence)]

    def _rule_rolling(
        self,
        request: ReviewRequest,
        operations: list[Operation],
        simulation: SimulationResult,
    ) -> list[RuleHit]:
        if simulation.failed:
            return []
        ratio = float(simulation.rolling_exposure_ratio)
        thresholds = self.thresholds
        evidence = [
            f"rolling_exposure_ratio={ratio}",
            f"window_hours={simulation.window_hours}",
            f"rolling_step_up={thresholds.rolling_step_up}",
            f"rolling_block={thresholds.rolling_block}",
            f"当前暴露比例={simulation.exposure_ratio}",
        ]
        if ratio >= thresholds.rolling_block:
            evidence.append("窗口内累计暴露达到阻断阈值")
            return [self._hit("rolling_exposure", "critical", "BLOCK", evidence)]
        if ratio >= thresholds.rolling_step_up:
            evidence.append("窗口内累计暴露达到二次确认阈值")
            return [self._hit("rolling_exposure", "high", "STEP_UP", evidence)]
        return []

    def _rule_batch_nonce(
        self,
        request: ReviewRequest,
        operations: list[Operation],
        simulation: SimulationResult,
    ) -> list[RuleHit]:
        span = _nonce_span(operations, simulation)
        thresholds = self.thresholds
        if span <= thresholds.batch_nonce_warn:
            return []
        evidence = [
            f"nonce_span={span}",
            f"batch_nonce_warn={thresholds.batch_nonce_warn}",
            f"batch_nonce_block={thresholds.batch_nonce_block}",
            f"操作数={len(operations)}",
        ]
        nonce_operations = [operation for operation in operations if operation.nonce is not None]
        for operation in nonce_operations[:_MAX_EVIDENCE_ITEMS]:
            evidence.append(f"{operation.kind}@{operation.chain} nonce={operation.nonce}")
        if len(nonce_operations) > _MAX_EVIDENCE_ITEMS:
            evidence.append(f"…共 {len(nonce_operations)} 条带 nonce 的操作（其余明细省略）")
        if span >= thresholds.batch_nonce_block:
            evidence.append("批量 nonce 跨度达到阻断阈值（疑似批量签发）")
            return [self._hit("batch_nonce", "critical", "BLOCK", evidence)]
        evidence.append("批量 nonce 跨度超过告警阈值，需二次确认")
        return [self._hit("batch_nonce", "high", "STEP_UP", evidence)]

    def _rule_new_contract(
        self,
        request: ReviewRequest,
        operations: list[Operation],
        simulation: SimulationResult,
    ) -> list[RuleHit]:
        context = request.context
        trusted = _address_set(
            [
                *self.settings.allowlist,
                *context.allowlist,
                *context.preapproved_contracts,
                *context.known_counterparties,
            ]
        )
        evidence: list[str] = []
        blocking = False
        seen: set[str] = set()
        for operation in operations:
            target = operation.target or operation.delegate or operation.new_owner
            if not target:
                continue
            key = _normalize_address(target)
            if not key or key in seen or _is_zero_address(target):
                continue
            seen.add(key)
            if key in trusted:
                continue
            state = _chain_state(request, operation.chain)
            age = _contract_age(state, target)
            if age is not None and int(age) >= self.thresholds.new_contract_days:
                continue
            age_text = "缺失" if age is None else f"{age} 天"
            is_large = simulation.exposure_usd >= self.thresholds.large_amount_usd
            blocking = blocking or is_large
            evidence.append(
                f"{operation.kind}@{operation.chain} target={target} "
                f"合约年龄={age_text}（观察期 {self.thresholds.new_contract_days} 天）"
            )
        if not evidence:
            return []
        evidence.append("目标不在预登记合约/已知对手方/白名单中")
        decision: Decision = "BLOCK" if blocking else "STEP_UP"
        if blocking:
            evidence.append(
                f"暴露 {simulation.exposure_usd} USD 达到大额阈值 "
                f"{self.thresholds.large_amount_usd} USD，新合约/新对手方直接阻断"
            )
        return [
            self._hit(
                "new_contract",
                "critical" if blocking else "high",
                decision,
                evidence,
            )
        ]

    def _rule_allowlist(
        self,
        request: ReviewRequest,
        operations: list[Operation],
        simulation: SimulationResult,
    ) -> list[RuleHit]:
        if simulation.failed:
            return []
        outbound = [operation for operation in operations if operation.kind in _OUTBOUND_KINDS]
        if not outbound:
            return []
        if simulation.exposure_usd < self.thresholds.large_amount_usd:
            return []
        allowlist = _address_set([*self.settings.allowlist, *request.context.allowlist])
        targets = [_operation_target(operation) for operation in outbound]
        bad_targets = [target for target in targets if target is None or _normalize_address(target) not in allowlist]
        if not bad_targets:
            return []
        evidence = [
            f"大额转出 exposure_usd={simulation.exposure_usd}",
            f"large_amount_usd={self.thresholds.large_amount_usd}",
            "目标未在白名单: " + ", ".join(str(target) for target in bad_targets),
        ]
        return [self._hit("allowlist", "high", "STEP_UP", evidence)]

    def _rule_approval_ticket(
        self,
        request: ReviewRequest,
        operations: list[Operation],
        simulation: SimulationResult,
    ) -> list[RuleHit]:
        ticket = request.context.approval
        irreversible = set(self.settings.irreversible_actions)
        irreversible_ops = [
            operation
            for operation in operations
            if operation.kind in irreversible
            and irreversible_operation_changes_state(request, operation)
        ]
        hits: list[RuleHit] = []
        if irreversible_ops and (ticket is None or not ticket.irreversible_ok):
            hits.append(
                self._hit(
                    "approval_ticket",
                    "critical",
                    "BLOCK",
                    [
                        "不可逆操作缺少 irreversible_ok 审批",
                        "操作类型: " + ", ".join(sorted({operation.kind for operation in irreversible_ops})),
                    ],
                )
            )
        if ticket is None:
            return hits
        if ticket.amount_usd is not None and not simulation.failed:
            approved_amount = float(ticket.amount_usd)
            if float(simulation.exposure_usd) > approved_amount + 1e-6:
                hits.append(
                    self._hit(
                        "approval_ticket",
                        "high",
                        "STEP_UP",
                        [
                            f"请求暴露 {simulation.exposure_usd} USD 超过审批金额 {approved_amount} USD",
                            f"ticket_id={ticket.ticket_id}",
                        ],
                    )
                )
        if ticket.recipient:
            expected = _normalize_address(ticket.recipient)
            mismatched = [
                operation.target
                for operation in operations
                if operation.target and _normalize_address(operation.target) != expected
            ]
            if mismatched:
                hits.append(
                    self._hit(
                        "approval_ticket",
                        "high",
                        "STEP_UP",
                        [
                            f"审批收款方 {ticket.recipient} 与实际目标不一致: "
                            + ", ".join(str(target) for target in mismatched),
                            f"ticket_id={ticket.ticket_id}",
                        ],
                    )
                )
        if ticket.allowed_operations:
            allowed = set(ticket.allowed_operations)
            unexpected = sorted({operation.kind for operation in operations if operation.kind not in allowed})
            if unexpected:
                hits.append(
                    self._hit(
                        "approval_ticket",
                        "high",
                        "STEP_UP",
                        [
                            "审批未覆盖的操作类型: " + ", ".join(unexpected),
                            f"ticket_id={ticket.ticket_id}",
                        ],
                    )
                )
        expiry = _parse_timestamp(ticket.expires_at)
        request_time = _parse_timestamp(request.timestamp)
        if expiry is not None and request_time is not None and request_time > expiry:
            hits.append(
                self._hit(
                    "approval_ticket",
                    "high",
                    "STEP_UP",
                    [
                        f"审批票据已过期: expires_at={ticket.expires_at}, 请求时间={request.timestamp}",
                        f"ticket_id={ticket.ticket_id}",
                    ],
                )
            )
        return hits

    def _rule_origin_anomaly(
        self,
        request: ReviewRequest,
        operations: list[Operation],
        simulation: SimulationResult,
    ) -> list[RuleHit]:
        if request.context.approval is not None:
            return []
        if not _is_service_origin(request):
            return []
        evidence = [
            f"来源 origin={request.context.origin}",
            f"账户角色 account_role={request.context.account_role}",
            f"渠道 channel={request.context.channel}",
            "service account 发起且无审批票据",
        ]
        return [self._hit("origin_anomaly", "medium", "STEP_UP", evidence)]

    def _rule_untrusted_text(
        self,
        request: ReviewRequest,
        operations: list[Operation],
        simulation: SimulationResult,
    ) -> list[RuleHit]:
        matches: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for field_name, text in _untrusted_texts(request, operations):
            for pattern, compiled in self._patterns:
                if compiled.search(text) and (field_name, pattern) not in seen:
                    seen.add((field_name, pattern))
                    matches.append((field_name, pattern))
        if not matches:
            return []
        evidence = [f"不可信字段 {field_name} 命中可疑文本模式 {pattern!r}" for field_name, pattern in matches]
        evidence.append("外部文本只作为数据处理，不改变确定性规则；该命中仅标记可疑并要求二次确认")
        return [self._hit("untrusted_text", "medium", "STEP_UP", evidence)]

    def _rule_invalid_request(
        self,
        request: ReviewRequest,
        operations: list[Operation],
        simulation: SimulationResult,
    ) -> list[RuleHit]:
        hits: list[RuleHit] = []
        if simulation.failed:
            hits.append(
                self._hit(
                    "invalid_request",
                    "critical",
                    "ESCALATE",
                    [
                        "模拟失败，无法确认签名会让渡的能力与资产暴露",
                        f"failure_reason={simulation.failure_reason}",
                    ],
                )
            )
        unknown = [operation for operation in operations if operation.kind == "unknown"]
        if unknown:
            hits.append(
                self._hit(
                    "invalid_request",
                    "high",
                    "ESCALATE",
                    [
                        f"存在 {len(unknown)} 条无法识别的操作类型，不能自动放行",
                        "涉及链: " + ", ".join(sorted({operation.chain for operation in unknown})),
                    ],
                )
            )
        return hits

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _hit(
        self,
        enabled_name: str,
        severity: Severity,
        suggested_decision: Decision,
        evidence: list[str],
    ) -> RuleHit:
        rule_id, name = RULE_DEFS[enabled_name]
        return RuleHit(
            rule_id=rule_id,
            name=name,
            severity=severity,
            evidence=evidence,
            rule_version=self.settings.version,
            suggested_decision=suggested_decision,
        )


def evaluate_rules(
    request: ReviewRequest,
    operations: list[Operation],
    simulation: SimulationResult,
    rules: RuleSettings,
) -> list[RuleHit]:
    """便捷入口：构造 `RuleEngine` 并执行全部启用规则。"""
    return RuleEngine(rules).evaluate(request, operations, simulation)


# ---------------------------------------------------------------------------
# 纯工具函数
# ---------------------------------------------------------------------------


def _nonce_span(operations: list[Operation], simulation: SimulationResult) -> int:
    """批量 nonce 跨度：显式 nonce_span 优先；batch_size 仅在存在 nonce 时参与推断。"""
    span = max((delta.nonce_span for delta in simulation.per_chain), default=0)
    for operation in operations:
        operation_span = int(operation.nonce_span or 0)
        if operation_span <= 0 and operation.nonce is not None and int(operation.batch_size or 1) > 1:
            operation_span = int(operation.batch_size or 1)
        if operation.kind == "eip7702_authorization" and operation_span <= 0:
            operation_span = 1
        span = max(span, operation_span)
    return span


def _chain_state(request: ReviewRequest, chain: str):
    from .decode import chain_state  # noqa: PLC0415 - 避免 rules→decode 顶层循环依赖

    return chain_state(request, chain)


def _contract_age(state: Any, target: str) -> int | None:
    if state is None:
        return None
    key = _normalize_address(target)
    for address, age in state.contract_age_days.items():
        if _normalize_address(address) == key:
            return int(age)
    return None


def _operation_target(operation: Operation) -> str | None:
    if operation.target:
        return operation.target
    items = operation.metadata.get("items") or operation.metadata.get("withdrawals")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and item.get("target"):
                return str(item["target"])
    return None


def _address_set(values: Any) -> set[str]:
    return {_normalize_address(str(value)) for value in values if value}


def _normalize_address(address: str | None) -> str:
    return str(address or "").strip().lower()


def _is_zero_address(address: str | None) -> bool:
    """零地址/空地址代表「无对手方」（如撤销委托、renounce），不算新合约。"""
    normalized = _normalize_address(address)
    if not normalized:
        return False
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    return bool(normalized) and set(normalized) == {"0"}


def _is_service_origin(request: ReviewRequest) -> bool:
    if request.context.account_role == "ops":
        return True
    origin = str(request.context.origin or "").strip().lower()
    return any(marker in origin for marker in _SERVICE_ORIGIN_MARKERS)


def _untrusted_texts(
    request: ReviewRequest, operations: list[Operation]
) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    if request.context.notes:
        texts.append(("context.notes", str(request.context.notes)))
    if request.context.ui_text:
        texts.append(("context.ui_text", str(request.context.ui_text)))
    for index, operation in enumerate(operations):
        _walk_strings(operation.metadata, f"operations[{index}].metadata", texts)
    return texts


def _walk_strings(value: Any, path: str, texts: list[tuple[str, str]]) -> None:
    if isinstance(value, str):
        texts.append((path, value))
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            _walk_strings(value[key], f"{path}.{key}", texts)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_strings(item, f"{path}[{index}]", texts)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


__all__ = [
    "RULE_DEFS",
    "RuleConfigError",
    "RuleEngine",
    "evaluate_rules",
    "validate_rule_settings",
]
