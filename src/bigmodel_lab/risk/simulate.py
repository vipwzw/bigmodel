"""确定性状态模拟：基于 world snapshot 计算签名后的资产/授权/权限/代码差分。

教学模拟器是**语义级**而非完整 EVM 执行器：只建模审查关心的状态转移
（转账、批量提现、approve/permit、EIP-7702 委托、owner 变更、代理升级、
跨链组合），并据此计算能力变化与最坏情况资产暴露。所有输入来自请求携带的
世界状态快照，不访问 RPC、不广播交易，结果可复现。

暴露口径（USD）：
- 普通转账：转出金额的 USD 价值；
- 批量提现：所有提现腿合计 USD；
- approve/permit：`min(额度, 余额) * 价格`，无限额度取全部余额；
- EIP-7702 委托：该链全部资产净值（capability=full_control）；
- owner 变更 / 代理升级：该链全部资产净值（capability=control_grant）；
- 跨链组合：所有相关链的暴露合计。

`exposure_ratio = exposure_usd / nav_usd`（nav 仅统计相关链，为 0 时记 0，
上限 1.0）；模拟失败（缺链快照/没有操作/异常）时 `failed=True`，由规则 fail-closed。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .decode import resolve_chain, token_balance, token_price
from .schema import (
    Allowance,
    CapabilityDelta,
    ChainDelta,
    Operation,
    ReviewRequest,
    SimulationResult,
    WorldState,
)

_CAPABILITY_RANK: dict[str, int] = {
    "none": 0,
    "value_transfer": 1,
    "allowance_grant": 2,
    "control_grant": 3,
    "full_control": 4,
}

_TRANSFER_KINDS = frozenset({"native_transfer", "erc20_transfer"})


class _MissingChainError(RuntimeError):
    """请求引用的链在 world snapshot 中不存在。"""


@dataclass
class _ChainAccumulator:
    """单链差分累加器。"""

    chain: str
    balance_delta: dict[str, float] = field(default_factory=dict)
    allowance_delta: list[Allowance] = field(default_factory=list)
    authority_delta: dict[str, str | None] = field(default_factory=dict)
    code_delta: dict[str, str] = field(default_factory=dict)
    usd_at_risk: float = 0.0
    nonce_span: int = 0
    notes: list[str] = field(default_factory=list)

    def to_delta(self) -> ChainDelta:
        return ChainDelta(
            chain=self.chain,
            balance_delta={key: round(self.balance_delta[key], 6) for key in sorted(self.balance_delta)},
            allowance_delta=list(self.allowance_delta),
            authority_delta={key: self.authority_delta[key] for key in sorted(self.authority_delta)},
            code_delta={key: self.code_delta[key] for key in sorted(self.code_delta)},
            usd_at_risk=round(self.usd_at_risk, 6),
            nonce_span=self.nonce_span,
            notes=list(self.notes),
        )


class _Simulator:
    """把操作依次作用到世界状态快照上，累积每链差分。"""

    def __init__(self, request: ReviewRequest, nav_by_chain: dict[str, float]):
        self.request = request
        self.world = request.world
        self.nav_by_chain = nav_by_chain
        self.accs: dict[str, _ChainAccumulator] = {}

    def resolve(self, chain: str | None) -> str:
        resolved = resolve_chain(self.world, chain)
        if resolved is None:
            raise _MissingChainError(f"世界状态缺少链快照: {chain!r}")
        return resolved

    def accumulator(self, chain: str) -> _ChainAccumulator:
        if chain not in self.accs:
            self.accs[chain] = _ChainAccumulator(chain=chain)
        return self.accs[chain]


def simulate_request(request: ReviewRequest, operations: list[Operation]) -> SimulationResult:
    """模拟一批操作对 `request.world` 的影响。

    Args:
        request: 审查请求（携带账户世界状态快照与账户地址）。
        operations: 已解码的操作列表。

    Returns:
        `SimulationResult`；任何无法安全建模的情况返回 `failed=True` 且写明原因，
        调用方不得据此放行。
    """
    try:
        return _simulate(request, operations)
    except _MissingChainError as exc:
        return SimulationResult(failed=True, failure_reason=str(exc))
    except Exception as exc:  # noqa: BLE001 - fail-closed：任何模拟异常都不放行
        return SimulationResult(failed=True, failure_reason=f"模拟异常: {type(exc).__name__}: {exc}")


def _simulate(request: ReviewRequest, operations: list[Operation]) -> SimulationResult:
    if not operations:
        return SimulationResult(failed=True, failure_reason="请求不包含可模拟的操作")

    involved, missing = _involved_chains(request.world, operations)
    if missing:
        return SimulationResult(
            failed=True,
            failure_reason=f"世界状态缺少链快照: {sorted(set(missing))}",
        )
    nav_by_chain = {
        chain: float(request.world.chains[chain].nav_usd()) for chain in involved
    }
    nav_usd = round(sum(nav_by_chain.values()), 6)

    simulator = _Simulator(request, nav_by_chain)
    capability: CapabilityDelta = "none"
    for index, operation in enumerate(operations):
        operation_capability = _apply_operation(simulator, request, index, operation)
        capability = _max_capability(capability, operation_capability)

    exposure_usd = round(sum(acc.usd_at_risk for acc in simulator.accs.values()), 6)
    if nav_usd > 0:
        exposure_ratio = round(min(exposure_usd / nav_usd, 1.0), 6)
        worst_loss_usd = round(min(exposure_usd, nav_usd), 6)
    else:
        exposure_ratio = 0.0
        worst_loss_usd = exposure_usd

    per_chain = [simulator.accs[chain].to_delta() for chain in sorted(simulator.accs)]
    return SimulationResult(
        capability_delta=capability,
        exposure_usd=exposure_usd,
        exposure_ratio=exposure_ratio,
        nav_usd=nav_usd,
        worst_loss_usd=worst_loss_usd,
        per_chain=per_chain,
        failed=False,
    )


def _involved_chains(
    world: WorldState, operations: list[Operation]
) -> tuple[list[str], list[str]]:
    """解析所有涉及的链；返回 (已解析链列表, 缺失链列表)。"""
    involved: list[str] = []
    missing: list[str] = []
    for operation in operations:
        candidates: list[str | None] = [operation.chain]
        for leg in _cross_chain_legs(operation):
            candidates.append(leg.get("chain") or operation.chain)
        for raw_chain in candidates:
            resolved = resolve_chain(world, raw_chain)
            if resolved is None:
                missing.append(str(raw_chain))
            elif resolved not in involved:
                involved.append(resolved)
    return involved, missing


def _cross_chain_legs(operation: Operation) -> list[dict[str, Any]]:
    if operation.kind != "cross_chain_transfer":
        return []
    legs = operation.metadata.get("legs") or operation.metadata.get("items")
    if isinstance(legs, list):
        return [leg for leg in legs if isinstance(leg, dict)]
    return []


def _apply_operation(
    simulator: _Simulator, request: ReviewRequest, index: int, operation: Operation
) -> CapabilityDelta:
    """应用一条操作并返回其能力变化。"""
    kind = operation.kind
    if kind in _TRANSFER_KINDS:
        return _apply_transfer(simulator, index, operation)
    if kind == "batch_withdrawal":
        return _apply_batch_withdrawal(simulator, index, operation)
    if kind in {"approve", "permit"}:
        return _apply_allowance(simulator, index, operation)
    if kind == "eip7702_authorization":
        return _apply_eip7702(simulator, index, operation)
    if kind == "ownership_change":
        return _apply_ownership_change(simulator, index, operation)
    if kind == "proxy_upgrade":
        return _apply_proxy_upgrade(simulator, index, operation)
    if kind == "cross_chain_transfer":
        return _apply_cross_chain(simulator, index, operation)
    # unknown：不猜测语义，记录后由规则 R-010 走 ESCALATE
    chain = simulator.resolve(operation.chain)
    accumulator = simulator.accumulator(chain)
    accumulator.notes.append(f"操作 {index}: 未知类型 {kind}，无法评估资产暴露")
    return "none"


def _apply_transfer(
    simulator: _Simulator, index: int, operation: Operation
) -> CapabilityDelta:
    chain = simulator.resolve(operation.chain)
    state = simulator.world.chains[chain]
    token = operation.token or "native"
    amount = float(operation.amount or 0.0)
    usd = _operation_usd(operation, state, token)
    accumulator = simulator.accumulator(chain)
    accumulator.usd_at_risk += usd
    accumulator.balance_delta[token] = accumulator.balance_delta.get(token, 0.0) - amount
    accumulator.nonce_span = max(accumulator.nonce_span, _operation_nonce_span(operation))
    note = f"操作 {index}: {operation.kind} 转出 {amount:g} {token}（≈{usd:g} USD）"
    if usd == 0.0 and amount > 0.0:
        note += "；缺少价格快照，暴露可能被低估"
    accumulator.notes.append(note)
    return "value_transfer"


def _apply_batch_withdrawal(
    simulator: _Simulator, index: int, operation: Operation
) -> CapabilityDelta:
    items = operation.metadata.get("items") or operation.metadata.get("withdrawals")
    accumulator = simulator.accumulator(simulator.resolve(operation.chain))
    total_usd = 0.0
    if isinstance(items, list) and items:
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                raise _MissingChainError(f"操作 {index} 的批量提现第 {item_index} 项不是对象")
            chain = simulator.resolve(str(item.get("chain") or operation.chain))
            state = simulator.world.chains[chain]
            token = str(item.get("token") or operation.token or "native")
            amount = _as_float(item.get("amount"), 0.0)
            usd = _as_float(item.get("amount_usd"), None)
            if usd is None:
                usd = round(amount * token_price(state, token), 6)
            target_acc = simulator.accumulator(chain)
            target_acc.usd_at_risk += usd
            target_acc.balance_delta[token] = target_acc.balance_delta.get(token, 0.0) - amount
            total_usd += usd
        accumulator.notes.append(f"操作 {index}: 批量提现 {len(items)} 笔，合计≈{round(total_usd, 6):g} USD")
    else:
        state = simulator.world.chains[accumulator.chain]
        token = operation.token or "native"
        amount = float(operation.amount or 0.0)
        usd = _operation_usd(operation, state, token)
        accumulator.usd_at_risk += usd
        accumulator.balance_delta[token] = accumulator.balance_delta.get(token, 0.0) - amount
        accumulator.notes.append(f"操作 {index}: 批量提现 {amount:g} {token}（≈{usd:g} USD）")
    accumulator.nonce_span = max(accumulator.nonce_span, _operation_nonce_span(operation))
    return "value_transfer"


def _apply_allowance(
    simulator: _Simulator, index: int, operation: Operation
) -> CapabilityDelta:
    chain = simulator.resolve(operation.chain)
    state = simulator.world.chains[chain]
    token = operation.token
    allowance = operation.allowance if operation.allowance is not None else operation.amount
    unlimited = bool(operation.unlimited_allowance)
    balance = token_balance(state, token)
    price = token_price(state, token)
    if unlimited:
        exposure = round(balance * price, 6)
        allowance_value: float | None = None
    else:
        allowance_value = float(allowance or 0.0)
        exposure = round(min(allowance_value, balance) * price, 6)
    if exposure == 0.0 and operation.amount_usd is not None:
        # token/余额快照缺失时，允许请求携带的预计算暴露兜底
        exposure = round(float(operation.amount_usd), 6)
    accumulator = simulator.accumulator(chain)
    accumulator.usd_at_risk += exposure
    accumulator.allowance_delta.append(
        Allowance(
            token=token or "",
            spender=operation.target or "",
            amount=allowance_value,
            unlimited=unlimited,
        )
    )
    accumulator.nonce_span = max(accumulator.nonce_span, _operation_nonce_span(operation))
    authority_kind = "无限额度授权" if unlimited else f"授权额度 {allowance_value:g}"
    accumulator.notes.append(
        f"操作 {index}: {operation.kind} {authority_kind} 给 {operation.target or '未知对象'}"
        f"（token={token or '未知'}，最坏暴露≈{exposure:g} USD）"
    )
    return "allowance_grant"


def _apply_eip7702(
    simulator: _Simulator, index: int, operation: Operation
) -> CapabilityDelta:
    chain = simulator.resolve(operation.chain)
    state = simulator.world.chains[chain]
    accumulator = simulator.accumulator(chain)
    delegate = operation.delegate or operation.target
    current = state.authority.delegate
    accumulator.nonce_span = max(accumulator.nonce_span, _operation_nonce_span(operation))
    if operation.nonce is not None:
        accumulator.notes.append(f"操作 {index}: 授权 nonce={operation.nonce}")
    if _is_zero_address(delegate):
        if current:
            accumulator.authority_delta["delegate"] = None
            accumulator.notes.append(
                f"操作 {index}: 撤销既有 7702 委托（原委托 {current}），"
                "账户执行权收回，本次签名无新增暴露"
            )
        else:
            accumulator.notes.append(f"操作 {index}: 7702 委托目标为空地址，状态无变化")
        return "none"
    if _same_address(current, delegate):
        accumulator.notes.append(
            f"操作 {index}: 账户已委托给 {delegate}，本次签名不新增执行权（幂等操作）"
        )
        return "none"
    exposure = round(simulator.nav_by_chain.get(chain, 0.0), 6)
    accumulator.usd_at_risk += exposure
    accumulator.authority_delta["delegate"] = delegate
    accumulator.notes.append(
        f"操作 {index}: EIP-7702 委托给 {delegate}，"
        f"账户执行权完全让渡（该链 NAV≈{exposure:g} USD）"
    )
    return "full_control"


def _apply_ownership_change(
    simulator: _Simulator, index: int, operation: Operation
) -> CapabilityDelta:
    chain = simulator.resolve(operation.chain)
    state = simulator.world.chains[chain]
    accumulator = simulator.accumulator(chain)
    new_owner = operation.new_owner or operation.target
    current = state.authority.owner
    if _is_zero_address(new_owner):
        if current:
            accumulator.authority_delta["owner"] = None
            accumulator.notes.append(
                f"操作 {index}: owner 变更为零地址（renounce），控制权交出，无新增暴露"
            )
        else:
            accumulator.notes.append(f"操作 {index}: owner 本为空，状态无变化")
        return "none"
    if _same_address(current, new_owner):
        accumulator.notes.append(
            f"操作 {index}: owner 已是 {new_owner}，本次签名不新增控制权（幂等操作）"
        )
        return "none"
    exposure = round(simulator.nav_by_chain.get(chain, 0.0), 6)
    accumulator.usd_at_risk += exposure
    accumulator.authority_delta["owner"] = new_owner
    accumulator.notes.append(
        f"操作 {index}: owner 变更为 {new_owner or '未知地址'}，控制权转移（NAV≈{exposure:g} USD）"
    )
    return "control_grant"


def _apply_proxy_upgrade(
    simulator: _Simulator, index: int, operation: Operation
) -> CapabilityDelta:
    chain = simulator.resolve(operation.chain)
    state = simulator.world.chains[chain]
    accumulator = simulator.accumulator(chain)
    implementation = (
        operation.implementation
        or (operation.metadata.get("implementation") if operation.metadata else None)
        or "unknown"
    )
    current = state.authority.implementation
    if _same_address(current, str(implementation)):
        accumulator.notes.append(
            f"操作 {index}: 代理实现已是 {implementation}，本次升级无代码变化（幂等操作）"
        )
        return "none"
    exposure = round(simulator.nav_by_chain.get(chain, 0.0), 6)
    accumulator.usd_at_risk += exposure
    accumulator.code_delta[operation.target or "proxy"] = str(implementation)
    accumulator.authority_delta["implementation"] = str(implementation)
    accumulator.notes.append(
        f"操作 {index}: 代理实现升级为 {implementation}，逻辑控制权变更（NAV≈{exposure:g} USD）"
    )
    return "control_grant"


def _apply_cross_chain(
    simulator: _Simulator, index: int, operation: Operation
) -> CapabilityDelta:
    legs = _cross_chain_legs(operation)
    if not legs:
        return _apply_transfer(simulator, index, operation)
    for leg_index, leg in enumerate(legs):
        chain = simulator.resolve(str(leg.get("chain") or operation.chain))
        state = simulator.world.chains[chain]
        token = str(leg.get("token") or operation.token or "native")
        amount = _as_float(leg.get("amount"), 0.0)
        usd = _as_float(leg.get("amount_usd"), None)
        if usd is None:
            usd = round(amount * token_price(state, token), 6)
        accumulator = simulator.accumulator(chain)
        accumulator.usd_at_risk += usd
        accumulator.balance_delta[token] = accumulator.balance_delta.get(token, 0.0) - amount
        accumulator.notes.append(
            f"操作 {index}: 跨链第 {leg_index + 1} 腿在 {chain} 转出 {amount:g} {token}（≈{usd:g} USD）"
        )
    return "value_transfer"


def _operation_usd(operation: Operation, state: Any, token: str) -> float:
    if operation.amount_usd is not None:
        return round(float(operation.amount_usd), 6)
    amount = float(operation.amount or 0.0)
    return round(amount * token_price(state, token), 6)


def _operation_nonce_span(operation: Operation) -> int:
    """nonce 跨度：优先用显式 nonce_span；仅当存在 nonce 时才用 batch_size 推断。

    注意 `batch_size > 1` 也可能只是「一笔交易含多个收款人」（批量提现），
    并不代表连续 nonce 批量签发，因此不能无条件计入 R-004。
    """
    span = int(operation.nonce_span or 0)
    if span <= 0 and operation.nonce is not None and int(operation.batch_size or 1) > 1:
        span = int(operation.batch_size or 1)
    if operation.kind == "eip7702_authorization" and span <= 0:
        span = 1
    return span


def irreversible_operation_changes_state(
    request: ReviewRequest, operation: Operation
) -> bool:
    """不可逆操作是否真的改变了权限状态。

    对「幂等重复签名 / 撤销委托 / renounce ownership / 同实现升级」返回 False：
    这类操作不新增能力，不应被 R-001/R-007 当成不可逆授权阻断。
    无法确认世界状态时保守返回 True（fail-closed）。
    """
    chain = resolve_chain(request.world, operation.chain)
    state = request.world.chains.get(chain) if chain else None
    if state is None:
        return True
    authority = state.authority
    if operation.kind == "eip7702_authorization":
        delegate = operation.delegate or operation.target
        if _is_zero_address(delegate):
            return False
        return not _same_address(authority.delegate, delegate)
    if operation.kind == "ownership_change":
        new_owner = operation.new_owner or operation.target
        if _is_zero_address(new_owner):
            return False
        return not _same_address(authority.owner, new_owner)
    if operation.kind == "proxy_upgrade":
        implementation = operation.implementation or (
            operation.metadata.get("implementation") if operation.metadata else None
        )
        if implementation is None:
            return True
        return not _same_address(authority.implementation, str(implementation))
    return True


def _normalize_address(address: str | None) -> str:
    return str(address or "").strip().lower()


def _is_zero_address(address: str | None) -> bool:
    normalized = _normalize_address(address)
    if not normalized:
        return False
    if normalized in {"0x0", "0x00"}:
        return True
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    return bool(normalized) and set(normalized) == {"0"}


def _same_address(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return _normalize_address(left) == _normalize_address(right)


def _max_capability(left: CapabilityDelta, right: CapabilityDelta) -> CapabilityDelta:
    if _CAPABILITY_RANK.get(right, 0) > _CAPABILITY_RANK.get(left, 0):
        return right
    return left


def _as_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["irreversible_operation_changes_state", "simulate_request"]
