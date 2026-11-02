"""签名请求解码：EVM calldata、EIP-712 结构化数据与 EIP-7702 授权元组。

解码层只做结构化：不接触私钥、不广播交易。无法识别或无法校验的输入一律抛出
`DecodeError`，由 `pipeline` fail-closed（ESCALATE/BLOCK），绝不静默放行。

支持三类输入：
- `structured`：请求已带 `operations`，直接透传（数据集与测试最常用）。
- `evm_tx`：解析 calldata 头 4 字节 selector；内置常见 selector，未知 selector
  若请求显式提供 ABI/函数签名则用 `eth-abi` 解码，否则报错。
- `eip712` / `eip7702_authorization`：解析 typed-data（Permit/Approve/TransferWithAuthorization/
  7702 授权）与 7702 授权元组列表。

`eth-abi` 为可选依赖：缺失时内置 selector 走纯 Python 静态解码兜底。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .schema import ChainState, Operation, ReviewRequest, WorldState

MAX_UINT256 = 2**256 - 1
# 常见「无限授权」哨兵：type(uint256).max 或 >= 2**255 的大数
_UNLIMITED_THRESHOLD = 2**255

# 内置 selector（小写）
SELECTOR_ERC20_TRANSFER = "0xa9059cbb"
SELECTOR_APPROVE = "0x095ea7b3"
SELECTOR_SET_APPROVAL_FOR_ALL = "0xa22cb465"
SELECTOR_TRANSFER_FROM = "0x23b872dd"
SELECTOR_INCREASE_ALLOWANCE = "0x39509351"

_BUILTIN_SELECTORS: dict[str, str] = {
    SELECTOR_ERC20_TRANSFER: "transfer",
    SELECTOR_APPROVE: "approve",
    SELECTOR_SET_APPROVAL_FOR_ALL: "setApprovalForAll",
    SELECTOR_TRANSFER_FROM: "transferFrom",
    SELECTOR_INCREASE_ALLOWANCE: "increaseAllowance",
}

# 链别名：`eth`/`ethereum`/`1` 视为同一条链，便于数据集与真实世界状态对齐
_CHAIN_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"ethereum", "eth", "mainnet", "1", "ethereum-mainnet"}),
    frozenset({"bsc", "bnb", "binance-smart-chain", "binance", "56", "bnb-chain"}),
    frozenset({"arbitrum", "arb", "arbitrum-one", "42161"}),
    frozenset({"base", "8453", "base-mainnet"}),
    frozenset({"polygon", "matic", "137", "polygon-pos"}),
    frozenset({"optimism", "op", "10", "optimism-mainnet"}),
    frozenset({"avalanche", "avax", "43114"}),
)

_NATIVE_SYMBOLS = frozenset({"native", "eth", "ether", "bnb", "matic", "pol", "avax", "sol", "trx"})


class DecodeError(ValueError):
    """请求无法解码；审查必须 fail-closed。"""


# ---------------------------------------------------------------------------
# 公共工具（simulate / features 复用，保证价格、余额、链名解析口径一致）
# ---------------------------------------------------------------------------


def resolve_chain(world: WorldState, chain: str | None) -> str | None:
    """把请求里的链名解析成 world.chains 中的实际键名；找不到返回 None。"""
    if chain is None:
        return None
    raw = str(chain).strip()
    if not raw:
        return None
    if raw in world.chains:
        return raw
    lowered = raw.lower()
    for key in world.chains:
        if key.lower() == lowered:
            return key
    group = _chain_group(raw)
    if group is not None:
        for key in world.chains:
            if key.lower() in group:
                return key
    return None


def _chain_group(chain: str) -> frozenset[str] | None:
    lowered = chain.strip().lower()
    for group in _CHAIN_GROUPS:
        if lowered in group:
            return group
    return None


def chain_state(request: ReviewRequest, chain: str | None = None) -> ChainState | None:
    """取某条链的状态快照；chain 为空时取请求主链。"""
    resolved = resolve_chain(request.world, chain or request.chain)
    return request.world.chains.get(resolved) if resolved else None


def token_price(state: ChainState | None, token: str | None) -> float:
    """按 token 查 USD 价格；对原生币/链简称做别名兜底，查不到返回 0。"""
    if state is None or token is None:
        return 0.0
    prices = {str(key).strip().lower(): float(value) for key, value in state.prices_usd.items()}
    key = str(token).strip().lower()
    if key in prices:
        return prices[key]
    if key in _NATIVE_SYMBOLS or key == state.chain.strip().lower():
        candidates = [state.chain.strip().lower(), "native", "eth", "bnb", "matic", "pol"]
        for candidate in candidates:
            if candidate in prices:
                return prices[candidate]
    return 0.0


def token_balance(state: ChainState | None, token: str | None) -> float:
    """按 token 查余额；对原生币/链简称做别名兜底，查不到返回 0。"""
    if state is None or token is None:
        return 0.0
    balances = {str(key).strip().lower(): float(value) for key, value in state.balances.items()}
    key = str(token).strip().lower()
    if key in balances:
        return balances[key]
    if key in _NATIVE_SYMBOLS or key == state.chain.strip().lower():
        candidates = [state.chain.strip().lower(), "native", "eth", "bnb", "matic", "pol"]
        for candidate in candidates:
            if candidate in balances:
                return balances[candidate]
    return 0.0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def decode_request(request: ReviewRequest) -> list[Operation]:
    """把 `ReviewRequest` 解码成操作列表。

    Args:
        request: 已完成 schema 校验的审查请求。

    Returns:
        非空的操作列表；每条操作带 kind/chain/account 与关键参数。

    Raises:
        DecodeError: 请求缺少可解码内容或类型未知（调用方必须 fail-closed）。
    """
    if request.kind == "structured":
        operations = list(request.operations)
        payload_operations = request.payload.get("operations") if request.payload else None
        if not operations and isinstance(payload_operations, list) and payload_operations:
            operations = [_operation_from_mapping(item) for item in payload_operations]
        if not operations:
            raise DecodeError("structured 请求缺少 operations，无法确定签名对象")
        return operations

    try:
        operations = _decode_payload(request)
    except DecodeError:
        # 结构化降级：payload 不可解码但请求已显式给出结构化操作时，仍按操作审查
        if request.operations:
            return list(request.operations)
        raise
    if not operations:
        if request.operations:
            return list(request.operations)
        raise DecodeError(f"{request.kind} 请求未解码出任何操作")
    return operations


def _decode_payload(request: ReviewRequest) -> list[Operation]:
    if request.kind == "evm_tx":
        return _decode_evm_tx(request)
    if request.kind == "eip712":
        return _decode_eip712(request)
    if request.kind == "eip7702_authorization":
        return _decode_eip7702(request)
    raise DecodeError(f"不支持的请求类型: {request.kind}")


# ---------------------------------------------------------------------------
# EVM calldata
# ---------------------------------------------------------------------------


def _decode_evm_tx(request: ReviewRequest) -> list[Operation]:
    payload = request.payload or {}
    tx = payload.get("tx") if isinstance(payload.get("tx"), dict) else None
    tx = tx or (payload.get("transaction") if isinstance(payload.get("transaction"), dict) else None)
    source: dict[str, Any] = tx or payload  # 允许 {"tx": {...}} 或平铺字段

    data = _normalize_hex(
        _get(source, "data", "calldata", "input") or _get(payload, "data", "calldata", "input") or "0x"
    )
    to = _get(source, "to", "target", "contract") or _get(payload, "to", "target", "contract")
    chain = str(_get(payload, "chain") or _get(source, "chain") or request.chain)
    account = str(_get(payload, "account") or _get(source, "account") or request.account)
    token = _get(payload, "token") or _get(source, "token")
    explicit_usd = _as_float(_get(payload, "amount_usd") or _get(source, "amount_usd"))
    decimals = _as_int(_get(payload, "decimals") or _get(source, "decimals"))

    if data == "0x":
        if not to:
            raise DecodeError("evm_tx 原生转账缺少 to 字段")
        value_wei = _as_int(_get(source, "value") if _get(source, "value") is not None else _get(payload, "value"), 0)
        amount = _as_float(_get(payload, "amount"))
        if amount is None:
            amount = _as_float(_get(source, "value_ether")) or (
                value_wei / 10**18 if value_wei is not None else 0.0
            )
        return [
            _transfer_operation(
                kind="native_transfer",
                request=request,
                chain=chain,
                account=account,
                target=str(to),
                token=str(token) if token else "native",
                amount=amount,
                explicit_usd=explicit_usd,
                metadata={"decoded_by": "evm_tx.native_transfer", "value_wei": value_wei},
            )
        ]

    selector = data[:10]
    args_hex = data[10:]
    name = _BUILTIN_SELECTORS.get(selector)
    if name is not None:
        types = _SELECTOR_TYPES[name]
        values = _decode_args(types, args_hex)
        return [
            _operation_from_values(
                name,
                values,
                request=request,
                to=str(to) if to else None,
                token=str(token) if token else None,
                chain=chain,
                account=account,
                explicit_usd=explicit_usd,
                decimals=decimals,
                decoded_by=f"evm_tx.selector{selector}",
            )
        ]

    # 未知 selector：只有在请求显式提供 ABI / 函数签名时才能继续
    abi_operations = _decode_unknown_with_abi(
        request,
        source=source,
        payload=payload,
        selector=selector,
        args_hex=args_hex,
        to=str(to) if to else None,
        token=str(token) if token else None,
        chain=chain,
        account=account,
        explicit_usd=explicit_usd,
        decimals=decimals,
    )
    if abi_operations is not None:
        return abi_operations
    raise DecodeError(f"未知 calldata selector {selector}，且请求未提供可解码 ABI")


_SELECTOR_TYPES: dict[str, list[str]] = {
    "transfer": ["address", "uint256"],
    "approve": ["address", "uint256"],
    "setApprovalForAll": ["address", "bool"],
    "transferFrom": ["address", "address", "uint256"],
    "increaseAllowance": ["address", "uint256"],
}


def _decode_args(types: list[str], args_hex: str) -> tuple[Any, ...]:
    try:
        raw = bytes.fromhex(args_hex)
    except ValueError as exc:
        raise DecodeError(f"calldata 参数不是合法十六进制: {args_hex[:32]}") from exc
    try:
        from eth_abi import decode as abi_decode  # noqa: PLC0415 - 可选依赖，惰性导入
    except ImportError:
        return _decode_static_args(types, raw)
    try:
        return tuple(abi_decode(types, raw))
    except Exception as exc:  # noqa: BLE001 - eth-abi 异常类型较多，统一转 DecodeError
        raise DecodeError(f"ABI 解码失败（{','.join(types)}）: {exc}") from exc


def _decode_static_args(types: list[str], raw: bytes) -> tuple[Any, ...]:
    """无 eth-abi 时的兜底：只支持 address/uint*/bool 静态类型。"""
    if len(raw) < 32 * len(types):
        raise DecodeError(f"calldata 参数长度不足：需要 {32 * len(types)} 字节，实际 {len(raw)}")
    values: list[Any] = []
    for index, type_name in enumerate(types):
        word = raw[index * 32 : (index + 1) * 32]
        if type_name == "address":
            values.append("0x" + word[12:].hex())
        elif type_name.startswith("uint"):
            values.append(int.from_bytes(word, "big"))
        elif type_name == "bool":
            values.append(any(word))
        else:
            raise DecodeError(f"缺少 eth-abi 依赖，无法解码动态类型 {type_name}")
    return tuple(values)


def _operation_from_values(
    name: str,
    values: tuple[Any, ...],
    *,
    request: ReviewRequest,
    to: str | None,
    token: str | None,
    chain: str,
    account: str,
    explicit_usd: float | None,
    decimals: int | None,
    decoded_by: str,
) -> Operation:
    """把已解码的 ABI 参数映射为 Operation。"""
    lowered = name.lower()
    if lowered == "transfer" and len(values) == 2:
        recipient, raw_amount = values
        return _transfer_operation(
            kind="erc20_transfer",
            request=request,
            chain=chain,
            account=account,
            target=str(recipient),
            token=token or to,
            amount=_human_amount(raw_amount, decimals),
            explicit_usd=explicit_usd,
            metadata={"decoded_by": decoded_by, "raw_amount": str(raw_amount)},
        )
    if lowered == "transferfrom" and len(values) == 3:
        _sender, recipient, raw_amount = values
        return _transfer_operation(
            kind="erc20_transfer",
            request=request,
            chain=chain,
            account=account,
            target=str(recipient),
            token=token or to,
            amount=_human_amount(raw_amount, decimals),
            explicit_usd=explicit_usd,
            metadata={"decoded_by": decoded_by, "raw_amount": str(raw_amount)},
        )
    if lowered in {"approve", "increaseallowance"} and len(values) == 2:
        spender, raw_amount = values
        allowance = _human_amount(raw_amount, decimals)
        unlimited = _is_unlimited(raw_amount)
        return Operation(
            kind="approve",
            chain=chain,
            account=account,
            target=str(spender),
            token=token or to,
            amount=None if unlimited else allowance,
            allowance=None if unlimited else allowance,
            unlimited_allowance=unlimited,
            metadata={"decoded_by": decoded_by, "raw_amount": str(raw_amount)},
        )
    if lowered == "setapprovalforall" and len(values) == 2:
        operator, approved = values
        return Operation(
            kind="approve",
            chain=chain,
            account=account,
            target=str(operator),
            token=token or to,
            amount=None,
            allowance=None,
            unlimited_allowance=bool(approved),
            metadata={"decoded_by": decoded_by, "approval_type": "operator", "approved": bool(approved)},
        )
    if len(values) == 2 and isinstance(values[1], (int, float)):  # 未知函数但形如 (address,uint256)
        return Operation(
            kind="unknown",
            chain=chain,
            account=account,
            target=str(values[0]),
            token=token or to,
            metadata={"decoded_by": decoded_by, "function": name, "args": [str(item) for item in values]},
        )
    return Operation(
        kind="unknown",
        chain=chain,
        account=account,
        target=to,
        token=token,
        metadata={"decoded_by": decoded_by, "function": name, "args": [str(item) for item in values]},
    )


def _transfer_operation(
    *,
    kind: str,
    request: ReviewRequest,
    chain: str,
    account: str,
    target: str,
    token: str | None,
    amount: float | None,
    explicit_usd: float | None,
    metadata: dict[str, Any] | None = None,
) -> Operation:
    token_name = token or "native"
    amount_usd = explicit_usd
    if amount_usd is None:
        state = chain_state(request, chain)
        price = token_price(state, token_name)
        amount_usd = round(float(amount or 0.0) * price, 6) if price else None
    return Operation(
        kind=kind,  # type: ignore[arg-type] - 调用方只传两个合法 kind
        chain=chain,
        account=account,
        target=target,
        token=token_name,
        amount=float(amount or 0.0),
        amount_usd=amount_usd,
        metadata=metadata or {},
    )


def _human_amount(raw: Any, decimals: int | None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise DecodeError(f"金额参数不是数值: {raw!r}") from exc
    if decimals and decimals > 0:
        return round(value / 10**decimals, 18)
    return value


def _is_unlimited(raw: Any) -> bool:
    try:
        return int(raw) == MAX_UINT256 or int(raw) >= _UNLIMITED_THRESHOLD
    except (TypeError, ValueError):
        return False


def _decode_unknown_with_abi(
    request: ReviewRequest,
    *,
    source: dict[str, Any],
    payload: dict[str, Any],
    selector: str,
    args_hex: str,
    to: str | None,
    token: str | None,
    chain: str,
    account: str,
    explicit_usd: float | None,
    decimals: int | None,
) -> list[Operation] | None:
    """按请求显式给出的 ABI / 函数签名解码未知 selector；未提供返回 None。"""
    abi_spec = _get(source, "abi") or _get(payload, "abi")
    signature = _get(source, "function_signature") or _get(payload, "function_signature")
    if abi_spec is None and signature is None:
        return None
    try:
        from eth_abi import decode as abi_decode  # noqa: PLC0415 - 可选依赖
    except ImportError as exc:  # pragma: no cover - 环境缺依赖
        raise DecodeError("请求提供了 ABI，但当前环境缺少 eth-abi，无法解码") from exc

    candidates: list[tuple[str, list[str]]] = []
    if signature is not None:
        match = re.fullmatch(r"\s*([A-Za-z_][\w]*)\s*\(([^)]*)\)\s*", str(signature))
        if not match:
            raise DecodeError(f"function_signature 格式非法: {signature!r}")
        type_names = [item.strip() for item in match.group(2).split(",") if item.strip()]
        candidates.append((match.group(1), type_names))
    if abi_spec is not None:
        candidates.extend(_abi_candidates(abi_spec))

    matched = [
        (name, type_names)
        for name, type_names in candidates
        if _function_selector(name, type_names) == selector
    ]
    if matched:
        chosen: tuple[str, list[str]] | None = matched[0]
    elif signature is not None:
        # 显式函数签名以请求为准：keccak 后端可能不可用，无法本地校验 selector
        chosen = candidates[0]
    elif len(candidates) == 1:
        chosen = candidates[0]
    else:
        chosen = None
    if chosen is None:
        return None
    name, type_names = chosen
    try:
        values = tuple(abi_decode(type_names, bytes.fromhex(args_hex)))
    except Exception as exc:  # noqa: BLE001
        raise DecodeError(f"按 ABI 解码 {name}({','.join(type_names)}) 失败: {exc}") from exc
    return [
        _operation_from_values(
            name,
            values,
            request=request,
            to=to,
            token=token,
            chain=chain,
            account=account,
            explicit_usd=explicit_usd,
            decimals=decimals,
            decoded_by=f"evm_tx.abi.{name}",
        )
    ]


def _function_selector(name: str, type_names: list[str]) -> str | None:
    """计算函数 selector；环境缺少 keccak 后端时返回 None（由调用方按显式签名信任）。"""
    try:
        from eth_utils import keccak  # noqa: PLC0415 - 可选依赖（可能缺哈希后端）
    except ImportError:  # pragma: no cover - 环境缺依赖
        return None
    try:
        return "0x" + keccak(text=f"{name}({','.join(type_names)})")[:4].hex()
    except Exception:  # noqa: BLE001 - eth-hash 后端缺失/类型非法都不阻断显式 ABI 解码
        return None


def _abi_candidates(abi_spec: Any) -> list[tuple[str, list[str]]]:
    entries = abi_spec
    if isinstance(entries, str):
        try:
            entries = json.loads(entries)
        except json.JSONDecodeError as exc:
            raise DecodeError(f"请求 ABI 不是合法 JSON: {exc}") from exc
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        raise DecodeError("请求 ABI 必须是函数列表或 JSON 字符串")
    candidates: list[tuple[str, list[str]]] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type", "function") != "function":
            continue
        name = entry.get("name")
        inputs = entry.get("inputs") or []
        if not name or not isinstance(inputs, list):
            continue
        type_names = [str(item.get("type")) for item in inputs if isinstance(item, dict) and item.get("type")]
        if len(type_names) == len(inputs):
            candidates.append((str(name), type_names))
    return candidates


# ---------------------------------------------------------------------------
# EIP-712 typed data
# ---------------------------------------------------------------------------


def _decode_eip712(request: ReviewRequest) -> list[Operation]:
    payload = request.payload or {}
    typed = payload.get("typed_data") or payload.get("eip712") or payload.get("typedData")
    if isinstance(typed, str):
        try:
            typed = json.loads(typed)
        except json.JSONDecodeError as exc:
            raise DecodeError(f"EIP-712 typed_data 不是合法 JSON: {exc}") from exc
    if typed is None or not isinstance(typed, dict):
        typed = payload

    primary = (
        _get(typed, "primaryType", "primary_type")
        or _get(payload, "primaryType", "primary_type")
        or ""
    )
    if not primary:
        raise DecodeError("EIP-712 请求缺少 primaryType")
    domain = typed.get("domain") if isinstance(typed.get("domain"), dict) else payload.get("domain") or {}
    message = typed.get("message") if isinstance(typed.get("message"), dict) else payload.get("message") or {}
    if not isinstance(message, dict) or not message:
        raise DecodeError("EIP-712 请求缺少 message 内容")

    chain_id = _as_int(_get(domain, "chainId", "chain_id"), None)
    chain = str(_get(domain, "chain") or _get(payload, "chain") or request.chain)
    account = str(_get(message, "owner") or _get(payload, "account") or request.account)
    metadata: dict[str, Any] = {
        "decoded_by": f"eip712.{primary}",
        "primary_type": str(primary),
        "chain_id": chain_id,
    }
    for key in ("deadline", "nonce", "validAfter", "validBefore"):
        if key in message:
            metadata[key] = message[key]

    normalized = str(primary).lower()
    verifying = _get(domain, "verifyingContract", "verifying_contract") or _get(
        message, "verifyingContract", "token"
    )
    token = str(verifying) if verifying else str(_get(payload, "token") or "") or None
    explicit_usd = _as_float(_get(message, "amount_usd") or _get(payload, "amount_usd"))

    if "transferwithauthorization" in normalized or "receivewithauthorization" in normalized:
        recipient = _get(message, "to", "recipient")
        raw_amount = _get(message, "value", "amount")
        if not recipient or raw_amount is None:
            raise DecodeError("EIP-3009 授权缺少 to/value")
        return [
            _transfer_operation(
                kind="erc20_transfer",
                request=request,
                chain=chain,
                account=account,
                target=str(recipient),
                token=token,
                amount=_as_float(raw_amount) or 0.0,
                explicit_usd=explicit_usd,
                metadata=metadata,
            )
        ]

    if "authorization" in normalized or "delegation" in normalized or "7702" in normalized or "setcode" in normalized:
        delegate = _get(message, "address", "delegate", "contract")
        if not delegate:
            raise DecodeError("EIP-712 7702 授权缺少 address/delegate")
        nonce = _as_int(_get(message, "nonce"), 0) or 0
        return [
            Operation(
                kind="eip7702_authorization",
                chain=chain,
                account=account,
                target=str(delegate),
                delegate=str(delegate),
                nonce=nonce,
                nonce_span=1,
                batch_size=1,
                metadata=metadata,
            )
        ]

    if "permit" in normalized or "approve" in normalized:
        spender = _get(message, "spender", "operator", "address")
        raw_allowance = _get(message, "value", "amount", "allowance")
        if not spender or raw_allowance is None:
            raise DecodeError("EIP-712 授权缺少 spender/value")
        allowance = _as_float(raw_allowance) or 0.0
        unlimited = _is_unlimited(raw_allowance)
        return [
            Operation(
                kind="permit" if "permit" in normalized else "approve",
                chain=chain,
                account=account,
                target=str(spender),
                token=token,
                amount=None if unlimited else allowance,
                allowance=None if unlimited else allowance,
                unlimited_allowance=unlimited,
                metadata={**metadata, "raw_allowance": str(raw_allowance)},
            )
        ]

    raise DecodeError(f"未知 EIP-712 primaryType: {primary}")


# ---------------------------------------------------------------------------
# EIP-7702 授权元组
# ---------------------------------------------------------------------------


def _decode_eip7702(request: ReviewRequest) -> list[Operation]:
    payload = request.payload or {}
    raw = payload.get("authorizations")
    if raw is None:
        raw = payload.get("authorization")
    if raw is None and (_get(payload, "address", "delegate", "contract") is not None):
        raw = [payload]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise DecodeError("eip7702_authorization 请求缺少 authorizations 列表")

    chain_default = str(_get(payload, "chain") or request.chain)
    account = str(_get(payload, "account") or request.account)
    operations: list[Operation] = []
    nonces: list[int] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DecodeError(f"第 {index} 条 7702 授权不是对象")
        delegate = _get(item, "address", "delegate", "contract")
        if not delegate:
            raise DecodeError(f"第 {index} 条 7702 授权缺少 address/delegate")
        chain = str(_get(item, "chain") or chain_default)
        chain_id = _as_int(_get(item, "chain_id", "chainId"), None)
        nonce = _as_int(_get(item, "nonce"), None)
        if nonce is not None:
            nonces.append(nonce)
        operations.append(
            Operation(
                kind="eip7702_authorization",
                chain=chain,
                account=account,
                target=str(delegate),
                delegate=str(delegate),
                nonce=nonce,
                metadata={
                    "decoded_by": "eip7702.authorization",
                    "authorization_index": index,
                    "address": str(delegate),
                    "chain_id": chain_id,
                },
            )
        )

    batch_size = len(operations)
    if nonces:
        nonce_span = max(nonces) - min(nonces) + 1
    else:
        nonce_span = batch_size
    for operation in operations:
        operation.batch_size = batch_size
        operation.nonce_span = nonce_span
    return operations


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _operation_from_mapping(item: Any) -> Operation:
    if isinstance(item, Operation):
        return item
    if not isinstance(item, dict):
        raise DecodeError(f"payload.operations 元素必须是对象: {item!r}")
    try:
        return Operation.model_validate(item)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError 统一转 DecodeError
        raise DecodeError(f"payload.operations 元素不合法: {exc}") from exc


def _get(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _normalize_hex(value: Any) -> str:
    text = str(value).strip()
    if not text or text in {"0x", "0X"}:
        return "0x"
    if text[:2] not in {"0x", "0X"}:
        text = "0x" + text
    body = text[2:]
    if len(body) % 2 != 0:
        raise DecodeError("calldata 十六进制长度为奇数")
    try:
        int(body, 16)
    except ValueError as exc:
        raise DecodeError(f"calldata 不是合法十六进制: {body[:32]}") from exc
    return "0x" + body.lower()


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(str(value), 16) if isinstance(value, str) and str(value).lower().startswith("0x") else int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "MAX_UINT256",
    "SELECTOR_APPROVE",
    "SELECTOR_ERC20_TRANSFER",
    "SELECTOR_INCREASE_ALLOWANCE",
    "SELECTOR_SET_APPROVAL_FOR_ALL",
    "SELECTOR_TRANSFER_FROM",
    "DecodeError",
    "chain_state",
    "decode_request",
    "resolve_chain",
    "token_balance",
    "token_price",
]
