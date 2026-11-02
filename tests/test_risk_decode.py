"""解码测试：内置 selector、可选 eth-abi、EIP-712、EIP-7702 批量授权与结构化降级。"""

from __future__ import annotations

import pytest

from bigmodel_lab.risk.decode import (
    MAX_UINT256,
    DecodeError,
    decode_request,
    resolve_chain,
    token_price,
)
from bigmodel_lab.risk.schema import (
    ChainState,
    Operation,
    RequestContext,
    ReviewRequest,
    WorldState,
)

TS = "2026-11-01T00:00:00Z"


def _world() -> WorldState:
    return WorldState(
        chains={
            "ethereum": ChainState(
                chain="ethereum",
                balances={"eth": 100.0, "usdc": 100_000.0},
                prices_usd={"eth": 3000.0, "usdc": 1.0},
            ),
            "bsc": ChainState(chain="bsc", balances={"bnb": 1000.0}, prices_usd={"bnb": 600.0}),
        }
    )


def _context() -> RequestContext:
    return RequestContext(origin="api", timestamp=TS, account_role="personal")


def _request(
    *,
    kind: str = "structured",
    payload: dict | None = None,
    operations: list[Operation] | None = None,
    chain: str = "ethereum",
    world: WorldState | None = None,
) -> ReviewRequest:
    return ReviewRequest(
        request_id="dec-1",
        timestamp=TS,
        account="0xacc",
        chain=chain,
        kind=kind,  # type: ignore[arg-type]
        payload=payload or {},
        operations=operations or [],
        context=_context(),
        world=world or _world(),
    )


def _address(tag: int) -> str:
    return "0x" + f"{tag:040x}"


def _word_address(address: str) -> bytes:
    return bytes(12) + bytes.fromhex(address[2:])


def _word_uint(value: int) -> bytes:
    return value.to_bytes(32, "big")


def _calldata(selector: str, *words: bytes) -> str:
    return selector + b"".join(words).hex()


# ---------------------------------------------------------------------------
# structured
# ---------------------------------------------------------------------------


def test_structured_operations_passthrough():
    operations = [
        Operation(kind="native_transfer", chain="ethereum", account="0xacc", target=_address(1), amount=1.0),
        Operation(kind="approve", chain="ethereum", account="0xacc", target=_address(2), token="usdc"),
    ]
    decoded = decode_request(_request(operations=operations))
    assert [item.kind for item in decoded] == ["native_transfer", "approve"]
    assert decoded[0].amount == 1.0


def test_structured_payload_operations_are_validated():
    request = _request(
        payload={
            "operations": [
                {"kind": "erc20_transfer", "chain": "ethereum", "account": "0xacc", "target": _address(3)}
            ]
        }
    )
    decoded = decode_request(request)
    assert len(decoded) == 1
    assert decoded[0].kind == "erc20_transfer"


def test_structured_without_operations_raises():
    with pytest.raises(DecodeError, match="缺少 operations"):
        decode_request(_request(payload={}))


# ---------------------------------------------------------------------------
# EVM calldata
# ---------------------------------------------------------------------------


def test_evm_tx_native_transfer():
    request = _request(
        kind="evm_tx",
        payload={"to": _address(9), "value": 10**18},
    )
    decoded = decode_request(request)
    assert len(decoded) == 1
    operation = decoded[0]
    assert operation.kind == "native_transfer"
    assert operation.target == _address(9)
    assert operation.amount == pytest.approx(1.0)
    assert operation.amount_usd == pytest.approx(3000.0)
    assert operation.metadata["value_wei"] == 10**18


def test_evm_tx_erc20_transfer_selector():
    recipient = _address(0xABC)
    data = _calldata("0xa9059cbb", _word_address(recipient), _word_uint(500))
    request = _request(
        kind="evm_tx",
        payload={"to": "0xusdc", "data": data, "token": "usdc"},
    )
    decoded = decode_request(request)
    operation = decoded[0]
    assert operation.kind == "erc20_transfer"
    assert operation.target.lower() == recipient.lower()
    assert operation.amount == pytest.approx(500.0)
    assert operation.amount_usd == pytest.approx(500.0)
    assert operation.metadata["decoded_by"] == "evm_tx.selector0xa9059cbb"


def test_evm_tx_approve_unlimited_and_limited():
    spender = _address(0xBEEF)
    unlimited = _calldata("0x095ea7b3", _word_address(spender), _word_uint(MAX_UINT256))
    decoded = decode_request(_request(kind="evm_tx", payload={"to": "0xusdc", "data": unlimited}))
    assert decoded[0].kind == "approve"
    assert decoded[0].unlimited_allowance is True
    assert decoded[0].allowance is None

    limited = _calldata("0x095ea7b3", _word_address(spender), _word_uint(1234))
    decoded = decode_request(_request(kind="evm_tx", payload={"to": "0xusdc", "data": limited}))
    assert decoded[0].unlimited_allowance is False
    assert decoded[0].allowance == pytest.approx(1234.0)


def test_evm_tx_set_approval_for_all():
    operator = _address(0xC0FFEE)
    data = _calldata("0xa22cb465", _word_address(operator), _word_uint(1))
    decoded = decode_request(_request(kind="evm_tx", payload={"to": "0xnft", "data": data}))
    operation = decoded[0]
    assert operation.kind == "approve"
    assert operation.unlimited_allowance is True
    assert operation.metadata["approval_type"] == "operator"
    assert operation.metadata["approved"] is True


def test_evm_tx_transfer_from_selector():
    sender = _address(0x1)
    recipient = _address(0x2)
    data = _calldata(
        "0x23b872dd", _word_address(sender), _word_address(recipient), _word_uint(77)
    )
    decoded = decode_request(_request(kind="evm_tx", payload={"to": "0xusdc", "data": data}))
    assert decoded[0].kind == "erc20_transfer"
    assert decoded[0].target.lower() == recipient.lower()
    assert decoded[0].amount == pytest.approx(77.0)


def test_evm_tx_unknown_selector_raises():
    request = _request(kind="evm_tx", payload={"to": _address(1), "data": "0xdeadbeef" + "00" * 32})
    with pytest.raises(DecodeError, match="未知 calldata selector"):
        decode_request(request)


def test_evm_tx_invalid_hex_raises():
    request = _request(kind="evm_tx", payload={"to": _address(1), "data": "0xzzzz"})
    with pytest.raises(DecodeError):
        decode_request(request)


def test_evm_tx_native_missing_to_raises():
    with pytest.raises(DecodeError, match="缺少 to"):
        decode_request(_request(kind="evm_tx", payload={"value": 1}))


def test_evm_tx_abi_decodes_known_function():
    """未知 selector + 显式 ABI：按 transfer 语义映射为 erc20_transfer。"""
    eth_abi = pytest.importorskip("eth_abi")
    recipient = _address(0x5151)
    signature = "transfer(address,uint256)"
    data = "0xdeadbeef" + eth_abi.encode(["address", "uint256"], [recipient, 321]).hex()
    request = _request(
        kind="evm_tx",
        payload={
            "to": "0xusdc",
            "data": data,
            "function_signature": signature,
            "token": "usdc",
        },
    )
    decoded = decode_request(request)
    assert decoded[0].kind == "erc20_transfer"
    assert decoded[0].target.lower() == recipient.lower()
    assert decoded[0].amount == pytest.approx(321.0)


def test_evm_tx_abi_unknown_function_becomes_unknown_operation():
    """显式 ABI 能解码但语义未知：保留参数并标记 unknown，由规则 fail-closed。"""
    eth_abi = pytest.importorskip("eth_abi")
    target = _address(0x7171)
    data = "0xdeadbeef" + eth_abi.encode(["address", "uint256"], [target, 9]).hex()
    request = _request(
        kind="evm_tx",
        payload={
            "to": _address(0x3),
            "data": data,
            "abi": [
                {
                    "type": "function",
                    "name": "execute",
                    "inputs": [
                        {"name": "to", "type": "address"},
                        {"name": "amount", "type": "uint256"},
                    ],
                }
            ],
        },
    )
    decoded = decode_request(request)
    assert decoded[0].kind == "unknown"
    assert decoded[0].metadata["function"] == "execute"


def test_evm_tx_broken_payload_falls_back_to_structured_operations():
    operation = Operation(kind="native_transfer", chain="ethereum", account="0xacc", target=_address(4))
    request = _request(kind="evm_tx", payload={"data": "0xdeadbeef" + "00" * 32}, operations=[operation])
    decoded = decode_request(request)
    assert [item.kind for item in decoded] == ["native_transfer"]


# ---------------------------------------------------------------------------
# EIP-712
# ---------------------------------------------------------------------------


def test_eip712_permit():
    request = _request(
        kind="eip712",
        payload={
            "primaryType": "Permit",
            "domain": {"chainId": 1, "verifyingContract": "0xusdc"},
            "message": {
                "owner": "0xacc",
                "spender": _address(0xBAD),
                "value": str(MAX_UINT256),
                "deadline": 1900000000,
            },
        },
    )
    decoded = decode_request(request)
    operation = decoded[0]
    assert operation.kind == "permit"
    assert operation.target.lower() == _address(0xBAD).lower()
    assert operation.token == "0xusdc"
    assert operation.unlimited_allowance is True
    assert operation.metadata["primary_type"] == "Permit"
    assert operation.metadata["chain_id"] == 1


def test_eip712_approve_named_primary_type():
    request = _request(
        kind="eip712",
        payload={
            "primary_type": "Approve",
            "domain": {"verifyingContract": "0xusdc"},
            "message": {"spender": _address(0x77), "value": 250},
        },
    )
    decoded = decode_request(request)
    assert decoded[0].kind == "approve"
    assert decoded[0].allowance == pytest.approx(250.0)
    assert decoded[0].unlimited_allowance is False


def test_eip712_transfer_with_authorization():
    request = _request(
        kind="eip712",
        payload={
            "primaryType": "TransferWithAuthorization",
            "domain": {"verifyingContract": "0xusdc"},
            "message": {
                "from": "0xacc",
                "to": _address(0x42),
                "value": 1234,
                "validAfter": 0,
                "validBefore": 1900000000,
                "nonce": "0x" + "11" * 32,
            },
        },
    )
    decoded = decode_request(request)
    assert decoded[0].kind == "erc20_transfer"
    assert decoded[0].target.lower() == _address(0x42).lower()
    assert decoded[0].amount == pytest.approx(1234.0)


def test_eip712_7702_authorization():
    request = _request(
        kind="eip712",
        payload={
            "primaryType": "Authorization",
            "message": {"address": _address(0xDEAD), "nonce": 5},
        },
    )
    decoded = decode_request(request)
    operation = decoded[0]
    assert operation.kind == "eip7702_authorization"
    assert operation.delegate.lower() == _address(0xDEAD).lower()
    assert operation.nonce == 5
    assert operation.batch_size == 1
    assert operation.nonce_span == 1


def test_eip712_unknown_primary_type_raises():
    request = _request(
        kind="eip712",
        payload={"primaryType": "Mystery", "message": {"foo": "bar"}},
    )
    with pytest.raises(DecodeError, match="未知 EIP-712 primaryType"):
        decode_request(request)


def test_eip712_missing_primary_type_raises():
    with pytest.raises(DecodeError, match="primaryType"):
        decode_request(_request(kind="eip712", payload={"message": {"a": 1}}))


# ---------------------------------------------------------------------------
# EIP-7702 授权元组
# ---------------------------------------------------------------------------


def test_eip7702_batch_authorizations():
    payload = {
        "authorizations": [
            {"chain": "ethereum", "address": _address(0xA1), "nonce": 10, "chain_id": 1},
            {"chain": "ethereum", "address": _address(0xA2), "nonce": 11, "chain_id": 1},
            {"chain": "bsc", "address": _address(0xA3), "nonce": 12, "chain_id": 56},
        ]
    }
    decoded = decode_request(_request(kind="eip7702_authorization", payload=payload))
    assert len(decoded) == 3
    assert {operation.chain for operation in decoded} == {"ethereum", "bsc"}
    assert all(operation.batch_size == 3 for operation in decoded)
    assert all(operation.nonce_span == 3 for operation in decoded)
    assert [operation.nonce for operation in decoded] == [10, 11, 12]
    assert decoded[2].metadata["chain_id"] == 56
    assert decoded[0].delegate.lower() == _address(0xA1).lower()


def test_eip7702_single_authorization_wrapper():
    payload = {"authorization": {"address": _address(0xB2), "nonce": 0}}
    decoded = decode_request(_request(kind="eip7702_authorization", payload=payload))
    assert len(decoded) == 1
    assert decoded[0].batch_size == 1
    assert decoded[0].nonce_span == 1
    assert decoded[0].nonce == 0


def test_eip7702_missing_address_raises():
    payload = {"authorizations": [{"nonce": 1}]}
    with pytest.raises(DecodeError, match="address"):
        decode_request(_request(kind="eip7702_authorization", payload=payload))


def test_eip7702_empty_list_raises():
    with pytest.raises(DecodeError):
        decode_request(_request(kind="eip7702_authorization", payload={"authorizations": []}))


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------


def test_resolve_chain_aliases():
    world = _world()
    assert resolve_chain(world, "ethereum") == "ethereum"
    assert resolve_chain(world, "ETH") == "ethereum"
    assert resolve_chain(world, "1") == "ethereum"
    assert resolve_chain(world, "mainnet") == "ethereum"
    assert resolve_chain(world, "bnb") == "bsc"
    assert resolve_chain(world, "solana") is None


def test_token_price_native_alias():
    world = _world()
    chain = world.chains["ethereum"]
    assert token_price(chain, "eth") == pytest.approx(3000.0)
    assert token_price(chain, "native") == pytest.approx(3000.0)
    assert token_price(chain, "usdc") == pytest.approx(1.0)
    assert token_price(chain, "unknown") == 0.0
