"""模拟测试：各类操作的资产/授权/权限/代码差分与暴露计算。"""

from __future__ import annotations

import pytest

from bigmodel_lab.risk.schema import (
    ChainState,
    Operation,
    RequestContext,
    ReviewRequest,
    WorldState,
)
from bigmodel_lab.risk.simulate import simulate_request

TS = "2026-11-01T00:00:00Z"


def _world() -> WorldState:
    return WorldState(
        chains={
            "ethereum": ChainState(
                chain="ethereum",
                balances={"eth": 100.0, "usdc": 100_000.0},
                prices_usd={"eth": 3000.0, "usdc": 1.0},
                contract_age_days={"0xknown": 30},
                nonce=7,
            ),
            "bsc": ChainState(chain="bsc", balances={"bnb": 1000.0}, prices_usd={"bnb": 600.0}),
        }
    )


# ethereum NAV = 100*3000 + 100000 = 400000；bsc NAV = 600000；全部 = 1,000,000


def _request(
    operations: list[Operation],
    *,
    world: WorldState | None = None,
    chain: str = "ethereum",
) -> ReviewRequest:
    return ReviewRequest(
        request_id="sim-1",
        timestamp=TS,
        account="0xacc",
        chain=chain,
        kind="structured",
        operations=operations,
        context=RequestContext(origin="api", timestamp=TS, account_role="personal"),
        world=world or _world(),
    )


def _op(kind: str, **kwargs) -> Operation:
    data = {"kind": kind, "chain": "ethereum", "account": "0xacc"}
    data.update(kwargs)
    return Operation(**data)


def test_native_transfer_exposure_and_balance_delta():
    request = _request([_op("native_transfer", target="0xknown", token="eth", amount=1.0)])
    result = simulate_request(request, request.operations)
    assert result.failed is False
    assert result.capability_delta == "value_transfer"
    assert result.exposure_usd == pytest.approx(3000.0)
    assert result.exposure_ratio == pytest.approx(3000.0 / 400000.0, abs=1e-6)
    assert result.nav_usd == pytest.approx(400000.0)
    assert result.worst_loss_usd == pytest.approx(3000.0)
    delta = result.per_chain[0]
    assert delta.chain == "ethereum"
    assert delta.balance_delta["eth"] == pytest.approx(-1.0)
    assert delta.usd_at_risk == pytest.approx(3000.0)


def test_erc20_transfer_uses_explicit_amount_usd():
    operation = _op(
        "erc20_transfer", target="0xknown", token="usdc", amount=20_000.0, amount_usd=20_000.0
    )
    result = simulate_request(_request([operation]), [operation])
    assert result.exposure_usd == pytest.approx(20_000.0)
    assert result.per_chain[0].balance_delta["usdc"] == pytest.approx(-20_000.0)


def test_batch_withdrawal_sums_all_legs():
    operation = _op(
        "batch_withdrawal",
        chain="ethereum",
        metadata={
            "items": [
                {"chain": "ethereum", "token": "eth", "amount": 1.0},
                {"chain": "ethereum", "token": "usdc", "amount": 5000.0},
            ]
        },
    )
    result = simulate_request(_request([operation]), [operation])
    assert result.failed is False
    assert result.exposure_usd == pytest.approx(8000.0)
    delta = result.per_chain[0]
    assert delta.balance_delta["eth"] == pytest.approx(-1.0)
    assert delta.balance_delta["usdc"] == pytest.approx(-5000.0)
    assert delta.usd_at_risk == pytest.approx(8000.0)


def test_batch_withdrawal_without_items_uses_amount():
    operation = _op("batch_withdrawal", token="usdc", amount=400.0)
    result = simulate_request(_request([operation]), [operation])
    assert result.exposure_usd == pytest.approx(400.0)


def test_approve_exposure_is_min_allowance_balance():
    operation = _op("approve", target="0xspender", token="usdc", allowance=250_000.0)
    result = simulate_request(_request([operation]), [operation])
    assert result.failed is False
    assert result.capability_delta == "allowance_grant"
    assert result.exposure_usd == pytest.approx(100_000.0)  # min(250000, 100000)
    allowance = result.per_chain[0].allowance_delta[0]
    assert allowance.token == "usdc"
    assert allowance.spender == "0xspender"
    assert allowance.unlimited is False
    assert allowance.amount == pytest.approx(250_000.0)
    assert result.per_chain[0].balance_delta == {}


def test_approve_unlimited_exposes_full_balance():
    operation = _op(
        "approve", target="0xspender", token="usdc", unlimited_allowance=True
    )
    result = simulate_request(_request([operation]), [operation])
    assert result.capability_delta == "allowance_grant"
    assert result.exposure_usd == pytest.approx(100_000.0)
    assert result.per_chain[0].allowance_delta[0].unlimited is True
    assert result.per_chain[0].allowance_delta[0].amount is None


def test_permit_behaves_like_allowance_grant():
    operation = _op("permit", target="0xspender", token="eth", amount=10.0)
    result = simulate_request(_request([operation]), [operation])
    assert result.capability_delta == "allowance_grant"
    assert result.exposure_usd == pytest.approx(30_000.0)


def test_eip7702_exposes_full_nav_and_full_control():
    operation = _op(
        "eip7702_authorization", target="0xdelegate", delegate="0xdelegate", nonce=0
    )
    result = simulate_request(_request([operation]), [operation])
    assert result.failed is False
    assert result.capability_delta == "full_control"
    assert result.exposure_usd == pytest.approx(400000.0)
    assert result.exposure_ratio == pytest.approx(1.0)
    assert result.worst_loss_usd == pytest.approx(400000.0)
    delta = result.per_chain[0]
    assert delta.authority_delta["delegate"] == "0xdelegate"
    assert delta.usd_at_risk == pytest.approx(400000.0)


def test_ownership_change_exposes_nav_and_control_grant():
    operation = _op("ownership_change", target="0xproxy", new_owner="0xattacker")
    result = simulate_request(_request([operation]), [operation])
    assert result.capability_delta == "control_grant"
    assert result.exposure_usd == pytest.approx(400000.0)
    assert result.exposure_ratio == pytest.approx(1.0)
    assert result.per_chain[0].authority_delta["owner"] == "0xattacker"


def test_proxy_upgrade_exposes_nav_and_code_delta():
    operation = _op(
        "proxy_upgrade", target="0xproxy", implementation="0xnewimpl"
    )
    result = simulate_request(_request([operation]), [operation])
    assert result.capability_delta == "control_grant"
    assert result.exposure_usd == pytest.approx(400000.0)
    delta = result.per_chain[0]
    assert delta.code_delta["0xproxy"] == "0xnewimpl"
    assert delta.authority_delta["implementation"] == "0xnewimpl"


def test_cross_chain_combo_aggregates_related_chains():
    operations = [
        _op("native_transfer", chain="ethereum", target="0xknown", token="eth", amount=1.0),
        _op("erc20_transfer", chain="bsc", target="0xknown", token="bnb", amount=10.0, amount_usd=6000.0),
    ]
    result = simulate_request(_request(operations), operations)
    assert result.failed is False
    assert result.nav_usd == pytest.approx(1_000_000.0)
    assert result.exposure_usd == pytest.approx(9000.0)
    assert result.exposure_ratio == pytest.approx(0.009)
    assert [delta.chain for delta in result.per_chain] == ["bsc", "ethereum"]


def test_cross_chain_transfer_legs():
    operation = _op(
        "cross_chain_transfer",
        chain="ethereum",
        metadata={
            "legs": [
                {"chain": "ethereum", "token": "eth", "amount": 1.0},
                {"chain": "bsc", "token": "bnb", "amount": 5.0},
            ]
        },
    )
    result = simulate_request(_request([operation]), [operation])
    assert result.exposure_usd == pytest.approx(3000.0 + 3000.0)
    assert result.nav_usd == pytest.approx(1_000_000.0)
    assert {delta.chain for delta in result.per_chain} == {"ethereum", "bsc"}


def test_chain_alias_resolves_to_world_key():
    operation = _op("native_transfer", chain="eth", target="0xknown", token="eth", amount=1.0)
    result = simulate_request(_request([operation]), [operation])
    assert result.failed is False
    assert result.per_chain[0].chain == "ethereum"


def test_missing_chain_state_fails_closed():
    operation = _op("native_transfer", chain="solana", target="0xknown", token="sol", amount=1.0)
    result = simulate_request(_request([operation]), [operation])
    assert result.failed is True
    assert "solana" in (result.failure_reason or "")
    assert result.exposure_ratio == 0.0


def test_no_operations_fails_closed():
    request = _request([])
    result = simulate_request(request, [])
    assert result.failed is True
    assert "没有可模拟" in (result.failure_reason or "") or "不包含" in (result.failure_reason or "")


def test_unknown_operation_is_not_failed_but_has_no_capability():
    operation = _op("unknown", target="0xcontract")
    result = simulate_request(_request([operation]), [operation])
    assert result.failed is False
    assert result.capability_delta == "none"
    assert result.exposure_usd == 0.0
    assert "未知类型" in "；".join(result.per_chain[0].notes)


def test_simulation_is_deterministic():
    operations = [
        _op("native_transfer", target="0xknown", token="eth", amount=2.5),
        _op("approve", target="0xspender", token="usdc", allowance=500.0),
    ]
    request = _request(operations)
    first = simulate_request(request, operations).model_dump()
    second = simulate_request(request, operations).model_dump()
    assert first == second


# ---------------------------------------------------------------------------
# 状态差分语义：幂等/撤销操作不新增能力，batch_size 不等于 nonce 跨度
# ---------------------------------------------------------------------------


def test_eip7702_revoke_existing_delegation_has_no_new_exposure():
    world = _world()
    world.chains["ethereum"].authority.delegate = "0x" + "dd" * 20
    operation = _op(
        "eip7702_authorization",
        target="0x" + "00" * 20,
        delegate="0x" + "00" * 20,
        nonce=1300,
    )
    request = _request([operation], world=world)
    result = simulate_request(request, [operation])
    assert result.failed is False
    assert result.capability_delta == "none"
    assert result.exposure_usd == 0.0
    assert result.per_chain[0].authority_delta["delegate"] is None
    assert "撤销" in "；".join(result.per_chain[0].notes)


def test_eip7702_same_delegate_is_idempotent():
    world = _world()
    delegate = "0x" + "dd" * 20
    world.chains["ethereum"].authority.delegate = delegate
    operation = _op("eip7702_authorization", target=delegate, delegate=delegate, nonce=1)
    request = _request([operation], world=world)
    result = simulate_request(request, [operation])
    assert result.capability_delta == "none"
    assert result.exposure_usd == 0.0
    assert "幂等" in "；".join(result.per_chain[0].notes)


def test_ownership_change_to_same_owner_is_idempotent():
    owner = "0x" + "ee" * 20
    world = _world()
    world.chains["ethereum"].authority.owner = owner
    operation = _op("ownership_change", target=owner, new_owner=owner)
    request = _request([operation], world=world)
    result = simulate_request(request, [operation])
    assert result.capability_delta == "none"
    assert result.exposure_usd == 0.0


def test_proxy_upgrade_to_same_implementation_is_idempotent():
    implementation = "0x" + "ff" * 20
    world = _world()
    world.chains["ethereum"].authority.implementation = implementation
    operation = _op("proxy_upgrade", target="0xproxy", implementation=implementation)
    request = _request([operation], world=world)
    result = simulate_request(request, [operation])
    assert result.capability_delta == "none"
    assert result.exposure_usd == 0.0


def test_batch_withdrawal_batch_size_alone_is_not_nonce_span():
    operation = _op(
        "batch_withdrawal",
        target="0x" + "11" * 20,
        amount_usd=9000.0,
        batch_size=3,
    )
    request = _request([operation])
    result = simulate_request(request, [operation])
    assert result.exposure_usd == pytest.approx(9000.0)
    assert result.per_chain[0].nonce_span == 0


def test_explicit_nonce_span_is_recorded_per_chain():
    operation = _op(
        "eip7702_authorization",
        target="0xbad",
        delegate="0xbad",
        nonce=1300,
        nonce_span=50,
        batch_size=50,
    )
    request = _request([operation])
    result = simulate_request(request, [operation])
    assert result.per_chain[0].nonce_span == 50
