"""审查样本 schema：合法/非法/未知版本与决策工具测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bigmodel_lab.risk.schema import (
    SCENARIO_FAMILIES,
    SCHEMA_VERSION,
    ChainState,
    Label,
    RequestContext,
    ReviewRequest,
    RiskSample,
    WorldState,
    canonical_json,
    exit_code_for,
    max_decision,
)

BASE_TIMESTAMP = "2026-10-30T00:00:00Z"


def _request(**overrides: object) -> ReviewRequest:
    payload: dict[str, object] = {
        "request_id": "req-1",
        "timestamp": BASE_TIMESTAMP,
        "account": "0xabc",
        "chain": "ethereum",
        "context": RequestContext(origin="ops-console@internal", timestamp=BASE_TIMESTAMP),
        "world": WorldState(),
    }
    payload.update(overrides)
    return ReviewRequest(**payload)


def _label(**overrides: object) -> Label:
    payload: dict[str, object] = {
        "action_type": "native_transfer",
        "capability_delta": "value_transfer",
        "exposure_ratio": 0.0,
        "risk_level": "low",
        "expected_decision": "ALLOW",
        "scenario_family": "native_transfer",
    }
    payload.update(overrides)
    return Label(**payload)


def _sample(**overrides: object) -> RiskSample:
    payload: dict[str, object] = {"sample_id": "sample-1", "request": _request(), "label": _label()}
    payload.update(overrides)
    return RiskSample(**payload)


def test_valid_sample_roundtrip():
    sample = _sample()
    assert sample.schema_version == SCHEMA_VERSION
    restored = RiskSample.model_validate_json(sample.model_dump_json())
    assert restored == sample
    assert restored.request.context.origin == "ops-console@internal"


def test_scenario_families_cover_nine_categories():
    assert len(SCENARIO_FAMILIES) == 9
    assert set(SCENARIO_FAMILIES) == {
        "normal_ops",
        "native_transfer",
        "batch_withdrawal",
        "approve_permit",
        "eip7702_delegation",
        "ownership_change",
        "proxy_upgrade",
        "batch_nonce",
        "cross_chain_combo",
    }


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError) as exc_info:
        RiskSample(sample_id="s1", request=_request())  # type: ignore[call-arg]
    assert "label" in str(exc_info.value)


def test_invalid_exposure_ratio_rejected():
    with pytest.raises(ValidationError, match="exposure_ratio"):
        _label(exposure_ratio=-0.1)


def test_invalid_enum_value_rejected():
    with pytest.raises(ValidationError):
        _label(risk_level="extreme")
    with pytest.raises(ValidationError):
        _label(expected_decision="REJECT")
    with pytest.raises(ValidationError):
        _request(kind="bogus")


def test_unknown_schema_version_rejected():
    with pytest.raises(ValidationError, match="不支持的 schema_version"):
        _sample(schema_version=2)
    with pytest.raises(ValidationError, match="不支持的 schema_version"):
        _request(schema_version=99)


def test_extra_field_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        _sample(unexpected="x")


def test_world_nav_and_token_usd():
    world = WorldState(
        chains={
            "ethereum": ChainState(
                chain="ethereum",
                balances={"ETH": 2.0, "USDT": 100.0},
                prices_usd={"ETH": 2500.0, "USDT": 1.0},
            ),
            "bsc": ChainState(chain="bsc", balances={"BNB": 1.0}, prices_usd={"BNB": 600.0}),
        }
    )
    assert world.nav_usd("ethereum") == 5100.0
    assert world.nav_usd() == 5700.0
    assert world.chain("ethereum").token_usd("ETH") == 5000.0
    assert world.chain("missing") is None


def test_decision_helpers():
    assert exit_code_for("ALLOW") == 0
    assert exit_code_for("STEP_UP") == 0
    assert exit_code_for("BLOCK") == 2
    assert exit_code_for("ESCALATE") == 3
    assert exit_code_for("UNKNOWN") == 3
    assert max_decision("ALLOW", "STEP_UP", "BLOCK") == "BLOCK"
    assert max_decision("ALLOW", None) == "ALLOW"


def test_canonical_json_is_stable_and_unescaped():
    assert canonical_json({"b": 1, "a": [2]}) == '{"a":[2],"b":1}'
    assert canonical_json({"名称": "测试"}) == '{"名称":"测试"}'
