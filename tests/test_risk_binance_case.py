"""币安 7702 回归用例 fixture 测试：关键特征、变体独立性与合成数据边界。"""

from __future__ import annotations

import copy

from bigmodel_lab.risk.dataset import (
    BINANCE_FIXTURE_PATH,
    binance_case_gaps,
    load_binance_fixture,
    load_binance_samples,
)
from bigmodel_lab.risk.schema import ApprovalTicket, RiskSample


def _instance_digest(sample: RiskSample) -> tuple[str, str, int, float]:
    operation = sample.request.operations[0]
    return (
        sample.request.account,
        operation.target or operation.delegate or "",
        int(operation.nonce_span),
        round(sample.request.world.nav_usd(), 2),
    )


def test_fixture_file_and_metadata():
    assert BINANCE_FIXTURE_PATH.is_file()
    fixture = load_binance_fixture()
    assert fixture["fixture_version"] == 1
    assert fixture["event_date"] == "2026-10-30"
    assert "公开" in fixture["source"]
    assert "不含真实" in fixture["disclaimer"]
    assert fixture["license"]
    assert len(fixture["key_features"]) >= 5
    raw_text = BINANCE_FIXTURE_PATH.read_text(encoding="utf-8")
    for forbidden in ("private_key", "privateKey", "mnemonic", "seed phrase"):
        assert forbidden not in raw_text
    assert len(fixture["variants"]) >= 3
    for item in [fixture["original"], *fixture["variants"]]:
        RiskSample.model_validate(item)


def test_original_case_has_all_required_features():
    samples = load_binance_samples()
    original = samples[0]
    request = original.request
    assert request.kind == "eip7702_authorization"
    operations = [operation for operation in request.operations if operation.kind == "eip7702_authorization"]
    assert operations
    operation = operations[0]
    assert operation.nonce_span == 50
    assert operation.batch_size == 50
    assert operation.target == operation.delegate
    assert request.context.approval is None
    origin = (request.context.origin or "").lower()
    assert any(marker in origin for marker in ("service", "svc", "container", "runner", "devops"))
    assert request.context.account_role == "hot_wallet"

    chain_state = request.world.chain(operation.chain)
    assert chain_state is not None
    assert chain_state.contract_age_days.get(operation.target) == 0
    assert 5.5e9 <= request.world.nav_usd() <= 7.0e9

    payload = request.payload
    assert payload["regression_gate"] is True
    assert payload["binance_case"] == "original"
    assert len(payload["authorizations"]) == 50
    assert original.label.expected_decision == "BLOCK"
    assert original.label.exposure_ratio == 1.0
    assert original.label.capability_delta == "full_control"
    assert original.label.risk_level == "critical"
    assert original.label.scenario_family == "eip7702_delegation"
    assert binance_case_gaps(original) == []


def test_variants_are_distinct_instances_and_differ_from_original():
    samples = load_binance_samples()
    original, variants = samples[0], samples[1:]
    assert len(variants) >= 3
    assert len({sample.sample_id for sample in samples}) == len(samples)
    assert len({sample.request.payload["instance_id"] for sample in samples}) == len(samples)

    original_digest = _instance_digest(original)
    variant_digests = [_instance_digest(variant) for variant in variants]
    for variant, digest in zip(variants, variant_digests, strict=True):
        assert digest != original_digest
        differences = [
            left != right for left, right in zip(original_digest, digest, strict=True)
        ]
        # 变体至少在地址、目标合约、nonce 数量、金额（NAV）之一与原始用例不同。
        assert any(differences)
        assert binance_case_gaps(variant) == []
        assert variant.label.expected_decision == "BLOCK"
        assert variant.label.exposure_ratio == 1.0
        assert variant.label.capability_delta == "full_control"
    assert len(set(variant_digests)) == len(variants)


def test_gap_checker_reports_specific_missing_features():
    original = load_binance_samples()[0]
    mutated = copy.deepcopy(original)
    mutated.request.operations[0].nonce_span = 1
    mutated.label.exposure_ratio = 0.5
    mutated.request.context.origin = "personal-laptop@home"
    mutated.request.payload["authorizations"] = []
    mutated.request.context.approval = ApprovalTicket(ticket_id="tkt-1")
    gaps = binance_case_gaps(mutated)
    text = "；".join(gaps)
    assert gaps
    assert "nonce" in text
    assert "100%" in text or "暴露" in text
    assert "服务账号" in text
    assert "authorizations" in text
    assert "审批凭证" in text
