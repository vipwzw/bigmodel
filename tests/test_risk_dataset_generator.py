"""确定性场景生成器测试：9 类覆盖、正负样本、逐字节复现与标签自洽。"""

from __future__ import annotations

import pytest

from bigmodel_lab.risk.dataset import (
    BASE_TIMESTAMP_ISO,
    dataset_stats,
    generate_dataset,
    scenario_coverage,
)
from bigmodel_lab.risk.schema import DECISION_RANK, RISK_LEVEL_RANK, SCENARIO_FAMILIES

_CONTROL_CAPABILITIES = {"control_grant", "full_control"}


def _jsonl(samples) -> str:  # noqa: ANN001 - 测试辅助
    return "".join(sample.model_dump_json() + "\n" for sample in samples)


def test_covers_all_nine_families_with_positive_and_negative():
    samples = generate_dataset(seed=42, samples_per_family=6)
    assert len(samples) == 54
    coverage = scenario_coverage(samples)
    assert coverage["complete"] is True
    assert coverage["missing"] == []
    assert set(coverage["families"]) == set(SCENARIO_FAMILIES)
    for family, counts in coverage["families"].items():
        assert counts == {"total": 6, "positive": 3, "negative": 3}, family
    assert len({sample.sample_id for sample in samples}) == len(samples)


def test_same_seed_is_byte_identical():
    first = _jsonl(generate_dataset(seed=42, samples_per_family=6))
    second = _jsonl(generate_dataset(seed=42, samples_per_family=6))
    assert first == second
    assert _jsonl(generate_dataset(seed=7, samples_per_family=6)) != first


def test_timestamps_derive_from_fixed_base():
    samples = generate_dataset(seed=42, samples_per_family=2)
    assert all(sample.request.timestamp.startswith("2026-10-30T") for sample in samples)
    assert all(sample.request.context.timestamp == sample.request.timestamp for sample in samples)
    assert BASE_TIMESTAMP_ISO == "2026-10-30T00:00:00Z"


def test_labels_and_requests_are_self_consistent():
    for sample in generate_dataset(seed=42, samples_per_family=6):
        request, label = sample.request, sample.label
        assert request.operations, sample.sample_id
        assert label.scenario_family in SCENARIO_FAMILIES
        assert request.payload["scenario_family"] == label.scenario_family
        assert request.payload["instance_id"] == sample.sample_id
        assert request.payload["positive"] is (label.expected_decision != "ALLOW")
        assert request.world.nav_usd() > 0
        for operation in request.operations:
            assert operation.chain in request.world.chains
            assert operation.account == request.account
        if label.capability_delta not in _CONTROL_CAPABILITIES:
            exposure_usd = sum(float(operation.amount_usd or 0.0) for operation in request.operations)
            expected_ratio = min(round(exposure_usd / request.world.nav_usd(), 6), 1.0)
            assert label.exposure_ratio == pytest.approx(expected_ratio, abs=1e-6)
            assert label.exposure_ratio >= 0.0
        assert label.evidence
        if label.expected_decision == "ALLOW":
            assert label.risk_level == "low"
            assert label.exposure_ratio <= 0.01 + 1e-9
        else:
            assert label.risk_factors
            assert DECISION_RANK[label.expected_decision] > DECISION_RANK["ALLOW"]
        if label.expected_decision == "BLOCK":
            assert RISK_LEVEL_RANK[label.risk_level] >= RISK_LEVEL_RANK["high"]


def test_eip7702_positive_samples_are_full_exposure_block():
    positives = [
        sample
        for sample in generate_dataset(seed=42, samples_per_family=6)
        if sample.label.scenario_family == "eip7702_delegation"
        and sample.label.expected_decision != "ALLOW"
    ]
    assert len(positives) == 3
    for sample in positives:
        assert sample.label.exposure_ratio == 1.0
        assert sample.label.expected_decision == "BLOCK"
        assert sample.label.capability_delta == "full_control"
        assert any(operation.kind == "eip7702_authorization" for operation in sample.request.operations)


def test_batch_nonce_family_covers_large_nonce_span():
    positives = [
        sample
        for sample in generate_dataset(seed=42, samples_per_family=6)
        if sample.label.scenario_family == "batch_nonce" and sample.label.expected_decision != "ALLOW"
    ]
    spans = {operation.nonce_span for sample in positives for operation in sample.request.operations}
    assert 50 in spans


def test_seed_changes_sample_content():
    first = generate_dataset(seed=42, samples_per_family=4)
    second = generate_dataset(seed=43, samples_per_family=4)
    assert [sample.sample_id for sample in first] != [sample.sample_id for sample in second]
    first_amounts = [
        operation.amount_usd for sample in first for operation in sample.request.operations if operation.amount_usd
    ]
    second_amounts = [
        operation.amount_usd
        for sample in second
        for operation in sample.request.operations
        if operation.amount_usd
    ]
    assert first_amounts != second_amounts


def test_family_subset_generation_reports_missing_coverage():
    samples = generate_dataset(seed=1, samples_per_family=2, families=["normal_ops"])
    assert len(samples) == 2
    coverage = scenario_coverage(samples)
    assert coverage["complete"] is False
    assert "native_transfer" in coverage["missing"]
    assert any(sample.label.expected_decision != "ALLOW" for sample in samples)


def test_invalid_generator_arguments_rejected():
    with pytest.raises(ValueError, match="samples_per_family"):
        generate_dataset(seed=1, samples_per_family=1)
    with pytest.raises(ValueError, match="未知场景族"):
        generate_dataset(seed=1, samples_per_family=2, families=["not_a_family"])
    with pytest.raises(ValueError, match="未知场景族"):
        generate_dataset(seed=1, samples_per_family=2, families=["normal_ops", "bogus"])


def test_stats_summarizes_dataset():
    samples = generate_dataset(seed=42, samples_per_family=4)
    stats = dataset_stats(samples, seed=42, samples_per_family=4)
    assert stats["total"] == 36
    assert stats["synthetic_total"] == 36
    assert stats["binance_regression_total"] == 0
    assert sum(item["total"] for item in stats["families"].values()) == 36
    assert sum(stats["labels"]["expected_decision"].values()) == 36
    assert stats["coverage"]["complete"] is True
    assert stats["scenario_instances"] == 36
