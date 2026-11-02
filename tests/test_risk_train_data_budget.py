"""训练样本预算测试：sanity/standard 上限、按场景族分层与统计可复现性。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_risk_model_data import make_samples

from bigmodel_lab.risk.config import load_risk_config
from bigmodel_lab.risk.model_data import (
    build_sft_records,
    stratified_sample,
    write_training_data,
)
from bigmodel_lab.risk.schema import SCENARIO_FAMILIES


def test_profile_sample_budgets_from_config() -> None:
    cfg = load_risk_config()
    train = cfg.settings.model.train
    assert train.sanity.max_train_samples == 160
    assert train.sanity.max_val_samples == 32
    assert train.standard.max_train_samples == 2000
    assert train.standard.max_val_samples == 200
    # sanity 档必须明显小于 standard 档（保证 ≤10 分钟可跑）
    assert train.sanity.max_train_samples < train.standard.max_train_samples
    assert train.sanity.max_steps < train.standard.max_steps


def test_stratified_sample_covers_all_families_and_is_deterministic() -> None:
    samples = make_samples(len(SCENARIO_FAMILIES) * 4)
    selected = stratified_sample(samples, len(SCENARIO_FAMILIES), seed=42)
    assert len(selected) == len(SCENARIO_FAMILIES)
    assert {sample.label.scenario_family for sample in selected} == set(SCENARIO_FAMILIES)
    again = stratified_sample(samples, len(SCENARIO_FAMILIES), seed=42)
    assert [sample.sample_id for sample in selected] == [sample.sample_id for sample in again]
    # 不超上限 / 上限大于样本数时原样返回
    assert len(stratified_sample(samples, 7, seed=42)) == 7
    assert stratified_sample(samples, None) == samples
    assert stratified_sample(samples, 0) == []


def test_sanity_budget_caps_training_data(tmp_path: Path) -> None:
    cfg = load_risk_config()
    sanity = cfg.settings.model.train.sanity
    samples = make_samples(240)
    stats = write_training_data(
        samples,
        tmp_path / "sanity-data",
        max_train=sanity.max_train_samples,
        max_val=sanity.max_val_samples,
        seed=cfg.settings.dataset.seed,
    )
    assert stats["train_samples"] <= sanity.max_train_samples
    assert stats["val_samples"] <= sanity.max_val_samples
    assert stats["train_samples"] + stats["val_samples"] <= 240
    # 分层抽样：训练集应覆盖 ≥ 一半场景族（240 样本足够覆盖全部）
    assert set(stats["train_families"]) >= set(SCENARIO_FAMILIES[:5])
    assert sum(stats["train_families"].values()) == stats["train_samples"]
    assert sum(stats["train_labels"].values()) == stats["train_samples"]
    assert sum(stats["val_families"].values()) == stats["val_samples"]


def test_standard_budget_uses_more_samples_than_sanity(tmp_path: Path) -> None:
    cfg = load_risk_config()
    train = cfg.settings.model.train
    samples = make_samples(240)
    sanity_stats = write_training_data(
        samples,
        tmp_path / "sanity",
        max_train=train.sanity.max_train_samples,
        max_val=train.sanity.max_val_samples,
    )
    standard_stats = write_training_data(
        samples,
        tmp_path / "standard",
        max_train=train.standard.max_train_samples,
        max_val=train.standard.max_val_samples,
    )
    assert standard_stats["train_samples"] > sanity_stats["train_samples"]
    assert standard_stats["train_samples"] <= train.standard.max_train_samples
    assert standard_stats["val_samples"] <= train.standard.max_val_samples
    assert standard_stats["train_samples"] + standard_stats["val_samples"] <= 240


def test_budget_stats_are_reproducible(tmp_path: Path) -> None:
    samples = make_samples(90)
    first = write_training_data(samples, tmp_path / "a", max_train=40, max_val=8, seed=42)
    second = write_training_data(samples, tmp_path / "b", max_train=40, max_val=8, seed=42)
    assert {k: v for k, v in first.items() if k != "files"} == {
        k: v for k, v in second.items() if k != "files"
    }
    assert (tmp_path / "a" / "train.jsonl").read_bytes() == (tmp_path / "b" / "train.jsonl").read_bytes()


def test_build_sft_records_uses_budget() -> None:
    cfg = load_risk_config()
    samples = make_samples(300)
    records = build_sft_records(samples, max_records=cfg.settings.model.train.sanity.max_train_samples)
    assert len(records) == cfg.settings.model.train.sanity.max_train_samples
    # 记录仍是合法 SFT 结构且标签未泄漏
    for record in records[:3]:
        user_content = record["messages"][1]["content"]
        assert '"expected_decision"' not in user_content


def test_budget_stats_written_to_json(tmp_path: Path) -> None:
    stats = write_training_data(make_samples(30), tmp_path / "data", max_train=10, max_val=3)
    payload = json.loads((tmp_path / "data" / "stats.json").read_text(encoding="utf-8"))
    assert payload["max_train"] == 10 and payload["max_val"] == 3
    assert payload["prompt_version"].startswith("risk-review-prompt")
    assert payload["schema_version"] == 1
    assert payload["format"] == "sft"
    assert payload["label_leak_check"]["ok"] is True
    assert payload["train_samples"] == stats["train_samples"]
    assert Path(payload["files"]["train"]).exists()


def test_dpo_budget_format(tmp_path: Path) -> None:
    stats = write_training_data(
        make_samples(40), tmp_path / "dpo", max_train=12, max_val=4, format="dpo"
    )
    assert stats["format"] == "dpo"
    assert stats["train_samples"] <= 12
    first_line = (tmp_path / "dpo" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    assert set(record) == {"prompt", "chosen", "rejected"}


def test_invalid_budget_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_training_data(make_samples(3), tmp_path / "bad", max_train=-1, max_val=1)
