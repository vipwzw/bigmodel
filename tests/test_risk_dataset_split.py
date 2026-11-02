"""数据集划分、泄漏检查、数据卡与加载器测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigmodel_lab.risk.dataset import (
    build_splits,
    dataset_stats,
    generate_dataset,
    leak_check,
    load_binance_samples,
    load_dataset,
    load_splits,
    render_data_card,
)


def _all_samples(seed: int = 42, samples_per_family: int = 4):  # noqa: ANN202 - 测试辅助
    return [
        *generate_dataset(seed=seed, samples_per_family=samples_per_family),
        *load_binance_samples(),
    ]


def test_split_is_reproducible():
    samples = _all_samples()
    first = build_splits(samples, seed=42)
    second = build_splits(samples, seed=42)
    assert first == second
    assert build_splits(samples, seed=7) != first


def test_split_is_disjoint_and_covers_all_samples():
    samples = _all_samples()
    result = build_splits(samples, seed=42)
    train = set(result["splits"]["train"])
    val = set(result["splits"]["val"])
    test = set(result["splits"]["test"])
    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == {sample.sample_id for sample in samples}
    assert result["counts"]["total"] == len(samples)
    assert sum(result["counts"][name] for name in ("train", "val", "test")) == len(samples)


def test_binance_original_case_only_in_test_split():
    samples = _all_samples()
    result = build_splits(samples, seed=42)
    original, variants = load_binance_samples()[0], load_binance_samples()[1:]
    assert original.sample_id in result["splits"]["test"]
    assert original.sample_id not in result["splits"]["train"]
    assert original.sample_id not in result["splits"]["val"]
    assert original.sample_id in result["forced_test"]
    assert all(sample.sample_id not in result["forced_test"] for sample in variants)


def test_split_ratio_and_duplicate_id_validation():
    samples = _all_samples()
    assert leak_check(build_splits(samples, seed=42)) == []
    with pytest.raises(ValueError, match="比例"):
        build_splits(samples, seed=42, train_ratio=0.5)
    with pytest.raises(ValueError, match="重复"):
        build_splits([*samples, samples[0]], seed=42)
    with pytest.raises(ValueError, match="空样本集"):
        build_splits([], seed=42)


def test_leak_check_detects_cross_split_duplicates():
    fake = {"splits": {"train": ["a", "b"], "val": ["c"], "test": ["b", "d"]}}
    leaks = leak_check(fake)
    assert {"kind": "sample_id", "value": "b", "splits": ["test", "train"]} in leaks

    fake_instances = {
        "splits": {"train": ["a"], "val": [], "test": ["b"]},
        "instances": {"instance-x": ["a", "b"]},
    }
    instance_leaks = leak_check(fake_instances)
    assert {"kind": "instance", "value": "instance-x", "splits": ["test", "train"]} in instance_leaks


def test_small_dataset_has_all_three_splits_nonempty():
    samples = generate_dataset(seed=3, samples_per_family=6, families=["normal_ops", "native_transfer"])
    result = build_splits(samples, seed=3)
    assert result["counts"]["train"] >= 1
    assert result["counts"]["val"] >= 1
    assert result["counts"]["test"] >= 1


def test_stats_and_data_card_cover_required_sections():
    samples = _all_samples()
    splits = build_splits(samples, seed=42)
    stats = dataset_stats(samples, splits, seed=42, samples_per_family=4)
    assert stats["total"] == len(samples)
    assert stats["binance_regression_total"] == len(load_binance_samples())
    assert stats["coverage"]["complete"] is True
    assert stats["splits"]["leaks"] == []
    assert sum(stats["labels"]["exposure_buckets"].values()) == len(samples)
    card = render_data_card(stats)
    for section in ("来源与许可", "场景构成", "标签分布", "已知偏差与限制", "划分"):
        assert section in card
    assert "公开事件复盘" in card
    assert render_data_card(stats) == card


def test_load_dataset_and_splits_from_run_and_artifacts(tmp_path: Path):
    samples = generate_dataset(seed=1, samples_per_family=2, families=["normal_ops", "native_transfer"])
    splits = build_splits(samples, seed=1)
    artifacts = tmp_path / "run-1" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "samples.jsonl").write_text(
        "".join(sample.model_dump_json() + "\n" for sample in samples), encoding="utf-8"
    )
    (artifacts / "splits.json").write_text(json.dumps(splits, ensure_ascii=False), encoding="utf-8")

    assert load_dataset(tmp_path / "run-1") == samples
    assert load_dataset(artifacts) == samples
    assert load_dataset(artifacts / "samples.jsonl") == samples
    assert load_splits(tmp_path / "run-1")["counts"] == splits["counts"]
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        load_splits(tmp_path)


def test_load_dataset_rejects_invalid_jsonl(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "samples.jsonl").write_text('{"bad": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="不是合法"):
        load_dataset(artifacts)
    (artifacts / "samples.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="为空"):
        load_dataset(artifacts)
