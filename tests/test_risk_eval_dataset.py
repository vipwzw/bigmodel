"""`risk.evaluate` 评测集加载测试：划分切片、场景族/时间过滤、泄漏检测与标签校验。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bigmodel_lab.risk.dataset import (
    BINANCE_CASES_FILENAME,
    SAMPLES_FILENAME,
    SPLITS_FILENAME,
    STATS_FILENAME,
    build_splits,
    dataset_stats,
    generate_dataset,
    load_binance_samples,
)
from bigmodel_lab.risk.evaluate import EvalDatasetError, load_eval_dataset
from bigmodel_lab.risk.schema import RiskSample

SEED = 7


def _parse(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _jsonl(samples: list[RiskSample]) -> str:
    return "".join(f"{sample.model_dump_json()}\n" for sample in samples)


def _write_run_dir(
    base: Path,
    *,
    samples: list[RiskSample] | None = None,
    splits: dict | None = None,
    stats: dict | None = None,
    binance: list[RiskSample] | None = None,
) -> Path:
    """把样本、划分、统计与币安用例写入 run 目录的 artifacts/。"""
    samples = samples if samples is not None else _full_samples()
    if splits is None:
        try:
            splits = build_splits(samples, seed=SEED)
        except ValueError:  # 重复 sample_id 等非法输入：写入最小可加载划分，交由加载器报错
            splits = {
                "splits": {
                    "train": [],
                    "val": [],
                    "test": sorted({sample.sample_id for sample in samples}),
                }
            }
    stats = (
        stats
        if stats is not None
        else dataset_stats(samples, splits, seed=SEED, samples_per_family=2)
    )
    binance = binance if binance is not None else load_binance_samples()
    artifacts = base / "dataset-run" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / SAMPLES_FILENAME).write_text(_jsonl(samples), encoding="utf-8")
    (artifacts / SPLITS_FILENAME).write_text(json.dumps(splits, ensure_ascii=False), encoding="utf-8")
    (artifacts / STATS_FILENAME).write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    if binance:
        (artifacts / BINANCE_CASES_FILENAME).write_text(_jsonl(binance), encoding="utf-8")
    return base / "dataset-run"


def _full_samples() -> list[RiskSample]:
    return [
        *generate_dataset(seed=SEED, samples_per_family=2),
        *load_binance_samples(),
    ]


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    return _write_run_dir(tmp_path)


# ---------------------------------------------------------------------------
# 加载与划分
# ---------------------------------------------------------------------------


def test_load_holdout_split_matches_splits_json(run_dir: Path):
    samples = _full_samples()
    splits = build_splits(samples, seed=SEED)
    dataset = load_eval_dataset(run_dir, split="test")

    assert dataset.split == "test"
    assert {sample.sample_id for sample in dataset.samples} == set(splits["splits"]["test"])
    assert dataset.leaks == []
    assert dataset.stats["splits"]["counts"]["test"] == len(dataset.samples)
    assert len(dataset.samples) > 0

    describe = dataset.describe()
    assert describe["source"].endswith(SAMPLES_FILENAME)
    assert describe["split"] == "test"
    assert describe["total"] == len(dataset.samples)
    assert describe["positive"] + describe["negative"] == describe["total"]
    assert describe["stats"]["total"] == len(dataset.samples)


def test_load_supports_run_artifacts_and_direct_jsonl_paths(run_dir: Path):
    from_run = load_eval_dataset(run_dir, split="test")
    from_artifacts = load_eval_dataset(run_dir / "artifacts", split="test")
    from_file = load_eval_dataset(run_dir / "artifacts" / SAMPLES_FILENAME, split="test")

    expected = [sample.sample_id for sample in from_run.samples]
    assert [sample.sample_id for sample in from_artifacts.samples] == expected
    assert [sample.sample_id for sample in from_file.samples] == expected


def test_load_without_split_keeps_all_samples_and_reports_leaks(run_dir: Path):
    samples = _full_samples()
    splits = build_splits(samples, seed=SEED)
    dataset = load_eval_dataset(run_dir)

    assert dataset.split is None
    assert len(dataset.samples) == len(samples)
    assert dataset.leaks, "全量评测必然与 train 划分有重叠，应报告泄漏"
    train_ids = set(splits["splits"]["train"])
    leak_ids = {item["value"] for item in dataset.leaks if item["kind"] == "sample_id"}
    assert leak_ids
    assert leak_ids <= train_ids
    assert any("泄漏" in warning for warning in dataset.warnings)


def test_strict_mode_rejects_train_leak_but_holdout_is_clean(run_dir: Path):
    with pytest.raises(EvalDatasetError, match="泄漏"):
        load_eval_dataset(run_dir, strict=True)
    # holdout 划分与训练集不共享实例，严格模式应通过
    dataset = load_eval_dataset(run_dir, split="test", strict=True)
    assert dataset.leaks == []


def test_instance_level_leak_detection(tmp_path: Path):
    base = generate_dataset(seed=SEED, samples_per_family=2, families=["normal_ops", "native_transfer"])
    first, second = base[0], base[1]
    first.request.payload["instance_id"] = "shared-instance"
    second.request.payload["instance_id"] = "shared-instance"
    samples = [first, second]
    splits = {
        "schema_version": 1,
        "splits": {"train": [first.sample_id], "val": [], "test": [second.sample_id]},
        "instances": {
            "shared-instance": [first.sample_id, second.sample_id],
            "normal_ops-other": [first.sample_id],
        },
    }
    run = _write_run_dir(tmp_path, samples=samples, splits=splits, binance=[])

    dataset = load_eval_dataset(run, split="test")
    instance_leaks = [item for item in dataset.leaks if item["kind"] == "instance"]
    assert instance_leaks
    assert instance_leaks[0]["value"] == "shared-instance"
    assert instance_leaks[0]["splits"] == ["eval", "train"]
    with pytest.raises(EvalDatasetError, match="泄漏"):
        load_eval_dataset(run, split="test", strict=True)


# ---------------------------------------------------------------------------
# 场景族 / 时间过滤
# ---------------------------------------------------------------------------


def test_family_filter_limits_samples(run_dir: Path):
    dataset = load_eval_dataset(run_dir, families=["normal_ops", "native_transfer"])

    assert dataset.families == ("normal_ops", "native_transfer")
    assert {sample.label.scenario_family for sample in dataset.samples} <= {
        "normal_ops",
        "native_transfer",
    }
    assert len(dataset.samples) == sum(
        1 for sample in _full_samples() if sample.label.scenario_family in {"normal_ops", "native_transfer"}
    )


def test_family_filter_rejects_unknown_family(run_dir: Path):
    with pytest.raises(EvalDatasetError, match="未知场景族"):
        load_eval_dataset(run_dir, families=["quantum_heist"])


def test_time_window_filter(run_dir: Path):
    all_dataset = load_eval_dataset(run_dir)
    stamps = sorted(_parse(sample.request.timestamp) for sample in all_dataset.samples)
    cut = stamps[len(stamps) // 2]

    dataset = load_eval_dataset(run_dir, since=cut.isoformat())
    assert dataset.time_window["since"] == cut.isoformat()
    assert dataset.samples
    assert all(_parse(sample.request.timestamp) >= cut for sample in dataset.samples)
    assert len(dataset.samples) < len(all_dataset.samples)

    single = load_eval_dataset(run_dir, since=stamps[0].isoformat(), until=stamps[0].isoformat())
    assert len(single.samples) == sum(1 for stamp in stamps if stamp == stamps[0])

    with pytest.raises(EvalDatasetError, match="时间窗口非法"):
        load_eval_dataset(run_dir, since=stamps[1].isoformat(), until=stamps[0].isoformat())
    # 上界早于所有样本时间时过滤为空
    with pytest.raises(EvalDatasetError, match="没有样本"):
        load_eval_dataset(run_dir, until="2020-01-01T00:00:00Z")


# ---------------------------------------------------------------------------
# 标签与场景族分布校验
# ---------------------------------------------------------------------------


def _tamper(sample: RiskSample, **label_updates) -> RiskSample:  # noqa: ANN003 - 测试辅助
    return sample.model_copy(update={"label": sample.label.model_copy(update=label_updates)})


def test_missing_family_distribution_is_reported_as_warning(tmp_path: Path):
    samples = generate_dataset(seed=SEED, samples_per_family=2, families=["normal_ops"])
    run = _write_run_dir(tmp_path, samples=samples, binance=[])
    dataset = load_eval_dataset(run)

    missing = dataset.describe()["missing_families"]
    assert len(missing) == 8
    assert "normal_ops" not in missing
    assert any("缺少场景族" in warning for warning in dataset.warnings)


def test_duplicate_sample_id_rejected(tmp_path: Path):
    samples = generate_dataset(seed=SEED, samples_per_family=2, families=["normal_ops"])
    run = _write_run_dir(tmp_path, samples=[*samples, samples[0]], binance=[])
    with pytest.raises(EvalDatasetError, match="重复"):
        load_eval_dataset(run)


def test_unknown_scenario_family_rejected(tmp_path: Path):
    samples = [_tamper(generate_dataset(seed=1, samples_per_family=2, families=["normal_ops"])[0], scenario_family="quantum_heist")]
    run = _write_run_dir(tmp_path, samples=samples, binance=[])
    with pytest.raises(EvalDatasetError, match="场景族未知"):
        load_eval_dataset(run)


def test_exposure_ratio_out_of_range_rejected(tmp_path: Path):
    samples = [_tamper(generate_dataset(seed=1, samples_per_family=2, families=["normal_ops"])[0], exposure_ratio=1.5)]
    run = _write_run_dir(tmp_path, samples=samples, binance=[])
    with pytest.raises(EvalDatasetError, match="exposure_ratio"):
        load_eval_dataset(run)


def test_empty_label_text_rejected(tmp_path: Path):
    samples = [_tamper(generate_dataset(seed=1, samples_per_family=2, families=["normal_ops"])[0], action_type="")]
    run = _write_run_dir(tmp_path, samples=samples, binance=[])
    with pytest.raises(EvalDatasetError, match="action_type"):
        load_eval_dataset(run)


def test_unknown_split_name_rejected(run_dir: Path):
    with pytest.raises(EvalDatasetError, match="未知评测划分"):
        load_eval_dataset(run_dir, split="holdout")


def test_cli_strict_leak_returns_nonzero(tmp_path: Path):
    """严格模式下训练集泄漏必须转为 CLI 非零退出码。"""
    from bigmodel_lab.cli import main  # noqa: PLC0415 - 仅该用例需要 CLI

    samples = _full_samples()
    splits = build_splits(samples, seed=SEED)
    # 人为制造 test 与 train 重叠，模拟数据划分被污染
    splits["splits"]["test"] = [*splits["splits"]["test"], splits["splits"]["train"][0]]
    run = _write_run_dir(tmp_path, samples=samples, splits=splits)

    code = main(["risk", "eval", "--dataset", str(run), "--out", str(tmp_path / "out")])
    assert code == 2
