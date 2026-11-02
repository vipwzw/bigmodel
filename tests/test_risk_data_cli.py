"""`lab risk data` CLI 行为测试：generate/validate/show 与产物完整性。"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
from argparse import Namespace
from pathlib import Path

import pytest

from bigmodel_lab.cli import main
from bigmodel_lab.risk.config import RISK_CONFIG_PATH
from bigmodel_lab.risk.dataset import (
    BINANCE_CASES_FILENAME,
    DATA_CARD_FILENAME,
    SAMPLES_FILENAME,
    SPLITS_FILENAME,
    STATS_FILENAME,
    run_data_command,
)

ARTIFACT_NAMES = (
    SAMPLES_FILENAME,
    SPLITS_FILENAME,
    DATA_CARD_FILENAME,
    STATS_FILENAME,
    BINANCE_CASES_FILENAME,
)


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _generate(
    out: Path, *, seed: int = 7, samples_per_family: int = 3, as_json: bool = True
) -> tuple[int, dict, str]:
    argv = [
        "risk",
        "data",
        "generate",
        "--out",
        str(out),
        "--seed",
        str(seed),
        "--samples-per-family",
        str(samples_per_family),
    ]
    if as_json:
        argv.append("--json")
    code, stdout, stderr = _run_cli(argv)
    summary = json.loads(stdout) if as_json and code == 0 and stdout.strip().startswith("{") else {}
    return code, summary, stderr


@pytest.fixture(scope="module")
def dataset_run(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("risk-data-cli")
    code, summary, stderr = _generate(base / "runs")
    assert code == 0, stderr
    summary["base_dir"] = str(base)
    return summary


def _artifacts(summary: dict) -> Path:
    return Path(summary["artifacts_dir"])


def _run_dir(summary: dict) -> Path:
    return Path(summary["run_dir"])


def _copy_artifacts(summary: dict, destination: Path) -> Path:
    target = destination / "artifacts"
    shutil.copytree(_artifacts(summary), target)
    return target


def test_generate_writes_complete_artifacts(dataset_run):
    summary = dataset_run
    run_dir, artifacts = _run_dir(summary), _artifacts(summary)
    for name in ARTIFACT_NAMES:
        assert (artifacts / name).is_file(), name
    assert run_dir.parent == (Path(summary["base_dir"]) / "runs" / "risk" / "dataset").resolve()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["stage"] == "risk/dataset"
    assert summary["samples"] == 31
    assert summary["synthetic_samples"] == 27
    assert summary["binance_cases"] == 4
    assert summary["seed"] == 7
    assert summary["coverage_complete"] is True
    assert summary["splits"]["total"] == 31
    assert set(summary["files"]) == set(ARTIFACT_NAMES)

    stats = json.loads((artifacts / STATS_FILENAME).read_text(encoding="utf-8"))
    assert stats["seed"] == 7
    assert stats["total"] == 31
    samples = [
        json.loads(line)
        for line in (artifacts / SAMPLES_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(samples) == 31
    assert all(sample["schema_version"] == 1 for sample in samples)
    splits = json.loads((artifacts / SPLITS_FILENAME).read_text(encoding="utf-8"))
    assert sum(splits["counts"][name] for name in ("train", "val", "test")) == 31


def test_generate_same_seed_is_byte_identical(tmp_path):
    first_code, first, first_error = _generate(tmp_path / "runs-a")
    second_code, second, second_error = _generate(tmp_path / "runs-b")
    assert first_code == 0, first_error
    assert second_code == 0, second_error
    for name in ARTIFACT_NAMES:
        assert (_artifacts(first) / name).read_bytes() == (_artifacts(second) / name).read_bytes(), name


def test_generate_text_summary_when_not_json(tmp_path):
    code, stdout, stderr = _run_cli(
        [
            "risk",
            "data",
            "generate",
            "--out",
            str(tmp_path / "runs"),
            "--seed",
            "3",
            "--samples-per-family",
            "2",
        ]
    )
    assert code == 0, stderr
    assert "生成完成" in stdout
    assert "samples.jsonl" in stdout
    assert "划分:" in stdout


def test_generate_rejects_too_few_samples_per_family(tmp_path):
    code, stdout, stderr = _run_cli(
        [
            "risk",
            "data",
            "generate",
            "--out",
            str(tmp_path / "runs"),
            "--samples-per-family",
            "1",
        ]
    )
    assert code == 2
    assert "参数错误" in stderr
    assert stdout == ""


def test_validate_ok(dataset_run):
    code, stdout, stderr = _run_cli(
        ["risk", "data", "validate", "--dataset", str(_run_dir(dataset_run)), "--json"]
    )
    assert code == 0, stderr
    report = json.loads(stdout)
    assert report["ok"] is True
    assert report["invalid"] == 0
    assert report["splits"]["present"] is True
    assert report["leaks"] == []
    assert report["coverage"]["complete"] is True
    assert report["binance_cases"]["ok"] is True

    args = Namespace(
        data_command="validate", config=[], dataset=str(_artifacts(dataset_run)), json=True
    )
    assert run_data_command(args) == 0


def test_validate_reports_invalid_samples(tmp_path, dataset_run):
    artifacts = _copy_artifacts(dataset_run, tmp_path)
    samples_path = artifacts / SAMPLES_FILENAME
    lines = [line for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples_path.write_text("\n".join(lines) + '\n{"bad": true}\n', encoding="utf-8")
    code, stdout, stderr = _run_cli(
        ["risk", "data", "validate", "--dataset", str(artifacts), "--json"]
    )
    assert code == 1, stderr
    report = json.loads(stdout)
    assert report["invalid"] == 1
    assert report["ok"] is False
    assert report["errors"][0]["line"] == len(lines) + 1


def test_validate_detects_split_leak(tmp_path, dataset_run):
    artifacts = _copy_artifacts(dataset_run, tmp_path)
    splits_path = artifacts / SPLITS_FILENAME
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    splits["splits"]["test"].append(splits["splits"]["train"][0])
    splits_path.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    code, stdout, _stderr = _run_cli(
        ["risk", "data", "validate", "--dataset", str(artifacts), "--json"]
    )
    assert code == 1
    report = json.loads(stdout)
    assert report["leaks"]
    assert report["ok"] is False


def test_validate_missing_dataset_returns_error_code(tmp_path):
    code, _stdout, stderr = _run_cli(
        ["risk", "data", "validate", "--dataset", str(tmp_path / "missing"), "--json"]
    )
    assert code == 2
    assert "文件不存在" in stderr


def test_show_outputs_stats_and_data_card(dataset_run):
    run_dir = _run_dir(dataset_run)
    code, stdout, stderr = _run_cli(["risk", "data", "show", "--dataset", str(run_dir), "--json"])
    assert code == 0, stderr
    payload = json.loads(stdout)
    assert payload["stats"]["total"] == 31
    assert payload["stats"]["seed"] == 7
    card = payload["data_card"]
    assert card["path"]
    assert "来源与许可" in card["sections"]

    code, stdout, _stderr = _run_cli(["risk", "data", "show", "--dataset", str(run_dir)])
    assert code == 0
    assert "数据卡章节" in stdout


def test_show_missing_dataset_returns_error_code(tmp_path):
    code, _stdout, stderr = _run_cli(
        ["risk", "data", "show", "--dataset", str(tmp_path / "missing")]
    )
    assert code == 2
    assert "文件不存在" in stderr


def test_generate_uses_config_seed_override(tmp_path):
    override = tmp_path / "risk-override.yaml"
    override.write_text("risk:\n  dataset:\n    seed: 99\n", encoding="utf-8")
    code, stdout, stderr = _run_cli(
        [
            "risk",
            "data",
            "generate",
            "--config",
            str(RISK_CONFIG_PATH),
            "--config",
            str(override),
            "--out",
            str(tmp_path / "runs"),
            "--samples-per-family",
            "2",
            "--json",
        ]
    )
    assert code == 0, stderr
    summary = json.loads(stdout)
    assert summary["seed"] == 99
    stats = json.loads((_artifacts(summary) / STATS_FILENAME).read_text(encoding="utf-8"))
    assert stats["seed"] == 99


def test_run_data_command_rejects_unknown_subcommand():
    with pytest.raises(ValueError, match="未知 data 子命令"):
        run_data_command(Namespace(data_command="bogus"))
