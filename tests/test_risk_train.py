"""`lab risk train` 测试：参数装配、数据预算、manifest/摘要与失败路径（monkeypatch 训练栈）。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_risk_model_data import make_samples, write_sample_dataset

from bigmodel_lab.budget import BudgetExceeded
from bigmodel_lab.risk import train
from bigmodel_lab.risk.config import load_risk_config


class FakeSFTTrainer:
    """替换真实 SFTTrainer：记录配置、产出 adapter 与摘要，不加载模型。"""

    instances: list[FakeSFTTrainer] = []

    def __init__(self, config, *, ctx, budget, train_records, val_records):  # noqa: ANN001
        self.config = config
        self.ctx = ctx
        self.budget = budget
        self.train_records = train_records
        self.val_records = val_records
        FakeSFTTrainer.instances.append(self)

    def train(self) -> dict[str, Any]:
        adapter = self.ctx.artifact_path("adapter")
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        self.ctx.add_artifact(adapter)
        return {
            "profile": self.config.profile,
            "steps": self.config.max_steps,
            "adapter": str(adapter),
            "train_samples": len(self.train_records),
            "val_samples": len(self.val_records),
            "final_train_loss": 0.42,
            "final_val_loss": 0.5,
            "wall_seconds": 1.0,
            "device": self.config.device,
            "lora": {"r": self.config.lora.r},
        }


class FakeDPOTrainer:
    instances: list[FakeDPOTrainer] = []

    def __init__(self, config, *, ctx, budget, train_records, val_records):  # noqa: ANN001
        self.config = config
        self.ctx = ctx
        self.train_records = train_records
        self.val_records = val_records
        FakeDPOTrainer.instances.append(self)

    def train(self) -> dict[str, Any]:
        adapter = self.ctx.artifact_path("dpo_adapter")
        adapter.mkdir(parents=True, exist_ok=True)
        self.ctx.add_artifact(adapter)
        return {"adapter": str(adapter), "steps": self.config.max_steps, "final_train_loss": 0.3}


@pytest.fixture()
def fake_trainer(monkeypatch: pytest.MonkeyPatch) -> type[FakeSFTTrainer]:
    FakeSFTTrainer.instances = []
    monkeypatch.setattr(train, "SFTTrainer", FakeSFTTrainer)
    return FakeSFTTrainer


@pytest.fixture()
def risk_cfg(tmp_path: Path):
    cfg = load_risk_config()
    cfg.lab.runs_dir = str(tmp_path / "runs")
    cfg.settings.model.train.base_model_dir = str(tmp_path / "base-model")
    cfg.settings.model.train.export_dir = str(tmp_path / "export-model")
    return cfg


def _args(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "profile": "sanity",
        "dataset": None,
        "max_steps": None,
        "export": False,
        "quantize": False,
        "json": False,
        "config": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    risk_cfg,  # noqa: ANN001
    *,
    sample_count: int = 18,
    splits: dict[str, list[str]] | None = None,
) -> Path:
    samples = make_samples(sample_count)
    dataset_dir = write_sample_dataset(tmp_path / "dataset-run", samples, splits=splits)
    monkeypatch.setattr(train, "load_risk_config", lambda paths=None: risk_cfg)  # noqa: ARG005
    monkeypatch.setattr(train, "default_device", lambda: "cpu")
    return dataset_dir


def _run_dir(tmp_path: Path) -> Path:
    run_dirs = sorted((tmp_path / "runs" / "risk" / "train").iterdir())
    assert len(run_dirs) == 1
    return run_dirs[0]


def test_train_command_wires_sft_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, fake_trainer: type[FakeSFTTrainer]
) -> None:
    dataset_dir = _setup(tmp_path, monkeypatch, risk_cfg)
    exit_code = train.run_train_command(_args(dataset=str(dataset_dir), max_steps=2))
    assert exit_code == 0

    instance = fake_trainer.instances[0]
    sanity = risk_cfg.settings.model.train.sanity
    config = instance.config
    assert config.max_steps == 2
    assert config.device == "cpu"
    assert config.profile == "sanity"
    assert config.max_samples == sanity.max_train_samples
    assert config.max_seq_len == sanity.max_seq_len
    assert config.batch_size == sanity.batch_size
    assert config.grad_accum == sanity.grad_accum
    assert config.learning_rate == sanity.learning_rate
    assert config.lora.r == risk_cfg.settings.model.train.lora.r
    assert config.model_dir == Path(risk_cfg.settings.model.train.base_model_dir)
    assert config.seed == risk_cfg.settings.dataset.seed
    assert len(instance.train_records) <= sanity.max_train_samples
    assert len(instance.val_records) <= sanity.max_val_samples
    assert config.max_samples == sanity.max_train_samples

    run_dir = _run_dir(tmp_path)
    assert (run_dir / "artifacts" / "train_summary.json").exists()
    assert (run_dir / "artifacts" / "data" / "train.jsonl").exists()
    assert (run_dir / "artifacts" / "data" / "stats.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["exit_code"] == 0
    summary = json.loads((run_dir / "artifacts" / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["profile"] == "sanity"
    assert summary["dataset"] == str(dataset_dir)
    assert summary["sft"]["steps"] == 2


def test_train_command_default_max_steps_from_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, fake_trainer: type[FakeSFTTrainer]
) -> None:
    dataset_dir = _setup(tmp_path, monkeypatch, risk_cfg)
    assert train.run_train_command(_args(dataset=str(dataset_dir))) == 0
    assert fake_trainer.instances[0].config.max_steps == risk_cfg.settings.model.train.sanity.max_steps


def test_train_command_uses_latest_dataset_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, fake_trainer: type[FakeSFTTrainer]
) -> None:
    dataset_dir = _setup(tmp_path, monkeypatch, risk_cfg)
    monkeypatch.setattr(train, "latest_run", lambda stage, runs_dir=None: dataset_dir)  # noqa: ARG005
    assert train.run_train_command(_args()) == 0
    assert fake_trainer.instances[0].config.max_steps > 0
    summary = json.loads(
        (_run_dir(tmp_path) / "artifacts" / "train_summary.json").read_text(encoding="utf-8")
    )
    assert summary["dataset"] == str(dataset_dir)


def test_train_command_respects_dataset_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, fake_trainer: type[FakeSFTTrainer]
) -> None:
    samples = make_samples(12)
    train_ids = [sample.sample_id for sample in samples[:8]]
    val_ids = [sample.sample_id for sample in samples[8:]]
    dataset_dir = _setup(
        tmp_path, monkeypatch, risk_cfg, splits={"train": train_ids, "val": val_ids, "test": []}
    )
    assert train.run_train_command(_args(dataset=str(dataset_dir), max_steps=1)) == 0
    instance = fake_trainer.instances[0]
    assert len(instance.train_records) == 8
    assert len(instance.val_records) == 4
    for record in instance.train_records:
        request_id = record["messages"][1]["content"].split('"request_id":"req-', 1)[1].split('"', 1)[0]
        assert request_id in train_ids
    for record in instance.val_records:
        request_id = record["messages"][1]["content"].split('"request_id":"req-', 1)[1].split('"', 1)[0]
        assert request_id in val_ids


def test_train_command_missing_dataset_returns_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(train, "load_risk_config", lambda paths=None: risk_cfg)  # noqa: ARG005
    exit_code = train.run_train_command(_args(dataset=str(tmp_path / "no-such-dataset")))
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "lab risk data generate" in captured.err
    assert not (tmp_path / "runs" / "risk" / "train").exists()


def test_train_command_no_dataset_run_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(train, "load_risk_config", lambda paths=None: risk_cfg)  # noqa: ARG005
    monkeypatch.setattr(train, "latest_run", lambda stage, runs_dir=None: None)  # noqa: ARG005
    assert train.run_train_command(_args()) == 1
    assert "lab risk data generate" in capsys.readouterr().err


def test_train_command_empty_dataset_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_dir = tmp_path / "empty-dataset"
    (dataset_dir / "artifacts").mkdir(parents=True)
    (dataset_dir / "artifacts" / "samples.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(train, "load_risk_config", lambda paths=None: risk_cfg)  # noqa: ARG005
    assert train.run_train_command(_args(dataset=str(dataset_dir))) == 1
    assert "数据集为空" in capsys.readouterr().err


def test_train_command_rejects_unknown_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(train, "load_risk_config", lambda paths=None: risk_cfg)  # noqa: ARG005
    assert train.run_train_command(_args(profile="turbo")) == 1
    assert "sanity/standard" in capsys.readouterr().err


def test_train_command_failure_marks_manifest_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_dir = _setup(tmp_path, monkeypatch, risk_cfg)

    class BrokenTrainer(FakeSFTTrainer):
        def train(self) -> dict[str, Any]:
            raise RuntimeError("显存不足")

    monkeypatch.setattr(train, "SFTTrainer", BrokenTrainer)
    assert train.run_train_command(_args(dataset=str(dataset_dir))) == 1
    assert "显存不足" in capsys.readouterr().err
    manifest = json.loads((_run_dir(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "RuntimeError" in manifest["error"]


def test_train_command_budget_exceeded_marks_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_dir = _setup(tmp_path, monkeypatch, risk_cfg)

    class TimeoutTrainer(FakeSFTTrainer):
        def train(self) -> dict[str, Any]:
            raise BudgetExceeded("超过 10 分钟")

    monkeypatch.setattr(train, "SFTTrainer", TimeoutTrainer)
    assert train.run_train_command(_args(dataset=str(dataset_dir))) == 1
    assert "训练超时" in capsys.readouterr().err
    manifest = json.loads((_run_dir(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "timeout"


def test_train_command_json_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, fake_trainer: type[FakeSFTTrainer], capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_dir = _setup(tmp_path, monkeypatch, risk_cfg)
    assert train.run_train_command(_args(dataset=str(dataset_dir), max_steps=1, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "sanity"
    assert payload["sft"]["steps"] == 1
    assert payload["data"]["train_samples"] > 0


def test_train_command_optional_dpo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg, fake_trainer: type[FakeSFTTrainer]
) -> None:
    dataset_dir = _setup(tmp_path, monkeypatch, risk_cfg)
    risk_cfg.settings.model.train.dpo.enabled = True
    FakeDPOTrainer.instances = []
    monkeypatch.setattr(train, "DPOTrainer", FakeDPOTrainer)
    merge_calls: list[dict[str, Any]] = []

    def fake_merge_and_export(*, adapter_dir, base_model_dir, output_dir, device):  # noqa: ANN001
        merge_calls.append(
            {"adapter_dir": adapter_dir, "base_model_dir": base_model_dir, "output_dir": output_dir, "device": device}
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"output_dir": str(output_dir), "loadable": True}

    monkeypatch.setattr(train, "merge_and_export", fake_merge_and_export)
    assert train.run_train_command(_args(dataset=str(dataset_dir), max_steps=1)) == 0
    assert merge_calls and Path(merge_calls[0]["output_dir"]).name == "dpo_base_merged"
    assert len(FakeDPOTrainer.instances) == 1
    dpo_summary = json.loads(
        (_run_dir(tmp_path) / "artifacts" / "dpo_summary.json").read_text(encoding="utf-8")
    )
    assert dpo_summary["steps"] == risk_cfg.settings.model.train.dpo.max_steps
    # DPO 数据以 prompt/chosen/rejected 格式写出
    dpo_line = (
        _run_dir(tmp_path) / "artifacts" / "dpo_data" / "train.jsonl"
    ).read_text(encoding="utf-8").splitlines()[0]
    assert set(json.loads(dpo_line)) == {"prompt", "chosen", "rejected"}
