"""`lab risk train --export/--quantize` 测试：合并导出路径、量化装配与失败处理。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_risk_model_data import make_samples, write_sample_dataset
from test_risk_train import FakeSFTTrainer, _args

from bigmodel_lab.risk import infer, train
from bigmodel_lab.risk.config import load_risk_config


@pytest.fixture()
def risk_cfg(tmp_path: Path):
    cfg = load_risk_config()
    cfg.lab.runs_dir = str(tmp_path / "runs")
    cfg.settings.model.train.base_model_dir = str(tmp_path / "base-model")
    cfg.settings.model.train.export_dir = str(tmp_path / "exports" / "risk-review")
    return cfg


@pytest.fixture()
def prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, risk_cfg
) -> tuple[Path, dict[str, Any]]:
    samples = make_samples(12)
    dataset_dir = write_sample_dataset(tmp_path / "dataset-run", samples)
    monkeypatch.setattr(train, "load_risk_config", lambda paths=None: risk_cfg)  # noqa: ARG005
    monkeypatch.setattr(train, "default_device", lambda: "cpu")
    FakeSFTTrainer.instances = []
    monkeypatch.setattr(train, "SFTTrainer", FakeSFTTrainer)
    calls: dict[str, Any] = {}

    def fake_merge_and_export(*, adapter_dir, base_model_dir, output_dir, device):  # noqa: ANN001
        calls.update(
            {
                "adapter_dir": Path(adapter_dir),
                "base_model_dir": Path(base_model_dir),
                "output_dir": Path(output_dir),
                "device": device,
            }
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "config.json").write_text("{}", encoding="utf-8")
        return {"output_dir": str(output_dir), "loadable": True, "sample": "ok"}

    monkeypatch.setattr(train, "merge_and_export", fake_merge_and_export)
    return dataset_dir, calls


def _latest_run_dir(tmp_path: Path) -> Path:
    return sorted((tmp_path / "runs" / "risk" / "train").iterdir())[-1]


def test_export_uses_config_export_dir_and_merge_stack(
    tmp_path: Path,
    prepared: tuple[Path, dict[str, Any]],
    risk_cfg,  # noqa: ANN001
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir, calls = prepared
    exit_code = train.run_train_command(_args(dataset=str(dataset_dir), max_steps=1, export=True))
    assert exit_code == 0
    # merge_and_export 收到 adapter / 基座 / 配置中的导出目录
    assert calls["output_dir"] == Path(risk_cfg.settings.model.train.export_dir)
    assert calls["base_model_dir"] == Path(risk_cfg.settings.model.train.base_model_dir)
    assert calls["adapter_dir"].name == "adapter"
    assert calls["device"] == "cpu"

    run_dir = _latest_run_dir(tmp_path)
    export_summary = json.loads((run_dir / "artifacts" / "export.json").read_text(encoding="utf-8"))
    assert export_summary["export_dir"] == str(calls["output_dir"])
    assert export_summary["loadable"] is True
    train_summary = json.loads((run_dir / "artifacts" / "train_summary.json").read_text(encoding="utf-8"))
    assert train_summary["export"]["loadable"] is True
    assert "导出模型" in capsys.readouterr().out
    # 导出后 infer.build_reviewer 必须优先选中导出目录
    reviewer = infer.build_reviewer(risk_cfg)
    assert reviewer.source == "exported"
    assert reviewer.model_dir == Path(risk_cfg.settings.model.train.export_dir)


def test_export_failure_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared: tuple[Path, dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir, _ = prepared

    def broken_merge_and_export(**_kwargs):  # noqa: ANN003
        raise RuntimeError("合并导出失败")

    monkeypatch.setattr(train, "merge_and_export", broken_merge_and_export)
    assert train.run_train_command(_args(dataset=str(dataset_dir), export=True)) == 1
    assert "合并导出失败" in capsys.readouterr().err
    manifest = json.loads((_latest_run_dir(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_quantize_unavailable_fails_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared: tuple[Path, dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir, _ = prepared
    monkeypatch.setattr(train, "_mlx_available", lambda: False)
    assert train.run_train_command(_args(dataset=str(dataset_dir), quantize=True)) == 1
    captured = capsys.readouterr()
    assert "mlx-lm" in captured.err
    # 快速失败：未创建训练 run、未调用训练器
    assert not (tmp_path / "runs" / "risk" / "train").exists()
    assert FakeSFTTrainer.instances == []


def test_quantize_auto_exports_then_converts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared: tuple[Path, dict[str, Any]],
    risk_cfg,  # noqa: ANN001
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir, calls = prepared
    monkeypatch.setattr(train, "_mlx_available", lambda: True)
    quantized: list[Path] = []

    def fake_quantize(model_dir: Path, *, bits: int = 4) -> Path:
        quantized.append(Path(model_dir))
        output = tmp_path / "mlx-quantized"
        output.mkdir(parents=True, exist_ok=True)
        return output

    monkeypatch.setattr(train, "_quantize_model", fake_quantize)
    exit_code = train.run_train_command(_args(dataset=str(dataset_dir), max_steps=1, quantize=True))
    assert exit_code == 0
    # --quantize 未显式 --export 时自动先导出
    assert calls["output_dir"] == Path(risk_cfg.settings.model.train.export_dir)
    assert quantized == [Path(risk_cfg.settings.model.train.export_dir)]
    run_dir = _latest_run_dir(tmp_path)
    quantize_summary = json.loads((run_dir / "artifacts" / "quantize.json").read_text(encoding="utf-8"))
    assert quantize_summary["bits"] == 4
    assert quantize_summary["path"] == str(tmp_path / "mlx-quantized")
    assert str(tmp_path / "mlx-quantized") in capsys.readouterr().out


def test_quantize_model_calls_mlx_convert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_convert(model_path: str, output_path: str, *, quantize: bool, q_bits: int) -> None:
        calls.append({"model": model_path, "output": output_path, "quantize": quantize, "bits": q_bits})
        Path(output_path).mkdir(parents=True, exist_ok=True)

    fake_module = SimpleNamespace(convert=fake_convert)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)
    monkeypatch.setattr(train, "PROJECT_ROOT", tmp_path)
    model_dir = tmp_path / "merged-model"
    model_dir.mkdir()
    output = train._quantize_model(model_dir, bits=4)
    assert output == tmp_path / "models" / "exports" / "mlx-merged-model-4bit"
    assert calls == [
        {
            "model": str(model_dir),
            "output": str(output),
            "quantize": True,
            "bits": 4,
        }
    ]


def test_mlx_available_reflects_import(monkeypatch: pytest.MonkeyPatch) -> None:
    # 已安装：find_spec 有结果且可导入
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(convert=lambda *args, **kwargs: None))
    monkeypatch.setattr(train.importlib.util, "find_spec", lambda name: object())  # noqa: ARG005
    assert train._mlx_available() is True
    # 未安装：find_spec 返回 None
    monkeypatch.setattr(train.importlib.util, "find_spec", lambda name: None)  # noqa: ARG005
    assert train._mlx_available() is False
