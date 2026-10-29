"""实验目录与 manifest 测试。"""

from __future__ import annotations

import json

from bigmodel_lab.manifest import RunContext, code_fingerprint, config_fingerprint


def test_run_manifest_roundtrip(tmp_path):
    config = {"runs_dir": str(tmp_path), "seed": 7, "profile": "sanity", "offline": True}
    ctx = RunContext.create("stage-1", config, runs_dir=tmp_path)
    assert ctx.run_dir.parent.name == "stage-1"
    assert (ctx.run_dir / "config.yaml").exists()
    assert (ctx.run_dir / "manifest.json").exists()
    assert (ctx.run_dir / "artifacts").is_dir()
    assert (ctx.run_dir / "checkpoints").is_dir()

    ctx.log_metrics(step=1, loss=1.23, tokens_per_second=100.0)
    ctx.write_artifact("report.md", "# ok")
    ctx.write_json_artifact("data.json", {"a": 1})
    manifest = ctx.finish("success", 0)

    loaded = json.loads((ctx.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded["status"] == "success"
    assert loaded["exit_code"] == 0
    assert loaded["seed"] == 7
    assert loaded["profile"] == "sanity"
    assert loaded["offline"] is True
    assert loaded["metrics_records"] == 1
    assert "artifacts/report.md" in loaded["artifacts"]
    assert "artifacts/data.json" in loaded["artifacts"]
    assert loaded["finished_at"] is not None
    assert loaded["duration_seconds"] >= 0
    assert loaded["environment"]["python"]
    assert loaded["code"]["fingerprint_sha256"]
    assert manifest["status"] == "success"

    metrics = (ctx.run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics) == 1
    assert json.loads(metrics[0])["loss"] == 1.23


def test_context_manager_marks_success(tmp_path):
    with RunContext.create("stage-x", {"runs_dir": str(tmp_path)}, runs_dir=tmp_path) as ctx:
        ctx.write_artifact("a.txt", "hello")
    loaded = json.loads((ctx.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded["status"] == "success"


def test_context_manager_marks_failure(tmp_path):
    ctx = None
    try:
        with RunContext.create("stage-x", {"runs_dir": str(tmp_path)}, runs_dir=tmp_path) as ctx:
            raise ValueError("boom")
    except ValueError:
        pass
    loaded = json.loads((ctx.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded["status"] == "failed"
    assert "ValueError" in loaded["error"]


def test_fingerprints_are_stable_and_sensitive(tmp_path):
    config_a = {"seed": 1}
    config_b = {"seed": 2}
    assert config_fingerprint(config_a) == config_fingerprint(dict(reversed(list(config_a.items()))))
    assert config_fingerprint(config_a) != config_fingerprint(config_b)

    source = tmp_path / "src"
    source.mkdir()
    (source / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")
    first = code_fingerprint(tmp_path)
    (source / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert code_fingerprint(tmp_path) != first
