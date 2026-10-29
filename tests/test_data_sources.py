"""资源获取（多源/缓存/离线/代理）测试，不访问真实网络。"""

from __future__ import annotations

import os

import pytest

from bigmodel_lab.config import NetworkConfig
from bigmodel_lab.data import sources
from bigmodel_lab.data.specs import ResourceSpec


def _spec() -> ResourceSpec:
    return ResourceSpec(
        key="demo",
        repo_id="demo/x",
        files=["a.txt"],
        text_field="text",
        language="zh",
        license="test",
        approx_size_gb=0.001,
    )


def test_download_then_cache_hit(tmp_path, monkeypatch):
    spec = _spec()

    def fake_download(spec, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "a.txt").write_text("hello world", encoding="utf-8")
        (dest / "fetch_manifest.json").unlink(missing_ok=True)

    monkeypatch.setattr(sources, "_download_modelscope", fake_download)
    first = sources.fetch_resource(spec, data_dir=tmp_path, network=NetworkConfig(), offline=False)
    assert first["status"] == "downloaded"
    assert first["source"] == "modelscope"
    assert (tmp_path / "raw" / "demo" / "a.txt").read_text(encoding="utf-8") == "hello world"
    assert (tmp_path / "raw" / "demo" / "fetch_manifest.json").exists()

    second = sources.fetch_resource(spec, data_dir=tmp_path, network=NetworkConfig(), offline=False)
    assert second["status"] == "cache_hit"
    assert second["files"][0]["path"] == "a.txt"


def test_offline_without_cache_raises(tmp_path):
    with pytest.raises(RuntimeError, match="离线模式"):
        sources.fetch_resource(_spec(), data_dir=tmp_path, network=NetworkConfig(), offline=True)


def test_fallback_to_next_source(tmp_path, monkeypatch):
    spec = _spec()
    spec.source = "modelscope"
    network = NetworkConfig(priority=["cache", "modelscope", "hf-mirror", "hf"])

    def failing_modelscope(spec, dest):
        raise RuntimeError("modelscope 挂了")

    def fake_hf(spec, dest, endpoint):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "a.txt").write_text("from hf mirror", encoding="utf-8")

    monkeypatch.setattr(sources, "_download_modelscope", failing_modelscope)
    monkeypatch.setattr(sources, "_download_hf", fake_hf)
    entry = sources.fetch_resource(spec, data_dir=tmp_path, network=network, offline=False)
    assert entry["status"] == "downloaded"
    assert entry["source"] == "hf-mirror"


def test_proxy_environment_sets_and_restores(monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    with sources.proxy_environment("http://127.0.0.1:1087"):
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:1087"
        assert "modelscope.cn" in os.environ["NO_PROXY"]
    assert "HTTPS_PROXY" not in os.environ


def test_manifest_detects_truncated_file(tmp_path):
    spec = _spec()

    def fake_download(spec, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "a.txt").write_text("hello", encoding="utf-8")

    sources._download_modelscope = fake_download  # type: ignore[assignment]
    try:
        sources.fetch_resource(spec, data_dir=tmp_path, network=NetworkConfig(), offline=False)
    finally:
        del sources._download_modelscope
    dest = tmp_path / "raw" / "demo"
    (dest / "a.txt").write_text("hel", encoding="utf-8")
    assert sources.manifest_complete(dest) is None
