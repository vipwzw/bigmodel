"""配置加载与校验测试。"""

from __future__ import annotations

import pytest

from bigmodel_lab.config import ConfigError, deep_merge, load_config


def test_default_config_loads():
    raw, config = load_config()
    assert config.seed == 42
    assert config.profile == "standard"
    assert config.data_dir == "data"
    assert config.network.priority[0] == "cache"
    assert raw["network"]["hf_endpoint"].startswith("https://")


def test_overrides_support_dotted_paths():
    raw, config = load_config(overrides={"profile": "sanity", "network.proxy": "http://127.0.0.1:1087"})
    assert config.profile == "sanity"
    assert config.network.proxy == "http://127.0.0.1:1087"
    assert raw["network"]["proxy"] == "http://127.0.0.1:1087"


def test_invalid_profile_raises_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("profile: turbo\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_config(bad)
    message = str(exc_info.value)
    assert "profile" in message
    assert "配置校验失败" in message


def test_invalid_type_raises_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: not-an-int\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_missing_config_file_raises():
    with pytest.raises(ConfigError, match="不存在"):
        load_config("/tmp/definitely-missing-bigmodel-lab.yaml")


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 3, "z": 4}}
    merged = deep_merge(base, override)
    assert merged == {"a": 1, "nested": {"x": 1, "y": 3, "z": 4}}
    assert base["nested"]["y"] == 2
    assert override["nested"] == {"y": 3, "z": 4}


def test_stage_config_merges_over_defaults(tmp_path):
    stage = tmp_path / "stage.yaml"
    stage.write_text("seed: 123\nnetwork:\n  proxy: http://localhost:9999\n", encoding="utf-8")
    _, config = load_config(stage)
    assert config.seed == 123
    assert config.network.proxy == "http://localhost:9999"
    assert config.data_dir == "data"  # 默认值保留
