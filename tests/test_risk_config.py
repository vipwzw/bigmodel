"""风险审查配置加载与校验测试。"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from bigmodel_lab.risk.config import (
    RISK_CONFIG_PATH,
    RISK_RULES_PATH,
    RiskConfigError,
    load_risk_config,
    load_rule_settings,
)


def _config_payload() -> dict:
    return yaml.safe_load(Path(RISK_CONFIG_PATH).read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict, name: str = "risk.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_default_config_loads():
    config = load_risk_config()
    assert config.settings.dataset.seed == 42
    assert config.settings.review.mode == "rules_only"
    assert config.settings.model.backend == "transformers"
    assert "eip7702_delegation" in config.settings.dataset.families
    assert config.rules.version == "risk-rules-v1"
    assert config.rules.thresholds.block_above == 0.5
    assert config.lab.profile == "standard"
    assert config.raw["risk"]["simulator"] == "offline-v1"


def test_multi_path_override_merges(tmp_path):
    override = _write(tmp_path, {"risk": {"eval": {"min_recall": 0.88}}}, "override.yaml")
    config = load_risk_config([RISK_CONFIG_PATH, override])
    assert config.settings.eval.min_recall == 0.88
    assert config.settings.eval.max_fpr == 0.10  # 未被覆盖字段保留默认


def test_invalid_ratio_raises(tmp_path):
    payload = _config_payload()
    payload["risk"]["dataset"]["train_ratio"] = 0.5
    with pytest.raises(RiskConfigError, match="比例"):
        load_risk_config(_write(tmp_path, payload))


def test_unknown_field_raises(tmp_path):
    payload = _config_payload()
    payload["risk"]["dataset"]["unexpected"] = 1
    with pytest.raises(RiskConfigError):
        load_risk_config(_write(tmp_path, payload))


def test_missing_rules_file_raises(tmp_path):
    payload = _config_payload()
    payload["risk"]["rules_file"] = "configs/definitely-missing-rules.yaml"
    with pytest.raises(RiskConfigError, match="规则文件不存在"):
        load_risk_config(_write(tmp_path, payload))


def test_invalid_rules_config_raises(tmp_path):
    bad = tmp_path / "bad-rules.yaml"
    bad.write_text("rules:\n  version: r1\n  thresholds:\n    block_above: 2.0\n", encoding="utf-8")
    with pytest.raises(RiskConfigError, match="规则配置校验失败"):
        load_rule_settings(bad)


def test_rule_settings_defaults():
    rules = load_rule_settings(RISK_RULES_PATH)
    assert "eip7702_authorization" in rules.irreversible_actions
    assert "irreversible_action" in rules.enabled
    assert rules.thresholds.rolling_window_hours >= 1


def test_config_payload_copy_is_not_mutated(tmp_path):
    payload = copy.deepcopy(_config_payload())
    payload["risk"]["review"]["max_batch"] = 7
    config = load_risk_config(_write(tmp_path, payload))
    assert config.settings.review.max_batch == 7
    assert _config_payload()["risk"]["review"]["max_batch"] != 7
