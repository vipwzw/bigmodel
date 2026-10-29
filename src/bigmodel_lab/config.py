"""配置加载与校验。

约定：
- 默认配置位于 `configs/default.yaml`，阶段配置可覆盖其中任意字段。
- 多份 YAML 按顺序深合并，CLI 传入的 key=value 覆盖最后生效。
- 使用 pydantic 校验已知字段；非法值给出明确的字段级错误。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

PROFILE_CHOICES = ("sanity", "standard")


class ConfigError(RuntimeError):
    """配置加载或校验失败。"""


class NetworkConfig(BaseModel):
    """资源获取来源与代理设置。"""

    model_config = ConfigDict(extra="forbid")

    priority: list[Literal["cache", "modelscope", "hf-mirror", "hf"]] = Field(
        default_factory=lambda: ["cache", "modelscope", "hf-mirror", "hf"]
    )
    hf_endpoint: str | None = "https://hf-mirror.com"
    modelscope_endpoint: str = "https://www.modelscope.cn"
    proxy: str | None = None  # 例如 "http://127.0.0.1:1087"；None 表示直连
    timeout_seconds: float = 8.0


class LabConfig(BaseModel):
    """实验项目级配置（未知字段忽略，供阶段配置透传）。"""

    model_config = ConfigDict(extra="ignore")

    seed: int = 42
    profile: Literal["sanity", "standard"] = "standard"
    max_minutes: float | None = None
    dry_run: bool = False
    offline: bool = False
    data_dir: str = "data"
    models_dir: str = "models"
    runs_dir: str = "runs"
    network: NetworkConfig = Field(default_factory=NetworkConfig)

    def resolved_dir(self, field_name: Literal["data_dir", "models_dir", "runs_dir"]) -> Path:
        path = Path(getattr(self, field_name))
        return path if path.is_absolute() else PROJECT_ROOT / path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - 依赖 PyYAML 的错误类型
        raise ConfigError(f"配置文件 YAML 解析失败: {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射（dict）: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个映射，override 优先；返回新对象。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """应用 CLI 覆盖项；key 支持点号路径，例如 `network.proxy`。"""
    result = copy.deepcopy(data)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        cursor = result
        for part in parts[:-1]:
            node = cursor.get(part)
            if not isinstance(node, dict):
                node = {}
                cursor[part] = node
            cursor = node
        cursor[parts[-1]] = value
    return result


def load_config(
    *paths: str | Path,
    overrides: dict[str, Any] | None = None,
    use_default: bool = True,
) -> tuple[dict[str, Any], LabConfig]:
    """加载并合并配置。

    返回 (原始合并结果, 校验后的 LabConfig)。阶段专属字段保留在原始结果中。
    """
    raw: dict[str, Any] = _read_yaml(DEFAULT_CONFIG_PATH) if use_default else {}
    for path in paths:
        raw = deep_merge(raw, _read_yaml(Path(path)))
    if overrides:
        raw = apply_overrides(raw, overrides)
    try:
        config = LabConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
    return raw, config


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["配置校验失败:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {location}: {error['msg']} (输入值: {error.get('input')!r})")
    return "\n".join(lines)


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
