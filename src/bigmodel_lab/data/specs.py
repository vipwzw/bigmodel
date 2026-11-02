"""数据资源规格（由 `configs/default.yaml` 的 `data.resources` 驱动）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import ConfigError


@dataclass
class ResourceSpec:
    key: str
    repo_id: str
    files: list[str]
    language: str = "zh"
    license: str = "unknown"
    repo_type: str = "dataset"
    text_field: str | None = None
    source: str = "modelscope"
    approx_size_gb: float = 0.5
    notes: str = ""
    dest: str | None = None

    @classmethod
    def from_dict(cls, key: str, payload: dict[str, Any]) -> ResourceSpec:
        if not payload.get("repo_id"):
            raise ConfigError(f"data.resources.{key} 缺少 repo_id")
        files = payload.get("files") or []
        if not files:
            raise ConfigError(f"data.resources.{key} 缺少 files")
        return cls(
            key=key,
            repo_id=payload["repo_id"],
            files=list(files),
            language=payload.get("language", "zh"),
            license=payload.get("license", "unknown"),
            repo_type=payload.get("repo_type", "dataset"),
            text_field=payload.get("text_field"),
            source=payload.get("source", "modelscope"),
            approx_size_gb=float(payload.get("approx_size_gb", 0.5)),
            notes=payload.get("notes", ""),
            dest=payload.get("dest"),
        )


def load_resources(config: dict[str, Any]) -> dict[str, ResourceSpec]:
    data_config = config.get("data") or {}
    raw_resources = data_config.get("resources") or {}
    return {
        key: ResourceSpec.from_dict(key, value)
        for key, value in raw_resources.items()
    }


def data_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("data") or {}


def limit_for(key: str, config: dict[str, Any], profile: str) -> int | None:
    """按档位返回文档数上限；None 表示不限制。"""
    limits = (data_config(config).get("sanity_max_docs") or {})
    if profile == "sanity":
        value = limits.get(key)
        return int(value) if value else None
    return None


__all__ = ["ResourceSpec", "load_resources", "data_config", "limit_for"]
