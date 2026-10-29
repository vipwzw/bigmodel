"""实验产物与可复现 manifest。

每次运行创建 `runs/<stage>/<run-id>/`，固定包含：
- config.yaml     本次运行的合并配置快照
- manifest.json   环境、代码版本、时间、状态、产物清单
- metrics.jsonl   逐步/逐评测指标（追加写入）
- train.log       日志
- artifacts/      阶段产物（报告、图、样例）
- checkpoints/    检查点
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, dump_yaml

MANIFEST_SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_info(root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        info["commit"] = commit.stdout.strip() or None
        branch = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        info["branch"] = branch.stdout.strip() or None
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        info["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 环境相关
        pass
    return info


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _total_memory_gb() -> float | None:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        return round(int(out.stdout.strip()) / 1024**3, 1)
    except (OSError, ValueError, subprocess.SubprocessError):  # pragma: no cover
        return None


def _environment_snapshot(root: Path) -> dict[str, Any]:
    versions = {
        pkg: _package_version(pkg)
        for pkg in (
            "torch", "transformers", "datasets", "tokenizers", "peft", "trl",
            "accelerate", "mlx", "mlx-lm", "modelscope", "numpy", "pydantic",
        )
    }
    snapshot: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "total_memory_gb": _total_memory_gb(),
        "disk_free_gb": round(shutil.disk_usage(root).free / 1024**3, 1),
        "packages": versions,
    }
    try:
        import torch  # noqa: PLC0415 - 仅环境快照需要

        snapshot["mps_available"] = bool(torch.backends.mps.is_available())
        snapshot["torch_mps_built"] = bool(torch.backends.mps.is_built())
    except Exception:  # noqa: BLE001 - torch 可能未安装或初始化失败
        snapshot["mps_available"] = None
        snapshot["torch_mps_built"] = None
    return snapshot


_TRACKED_CODE_PATTERNS = ("src", "configs", "scripts", "pyproject.toml", "uv.lock")


def code_fingerprint(root: Path = PROJECT_ROOT) -> str:
    """对参与实验的代码与配置做内容哈希；忽略缓存、资源与运行产物。"""
    digest = hashlib.sha256()
    tracked: list[Path] = []
    for pattern in _TRACKED_CODE_PATTERNS:
        target = root / pattern
        if target.is_file():
            tracked.append(target)
        elif target.is_dir():
            tracked.extend(
                path
                for path in sorted(target.rglob("*"))
                if path.is_file()
                and "__pycache__" not in path.parts
                and not path.name.endswith((".pyc", ".pyo"))
            )
    for path in sorted(tracked):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class RunContext:
    """一次实验运行的上下文与产物写入器。"""

    def __init__(self, stage: str, run_dir: Path, config: dict[str, Any]):
        self.stage = stage
        self.run_dir = run_dir
        self.config = config
        self.artifacts_dir = run_dir / "artifacts"
        self.checkpoints_dir = run_dir / "checkpoints"
        self.metrics_path = run_dir / "metrics.jsonl"
        self.log_path = run_dir / "train.log"
        self.manifest_path = run_dir / "manifest.json"
        self._artifacts: list[str] = []
        self._metrics_count = 0
        self._started_wall = time.time()
        self._started_mono = time.monotonic()
        self._manifest: dict[str, Any] = {}

    # ---------- 构造 ----------
    @classmethod
    def create(
        cls,
        stage: str,
        config: dict[str, Any],
        *,
        root: Path = PROJECT_ROOT,
        runs_dir: Path | None = None,
        run_id: str | None = None,
    ) -> RunContext:
        runs_dir = runs_dir or root / str(config.get("runs_dir", "runs"))
        stage_dir = runs_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{config_fingerprint(config)[:8]}"
        run_dir = stage_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        ctx = cls(stage=stage, run_dir=run_dir, config=config)
        ctx.artifacts_dir.mkdir(exist_ok=True)
        ctx.checkpoints_dir.mkdir(exist_ok=True)
        ctx._write_snapshot()
        return ctx

    # ---------- 写入 ----------
    def _write_snapshot(self) -> None:
        _atomic_write_text(self.run_dir / "config.yaml", dump_yaml(self.config))
        code = code_fingerprint()
        self._manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_dir.name,
            "stage": self.stage,
            "status": "running",
            "created_at": _now_iso(),
            "finished_at": None,
            "duration_seconds": None,
            "exit_code": None,
            "error": None,
            "seed": self.config.get("seed"),
            "profile": self.config.get("profile"),
            "config_fingerprint": config_fingerprint(self.config),
            "offline": bool(self.config.get("offline", False)),
            "network": self.config.get("network"),
            "code": {"fingerprint_sha256": code, **_git_info(PROJECT_ROOT)},
            "environment": _environment_snapshot(PROJECT_ROOT),
            "paths": {
                "run_dir": str(self.run_dir),
                "config": "config.yaml",
                "metrics": "metrics.jsonl",
                "log": "train.log",
                "artifacts": "artifacts",
                "checkpoints": "checkpoints",
            },
            "artifacts": [],
            "metrics_records": 0,
        }
        self._flush_manifest()

    def _flush_manifest(self) -> None:
        self._manifest["artifacts"] = list(self._artifacts)
        self._manifest["metrics_records"] = self._metrics_count
        _atomic_write_text(self.manifest_path, json.dumps(self._manifest, ensure_ascii=False, indent=2, default=str))

    # ---------- 指标与产物 ----------
    def log_metrics(self, step: int | None = None, **metrics: Any) -> None:
        record = {"time": _now_iso(), "stage": self.stage, "step": step, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._metrics_count += 1

    def add_artifact(self, path: str | Path) -> Path:
        artifact = Path(path)
        if artifact.is_absolute():
            relative = artifact.relative_to(self.run_dir) if artifact.is_relative_to(self.run_dir) else artifact
        else:
            relative = artifact
        entry = str(relative)
        if entry not in self._artifacts:
            self._artifacts.append(entry)
            self._flush_manifest()
        return artifact

    def artifact_path(self, name: str) -> Path:
        path = self.artifacts_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_artifact(self, name: str, text: str) -> Path:
        path = self.artifact_path(name)
        _atomic_write_text(path, text)
        return self.add_artifact(path)

    def write_json_artifact(self, name: str, data: Any) -> Path:
        return self.write_artifact(name, json.dumps(data, ensure_ascii=False, indent=2, default=str))

    # ---------- 结束 ----------
    def finish(self, status: str, exit_code: int = 0, error: str | None = None) -> dict[str, Any]:
        self._manifest["status"] = status
        self._manifest["exit_code"] = exit_code
        self._manifest["error"] = error
        self._manifest["finished_at"] = _now_iso()
        self._manifest["duration_seconds"] = round(time.monotonic() - self._started_mono, 3)
        self._flush_manifest()
        return self._manifest

    def __enter__(self) -> RunContext:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001 - contextlib 协议
        if self._manifest.get("status") == "running":
            if exc is None:
                self.finish("success", 0)
            else:
                self.finish("failed", 1, f"{exc_type.__name__}: {exc}")


def latest_run(stage: str, runs_dir: Path | None = None) -> Path | None:
    base = (runs_dir or PROJECT_ROOT / "runs") / stage
    if not base.exists():
        return None
    candidates = sorted((path for path in base.iterdir() if path.is_dir()), key=lambda p: p.name)
    return candidates[-1] if candidates else None


def list_runs(stage: str | None = None, runs_dir: Path | None = None) -> list[Path]:
    base = runs_dir or PROJECT_ROOT / "runs"
    if not base.exists():
        return []
    if stage:
        dirs = base / stage
        return sorted(path for path in dirs.iterdir() if path.is_dir()) if dirs.exists() else []
    return sorted(path for path in base.rglob("manifest.json"))
