"""资源获取：多源优先级、缓存复用、校验和、磁盘预检、代理支持。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..budget import Budget, check_disk
from ..config import PROJECT_ROOT, NetworkConfig
from .io import resolve_files
from .specs import ResourceSpec

MANIFEST_NAME = "fetch_manifest.json"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
_NO_PROXY = "localhost,127.0.0.1,.cn,modelscope.cn"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def proxy_environment(proxy: str | None):
    """临时设置 HTTP(S)_PROXY；对国内源保持直连（NO_PROXY）。"""
    if not proxy:
        yield
        return
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy")
    saved = {key: os.environ.get(key) for key in keys}
    os.environ.update({
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "ALL_PROXY": proxy,
        "NO_PROXY": _NO_PROXY,
        "http_proxy": proxy,
        "https_proxy": proxy,
        "all_proxy": proxy,
        "no_proxy": _NO_PROXY,
    })
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _download_modelscope(spec: ResourceSpec, dest: Path) -> None:
    from modelscope import snapshot_download  # noqa: PLC0415 - 延迟导入

    snapshot_download(
        repo_id=spec.repo_id,
        repo_type=spec.repo_type,
        local_dir=str(dest),
        allow_patterns=spec.files,
        max_workers=4,
    )


def _download_hf(spec: ResourceSpec, dest: Path, endpoint: str | None) -> None:
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - 延迟导入

    snapshot_download(
        repo_id=spec.repo_id,
        repo_type=spec.repo_type,
        local_dir=str(dest),
        allow_patterns=spec.files,
        endpoint=endpoint,
    )


def _write_manifest(dest: Path, spec: ResourceSpec, source: str, files: list[Path], *, verify_hashes: bool = True) -> dict[str, Any]:
    entries = []
    for path in files:
        entry = {
            "path": str(path.relative_to(dest)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path) if verify_hashes else None,
        }
        entries.append(entry)
    manifest = {
        "schema_version": 1,
        "key": spec.key,
        "repo_id": spec.repo_id,
        "repo_type": spec.repo_type,
        "source": source,
        "downloaded_at": _now_iso(),
        "language": spec.language,
        "license": spec.license,
        "files": entries,
    }
    tmp = dest / f"{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dest / MANIFEST_NAME)
    return manifest


def _load_manifest(dest: Path) -> dict[str, Any] | None:
    path = dest / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def manifest_complete(dest: Path, *, verify_hashes: bool = False) -> dict[str, Any] | None:
    """缓存命中判定：manifest 存在、文件齐全且大小一致（可选校验 hash）。"""
    manifest = _load_manifest(dest)
    if not manifest:
        return None
    for entry in manifest.get("files", []):
        path = dest / entry["path"]
        if not path.exists() or path.stat().st_size != entry["size"]:
            return None
        if verify_hashes and entry.get("sha256") and sha256_file(path) != entry["sha256"]:
            return None
    return manifest


def _source_order(spec: ResourceSpec, network: NetworkConfig) -> list[str]:
    order = [spec.source]
    order.extend(source for source in network.priority if source not in ("cache", spec.source))
    return [source for source in order if source in {"modelscope", "hf-mirror", "hf"}]


def fetch_resource(
    spec: ResourceSpec,
    *,
    data_dir: Path,
    network: NetworkConfig,
    offline: bool = False,
    budget: Budget | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """确保资源已缓存并返回获取记录；按优先级尝试多个来源。"""
    if spec.repo_type == "model" and spec.dest:
        dest = PROJECT_ROOT / spec.dest
    else:
        dest = data_dir / "raw" / spec.key
    dest.mkdir(parents=True, exist_ok=True)

    if not force:
        cached = manifest_complete(dest)
        if cached:
            return {"key": spec.key, "status": "cache_hit", "dest": str(dest), "source": cached.get("source"), "files": cached["files"]}

    if offline:
        raise RuntimeError(
            f"离线模式下资源 {spec.key}（{spec.repo_id}）尚未缓存于 {dest}；"
            f"请先联网运行一次 lab run 2 或执行 lab run 2 --offline=false"
        )

    if budget is not None:
        budget.check()
    ok, message = check_disk(dest.parent, spec.approx_size_gb)
    if not ok:
        raise RuntimeError(f"{spec.key} 下载前磁盘预检失败：{message}")

    errors: list[str] = []
    for source in _source_order(spec, network):
        try:
            if source == "modelscope":
                _download_modelscope(spec, dest)
            elif source == "hf-mirror":
                os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
                with proxy_environment(network.proxy):
                    _download_hf(spec, dest, endpoint=HF_MIRROR_ENDPOINT)
            else:  # 官方 HuggingFace，通常需要代理
                with proxy_environment(network.proxy):
                    _download_hf(spec, dest, endpoint=network.hf_endpoint if network.hf_endpoint and network.hf_endpoint != HF_MIRROR_ENDPOINT else None)
            files = resolve_files(dest, spec.files)
            manifest = _write_manifest(dest, spec, source, files)
            return {"key": spec.key, "status": "downloaded", "dest": str(dest), "source": source, "files": manifest["files"]}
        except Exception as exc:  # noqa: BLE001 - 逐个来源尝试
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
            if budget is not None:
                budget.check()
    raise RuntimeError(
        f"资源 {spec.key}（{spec.repo_id}）下载失败；已尝试来源: {'; '.join(errors) or '无'}"
    )


def fetch_resources(
    specs: list[ResourceSpec],
    *,
    data_dir: Path,
    network: NetworkConfig,
    offline: bool = False,
    budget: Budget | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for spec in specs:
        results[spec.key] = fetch_resource(spec, data_dir=data_dir, network=network, offline=offline, budget=budget)
    return results


def cleanup_manifest(dest: Path) -> None:
    manifest_path = dest / MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()


def copy_manifest(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_dir / MANIFEST_NAME, dst_dir / MANIFEST_NAME)
