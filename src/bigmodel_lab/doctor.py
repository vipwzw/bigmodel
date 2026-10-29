"""`lab doctor`：环境与资源来源自检。"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .config import PROJECT_ROOT, LabConfig, NetworkConfig

Status = Literal["PASS", "WARN", "FAIL"]

_STATUS_ORDER: dict[str, int] = {"PASS": 0, "WARN": 1, "FAIL": 2}

CORE_PACKAGES: dict[str, str] = {
    "yaml": "pyyaml",
    "pydantic": "pydantic",
    "numpy": "numpy",
}
TRAIN_PACKAGES: dict[str, str] = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "tokenizers": "tokenizers",
    "accelerate": "accelerate",
    "peft": "peft",
    "trl": "trl",
    "modelscope": "modelscope",
}
OPTIONAL_PACKAGES: dict[str, tuple[str, str]] = {
    "mlx": ("mlx", "MLX 对照实验（mlx 依赖组）"),
    "mlx_lm": ("mlx-lm", "MLX 量化推理（mlx 依赖组）"),
    "matplotlib": ("matplotlib", "曲线绘制（eval 依赖组）"),
    "fastapi": ("fastapi", "OpenAI 兼容服务（serve 依赖组）"),
    "uvicorn": ("uvicorn", "OpenAI 兼容服务（serve 依赖组）"),
}

MPS_MIN_FREE_GB = 10.0
DISK_WARN_GB = 30.0
MEMORY_WARN_GB = 24.0

PROBE_TARGETS: dict[str, str] = {
    "modelscope": "/api/v1/models/Qwen/Qwen3-0.6B",
    "hf-mirror": "/api/models/openai-community/gpt2",
    "huggingface": "/api/models/openai-community/gpt2",
}


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str
    fix: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _version_of(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python() -> CheckResult:
    version = platform.python_version()
    major, minor = (int(part) for part in version.split(".")[:2])
    if (major, minor) >= (3, 12):
        return CheckResult("python", "PASS", f"Python {version}")
    return CheckResult(
        "python", "FAIL", f"Python {version} 过旧（需要 ≥3.12）",
        fix="用 uv 创建 3.12 环境：uv venv --python 3.12 .venv && uv sync --all-groups",
    )


def check_packages() -> list[CheckResult]:
    results: list[CheckResult] = []
    missing_core = [dist for module, dist in CORE_PACKAGES.items() if not _importable(module)]
    if missing_core:
        results.append(CheckResult(
            "core-packages", "FAIL", f"缺少核心包: {', '.join(missing_core)}",
            fix="uv sync --all-groups",
        ))
    else:
        versions = ", ".join(f"{name} {_version_of(dist) or '?'}" for name, dist in CORE_PACKAGES.items())
        results.append(CheckResult("core-packages", "PASS", versions))

    missing_train = [dist for module, dist in TRAIN_PACKAGES.items() if not _importable(module)]
    if missing_train:
        results.append(CheckResult(
            "train-packages", "WARN", f"缺少训练依赖: {', '.join(missing_train)}",
            fix="uv sync --all-groups（训练阶段需要）",
        ))
    else:
        detail = ", ".join(
            f"{name} {_version_of(dist) or '?'}"
            for name, dist in list(TRAIN_PACKAGES.items())[:4]
        )
        results.append(CheckResult("train-packages", "PASS", detail + " …"))

    missing_optional = [
        f"{dist}（{purpose}）" for module, (dist, purpose) in OPTIONAL_PACKAGES.items() if not _importable(module)
    ]
    if missing_optional:
        results.append(CheckResult(
            "optional-packages", "WARN", f"缺少可选依赖: {'; '.join(missing_optional)}",
            fix="uv sync --all-groups",
        ))
    else:
        results.append(CheckResult("optional-packages", "PASS", "mlx / mlx-lm / matplotlib / fastapi 均已安装"))
    return results


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:  # noqa: BLE001 - import 失败原因不重要
        return False


def check_mps() -> CheckResult:
    if not _importable("torch"):
        return CheckResult(
            "mps", "FAIL", "未安装 torch，无法使用 Apple Silicon GPU",
            fix="uv sync --all-groups",
        )
    import torch  # noqa: PLC0415

    available = bool(torch.backends.mps.is_available())
    built = bool(torch.backends.mps.is_built())
    if available:
        return CheckResult(
            "mps", "PASS", f"torch {torch.__version__}，MPS 可用（built={built}）",
            data={"torch": torch.__version__, "built": built},
        )
    return CheckResult(
        "mps", "FAIL", f"torch {torch.__version__} 但 MPS 不可用（built={built}）",
        fix="确认 macOS ≥ 13 且安装的是 arm64 版 torch：python -c \"import torch; print(torch.__version__)\"",
    )


def check_mlx() -> CheckResult:
    if not _importable("mlx"):
        return CheckResult(
            "mlx", "WARN", "未安装 MLX（可选的 Apple 原生对照后端）",
            fix="uv sync --all-groups 或 uv sync --group mlx",
        )
    try:
        import mlx.core as mx  # noqa: PLC0415

        version = _version_of("mlx") or "unknown"
        metal = getattr(mx, "metal", None)
        return CheckResult(
            "mlx", "PASS", f"mlx {version}，metal={metal is not None}",
            data={"mlx": version},
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("mlx", "WARN", f"mlx 导入失败: {exc}", fix="uv sync --group mlx")


def check_memory() -> CheckResult:
    total_gb = _total_memory_gb()
    if total_gb is None:
        return CheckResult("memory", "WARN", "无法读取物理内存大小")
    if total_gb >= 32:
        return CheckResult("memory", "PASS", f"统一内存 {total_gb:.0f} GB", data={"total_memory_gb": total_gb})
    if total_gb >= MEMORY_WARN_GB:
        return CheckResult(
            "memory", "WARN", f"统一内存 {total_gb:.0f} GB，建议 32 GB 以上以运行 1B 级模型微调",
        )
    return CheckResult(
        "memory", "WARN", f"统一内存仅 {total_gb:.0f} GB，建议使用 sanity 档并跳过 1.7B 微调",
        fix="configs 中改用更小模型或减小 batch",
    )


def check_disk(path: Path = PROJECT_ROOT) -> CheckResult:
    free_gb = round(shutil.disk_usage(path).free / 1024**3, 1)
    if free_gb >= DISK_WARN_GB:
        return CheckResult("disk", "PASS", f"可用磁盘 {free_gb} GB", data={"free_gb": free_gb})
    if free_gb >= MPS_MIN_FREE_GB:
        return CheckResult("disk", "WARN", f"可用磁盘 {free_gb} GB，低于建议的 {DISK_WARN_GB:.0f} GB")
    return CheckResult(
        "disk", "FAIL", f"可用磁盘仅 {free_gb} GB，不足以缓存模型/数据",
        fix="清理空间或执行 lab clean",
    )


def _total_memory_gb() -> float | None:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return round(int(out.stdout.strip()) / 1024**3, 1)
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round(page_size * pages / 1024**3, 1)
    except (OSError, ValueError, subprocess.SubprocessError):  # pragma: no cover
        return None


def _http_probe(url: str, timeout: float, proxy: str | None = None) -> tuple[bool, str, float]:
    handlers: list[urllib.request.BaseHandler] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, headers={"User-Agent": "bigmodel-lab-doctor/0.1"})
    started = time.monotonic()
    try:
        with opener.open(request, timeout=timeout) as response:
            elapsed = time.monotonic() - started
            return True, f"HTTP {response.status}", elapsed
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        elapsed = time.monotonic() - started
        reason = getattr(exc, "reason", exc)
        return False, f"{type(exc).__name__}: {reason}", elapsed


def check_network(network: NetworkConfig, offline: bool = False) -> list[CheckResult]:
    if offline:
        return [CheckResult("network", "PASS", "离线模式：跳过网络探测")]
    results: list[CheckResult] = []
    timeout = network.timeout_seconds
    direct: dict[str, tuple[bool, str, float]] = {}

    endpoints = {
        "modelscope": network.modelscope_endpoint,
        "hf-mirror": "https://hf-mirror.com",
        "huggingface": "https://huggingface.co",
    }
    for name, base_url in endpoints.items():
        ok, detail, elapsed = _http_probe(base_url + PROBE_TARGETS[name], timeout)
        direct[name] = (ok, detail, elapsed)
        if ok:
            results.append(CheckResult(name, "PASS", f"{detail}（{elapsed * 1000:.0f} ms）", data={"direct": True}))
        elif name == "huggingface":
            # 官方 Hub 直连通常需要代理，先不判失败
            proxy_result = None
            if network.proxy:
                proxy_ok, proxy_detail, proxy_elapsed = _http_probe(
                    base_url + PROBE_TARGETS[name], timeout, proxy=network.proxy
                )
                proxy_result = (proxy_ok, proxy_detail, proxy_elapsed)
            if proxy_result and proxy_result[0]:
                results.append(CheckResult(
                    "huggingface", "PASS",
                    f"直连失败，经代理 {network.proxy} 可达（{proxy_result[2] * 1000:.0f} ms）",
                    data={"direct": False, "proxy": True},
                ))
            else:
                results.append(CheckResult(
                    "huggingface", "WARN",
                    f"直连不可达：{detail}",
                    fix="可选：在 configs/default.yaml 设置 network.proxy（如 http://127.0.0.1:1087）；"
                        "或继续使用 hf-mirror/ModelScope",
                    data={"direct": False, "proxy": False},
                ))
        else:
            results.append(
                CheckResult(name, "WARN", f"不可达：{detail}", data={"direct": False})
            )

    if network.proxy:
        proxy_ok, proxy_detail, proxy_elapsed = _http_probe(
            "https://huggingface.co/api/models/openai-community/gpt2",
            timeout,
            proxy=network.proxy,
        )
        results.append(CheckResult(
            "proxy", "PASS" if proxy_ok else "WARN",
            f"{network.proxy} → {'可达' if proxy_ok else proxy_detail}（{proxy_elapsed * 1000:.0f} ms）",
            fix=None if proxy_ok else "检查代理进程与端口",
        ))

    pypi_ok, pypi_detail, pypi_elapsed = _http_probe("https://pypi.org/simple/", timeout)
    mirror_ok, mirror_detail, mirror_elapsed = _http_probe(
        "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/", timeout
    )
    if pypi_ok or mirror_ok:
        detail = f"pypi.org {'可达' if pypi_ok else '不可达'}；清华镜像 {'可达' if mirror_ok else '不可达'}"
        results.append(CheckResult("pypi", "PASS", detail))
    else:
        results.append(CheckResult(
            "pypi", "WARN", f"两个 PyPI 源均不可达（{pypi_detail}）",
            fix="检查网络或配置代理后重试",
        ))
    return results


def run_checks(config: LabConfig, *, offline: bool = False) -> list[CheckResult]:
    results = [check_python(), *check_packages(), check_mps(), check_mlx(), check_memory(), check_disk()]
    results.extend(check_network(config.network, offline=offline))
    return results


def overall_status(results: list[CheckResult]) -> Status:
    worst = max((_STATUS_ORDER[result.status] for result in results), default=0)
    return ["PASS", "WARN", "FAIL"][worst]


def format_report(results: list[CheckResult]) -> str:
    lines = ["bigmodel-lab 环境自检", "=" * 60]
    for result in results:
        lines.append(f"[{result.status:4}] {result.name:<18} {result.detail}")
        if result.fix and result.status != "PASS":
            lines.append(f"       ↳ 修复建议: {result.fix}")
    status = overall_status(results)
    counts = {status_name: sum(1 for item in results if item.status == status_name) for status_name in _STATUS_ORDER}
    lines.append("=" * 60)
    lines.append(f"结论: {status}  (PASS {counts['PASS']} / WARN {counts['WARN']} / FAIL {counts['FAIL']})")
    return "\n".join(lines)


def report_json(results: list[CheckResult]) -> str:
    payload = {
        "status": overall_status(results),
        "checks": [result.to_dict() for result in results],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
