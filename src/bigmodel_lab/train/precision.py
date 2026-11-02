"""精度、设备同步与显存采样——CUDA / Apple MPS / CPU 三后端统一入口。

设计目标：
- 训练代码只依赖本模块解析设备、精度与显存，不在业务代码里写死 `mps` 或 `bfloat16`；
- CUDA：Ampere 及以上（A10/A100/4090…）用 bf16 autocast；V100 等 sm70 老卡自动回落
  fp16 autocast + GradScaler（V100 不支持 bf16 运算）；
- Apple MPS：沿用原行为（bf16 → fp16 → fp32），不做梯度缩放；
- CPU：不启用 autocast，保证单元测试与无 GPU 环境可运行。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - 某些 CPU-only 构建会直接抛错
        return False


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def default_device() -> str:
    """优先级：CUDA → MPS → CPU。"""
    if cuda_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def device_type(device: str | None = None) -> str:
    """把 `cuda:0` 这类设备名归一化为 `cuda` / `mps` / `cpu`。"""
    name = (device or default_device()).strip()
    return name.split(":", 1)[0]


def resolve_device(device: str | None = None) -> str:
    """解析配置中的设备名；显式指定但当前机器不可用时回退 CPU。"""
    if not device:
        return default_device()
    kind = device_type(device)
    if kind == "cuda" and not cuda_available():
        return "cpu"
    if kind == "mps" and not mps_available():
        return "cpu"
    return device if kind != "cpu" else "cpu"


def device_name(device: str | None = None) -> str | None:
    kind = device_type(device)
    if kind == "cuda" and cuda_available():
        try:
            return torch.cuda.get_device_name(torch.device(kind))
        except Exception:  # noqa: BLE001
            return None
    if kind == "mps":
        return "Apple MPS"
    return None


def cuda_autocast_name(bf16_supported: bool) -> str:
    """CUDA 上按硬件能力选 autocast 精度。

    - Ampere+（A10/A100/4090 等）支持 bf16 → 用 bf16（无需梯度缩放）；
    - V100/T4 等 sm70/sm75 老卡不支持 bf16 → 回落 fp16（必须配 GradScaler）。
    """
    return "bfloat16" if bf16_supported else "float16"


def cuda_bf16_supported() -> bool:
    """CUDA GPU 是否**原生**支持 bf16。

    注意：torch ≥2.4 的 `torch.cuda.is_bf16_supported()` 默认包含「软件模拟」，
    会让 V100（sm70）也返回 True；但模拟路径很慢，训练应回落 fp16。
    这里显式 `including_emulation=False`，只在 Ampere+ 上返回 True。
    """
    if not cuda_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # pragma: no cover - torch < 2.4 没有该参数
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:  # noqa: BLE001
            return False
    except Exception:  # noqa: BLE001
        return False


def detect_autocast_dtype(device: str | None = None) -> tuple[Any, str]:
    """探测可用的最低精度：bf16 → fp16 → fp32（None 表示不做 autocast）。

    CUDA 上优先原生 bf16；不支持 bf16 的卡（如 V100 / sm70）回落 fp16，
    调用方需按同一 dtype 决定是否启用 GradScaler（见 :func:`make_grad_scaler`）。
    """
    device = resolve_device(device)
    kind = device_type(device)
    if kind == "cpu":
        return None, "float32"
    if kind == "cuda":
        choice = cuda_autocast_name(cuda_bf16_supported())
        dtype = torch.bfloat16 if choice == "bfloat16" else torch.float16
        try:
            probe = torch.randn(64, 64, device=device, dtype=dtype)
            _ = (probe @ probe).sum().item()
            sync(device)
            return dtype, choice
        except Exception:  # noqa: BLE001 - 尝试 fp32
            return None, "float32"
    for dtype, name in ((torch.bfloat16, "bfloat16"), (torch.float16, "float16")):
        try:
            probe = torch.randn(64, 64, device=device, dtype=dtype)
            _ = (probe @ probe).sum().item()
            sync(device)
            return dtype, name
        except Exception:  # noqa: BLE001 - 尝试下一种精度
            continue
    return None, "float32"


def default_dtype_name(device: str | None = None) -> str:
    """加载模型权重时的默认 dtype：CUDA 原生 bf16 可用则 bf16，否则 fp16；MPS bf16；CPU fp32。"""
    device = resolve_device(device)
    kind = device_type(device)
    if kind == "cuda":
        return "bfloat16" if cuda_bf16_supported() else "float16"
    if kind == "mps":
        return "bfloat16"
    return "float32"


def needs_grad_scaler(dtype: Any) -> bool:
    """只有 fp16 autocast 需要梯度缩放；bf16/fp32 不需要。"""
    return dtype is torch.float16


def make_grad_scaler(device: str | None = None, dtype: Any = None):
    """按设备/精度创建 GradScaler；不需要时返回 None。"""
    if not needs_grad_scaler(dtype) or device_type(device) != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # pragma: no cover - 老版本 torch 兜底
        return torch.cuda.amp.GradScaler()


def autocast_context(device: str, dtype: Any):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device_type(device), dtype=dtype)


def sync(device: str | None = None) -> None:
    kind = device_type(device)
    if kind == "cuda" and cuda_available():
        torch.cuda.synchronize()
    elif kind == "mps" and mps_available():
        torch.mps.synchronize()


def empty_cache(device: str | None = None) -> None:
    kind = device_type(device)
    if kind == "cuda" and cuda_available():
        torch.cuda.empty_cache()
    elif kind == "mps" and mps_available():
        torch.mps.empty_cache()


def reset_peak_memory(device: str | None = None) -> None:
    kind = device_type(device)
    if kind == "cuda" and cuda_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001 - CPU-only 或无统计支持
            pass


def allocated_memory_gb(device: str | None = None) -> float:
    kind = device_type(device)
    if kind == "cuda" and cuda_available():
        return round(torch.cuda.memory_allocated() / 1024**3, 3)
    if kind == "mps" and mps_available():
        return round(torch.mps.current_allocated_memory() / 1024**3, 3)
    return 0.0


def driver_memory_gb(device: str | None = None) -> float:
    """当前进程占用的显存/统一内存（CUDA 与 MPS 语义对齐）。"""
    kind = device_type(device)
    if kind == "cuda" and cuda_available():
        return round(torch.cuda.memory_allocated() / 1024**3, 3)
    if kind == "mps" and mps_available():
        return round(torch.mps.driver_allocated_memory() / 1024**3, 3)
    return 0.0


def peak_memory_gb(device: str | None = None) -> float:
    """自上次 reset 以来的峰值显存；MPS 没有峰值接口，返回当前驱动占用。"""
    kind = device_type(device)
    if kind == "cuda" and cuda_available():
        return round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    if kind == "mps" and mps_available():
        return round(torch.mps.driver_allocated_memory() / 1024**3, 3)
    return 0.0


def device_report(device: str | None = None) -> dict[str, Any]:
    """设备信息快照，写入 manifest/实验报告。"""
    resolved = resolve_device(device)
    kind = device_type(resolved)
    report: dict[str, Any] = {"device": resolved, "kind": kind, "backend": kind}
    if kind == "cuda" and cuda_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        report.update(
            {
                "name": properties.name,
                "capability": f"{properties.major}.{properties.minor}",
                "total_memory_gb": round(properties.total_memory / 1024**3, 1),
                "cuda_version": torch.version.cuda,
                "bf16_supported": cuda_bf16_supported(),
                "multi_processor_count": properties.multi_processor_count,
            }
        )
    elif kind == "mps":
        report.update({"name": "Apple MPS", "total_memory_gb": None})
    else:
        report.update({"name": "CPU"})
    return report
