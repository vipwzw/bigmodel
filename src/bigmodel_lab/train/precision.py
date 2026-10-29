"""精度、设备同步与显存采样。"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def mps_available() -> bool:
    return bool(torch.backends.mps.is_available())


def default_device() -> str:
    return "mps" if mps_available() else "cpu"


def detect_autocast_dtype(device: str = "mps") -> tuple[Any, str]:
    """探测可用的最低精度：bf16 → fp16 → fp32（None 表示不做 autocast）。"""
    if device != "mps":
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


def autocast_context(device: str, dtype: Any):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device, dtype=dtype)


def sync(device: str) -> None:
    if device == "mps" and mps_available():
        torch.mps.synchronize()


def empty_cache(device: str) -> None:
    if device == "mps" and mps_available():
        torch.mps.empty_cache()


def allocated_memory_gb(device: str) -> float:
    if device == "mps" and mps_available():
        return round(torch.mps.current_allocated_memory() / 1024**3, 3)
    return 0.0


def driver_memory_gb(device: str) -> float:
    if device == "mps" and mps_available():
        return round(torch.mps.driver_allocated_memory() / 1024**3, 3)
    return 0.0
