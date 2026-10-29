"""运行档位、时间预算与磁盘预检。"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

# 每阶段默认时间预算（分钟）；sanity 用于快速回归，standard 由命令行或配置控制
PROFILES: dict[str, dict[str, Any]] = {
    "sanity": {
        "max_minutes": 10.0,
        "description": "快速验证链路：小数据子集、少步数，全阶段 ≤30 分钟",
    },
    "standard": {
        "max_minutes": None,
        "description": "正式实验：按设计 token 预算与真实数据规模运行",
    },
}

DISK_RESERVE_GB = 5.0
DISK_WARN_GB = 30.0


class BudgetExceeded(RuntimeError):
    """运行超过时间预算。"""


@dataclass
class Budget:
    profile: str = "standard"
    max_minutes: float | None = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(f"未知 profile: {self.profile}，可选: {sorted(PROFILES)}")
        if self.max_minutes is None:
            self.max_minutes = PROFILES[self.profile]["max_minutes"]
        self._start = time.monotonic()

    @property
    def elapsed_minutes(self) -> float:
        return (time.monotonic() - self._start) / 60.0

    @property
    def remaining_minutes(self) -> float | None:
        if self.max_minutes is None:
            return None
        return max(0.0, self.max_minutes - self.elapsed_minutes)

    def check(self) -> None:
        """超时则抛出 BudgetExceeded（调用方负责优雅停止）。"""
        if self.max_minutes is not None and self.elapsed_minutes >= self.max_minutes:
            raise BudgetExceeded(
                f"运行超过时间预算 {self.max_minutes:.1f} 分钟（已用 {self.elapsed_minutes:.1f} 分钟）"
            )

    def scale(self, base: int | float, *, minimum: int = 1) -> int:
        """按档位缩放步数/样本数：sanity 约为标准的 5%，至少 minimum。"""
        if self.profile == "sanity":
            return max(minimum, int(base * 0.05))
        return int(base)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "max_minutes": self.max_minutes,
            "dry_run": self.dry_run,
            "elapsed_minutes": round(self.elapsed_minutes, 3),
        }


def disk_free_gb(path: str | Path = PROJECT_ROOT) -> float:
    return round(shutil.disk_usage(Path(path)).free / 1024**3, 1)


def check_disk(path: str | Path, required_gb: float, *, reserve_gb: float = DISK_RESERVE_GB) -> tuple[bool, str]:
    """下载/写入前检查磁盘余量。返回 (是否足够, 说明)。"""
    free = disk_free_gb(path)
    if required_gb <= 0:
        return True, f"无需额外磁盘（可用 {free} GB）"
    if free - required_gb < reserve_gb:
        return False, f"磁盘不足：需要约 {required_gb:.1f} GB，当前可用 {free} GB，保留余量 {reserve_gb} GB"
    return True, f"磁盘充足：需要约 {required_gb:.1f} GB，可用 {free} GB"
