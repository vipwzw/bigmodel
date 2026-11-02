#!/usr/bin/env python
"""GPU 计算性能基准：矩阵乘 GFLOPS/TFLOPS、显存带宽与设备信息。

面向远程 CUDA 环境（如 NVIDIA A10），也支持 --device mps/cpu 在无 GPU 的机器上自检。
只依赖 PyTorch，不需要 nvcc / 编译工具链。

测什么：
  1. 设备信息：torch/CUDA 版本、设备名、计算能力、SM 数、显存、nvidia-smi 实时频率/功耗/温度；
  2. 矩阵乘：n×n 方阵乘，分别测 fp32（关闭 TF32）、tf32、fp16、bf16 的最佳/平均 TFLOPS；
     对 A10 会附官方参考峰值与达成率；
  3. 显存带宽：copy_（读 1 写 1）、add(out=)（读 2 写 1）、sum（只读）三种访问模式的 GB/s。

计时方式：CUDA 上用 cuda event，其它设备用墙钟 + synchronize；每个用例先预热，再重复计时取最佳值。

用法：
    python scripts/bench_gpu.py                       # 自动选设备（cuda > mps > cpu），n=8192
    python scripts/bench_gpu.py --size 4096 --reps 20
    python scripts/bench_gpu.py --gpu 1 --json ~/a10.json
    python scripts/bench_gpu.py --device cpu --size 1024 --reps 3   # 本地自检
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import torch

# NVIDIA A10 官方规格（NVIDIA / PNY A10 product brief；带 * 的稀疏算力不参与对比）
_REFERENCE_SPECS: dict[str, dict[str, float]] = {
    "a10": {
        "fp32_tflops": 31.2,
        "tf32_tflops": 62.5,
        "fp16_tflops": 165.0,
        "bf16_tflops": 126.0,
        "bandwidth_gbs": 600.0,
    },
}

# 需要测量的精度；tf32 只在 CUDA 上有效
_MATMUL_PRECISIONS = ("fp32", "tf32", "fp16", "bf16")
_TORCH_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_PRECISION_LABELS: dict[str, str] = {
    "fp32": "fp32 (TF32 关)",
    "tf32": "tf32 (TF32 开)",
    "fp16": "fp16 (Tensor Core)",
    "bf16": "bf16 (Tensor Core)",
}

_SMI_FIELDS_FULL = (
    "name,driver_version,memory.total,memory.used,"
    "clocks.current.sm,clocks.current.memory,power.draw,power.limit,"
    "temperature.gpu,utilization.gpu"
)
_SMI_FIELDS_BASIC = "name,driver_version,memory.total,memory.used"


# --------------------------------------------------------------------------- #
# 设备与计时工具
# --------------------------------------------------------------------------- #


def resolve_device(name: str, gpu_index: int) -> torch.device:
    """把 --device 解析为 torch.device，并检查可用性。"""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", gpu_index)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("错误：torch.cuda 不可用（没有 CUDA 设备或安装的是 CPU 版 torch）")
        return torch.device("cuda", gpu_index)
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise SystemExit("错误：MPS 后端不可用")
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """等待设备上已提交的计算全部完成。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """CUDA 用 cuda event 计时（GPU 时间），其它设备用墙钟计时。"""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.use_event = device.type == "cuda"

    def start(self) -> Any:
        if self.use_event:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, token: Any) -> float:
        """返回从 start 到 stop 的秒数。"""
        if self.use_event:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            return token.elapsed_time(end) / 1e3
        sync_device(self.device)
        return time.perf_counter() - token


def timed(fn: partial, timer: Timer, warmup: int, reps: int) -> list[float]:
    """预热 warmup 次后连续计时 reps 次，返回每次耗时（秒）。"""
    for _ in range(warmup):
        fn()
    sync_device(timer.device)
    times: list[float] = []
    for _ in range(reps):
        token = timer.start()
        fn()
        times.append(timer.stop(token))
    return times


# --------------------------------------------------------------------------- #
# 设备信息
# --------------------------------------------------------------------------- #


def _run_nvidia_smi(gpu_index: int, fields: str) -> dict[str, str] | None:
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    first = proc.stdout.strip().splitlines()
    if not first:
        return None
    values = [part.strip() for part in first[0].split(",")]
    keys = [part.strip() for part in fields.split(",")]
    if len(values) != len(keys):
        return None
    return dict(zip(keys, values, strict=True))


def query_nvidia_smi(gpu_index: int) -> dict[str, str] | None:
    """读取实时频率/功耗/温度；不支持的字段自动降级，失败返回 None。"""
    return _run_nvidia_smi(gpu_index, _SMI_FIELDS_FULL) or _run_nvidia_smi(gpu_index, _SMI_FIELDS_BASIC)


def _compute_capability_name(major: int, minor: int) -> str:
    names = {
        (7, 0): "Volta",
        (7, 5): "Turing",
        (8, 0): "Ampere",
        (8, 6): "Ampere",
        (8, 9): "Ada Lovelace",
        (9, 0): "Hopper",
        (10, 0): "Blackwell",
        (12, 0): "Blackwell",
    }
    return names.get((major, minor), f"sm_{major}{minor}")


def collect_device_info(device: torch.device, gpu_index: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.machine()}",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": None,
        "device_type": device.type,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.backends.cudnn.is_available():
        info["cudnn"] = torch.backends.cudnn.version()

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(gpu_index)
        total_mem = getattr(props, "total_memory", 0)
        l2_cache = getattr(props, "L2_cache_size", 0)
        info.update(
            {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "architecture": _compute_capability_name(props.major, props.minor),
                "sm_count": getattr(props, "multi_processor_count", None),
                "total_memory_gb": round(total_mem / 2**30, 2) if total_mem else None,
                "l2_cache_mb": round(l2_cache / 2**20, 2) if l2_cache else None,
                "tf32_allowed_default": torch.backends.cuda.matmul.allow_tf32,
                "nvidia_smi": query_nvidia_smi(gpu_index),
            }
        )
    elif device.type == "mps":
        info["name"] = "Apple GPU (MPS)"
    else:
        info["name"] = f"CPU ({platform.machine()})"

    info["reference"] = _lookup_reference(str(info["name"]))
    return info


def _lookup_reference(gpu_name: str) -> dict[str, float] | None:
    """按设备名匹配参考规格；用词边界避免 A10 误匹配 A100。"""
    low = gpu_name.lower()
    for key, spec in _REFERENCE_SPECS.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            return spec
    return None


# --------------------------------------------------------------------------- #
# 基准项
# --------------------------------------------------------------------------- #


def _set_tf32(enabled: bool) -> None:
    """切换 CUDA 上 fp32 矩阵乘是否允许 TF32。"""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = enabled
    # 新版 PyTorch（>= 2.9）改用 fp32_precision 控制，存在时一并设置
    fp32_precision = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    if fp32_precision is not None:
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
        except (RuntimeError, ValueError):
            pass


def bench_matmul(
    device: torch.device,
    timer: Timer,
    size: int,
    reps: int,
    warmup: int,
) -> list[dict[str, Any]]:
    """测量 n×n 方阵乘的 TFLOPS（每次矩阵乘 FLOPs = 2·n³）。"""
    flops = 2.0 * size**3
    results: list[dict[str, Any]] = []
    original_tf32 = torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None

    try:
        for precision in _MATMUL_PRECISIONS:
            result: dict[str, Any] = {
                "precision": precision,
                "label": _PRECISION_LABELS[precision],
                "size": size,
                "flops_per_call": flops,
                "skipped": None,
                "tflops_best": None,
                "tflops_avg": None,
            }
            if precision == "tf32" and device.type != "cuda":
                result["skipped"] = "TF32 仅 CUDA 支持"
                results.append(result)
                continue
            if precision in ("fp16", "bf16") and device.type == "cpu":
                result["skipped"] = "CPU 无 Tensor Core，跳过"
                results.append(result)
                continue

            dtype = _TORCH_DTYPES[precision]
            tensors: tuple[torch.Tensor, torch.Tensor] | None = None
            try:
                if device.type == "cuda":
                    _set_tf32(precision == "tf32")
                a = torch.randn(size, size, device=device, dtype=dtype)
                b = torch.randn(size, size, device=device, dtype=dtype)
                tensors = (a, b)
                times = timed(partial(torch.matmul, a, b), timer, warmup, reps)
            except (RuntimeError, TypeError, NotImplementedError) as exc:
                result["skipped"] = f"{type(exc).__name__}: {exc}"
            else:
                result["tflops_best"] = flops / min(times) / 1e12
                result["tflops_avg"] = flops / (sum(times) / len(times)) / 1e12
                result["ms_best"] = min(times) * 1e3
            finally:
                if tensors is not None:
                    del tensors
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            results.append(result)
    finally:
        if original_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = original_tf32
            _set_tf32(original_tf32)
    return results


def bench_bandwidth(
    device: torch.device,
    timer: Timer,
    target_mb: int,
    reps: int,
    warmup: int,
) -> list[dict[str, Any]]:
    """用 1D float32 张量测显存带宽（GB/s = 字节数 / 秒，按 1e9 计）。"""
    count = max(1, target_mb * 1024 * 1024 // 4)
    a = torch.randn(count, device=device, dtype=torch.float32)
    b = torch.randn(count, device=device, dtype=torch.float32)
    out = torch.empty_like(a)

    # (名称, 可调用对象, 每次调用搬运字节数)
    cases: list[tuple[str, partial, int]] = [
        ("copy_ (读1写1)", partial(out.copy_, a), 2 * count * 4),
        ("add out= (读2写1)", partial(torch.add, a, b, out=out), 3 * count * 4),
        ("sum (只读)", partial(torch.sum, a), count * 4),
    ]
    results: list[dict[str, Any]] = []
    for label, fn, nbytes in cases:
        try:
            times = timed(fn, timer, warmup, reps)
        except (RuntimeError, NotImplementedError) as exc:
            results.append({"case": label, "skipped": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(
            {
                "case": label,
                "bytes_per_call": nbytes,
                "gbs_best": nbytes / min(times) / 1e9,
                "gbs_avg": nbytes / (sum(times) / len(times)) / 1e9,
                "ms_best": min(times) * 1e3,
            }
        )
    del a, b, out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def _efficiency(measured: float | None, reference: float | None) -> str:
    if measured is None or not reference:
        return "-"
    return f"{measured / reference * 100:.1f}%"


def _display_width(text: str) -> int:
    """按终端显示宽度计算长度（中日韩全角字符按 2 列算）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, *, align: str = "left") -> str:
    """按显示宽度补空格，保证中英混排的表格对齐。"""
    spaces = " " * max(0, width - _display_width(text))
    return text + spaces if align == "left" else spaces + text


def print_device_info(info: dict[str, Any]) -> None:
    print("=" * 78)
    print(" GPU 性能基准")
    print("=" * 78)
    print("[设备]")
    print(f"  主机            : {info['hostname']}  ({info['platform']})")
    print(
        f"  PyTorch / CUDA  : {info['torch']} / {info['cuda_runtime'] or '无'}  (cuDNN: {info['cudnn'] or '无'})"
    )
    print(f"  设备            : {info['name']}")
    if info["device_type"] == "cuda":
        print(f"  计算能力        : {info['compute_capability']} ({info['architecture']})")
        print(
            f"  SM 数量         : {info['sm_count']}    显存: {info['total_memory_gb']} GB    L2: {info['l2_cache_mb']} MB"
        )
        print(f"  TF32 默认       : {'开' if info.get('tf32_allowed_default') else '关'}")
        smi = info.get("nvidia_smi") or {}
        if smi:
            print(
                "  nvidia-smi      : "
                f"驱动 {smi.get('driver_version', '?')} | SM {smi.get('clocks.current.sm', '?')} MHz"
                f" | 显存 {smi.get('clocks.current.memory', '?')} MHz"
                f" | 功耗 {smi.get('power.draw', '?')}/{smi.get('power.limit', '?')} W"
                f" | {smi.get('temperature.gpu', '?')}°C"
                f" | 利用率 {smi.get('utilization.gpu', '?')}%"
            )
    reference = info.get("reference")
    if reference:
        print(
            "  参考规格        : "
            f"FP32 {reference['fp32_tflops']} | TF32 {reference['tf32_tflops']} | "
            f"FP16 {reference['fp16_tflops']} | BF16 {reference['bf16_tflops']} TFLOPS, "
            f"带宽 {reference['bandwidth_gbs']} GB/s"
        )
    print()


def print_matmul_results(results: list[dict[str, Any]], reference: dict[str, float] | None) -> None:
    size = results[0]["size"]
    flops = results[0]["flops_per_call"]
    print(f"[矩阵乘] n={size}，单次 {flops / 1e12:.3f} TFLOP（=2n³）")
    header = (
        _pad("精度", 20)
        + _pad("最佳 TFLOPS", 14, align="right")
        + _pad("平均 TFLOPS", 14, align="right")
        + _pad("GFLOPS(最佳)", 14, align="right")
        + _pad("参考峰值", 10, align="right")
        + _pad("达成率", 9, align="right")
    )
    print("  " + header)
    for item in results:
        if item.get("skipped"):
            print("  " + _pad(item["label"], 20) + item["skipped"])
            continue
        ref_value = reference.get(f"{item['precision']}_tflops") if reference else None
        row = (
            _pad(item["label"], 20)
            + _pad(f"{item['tflops_best']:.2f}", 14, align="right")
            + _pad(f"{item['tflops_avg']:.2f}", 14, align="right")
            + _pad(f"{item['tflops_best'] * 1e3:.0f}", 14, align="right")
            + _pad(f"{ref_value:.1f}" if ref_value else "-", 10, align="right")
            + _pad(_efficiency(item["tflops_best"], ref_value), 9, align="right")
        )
        print("  " + row)
    print()


def print_bandwidth_results(results: list[dict[str, Any]], reference: dict[str, float] | None) -> None:
    print("[显存带宽] 1D fp32 张量，单次搬运量见下")
    header = (
        _pad("操作", 20)
        + _pad("最佳 GB/s", 12, align="right")
        + _pad("平均 GB/s", 12, align="right")
        + _pad("参考峰值", 10, align="right")
        + _pad("达成率", 9, align="right")
    )
    print("  " + header)
    ref_bw = reference.get("bandwidth_gbs") if reference else None
    for item in results:
        if item.get("skipped"):
            print("  " + _pad(item["case"], 20) + item["skipped"])
            continue
        row = (
            _pad(item["case"], 20)
            + _pad(f"{item['gbs_best']:.1f}", 12, align="right")
            + _pad(f"{item['gbs_avg']:.1f}", 12, align="right")
            + _pad(f"{ref_bw:.0f}" if ref_bw else "-", 10, align="right")
            + _pad(_efficiency(item["gbs_best"], ref_bw), 9, align="right")
        )
        print("  " + row)
    print()


def print_summary(
    info: dict[str, Any], matmul: list[dict[str, Any]], bandwidth: list[dict[str, Any]]
) -> None:
    reference = info.get("reference")
    parts: list[str] = []
    for item in matmul:
        if item.get("skipped") is None and item.get("tflops_best"):
            parts.append(
                f"{item['precision']} {item['tflops_best']:.2f} TFLOPS"
                f" ({item['tflops_best'] * 1e3:.0f} GFLOPS)"
            )
    best_bw = max((item["gbs_best"] for item in bandwidth if item.get("gbs_best")), default=None)
    print("[汇总]")
    print("  " + " | ".join(parts))
    if best_bw is not None:
        line = f"  显存带宽 {best_bw:.1f} GB/s"
        if reference:
            line += f"（参考 {reference['bandwidth_gbs']:.0f} GB/s，达成率 {best_bw / reference['bandwidth_gbs'] * 100:.1f}%）"
        print(line)
    print()


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_gpu.py",
        description="GPU 计算性能基准（矩阵乘 TFLOPS/GFLOPS、显存带宽、设备信息）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto", help="目标设备")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA 设备序号")
    parser.add_argument("--size", type=int, default=8192, help="矩阵乘方阵边长 n")
    parser.add_argument("--reps", type=int, default=10, help="每个用例计时次数（取最佳）")
    parser.add_argument("--warmup", type=int, default=3, help="每个用例预热次数")
    parser.add_argument("--bandwidth-mb", type=int, default=512, help="带宽测试张量大小（MB）")
    parser.add_argument("--skip-bandwidth", action="store_true", help="跳过显存带宽测试")
    parser.add_argument("--json", type=Path, default=None, help="把结果写入 JSON 文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.size < 256:
        raise SystemExit("错误：--size 至少为 256")
    if args.reps < 1 or args.warmup < 0:
        raise SystemExit("错误：--reps 需 >=1，--warmup 需 >=0")

    device = resolve_device(args.device, args.gpu)
    timer = Timer(device)
    info = collect_device_info(device, args.gpu)
    print_device_info(info)

    matmul = bench_matmul(device, timer, args.size, args.reps, args.warmup)
    print_matmul_results(matmul, info.get("reference"))

    bandwidth: list[dict[str, Any]] = []
    if not args.skip_bandwidth:
        bandwidth = bench_bandwidth(device, timer, args.bandwidth_mb, args.reps, args.warmup)
        print_bandwidth_results(bandwidth, info.get("reference"))

    print_summary(info, matmul, bandwidth)

    if args.json is not None:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv[1:],
            "device": info,
            "matmul": matmul,
            "bandwidth": bandwidth,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
