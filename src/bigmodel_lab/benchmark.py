"""Stage 1 性能基线：矩阵运算 GFLOPS、tiny GPT 训练步吞吐，并据此校准 token 预算。

CUDA 与 Apple MPS 共用同一套基线逻辑；MLX 仅在 macOS（Apple Silicon）上作为对照后端。
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

from .budget import Budget
from .models.gpt import GPT, PRESETS, STANDARD_TOKEN_BUDGET, GPTConfig
from .train.precision import (
    allocated_memory_gb,
    autocast_context,
    cuda_available,
    cuda_bf16_supported,
    default_device,
    detect_autocast_dtype,
    empty_cache,
    mps_available,
    peak_memory_gb,
    reset_peak_memory,
    sync,
)

MATMUL_SIZE = 4096
SANITY_MATMUL_SIZE = 2048


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def gpu_backend() -> str | None:
    """返回可用 GPU 后端：cuda → mps → None（CPU）。"""
    if cuda_available():
        return "cuda"
    if mps_available():
        return "mps"
    return None


def _torch_matmul_gflops(
    size: int, dtype_name: str, repeats: int = 8, *, device: str | None = None
) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    device = device or default_device()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    try:
        a = torch.randn(size, size, device=device, dtype=dtype)
        b = torch.randn(size, size, device=device, dtype=dtype)
        sync(device)
        for _ in range(2):
            torch.mm(a, b)
        sync(device)
        started = time.perf_counter()
        for _ in range(repeats):
            torch.mm(a, b)
            sync(device)
        elapsed = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001 - 某些 dtype 在旧 GPU（如 V100 的 bf16）上不可用
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    flops = 2 * size**3 * repeats
    return {
        "available": True,
        "size": size,
        "dtype": dtype_name,
        "repeats": repeats,
        "elapsed_seconds": round(elapsed, 4),
        "gflops": round(flops / elapsed / 1e9, 2),
    }


def _mlx_matmul_gflops(size: int, dtype_name: str, repeats: int = 8) -> dict[str, Any] | None:
    try:
        import mlx.core as mx  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"mlx 不可用: {exc}"}
    dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype_name]
    try:
        a = mx.random.normal((size, size)).astype(dtype)
        b = mx.random.normal((size, size)).astype(dtype)
        mx.eval(a, b)
        for _ in range(2):
            mx.eval(mx.matmul(a, b))
        started = time.perf_counter()
        for _ in range(repeats):
            mx.eval(mx.matmul(a, b))
        elapsed = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    flops = 2 * size**3 * repeats
    return {
        "available": True,
        "size": size,
        "dtype": dtype_name,
        "repeats": repeats,
        "elapsed_seconds": round(elapsed, 4),
        "gflops": round(flops / elapsed / 1e9, 2),
    }


def _torch_train_step(
    preset: str,
    *,
    batch_size: int,
    steps: int,
    profile: str,
    device: str | None = None,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    device = device or default_device()
    config: GPTConfig = PRESETS[preset]
    autocast_dtype, dtype_name = detect_autocast_dtype(device)
    model = GPT(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    generator = torch.Generator(device="cpu").manual_seed(1234)
    idx = torch.randint(0, config.vocab_size, (batch_size, config.block_size), generator=generator).to(device)
    targets = torch.randint(0, config.vocab_size, (batch_size, config.block_size), generator=generator).to(device)
    reset_peak_memory(device)
    empty_cache(device)
    timings: list[float] = []
    burned = 0
    peak_driver = 0.0
    peak_allocated = 0.0
    try:
        for step_index in range(steps + 2):
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            with autocast_context(device, autocast_dtype):
                loss, _ = model(idx, targets)
            loss.backward()
            optimizer.step()
            sync(device)
            elapsed = time.perf_counter() - started
            peak_driver = max(peak_driver, peak_memory_gb(device))
            peak_allocated = max(peak_allocated, allocated_memory_gb(device))
            if step_index >= 2:  # 前两步预热
                timings.append(elapsed)
                burned += batch_size * config.block_size
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"出现非有限 loss: {loss.item()}")
        tokens_per_second = burned / sum(timings)
        final_loss = float(loss.item())
    finally:
        del model, optimizer
        empty_cache(device)
    return {
        "available": True,
        "preset": preset,
        "params": config.num_params(),
        "flops_per_token": config.flops_per_token(),
        "block_size": config.block_size,
        "batch_size": batch_size,
        "device": device,
        "timed_steps": len(timings),
        "autocast_dtype": dtype_name,
        "tokens_per_second": round(tokens_per_second, 1),
        "step_seconds": [round(value, 4) for value in timings],
        "peak_allocated_gb": round(peak_allocated, 3),
        "peak_driver_gb": round(peak_driver, 3),
        "final_loss": round(final_loss, 4),
        "profile": profile,
    }


def calibrate_budget(throughput: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """用实测 tokens/s 校准 design 中各档 standard token 预算的预计耗时。"""
    calibration: dict[str, Any] = {}
    for preset, budget_tokens in STANDARD_TOKEN_BUDGET.items():
        measured = throughput.get(preset)
        if not measured or not measured.get("available"):
            calibration[preset] = {"available": False}
            continue
        tokens_per_second = measured["tokens_per_second"]
        hours = budget_tokens / tokens_per_second / 3600
        calibration[preset] = {
            "available": True,
            "standard_token_budget": budget_tokens,
            "estimated_hours": round(hours, 2),
            "measured_tokens_per_second": tokens_per_second,
        }
    return calibration


def run_benchmark(budget: Budget) -> dict[str, Any]:
    backend = gpu_backend()
    if backend is None:
        raise RuntimeError("未检测到 GPU（CUDA/MPS），无法运行性能基线")
    device = default_device()
    size = SANITY_MATMUL_SIZE if budget.profile == "sanity" else MATMUL_SIZE
    repeats = 4 if budget.profile == "sanity" else 8
    steps = 2 if budget.profile == "sanity" else 3
    batch_sizes = {"s": 2, "m": 2, "l": 1} if budget.profile == "sanity" else {"s": 8, "m": 4, "l": 2}

    matmul: dict[str, Any] = {backend: {}}
    for dtype_name in ("float32", "float16", "bfloat16"):
        budget.check()
        matmul[backend][dtype_name] = _torch_matmul_gflops(size, dtype_name, repeats, device=device)
    if mps_available():
        matmul["mlx"] = {}
        for dtype_name in ("float32", "float16", "bfloat16"):
            budget.check()
            matmul["mlx"][dtype_name] = _mlx_matmul_gflops(size, dtype_name, repeats)
    # V100 等无原生 bf16 的卡上，bf16 矩阵运算走软件模拟，标注出来避免误读
    if backend == "cuda" and not cuda_bf16_supported():
        bf16_entry = matmul[backend].get("bfloat16")
        if bf16_entry and bf16_entry.get("available"):
            bf16_entry["emulated"] = True
            bf16_entry["note"] = "软件模拟，训练会自动改用 fp16 + GradScaler"

    pretrain: dict[str, Any] = {}
    for preset, batch_size in batch_sizes.items():
        budget.check()
        try:
            pretrain[preset] = _torch_train_step(
                preset, batch_size=batch_size, steps=steps, profile=budget.profile, device=device
            )
        except Exception as exc:  # noqa: BLE001 - 记录失败但保留其余结果
            pretrain[preset] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    calibration = calibrate_budget(pretrain)
    best_matmul = {
        name: max(
            (entry["gflops"] for entry in entries.values() if entry and entry.get("available")),
            default=None,
        )
        for name, entries in matmul.items()
    }
    notes = [
        "tokens/s 为 GPT 训练（前向+反向+优化器步）实测吞吐，已包含梯度同步开销",
        "校准耗时 = standard token 预算 / 实测 tokens/s；正式训练受数据加载、检查点写入影响可能更慢",
        "CUDA 后端在 Ampere+ 用 bf16 autocast，V100 等 sm70 卡自动回落 fp16+梯度缩放；"
        "MLX 仅 macOS 可用，作为 Apple 原生对照",
    ]
    return {
        "generated_at": _now_iso(),
        "profile": budget.profile,
        "device": device,
        "device_name": _device_name(device),
        "matmul_size": size,
        "matmul": matmul,
        "matmul_best_gflops": best_matmul,
        "pretrain_step": pretrain,
        "calibration": calibration,
        "notes": notes,
    }


def _device_name(device: str) -> str:
    import torch  # noqa: PLC0415

    try:
        if device.startswith("cuda"):
            return torch.cuda.get_device_name(torch.device("cuda"))
        if device == "mps":
            return "Apple MPS"
    except Exception:  # noqa: BLE001
        return device
    return device


def to_markdown(result: dict[str, Any]) -> str:
    lines = ["# Stage 1 性能基线报告", ""]
    lines.append(f"- 生成时间：{result['generated_at']}")
    lines.append(f"- 档位：{result['profile']}")
    lines.append(f"- 设备：{result.get('device_name') or result.get('device')}")
    lines.append(f"- 矩阵规模：{result['matmul_size']}×{result['matmul_size']}")
    lines.append("")
    lines.append("## 矩阵运算 GFLOPS")
    lines.append("")
    lines.append("| 后端 | dtype | GFLOPS | 是否可用 |")
    lines.append("|---|---|---|---|")
    for backend, entries in result["matmul"].items():
        for dtype_name, entry in entries.items():
            if entry and entry.get("available"):
                note = f"（{entry['note']}）" if entry.get("note") else ""
                lines.append(f"| {backend} | {dtype_name} | {entry['gflops']} | ✅{note} |")
            else:
                reason = (entry or {}).get("reason", "未知原因")
                lines.append(f"| {backend} | {dtype_name} | - | ❌ {reason[:60]} |")
    lines.append("")
    lines.append("## tiny GPT 训练步吞吐与预算校准")
    lines.append("")
    lines.append("| 档位 | 参数量 | dtype | batch×ctx | tokens/s | 峰值显存(GB) | standard 预算耗时(h) |")
    lines.append("|---|---|---|---|---|---|---|")
    for preset, entry in result["pretrain_step"].items():
        calibration = result["calibration"].get(preset, {})
        if not entry.get("available"):
            lines.append(f"| {preset} | - | - | - | - | - | ❌ {entry.get('reason', '')[:40]} |")
            continue
        hours = calibration.get("estimated_hours", "-")
        lines.append(
            f"| {preset} | {entry['params'] / 1e6:.1f}M | {entry['autocast_dtype']} | "
            f"{entry['batch_size']}×{entry['block_size']} | {entry['tokens_per_second']:.0f} | "
            f"{entry['peak_driver_gb']:.2f} | {hours} |"
        )
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.extend(f"- {note}" for note in result["notes"])
    return "\n".join(lines)


def save_report(result: dict[str, Any], *, json_path, markdown_path) -> None:
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(to_markdown(result), encoding="utf-8")
