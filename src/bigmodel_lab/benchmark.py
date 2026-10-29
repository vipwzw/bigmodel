"""Stage 1 性能基线：矩阵运算 GFLOPS、tiny GPT 训练步吞吐，并据此校准 token 预算。"""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Any

from .budget import Budget
from .models.gpt import PRESETS, STANDARD_TOKEN_BUDGET, GPT, GPTConfig

MATMUL_SIZE = 4096
SANITY_MATMUL_SIZE = 2048


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sync_mps() -> None:
    import torch  # noqa: PLC0415

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _mps_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        return False


def _mps_matmul_gflops(size: int, dtype_name: str, repeats: int = 8) -> dict[str, Any] | None:
    import torch  # noqa: PLC0415

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    try:
        a = torch.randn(size, size, device="mps", dtype=dtype)
        b = torch.randn(size, size, device="mps", dtype=dtype)
        _sync_mps()
        for _ in range(2):
            torch.mm(a, b)
        _sync_mps()
        started = time.perf_counter()
        for _ in range(repeats):
            torch.mm(a, b)
            _sync_mps()
        elapsed = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001 - 某些 dtype 在 MPS 上不可用
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


def _detect_autocast_dtype() -> tuple[Any, str]:
    """探测 MPS 上可用的最低精度（bf16 → fp16 → fp32）。"""
    import torch  # noqa: PLC0415

    for dtype, name in ((torch.bfloat16, "bfloat16"), (torch.float16, "float16")):
        try:
            x = torch.randn(64, 64, device="mps", dtype=dtype)
            _ = (x @ x).sum().item()
            _sync_mps()
            return dtype, name
        except Exception:  # noqa: BLE001 - 尝试下一种精度
            continue
    return None, "float32"


def _torch_train_step(
    preset: str,
    *,
    batch_size: int,
    steps: int,
    profile: str,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    config: GPTConfig = PRESETS[preset]
    model = GPT(config).to("mps")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    generator = torch.Generator(device="cpu").manual_seed(1234)
    idx = torch.randint(0, config.vocab_size, (batch_size, config.block_size), generator=generator).to("mps")
    targets = torch.randint(0, config.vocab_size, (batch_size, config.block_size), generator=generator).to("mps")
    autocast_dtype, dtype_name = _detect_autocast_dtype()
    context = (
        torch.autocast(device_type="mps", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    torch.mps.empty_cache()
    timings: list[float] = []
    burned = 0
    peak_driver = 0.0
    peak_allocated = 0.0
    try:
        for step_index in range(steps + 2):
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            with context:
                loss, _ = model(idx, targets)
            loss.backward()
            optimizer.step()
            _sync_mps()
            elapsed = time.perf_counter() - started
            peak_driver = max(peak_driver, torch.mps.driver_allocated_memory() / 1024**3)
            peak_allocated = max(peak_allocated, torch.mps.current_allocated_memory() / 1024**3)
            if step_index >= 2:  # 前两步预热
                timings.append(elapsed)
                burned += batch_size * config.block_size
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"出现非有限 loss: {loss.item()}")
        tokens_per_second = burned / sum(timings)
        final_loss = float(loss.item())
    finally:
        del model, optimizer
        torch.mps.empty_cache()
    return {
        "available": True,
        "preset": preset,
        "params": config.num_params(),
        "flops_per_token": config.flops_per_token(),
        "block_size": config.block_size,
        "batch_size": batch_size,
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
    if not _mps_available():
        raise RuntimeError("MPS 不可用，无法运行 Apple Silicon 性能基线")
    size = SANITY_MATMUL_SIZE if budget.profile == "sanity" else MATMUL_SIZE
    repeats = 4 if budget.profile == "sanity" else 8
    steps = 2 if budget.profile == "sanity" else 3
    batch_sizes = {"s": 2, "m": 2, "l": 1} if budget.profile == "sanity" else {"s": 8, "m": 4, "l": 2}

    matmul: dict[str, Any] = {}
    for backend, fn in (("mps", _mps_matmul_gflops), ("mlx", _mlx_matmul_gflops)):
        matmul[backend] = {}
        for dtype_name in ("float32", "float16", "bfloat16"):
            budget.check()
            matmul[backend][dtype_name] = fn(size, dtype_name, repeats)

    pretrain: dict[str, Any] = {}
    for preset, batch_size in batch_sizes.items():
        budget.check()
        try:
            pretrain[preset] = _torch_train_step(
                preset, batch_size=batch_size, steps=steps, profile=budget.profile
            )
        except Exception as exc:  # noqa: BLE001 - 记录失败但保留其余结果
            pretrain[preset] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    calibration = calibrate_budget(pretrain)
    best_mps = max(
        (item["gflops"] for item in matmul["mps"].values() if item and item.get("available")),
        default=None,
    )
    best_mlx = max(
        (item["gflops"] for item in matmul["mlx"].values() if item and item.get("available")),
        default=None,
    )
    return {
        "generated_at": _now_iso(),
        "profile": budget.profile,
        "matmul_size": size,
        "matmul": matmul,
        "matmul_best_gflops": {"mps": best_mps, "mlx": best_mlx},
        "pretrain_step": pretrain,
        "calibration": calibration,
        "notes": [
            "tokens/s 为 GPT 训练（前向+反向+优化器步）实测吞吐，已包含梯度同步开销",
            "校准耗时 = standard token 预算 / 实测 tokens/s；正式训练受数据加载、检查点写入影响可能更慢",
            "MLX 仅测矩阵运算基线；完整 MLX 训练对照在 Stage 4 实现",
        ],
    }


def to_markdown(result: dict[str, Any]) -> str:
    lines = ["# Stage 1 性能基线报告", ""]
    lines.append(f"- 生成时间：{result['generated_at']}")
    lines.append(f"- 档位：{result['profile']}")
    lines.append(f"- 矩阵规模：{result['matmul_size']}×{result['matmul_size']}")
    lines.append("")
    lines.append("## 矩阵运算 GFLOPS")
    lines.append("")
    lines.append("| 后端 | dtype | GFLOPS | 是否可用 |")
    lines.append("|---|---|---|---|")
    for backend, entries in result["matmul"].items():
        for dtype_name, entry in entries.items():
            if entry and entry.get("available"):
                lines.append(f"| {backend} | {dtype_name} | {entry['gflops']} | ✅ |")
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
