"""Stage 4：从零预训练（S/M/L 三档）、MLX 数值与吞吐对照、规模曲线。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..budget import Budget
from ..config import LabConfig
from ..manifest import RunContext
from ..models.gpt import GPT, PRESETS, GPTConfig
from ..models.mlx_gpt import MLX_AVAILABLE, MLXGPT, load_from_torch, mlx_throughput, parity_check
from ..train.precision import default_device
from ..train.pretrain import PretrainConfig, Pretrainer

logger = logging.getLogger(__name__)


def _stage_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("stage4") or {}


def _pretrain_config(preset_name: str, preset_payload: dict[str, Any], *, profile: str, seed: int, device: str) -> PretrainConfig:
    max_tokens_budget = preset_payload.get("max_tokens") or {}
    if isinstance(max_tokens_budget, dict):
        max_tokens = int(max_tokens_budget.get(profile, max_tokens_budget.get("sanity", 1_000_000)))
    else:
        max_tokens = int(max_tokens_budget)
    return PretrainConfig(
        preset=preset_name,
        data_key=str(preset_payload.get("data_key")),
        batch_size=int(preset_payload.get("batch_size", 8)),
        grad_accum=int(preset_payload.get("grad_accum", 1)),
        lr=float(preset_payload.get("lr", 6e-4)),
        min_lr_ratio=float(preset_payload.get("min_lr_ratio", 0.1)),
        warmup_steps=int(preset_payload.get("warmup_steps", 100)),
        max_tokens=max_tokens,
        grad_clip=float(preset_payload.get("grad_clip", 1.0)),
        seed=seed,
        grad_checkpoint=bool(preset_payload.get("grad_checkpoint", False)),
        log_every=int(preset_payload.get("log_every", 10)),
        checkpoint_every=int(preset_payload.get("checkpoint_every", 200)),
        keep_last=int(preset_payload.get("keep_last", 2)),
        eval_every=preset_payload.get("eval_every"),
        eval_batches=int(preset_payload.get("eval_batches", 8)),
        device=device,
        profile=profile,
        resume_from=preset_payload.get("resume_from"),
    )


def _train_preset(
    *,
    ctx: RunContext,
    budget: Budget,
    config: PretrainConfig,
    packed_dir: Path,
) -> dict[str, Any]:
    trainer = Pretrainer(config, ctx=ctx, budget=budget, packed_dir=packed_dir)
    summary = trainer.train()
    artifact = ctx.artifact_path(f"pretrain_{config.preset}_summary.json")
    artifact.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    ctx.add_artifact(artifact)
    return summary


def _mlx_parity_report(config: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    payload = _stage_config(config).get("mlx_parity") or {}
    preset_name = str(payload.get("preset", "s"))
    seq_len = int(payload.get("seq_len", 128))
    batch_size = int(payload.get("batch_size", 2))
    if not MLX_AVAILABLE:
        return {"available": False, "reason": "未安装 MLX"}
    base = PRESETS[preset_name]
    model_config = GPTConfig(
        vocab_size=base.vocab_size,
        block_size=seq_len,
        n_layer=base.n_layer,
        n_head=base.n_head,
        n_embd=base.n_embd,
        ffn_hidden=base.ffn_hidden,
        n_kv_head=base.n_kv_head,
        tie_embeddings=True,
    )
    torch_model = GPT(model_config)
    mlx_model = MLXGPT(model_config)
    load_from_torch(mlx_model, torch_model)
    rng = np.random.default_rng(int(config.get("seed", 42)))
    inputs = rng.integers(0, model_config.vocab_size, (batch_size, seq_len - 1), dtype=np.int32)
    targets = rng.integers(0, model_config.vocab_size, (batch_size, seq_len - 1), dtype=np.int32)
    loss_mask = np.ones((batch_size, seq_len - 1), dtype=np.float32)
    report = parity_check(torch_model, mlx_model, inputs, targets, loss_mask)
    report["model"] = model_config.to_dict()
    report["params"] = model_config.num_params()
    report["batch"] = [batch_size, seq_len - 1]
    tolerance = report.get("tolerance", {})
    report["passed"] = (
        report["loss_rel_diff"] <= tolerance.get("loss_rel_diff", 1e-3)
        and report["grad_norm_rel_diff"] <= tolerance.get("grad_norm_rel_diff", 1e-2)
    )
    return report


def _mlx_performance(config: dict[str, Any], ctx: RunContext, mps_tokens_per_second: float | None) -> dict[str, Any]:
    payload = _stage_config(config).get("mlx_throughput") or {}
    if not MLX_AVAILABLE:
        return {"available": False, "reason": "未安装 MLX"}
    preset_name = str(payload.get("preset", "s"))
    preset = PRESETS[preset_name]
    model_config = GPTConfig(
        vocab_size=preset.vocab_size,
        block_size=int(payload.get("seq_len", 256)),
        n_layer=preset.n_layer,
        n_head=preset.n_head,
        n_embd=preset.n_embd,
        ffn_hidden=preset.ffn_hidden,
        n_kv_head=preset.n_kv_head,
    )
    result = mlx_throughput(
        model_config,
        batch_size=int(payload.get("batch_size", 4)),
        seq_len=int(payload.get("seq_len", 256)),
        steps=int(payload.get("steps", 5)),
    )
    if mps_tokens_per_second:
        result["mps_tokens_per_second"] = mps_tokens_per_second
        result["mlx_vs_mps_ratio"] = round(result["tokens_per_second"] / mps_tokens_per_second, 3)
    return result


def _scaling_report(results: dict[str, Any], parity: dict[str, Any], performance: dict[str, Any], ctx: RunContext) -> None:
    lines = ["# Stage 4 规模实验报告", "", "## 三档规模训练结果", ""]
    lines.append("| 档位 | 参数量 | FLOPs/token | tokens | final train loss | val loss | 耗时(s) | tokens/s |")
    lines.append("|---|---|---|---|---|---|---|---|")
    points: list[tuple[float, float, str]] = []
    for name, summary in results.items():
        compute = summary["flops_per_token"] * summary["tokens_seen"]
        points.append((compute, summary["final_val_loss"], name))
        lines.append(
            f"| {name} | {summary['params'] / 1e6:.1f}M | {summary['flops_per_token'] / 1e6:.1f}M | "
            f"{summary['tokens_seen']:,} | {summary['final_train_loss']} | {summary['final_val_loss']} | "
            f"{summary['wall_seconds']} | {summary['tokens_per_second']} |"
        )
    lines.append("")
    lines.append("## MLX 数值一致性")
    lines.append("")
    if parity.get("available", True):
        lines.append(f"- loss 相对误差：{parity['loss_rel_diff']:.3e}（容差 {parity['tolerance']['loss_rel_diff']}）")
        lines.append(f"- logits 最大绝对误差：{parity['logits_max_abs_diff']:.3e}")
        lines.append(f"- 梯度范数相对误差：{parity['grad_norm_rel_diff']:.3e}（容差 {parity['tolerance']['grad_norm_rel_diff']}）")
        lines.append(f"- 结论：{'通过' if parity.get('passed') else '未通过'}")
    else:
        lines.append(f"- MLX 不可用：{parity.get('reason')}")
    lines.append("")
    lines.append("## MPS vs MLX 训练吞吐（同模型同配置）")
    lines.append("")
    if performance.get("available", True):
        lines.append(f"- MLX tokens/s：{performance['tokens_per_second']}")
        if performance.get("mps_tokens_per_second"):
            lines.append(f"- MPS tokens/s：{performance['mps_tokens_per_second']}")
            lines.append(f"- MLX/MPS 比值：{performance.get('mlx_vs_mps_ratio')}")
        lines.append(f"- 测试配置：batch={performance['batch_size']}，seq_len={performance['seq_len']}，{performance['timed_steps']} 步计时")
    else:
        lines.append(f"- MLX 不可用：{performance.get('reason')}")
    lines.append("")
    lines.append("## loss vs 累计计算量（小规模趋势）")
    lines.append("")
    lines.append("| 档位 | 累计计算量（TFLOPs） | val loss |")
    lines.append("|---|---|---|")
    for compute, loss, name in points:
        lines.append(f"| {name} | {compute / 1e12:.2f} | {loss} |")
    lines.append("")
    lines.append("> 注：三档使用相同语料，但 token 预算按算力裁剪（sanity 档），因此曲线只用于观察趋势，不作为规模律拟合结论。")
    path = ctx.artifact_path("scaling_report.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    ctx.add_artifact(path)

    if points:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        points.sort()
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot([item[0] / 1e12 for item in points], [item[1] for item in points], marker="o")
        for compute, loss, name in points:
            axis.annotate(name.upper(), (compute / 1e12, loss))
        axis.set_xlabel("cumulative compute (TFLOPs)")
        axis.set_ylabel("val loss")
        axis.set_title("loss vs compute (sanity scale)")
        figure.tight_layout()
        chart = ctx.artifact_path("scaling_curve.png")
        figure.savefig(chart, dpi=120)
        plt.close(figure)
        ctx.add_artifact(chart)


def run(*, ctx: RunContext, config: dict[str, Any], budget: Budget) -> dict[str, Any]:
    lab_config = LabConfig.model_validate(config)
    data_dir = lab_config.resolved_dir("data_dir")
    packed_dir = data_dir / "processed" / "stage3" / "packed"
    if not packed_dir.exists():
        raise RuntimeError(f"缺少 Stage 3 产物 {packed_dir}，请先运行 lab run 3")
    payload = _stage_config(config)
    presets = payload.get("presets") or {}
    scaling = payload.get("scaling") or {}
    order = list(scaling.get("presets") or presets.keys())
    seed = int(config.get("seed", 42))
    device = default_device()

    results: dict[str, Any] = {}
    for preset_name in order:
        preset_payload = presets.get(preset_name)
        if not preset_payload:
            raise RuntimeError(f"configs/stage4.yaml 缺少 presets.{preset_name}")
        trainer_config = _pretrain_config(preset_name, preset_payload, profile=budget.profile, seed=seed, device=device)
        logger.info(
            "开始预训练 preset=%s，%d tokens，batch=%d×%d，grad_accum=%d，grad_checkpoint=%s",
            preset_name, trainer_config.max_tokens, trainer_config.batch_size,
            PRESETS[preset_name].block_size, trainer_config.grad_accum, trainer_config.grad_checkpoint,
        )
        results[preset_name] = _train_preset(
            ctx=ctx,
            budget=budget,
            config=trainer_config,
            packed_dir=packed_dir / trainer_config.data_key,
        )
        ctx.log_metrics(preset=preset_name, kind="pretrain_summary", **{
            key: results[preset_name][key]
            for key in ("params", "tokens_seen", "final_val_loss", "wall_seconds", "tokens_per_second", "peak_memory_gb")
        })

    logger.info("运行 MLX 数值对齐与吞吐对照")
    parity = _mlx_parity_report(config, ctx)
    ctx.write_json_artifact("mlx_parity.json", parity)
    performance = _mlx_performance(config, ctx, results.get("s", {}).get("tokens_per_second"))
    ctx.write_json_artifact("mlx_performance.json", performance)
    _scaling_report(results, parity, performance, ctx)

    summary = {
        "profile": budget.profile,
        "presets": {
            name: {
                key: value[key]
                for key in ("params", "tokens_seen", "final_train_loss", "final_val_loss", "wall_seconds", "tokens_per_second", "peak_memory_gb", "checkpoint")
            }
            for name, value in results.items()
        },
        "mlx_parity": {key: parity.get(key) for key in ("available", "passed", "loss_rel_diff", "grad_norm_rel_diff")},
        "mlx_vs_mps": performance.get("mlx_vs_mps_ratio"),
    }
    ctx.write_json_artifact("stage_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary
