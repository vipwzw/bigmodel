"""审查模型训练：复用既有 SFT（可选 DPO）链路、manifest 与预算。

`lab risk train` 的职责：

1. 加载 `risk` 配置，确定数据集（`--dataset` 或 `runs/risk/dataset` 最近一次生成）。
2. 用 `risk.model_data` 生成 SFT（可选 DPO）训练/验证集并写入本次 run 的 artifacts。
3. 复用 `finetune.sft.SFTConfig/SFTTrainer` 做 LoRA SFT（sanity/standard 档），
   `--export` 复用 `merge_and_export` 合并导出，`--quantize` 用 `mlx_lm.convert` 产出 4bit。
4. 产物写入 `runs/risk/<run-id>/`，摘要、配置、manifest、metrics 与检查点齐全。
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from ..budget import Budget, BudgetExceeded
from ..config import PROJECT_ROOT
from ..data.chat import load_sft_records
from ..finetune.align import DPOConfig, DPOTrainer
from ..finetune.sft import LoraConfigSpec, SFTConfig, SFTTrainer, merge_and_export
from ..manifest import RunContext, latest_run
from ..train.precision import default_device
from .config import RiskConfig, load_risk_config, resolve_project_path
from .model_data import load_samples, load_splits, write_training_data
from .schema import RiskSample

logger = logging.getLogger(__name__)

DATASET_HINT = "请先运行 `lab risk data generate` 生成数据集"
QUANTIZE_BITS = 4


# ---------------------------------------------------------------------------
# 配置 / 数据集 / 记录装配
# ---------------------------------------------------------------------------


def _resolve_dataset_dir(dataset: str | None, runs_dir: Path) -> Path:
    """确定训练数据集目录：显式 --dataset 优先，否则取最近一次 risk/dataset 生成。"""
    if dataset:
        path = Path(dataset)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"数据集不存在: {path}；{DATASET_HINT}，或检查 --dataset 参数")
        return path
    latest = latest_run("risk/dataset", runs_dir)
    if latest is None:
        raise FileNotFoundError(f"未找到风险数据集（{runs_dir / 'risk' / 'dataset'}）；{DATASET_HINT}")
    return latest


def _build_lora(spec: Any) -> LoraConfigSpec:
    return LoraConfigSpec(
        r=int(spec.r),
        alpha=int(spec.alpha),
        dropout=float(spec.dropout),
        target_modules=list(spec.target_modules),
    )


def _build_sft_config(
    cfg: RiskConfig,
    *,
    profile: str,
    profile_settings: Any,
    max_steps: int,
    ctx: RunContext,
    seed: int,
) -> SFTConfig:
    """把 risk 配置映射到现有 SFTConfig（device 使用 train.precision.default_device）。"""
    train_settings = cfg.settings.model.train
    return SFTConfig(
        model_dir=resolve_project_path(train_settings.base_model_dir),
        output_dir=ctx.artifact_path("adapter"),
        lora=_build_lora(train_settings.lora),
        max_steps=max_steps,
        batch_size=int(profile_settings.batch_size),
        grad_accum=int(profile_settings.grad_accum),
        max_seq_len=int(profile_settings.max_seq_len),
        learning_rate=float(profile_settings.learning_rate),
        warmup_steps=int(profile_settings.warmup_steps),
        log_every=int(profile_settings.log_every),
        eval_every=int(profile_settings.eval_every),
        eval_batches=int(profile_settings.eval_batches),
        save_every=int(profile_settings.save_every),
        seed=seed,
        device=default_device(),
        profile=profile,
        max_samples=int(profile_settings.max_train_samples),
    )


def _prepare_training_data(
    samples: list[RiskSample],
    *,
    ctx: RunContext,
    profile_settings: Any,
    seed: int,
    splits: dict[str, list[str]] | None,
) -> dict[str, Any]:
    """写出 SFT 训练/验证 JSONL 与 stats，返回统计信息。"""
    data_dir = ctx.artifact_path("data")
    stats = write_training_data(
        samples,
        data_dir,
        max_train=int(profile_settings.max_train_samples),
        max_val=int(profile_settings.max_val_samples),
        format="sft",
        seed=seed,
        train_ids=splits.get("train") if splits else None,
        val_ids=splits.get("val") if splits else None,
    )
    for name in ("train.jsonl", "val.jsonl", "stats.json"):
        path = data_dir / name
        if path.exists():
            ctx.add_artifact(path)
    return stats


def _load_records(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从写出的 SFT JSONL 读回 trainer 记录；验证集为空时回退训练集少量样本。"""
    train_records = load_sft_records([data_dir / "train.jsonl"])
    val_records = load_sft_records([data_dir / "val.jsonl"]) if (data_dir / "val.jsonl").exists() else []
    if not val_records:
        val_records = train_records[: min(16, len(train_records))]
    return train_records, val_records


def _count_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# ---------------------------------------------------------------------------
# DPO（可选）
# ---------------------------------------------------------------------------


def _run_dpo(
    cfg: RiskConfig,
    *,
    ctx: RunContext,
    budget: Budget,
    profile: str,
    profile_settings: Any,
    samples: list[RiskSample],
    adapter_dir: Path,
    device: str,
    splits: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """可选 DPO：先把 SFT adapter 合并成独立模型，再复用 `DPOTrainer`。"""
    train_settings = cfg.settings.model.train
    dpo_settings = train_settings.dpo
    base_model_dir = resolve_project_path(train_settings.base_model_dir)
    merged_dir = ctx.artifact_path("dpo_base_merged")
    logger.info("DPO 前置：合并 SFT adapter → %s", merged_dir)
    merge_and_export(
        adapter_dir=adapter_dir,
        base_model_dir=base_model_dir,
        output_dir=merged_dir,
        device=device,
    )
    dpo_data_dir = ctx.artifact_path("dpo_data")
    stats = write_training_data(
        samples,
        dpo_data_dir,
        max_train=int(profile_settings.max_train_samples),
        max_val=int(profile_settings.max_val_samples),
        format="dpo",
        seed=int(cfg.settings.dataset.seed),
        train_ids=splits.get("train") if splits else None,
        val_ids=splits.get("val") if splits else None,
    )
    train_records = _load_jsonl(dpo_data_dir / "train.jsonl")
    val_records = _load_jsonl(dpo_data_dir / "val.jsonl")
    if not val_records:
        val_records = train_records[: min(16, len(train_records))]
    dpo_config = DPOConfig(
        model_dir=merged_dir,
        output_dir=ctx.artifact_path("dpo_adapter"),
        beta=float(dpo_settings.beta),
        max_steps=int(dpo_settings.max_steps),
        batch_size=int(dpo_settings.batch_size),
        grad_accum=int(dpo_settings.grad_accum),
        max_seq_len=int(dpo_settings.max_seq_len),
        learning_rate=float(dpo_settings.learning_rate),
        warmup_steps=int(dpo_settings.warmup_steps),
        log_every=int(dpo_settings.log_every),
        eval_every=int(dpo_settings.eval_every),
        eval_batches=int(dpo_settings.eval_batches),
        save_every=int(dpo_settings.save_every),
        seed=int(cfg.settings.dataset.seed),
        device=device,
        profile=profile,
        lora=_build_lora(train_settings.lora),
    )
    summary = DPOTrainer(
        dpo_config, ctx=ctx, budget=budget, train_records=train_records, val_records=val_records
    ).train()
    summary["data"] = stats
    ctx.write_json_artifact("dpo_summary.json", summary)
    return summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# 导出 / 量化
# ---------------------------------------------------------------------------


def _export_model(cfg: RiskConfig, *, adapter_dir: Path, device: str) -> dict[str, Any]:
    """复用 `merge_and_export` 把 LoRA 合并进基座并导出到配置目录。"""
    train_settings = cfg.settings.model.train
    export_dir = resolve_project_path(train_settings.export_dir)
    result = merge_and_export(
        adapter_dir=adapter_dir,
        base_model_dir=resolve_project_path(train_settings.base_model_dir),
        output_dir=export_dir,
        device=device,
    )
    return {"export_dir": str(export_dir), **result}


def _mlx_available() -> bool:
    """检查 mlx-lm 是否可导入（--quantize 的前置条件）。"""
    try:
        spec = importlib.util.find_spec("mlx_lm")
    except (ImportError, ValueError, AttributeError):  # pragma: no cover - 异常环境
        spec = None
    if spec is None:
        return False
    try:
        import mlx_lm  # noqa: F401, PLC0415

        return True
    except Exception:  # noqa: BLE001 - 非 Apple Silicon 或依赖损坏时导入失败
        return False


def _quantize_model(model_dir: Path, *, bits: int = QUANTIZE_BITS) -> Path:
    """用 `mlx_lm.convert` 生成 bits-bit 量化模型，输出到 models/exports/。"""
    from mlx_lm import convert  # noqa: PLC0415

    output_dir = PROJECT_ROOT / "models" / "exports" / f"mlx-{model_dir.name}-{bits}bit"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("MLX %dbit 量化: %s → %s", bits, model_dir, output_dir)
    convert(str(model_dir), str(output_dir), quantize=True, q_bits=bits)
    return output_dir


def _attach_logging(log_path: Path) -> logging.Handler:
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def _print_summary(summary: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return
    sft = summary.get("sft", {})
    data = summary.get("data", {})
    print(
        f"✓ 审查模型训练完成（profile={summary.get('profile')}，dataset={summary.get('dataset')}）\n"
        f"  train/val 样本: {data.get('train_samples')} / {data.get('val_samples')}\n"
        f"  steps: {sft.get('steps')}，final train loss: {sft.get('final_train_loss')}，"
        f"final val loss: {sft.get('final_val_loss')}，耗时: {sft.get('wall_seconds')}s\n"
        f"  adapter: {sft.get('adapter')}"
    )
    if "export" in summary:
        print(f"  导出模型: {summary['export'].get('export_dir')}")
    if "quantize" in summary:
        print(f"  量化模型: {summary['quantize'].get('path')}")
    if "dpo" in summary:
        print(f"  DPO adapter: {summary['dpo'].get('adapter')}")
    print(f"  实验目录: {summary.get('run_dir')}")


def run_train_command(args: Any) -> int:
    """`lab risk train` 实现；成功返回 0，任何失败返回 1（配置错误由 CLI 层转 2）。"""
    started = time.time()
    cfg = load_risk_config(list(getattr(args, "config", []) or []) or None)
    profile = getattr(args, "profile", None) or cfg.lab.profile
    if profile not in ("sanity", "standard"):
        print(f"参数错误: 未知 profile={profile!r}（可选 sanity/standard）", file=sys.stderr)
        return 1
    quantize_requested = bool(getattr(args, "quantize", False))
    if quantize_requested and not _mlx_available():
        print(
            "量化不可用：未安装 mlx-lm（`uv sync --group mlx`）；"
            "已跳过 --quantize，若只需训练请去掉该参数",
            file=sys.stderr,
        )
        return 1

    runs_dir = cfg.lab.resolved_dir("runs_dir")
    try:
        dataset_dir = _resolve_dataset_dir(getattr(args, "dataset", None), runs_dir)
        samples = load_samples(dataset_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"训练失败: {exc}", file=sys.stderr)
        return 1
    if not samples:
        print(f"训练失败: 数据集为空 {dataset_dir}；{DATASET_HINT}", file=sys.stderr)
        return 1

    run_config = {**cfg.raw, "profile": profile}
    ctx = RunContext.create("risk/train", run_config, runs_dir=runs_dir)
    handler = _attach_logging(ctx.log_path)
    export_dir = resolve_project_path(cfg.settings.model.train.export_dir)
    try:
        train_settings = cfg.settings.model.train
        profile_settings = getattr(train_settings, profile)
        max_steps = int(getattr(args, "max_steps", None) or profile_settings.max_steps)
        if max_steps < 1:
            raise ValueError(f"--max-steps 必须为正整数，收到 {max_steps}")
        seed = int(cfg.settings.dataset.seed)
        device = default_device()
        logger.info(
            "开始审查模型训练: profile=%s dataset=%s samples=%d max_steps=%d device=%s run=%s",
            profile, dataset_dir, len(samples), max_steps, device, ctx.run_dir,
        )

        splits = load_splits(dataset_dir)
        data_stats = _prepare_training_data(
            samples, ctx=ctx, profile_settings=profile_settings, seed=seed, splits=splits
        )
        train_records, val_records = _load_records(ctx.artifact_path("data"))
        if not train_records:
            raise ValueError("SFT 训练记录为空（可能 max_seq_len 过小导致全部样本被丢弃）")

        sft_config = _build_sft_config(
            cfg, profile=profile, profile_settings=profile_settings, max_steps=max_steps, ctx=ctx, seed=seed
        )
        budget = Budget(profile=profile, max_minutes=cfg.lab.max_minutes)
        logger.info(
            "SFT 配置: model=%s max_steps=%d max_seq_len=%d lora_r=%d max_train_samples=%d",
            sft_config.model_dir, sft_config.max_steps, sft_config.max_seq_len,
            sft_config.lora.r, sft_config.max_samples,
        )
        sft_summary = SFTTrainer(
            sft_config, ctx=ctx, budget=budget, train_records=train_records, val_records=val_records
        ).train()
        adapter_dir = Path(str(sft_summary.get("adapter") or ctx.artifact_path("adapter")))
        final: dict[str, Any] = {
            "profile": profile,
            "dataset": str(dataset_dir),
            "run_dir": str(ctx.run_dir),
            "device": device,
            "data": data_stats,
            "sft": sft_summary,
        }
        if train_settings.dpo.enabled:
            final["dpo"] = _run_dpo(
                cfg,
                ctx=ctx,
                budget=budget,
                profile=profile,
                profile_settings=profile_settings,
                samples=samples,
                adapter_dir=adapter_dir,
                device=device,
                splits=splits,
            )

        export_result: dict[str, Any] | None = None
        if bool(getattr(args, "export", False)) or quantize_requested:
            logger.info("合并导出审查模型 → %s", export_dir)
            export_result = _export_model(cfg, adapter_dir=adapter_dir, device=device)
            final["export"] = export_result
            ctx.write_json_artifact("export.json", export_result)

        if quantize_requested:
            if export_result is None:  # pragma: no cover - 上面已保证自动导出
                raise ValueError("--quantize 需要先导出合并模型")
            quant_path = _quantize_model(Path(export_result["export_dir"]), bits=QUANTIZE_BITS)
            final["quantize"] = {
                "bits": QUANTIZE_BITS,
                "path": str(quant_path),
                "source": export_result["export_dir"],
            }
            ctx.write_json_artifact("quantize.json", final["quantize"])

        ctx.write_json_artifact("train_summary.json", final)
        ctx.finish("success", 0)
        logger.info("审查模型训练完成，耗时 %.1fs", time.time() - started)
        _print_summary(final, as_json=bool(getattr(args, "json", False)))
        return 0
    except BudgetExceeded as exc:
        ctx.finish("timeout", 1, str(exc))
        print(f"训练超时：{exc}（已保留产物: {ctx.run_dir}）", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        ctx.finish("interrupted", 130, "用户中断")
        print("训练被用户中断", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 边界统一转成退出码
        ctx.finish("failed", 1, f"{type(exc).__name__}: {exc}")
        logger.exception("审查模型训练失败")
        print(f"训练失败: {exc}（产物: {ctx.run_dir}）", file=sys.stderr)
        return 1
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


__all__ = [
    "DATASET_HINT",
    "QUANTIZE_BITS",
    "run_train_command",
]
