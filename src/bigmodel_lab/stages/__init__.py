"""8 个课程阶段的注册表。"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from ..budget import Budget
from ..manifest import RunContext


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    module: str
    requires: tuple[int, ...]
    artifacts: tuple[str, ...]
    summary: str

    @property
    def slug(self) -> str:
        return f"stage-{self.number}"


STAGES: tuple[Stage, ...] = (
    Stage(
        number=1,
        name="环境与实验基建",
        module="bigmodel_lab.stages.stage1",
        requires=(),
        artifacts=("doctor.json", "doctor.md", "benchmark.json", "benchmark.md"),
        summary="环境自检、MPS/MLX 性能基线、token 预算校准",
    ),
    Stage(
        number=2,
        name="数据工程",
        module="bigmodel_lab.stages.stage2",
        requires=(1,),
        artifacts=("dataset_card.md", "clean_stats.json", "fetch_manifest.json"),
        summary="多源获取语料、清洗、去重、统计与数据卡",
    ),
    Stage(
        number=3,
        name="分词器与数据打包",
        module="bigmodel_lab.stages.stage3",
        requires=(2,),
        artifacts=("tokenizer_report.json", "pack_report.json", "split_manifest.json"),
        summary="训练 BPE 分词器、指令/偏好格式化、序列打包与数据划分",
    ),
    Stage(
        number=4,
        name="从零预训练",
        module="bigmodel_lab.stages.stage4",
        requires=(3,),
        artifacts=("metrics.jsonl", "summary.json", "mlx_parity.json", "scaling_report.md"),
        summary="S/M/L 三档预训练、MLX 数值与性能对照、规模曲线",
    ),
    Stage(
        number=5,
        name="指令微调",
        module="bigmodel_lab.stages.stage5",
        requires=(1, 3),
        artifacts=("adapter", "sft_summary.json", "before_after.md", "mask_check.json"),
        summary="Qwen3 小模型 LoRA/QLoRA SFT、模板与掩码自检、超参对比",
    ),
    Stage(
        number=6,
        name="偏好优化",
        module="bigmodel_lab.stages.stage6",
        requires=(5,),
        artifacts=("dpo_adapter", "dpo_summary.json", "side_by_side.md", "monitor.json"),
        summary="偏好对构造、DPO 训练、稳定性监控与对齐对比",
    ),
    Stage(
        number=7,
        name="评测与实验追踪",
        module="bigmodel_lab.stages.stage7",
        requires=(4,),
        artifacts=("ppl_report.json", "task_report.json", "generation_report.md", "compare_report.md"),
        summary="困惑度、本地基准、生成评审、跨实验对比报告",
    ),
    Stage(
        number=8,
        name="量化推理与服务",
        module="bigmodel_lab.stages.stage8",
        requires=(5, 7),
        artifacts=("quant_report.json", "bench_report.json", "server_contract.json"),
        summary="MLX 量化、吞吐/内存基准、OpenAI 兼容服务",
    ),
)

_STAGE_BY_NUMBER: dict[int, Stage] = {stage.number: stage for stage in STAGES}


def get_stage(number: int) -> Stage | None:
    return _STAGE_BY_NUMBER.get(number)


def stage_slug(number: int) -> str:
    stage = get_stage(number)
    if stage is None:
        raise ValueError(f"未知阶段: {number}")
    return stage.slug


def load_runner(stage: Stage) -> Callable[..., Any]:
    """按注册表动态导入阶段模块，返回其 run(ctx, config, budget) 函数。"""
    module = importlib.import_module(stage.module)
    runner = getattr(module, "run", None)
    if runner is None:
        raise RuntimeError(f"阶段模块 {stage.module} 缺少 run() 实现")
    return runner


def run_stage(stage: Stage, ctx: RunContext, config: dict[str, Any], budget: Budget) -> Any:
    return load_runner(stage)(ctx=ctx, config=config, budget=budget)


def format_stage_list() -> str:
    lines = ["编号  阶段                    依赖      产物", "-" * 78]
    for stage in STAGES:
        requires = ",".join(str(item) for item in stage.requires) or "-"
        lines.append(f"{stage.number:<4} {stage.name:<22} {requires:<8} {', '.join(stage.artifacts)}")
        lines.append(f"     {stage.summary}")
    return "\n".join(lines)
