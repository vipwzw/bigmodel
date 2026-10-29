# Stage 1：环境与实验基建

> 对应 capability：`lab-runtime`（环境自检、实验产物、时间预算、性能基线）
> 前置：无。后续阶段全部依赖本阶段的基座。

## 学习目标

1. 理解训练大模型前必须确认的机器事实：加速后端、精度支持、内存/磁盘、性能上限。
2. 建立**可复现实验**的基本习惯：每次运行有独立目录、配置快照、环境与代码版本、指标与产物清单。
3. 用实测吞吐校准训练预算：知道"某个规模模型训多少 token 要多久"，而不是凭感觉。

## 已验证的本机环境

| 项目 | 实测值 |
|---|---|
| 芯片 / 内存 | Apple M4 Pro（20 核 GPU，Metal 4）/ 48 GB 统一内存 |
| 系统 | macOS 26.6.2 |
| Python / 包管理 | Python 3.12.13 + uv（`.venv`），依赖锁定在 `uv.lock` |
| 训练框架 | torch 2.15.0（MPS 可用，bf16 autocast 可用） |
| Apple 原生框架 | mlx 0.32.4 / mlx-lm |
| HF 生态 | transformers 5.21.0 / peft / trl / datasets / tokenizers |
| 资源来源 | ModelScope 直连、hf-mirror 直连、PyPI/清华镜像；官方 HF/GitHub 需本地代理 |

> 注：本机 `huggingface.co` 直连不可达（DNS 污染），但 hf-mirror 与 ModelScope 可直连。
> 如需官方 Hub 或 GitHub，在 `configs/default.yaml` 设置 `network.proxy`（本机实测 `http://127.0.0.1:1087`）。

## 运行命令

```bash
# 1) 环境自检（含网络来源探测）
.venv/bin/lab doctor

# 2) 阶段 1 完整运行：自检 + MPS/MLX 性能基线 + token 预算校准
.venv/bin/lab run 1 --profile sanity

# 3) 不产生任何实验目录的干跑检查
.venv/bin/lab run 1 --dry-run

# 4) 列出所有阶段
.venv/bin/lab stages
```

## 产物说明

运行后在 `runs/stage-1/<run-id>/` 中生成：

| 路径 | 内容 |
|---|---|
| `config.yaml` | 本次运行的合并配置快照 |
| `manifest.json` | 环境快照、代码指纹、git 版本、时间、状态、产物清单 |
| `metrics.jsonl` | 结构化的自检与基准指标 |
| `train.log` | 运行日志 |
| `artifacts/doctor.json` / `doctor.md` | 环境自检报告 |
| `artifacts/benchmark.json` / `benchmark.md` | 性能基线与预算校准报告 |
| `artifacts/stage_summary.json` | 阶段汇总 |

## 实测基线（2026-10-29，sanity 档）

矩阵运算峰值（4096³ → sanity 档为 2048³）：

| 后端 | 最佳 GFLOPS |
|---|---|
| torch MPS | ~6000（bf16） |
| MLX | ~5930（bf16） |

tiny GPT 训练吞吐（前向 + 反向 + AdamW 一步）：

| 档位 | 参数量 | 精度 | batch×ctx | tokens/s | 峰值显存 | standard token 预算 | 预计耗时 |
|---|---|---|---|---|---|---|---|
| S | 16.9M | bf16 | 2×512 | 19,263 | 1.19 GB | 100M | ≈1.4 h |
| M | 41.7M | bf16 | 2×768 | 9,131 | 3.23 GB | 120M | ≈3.7 h |
| L | 110.1M | bf16 | 1×1024 | 3,495 | 4.10 GB | 80M | ≈6.4 h |

> 校准结论：`design.md` D3 的耗时预算（1–2 / 2–4 / 3–6 小时）成立。48 GB 统一内存在三档上都不是瓶颈，
> **真正约束是单 GPU 吞吐**；标准档 L 训练建议放在夜间运行并依赖检查点续训。

## 本机做法 ↔ 生产栈对照

| 本机（Apple Silicon 单机） | 生产级大模型训练栈 | 差异与说明 |
|---|---|---|
| Metal / MPS 后端 | CUDA + NCCL | 无多卡通信；本机用梯度累积模拟"更大 batch" |
| 48 GB 统一内存（CPU/GPU 共享） | HBM 80 GB × N | 显存即内存，不需要显式 host↔device 拷贝 |
| torch autocast bf16（MPS） | fp8/bf16 混合精度 + AMP | 本机不支持 fp8；bf16 已可用 |
| 单进程训练循环 | FSDP / ZeRO / Megatron 流水并行 | 本机只能体验单卡数据并行语义（梯度累积） |
| `tokens/s` 实测校准预算 | MFU（模型 FLOPs 利用率） | 本机可用 FLOPs/token 反推 MFU |
| 本地 `metrics.jsonl` + manifest | W&B / MLflow + 对象存储 | 结构对齐，便于后续迁移 |

## 常见问题

- **`lab doctor` 报 huggingface WARN**：正常。直连不可达，改用 hf-mirror/ModelScope 即可；需要官方 Hub 时配置代理。
- **MPS 报某算子不支持**：先在 `sanity` 档复现，确认是否为精度/算子问题；后续阶段会记录回退 CPU 的情况。
- **磁盘不足**：阶段 2 起会下载语料与模型，本机当前约 176 GB 可用；下载前会做磁盘预检。
