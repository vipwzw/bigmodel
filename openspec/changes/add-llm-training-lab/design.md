# Design

## Context

见 `proposal.md` 的动因与范围；本节只记录塑造技术方案的事实与约束。

**项目现状**
- `/Users/king/bigmodel` 目前只有 OpenSpec 脚手架（`openspec/`、`.dsh/skills/`），无代码、无 git 仓库、无 Python 虚拟环境。
- 系统自带 Python 3.9.6（过旧，不使用）；`uv 0.11.8` 已安装且本地已有 `cpython-3.12.13`。

**硬件（实测）**
- Apple M4 Pro，20 核 GPU（Metal 4），48 GB 统一内存，macOS 26.6.2，剩余磁盘 178 GB。
- 单 GPU、无 CUDA；内存不是主要瓶颈，**单机算力与训练时间是**——所有规模/预算设计围绕此约束。

**网络（2026-10-29 复测）**
- 直连可达：`modelscope.cn`、`hf-mirror.com`（HF 镜像，实测下载约 0.4–0.5 MB/s）、`pypi.org`、`mirrors.tuna.tsinghua.edu.cn`，以及 `api.github.com`/`raw.githubusercontent.com`/`codeload.github.com`。
- 直连不可达：`huggingface.co`（DNS 被污染）、`github.com` 主站；本机 ShadowsocksX-NG 已运行，HTTP 代理 `127.0.0.1:1087` 可访问 HuggingFace（实测下载约 1.5 MB/s）与 GitHub（git clone 正常）。
- 因此：**默认走直连源（ModelScope 为主、hf-mirror 为 HF 生态备用）；官方 HF 与 GitHub 经可选的本地代理访问，代理为显式配置、不得硬编码或作为硬依赖**。以下候选已实测存在：模型 `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-1.7B`、`Qwen/Qwen3-4B`、`Qwen/Qwen2.5-0.5B-Instruct`；预训练语料 `AI-ModelScope/wikipedia-cn-20230720-filtered`、`AI-ModelScope/TinyStories`、`AI-ModelScope/fineweb-edu`、`AI-ModelScope/c4`；指令数据 `AI-ModelScope/alpaca-data-gpt4-chinese`、`AI-ModelScope/alpaca-gpt4-data-zh`、`swift/self-cognition`；偏好数据 `AI-ModelScope/DPO-En-Zh-20k`、`AI-ModelScope/ultrafeedback`、`AI-ModelScope/hh-rlhf`、`AI-ModelScope/orpo-dpo-mix-40k`。

## Goals / Non-Goals

**Goals:**
- 一条在单机上**真实跑通**的 8 阶段课程链路，每阶段命令化、产物化、可验收、可中断恢复。
- 主栈对齐主流大模型训练生态；MLX 作为 Apple 原生对照，覆盖训练与推理两端。
- 所有默认配置在 48 GB 统一内存与单机时间预算内（标准档单阶段 ≤ 数小时，sanity 档 ≤ 10 分钟）；资源获取多源优先（直连源为主、代理可选）且离线可复现。
- 每阶段文档给出"本机做法 ↔ 生产栈"的逐项对照，明确哪些是真实体验、哪些是模拟、哪些只能文档说明。

**Non-Goals:**
- 不执行真实多卡/多机分布式训练、CUDA 专属内核；不追求 SOTA 指标或复现具体大模型。
- 不做生产级推理吞吐优化与部署治理；不做完整 MLOps/超参搜索平台。
- 不追求 token 预算等同真实大模型训练量级，只观察小规模下的技术现象与趋势。

## Decisions

### D1 环境与依赖管理：uv + Python 3.12 + 镜像源
- `.venv` 由 `uv` 创建，固定 Python 3.12（复用已存在的 `cpython-3.12.13`，避免 `uv python install` 走 GitHub 下载）；依赖锁定在 `uv.lock`。
- `UV_DEFAULT_INDEX`/`PIP_INDEX_URL` 默认指向清华 PyPI 镜像，`pypi.org` 兜底；版本以安装时最新稳定版为准并锁定，不写死具体小版本。
- 备选：Homebrew Python 3.11（可运行但较旧）；conda（未安装）；`pip + venv`（可用但慢）。选择 uv 因已在、速度快、锁文件可复现。

### D2 训练框架分工：自研轻量预训练循环 + HF 微调栈 + MLX 对照
- **从零预训练**：自研训练循环（模型、数据加载、AMP、梯度累积、调度、检查点），约数百行，显式暴露机制；不经过 `transformers.Trainer`。
- **SFT / DPO**：`transformers` + `peft` + `trl`（`SFTTrainer`/`DPOTrainer`），与主流微调栈一致。
- **MLX 对照**：用 `mlx` 重写同结构 tiny GPT 训练循环；用 `mlx-lm` 做 LoRA 与量化推理对照。
- 备选：全部用 HF Trainer（MPS 行为不透明、掩盖机制）；nanoGPT（可经代理获取，但代码结构与课程目标不一致，仍不采用）；llama.cpp 训练（非训练导向，且本机无 cmake）。选择当前方案是为了同时满足"看见机制"与"贴近主流栈"。

### D3 模型规模与算力预算
从零预训练三档模型（decoder-only，RoPE + RMSNorm + SwiGLU + 因果注意力）：

| 档位 | 层数 | 隐藏维 | 头数 | 上下文 | 词表 | 参数量(约) | standard token 预算 | 标准档预计耗时 |
|---|---|---|---|---|---|---|---|---|
| S | 6 | 384 | 6 | 512 | 16k | ~17M | 100M | 1–2 h |
| M | 8 | 512 | 8 | 768 | 32k | ~41M | 120M | 2–4 h |
| L | 12 | 768 | 12 | 1024 | 32k | ~110M | 80M | 3–6 h |

- `sanity` 档：每档 ≤5M tokens、≤10 分钟，用于验证链路与快速回归。
- 预算依赖实测吞吐：Stage 0 与 Stage 4 先测 tokens/秒，再按实测值校准 token 预算（允许 ±2 倍浮动），并把校准结果写入课程文档。
- 内存估算：L 档 bf16 权重约 0.22 GB，Adam 状态约 0.9 GB，激活经梯度检查点后估计 < 6 GB；48 GB 充裕。**因此不做内存极限实验，时间才是约束**。
- 备选：更大模型（时间不可行）；更小模型（趋势不可见，难形成有效学习曲线）。

### D4 数据与分词器
- 预训练语料优先级：`AI-ModelScope/wikipedia-cn-20230720-filtered` → `AI-ModelScope/TinyStories`（快速兜底）→ 本地合成语料（完全离线兜底，保证课程可跑通）。
- SFT 数据：`AI-ModelScope/alpaca-data-gpt4-chinese` 为主，`swift/self-cognition` 可选补充。
- 偏好数据：`AI-ModelScope/hh-rlhf` 测试分片为主（`AI-ModelScope/DPO-En-Zh-20k` 仓库实测为空，弃用），`AI-ModelScope/ultrafeedback` 备选；另实现 self-play 自造偏好对，使 DPO 阶段不依赖公开偏好数据。
- 分词器：预训练阶段用 `tokenizers` 在本地语料训练 BPE（16k/32k 两档）；微调阶段直接复用 Qwen3 自带分词器。
- 获取方式：默认 `modelscope` SDK 下载到项目内 `data/`、`models/`（均不入库）；HF 生态模型/数据集可经 `HF_ENDPOINT=https://hf-mirror.com` 直连，或经可选的本地代理访问官方 Hub；训练/评测阶段设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`，数据只从本地 parquet/json 读取，避免隐式联网挂起；来源、版本与校验和写入获取清单。
- 备选：只依赖 ModelScope（可用但失去 HF 生态备份）；只依赖官方 HF（直连不可达、强依赖代理）；自建爬取（合规与稳定性差）；tiktoken（下载源不可达风险）。

### D5 偏好优化路线：DPO 为主，RL 类作为对照
- 主线：在 SFT 检查点上做 DPO（`trl.DPOTrainer`，LoRA 或全参 0.6B），冻结参考模型，显存预算约 2.4 GB。
- 对照：GRPO/PPO 类 RL 方法在单机采样开销大，作为课程文档 + 可选最小实验，不进入验收项。
- 备选：PPO（实现复杂、单机不友好）；ORPO（可作为文档对比，不作为主线）。

### D6 配置、实验产物与追踪
- 配置：YAML + `pydantic` 校验；运行参数（profile、max-minutes、seed、后端）可命令行覆盖。
- 产物布局：`runs/<stage>/<run-id>/` 下固定为 `config.yaml`、`manifest.json`、`metrics.jsonl`、`train.log`、`artifacts/`、`checkpoints/`；`run-id` = 时间戳 + 配置短哈希；manifest 记录代码树哈希（git 提交哈希可用时优先）。
- 追踪：本地 JSONL + matplotlib 曲线；可选 TensorBoard；`wandb` 仅在安装且可用时以 offline 模式记录，不作为依赖。
- 备选：Hydra（依赖重、对 MPS 无增益）；MLflow（偏服务端、过重）；纯 argparse（配置复用与产物规范难统一）。

### D7 评测方案（不依赖外网）
- 语言建模：验证集困惑度（按 token 计），并做评测集与训练集重叠检查。
- 任务评测：自建中文小基准（选择题 + 指令遵循规则任务）为主路径；需要 HF 数据集时可经 `HF_ENDPOINT=https://hf-mirror.com` 或本地代理补充运行 `lm-evaluation-harness` 子集，不作为验收必需。
- 生成评测：固定提示集批量生成 + 规则评分（长度、重复率、关键词命中）+ 人工评审表；LLM-as-judge 仅作为文档讨论/可选（本机无可靠裁判模型）。
- 对比：统一指标 schema，支持跨 run 表格 + 曲线叠加 + 预置对比场景（规模、SFT、DPO、后端）。
- 备选：`lm-evaluation-harness`（经 hf-mirror/代理可用，但 MPS 支持有限，仅作补充，不作为主线）。

### D8 推理与服务
- 量化与推理：`mlx-lm` 4bit 量化为主（Apple 原生、PyPI 可装）；`transformers` + MPS bf16 作基线对照。
- 服务：优先使用 `mlx_lm.server`（内置 OpenAI 兼容接口）做契约验证；若字段/流式行为不满足 spec，则用 FastAPI 薄封装（PyPI 可装）补齐。
- GGUF / llama.cpp：本机无 cmake，但 GitHub 与 brew 源经代理可达，可在代理可用时作为可选加分实验（安装 cmake 后源码构建）；不作为验收必需。
- 备选：vLLM（无 CUDA 不可用）；Ollama（黑盒、不适合展示机制）。

### D9 网络与离线策略
- 来源优先级：本地缓存 → ModelScope/hf-mirror 直连 → 官方 HF/GitHub（需显式启用本地代理）；来源与代理集中在配置/环境变量，`lab doctor` 逐项探测并报告。
- 代理默认不开启、不硬编码；通过显式配置（如 `HTTPS_PROXY=http://127.0.0.1:1087`）或命令参数启用；代理不可用时自动回落到直连源与缓存。
- 下载失败时给出"候选替代来源 + 重试命令"，不产生不完整缓存文件；缓存命中时跳过网络；所有远程资源记录版本与校验和。
- 训练/评测阶段默认离线标志开启，manifest 记录本次使用的来源与代理模式，确保无外网也能复现。

### D10 课程阶段与能力映射

| 阶段 | 对应 capability | 核心内容 | 验收产物 |
|---|---|---|---|
| 1 环境与实验基建 | `lab-runtime` | doctor、MPS/MLX 能力与吞吐基准、配置与 manifest 骨架 | 环境报告、基准表、可运行骨架 |
| 2 数据工程 | `data-pipeline` | 多源获取（ModelScope 为主）、清洗去重、统计 | 数据卡、统计报告 |
| 3 分词器与打包 | `data-pipeline` | BPE 训练、chat template、packing、数据划分 | 分词器、packed 数据集、校验报告 |
| 4 从零预训练 | `pretraining` | S/M/L 三档、MPS 训练、MLX 对照、规模曲线 | 检查点、曲线、对照报告 |
| 5 指令微调 | `supervised-finetuning` | Qwen3-0.6B/1.7B LoRA、模板掩码、超参对比 | 适配器、前后对比样例 |
| 6 偏好优化 | `preference-optimization` | 偏好对构造、DPO、稳定性监控 | DPO 适配器、并排对比与评审表 |
| 7 评测与追踪 | `evaluation` | 困惑度、本地基准、生成评审、跨实验对比 | 指标报告、对比报告 |
| 8 量化推理与服务 | `inference-serving` | MLX 4bit、吞吐/内存基准、OpenAI 兼容服务 | 量化模型、基准、服务调用示例 |

微调与对齐阶段资源预算（估算）：

| 任务 | 模型 | 方法 | 数据规模 | 预计显存 | 预计耗时 |
|---|---|---|---|---|---|
| SFT | Qwen3-0.6B | LoRA bf16 | 中文指令 ~20k 条 | 8–12 GB | 1–2 h |
| SFT 进阶 | Qwen3-1.7B | LoRA bf16 | 子集 ~10k 条 | 14–20 GB | 2–4 h |
| DPO | Qwen3-0.6B(SFT) | DPO + LoRA | 偏好对 ~5k | 12–18 GB | 1–3 h |
| 推理 | Qwen3-0.6B/1.7B | MLX 4bit | — | < 4 GB | — |

### D11 仓库与可复现基线
- 阶段 0 初始化 git 仓库（无远程）与 `.gitignore`：排除 `data/`、`models/`、`runs/`、`.venv/`、`__pycache__/`、`*.safetensors` 等大文件。
- manifest 记录 git 提交哈希；存在未提交改动时追加代码树哈希，保证实验可追溯。
- 回滚策略：代码回滚到上一提交；删除 `data/`、`models/`、`runs/` 不影响代码；无生产流量迁移。

## Risks / Trade-offs

- [MPS 算子覆盖/性能不达预期] → Stage 0 跑算子与吞吐基准，sanity 档先行；不支持的算子回退 CPU 并记录；所有基准留档，据此校准 token 预算。
- [网络可用性波动（直连源、镜像、代理任一失效）] → 多源优先级 + doctor 前置探测 + 缓存与合成语料兜底；代理不可用时自动回落；单源失败不阻断已完成阶段。
- [代理状态影响可复现] → manifest 记录本次来源与代理模式；验收场景优先使用直连源与离线模式，不依赖代理。
- [模型/数据集 ID 变更或被删] → ID 全部配置化 + 版本锁定 + 校验和；doctor 校验；文档记录备选 ID。
- [下载体积与磁盘] → 只下所需分片/子集；下载前检查余量；提供清理命令；总预算控制在 30–50 GB。
- [QLoRA 在 MPS 上可用性不确定（bitsandbytes 支持实验性）] → LoRA 作为验收路径；QLoRA 可选并在不可用时给出替代；量化体验前移到推理阶段（MLX 4bit 确定可用）。
- [训练时间超预算] → sanity/standard 双档 + `--max-minutes` + 检查点续训；超时优雅停止；必要时自动缩减 token 预算并记录。
- [DPO 在小模型上不稳定或退化] → 小 beta、步数上限、KL 与生成质量监控、SFT 基线对比；验收重点是"可观察现象"，不追求指标提升。
- [MLX 与 PyTorch 数值差异] → 以声明容差断言首步 loss/梯度范数，而非逐位一致；差异写入对照报告。
- [推理服务接口细节不兼容] → 以 spec 字段契约为准做契约测试；`mlx_lm.server` 不满足则 FastAPI 薄封装。
- [训练日志/指标量膨胀] → 指标写 JSONL 并按步节流，日志滚动；产物大小在 manifest 中记录。

## Migration Plan

- 全新项目、无生产流量：Stage 0 建立骨架并做 git 初始提交，其后每阶段一个增量提交（或至少一个可回滚提交点）。
- 每阶段产物写入独立 `runs/` 目录，失败可单独重跑；回滚到上一提交即可，不影响已下载资源缓存。
- 完成标志：8 阶段验收产物齐全 + 一份总复盘文档（含与生产栈对照、后续迁移到 CUDA 环境的建议）。

## Open Questions

- 预训练首选中文维基子集的规模与切分方式：实现时按实测吞吐与磁盘余量裁剪，不影响阶段设计。
- 是否启用 wandb 在线同步：默认离线，不影响任何验收项。
- 是否尝试 GGUF/llama.cpp：GitHub 与 brew 源经代理可达，具备可行性；仍属第 8 阶段的可选加分项。
