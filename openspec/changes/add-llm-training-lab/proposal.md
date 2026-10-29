# Proposal

## Why

只有一台 Apple Silicon Mac（M4 Pro / 48 GB 统一内存 / 20 核 GPU）时，无法接入 CUDA 集群，常见教程要么停留在"跑一个 demo"，要么默认多卡环境，无法系统体验大模型训练的真实流程与主流技术栈。本变更在单机上构建一条**真实可运行、分阶段可验收、并与生产训练栈逐层对照**的大模型训练实验课程，让学习者亲手完成"数据 → 分词 → 预训练 → SFT → 偏好优化 → 评测 → 量化推理"的完整闭环。

## What Changes

- 新建单机 LLM 训练实验项目 `bigmodel`：Python 3.12 + uv 管理环境，统一 `lab` CLI 与配置系统，所有阶段产物写入规范化实验目录（配置快照、日志、指标、模型、样本）。
- 8 个递进阶段，每阶段包含可运行脚本、课程文档、可验收产物：
  1. 环境与实验基建（MPS/MLX 能力探测、性能基线与时间预算）
  2. 数据工程（ModelScope 语料获取、清洗、去重、质量过滤）
  3. 分词器与数据打包（BPE 训练、chat template、packing 与 mask）
  4. 从零预训练 tiny GPT（MPS bf16/autocast、warmup+cosine、梯度累积、梯度检查点、checkpoint/resume），并用 MLX 实现对照训练
  5. 指令微调 SFT（Qwen3 小模型 + LoRA/QLoRA，peft/trl 主栈）
  6. 偏好优化 DPO（偏好对构造、DPO 训练、与 SFT 基线对比）
  7. 评测与实验追踪（困惑度、本地基准、生成样例评审、wandb/tensorboard、消融对比）
  8. 量化推理与服务（MLX 4bit 量化、吞吐/内存基准、OpenAI 兼容推理服务）
- 每阶段文档给出"本机做法 ↔ 生产级大模型栈"对照（如 MPS ↔ CUDA、单机梯度累积 ↔ FSDP/ZeRO、TRL ↔ Megatron/DeepSpeed、MLX-LM/llama.cpp ↔ vLLM/SGLang），并说明本机受哪些硬件限制、为什么。
- 资源获取多源可配：ModelScope 与 hf-mirror 可直连，HuggingFace 官方与 GitHub 可经本机代理访问（实测 HTTP `127.0.0.1:1087`）；模型/数据默认走 ModelScope 直连，HF 生态与 GitHub 作为可切换来源，代理为显式可选配置而非硬依赖，Python 依赖走 PyPI/清华镜像。
- **Out of scope**：真实千亿级训练、多机多卡、CUDA 专属内核（FlashAttention-3、DeepSpeed 等）的实际执行；这些仅做文档对照与可选机制演示，不作为验收项。

## Capabilities

### New Capabilities

- `lab-runtime`: 实验项目基座——uv 环境与依赖锁定、`lab` CLI、`doctor` 环境自检、实验目录与可复现 manifest、课程导航与阶段验收机制。
- `data-pipeline`: 语料与数据工程——ModelScope 数据获取、清洗/去重/质量过滤、BPE 分词器训练、指令/偏好数据格式化与打包。
- `pretraining`: 从零预训练——tiny GPT 模型定义、MPS 混合精度训练循环、LR 调度、梯度累积/检查点、断点续训、MLX 对照训练与性能对比。
- `supervised-finetuning`: 指令微调——基于 Qwen3 小模型的 LoRA/QLoRA SFT，chat template、数据打包、超参与显存/速度权衡。
- `preference-optimization`: 偏好优化——偏好对构造、DPO 训练、与 SFT 基线的行为对比。
- `evaluation`: 评测与实验追踪——困惑度、本地评测基准、生成样例评审、指标记录与实验对比。
- `inference-serving`: 量化推理与服务——模型量化、吞吐/内存基准、OpenAI 兼容推理接口与生产部署栈对照。

### Modified Capabilities

无（项目为全新代码库，`openspec/specs/` 当前为空）。

## Impact

- 新增代码与目录：`src/bigmodel_lab/`（CLI、各阶段模块）、`configs/`、`scripts/`、`docs/stages/`、`tests/`、`runs/`（实验产物，不入库）、`data/`、`models/`（下载资源，不入库）。
- 新增依赖（PyPI，经清华镜像安装）：`torch`、`transformers`、`datasets`、`tokenizers`、`accelerate`、`peft`、`trl`、`safetensors`、`numpy`、`matplotlib`、`mlx`、`mlx-lm`、`modelscope`、`tensorboard`、`wandb`、`pytest`。核心依赖不引入需要 CUDA 或 GitHub 源码构建的包（如 DeepSpeed、FlashAttention）；可选的 GGUF/llama.cpp 等 GitHub 源码工具在代理可用时作为加分实验，不进入验收。
- 环境约束：macOS 26.6.2 / Apple M4 Pro / 48 GB 统一内存 / 178 GB 可用磁盘；Python 3.12（uv 已安装）；不使用系统 Python 3.9。
- 资源预算：模型与数据集下载约 10–30 GB，检查点与实验产物约 10–20 GB；磁盘占用需可清理、可复现。
- 对现有系统无破坏性影响：项目目前仅有 OpenSpec 脚手架（`openspec/`、`.dsh/skills/`），本变更只做增量新增。
