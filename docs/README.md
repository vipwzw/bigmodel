# bigmodel-lab 课程总览：在 Apple Silicon 上体验大模型训练全流程

> 目标：只用一台 Apple Silicon Mac（本课程实测 M4 Pro / 48 GB 统一内存），**真实跑通**从语料到服务的
> 8 个阶段，并逐层对照生产级大模型训练/推理栈。所有产物可复现、可验收。

## 快速开始

```bash
# 1) 环境
uv sync --all-groups          # 已含 torch / transformers / peft / mlx / mlx-lm ...
.venv/bin/lab doctor          # 环境与资源源自检

# 2) 列出阶段
.venv/bin/lab stages

# 3) 逐阶段运行（sanity 档）
.venv/bin/lab run 1 --profile sanity
.venv/bin/lab run 2 --profile sanity
...
.venv/bin/lab run 8 --profile sanity

# 4) 或一键端到端（约 20 分钟；模型/数据已缓存时）
.venv/bin/lab run all --profile sanity
```

每个阶段运行后都会在 `runs/stage-N/<run-id>/` 生成完整实验目录（配置快照、manifest、指标、日志、产物）。
课程文档在 `docs/stages/`。

## 8 个阶段

| 阶段 | 主题 | 核心能力 | 文档 | 预计耗时（sanity） |
|---|---|---|---|---|
| 1 | 环境与实验基建 | 环境自检、MPS/MLX 性能基线、token 预算校准、实验 manifest | [01-env-and-baseline.md](stages/01-env-and-baseline.md) | ~30 s |
| 2 | 数据工程 | 多源获取（ModelScope/hf-mirror/代理）、清洗、MinHash 去重、数据卡 | [02-data-engineering.md](stages/02-data-engineering.md) | ~1 min |
| 3 | 分词器与数据打包 | BPE 训练、指令/偏好格式化、packing + 段掩码、可复现划分 | [03-tokenizer-and-packing.md](stages/03-tokenizer-and-packing.md) | ~25 s |
| 4 | 从零预训练 | S/M/L 三档、bf16 autocast、梯度累积/检查点、断点续训、MLX 对照、规模实验 | [04-pretraining.md](stages/04-pretraining.md) | ~7 min |
| 5 | 指令微调 | Qwen3-0.6B LoRA、模板与掩码、超参对比、合并导出 | [05-supervised-finetuning.md](stages/05-supervised-finetuning.md) | ~2.5 min |
| 6 | 偏好优化 | HH-RLHF + self-play 偏好对、DPO、稳定性监控、SFT vs DPO 对比 | [06-preference-optimization.md](stages/06-preference-optimization.md) | ~1 min |
| 7 | 评测与追踪 | 困惑度、选择题、指令遵循规则评测、跨实验对比、可复现 | [07-evaluation.md](stages/07-evaluation.md) | ~30 s |
| 8 | 量化推理与服务 | MLX 4bit/8bit 量化、吞吐/TTFT/显存基准、CLI 生成、OpenAI 兼容服务 | [08-inference-serving.md](stages/08-inference-serving.md) | ~3 min |

## 理论册与总复盘

- [PRINCIPLES.md](PRINCIPLES.md)：**原理与实测**——每个阶段背后的机器学习机制、公式推导，以及用本仓库真实数据做的验证。
- [CAPSTONE.md](CAPSTONE.md)：实测数据汇总、工程踩坑、本机限制与迁移生产环境的路径。

## 技术栈

| 环节 | 主栈 | Apple 原生对照 | 生产栈对照 |
|---|---|---|---|
| 环境 | uv + Python 3.12 | — | 容器/集群镜像 |
| 数据 | modelscope、datasets、tokenizers、numpy | — | Spark/Ray Data |
| 分词器 | tokenizers（byte-level BPE） | — | 多语言大词表分词器 |
| 预训练 | PyTorch MPS + 自研训练循环 | MLX（同结构、逐位对齐） | Megatron/DeepSpeed/FSDP |
| 微调 | transformers + peft（LoRA） | mlx-lm LoRA（可选） | 同左 + 分布式 |
| 对齐 | 自研 DPO 训练循环 | — | RLHF/GRPO/RLVR 框架 |
| 评测 | 自研评测 + matplotlib | — | lm-evaluation-harness |
| 推理 | transformers + MPS（基线） | mlx-lm 4bit/8bit | vLLM/SGLang/TGI |
| 服务 | FastAPI（OpenAI 兼容） | mlx-lm server | 多副本 + 调度 |

## 本机约束与实测结论

- **网络**：ModelScope、hf-mirror、PyPI/清华镜像可直连；HuggingFace 官方与 GitHub 需要本地代理
  （实测 `http://127.0.0.1:1087`）。课程默认直连源优先、代理可选。
- **算力**：单 GPU、无 CUDA。从零预训练 S/M/L 三档实测 19.7k / 7.2k / 2.9k tokens/s（bf16），
  与 design 的 token 预算校准吻合。
- **内存**：48 GB 统一内存不是瓶颈（L 档训练峰值仅 ~4.2 GB，Qwen3-0.6B LoRA 训练峰值 ~19 GB）。
  真正的约束是**单卡吞吐与时间**。
- **量化**：MLX 4bit/8bit 在本机稳定可用；GGUF/llama.cpp 因无 cmake 且需源码构建，列为可选扩展。

## 目录结构

```
configs/              数据源、各阶段参数
src/bigmodel_lab/     CLI、基座、data/train/finetune/evaluation/serve、8 个阶段模块
docs/stages/          8 篇课程文档
data/                 原始/处理数据、评测基准（不入库）
models/               下载的模型与导出模型（不入库）
runs/                 每次实验的完整产物（不入库）
openspec/             OpenSpec 规格与变更记录
tests/                单元与集成测试
```

## 复现与清理

```bash
# 复现某次实验：使用 runs/... 下的 config.yaml
.venv/bin/lab run 4 --config runs/stage-4/<run-id>/config.yaml

# 查看某阶段最近实验
ls runs/stage-4/

# 清理历史实验目录（默认 dry-run）
.venv/bin/lab clean --keep 3
.venv/bin/lab clean --keep 3 --apply
```
