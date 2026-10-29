# Stage 5：指令微调（Qwen3 + LoRA）

> 对应 capability：`supervised-finetuning`（基座加载、LoRA/QLoRA、模板与掩码、超参对比、导出）
> 前置：Stage 1（环境）、Stage 3（指令数据）。产物：LoRA 适配器、训练摘要、前后对比、合并导出的模型。

## 学习目标

1. 掌握**参数高效微调**：LoRA 的秩/缩放/目标模块如何影响可训练参数、显存与吞吐。
2. 理解**对话模板与损失掩码**：只有 assistant 回复计入损失，system/user/填充位置都必须屏蔽。
3. 学会在单机（MPS）上稳定训练大模型：fp32 主权重 + bf16 autocast，而不是直接 bf16 权重 + AdamW。
4. 完成"训练 → 对比 → 合并导出 → 可加载推理"的完整闭环。

## 运行命令

```bash
.venv/bin/lab run 5 --profile sanity   # 约 2.5 分钟（资源已缓存）
.venv/bin/lab run 5 --profile standard # 20k 样本、更多步数
```

## 产物说明

| 路径 | 内容 |
|---|---|
| `runs/stage-5/<run>/artifacts/mask_check.json` | 掩码自检（模板渲染、计损位置） |
| `runs/stage-5/<run>/artifacts/model_inventory.json` | 本地模型清单（参数量、显存、是否可用） |
| `runs/stage-5/<run>/artifacts/qlora_probe.json` | QLoRA 可用性探测与替代方案 |
| `runs/stage-5/<run>/artifacts/sft_summary_*.json` | 各变体训练摘要 |
| `runs/stage-5/<run>/artifacts/sft_comparison.md` | 超参对比（rank/损失/显存/吞吐） |
| `runs/stage-5/<run>/artifacts/before_after.md` | 基座 vs 微调生成对比 |
| `runs/stage-5/<run>/artifacts/adapter/` | LoRA 适配器（可继续训练或合并） |
| `models/exports/qwen3-0.6b-sft-lora-merged/` | 合并导出的完整模型（Stage 6/7/8 使用） |

## 实测结果（sanity 档，2026-10-29）

基座模型：Qwen3-0.6B（ModelScope 下载，离线加载）

| 项目 | 实测 |
|---|---|
| 参数量 | 596,049,920（≈596M） |
| 加载显存 | 1.13 GB（bf16；训练时使用 fp32 主权重） |
| 掩码自检 | prompt 38 tokens 全部屏蔽，response 7 tokens 计损，`ok=true` |

LoRA 超参对比（30 步、每步 8 条样本、max_seq_len=384）：

| 变体 | 可训练参数 | 占比 | final train loss | val loss | tokens/s | 峰值显存 | 耗时 |
|---|---|---|---|---|---|---|---|
| lora_r16 | 10,092,544 | 1.67% | 2.167 | 2.340 | 615 | 19.2 GB | 62 s |
| lora_r8 | 5,046,272 | 0.84% | 2.165 | 2.347 | 603 | 20.3 GB | 64 s |

> 30 步只够验证链路：两个 rank 的最终 loss 接近，差异主要看后续 standard 档。
> 峰值显存包含 fp32 主权重 + 激活 + MPS 缓存，48 GB 统一内存下可放心跑。

### 训练稳定性：一个真实踩坑

初版实现直接以 bf16 权重 + AdamW 训练，30 步内 loss 反而从 1.25 升到 3.0。
改为 **fp32 主权重 + bf16 autocast**（标准 AMP 做法）后，固定批次上 6 步 loss 从 3.43 降到 2.30，训练稳定。
这条经验已写入 `finetune/sft.py`（`dtype="float32"` + autocast），也是本课程最有价值的工程细节之一。

### 微调前后生成对比（节选）

| 提示 | 基座模型 | LoRA 微调后 |
|---|---|---|
| 用一句话解释什么是大语言模型。 | `<think>` 嗯，用户让我用一句话解释什么是大语言模型… | 得体的，简洁的，准确的…（出现重复，30 步训练不足） |
| 请用中文写一句关于秋天的诗句。 | `<think>` 好的，用户让我写秋天的诗句… | 秋风起，落叶舞，山河共秋色，心怀故土情。 |

> 结论：sanity 档的 30 步 LoRA 只能验证"训练-导出-推理"链路，**不足以提升效果**；standard 档（20k 样本 + 800 步）才有可比效果。
> 对比同时暴露了基座模型的 `<think>` 思考模式输出——这也是 Qwen3 系列的特点，推理阶段需要处理。

## QLoRA 可用性

本机未安装 bitsandbytes，QLoRA 探测结果为不可用。替代方案：**bf16 LoRA 直接训练**（Qwen3-0.6B/1.7B 在 48 GB 统一内存内都不需要 4bit 量化）。
量化体验放在 Stage 8（MLX 4bit/8bit 推理量化，确定可用）。

## 本机做法 ↔ 生产栈对照

| 本机 | 生产级微调栈 | 差异 |
|---|---|---|
| Qwen3-0.6B + LoRA（r=8/16） | 7B–70B + LoRA/全参 + 多卡 | 模型规模与并行不同 |
| fp32 主权重 + bf16 autocast | mixed precision + ZeRO/FSDP | 数学一致，显存策略不同 |
| 自研训练循环 | TRL SFTTrainer / 自研 pipeline | 本课为了透明与 MPS 可控性选择自研，行为对齐 |
| 合并导出 safetensors | 推送到模型仓库 + 版本管理 | 本机本地目录 |
| 30 步 sanity | 数千 step + 评测回归 | 预算差异 |

## 常见问题

- **loss 不降/上升**：检查是否 bf16 权重 + AdamW；改用 fp32 主权重 + autocast，并降低学习率（5e-5 量级）。
- **OOM**：减小 `max_seq_len`/`batch_size`，或提高 `grad_accum`。
- **生成出现 `<think>`**：Qwen3 思考模式；可在推理时用模板参数关闭（Stage 8 文档会说明）。
- **1.7B 暂无**：`models/Qwen3-1.7B` 需下载 ~3.4 GB；`configs/stage5.yaml` 的 `advance.enabled=true` 可跑进阶实验。
