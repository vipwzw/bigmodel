# Stage 4：从零预训练

> 对应 capability：`pretraining`（模型定义、训练循环、显存控制、断点续训、监控、MLX 对照、规模实验）
> 前置：Stage 3 的 packed 数据。产物：三档模型检查点、训练曲线、MLX 对齐/吞吐报告、规模报告。

## 学习目标

1. 亲手实现预训练循环：token 流采样、混合精度（bf16 autocast）、warmup+cosine 调度、
   梯度裁剪、梯度累积、梯度检查点。
2. 理解**检查点与断点续训**工程：保存模型/优化器/调度器/数据位置/随机状态，续训后 loss 连续。
3. 用 MLX 复刻同一模型结构，做**数值一致性**（loss/logits/梯度）与**吞吐对比**。
4. 观察小规模下的 loss–compute 趋势，理解"参数量 × token 数 = 计算量"的预算取舍。

## 运行命令

```bash
.venv/bin/lab run 4 --profile sanity   # 三档 + MLX 对照，约 7 分钟
.venv/bin/lab run 4 --profile standard # 正式预算（S 100M / M 120M / L 80M tokens，预计数小时）
```

单档/断点续训可在 `configs/stage4.yaml` 中设置 `presets.<档位>.resume_from` 指向检查点后重跑。

## 产物说明

| 路径 | 内容 |
|---|---|
| `runs/stage-4/<run>/artifacts/pretrain_{s,m,l}_summary.json` | 每档训练摘要（损失、吞吐、显存等） |
| `runs/stage-4/<run>/artifacts/pretrain_{s,m,l}_metrics.jsonl` | 逐步训练指标 |
| `runs/stage-4/<run>/artifacts/pretrain_{s,m,l}_curves.png` | loss / learning rate 曲线 |
| `runs/stage-4/<run>/checkpoints/{s,m,l}/ckpt_step*.pt` | 检查点（含优化器与随机状态，可续训） |
| `runs/stage-4/<run>/artifacts/mlx_parity.json` | MLX 与 PyTorch 数值一致性 |
| `runs/stage-4/<run>/artifacts/mlx_performance.json` | MLX vs MPS 吞吐 |
| `runs/stage-4/<run>/artifacts/scaling_report.md` / `scaling_curve.png` | 规模实验报告与曲线 |

## 实测结果（sanity 档，2026-10-29）

| 档位 | 参数量 | 上下文 | batch×grad_accum | tokens | final train loss | val loss | 耗时 | tokens/s | 峰值显存 |
|---|---|---|---|---|---|---|---|---|---|
| S | 16.9M | 512 | 8×1 | 2.00M | 6.92 | 6.89 | 102 s | 19,700 | 3.1 GB |
| M | 41.7M | 768 | 4×2 | 1.00M | 7.98 | 7.92 | 140 s | 7,165 | 4.2 GB |
| L | 110.1M | 1024 | 2×4 | 0.51M | 8.12 | 8.20 | 175 s | 2,902 | 4.2 GB |

> 解读：sanity 档的 token 预算按算力裁剪（S 2M / M 1M / L 0.5M），因此**不能**直接比较三档 loss
> 判断"越大越好"——L 档 token 少、参数多，loss 自然更高。规模曲线只用于观察"计算量增加、loss 下降"的趋势，
> 正式结论需要在 standard 档（相同 token 预算）下重跑。

### MLX 数值一致性（同一权重、fp32、batch=2×128）

| 指标 | 结果 | 容差 |
|---|---|---|
| loss 相对误差 | 0.0 | ≤1e-3 |
| logits 最大绝对误差 | 2.4e-6 | — |
| 梯度范数相对误差 | 1.2e-7 | ≤1e-2 |

### MPS vs MLX 训练吞吐（S 档，batch=4×256，5 步计时）

| 后端 | tokens/s |
|---|---|
| MLX | 23,180 |
| PyTorch MPS | 19,700 |
| MLX/MPS | **1.18×** |

> MLX 在该小模型/短序列配置下略快；大模型长序列下两者差距会变化，可在 `configs/stage4.yaml` 调整后复测。

## 训练机制说明

1. **无状态批采样**：第 i 个 batch 由全局 batch 序号唯一确定（固定种子随机置换），因此续训只需恢复
   `global_step`，数据位置天然对齐，不需要保存 dataloader 状态。
2. **梯度累积**：`tokens/step = batch × seq_len × grad_accum`，用累积模拟更大 batch。
3. **梯度检查点**：M/L 档默认开启，用计算换显存（本机 48GB 很充裕，主要是为演示机制）。
4. **检查点**：保存模型、优化器、调度器、`tokens_seen`、Python/NumPy/Torch/MPS 随机状态；按
   `keep_last` 保留最近检查点 + `best.pt`（最优 val loss）。
5. **监控**：每步记录 loss/lr/grad_norm/tokens/s/峰值显存；loss 非有限值直接中止；时间超预算时先保存检查点再抛出。
6. **MLX 对照**：`models/mlx_gpt.py` 用相同结构（RoPE/RMSNorm/SwiGLU/权重共享）重建模型，
   直接从 PyTorch state_dict 拷贝权重做逐位对齐验证。

## 本机做法 ↔ 生产栈对照

| 本机（单机） | 生产级预训练 | 差异与说明 |
|---|---|---|
| 单 GPU（MPS） | 千卡集群 + NVLink/InfiniBand | 本机无并行，只能体验单卡循环 |
| 梯度累积模拟大 batch | 数据并行 + 梯度 all-reduce | 数学等价、通信机制不同 |
| 梯度检查点（torch checkpoint） | Activation recomputation + ZeRO-Offload | 机制一致，规模不同 |
| bf16 autocast | bf16/fp8 混合精度 + 分布式优化器 | 本机无 fp8 |
| 自研 checkpoint | FSDP/ZeRO-3 分片检查点 | 本机全量保存，无需分片 |
| 单机 tokens/s 推算 MFU | MFU/吞吐/成本监控 | 可用 `FLOPs/token × tokens/s / 峰值算力` 估算 |
| 小规模 loss–compute 观察 | Scaling law 拟合（Chinchilla 等） | 本机数据不足以拟合指数，只观察趋势 |

## 常见问题

- **L 档 loss 比 M 高**：sanity 档 token 预算不同，不代表模型更差；standard 档同预算重跑再比较。
- **训练很慢**：MPS 单卡 theo 吞吐有限；先跑 sanity 验证链路，standard 档可放夜间并用 `resume_from` 续训。
- **内存不足**：`configs/stage4.yaml` 减小 `batch_size`、增大 `grad_accum`、开启 `grad_checkpoint`。
- **想偷懒**：`--max-minutes` 会在超时前保存检查点并优雅退出，下次从 `resume_from` 继续。
