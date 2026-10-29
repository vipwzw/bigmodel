# Stage 7：评测与实验追踪

> 对应 capability：`evaluation`（困惑度、本地基准、生成规则评测、追踪、跨实验对比、可复现）
> 前置：Stage 4 检查点、Stage 5/6 模型产物。产物：评测报告、对比报告、可复现性报告。

## 学习目标

1. 建立"训练 ≠ 有效"的意识：用**困惑度**、**任务准确率**、**指令遵循通过率**三类指标评估模型。
2. 理解评测的坑：数据泄漏、评测集与训练预算不匹配、指标不可比。
3. 学会**实验追踪**：本地 JSONL/report 即可复现，不依赖外部平台；可用时再接 TensorBoard/W&B。
4. 做**跨实验对比**：规模对比、SFT 前后、DPO 前后、后端对比。

## 运行命令

```bash
.venv/bin/lab run 7 --profile sanity    # 约 30 秒
.venv/bin/lab run 7 --offline           # 验证离线可复现（manifest 标记 offline）
```

## 产物说明

| 路径 | 内容 |
|---|---|
| `runs/stage-7/<run>/artifacts/ppl_report.json` | 各档 tiny 模型的验证集困惑度与数据重叠检查 |
| `runs/stage-7/<run>/artifacts/task_report.json` | 选择题 + 指令遵循生成评测明细 |
| `runs/stage-7/<run>/artifacts/generation_report.md` | 生成评测可读报告 |
| `runs/stage-7/<run>/artifacts/compare_report.md` | 跨实验对比（规模 / 后端 / SFT / DPO） |
| `runs/stage-7/<run>/artifacts/reproducibility.json` | 重复评测一致性 |
| `runs/stage-7/<run>/artifacts/tracking.json` | 追踪能力探测（TensorBoard/W&B/offline） |

## 实测结果（sanity 档，2026-10-29）

| 模型 | 验证集困惑度 | 中文选择题准确率（20 题） | 数据重叠率 |
|---|---|---|---|
| S（16.9M） | 997.8 | 35% | 0% |
| M（41.7M） | 3163.1 | 25% | 0% |
| L（110.1M） | 3849.1 | 40% | 0% |

- 随机基线：25%（4 选项）。
- 指令遵循生成评测：通过率 **37.5%**（3/8，SFT 合并模型，sanity 微调 30 步）。
- 可复现性：同一检查点重复评测，困惑度差异 **0.0**（容差 0.001），通过。
- 追踪：metrics JSONL 本地记录；TensorBoard/W&B 未安装（可选，不阻塞）。

> 解读：sanity 档训练 token 很少（S 2M / M 1M / L 0.5M），困惑度自然很高；**三个档位的训练预算不同，困惑度不可直接横比**。
> 选择题准确率在 25%–40% 之间说明小模型仅有微弱能力，这正是"从零训练小模型"的真实水平。
> 这些数字的意义在于**验证评测管线**，而不是证明模型质量。

## 机制说明

1. **困惑度**：在 Stage 3 的 packed val 集上按 token 加权计算 CE，再取 exp；同时做 train/val **行级 token 重叠检查**（0%）。
2. **选择题**：把题目与选项拼成 prompt，分别计算" A/B/C/D"续写的平均 logprob，取最大者；适用于任何分词器。
3. **生成规则评测**：8 条中文指令 + 规则（精确匹配/长度/包含/不包含/任一包含），对生成结果自动打分。
4. **跨实验对比**：自动读取最近成功的 Stage 4/5/6 运行产物（规模表、MLX 性能、SFT 前后、DPO 对比）并与本次评测合并成报告。
5. **可复现**：固定种子 + 确定性采样，重复评测指标完全一致。

## 本机做法 ↔ 生产栈对照

| 本机 | 生产评测栈 | 差异 |
|---|---|---|
| 自建中文小基准（20 题 + 8 指令） | C-Eval/CMMLU/MT-Bench/AlpacaEval | 规模与信度差异，方法一致 |
| 规则评分 + 人工评审表 | LLM-as-judge + 人工标注平台 | 本机无可靠裁判模型 |
| 困惑度 + 重叠检查 | 去污 + 泄漏检测 + 私有评测集 | 本机做了基础版 |
| 本地 JSONL + Markdown 报告 | W&B/MLflow + 看板 | 本机离线优先 |
| 手动跨实验对比 | 自动 leaderboard/回归门禁 | 本机用脚本拼接报告 |

## 常见问题

- **困惑度很高**：检查 token 预算与模型是否真的训练过；不同分词器的困惑度不可直接比较。
- **准确率等于随机**：小模型 + 极少 token 属于正常；先看训练 loss 是否下降，再看评测。
- **想接 TensorBoard/W&B**：`tensorboard`/`wandb` 未安装时报告会自动跳过；安装后可用 offline 模式。
