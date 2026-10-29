# Stage 6：偏好优化（DPO）

> 对应 capability：`preference-optimization`（偏好对构造、DPO、稳定性监控、对齐对比）
> 前置：Stage 5 的 SFT 模型。产物：DPO 适配器、损失/奖励间隔曲线、监控报告、SFT vs DPO 并排对比。

## 学习目标

1. 理解 DPO 在做什么：用偏好对直接优化策略，让"优选回答"的隐式奖励高于"劣选回答"，同时用参考模型约束偏离。
2. 亲手构造偏好数据：公开偏好数据集（HH-RLHF）+ **self-play 自造偏好对**。
3. 认识对齐阶段的核心风险：**过度优化（reward hacking）**，以及如何用奖励间隔 / 近似 KL / 生成质量做监控。
4. 完成 SFT 与 DPO 的可复现并排对比。

## 运行命令

```bash
.venv/bin/lab run 6 --profile sanity   # 约 1 分钟
.venv/bin/lab run 6 --profile standard # 400 步
```

## 产物说明

| 路径 | 内容 |
|---|---|
| `runs/stage-6/<run>/artifacts/selfplay_report.json` | self-play 偏好对生成统计与样例 |
| `runs/stage-6/<run>/artifacts/dpo_summary.json` | DPO 训练摘要（loss/奖励间隔/准确率/KL/显存） |
| `runs/stage-6/<run>/artifacts/dpo_adapter/` | DPO LoRA 适配器（可加载到 SFT 模型） |
| `runs/stage-6/<run>/artifacts/monitor.json` | 稳定性监控（阈值、历史、告警） |
| `runs/stage-6/<run>/artifacts/side_by_side.md` | SFT vs DPO 并排对比 + 人工评审表 |

## 实测结果（sanity 档，2026-10-29）

| 项目 | 实测 |
|---|---|
| 公开偏好对 | 2,850（HH-RLHF 测试分片，Stage 3 已格式化） |
| self-play 偏好对 | 3（3 个提示 × 2 次采样，启发式质量打分） |
| 训练步数 | 20（batch=2×grad_accum=2，max_seq_len=384） |
| final train loss | 0.680（初始 ≈0.693 = log2，符合 DPO 理论） |
| reward margin | ≈0.00（sanity 步数太少，尚未显著拉开） |
| 偏好准确率 | 0.50（随机水平，符合"几乎没训练"的预期） |
| 近似 KL | 0.117（相对参考模型偏离很小） |
| 峰值显存 | 11.0 GB（策略 + 参考模型两份 fp32 权重） |
| tokens/s | 339 |
| 稳定性监控 | `ok`（无告警） |

> 解读：DPO 在初始化时 loss=log2、margin=0、accuracy=0.5——这说明实现正确（策略与参考模型完全一致时，
> 偏好对没有信息增益）。sanity 档 20 步不足以让模型真正偏好优选回答；standard 档 400 步才会看到 margin 上升。
> 监控项（奖励间隔、近似 KL、生成质量）是比"loss 下降"更重要的对齐健康指标。

## 机制说明

1. **偏好数据**：HH-RLHF 的对话转录被解析为最后一轮 Human 作为 prompt，其后 Assistant 回复分别为 chosen/rejected。
2. **self-play 构造**：同一 SFT 模型用不同采样（温度 0.9）生成两条回复，按"长度合理 + 字符多样性 + 结尾完整性"启发式打分，
   高分作 chosen、低分作 rejected（见 `data/preference.py::make_selfplay_pairs`）。
3. **DPO 损失**：`-logsigmoid(β * ((logπ_c - logπ_r) - (logπ_ref_c - logπ_ref_r)))`；
   参考模型冻结，β=0.1。
4. **稳定性监控**：奖励间隔过大、近似 KL 过大、生成质量相对基线下降 → `warn`/`stop`；
   `stop_on_unhealthy=true` 时可自动中止（本机默认 false）。
5. **可复现对比**：生成采样固定种子与参数，side-by-side 表格带人工评审列。

## 本机做法 ↔ 生产栈对照

| 本机 | 生产级对齐栈 | 差异 |
|---|---|---|
| DPO（20–400 步，单卡） | DPO/RLHF/GRPO/RLVR 大规模训练 | 方法一致，规模与数据不同 |
| LoRA 策略 + 冻结参考模型（两份 0.6B） | 策略/参考/奖励模型多副本 + 分片 | 本机内存足够，无需分片 |
| 启发式质量监控 | 奖励模型打分 + 人工评审 + 红线指标 | 本机无可靠裁判模型 |
| self-play 自造偏好对 | 拒绝采样 + 更强模型蒸馏 | 本机用同模型自博弈，质量有限 |
| 单机 339 tokens/s | 数千 tokens/s（A100 集群） | 硬件差异 |

## 常见问题

- **margin 一直是 0**：检查 DPO 损失实现与学习率；sanity 步数太少也会如此，先看 standard 档。
- **KL 增长很快但 margin 不涨**：学习率/β 不匹配，或数据质量差（chosen/rejected 区分度低）。
- **想体验 RL（GRPO/PPO）**：单机采样成本高，本课在文档中说明，不进入验收；可在 Stage 7 用规则奖励做最小 GRPO 实验（扩展项）。
