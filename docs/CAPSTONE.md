# 总复盘：一台 Mac 上的大模型训练全流程实验

> 本文件是 8 个阶段完成后的**总复盘**：实测数据、工程结论、与本机限制、迁移生产环境的路径。
> 运行时间：2026-10-29；机器：Apple M4 Pro / 48 GB 统一内存 / macOS 26.6.2。

## 一、我们完成了什么

从空目录开始，在单台 Apple Silicon Mac 上真实跑通：

| # | 阶段 | 关键证据（实测） |
|---|---|---|
| 1 | 环境与实验基座 | 环境自检 12 项；MPS/MLX 矩阵 ~6 TFLOPs；S/M/L 训练吞吐 19.7k / 7.2k / 3.5k tokens/s |
| 2 | 数据工程 | 中文维基 20k 篇 → 保留 19,896 篇（14.9M 字符），近似去重 103 篇；英文 TinyStories 3k → 2,955 篇 |
| 3 | 分词器与打包 | BPE 16k/32k（1.46/1.62 字符每 token，0 UNK）；packed 中文 4M + 英文 1M token；指令 5k 条、偏好 3k 对 |
| 4 | 从零预训练 | S/M/L 三档真实训练（2M / 1M / 0.5M tokens），val loss 6.89 / 7.92 / 8.20；MLX 对齐 loss 误差 0.0、梯度误差 1.2e-7；MLX/MPS 吞吐 1.18× |
| 5 | 指令微调 | Qwen3-0.6B/1.7B 离线加载；LoRA r16（10.1M 可训练参数）30 步，val loss 2.34；合并导出可加载 |
| 6 | 偏好优化 | HH-RLHF 2,850 对 + self-play 3 对；DPO 20 步（初始 loss=log2 验证实现正确），奖励间隔/KL 监控正常 |
| 7 | 评测与追踪 | 三档 PPL、20 题选择题（25%–40%）、8 条指令遵循通过率 37.5%；重复评测差异 0.0；跨实验对比报告 |
| 8 | 量化推理服务 | MLX 4bit 体积 0.36 GB（16%）、**521 tokens/s**、0.61 GB 峰值显存；4bit 吞吐为 bf16 的 3.2–3.6 倍；OpenAI 兼容服务 4 项契约 + 并发测试全通过 |

**端到端验证**：`lab run all --profile sanity` 完整跑完 1→8 阶段，**12 分 4 秒**，8/8 阶段 success（要求 ≤30 分钟）。
所有实验目录都带 `manifest.json`（配置快照、环境、代码指纹、时间、状态、产物清单），可追溯。

## 二、最有价值的工程结论（踩坑与验证）

1. **单机训练的瓶颈不是内存，是算力与时间。**
   48 GB 统一内存下，110M 参数模型训练峰值仅 ~4.2 GB，Qwen3-0.6B LoRA 训练峰值 ~19 GB。
   所有预算设计都应围绕 tokens/s 与训练时间，而不是担心 OOM。

2. **bf16 直接训练大模型权重 + AdamW 不稳定，必须 fp32 主权重 + bf16 autocast。**
   本课 SFT 初版用 bf16 权重训练，30 步内 loss 从 1.25 升到 3.0；改为标准 AMP（fp32 权重 + autocast bf16）
   后固定批次 6 步 loss 3.43→2.30。这是 MPS 单机微调最容易踩的坑之一。

3. **MLX 与 PyTorch 可以做到逐位级对齐。**
   同结构、同权重（直接拷贝 state_dict）下，loss 相对误差 0.0、logits 最大误差 2.4e-6、梯度范数误差 1.2e-7。
   这证明"Apple 原生框架"不是黑盒，可以放心作为对照/替代实现。

4. **量化是单机推理最划算的优化。**
   4bit 体积降到 16%，峰值显存降到约 1/3，吞吐约为 bf16 的 3.2–3.6 倍（实测 465–521 vs 130–161 tokens/s）。

5. **DPO 的"零信息"基线是很好的实现正确性检查。**
   初始时策略=参考模型，loss 应等于 log 2 ≈ 0.693、margin=0、accuracy=0.5；实测完全吻合。

6. **sanity 档只能验证链路，不能证明效果。**
   30 步 LoRA、20 步 DPO、2M token 预训练得到的模型质量有限（困惑度很高、生成有重复）。
   课程的价值在于**每一条链路都真实跑通并可复现**，效果需要 standard 档（数小时到数天）才谈得上。

7. **可复现性在单机上受 MPS 非确定性影响，但影响很小。**
   两次独立 Stage 4 运行（同配置/种子）val loss 差异 0.003–0.008；评测本身重复差异 0.0。
   实验容差应据此设定。

## 三、本机限制（诚实清单）

| 限制 | 影响 | 已验证的替代方案 |
|---|---|---|
| 无 CUDA / 多卡 | 无法真实体验 FSDP/ZeRO/Megatron/FlashAttention 的分布式行为 | 梯度累积 + 单机 8 阶段；分布式作为文档对照（每篇阶段文档都有对照表） |
| 单 GPU 吞吐 ~20k tokens/s（17M 模型） | standard 档大模型训练需要数小时到数天 | `--max-minutes` + 检查点续训 + `resume_from` |
| 无可靠的本地裁判模型 | LLM-as-judge 只做文档讨论 | 规则评分 + 人工评审表 |
| bitsandbytes 未安装（QLoRA 不可用） | 无法体验 4bit 训练 | bf16 LoRA 直接训（48 GB 内存足够 0.6B/1.7B）；量化体验放在推理阶段 |
| 无 cmake / GitHub 直连 | GGUF/llama.cpp 未体验 | MLX 量化为主；GGUF 作为可选扩展 |
| HuggingFace 官方直连不可达 | HF Hub 相关生态需要代理 | hf-mirror/ModelScope 直连；代理显式配置 |

## 四、迁移到生产 GPU 集群的路径

| 本机组件 | 生产替换 | 迁移要点 |
|---|---|---|
| 自研预训练循环 | Megatron-LM / torchtitan / NanoGPT 训练栈 | 数据格式（packed + mask）可直接复用；替换并行与优化器分片 |
| 单卡 LoRA | FSDP/DeepSpeed + LoRA | 训练循环结构不变，加分布式包装；显存账要重算 |
| 自研 DPO | TRL DPOTrainer / 分布式 RLHF 框架 | 偏好数据格式（prompt/chosen/rejected）可直接复用 |
| 本地 JSONL 追踪 | W&B / MLflow + 对象存储 | manifest 字段已按此设计 |
| FastAPI 单进程 | vLLM / SGLang / TGI | OpenAI 接口契约一致，客户端代码无需改 |
| MLX 4bit | GPTQ/AWQ/FP8 + PagedAttention | 量化目标一致，重新校准精度 |

## 五、如何用这套课程（2–4 周建议）

- **第 1 天**：Stage 1–3（环境、数据、分词器/打包），理解"数据决定上限"。
- **第 2–3 天**：Stage 4 sanity，再挑一档跑 standard（建议 S 档，1–2 小时）。
- **第 4–5 天**：Stage 5，改成 standard 档跑 20k 指令样本，观察 LoRA rank/学习率对比。
- **第 6 天**：Stage 6，构造自己的偏好数据（self-play 或改写数据），跑 200–400 步 DPO。
- **第 7 天**：Stage 7–8，评测、量化、起服务，用 curl/Python 客户端调用。
- **第 2 周起**：选一个方向深入（分布式概念实验 / MoE 小模型 / GGUF 量化 / 更长训练），
  或把某个阶段迁移到 GPU 云机器上对照。

## 六、延伸阅读

- [PRINCIPLES.md](PRINCIPLES.md)：每个阶段的原理与公式，并用本仓库实测数据逐条验证（含 LoRA 参数量公式、量化体积/带宽账、DPO 推导等）。

## 七、复现与验证

```bash
# 全部测试
.venv/bin/python -m pytest -q
# 静态检查
.venv/bin/ruff check src tests
# 端到端（≤30 分钟）
.venv/bin/lab run all --profile sanity
```

已归档的验收证据：
- 8 个阶段的 `runs/stage-{1..8}/<run-id>/artifacts/` 报告与 `manifest.json`
- 课程文档：`docs/stages/01..08`
- 规格与变更记录：`openspec/changes/add-llm-training-lab/`
