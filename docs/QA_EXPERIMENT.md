# 训练一个语言模型，然后向它提问

> 实验日期：2026-10-30；机器：M4 Pro / 48 GB 统一内存。
> 目标：在只训练过"预训练 / 指令微调"两种方式的情况下，**真实测试能不能像 ChatGPT 一样向模型提问**，
> 并用同一套问题对比不同训练程度的模型。所有回答都是真实运行输出（原始记录见
> [QA_EXPERIMENT.json](QA_EXPERIMENT.json)，生成脚本见 `scripts/qa_eval.py`）。

## 一句话结论

**能提问，但"能提问"和"能答好"是两件事。**

| 模型 | 训练方式 | 能否提问 | 客观题 | 实际表现 |
|---|---|---|---|---|
| tiny S（16.9M） | 仅预训练 2M token | ❌ 不行 | 0/3 | 只是在续写词串，完全不理解"问题" |
| tiny S + 指令微调 | +800 步、4000 条指令 | ⚠️ 只学会"问答的壳" | 0/3 | 会输出"我…"式回答，但没有知识 |
| Qwen3-0.6B + 30 步 SFT（旧） | LoRA 30 步、400 条样本 | ⚠️ 能接话 | 1+1 答错、首都胡编 | 格式有了，内容不可靠 |
| **Qwen3-0.6B + 328 步 SFT（本次）** | LoRA 328 步、4736 条样本 | ✅ **可以** | **3/3**（1+1=2、2+3=5、北京、木星） | 简单问答可用；长回答会重复、部分概念解释错误 |

> 换句话说：**提问是"指令微调"给的接口，回答质量由"模型规模 × 训练数据 × 训练步数"决定。**
> 从零训练一个 17M 小模型，即使做了指令微调，也还不足以回答真实问题——这正是本课程的规模现实。

---

## 一、实验设置：四个模型、同一套问题

同一套 10 个中文问题（自我介绍、算术、事实、概念、指令遵循、创作、建议等），问四个模型：

1. **tiny-pre**：Stage 4 从零预训练的 S 档（16.9M 参数，2.00M token，489 步）；
2. **tiny-sft**：在 1 的基础上做 800 步指令微调（4000 条 Alpaca 中文指令，答案部分计损）；
3. **Qwen-30**：上一轮 Stage 5 的 30 步 LoRA（400 条样本）——旧版本，用于对比；
4. **Qwen-328**：本次 328 步 LoRA（4736 条样本），Q&A 主结果。

---

## 二、训练过程与指标

### 2.1 Qwen3-0.6B 指令微调（主实验）

| 项目 | 配置 |
|---|---|
| 基座 | Qwen3-0.6B（596M 参数，本地离线加载） |
| 方法 | LoRA r=16，可训练参数 **10,092,544（1.67%）** |
| 数据 | Stage 3 的 `sft_zh`：4,750 条训练 / 200 条验证（丢弃 14 条超长） |
| 超参 | batch 2 × grad_accum 4，max_seq_len 512，lr **1e-4**，warmup 20，fp32 主权重 + bf16 autocast |
| 目标步数 | 400 |
| **实际步数** | **328**（被 sanity 档 10 分钟预算中断，见下） |
| 吞吐 / 峰值显存 | ~672 tokens/s（中位数）/ **32.1 GB** |
| 耗时 | 600 秒（中断时） |

验证集损失（每 100 步评测一次）：

| step | 0 | 100 | 200 | 300 |
|---|---|---|---|---|
| val loss | 2.6476 | 1.9946 | 1.9573 | **1.9527** |

> 对比上一轮 30 步版本：val loss 2.340（38 条验证集）→ 本次 1.953（200 条验证集），**显著下降**。

**训练被预算中断的经过**：`--profile sanity` 默认 `max_minutes=10`，跑到 step 328/400、用满 600 秒时，
训练器自动保存了 LoRA 检查点（`runs/stage-5/<run>/checkpoints/sft/`）并以 `timeout` 优雅退出。
我们用这个检查点直接 `merge_and_export` 出完整模型（`models/exports/qwen3-0.6b-qa-sft-merged/`），
再量化为 MLX 4bit（`models/exports/mlx-qwen3-0.6b-qa-4bit/`，0.36 GB）用于问答。
**这就是检查点机制的实际价值**——不需要从头重训。

### 2.2 从零 tiny S 模型 + 指令微调

| 项目 | 配置 |
|---|---|
| 基座 | Stage 4 S 档：16.9M 参数，2.00M token 预训练（489 步） |
| 数据 | 4000 条 Alpaca 中文指令，模板 `问题：…\n回答：…`，只对答案计损 |
| 超参 | 800 步，batch 8，seq 512，lr 3e-4，bf16 autocast，AdamW |
| 耗时 | **104 秒**（MPS） |
| 训练 loss | **8.30 → 4.49** |

> 损失在降，但 17M 参数 + 2M token 的量级决定它学不会知识；这个实验的价值是展示"指令微调能让它学会
> 问答的**格式**，但填不进**内容**"。

---

## 三、怎么向这个模型提问（三种方式）

### 3.1 CLI（单轮）

```bash
.venv/bin/lab generate \
  --model-path models/exports/mlx-qwen3-0.6b-qa-4bit \
  --backend mlx \
  --prompt "用一句话解释什么是大语言模型。" \
  --max-new-tokens 80
```

实测输出（退出码 0，392 tokens/s，TTFT 0.053 s）：

```
大语言模型是一种人工智能，能够学习和理解语言并生成与人类语言相似的文本、文本和音频。
```

### 3.2 Python API（多轮会话）

```python
from bigmodel_lab.serve.generate import MLXGenerator, ChatSession, SamplingParams

gen = MLXGenerator("models/exports/mlx-qwen3-0.6b-qa-4bit")
session = ChatSession(gen, params=SamplingParams(max_tokens=100, temperature=0.3))
print(session.send("你好，我想学做大模型训练，应该从哪里开始？"))
print(session.send("我只有一台 Mac，没有 NVIDIA 显卡，还能练吗？"))
print(session.send("那第一步具体该做什么？"))
```

`enable_thinking=False` 已在生成器里默认打开（Qwen3 思考模式会输出 `<think>…</think>` 前缀，
问答场景关掉它更简洁；如果模板不支持该参数会自动回退）。

### 3.3 OpenAI 兼容服务（HTTP）

```bash
.venv/bin/lab serve --model-path models/exports/mlx-qwen3-0.6b-qa-4bit --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"中国的首都是哪座城市？"}],"max_tokens":50}'
```

实测：HTTP 200，返回 `chat.completion`；流式接口返回 `chat.completion.chunk` + `[DONE]`；
多轮时把历史消息一起传即可（服务器无状态）。

---

## 四、Qwen3-0.6B（328 步 SFT）问答记录

### 4.1 单轮 10 题（temperature=0.3，max_tokens=100，MLX 4bit，270–574 tokens/s）

| # | 问题 | 回答（原文，截断） | 评价 |
|---|---|---|---|
| 1 | 你好，你是谁？ | 我是一个人工智能助手，可以为您提供帮助和支持。我是基于自然语言处理技术开发的… | ✅ 像样（末尾有重复） |
| 2 | 1+1等于几？2+3等于几？ | `1 + 1 = 2`，`2 + 3 = 5` | ✅ 全对 |
| 3 | 中国的首都是哪座城市？ | 中国的首都是北京。 | ✅ |
| 4 | 什么是机器学习？ | 机器学习是一种人工智能技术，旨在通过分析数据和统计模型，从数据中学习并预测未来的结果… | ✅ 基本正确 |
| 5 | 用一句话解释什么是大语言模型。 | 大语言模型是一种人工智能技术，通过训练计算机来生成和理解文本… | ✅ |
| 6 | 给初学者三条学习深度学习的建议。 | 1. **从基础开始**… 2. **实践和探索**…（第 3 条被截断） | ✅ 格式正确 |
| 7 | 请用中文写一句关于秋天的诗句。 | 秋风送爽，落叶纷飞，大地披上金毯，秋意盎然，万物在秋日里欢欣。 | ✅ 押题 |
| 8 | 解释一下 LoRA 微调的原理。 | LoRA（LoRA）是一种基于神经网络的微调技术…（内容实际上在讲通用微调，**没有讲低秩分解**） | ❌ 概念错误 |
| 9 | 如果一个模型回答总是很啰嗦，你会怎么改进？ | 1. 明确回答需求 2. 简洁表达 3. 使用专业术语 4. 分段回答… | ✅ 合理 |
| 10 | 太阳系中体积最大的行星是哪个？ | 太阳系中体积最大的行星是木星。 | ✅ |

**客观题（算术/事实）3/3 全对**；指令遵循（三条建议、诗句、改进建议）基本满足；
**概念解释是弱项**（LoRA 答错），且长回答出现明显重复。

### 4.2 多轮对话（同一会话，3 轮）

| 轮 | 用户 | 模型（节选） |
|---|---|---|
| 1 | 你好，我想学做大模型训练，应该从哪里开始？ | 学习大模型训练是一个非常有挑战性的过程… 1. 了解大模型训练的类型…（列了 GPT/Bloom/Llama、BERT、ViT 等） |
| 2 | 我只有一台 Mac，没有 NVIDIA 显卡，还能练吗？ | 在您的 Mac 上运行大型模型训练是完全可行的，即使没有 NVIDIA 显卡…使用 PyTorch/TensorFlow/HuggingFace 等 |
| 3 | 那第一步具体该做什么？ | 1. 选择适合的模型… 2. 安装和配置环境… |

**能延续话题**（知道"Mac 没显卡"在问什么），但信息密度低、有重复、细节（如哪些框架能在 Mac 上跑）不够准确。

### 4.3 OpenAI 服务实测

- 两轮 HTTP 对话均 200；第二轮模型把"练"误解为翻译问题（多轮理解仍不稳）。
- 流式接口：HTTP 200，收到 `chat.completion.chunk` 与 `[DONE]`。

---

## 五、从零 tiny 模型问答记录（关键对照组）

### 5.1 仅预训练（2M token）：完全无法提问

| 问题 | 回答（原文） |
|---|---|
| 你好，你是谁？ | 的演员\n* 布·沃伊克(Tith)\n* 塞·克·莱·阿萨姆·萨(DE)…（在续写维基式列表） |
| 1+1等于几？ | 、新海、黄子、大阳、高、石、西西、南、林、刘东…（地名串） |
| 中国的首都是哪座城市？ | 饰 我:\n* 世界\n* 世界\n* 白安一区:美井… |
| 什么是机器学习？ | 科\n* 土夫\n* 小德(学名: )… |

**现象**：它只是在"续写训练语料里常见的结构"（列表、地名、诗句），
根本没有"回答问题"这个概念——因为预训练目标是 `P(next token)`，不是 `P(answer|question)`。

### 5.2 + 800 步指令微调：学会了"壳"，没有"内容"

| 问题 | 回答（原文） |
|---|---|
| 你好，你是谁？ | 我我可以从我感到守："你：我我我们希望你你/我/我，我让我们我我会在… |
| 1+1等于几？ | 给定的 2. 计算一个变量 2. `num`，将有一个输入数，`cinta`… |
| 中国的首都是哪座城市？ | 截至2021年，德国宪法战役是国家的一个重大影响… |
| 什么是机器学习？ | 计算机学习在日常行业中，包括人工智能、社交媒体、网络…（像百科句子） |

**现象**：相比 5.1，它开始输出"我…"、"1. 2."这种**回答句式**（说明格式被学到了），
但内容仍然是语料片段的拼接。17M 参数 + 2M token 的预训练量不足以支撑任何真实知识。

---

## 六、能力评估汇总

| 维度 | tiny-pre | tiny-SFT | Qwen-30（旧） | **Qwen-328（本次）** |
|---|---|---|---|---|
| 回答问题（格式） | ❌ | ⚠️ 部分 | ⚠️ 部分 | ✅ |
| 算术/事实客观题 | 0/3 | 0/3 | 1+1=1（错）、首都胡编 | **3/3** |
| 指令遵循（三条建议/诗句） | ❌ | ❌ | ⚠️ | ✅ |
| 多轮连贯 | ❌ | ❌ | — | ⚠️ 能接话题，细节不稳 |
| 概念解释 | ❌ | ❌ | ❌ | ⚠️ 部分错（LoRA） |
| 长回答重复 | — | — | 有 | 有（Qwen 也会复读） |
| 推理速度 | — | — | — | 270–574 tokens/s（4bit） |

**结论**：对于"简单问答、格式指令"已经可用；对于"知识准确性、长回答质量、多轮一致性"仍然很弱。
这是 0.6B 参数 + 4.7k 样本 + 328 步的现实水平，符合预期。

---

## 七、为什么差距这么大（原理）

1. **预训练 ≠ 会回答问题**：预训练只优化"续写下一 token"，模型学到的是文本分布；
   要让它"听懂问题并作答"，必须做**指令微调（SFT）**——用 `(问题, 答案)` 对话数据继续训练，
   并且只对答案 token 计损失（prompt 部分 mask 掉）。这就是 tiny-pre 与 tiny-SFT 的差别来源。
2. **规模决定知识容量**：17M 参数只能记住很少的模式；0.6B（大 35 倍）+ 指令数据才有一定知识与常识。
   我们对比中 Qwen 能答出"北京/木星/1+1=2"，tiny 模型连格式都刚学会。
3. **训练数据与步数决定"稳不稳"**：30 步（400 条样本）时 Qwen 还会把 1+1 答成 1、首都问题自相矛盾；
   328 步、4736 条样本之后，基础问答稳定下来。但 328 步仍远不是充分训练，所以长回答会复读、
   概念解释会错——**继续加数据/步数才能进一步改善**。
4. **没有做 RLHF/DPO**：本次流程里 DPO 阶段（Stage 6）没有并入这个 Q&A 模型，因此没有"偏好对齐"，
   模型不会"更简洁、更准确"地回答，只会模仿 SFT 数据的风格。
5. **思考模式**：Qwen3 默认会输出 `…</think>` 思考串；关闭 `enable_thinking` 后回答更直接，
   但小模型在非思考模式下的推理能力会再打折扣——这也是回答里偶尔出现事实错误的原因之一。

---

## 八、如何改进

| 方向 | 做法 | 预期 |
|---|---|---|
| 继续训练 | 把本次 400 步跑完，或跑 800–2000 步 | 格式更稳、复读减少 |
| 换更大模型 | 用已下载的 Qwen3-1.7B（`configs/stage5.yaml` 的 `advance.enabled=true`） | 知识与推理明显提升 |
| 更多/更好数据 | 全量 Alpaca 中文（52k）+ 领域数据 | 覆盖面提升 |
| 偏好对齐 | 接 Stage 6 的 DPO（400 步） | 回答更简洁、更"讨人喜欢" |
| 评测闭环 | 把 10 题扩展成 100+ 题带标准答案的评测集 | 可量化对比不同训练步数 |
| 系统提示 | 用 `DEFAULT_SYSTEM_PROMPT` 约束风格（我们已默认使用） | 回答更稳定 |

---

## 九、复现与产物

```bash
# 1) 训练（本次实际跑法；sanity 档会在 10 分钟处中断，可用 --max-minutes 加大预算）
.venv/bin/lab run 5 --profile sanity --config configs/stage5_qa.yaml

# 2) 用检查点合并 + 量化（本实验实际采用的恢复方式）
python - <<'PY'
from pathlib import Path
from bigmodel_lab.finetune.sft import merge_and_export
merge_and_export(
    adapter_dir=Path("runs/stage-5/<run-id>/checkpoints/sft"),
    base_model_dir=Path("models/Qwen3-0.6B"),
    output_dir=Path("models/exports/qwen3-0.6b-qa-sft-merged"),
    device="mps",
)
PY
.venv/bin/python -c "from mlx_lm import convert; convert('models/exports/qwen3-0.6b-qa-sft-merged', 'models/exports/mlx-qwen3-0.6b-qa-4bit', quantize=True, q_bits=4)"

# 3) 从零 tiny 模型指令微调（可选，约 2 分钟）
.venv/bin/python -m scripts.qa_experiment --mode tiny-sft --steps 800 --output models/exports/tiny-s-qa-sft.pt

# 4) 全量问答评测（生成原始 JSON + 基础 markdown）
.venv/bin/python -m scripts.qa_eval \
  --merged-dir models/exports/qwen3-0.6b-qa-sft-merged \
  --qwen-model-dir models/exports/mlx-qwen3-0.6b-qa-4bit \
  --tiny-checkpoint runs/stage-4/20261029-113020-ceeae475/checkpoints/s/ckpt_step0000489.pt \
  --tiny-sft models/exports/tiny-s-qa-sft.pt
```

| 产物 | 说明 |
|---|---|
| `docs/QA_EXPERIMENT.json` | 原始问答记录（机器可读，含 token/s、HTTP 状态等） |
| `runs/stage-5/20261030-111734-9df7ea97/` | 本次 SFT 运行（检查点、日志、指标、超时记录） |
| `models/exports/qwen3-0.6b-qa-sft-merged/` | 用检查点合并出的 Q&A 模型（fp32） |
| `models/exports/mlx-qwen3-0.6b-qa-4bit/` | 4bit MLX 量化版（问答演示实际使用） |
| `models/exports/tiny-s-qa-sft.pt` | tiny 模型指令微调检查点 |
| `scripts/qa_experiment.py` / `scripts/qa_eval.py` | 训练 + 评测脚本 |

---

## 十、一句话总结

**"训练一个语言模型并向它提问"——这个流程本身已经跑通了（CLI / Python / HTTP 三种提问方式都可用），
并且能明显看到"预训练 → 指令微调 → 更大模型/更多数据"每一步带来的能力变化。
但在 0.6B + 328 步的规模下，它只是"能答简单问题"，离"可靠的问答助手"还有明显差距——这正是规模、
数据与对齐训练的意义。**
