# Stage 3：分词器与数据打包

> 对应 capability：`data-pipeline`（分词器、指令/偏好格式化、packing、数据划分）
> 前置：Stage 2。产物：BPE 分词器、packed 训练数据、指令/偏好数据集。

## 学习目标

1. 理解**分词器是模型的一部分**：词表大小、压缩率、未知 token 如何影响训练成本。
2. 掌握指令微调数据格式（system/user/assistant messages）与偏好数据格式（prompt/chosen/rejected）。
3. 理解序列打包（packing）的收益与风险，并亲手实现**损失掩码 + 段掩码**。
4. 建立可复现的数据划分（固定种子）思维。

## 运行命令

```bash
.venv/bin/lab run 3 --profile sanity   # 约 25–45 秒（资源已缓存）
.venv/bin/lab run 3 --profile standard # 全量指令数据 + 完整语料
```

## 产物说明

| 路径 | 内容 |
|---|---|
| `data/processed/stage3/tokenizers/bpe_16384/` | 16k 词表 BPE 分词器（`tokenizer.json` + `meta.json`） |
| `data/processed/stage3/tokenizers/bpe_32768/` | 32k 词表 BPE 分词器 |
| `data/processed/stage3/packed/pretrain_zh_bpe{16384,32768}/` | packed 中文语料（train/val/test `.npz` + meta） |
| `data/processed/stage3/packed/pretrain_en_bpe16384/` | packed 英文语料 |
| `data/processed/stage3/instruction/sft_zh/` | Alpaca 中文指令 → chat messages（train/val） |
| `data/processed/stage3/preference/pref_en/` | HH-RLHF → prompt/chosen/rejected（train/val） |
| `runs/stage-3/<run-id>/artifacts/` | 分词器/打包/划分/指令/偏好报告 |

## 实测结果（sanity 档，2026-10-29）

分词器（在 3,000 篇中文维基文档上训练）：

| 分词器 | 词表 | 压缩率（字符/token） | 未知 token 率 | 每文档 token 数 p50 |
|---|---|---|---|---|
| bpe_16384 | 16,384 | 1.464 | 0% | 305 |
| bpe_32768 | 32,768 | 1.621 | 0% | 270 |

> 32k 词表压缩率更高（同样文本 token 更少），代价是 embedding 层参数更大。Stage 4 会看到两档词表分别用于 S 档与 M/L 档。

packed 数据（sanity 档按 token 上限裁剪）：

| 数据集 | 分词器 | train tokens | val tokens | test tokens |
|---|---|---|---|---|
| pretrain_zh | bpe_16384 | 4,000,000 | 104,575 | 97,246 |
| pretrain_zh | bpe_32768 | 4,000,000 | 96,795 | 89,735 |
| pretrain_en | bpe_16384 | 1,000,000 | 12,765 | 12,050 |

文档划分（种子 42，val/test 各 1%）：

| 语料 | train | val | test |
|---|---|---|---|
| pretrain_zh | 19,500 | 198 | 198 |
| pretrain_en | 2,897 | 29 | 29 |

指令与偏好数据：

| 数据集 | 输入 | 有效 | 保留率 | train / val |
|---|---|---|---|---|
| sft_zh（sanity 上限 5,000） | 5,000 | 5,000 | 100% | 4,750 / 250 |
| pref_en（上限 3,000 对） | 3,008 | 3,000 | 99.7% | 2,850 / 150 |

偏好数据丢弃原因：`missing_response`（择优/劣选回答缺失）、`identical_pair`（两者完全一致）。
另有 **self-play 偏好对构造工具**（`data/preference.py::make_selfplay_pairs`），在 Stage 6 作为无公开偏好数据时的兜底。

## 机制说明

1. **BPE 分词器**：byte-level BPE（`tokenizers`），特殊 token `<pad>/<bos>/<eos>/<unk>`，没有任何 UNK。
2. **打包**：把文档 token 流用 `<eos>` 分隔后拼成连续流，按 `block_size=512` 切块；
   - `loss_mask`：真实 token 为 1，末尾填充为 0（不参与损失）；
   - `segment_ids`：每个 token 所属文档编号，用于构造段掩码（避免跨文档注意力）；
   - `segment_attention_bias()`：返回 (T×T) 布尔掩码，`True` 表示允许 attend，同时满足 causal 与同段。
3. **划分**：先在文档级打乱（固定种子），再分别编码打包，避免同一文档同时出现在 train/val。
4. **指令格式**：`messages = [system, user, assistant]`，Stage 5 会再用 Qwen3 的 chat template 渲染。
5. **偏好格式**：从 HH-RLHF 的对话转录中取最后一轮 Human 作为 prompt，分别取其后的 Assistant 回复为 chosen/rejected。

> 注：`AI-ModelScope/DPO-En-Zh-20k` 仓库实测为空（仅 README），因此偏好数据改用 `AI-ModelScope/hh-rlhf` 测试分片 + self-play 工具。

## 本机做法 ↔ 生产栈对照

| 本机 | 生产级做法 | 差异 |
|---|---|---|
| `tokenizers` 本地训练 BPE（3k 文档） | 十亿级语料训练分词器 + 多语言词表 | 规模差异，方法一致 |
| numpy `.npz` 存 packed 数据 | 内存映射分片（Mosaic/WebDataset） | 本机数据小，直接落在本地盘 |
| 段掩码 + 损失掩码 | 序列打包 + 跨样本 mask（如 FlashAttention varlen） | 算法一致，本机用标准 attention + bias |
| 固定种子划分 | 数据版本 + 划分清单 | manifest 已记录划分统计 |

## 常见问题

- **分词器压缩率低**：中文维基熵高，1.4–1.6 字符/token 属正常；增大词表可提升，但 embedding 参数变多。
- **packed 数据太大**：sanity 档有 token 上限（`data.packing.sanity_max_tokens`）；standard 档为全量。
- **偏好数据格式不同**：HH-RLHF 是对话转录，需解析最后一轮；其他数据集（含 prompt/chosen/rejected 字段）可直接映射。
