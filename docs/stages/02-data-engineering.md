# Stage 2：数据工程

> 对应 capability：`data-pipeline`（资源获取、清洗、去重、统计、数据卡）
> 前置：Stage 1。产物：`data/processed/stage2/*.jsonl` + 数据卡/统计报告。

## 学习目标

1. 理解大模型训练中"数据决定上限"：语料来源、许可证、格式、规模与质量统计。
2. 亲手实现工业界数据管线的核心步骤：**获取 → 校验 → 规范化 → 过滤 → 精确/近似去重 → 数据卡**。
3. 学会用 manifest + 校验和保证"同一份数据可复现"，而不是每次重新下载。

## 数据资源（本机实测可用）

| 数据集 | 用途 | 来源 | 规模 | 许可证 |
|---|---|---|---|---|
| `AI-ModelScope/wikipedia-cn-20230720-filtered` | 中文预训练语料 | ModelScope | 496.6 MB（JSONL，字段 `completion`） | CC BY-SA 4.0 |
| `AI-ModelScope/TinyStories` | 英文小语料（快速兜底/对照） | ModelScope | 246.7 MB（2 个 parquet 分片） | CDLA-Sharing-1.0 |
| `AI-ModelScope/alpaca-data-gpt4-chinese` | 中文指令微调（Stage 3/5 用） | ModelScope | 74.8 MB（JSONL） | Apache-2.0 |
| `AI-ModelScope/hh-rlhf` | 偏好数据（Stage 3/6 用） | ModelScope | test 分片约 7 MB | MIT |

## 运行命令

```bash
# 完整运行（sanity 档：20k 中文文档 + 3k 英文文档，约 1 分钟，不含首次下载）
.venv/bin/lab run 2 --profile sanity

# 正式运行（全量，首次下载约 800 MB）
.venv/bin/lab run 2 --profile standard

# 离线复用缓存（训练/评测阶段默认离线；数据阶段可用 --offline 强制离线）
.venv/bin/lab run 2 --offline
```

首次运行会把资源下载到 `data/raw/<key>/`，并写入 `fetch_manifest.json`（来源、文件、大小、sha256）。
之后运行命中缓存，不再联网。

## 实测结果（sanity 档，2026-10-29）

| 数据集 | 输入文档 | 保留文档 | 保留字符 | 保留率 | 重复率（近似） | 输出大小 |
|---|---|---|---|---|---|---|
| pretrain_zh | 20,000 | 19,896 | 14,938,037 | 100% | 0.52%（103 篇） | 38 MB |
| pretrain_en | 3,000 | 2,955 | 2,493,373 | 98.5% | 0% | 2.7 MB |

> 维基中文长文本质量很高：只丢弃 1 篇（重复字符比例过高），近似去重命中 103 篇（模板化条目）。
> TinyStories 丢弃 45 篇，主要是过短/符号过多的故事。

## 产物说明

| 路径 | 内容 |
|---|---|
| `data/processed/stage2/pretrain_zh.jsonl` | 清洗去重后的中文语料（`doc_id`/`source`/`language`/`text`） |
| `data/processed/stage2/pretrain_en.jsonl` | 清洗去重后的英文语料 |
| `runs/stage-2/<run-id>/artifacts/fetch_manifest.json` | 资源获取记录（来源、文件、大小、sha256） |
| `runs/stage-2/<run-id>/artifacts/clean_stats.json` | 每个数据集的清洗/去重统计 |
| `runs/stage-2/<run-id>/artifacts/dataset_card.md` | 数据卡（来源、许可、规模、样例、已知偏差） |

## 管线机制说明

1. **多源优先级**：`cache → ModelScope → hf-mirror → 官方 HF（需代理）`；任一来源失败自动尝试下一个。
2. **校验和**：下载后记录每个文件的 sha256；缓存命中判定校验文件大小（可选校验 hash）。
3. **清洗规则**（`configs/default.yaml` 的 `data.clean`）：NFKC 规范化、控制字符清理、空白折叠、
   长度过滤（80–200,000 字符）、唯一字符比例、符号比例、中文 CJK 比例、英文字母比例。
4. **去重**：精确去重（规范化文本 sha1）+ MinHash/LSH 近似去重（默认 Jaccard ≥ 0.8）。
5. **数据卡**：每个数据集自动生成来源/许可证/规模/长度分位数/过滤原因/样例/已知偏差。

## 本机做法 ↔ 生产栈对照

| 本机（单机、小规模） | 生产级数据管线 | 差异 |
|---|---|---|
| ModelScope/HF 下载 + sha256 | 对象存储 + 数据湖（S3/OSS）+ 数据版本 | 本机无数据湖，用 manifest 代替版本管理 |
| Python 流式清洗（单进程） | Spark / Ray Data 分布式清洗 | 本机数据量小（GB 级），单进程足够 |
| MinHash/LSH 近似去重 | 大规模分布式 MinHash（如 datasketch + Spark） | 算法一致，规模不同 |
| 人工抽检 + 统计规则 | 数据质量门禁（规则 + 模型打分 + 人工评审） | 本机实现规则门禁，模型打分留作扩展 |
| `dataset_card.md` | 数据卡/数据血缘平台 | 格式对齐，无平台化 |

## 常见问题

- **首次下载慢**：ModelScope 直连约 4–8 MB/s；断点续传由 modelscope SDK 处理。
- **缓存命中失败**：检查 `fetch_manifest.json` 是否与文件大小一致；删除 manifest 可强制重新校验。
- **磁盘**：全量资源约 800 MB，处理后数据约 40 MB（sanity）/数百 MB（standard）。
