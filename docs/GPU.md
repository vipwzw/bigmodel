# CUDA 单卡（V100 / A10）运行指南

> 目标：同一套 `bigmodel-lab` 代码在 **Linux + NVIDIA 单卡** 上跑通
> 「下载数据 → 清洗/去重 → 分词/打包/指令数据 → LoRA SFT → 测试问答」，
> 同时兼容 **Tesla V100 16GB（sm70）** 与 **NVIDIA A10 24GB（sm86）**。
>
> 已在阿里云 PAI-DSW 的 **V100 与 A10 两台实例**上分别实测（Ubuntu 22.04，驱动 550.163.01，
> CUDA Toolkit 12.4，系统 Python 3.11，镜像预装 `torch 2.8.0+cu128`）。

## 一、双卡支持设计

程序不在代码里写死 GPU 型号或精度，而是运行时探测能力：

| 能力 | V100 16GB（sm70） | A10 24GB（sm86） | 程序行为 |
|---|---|---|---|
| 原生 bf16 | ❌（`is_bf16_supported(including_emulation=False)` = False） | ✅ | `detect_autocast_dtype()` 自动选：V100 → fp16 + `GradScaler`；A10 → bf16 |
| 显存 | 16 GB | 24 GB | `configs/cuda-v100.yaml`（micro-batch 2×512）/ `configs/cuda-a10.yaml`（2×1024） |
| 卡型参数 | 少、fp16 为主 | 多、bf16 | `lab run 1` 实测吞吐写入 benchmark，不套用 Mac 的数字 |

> ⚠️ 关键陷阱：`torch.cuda.is_bf16_supported()` 在 torch ≥2.4 默认包含「软件模拟」，
> V100 上也会返回 `True`，但模拟路径很慢。本仓库统一使用
> `torch.cuda.is_bf16_supported(including_emulation=False)`（封装为
> `bigmodel_lab.train.precision.cuda_bf16_supported()`）判断**原生** bf16。

涉及的自适应代码：

- `src/bigmodel_lab/train/precision.py`：设备解析（cuda → mps → cpu）、精度探测、GradScaler、显存统计；
- `finetune/sft.py`、`train/pretrain.py`：AMP + GradScaler（fp16 时自动启用）、OOM 提示；
- `doctor.py`：`cuda` 检查（设备/显存/驱动/CUDA 版本/原生 bf16），非 macOS 上 MPS/MLX 自动 SKIP；
- `benchmark.py`：CUDA 矩阵运算 + tiny GPT 训练吞吐基线；
- `train/__init__.py`：统一导出上述工具。

## 二、环境安装

### 2.1 推荐：一键脚本

```bash
bash scripts/setup_cuda.sh
```

脚本逻辑（自动适配两类机器）：

1. 若系统 Python 已预装 **CUDA 可用** 的 torch（如 PAI-DSW 镜像）→ 建 `--system-site-packages` 的
   `.venv` 复用，只补 `pytest/ruff`，然后可编辑安装本项目；
2. 否则 → 普通 `.venv` + `pip install -r requirements/cuda.txt`
   （`torch==2.6.0`，PyPI 默认 wheel 即 cu124，同时含 sm70/sm86 内核）；
3. 最后打印 GPU 型号、显存、原生 bf16 支持情况，并提示该用哪份 `configs/cuda-*.yaml`。

### 2.2 手动安装

```bash
# 有预装 CUDA torch 的镜像
python3 -m venv --system-site-packages .venv && . .venv/bin/activate
pip install -r requirements/cuda-runtime.txt
pip install --no-deps --ignore-requires-python -e .

# 干净机器（无 CUDA torch）
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements/cuda.txt
pip install -e .
```

> `--ignore-requires-python` 仅用于 Python 3.11 的 Linux 服务器：`pyproject.toml` 的
> `requires-python` 是 macOS 课程环境（3.12）的约定，代码本身兼容 3.11（`laboratory doctor`
> 在 Linux 上按 ≥3.11 校验）。

## 三、运行流程

> 一键运行：`bash scripts/run_cuda_pipeline.sh`（标准档）；`QUICK=1 bash scripts/run_cuda_pipeline.sh` 为冒烟档。
> V100 实测：QUICK 模式（30 步 LoRA ×2 变体）**5.6 分钟**跑完 Stage 1→3→5→QA，val loss 2.7962→2.3354，976.8 tokens/s，峰值 6.9 GB。

```bash
cd /mnt/workspace/bigmodel        # 持久盘目录（DSW 重启不丢）
.venv/bin/lab doctor              # 期望 cuda: PASS、mps/mlx: SKIP

# 1) 数据：下载语料 + 清洗/去重（ModelScope 直连）
.venv/bin/lab run 2 --profile sanity

# 2) 训练数据：BPE/打包 + 指令/偏好格式化
.venv/bin/lab run 3 --profile sanity

# 3) 环境基线与 GPU 性能（可选，用于校准吞吐）
.venv/bin/lab run 1 --profile sanity

# 4) Qwen3-0.6B LoRA SFT（V100 / A10 二选一）
.venv/bin/lab run 5 --config configs/cuda.yaml --config configs/cuda-v100.yaml --profile standard
# .venv/bin/lab run 5 --config configs/cuda.yaml --config configs/cuda-a10.yaml --profile standard

# 5) 测试数据：固定问题集问答
python scripts/qa_cuda.py --model-path models/exports/qwen3-0.6b-v100-sft-merged
```

产物位置：`runs/stage-N/<run-id>/`（配置快照、日志、指标、LoRA 检查点、生成对比）、
`models/exports/…`（合并后的完整模型）、`runs/qa-cuda/<时间戳>/qa_report.{json,md}`。

> 附：需要更细粒度的 GPU 微基准（fp32/TF32/fp16/bf16 矩阵、显存带宽、频率/功耗、A10 参考峰值）时，可运行 `python scripts/bench_gpu.py`（支持 cuda/mps/cpu，不依赖本项目代码）。

## 四、实测结果（V100 16GB 与 A10 24GB）

> 运行时间：2026-11-02；机器：阿里云 PAI-DSW，Tesla V100-SXM2-16GB（ECC 开启，8 vCPU / 31GB RAM）。
> 环境：Ubuntu 22.04、驱动 550.163.01、CUDA Toolkit 12.4、系统 Python 3.11.11、
> 预装 torch 2.8.0+cu128（arch_list 含 sm_70 与 sm_86）、transformers 4.57.1、peft 0.17.1、modelscope 1.31.0。

### 4.1 环境自检（`lab doctor`）

```
[PASS] cuda            Tesla V100-SXM2-16GB（16 GB，sm70，CUDA 12.8，bf16=不支持→fp16 AMP）
[SKIP] mps / mlx       非 macOS 平台
[WARN] memory          物理内存 31 GB（建议 32GB+；对 0.6B LoRA 无影响）
[WARN] huggingface     直连不可达（数据源默认走 ModelScope）
结论: WARN  (PASS 9 / WARN 2 / FAIL 0 / SKIP 2)
```

### 4.2 GPU 性能基线（Stage 1，sanity 档，2048×2048 矩阵）

| dtype | GFLOPS |
|---|---|
| float32 | 12,394 |
| float16 | **63,960** |
| bfloat16 | 9,139（软件模拟，训练自动改用 fp16） |

tiny GPT 训练吞吐：S 36,767 tok/s（2×512）、M 38,062（2×768）、L 17,680（1×1024）；
峰值显存 0.56 / 1.46 / 2.35 GB；standard token 预算校准 S/M/L ≈ 0.76 / 0.88 / 1.26 小时。

### 4.3 数据工程（Stage 2–3，sanity 档）

| 数据 | 结果 |
|---|---|
| 中文维基 | 20,000 篇 → 清洗保留 19,999 → MinHash 去重 103 篇 → **19,896 篇 / 1,493 万字符** |
| TinyStories | 3,000 篇 → 2,955 篇 |
| BPE 16k / 32k | 1.464 / 1.621 字符·token⁻¹，UNK = 0 |
| packed | 中文 4.0M token（val 104.6k / test 97.2k）、英文 1.0M token |
| 指令数据 | 5,000 条有效 → train 4,750 / val 250 |
| 偏好数据 | 3,000 对 → train 2,850 / val 150 |

### 4.4 Qwen3-0.6B LoRA SFT（Stage 5，standard 档）

| 指标 | V100 实测 |
|---|---|
| 配置 | LoRA r=16 α=32，可训练 **10,092,544（1.67%）**；batch 2 × grad_accum 4 × seq 512；lr 1e-4 |
| 数据 | train 4,736 / val 200（14 条超长丢弃） |
| 步数 / 时长 | 600 步 / **757.8 s（12.6 min）** |
| 吞吐 | **974.7 tokens/s**（738,576 tokens） |
| 显存 | torch 实际分配峰值 **7.9 GB**（`nvidia-smi` 显示 ~15.9 GB，含 CUDA context/缓存与镜像自身开销） |
| 精度路径 | `device=cuda`、`autocast=float16`、`grad_scaler=true` —— V100 自动回落 fp16 AMP 生效 |
| val loss | 2.6461 → 1.9956（100 步）→ 1.9579（200）→ 1.9529（300）→ **1.9344（600 步）** |
| 合并导出 | `models/exports/qwen3-0.6b-v100-sft-merged`，可重新加载生成 ✅ |

> 对照 Mac（M4 Pro，QA 实验 328 步）val loss 1.9527；V100 跑满 600 步后 1.9344，跨平台可复现性良好。
> 端到端冒烟（`QUICK=1`，2 个变体各 30 步）：5.6 分钟完成 Stage 1→3→5→QA，val loss 2.7962→2.3354，976.8 tokens/s，峰值 6.9 GB —— 见 `logs/pipeline-quick.log`。
> 微调前后生成对比见 `runs/stage-5/<run>/artifacts/before_after.md`：基座模型在 thinking 模式下啰嗦、答非所问，
> LoRA 微调后 10 条提示全部给出简洁中文回答（1+1=2、2+3=5、首都北京、最大行星木星等）。

### 4.5 测试数据：固定问题集问答（`scripts/qa_cuda.py`）

训练完成的模型用 10 条固定问题 + 3 轮多轮对话实测（完整记录见 [GPU_QA.md](GPU_QA.md)）：

| 项目 | 实测 |
|---|---|
| 生成模式 | 批量生成（10 条问题一次 forward，摊薄 CPU 调度开销） |
| 聚合吞吐 | **87.2 tokens/s**（逐条生成仅 ~26 tokens/s） |
| 单轮总耗时 | ~17 s（另多轮对话 ~8 s；总 24.8 s） |
| 空回答 | 0 / 10 |
| 客观题 | 1+1=2、2+3=5、首都北京、最大行星木星 —— **4/4 正确** |
| 长回答题 | 概念解释基本通顺，仍有重复与细节错误（600 步 / 4.7k 样本的预期上限） |

> 生成速度说明：本 DSW 共享 vCPU 上单 token 生成受 CPU 调度限制（单层 forward ~1.9 ms，其中 GPU 仅 ~0.27 ms），
> 批量生成把聚合吞吐从 26 提升到 87 tokens/s。该数字只反映「本机 + 本测试集」，不代表 A10/V100 硬件上限；
> 换机器请以 `lab run 1` 与 Stage 5 的 tokens/s 为准。

### 4.6 A10 实测结果（同一套代码，自动切到 bf16 路径）

A10 上运行 `bash scripts/run_cuda_pipeline.sh`（自动识别 A10 → `configs/cuda-a10.yaml`），
**无需改一行代码**；全流程（doctor→Stage1→2→3→5→QA，数据/模型已缓存）**约 9 分钟**。

| 项目 | A10 实测（对照 V100） |
|---|---|
| `lab doctor` cuda | `NVIDIA A10（22 GB，sm86，CUDA 12.8，bf16=支持）` |
| 矩阵 GFLOPS | fp32 **18,645** / fp16 **84,505** / bf16 **88,156**（原生，无模拟标注） |
| tiny GPT 训练 | bf16：S **55,125** / M **49,138** / L **19,390** tok/s（V100 为 36,767 / 38,062 / 17,680） |
| SFT 配置 | 400 步，seq 1024，batch 2 × grad_accum 4；train 4,750 / val 200 |
| 精度路径 | **`autocast=bfloat16`、`grad_scaler=false`**（与设计一致） |
| val loss | 2.6456 → 2.0034（100 步）→ 1.9780（200）→ 1.9657（300）→ **1.9598（400 步）** |
| 吞吐 / 时长 | **1,677.3 tokens/s** / 294.4 s（4.9 min）—— 约为 V100 的 **1.72 倍** |
| 显存 | torch 分配峰值 **10.5 GB**（seq 1024 比 V100 更长，故高于 V100 的 7.9 GB） |
| 测试问答（10 问） | 批量生成 **127.9 tokens/s**（V100 为 87.2）、总 14.8 s、0 空回答、客观题 4/4 正确 |
| 导出 | `models/exports/qwen3-0.6b-a10-sft-merged`，可加载生成 ✅ |

> 结论：同一份代码在 V100（fp16 + GradScaler）与 A10（bf16）上均按自动探测结果运行，
> A10 的 bf16 路径吞吐与生成速度均优于 V100，符合硬件预期。
> A10 问答全文见 [GPU_QA.md](GPU_QA.md) 第五节（多轮对话中出现了"40 个 CUDA 核心 / 16GB GDDR6"这类幻觉，
> 属 600 步级 SFT 的知识上限，非框架问题）。

## 五、省内存方案与更大模型（A10 24GB 实测）

A10 24GB 上能调多大的 Qwen，取决于「权重 + 激活 + 优化器」三层压缩。仓库已内置 4 个开关（无需改代码，配置即用）：

| 开关（YAML 键） | 作用 | 收益 | 代价 |
|---|---|---|---|
| `base_dtype: bfloat16` | 基座以 bf16 载入（原默认 fp32） | 权重显存 **-50%** | 无 |
| `gradient_checkpointing: true` | 激活重算 | 激活显存 **-50~70%** | 吞吐约 -45% |
| `quantization: 4bit` | QLoRA（NF4 + double quant，需 bitsandbytes） | 权重再 **-75%** | 略降质量；仅 CUDA |
| `attn_implementation: flash_attention_2` | FlashAttention-2（需 flash-attn） | 长序列激活更低、更快 | 编译安装较重；缺依赖自动回落 sdpa |
| `batch_size / max_seq_len / grad_accum` | 显存随 micro-batch 与序列线性变化 | 按需调节 | 训练变慢 |

### 5.1 A10 实测（同一套流水线，配置不同）

| 档位 | 配置 | 峰值显存 | 吞吐 | 训练结果 |
|---|---|---|---|---|
| Qwen3-0.6B（旧默认） | fp32 载入、无 checkpointing、bs2×ga4×seq1024 | **10.5 GB** | 1,677 tok/s | 400 步，val 2.6456→1.9598 |
| Qwen3-0.6B（省内存） | bf16 + checkpointing + sdpa，同 batch | **3.65 GB（-65%）** | 930 tok/s | 60 步，val 2.6482→2.0427 |
| **Qwen3-4B** | bf16 + checkpointing + FA2，bs1×ga8×seq512 | **9.16 GB** | 353 tok/s | 60 步，val 3.0803→1.5571（可训练 33.0M，0.81%） |
| **Qwen3-14B（QLoRA）** | bf16 载入 + **4bit** + checkpointing + FA2，bs1×ga8×seq512 | **15.9 GB** | 140 tok/s | 20 步，val 2.0746→1.3019（可训练 64.2M，0.78%） |

> 0.6B 省内存档把峰值从 10.5GB 压到 3.65GB（同样的 batch/seq），代价是吞吐从 1,677 降到 930 tok/s；
> 4B 与 14B 都在 24GB 内真实完成训练并导出 adapter，说明这套开关在 A10 上可用。
>
> 推理侧实测（10 条固定问题，批量生成）：0.6B **127.9** tokens/s；4B（bf16）**122.1** tokens/s、总 92.9s；
> 14B（4bit）**20.5** tokens/s、总 130.1s（反量化开销，比 4B 慢约 6 倍）。问答全文见
> [GPU_QA.md](GPU_QA.md) 第五~七节。

### 5.2 容量上限（A10 24GB，按实测外推）

| 方式 | 推荐 | 极限 | 说明 |
|---|---|---|---|
| bf16 LoRA | **Qwen3-4B**（实测 9.2GB） | Qwen3-8B（约 18–20GB，需 checkpointing + bs1） | 14B bf16 权重就 28GB，放不下 |
| **QLoRA 4bit** | **Qwen3-14B**（实测 15.9GB） | Qwen3-32B（约 22–24GB，seq 512/bs1） | 量化基座不能合并导出，推理用基座+adapter |
| 全参微调 | Qwen3-0.6B | Qwen3-1.7B（需 8bit Adam + checkpointing，边缘） | 优化器状态是大头 |

### 5.3 用法

```bash
# 0.6B 省内存回归（bf16 + checkpointing）
.venv/bin/lab run 5 --config configs/cuda.yaml --config configs/cuda-a10-0.6b-mem.yaml --profile standard

# 4B bf16 LoRA（自动下载模型；约 10 分钟/60 步）
.venv/bin/lab run 5 --config configs/cuda-a10-4b.yaml --profile standard
python scripts/qa_cuda.py --model-path models/exports/qwen3-4b-a10-sft-merged --out-dir runs/qa-cuda/4b

# 14B 4bit QLoRA（需 bitsandbytes；量化基座不合并，adapter 单独推理）
.venv/bin/lab run 5 --config configs/cuda-a10-14b-qlora.yaml --profile standard
python scripts/qa_cuda.py --model-path models/Qwen3-14B --quantization 4bit \
    --adapter runs/stage-5/<run>/artifacts/adapter_a10_14b_qlora --out-dir runs/qa-cuda/14b
```

### 5.4 踩坑与限制

- **24GB 上 14B QLoRA 不要同时加载两份模型**：stage5 的「基座 vs 微调生成对比」会依次加载两个 14B 4bit（约 20GB），
  第二个加载时 OOM。14B 配置已设 `generation.enabled: false`，训练后用 `qa_cuda.py` 单模型验证。
- **量化基座不合并导出**：stage5 会跳过 merge，只保留 adapter；推理用 `qa_cuda.py --quantization 4bit --adapter <dir>`。
- **checkpointing 是拿速度换显存**：0.6B 上吞吐 -45%；显存宽松时可关掉。
- **flash-attn 安装**：本机实测 `flash-attn 2.7.4.post1` 可在 A10 + torch 2.8/cu128 编译成功；未安装时程序自动回落 `sdpa`。
- **量化仅 CUDA**：MPS/CPU 上 `quantization` 会直接报错并提示改用 bf16 载入（Mac 48GB 统一内存也不需要量化）。

## 六、与 Apple Silicon 版本的差异

| 项目 | Apple Silicon（MPS/MLX） | CUDA 单卡（V100/A10） |
|---|---|---|
| 精度路径 | bf16 autocast（MPS） | A10 bf16；V100 fp16 + GradScaler |
| Stage 1 基线 | MPS + MLX 对照 | CUDA 矩阵/训练基线（MLX 自动跳过） |
| Stage 4/6/7/8 | 完整（含 MLX 量化） | 本次范围外：Stage 4/6/7 使用同一训练/评测代码可跑，Stage 8 的 MLX 量化在 Linux 不可用 |
| 显存统计 | `driver_allocated_memory` | `torch.cuda.memory_allocated / max_memory_allocated` |

## 七、常见问题

- **`uv python install 3.12` 卡住**：python-build-standalone 走 GitHub，大陆机器常超时；
  `scripts/setup_cuda.sh` 已改为直接用系统 Python（≥3.11），不依赖该下载。
- **pip 下载 torch 很慢**：优先复用镜像预装的 CUDA torch；必须装时用
  `PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/`（脚本默认）。
- **OOM**：`--set stage5.train.standard.batch_size=1 --set stage5.train.standard.max_seq_len=384`；
  训练器会拦截 OOM 并提示可调参数。
- **`lab doctor` 里 hf-mirror/huggingface 为 WARN**：服务器直连 ModelScope 即可，
  `data.resources` 默认来源就是 ModelScope，不影响数据下载。
