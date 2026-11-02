# bigmodel-lab

单机 Apple Silicon 大模型训练全流程实验课程（OpenSpec 规格驱动）。

> 课程文档见 `docs/README.md`；实现计划见 `openspec/changes/add-llm-training-lab/`。

## 快速开始

```bash
uv sync --all-groups
.venv/bin/lab doctor
.venv/bin/lab stages
```

## CUDA 单卡（V100 / A10）

同一套流程也支持 Linux + NVIDIA **V100 16GB（sm70）** 与 **A10 24GB（sm86）**：设备与精度在运行时自动选择
（A10 → bf16 autocast；V100 无原生 bf16 → fp16 autocast + GradScaler），无需改代码。

```bash
bash scripts/setup_cuda.sh            # 建 venv；已预装 CUDA torch 的镜像直接复用
.venv/bin/lab doctor                  # 期望 cuda 一项 PASS
.venv/bin/lab run 2 --profile sanity  # 下载语料 + 清洗/去重
.venv/bin/lab run 3 --profile sanity  # BPE/打包/指令数据
.venv/bin/lab run 5 --config configs/cuda.yaml --config configs/cuda-v100.yaml --profile standard
python scripts/qa_cuda.py --model-path models/exports/qwen3-0.6b-v100-sft-merged
```

A10 机器把 `cuda-v100.yaml` 换成 `configs/cuda-a10.yaml` 即可（V100/A10 均已实机验证）；也可一键跑完整流程：`bash scripts/run_cuda_pipeline.sh`（`QUICK=1` 冒烟档）。详见 [docs/GPU.md](docs/GPU.md) 与 [docs/GPU_QA.md](docs/GPU_QA.md)。

## 目录结构

```
configs/     各阶段默认配置（cuda.yaml / cuda-v100.yaml / cuda-a10.yaml 为 CUDA 单卡配置）
requirements/ CUDA 服务器依赖清单（有/无预装 CUDA torch 两种）
src/bigmodel_lab/  课程实现（CLI、基座、各阶段模块）
docs/stages/ 8 个阶段课程文档
tests/       测试
data/ models/ runs/  资源与实验产物（不入库）
```
