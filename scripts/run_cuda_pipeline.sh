#!/usr/bin/env bash
# CUDA 单卡一键完整流程（V100 / A10 自动识别）：
#   环境自检 → GPU 基线 → 数据下载/清洗 → 训练数据（分词/打包/指令）→ LoRA SFT → 测试数据问答
#
# 用法：
#   bash scripts/run_cuda_pipeline.sh             # 标准档（V100 约 15–20 分钟；A10 更快）
#   QUICK=1 bash scripts/run_cuda_pipeline.sh     # 快速冒烟（sanity：30 步 LoRA，约 5 分钟）
#
# 说明：数据与模型已缓存时重复运行会直接命中缓存；每一步的完整产物在 runs/stage-N/<run-id>/ 下。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "未找到 .venv：先执行 bash scripts/setup_cuda.sh" >&2
  exit 1
fi

GPU_NAME="$("$PY" -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null | tail -1)"

if [ "${QUICK:-0}" = "1" ]; then
  PROFILE=sanity
  STAGE5_ARGS=()
  MODEL_DIR="models/exports/qwen3-0.6b-sft-lora-merged"
else
  PROFILE=standard
  case "$GPU_NAME" in
    *V100*) STAGE5_ARGS=(--config configs/cuda.yaml --config configs/cuda-v100.yaml); GPU_OVERLAY=configs/cuda-v100.yaml ;;
    *A10*)  STAGE5_ARGS=(--config configs/cuda.yaml --config configs/cuda-a10.yaml);  GPU_OVERLAY=configs/cuda-a10.yaml ;;
    *)      STAGE5_ARGS=(--config configs/cuda.yaml); GPU_OVERLAY=""; echo "未识别 GPU（$GPU_NAME）：使用通用配置 configs/cuda.yaml" ;;
  esac
  export GPU_OVERLAY
  MODEL_DIR="$("$PY" - <<'PY'
import os

from bigmodel_lab.config import load_config

paths = ["configs/cuda.yaml"]
for name in ("configs/cuda-v100.yaml", "configs/cuda-a10.yaml"):
    if os.environ.get("GPU_OVERLAY") == name:
        paths.append(name)
raw, _ = load_config(*paths)
print((raw.get("stage5") or {}).get("export", {}).get("dir", "models/exports/qwen3-0.6b-cuda-sft-merged"))
PY
)"
fi

echo "== GPU: ${GPU_NAME:-未知} / profile: $PROFILE =="

echo "[1/5] 环境自检"
"$PY" -m bigmodel_lab.cli doctor || true

echo "[2/5] GPU 性能基线（Stage 1）"
"$PY" -m bigmodel_lab.cli run 1 --profile sanity

echo "[3/5] 数据下载 + 清洗/去重（Stage 2）"
"$PY" -m bigmodel_lab.cli run 2 --profile sanity

echo "[4/5] 训练数据：分词/打包/指令格式化（Stage 3）"
"$PY" -m bigmodel_lab.cli run 3 --profile sanity

echo "[5/5] Qwen3-0.6B LoRA SFT + 测试数据问答（Stage 5 + QA）"
"$PY" -m bigmodel_lab.cli run 5 "${STAGE5_ARGS[@]}" --profile "$PROFILE"
echo "测试数据：$MODEL_DIR"
"$PY" scripts/qa_cuda.py --model-path "$MODEL_DIR" --out-dir runs/qa-cuda/latest

echo ""
echo "完成：数据/训练产物在 runs/ 与 models/exports/，问答记录在 runs/qa-cuda/latest/qa_report.md"
