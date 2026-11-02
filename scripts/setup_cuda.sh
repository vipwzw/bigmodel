#!/usr/bin/env bash
# CUDA 单卡环境初始化：Ubuntu 22.04 + NVIDIA 驱动 550（CUDA 12.4）
#
# 同时支持 Tesla V100 16GB（sm70，无原生 bf16）与 NVIDIA A10 24GB（sm86，原生 bf16）：
#   - 若系统 Python 已预装 CUDA 版 torch（如 PAI-DSW 镜像）→ 复用，只补少量工具依赖；
#   - 否则安装 torch 2.6.0（PyPI cu124 wheel，含 sm70/sm86 内核）+ requirements/cuda.txt；
#   - 训练精度由程序运行时探测：A10 → bf16；V100 → fp16 + GradScaler（见 train/precision.py）。
#
# 用法：
#   bash scripts/setup_cuda.sh
#   PYTHON_BIN=/usr/bin/python3.11 bash scripts/setup_cuda.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_MIRROR="${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "未找到 $PYTHON_BIN；请安装 Python 3.11/3.12 后用 PYTHON_BIN=/path/to/python3 重试" >&2
  exit 1
fi
echo "[1/5] Python: $("$PYTHON_BIN" -V)"

HAS_CUDA_TORCH=0
if "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  HAS_CUDA_TORCH=1
  echo "[1/5] 检测到预装 CUDA 版 torch：$("$PYTHON_BIN" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null | tail -1)"
fi

if [ ! -d .venv ]; then
  echo "[2/5] 创建 .venv（复用系统已装 CUDA 依赖：$HAS_CUDA_TORCH）..."
  if [ "$HAS_CUDA_TORCH" = "1" ]; then
    "$PYTHON_BIN" -m venv --system-site-packages .venv
  else
    "$PYTHON_BIN" -m venv .venv
  fi
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install -q -U pip

if [ "$HAS_CUDA_TORCH" = "1" ]; then
  echo "[3/5] 复用预装 CUDA 依赖：只补 pytest/ruff 等工具包 ..."
  pip install -q pytest ruff
else
  echo "[3/5] 安装 CUDA 依赖（torch 2.6.0 = cu124，约 2.5GB）..."
  if [ -n "$PIP_MIRROR" ]; then
    pip config set global.index-url "$PIP_MIRROR" >/dev/null
  fi
  pip install -q -r requirements/cuda.txt
fi

echo "[4/5] 安装本项目（可编辑模式）..."
if python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  pip install -q --no-deps -e .
else
  # pyproject 要求 >=3.12（macOS 课程环境）；Linux CUDA 服务器常用 3.11，运行时代码已兼容
  pip install -q --no-deps --ignore-requires-python -e .
fi

echo "[5/5] 验证 CUDA 与 GPU 能力 ..."
python - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    sys.exit("CUDA 不可用：请检查 NVIDIA 驱动与 torch 版本是否匹配（python -c 'import torch;print(torch.version.cuda)'）")

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
native_bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"torch {torch.__version__} / CUDA {torch.version.cuda} / GPU {name} / sm{major}{minor}")
print(f"显存 {total_gb:.0f} GB；原生 bf16={'支持（走 bf16 autocast）' if native_bf16 else '不支持（走 fp16 + GradScaler）'}")

if "V100" in name:
    print("推荐配置：--config configs/cuda.yaml --config configs/cuda-v100.yaml")
elif "A10" in name:
    print("推荐配置：--config configs/cuda.yaml --config configs/cuda-a10.yaml")
else:
    print("未识别 GPU：先用 configs/cuda.yaml（micro-batch 保守），跑 lab run 1 校准吞吐后再调")
PY

echo ""
echo "完成。下一步：.venv/bin/lab doctor"
