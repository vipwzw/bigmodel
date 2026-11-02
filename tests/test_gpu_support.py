"""GPU 设备/精度抽象测试：V100（fp16+GradScaler）与 A10（bf16）两条路径的判定逻辑。

这些测试不依赖真实 GPU：
- 用 `cuda_autocast_name` 直接验证「是否原生 bf16」→「选 bf16 还是 fp16」的决策；
- 其余测试验证当前机器上的设备解析、CPU 回退与 doctor 状态汇总行为。
"""

from __future__ import annotations

import torch

from bigmodel_lab.benchmark import gpu_backend
from bigmodel_lab.doctor import CheckResult, check_cuda, check_mps, overall_status
from bigmodel_lab.train.precision import (
    autocast_context,
    cuda_autocast_name,
    cuda_available,
    cuda_bf16_supported,
    default_device,
    default_dtype_name,
    detect_autocast_dtype,
    device_report,
    device_type,
    make_grad_scaler,
    needs_grad_scaler,
    resolve_device,
)


def test_cuda_autocast_choice_covers_both_gpus():
    """A10（原生 bf16）→ bf16；V100（无原生 bf16）→ fp16。"""
    assert cuda_autocast_name(True) == "bfloat16"
    assert cuda_autocast_name(False) == "float16"


def test_native_bf16_check_is_bool():
    """V100 上 torch.cuda.is_bf16_supported() 默认含模拟会返回 True，本函数必须显式排除模拟。"""
    assert isinstance(cuda_bf16_supported(), bool)


def test_default_device_is_known():
    assert default_device() in {"cuda", "mps", "cpu"}
    assert device_type("cuda:1") == "cuda"
    assert device_type("cpu") == "cpu"


def test_resolve_device_falls_back_when_unavailable():
    assert resolve_device("cpu") == "cpu"
    if not cuda_available():
        assert resolve_device("cuda") == "cpu"


def test_cpu_path_has_no_autocast_or_scaler():
    assert detect_autocast_dtype("cpu") == (None, "float32")
    assert default_dtype_name("cpu") == "float32"
    assert make_grad_scaler("cpu", None) is None
    with autocast_context("cpu", None):
        pass


def test_grad_scaler_only_needed_for_fp16():
    assert needs_grad_scaler(torch.float16)
    assert not needs_grad_scaler(torch.bfloat16)
    assert not needs_grad_scaler(None)
    # CPU 上即使 dtype 是 fp16 也不做梯度缩放
    assert make_grad_scaler("cpu", torch.float16) is None


def test_device_report_shape():
    report = device_report("cpu")
    assert report["kind"] == "cpu"
    assert report["name"] == "CPU"
    assert report["device"] == "cpu"


def test_doctor_cuda_status_and_skip_handling():
    result = check_cuda()
    assert isinstance(result, CheckResult)
    assert result.name == "cuda"
    assert result.status in {"PASS", "WARN", "FAIL", "SKIP"}
    mps = check_mps()
    assert mps.name == "mps"
    # SKIP 不应把总体结论拉低
    assert overall_status([CheckResult("a", "PASS", ""), CheckResult("b", "SKIP", "")]) == "PASS"


def test_benchmark_gpu_backend_detection():
    assert gpu_backend() in {"cuda", "mps", None}
