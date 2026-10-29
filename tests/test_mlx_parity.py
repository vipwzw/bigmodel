"""MLX 数值对齐与吞吐测试。"""

from __future__ import annotations

import numpy as np
import pytest

from bigmodel_lab.models.gpt import GPT, GPTConfig
from bigmodel_lab.models.mlx_gpt import MLX_AVAILABLE, MLXGPT, load_from_torch, mlx_throughput, parity_check

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="未安装 MLX")


def _tiny_config() -> GPTConfig:
    return GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64, ffn_hidden=128)


def test_parity_loss_logits_and_gradients():
    config = _tiny_config()
    torch_model = GPT(config)
    mlx_model = MLXGPT(config)
    load_from_torch(mlx_model, torch_model)
    rng = np.random.default_rng(0)
    inputs = rng.integers(0, config.vocab_size, (2, 31), dtype=np.int32)
    targets = rng.integers(0, config.vocab_size, (2, 31), dtype=np.int32)
    loss_mask = np.ones((2, 31), dtype=np.float32)
    report = parity_check(torch_model, mlx_model, inputs, targets, loss_mask)
    assert report["loss_rel_diff"] < 1e-3
    assert report["logits_max_abs_diff"] < 1e-3
    assert report["grad_norm_rel_diff"] < 1e-2
    assert report["passed"] is True


def test_mlx_throughput_positive():
    config = GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=64, ffn_hidden=128)
    result = mlx_throughput(config, batch_size=2, seq_len=64, steps=3, warmup=1)
    assert result["backend"] == "mlx"
    assert result["tokens_per_second"] > 0
    assert result["timed_steps"] == 3
    assert result["step_seconds"] > 0
