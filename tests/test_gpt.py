"""GPT 模型结构与配置测试。"""

from __future__ import annotations

import pytest
import torch

from bigmodel_lab.models.gpt import GPT, PRESETS, GPTConfig


def test_preset_param_counts_match_design():
    expected = {"s": 16.9e6, "m": 41.7e6, "l": 110.1e6}
    for name, config in PRESETS.items():
        assert config.num_params() == pytest.approx(expected[name], rel=0.02)


def test_forward_shapes_and_loss():
    config = GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=32, ffn_hidden=64)
    model = GPT(config).eval()
    inputs = torch.randint(0, config.vocab_size, (3, 16))
    targets = torch.randint(0, config.vocab_size, (3, 16))
    loss_mask = torch.ones(3, 16)
    loss, logits = model(inputs, targets, loss_mask)
    assert logits.shape == (3, 16, config.vocab_size)
    assert torch.isfinite(loss)
    assert model.param_count() == config.num_params()


def test_causality_no_future_leak():
    config = GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=32, ffn_hidden=64)
    model = GPT(config).eval()
    first = torch.randint(0, config.vocab_size, (1, 16))
    second = first.clone()
    second[0, 10] = (second[0, 10] + 1) % config.vocab_size
    with torch.no_grad():
        _, logits_first = model(first)
        _, logits_second = model(second)
    assert torch.allclose(logits_first[0, :10], logits_second[0, :10], atol=1e-5)
    assert not torch.allclose(logits_first[0, 10:], logits_second[0, 10:], atol=1e-5)


def test_invalid_config_raises():
    with pytest.raises(ValueError, match="整除"):
        GPTConfig(vocab_size=100, block_size=8, n_layer=1, n_head=3, n_embd=32)
    with pytest.raises(ValueError, match="整除"):
        GPTConfig(vocab_size=100, block_size=8, n_layer=1, n_head=2, n_embd=32, n_kv_head=3)


def test_gqa_and_flops_report():
    config = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=4, n_embd=64, n_kv_head=2, ffn_hidden=128)
    model = GPT(config)
    assert config.flops_per_token() > 6 * config.num_params()
    assert sum(parameter.numel() for parameter in model.parameters()) == model.param_count()


def test_gradient_checkpointing_trains():
    config = GPTConfig(vocab_size=64, block_size=8, n_layer=2, n_head=2, n_embd=32, ffn_hidden=64)
    model = GPT(config).train()
    model.set_gradient_checkpointing(True)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    loss, _ = model(inputs, targets)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_generate_shapes():
    config = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32, ffn_hidden=64)
    model = GPT(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    output = model.generate(prompt, max_new_tokens=5, temperature=0.8)
    assert output.shape == (1, 9)
