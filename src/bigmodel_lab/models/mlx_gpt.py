"""MLX 对照实现：与 PyTorch GPT 同结构，可加载同一权重做数值对齐与吞吐对比。"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .gpt import GPT, GPTConfig

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import AdamW
    from mlx.utils import tree_flatten, tree_unflatten

    MLX_AVAILABLE = True
except Exception:  # noqa: BLE001 - 环境缺 mlx 时保持可导入
    MLX_AVAILABLE = False
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def require_mlx() -> None:
    if not MLX_AVAILABLE:
        raise RuntimeError("未安装 MLX：请执行 uv sync --group mlx")



if MLX_AVAILABLE:

    def _rope_cos_sin(seq_len: int, head_dim: int, theta: float):
        inverse = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        positions = mx.arange(seq_len).astype(mx.float32)
        frequencies = mx.outer(positions, inverse)
        return mx.cos(frequencies), mx.sin(frequencies)


    def _apply_rope(x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos = cos[: x.shape[-2]][None, None, :, :]
        sin = sin[: x.shape[-2]][None, None, :, :]
        rotated = mx.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return rotated.reshape(x.shape)


    class MLXAttention(nn.Module):
        def __init__(self, config: GPTConfig):
            super().__init__()
            self.n_head = config.n_head
            self.n_kv_head = int(config.n_kv_head)
            self.head_dim = config.head_dim
            kv_dim = self.n_kv_head * self.head_dim
            self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.k_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
            self.v_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
            self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.rope_theta = config.rope_theta

        def __call__(self, x):
            batch, seq_len, _ = x.shape
            cos, sin = _rope_cos_sin(seq_len, self.head_dim, self.rope_theta)
            query = self.q_proj(x).reshape(batch, seq_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            key = self.k_proj(x).reshape(batch, seq_len, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
            value = self.v_proj(x).reshape(batch, seq_len, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
            query = _apply_rope(query, cos, sin)
            key = _apply_rope(key, cos, sin)
            if self.n_kv_head != self.n_head:
                repeat = self.n_head // self.n_kv_head
                key = mx.repeat(key, repeat, axis=1)
                value = mx.repeat(value, repeat, axis=1)
            mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
            out = mx.fast.scaled_dot_product_attention(
                query, key, value, scale=1.0 / math.sqrt(self.head_dim), mask=mask
            )
            out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
            return self.o_proj(out)


    class MLXSwiGLU(nn.Module):
        def __init__(self, config: GPTConfig):
            super().__init__()
            hidden = int(config.ffn_hidden)
            self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
            self.w3 = nn.Linear(config.n_embd, hidden, bias=False)
            self.w2 = nn.Linear(hidden, config.n_embd, bias=False)

        def __call__(self, x):
            return self.w2(nn.silu(self.w1(x)) * self.w3(x))


    class MLXBlock(nn.Module):
        def __init__(self, config: GPTConfig):
            super().__init__()
            self.norm_1 = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
            self.attn = MLXAttention(config)
            self.norm_2 = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
            self.mlp = MLXSwiGLU(config)

        def __call__(self, x):
            x = x + self.attn(self.norm_1(x))
            x = x + self.mlp(self.norm_2(x))
            return x


    class MLXGPT(nn.Module):
        def __init__(self, config: GPTConfig):
            super().__init__()
            self.config = config
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)
            self.blocks = [MLXBlock(config) for _ in range(config.n_layer)]
            self.norm_f = nn.RMSNorm(config.n_embd, eps=config.norm_eps)

        def __call__(self, idx):
            hidden = self.wte(idx)
            for block in self.blocks:
                hidden = block(hidden)
            hidden = self.norm_f(hidden)
            # 与 PyTorch 实现一致：词嵌入与输出层共享权重
            return mx.matmul(hidden, self.wte.weight.T)


    def load_from_torch(mlx_model: MLXGPT, torch_model: GPT) -> None:
        """把 PyTorch 模型权重原样搬到 MLX 模型（结构一致时可直接对齐）。"""
        require_mlx()
        state = torch_model.state_dict()
        flat = dict(tree_flatten(mlx_model.parameters()))
        staged = []
        for name, _ in flat.items():
            if name not in state:
                raise KeyError(f"MLX 参数 {name} 在 PyTorch 模型中不存在")
            staged.append((name, mx.array(state[name].detach().cpu().float().numpy())))
        mlx_model.update(tree_unflatten(staged))
        mx.eval(mlx_model.parameters())


    def mlx_loss_fn(model: MLXGPT, inputs, targets, loss_mask):
        logits = model(inputs)
        vocab = logits.shape[-1]
        losses = nn.losses.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="none")
        mask = loss_mask.reshape(-1)
        return (losses * mask).sum() / mx.maximum(mask.sum(), 1.0)


    def _tree_l2_norm(tree) -> float:
        total = 0.0
        for _, array in tree_flatten(tree):
            total += float((array.astype(mx.float32) ** 2).sum().item())
        return math.sqrt(total)


    def parity_check(torch_model: GPT, mlx_model: MLXGPT, inputs: np.ndarray, targets: np.ndarray, loss_mask: np.ndarray) -> dict[str, Any]:
        """同一权重、同一批次、fp32 下比较 loss / logits / 梯度范数。"""
        import torch  # noqa: PLC0415

        torch_model = torch_model.to("cpu").float().train()
        loss_tensor, logits_torch = torch_model(
            torch.from_numpy(inputs.astype(np.int64)),
            torch.from_numpy(targets.astype(np.int64)),
            torch.from_numpy(loss_mask.astype(np.float32)),
        )
        torch_loss = float(loss_tensor.item())
        loss_tensor.backward()
        torch_grad_norm = math.sqrt(sum(float((parameter.grad**2).sum().item()) for parameter in torch_model.parameters() if parameter.grad is not None))
        torch_model.zero_grad(set_to_none=True)

        mx_inputs = mx.array(inputs)
        mx_targets = mx.array(targets)
        mx_mask = mx.array(loss_mask.astype(np.float32))
        logits_mlx = mlx_model(mx_inputs)
        mlx_loss = mlx_loss_fn(mlx_model, mx_inputs, mx_targets, mx_mask)
        mx.eval(logits_mlx, mlx_loss)
        loss_and_grad = nn.value_and_grad(mlx_model, mlx_loss_fn)
        mlx_loss_grad, mlx_grads = loss_and_grad(mlx_model, mx_inputs, mx_targets, mx_mask)
        mx.eval(mlx_loss_grad, mlx_grads)
        mlx_grad_norm = _tree_l2_norm(mlx_grads)

        logits_mlx_np = np.array(logits_mlx.astype(mx.float32).tolist())
        logits_torch_np = logits_torch.detach().float().numpy()
        loss_scale = max(abs(torch_loss), 1e-9)
        tolerance = {"loss_rel_diff": 1e-3, "grad_norm_rel_diff": 1e-2}
        report = {
            "torch_loss": torch_loss,
            "mlx_loss": float(mlx_loss_grad.item()),
            "loss_rel_diff": abs(torch_loss - float(mlx_loss_grad.item())) / loss_scale,
            "logits_max_abs_diff": float(np.max(np.abs(logits_mlx_np - logits_torch_np))),
            "logits_rel_diff": float(np.max(np.abs(logits_mlx_np - logits_torch_np)) / max(float(np.max(np.abs(logits_torch_np))), 1e-9)),
            "torch_grad_norm": torch_grad_norm,
            "mlx_grad_norm": mlx_grad_norm,
            "grad_norm_rel_diff": abs(torch_grad_norm - mlx_grad_norm) / max(torch_grad_norm, 1e-9),
            "tolerance": tolerance,
        }
        report["passed"] = (
            report["loss_rel_diff"] <= tolerance["loss_rel_diff"]
            and report["grad_norm_rel_diff"] <= tolerance["grad_norm_rel_diff"]
        )
        return report


    def mlx_throughput(
        config: GPTConfig,
        *,
        batch_size: int,
        seq_len: int,
        steps: int,
        seed: int = 42,
        warmup: int = 2,
    ) -> dict[str, Any]:
        """MLX 训练吞吐：训练步为前向+反向+AdamW 更新。"""
        require_mlx()
        key = mx.random.key(seed)
        keys = mx.random.split(key, 3)
        model = MLXGPT(config)
        optimizer = AdamW(learning_rate=3e-4)
        loss_and_grad = nn.value_and_grad(model, mlx_loss_fn)
        inputs = mx.random.randint(0, config.vocab_size, (batch_size, seq_len), key=keys[0])
        targets = mx.random.randint(0, config.vocab_size, (batch_size, seq_len), key=keys[1])
        loss_mask = mx.ones((batch_size, seq_len), dtype=mx.float32)
        timed_steps = 0
        started = None
        for step in range(steps + warmup):
            if step == warmup:
                started = time.perf_counter()
            loss_and_grad_value, grads = loss_and_grad(model, inputs, targets, loss_mask)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_and_grad_value)
            timed_steps += 1 if step >= warmup else 0
        elapsed = time.perf_counter() - started if started is not None else 0.0
        tokens = timed_steps * batch_size * seq_len
        return {
            "backend": "mlx",
            "params": config.num_params(),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "timed_steps": timed_steps,
            "tokens_per_second": round(tokens / max(elapsed, 1e-9), 1),
            "step_seconds": round(elapsed / max(timed_steps, 1), 4),
            "final_loss": float(loss_and_grad_value.item()) if hasattr(loss_and_grad_value, "item") else float(np.array(loss_and_grad_value)),
        }

else:  # pragma: no cover - Linux / 无 MLX 环境：保持模块可导入，调用时报错

    class MLXAttention:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_mlx()

    class MLXSwiGLU(MLXAttention):  # type: ignore[no-redef]
        ...

    class MLXBlock(MLXAttention):  # type: ignore[no-redef]
        ...

    class MLXGPT(MLXAttention):  # type: ignore[no-redef]
        ...

    def load_from_torch(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        require_mlx()

    def mlx_loss_fn(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        require_mlx()

    def _tree_l2_norm(tree):  # type: ignore[no-redef]
        require_mlx()

    def parity_check(*args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[no-redef]
        require_mlx()

    def mlx_throughput(*args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[no-redef]
        require_mlx()
