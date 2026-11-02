"""从零预训练用的 GPT 类模型（PyTorch / Apple Silicon MPS）。

结构取现代小模型常用组件：RoPE、RMSNorm、SwiGLU、因果注意力、可选 GQA。
模型定义在 Stage 1 的性能基线中即被使用，Stage 4 在此基础上实现训练循环。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    ffn_hidden: int | None = None
    n_kv_head: int | None = None
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd({self.n_embd}) 必须能被 n_head({self.n_head}) 整除")
        if self.ffn_hidden is None:
            self.ffn_hidden = _round_to_multiple(int(self.n_embd * 8 / 3), 64)
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head({self.n_head}) 必须能被 n_kv_head({self.n_kv_head}) 整除")
        if self.block_size <= 0 or self.vocab_size <= 0:
            raise ValueError("block_size 与 vocab_size 必须为正整数")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def num_params(self) -> int:
        # _linear_params 已按是否共享词嵌入计算 lm_head，无需再修正
        return self._linear_params()

    def _linear_params(self) -> int:
        qkv = self.n_embd * (self.n_head + 2 * self.n_kv_head) * self.head_dim
        attn_proj = self.n_embd * self.n_embd
        ffn = 3 * self.n_embd * int(self.ffn_hidden)
        norms = 2 * self.n_embd
        per_layer = qkv + attn_proj + ffn + norms
        final_norm = self.n_embd
        embeddings = self.vocab_size * self.n_embd
        lm_head = 0 if self.tie_embeddings else embeddings
        return embeddings + self.n_layer * per_layer + final_norm + lm_head

    def flops_per_token(self) -> float:
        # 训练（前向+反向）近似：6N，加上注意力项
        attention = self.n_layer * 12 * self.block_size * self.n_embd
        return 6 * self.num_params() + attention

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "ffn_hidden": self.ffn_hidden,
            "n_kv_head": self.n_kv_head,
            "rope_theta": self.rope_theta,
            "dropout": self.dropout,
            "tie_embeddings": self.tie_embeddings,
        }


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


# 与 OpenSpec design.md D3 的三档规模保持一致
PRESETS: dict[str, GPTConfig] = {
    "s": GPTConfig(vocab_size=16_384, block_size=512, n_layer=6, n_head=6, n_embd=384),
    "m": GPTConfig(vocab_size=32_768, block_size=768, n_layer=8, n_head=8, n_embd=512),
    "l": GPTConfig(vocab_size=32_768, block_size=1024, n_layer=12, n_head=12, n_embd=768),
}

# design.md 中 standard 档的 token 预算
STANDARD_TOKEN_BUDGET: dict[str, int] = {"s": 100_000_000, "m": 120_000_000, "l": 80_000_000}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def _rope_cache(block_size: int, head_dim: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(block_size, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    dtype = x.dtype
    x = x.float()
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2).to(dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = int(config.n_kv_head)
        self.head_dim = config.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        query = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        query = _apply_rope(query, cos, sin)
        key = _apply_rope(key, cos, sin)
        # 高效 causal attention；packing 时通过 attn_bias 额外屏蔽跨样本
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_bias is None,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, int(config.ffn_hidden), bias=False)
        self.w3 = nn.Linear(config.n_embd, int(config.ffn_hidden), bias=False)
        self.w2 = nn.Linear(int(config.ffn_hidden), config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), cos, sin, attn_bias)
        x = x + self.mlp(self.norm_2(x))
        return x


class GPT(nn.Module):
    """decoder-only 语言模型。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm_f = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        device = torch.device("cpu")
        cos, sin = _rope_cache(config.block_size, config.head_dim, config.rope_theta, device)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        # 残差投影缩放初始化（nanoGPT 风格）
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "w2.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def param_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        batch, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"序列长度 {seq_len} 超过 block_size {self.config.block_size}")
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        hidden = self.drop(self.wte(idx))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    block, hidden, cos, sin, attn_bias, use_reentrant=False
                )
            else:
                hidden = block(hidden, cos, sin, attn_bias)
        hidden = self.norm_f(hidden)
        logits = self.lm_head(hidden)
        if targets is None:
            return None, logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        loss = loss.view(batch, seq_len)
        if loss_mask is not None:
            loss = loss * loss_mask
            return loss.sum() / loss_mask.sum().clamp_min(1.0), logits
        return loss.mean(), logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size :]
            _, logits = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
