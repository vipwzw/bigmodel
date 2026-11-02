"""数据划分与序列打包（定长序列 + 损失掩码 + 段掩码）。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

PAD_SEGMENT = np.int32(-1)


def split_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float = 0.0,
) -> dict[str, list[dict[str, Any]]]:
    """按种子确定性划分数据集。"""
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    n_total = len(order)
    n_val = max(1, int(n_total * val_ratio)) if val_ratio > 0 else 0
    n_test = max(1, int(n_total * test_ratio)) if test_ratio > 0 else 0
    if n_val + n_test >= n_total:
        raise ValueError(f"划分比例过大：总数 {n_total}，val {n_val}，test {n_test}")
    splits = {
        "val": [records[index] for index in order[:n_val]],
        "test": [records[index] for index in order[n_val : n_val + n_test]],
        "train": [records[index] for index in order[n_val + n_test :]],
    }
    return splits


def encode_documents(
    documents: list[dict[str, Any]],
    tokenizer,
    *,
    eos_id: int | None,
) -> tuple[list[int], list[int]]:
    """把文档编码并拼接成连续 token 流，返回 (token_ids, segment_ids)。"""
    token_ids: list[int] = []
    segment_ids: list[int] = []
    for document_index, document in enumerate(documents):
        ids = tokenizer.encode(document["text"], add_special_tokens=False).ids
        if not ids:
            continue
        token_ids.extend(ids)
        segment_ids.extend([document_index] * len(ids))
        if eos_id is not None:
            token_ids.append(eos_id)
            segment_ids.append(document_index)
    return token_ids, segment_ids


def pack_sequences(
    token_ids: list[int],
    segment_ids: list[int],
    *,
    block_size: int,
    pad_id: int,
    max_tokens: int | None = None,
) -> dict[str, np.ndarray]:
    """把 token 流切分为定长序列，返回 tokens/loss_mask/segment_ids。"""
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
        segment_ids = segment_ids[:max_tokens]
    num_sequences = (len(token_ids) + block_size - 1) // block_size
    tokens = np.full((num_sequences, block_size), pad_id, dtype=np.int32)
    loss_mask = np.zeros((num_sequences, block_size), dtype=np.uint8)
    segments = np.full((num_sequences, block_size), PAD_SEGMENT, dtype=np.int32)
    for index, (token_id, segment_id) in enumerate(zip(token_ids, segment_ids, strict=True)):
        row, column = divmod(index, block_size)
        tokens[row, column] = token_id
        segments[row, column] = segment_id
        loss_mask[row, column] = 1
    return {"tokens": tokens, "loss_mask": loss_mask, "segment_ids": segments}


def segment_attention_bias(segment_ids: np.ndarray) -> np.ndarray:
    """构造注意力偏置：True 表示允许 attend（同段 + causal）。"""
    segment_ids = np.asarray(segment_ids)
    length = segment_ids.shape[0]
    causal = np.tril(np.ones((length, length), dtype=bool))
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    padding = segment_ids[:, None] == PAD_SEGMENT
    return causal & same_segment & ~padding


def validate_packed(arrays: dict[str, np.ndarray], *, vocab_size: int) -> dict[str, Any]:
    tokens = arrays["tokens"]
    loss_mask = arrays["loss_mask"]
    segments = arrays["segment_ids"]
    assert tokens.shape == loss_mask.shape == segments.shape, "tokens/loss_mask/segment_ids 形状不一致"
    real_tokens = int(loss_mask.sum())
    if real_tokens == 0:
        raise ValueError("打包结果没有有效 token")
    if tokens[loss_mask == 0].size and np.any(tokens[loss_mask == 0] != tokens[loss_mask == 0].flat[0]):
        pass  # 填充区允许全为 pad_id
    if int(tokens[loss_mask == 1].max()) >= vocab_size:
        raise ValueError("token id 超出词表范围")
    padding_positions = loss_mask == 0
    if np.any(segments[padding_positions] != PAD_SEGMENT):
        raise ValueError("填充位置应使用 PAD_SEGMENT=-1")
    for row in range(tokens.shape[0]):
        valid = segments[row][segments[row] != PAD_SEGMENT]
        if valid.size and np.any(np.diff(valid) < 0):
            raise ValueError(f"第 {row} 行 segment id 非单调不减")
    return {"sequences": int(tokens.shape[0]), "block_size": int(tokens.shape[1]), "real_tokens": real_tokens}


def save_packed(
    output_dir: Path,
    name: str,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    meta = {**meta, "file": str(path), "real_tokens": int(arrays["loss_mask"].sum())}
    (output_dir / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_packed(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as data:
        return {key: data[key] for key in data.files}
