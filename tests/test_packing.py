"""数据划分与序列打包测试。"""

from __future__ import annotations

import numpy as np
import pytest

from bigmodel_lab.data.packing import (
    PAD_SEGMENT,
    load_packed,
    pack_sequences,
    save_packed,
    segment_attention_bias,
    split_records,
    validate_packed,
)


def _records(count: int) -> list[dict]:
    return [{"text": f"文档 {index}"} for index in range(count)]


def test_split_records_deterministic():
    records = _records(100)
    first = split_records(records, seed=42, val_ratio=0.1, test_ratio=0.1)
    second = split_records(records, seed=42, val_ratio=0.1, test_ratio=0.1)
    assert first == second
    assert len(first["train"]) == 80
    assert len(first["val"]) == 10
    assert len(first["test"]) == 10
    assert len({row["text"] for row in first["train"]}) == 80


def test_split_records_rejects_bad_ratio():
    with pytest.raises(ValueError, match="划分比例过大"):
        split_records(_records(2), seed=1, val_ratio=0.6, test_ratio=0.6)


def test_pack_sequences_shapes_and_masks():
    token_ids = list(range(1, 11))
    segment_ids = [0] * 5 + [1] * 5
    arrays = pack_sequences(token_ids, segment_ids, block_size=4, pad_id=0)
    assert arrays["tokens"].shape == (3, 4)
    assert int(arrays["loss_mask"].sum()) == 10
    assert arrays["tokens"][2, 2] == 0 and arrays["loss_mask"][2, 2] == 0
    assert arrays["segment_ids"][2, 2] == PAD_SEGMENT
    validation = validate_packed(arrays, vocab_size=100)
    assert validation["real_tokens"] == 10


def test_pack_respects_max_tokens():
    token_ids = list(range(1, 21))
    segment_ids = [0] * 20
    arrays = pack_sequences(token_ids, segment_ids, block_size=8, pad_id=0, max_tokens=8)
    assert int(arrays["loss_mask"].sum()) == 8
    assert arrays["tokens"].shape == (1, 8)


def test_segment_attention_bias_blocks_cross_segment_and_future():
    segments = np.array([0, 0, 1, 1], dtype=np.int32)
    bias = segment_attention_bias(segments)
    # 位置 2 不能看位置 0/1（不同段）
    assert not bias[2, 0] and not bias[2, 1]
    # 位置 3 可以看位置 2（同段且因果）
    assert bias[3, 2]
    # 因果：位置 0 不能看位置 1
    assert not bias[0, 1]
    # 填充位不参与
    padded = np.array([0, PAD_SEGMENT, PAD_SEGMENT], dtype=np.int32)
    padded_bias = segment_attention_bias(padded)
    assert not padded_bias[1, 0] and not padded_bias[1, 1]


def test_validate_packed_rejects_bad_tokens():
    arrays = {
        "tokens": np.array([[10, 0]], dtype=np.int32),
        "loss_mask": np.array([[1, 0]], dtype=np.uint8),
        "segment_ids": np.array([[0, PAD_SEGMENT]], dtype=np.int32),
    }
    with pytest.raises(ValueError, match="词表范围"):
        validate_packed(arrays, vocab_size=10)
    arrays["tokens"] = np.array([[1, 0]], dtype=np.int32)
    arrays["segment_ids"] = np.array([[0, 5]], dtype=np.int32)
    with pytest.raises(ValueError, match="填充位置"):
        validate_packed(arrays, vocab_size=10)


def test_save_and_load_roundtrip(tmp_path):
    arrays = pack_sequences([1, 2, 3], [0, 0, 0], block_size=4, pad_id=0)
    meta = save_packed(tmp_path, "train", arrays, {"corpus": "demo"})
    assert meta["real_tokens"] == 3
    loaded = load_packed(tmp_path / "train.npz")
    assert np.array_equal(loaded["tokens"], arrays["tokens"])
    assert (tmp_path / "train.meta.json").exists()
