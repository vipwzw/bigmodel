"""Stage 3 管线集成测试：清洗语料 → 分词器 → 划分 → 打包 → 掩码校验。"""

from __future__ import annotations

import json
import time

import numpy as np

from bigmodel_lab.budget import Budget
from bigmodel_lab.data.packing import load_packed, segment_attention_bias
from bigmodel_lab.stages.stage3 import _pack_corpora, _train_tokenizers

BASE = "这是用于测试分词和打包的中文文档，内容需要足够长以覆盖多个 token，并验证划分与掩码是否正确。"


def _write_corpus(path, count: int, base: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            record = {
                "doc_id": f"d{index}",
                "source": path.stem,
                "language": "zh",
                "text": f"{base} 文档编号 {index}。",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_stage3_pipeline_fast(tmp_path):
    stage2 = tmp_path / "stage2"
    stage2.mkdir()
    _write_corpus(stage2 / "pretrain_zh.jsonl", 40, BASE)
    _write_corpus(stage2 / "pretrain_en.jsonl", 20, "This is a short English story for packing tests. " * 6)

    config = {
        "data": {
            "tokenizer": {
                "vocab_sizes": [512],
                "train_docs": {"sanity": 40},
                "eval_docs": 10,
                "min_frequency": 1,
            },
            "packing": {
                "block_size": 64,
                "sanity_max_tokens": {"pretrain_zh": 2000, "pretrain_en": 800},
            },
            "split": {"val_ratio": 0.1, "test_ratio": 0.1},
        }
    }
    started = time.perf_counter()
    reports, tokenizer_paths = _train_tokenizers(
        budget=Budget("sanity"), config=config, stage2_dir=stage2, output_dir=tmp_path / "tokenizers"
    )
    assert "bpe_512" in reports
    assert reports["bpe_512"]["chars_per_token"] > 0

    pack_report, split_manifest = _pack_corpora(
        budget=Budget("sanity"),
        config=config,
        stage2_dir=stage2,
        output_dir=tmp_path / "packed",
        tokenizer_paths=tokenizer_paths,
        seed=42,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 60, f"Stage 3 集成测试耗时过长: {elapsed:.1f}s"

    assert set(pack_report) == {"pretrain_zh_bpe512", "pretrain_en_bpe512"}
    zh = pack_report["pretrain_zh_bpe512"]["splits"]
    assert set(zh) == {"train", "val", "test"}
    assert zh["train"]["real_tokens"] <= 2000
    assert split_manifest["seed"] == 42
    assert split_manifest["corpora"]["pretrain_zh"]["train"] == 32  # 40 - 4 × 2

    arrays = load_packed(zh["train"]["file"])
    assert arrays["tokens"].shape[1] == 64
    # 找到包含多个文档段的序列，验证跨段注意力被屏蔽
    found_cross_segment = False
    for row in range(arrays["segment_ids"].shape[0]):
        segments = arrays["segment_ids"][row]
        valid = segments[segments != -1]
        if valid.size and len(np.unique(valid)) >= 2:
            bias = segment_attention_bias(segments)
            second_segment_first = int(np.argmax(valid != valid[0]))
            assert not bias[second_segment_first, 0]
            found_cross_segment = True
            break
    assert found_cross_segment, "打包数据中应出现跨文档序列，以便验证段掩码"
