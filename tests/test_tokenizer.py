"""分词器训练与评估测试。"""

from __future__ import annotations

import json

from bigmodel_lab.data.tokenizer import evaluate_tokenizer, load_tokenizer, train_bpe

TEXTS = [
    "大模型训练需要高质量语料，分词器决定 token 的粒度。" * 5,
    "预训练阶段使用 BPE 分词器，微调阶段复用基座模型的分词器。" * 5,
    "The quick brown fox jumps over the lazy dog. " * 5,
]


def test_train_bpe_and_evaluate(tmp_path):
    meta = train_bpe(TEXTS, vocab_size=300, output_dir=tmp_path / "tok")
    assert (tmp_path / "tok" / "tokenizer.json").exists()
    assert meta["vocab_size"] <= 300
    assert meta["special_token_ids"]["<pad>"] is not None
    tokenizer = load_tokenizer(tmp_path / "tok" / "tokenizer.json")
    ids = tokenizer.encode(TEXTS[0], add_special_tokens=False).ids
    assert tokenizer.decode(ids) == TEXTS[0]

    report = evaluate_tokenizer(tmp_path / "tok" / "tokenizer.json", TEXTS, max_docs=10)
    assert report["docs_evaluated"] == 3
    assert report["chars_per_token"] > 0
    assert report["unk_rate"] >= 0
    assert report["samples"][0]["decode_roundtrip_ok"] is True
    json.dumps(report)  # 可序列化


def test_vocab_size_respected(tmp_path):
    meta = train_bpe(TEXTS, vocab_size=280, output_dir=tmp_path / "tok")
    assert meta["vocab_size"] <= 280
    assert meta["requested_vocab_size"] == 280
