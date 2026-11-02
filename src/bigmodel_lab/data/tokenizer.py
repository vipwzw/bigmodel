"""BPE 分词器训练与评估。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int,
    output_dir: Path,
    min_frequency: int = 2,
) -> dict[str, Any]:
    """训练 byte-level BPE 分词器并保存 tokenizer.json。"""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_token_ids": {name: tokenizer.token_to_id(name) for name in SPECIAL_TOKENS},
        "tokenizer_path": str(output_dir / "tokenizer.json"),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer  # noqa: PLC0415

    return Tokenizer.from_file(str(path))


def evaluate_tokenizer(
    tokenizer_path: str | Path,
    texts: Iterable[str],
    *,
    max_docs: int = 200,
) -> dict[str, Any]:
    """评估压缩率、未知 token、编码/解码样例与每文档 token 数。"""
    tokenizer = load_tokenizer(tokenizer_path)
    vocab = tokenizer.get_vocab()
    unk_id = vocab.get("<unk>")
    total_chars = 0
    total_tokens = 0
    unk_tokens = 0
    tokens_per_doc: list[int] = []
    samples: list[dict[str, Any]] = []
    docs = 0
    for text in texts:
        if docs >= max_docs:
            break
        encoding = tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids
        docs += 1
        total_chars += len(text)
        total_tokens += len(ids)
        if unk_id is not None:
            unk_tokens += sum(1 for token_id in ids if token_id == unk_id)
        tokens_per_doc.append(len(ids))
        if len(samples) < 3:
            decoded = tokenizer.decode(ids)
            samples.append(
                {
                    "text_head": text[:120],
                    "num_tokens": len(ids),
                    "token_pieces_head": encoding.tokens[:24],
                    "decode_roundtrip_ok": decoded == text,
                }
            )
    tokens_per_doc.sort()
    percentile = lambda p: tokens_per_doc[min(len(tokens_per_doc) - 1, int(len(tokens_per_doc) * p))] if tokens_per_doc else 0  # noqa: E731
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "docs_evaluated": docs,
        "chars": total_chars,
        "tokens": total_tokens,
        "chars_per_token": round(total_chars / total_tokens, 3) if total_tokens else 0.0,
        "unk_rate": round(unk_tokens / total_tokens, 6) if total_tokens else 0.0,
        "tokens_per_doc": {"p50": percentile(0.5), "p90": percentile(0.9), "p99": percentile(0.99)},
        "samples": samples,
    }


def encode_with_specials(tokenizer, text: str, *, add_eos: bool = False) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if add_eos:
        eos_id = tokenizer.token_to_id("<eos>")
        if eos_id is not None:
            ids = [*ids, eos_id]
    return ids


def load_hf_tokenizer(path: str | Path):
    """把 tokenizer.json 包装成 transformers 风格 tokenizer（供评测/生成代码使用）。"""
    from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
    )
