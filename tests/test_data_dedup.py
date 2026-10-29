"""去重测试。"""

from __future__ import annotations

import numpy as np

from bigmodel_lab.data.dedup import DedupConfig, Deduper, minhash_signature

BASE = "这段文本用于测试去重逻辑，包含足够多的中文字符以便 MinHash 产生有区分度的签名。" * 6
MODIFIED = BASE + "（在末尾追加少量内容，模拟近似重复文档。）"
DISTINCT = "完全不同主题的一段文字：今天天气晴朗，适合去公园散步，顺便观察一下湖边的白鹭。" * 6


def test_exact_duplicate_detected():
    deduper = Deduper(DedupConfig(near=False))
    assert deduper.add(BASE) is not None
    assert deduper.add(BASE) is None
    summary = deduper.summary()
    assert summary["exact_duplicates"] == 1
    assert summary["docs_kept"] == 1


def test_near_duplicate_detected():
    deduper = Deduper(DedupConfig(exact=False, near=True, near_threshold=0.6))
    assert deduper.add(BASE) is not None
    assert deduper.add(MODIFIED) is None
    assert deduper.summary()["near_duplicates"] == 1


def test_distinct_documents_kept():
    deduper = Deduper(DedupConfig(near=True, near_threshold=0.8))
    assert deduper.add(BASE) is not None
    assert deduper.add(DISTINCT) is not None
    assert deduper.summary()["docs_kept"] == 2


def test_minhash_signature_deterministic():
    first = minhash_signature(BASE, num_perm=32)
    second = minhash_signature(BASE, num_perm=32)
    assert np.array_equal(first, second)
    assert len(first) == 32
    assert not np.array_equal(first, minhash_signature(DISTINCT, num_perm=32))
