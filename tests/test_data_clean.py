"""数据清洗测试。"""

from __future__ import annotations

from bigmodel_lab.data.clean import CleanConfig, CorpusCleaner, filter_reason, normalize_text
from bigmodel_lab.data.io import Document


def test_normalize_text_collapses_whitespace_and_control_chars():
    text = "  hello\r\n\r\n\r\nworld \t\t 测试\x00  "
    assert normalize_text(text) == "hello\n\nworld 测试"


def test_filter_too_short():
    assert filter_reason("太短", CleanConfig(min_chars=10), "zh") == "too_short"


def test_filter_low_unique_ratio():
    repeated = "a" * 500
    assert filter_reason(repeated, CleanConfig(), "en") == "low_unique_ratio"


def test_filter_low_cjk_ratio_for_chinese():
    english = "This is an English sentence only. " * 10
    assert filter_reason(english, CleanConfig(), "zh") == "low_cjk_ratio"


def test_filter_low_alpha_ratio_for_english():
    chinese = "这是一段完全由中文组成的文本，用来测试英文语料的过滤规则是否生效。" * 5
    assert filter_reason(chinese, CleanConfig(), "en") == "low_alpha_ratio"


def test_cleaner_streams_and_accumulates_stats():
    valid = "这是一段足够长的中文维基百科文本，用于测试清洗流程是否保留正常文档。" * 5
    documents = [
        Document(text=valid),
        Document(text="太短"),
        Document(text="This is English only. " * 10),
    ]
    cleaner = CorpusCleaner(dataset="demo", language="zh", config=CleanConfig())
    kept = list(cleaner.run(documents))
    assert len(kept) == 1
    summary = cleaner.summary()
    assert summary["docs_in"] == 3
    assert summary["docs_kept"] == 1
    assert summary["dropped"]["too_short"] == 1
    assert summary["dropped"]["low_cjk_ratio"] == 1
    assert summary["keep_rate"] == 0.3333
    assert summary["length"]["p50"] == len(valid)
    assert summary["samples"]


def test_cleaner_respects_max_docs():
    documents = (Document(text="有效的长中文文本。" * 20) for _ in range(100))
    cleaner = CorpusCleaner(dataset="demo", language="zh", config=CleanConfig(), max_docs=5)
    assert len(list(cleaner.run(documents))) == 5
    assert cleaner.summary()["docs_in"] == 5
