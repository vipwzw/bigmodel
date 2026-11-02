"""文本清洗与统计。"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_WHITESPACE_RE = re.compile(r"[ \t\u3000]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 非字母/数字/空白/CJK/常用标点的字符视为符号
_SYMBOL_RE = re.compile(r"[^\w\s\u4e00-\u9fff，。！？；：、…—·（）【】《》「」『』“”‘’\-.,!?;:'\"()\[\]%+]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ALPHA_RE = re.compile(r"[A-Za-z]")


@dataclass
class CleanConfig:
    min_chars: int = 80
    max_chars: int = 200_000
    min_unique_char_ratio: float = 0.02
    max_symbol_ratio: float = 0.35
    min_cjk_ratio: float = 0.10
    min_alpha_ratio: float = 0.50

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> CleanConfig:
        payload = payload or {}
        return cls(
            min_chars=int(payload.get("min_chars", 80)),
            max_chars=int(payload.get("max_chars", 200_000)),
            min_unique_char_ratio=float(payload.get("min_unique_char_ratio", 0.02)),
            max_symbol_ratio=float(payload.get("max_symbol_ratio", 0.35)),
            min_cjk_ratio=float(payload.get("min_cjk_ratio", 0.10)),
            min_alpha_ratio=float(payload.get("min_alpha_ratio", 0.50)),
        )


@dataclass
class CleanStats:
    dataset: str = ""
    docs_in: int = 0
    docs_kept: int = 0
    chars_in: int = 0
    chars_kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    kept_lengths: list[int] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)

    def drop(self, reason: str) -> None:
        self.dropped[reason] += 1

    def summary(self) -> dict[str, Any]:
        lengths = sorted(self.kept_lengths)
        percentile = lambda p: lengths[min(len(lengths) - 1, int(len(lengths) * p))] if lengths else 0  # noqa: E731
        return {
            "dataset": self.dataset,
            "docs_in": self.docs_in,
            "docs_kept": self.docs_kept,
            "keep_rate": round(self.docs_kept / self.docs_in, 4) if self.docs_in else 0.0,
            "chars_in": self.chars_in,
            "chars_kept": self.chars_kept,
            "dropped": dict(self.dropped),
            "length": {
                "min": lengths[0] if lengths else 0,
                "p50": percentile(0.50),
                "p90": percentile(0.90),
                "p99": percentile(0.99),
                "max": lengths[-1] if lengths else 0,
            },
            "samples": self.samples,
        }


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def ratio(pattern: re.Pattern[str], text: str) -> float:
    if not text:
        return 0.0
    return len(pattern.findall(text)) / len(text)


def filter_reason(text: str, config: CleanConfig, language: str) -> str | None:
    """返回丢弃原因；None 表示通过。"""
    length = len(text)
    if length < config.min_chars:
        return "too_short"
    if length > config.max_chars:
        return "too_long"
    if len(set(text)) / length < config.min_unique_char_ratio:
        return "low_unique_ratio"
    if ratio(_SYMBOL_RE, text) > config.max_symbol_ratio:
        return "too_many_symbols"
    if language == "zh" and ratio(_CJK_RE, text) < config.min_cjk_ratio:
        return "low_cjk_ratio"
    if language == "en" and ratio(_ALPHA_RE, text) < config.min_alpha_ratio:
        return "low_alpha_ratio"
    return None


class CorpusCleaner:
    """流式清洗器：逐条产出 (Document, cleaned_text)，结束后读取统计。"""

    def __init__(
        self,
        *,
        dataset: str,
        language: str,
        config: CleanConfig,
        max_docs: int | None = None,
    ):
        self.stats = CleanStats(dataset=dataset)
        self.language = language
        self.config = config
        self.max_docs = max_docs

    def run(self, documents):
        stats = self.stats
        for document in documents:
            if self.max_docs is not None and stats.docs_in >= self.max_docs:
                break
            stats.docs_in += 1
            stats.chars_in += len(document.text)
            cleaned = normalize_text(document.text)
            if not cleaned:
                stats.drop("empty_after_normalize")
                continue
            reason = filter_reason(cleaned, self.config, self.language)
            if reason:
                stats.drop(reason)
                continue
            stats.docs_kept += 1
            stats.chars_kept += len(cleaned)
            stats.kept_lengths.append(len(cleaned))
            if len(stats.samples) < 3:
                stats.samples.append(cleaned[:300])
            yield document, cleaned

    def summary(self) -> dict[str, Any]:
        return self.stats.summary()


def clean_documents(
    documents,
    *,
    dataset: str,
    language: str,
    config: CleanConfig,
    max_docs: int | None = None,
):
    """兼容函数式调用：返回 (生成器, cleaner)，统计需在消费完生成器后读取。"""
    cleaner = CorpusCleaner(dataset=dataset, language=language, config=config, max_docs=max_docs)
    return cleaner.run(documents), cleaner
