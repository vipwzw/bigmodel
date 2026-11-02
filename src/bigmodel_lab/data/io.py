"""本地数据读取：jsonl / parquet 统一为文档迭代器。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FILE_SUFFIXES = (".jsonl", ".json", ".parquet")


@dataclass
class Document:
    text: str
    source: str = ""
    row_index: int = 0
    meta: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}


def iter_jsonl(path: Path, text_field: str) -> Iterator[Document]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get(text_field) if isinstance(record, dict) else None
            if text is None and isinstance(record, dict):
                # 常见回退字段
                for candidate in ("text", "content", "completion", "prompt"):
                    if isinstance(record.get(candidate), str):
                        text = record[candidate]
                        break
            if isinstance(text, str) and text.strip():
                yield Document(text=text, row_index=index)


def iter_parquet(path: Path, text_field: str) -> Iterator[Document]:
    import pyarrow.parquet as pq  # noqa: PLC0415 - 延迟导入，核心流程不依赖

    table = pq.read_table(path, columns=None)
    column = text_field if text_field in table.column_names else None
    if column is None:
        for candidate in ("text", "content", "completion"):
            if candidate in table.column_names:
                column = candidate
                break
    if column is None:
        raise ValueError(f"parquet 文件缺少文本列 {text_field}: {path}，实际列: {table.column_names}")
    for index, value in enumerate(table.column(column).to_pylist()):
        if isinstance(value, str) and value.strip():
            yield Document(text=value, row_index=index)


def iter_documents(paths: list[Path], text_field: str | None) -> Iterator[Document]:
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            yield from iter_parquet(path, text_field or "text")
        elif suffix in (".jsonl", ".json"):
            yield from iter_jsonl(path, text_field or "text")
        else:
            raise ValueError(f"不支持的文本文件类型: {path}")


def resolve_files(directory: Path, patterns: list[str]) -> list[Path]:
    """在下载目录内按文件名/glob 定位已下载文件。"""
    from fnmatch import fnmatch  # noqa: PLC0415

    found: list[Path] = []
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            found.extend(sorted(path for path in directory.rglob("*") if path.is_file() and fnmatch(str(path.relative_to(directory)), pattern)))
        else:
            candidate = directory / pattern
            if candidate.exists():
                found.append(candidate)
    unique: list[Path] = []
    for path in found:
        if path not in unique:
            unique.append(path)
    if not unique:
        raise FileNotFoundError(f"{directory} 下找不到资源文件: {patterns}")
    return unique
