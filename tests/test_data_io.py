"""数据读取与文件定位测试。"""

from __future__ import annotations

import json

import pytest

from bigmodel_lab.data.io import iter_documents, iter_jsonl, iter_parquet, resolve_files


def test_iter_jsonl_uses_text_field_and_fallback(tmp_path):
    path = tmp_path / "docs.jsonl"
    rows = [
        {"completion": "第一段文本"},
        {"text": "第二段文本"},
        {"content": "第三段文本"},
        {"other": "没有文本字段"},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    docs = list(iter_jsonl(path, "completion"))
    assert [doc.text for doc in docs] == ["第一段文本", "第二段文本", "第三段文本"]


def test_iter_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "docs.parquet"
    pq.write_table(pa.table({"text": ["a", "b", "c"]}), path)
    docs = list(iter_parquet(path, "text"))
    assert [doc.text for doc in docs] == ["a", "b", "c"]


def test_iter_parquet_missing_column_raises(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "docs.parquet"
    pq.write_table(pa.table({"foo": ["a"]}), path)
    with pytest.raises(ValueError, match="缺少文本列"):
        list(iter_documents([path], "text"))


def test_resolve_files_supports_glob_and_nested(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train-00000.parquet").write_text("x")
    (tmp_path / "data" / "validation.parquet").write_text("y")
    (tmp_path / "README.md").write_text("z")

    found = resolve_files(tmp_path, ["data/train-*.parquet"])
    assert [path.name for path in found] == ["train-00000.parquet"]

    found = resolve_files(tmp_path, ["data/train-*.parquet", "data/validation.parquet"])
    assert len(found) == 2

    with pytest.raises(FileNotFoundError):
        resolve_files(tmp_path, ["missing.jsonl"])
