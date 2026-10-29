"""数据管线集成测试：原始子集 → 清洗 → 去重 → 统计（sanity 档 ≤2 分钟）。"""

from __future__ import annotations

import json
import time

from bigmodel_lab.budget import Budget
from bigmodel_lab.data.clean import CleanConfig
from bigmodel_lab.data.dedup import DedupConfig
from bigmodel_lab.data.specs import ResourceSpec
from bigmodel_lab.stages.stage2 import process_corpus

SENTENCE = "大模型训练实验课程需要使用真实语料来验证数据清洗与去重流程是否可靠，这条样例文本足够长。"
UNIQUE_SENTENCES = [
    f"第{i}条语料：{SENTENCE}主题编号 {i}，内容刻意保持互不相同以避免被误判为重复。"
    for i in range(120)
]


def _synthetic_corpus() -> list[dict[str, str]]:
    rows = [{"completion": text} for text in UNIQUE_SENTENCES]
    rows.extend({"completion": UNIQUE_SENTENCES[0]} for _ in range(20))  # 精确重复
    rows.extend({"completion": "太短"} for _ in range(10))  # 短文本
    rows.extend({"completion": "This is an English-only sentence with no CJK characters. " * 7} for _ in range(5))  # 语言不符
    return rows


def test_data_pipeline_sanity_under_two_minutes(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw" / "pretrain_demo"
    raw_dir.mkdir(parents=True)
    corpus_path = raw_dir / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in _synthetic_corpus()),
        encoding="utf-8",
    )

    spec = ResourceSpec(
        key="pretrain_demo",
        repo_id="demo/corpus",
        files=["corpus.jsonl"],
        text_field="completion",
        language="zh",
        license="test",
        approx_size_gb=0.001,
    )
    from bigmodel_lab.data import sources

    def fake_download(spec, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "corpus.jsonl").write_text(corpus_path.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(sources, "_download_modelscope", fake_download)
    entry = sources.fetch_resource(spec, data_dir=tmp_path, network=sources.NetworkConfig(), offline=False)

    started = time.perf_counter()
    result = process_corpus(
        key="pretrain_demo",
        files=[raw_dir / "corpus.jsonl"],
        text_field="completion",
        language="zh",
        output_path=tmp_path / "processed" / "pretrain_demo.jsonl",
        clean_config=CleanConfig(min_chars=40),
        dedup_config=DedupConfig(near=False),
        max_docs=500,
        budget=Budget("sanity"),
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 120, f"数据管线耗时过长: {elapsed:.1f}s"
    assert result["stats"]["docs_in"] == 155
    assert result["stats"]["dropped"]["too_short"] == 10
    assert result["stats"]["dropped"]["low_cjk_ratio"] == 5
    assert result["stats"]["docs_kept"] == 140
    assert result["dedup"]["exact_duplicates"] == 20
    assert result["stats"]["docs_written"] == 120
    output_lines = (tmp_path / "processed" / "pretrain_demo.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(output_lines) == 120
    first = json.loads(output_lines[0])
    assert set(first) == {"doc_id", "source", "language", "text"}
    assert entry["status"] == "downloaded"
