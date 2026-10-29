"""指令与偏好数据格式化测试。"""

from __future__ import annotations

import json

from bigmodel_lab.data.instruction import format_alpaca_row, format_instruction_files
from bigmodel_lab.data.preference import (
    format_hh_row,
    format_preference_files,
    make_selfplay_pairs,
    parse_transcript,
)


# ---------- 指令数据 ----------
def test_format_alpaca_row_uses_chinese_fields():
    row = {
        "instruction_zh": "给出三个保持健康的小贴士。",
        "input_zh": "",
        "output_zh": "均衡饮食、规律运动、充足睡眠。",
        "instruction": "Give three tips.",
        "output": "Eat well.",
    }
    record, reason = format_alpaca_row(row, system_prompt="系统提示", max_chars=1000)
    assert reason is None
    assert record["messages"][0] == {"role": "system", "content": "系统提示"}
    assert record["messages"][1]["role"] == "user"
    assert "保持健康" in record["messages"][1]["content"]
    assert record["messages"][2]["content"].startswith("均衡饮食")


def test_format_alpaca_row_rejects_missing_output():
    record, reason = format_alpaca_row({"instruction_zh": "问题", "output_zh": ""}, system_prompt="s", max_chars=100)
    assert record is None
    assert reason == "missing_output"


def test_format_alpaca_row_rejects_too_long():
    row = {"instruction_zh": "问题" * 100, "output_zh": "答案" * 500}
    record, reason = format_alpaca_row(row, system_prompt="s", max_chars=100)
    assert record is None
    assert reason == "too_long"


def test_format_instruction_files_and_stats(tmp_path):
    path = tmp_path / "sft.jsonl"
    rows = [
        {"instruction_zh": f"问题 {index}", "output_zh": f"答案 {index}" * 20}
        for index in range(20)
    ]
    rows.append({"instruction_zh": "", "output_zh": "缺指令"})
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    result = format_instruction_files([path], system_prompt="s", max_chars=4096, seed=1, val_ratio=0.1)
    assert result["stats"]["docs_in"] == 21
    assert result["stats"]["docs_valid"] == 20
    assert result["stats"]["invalid"]["missing_instruction"] == 1
    assert sum(result["stats"]["splits"].values()) == 20


# ---------- 偏好数据 ----------
TRANSCRIPT_CHOSEN = """\n\nHuman: 为什么感恩节要吃火鸡？

\n\nAssistant: 因为传统与历史习俗的延续。"""
TRANSCRIPT_REJECTED = """\n\nHuman: 为什么感恩节要吃火鸡？

\n\nAssistant: 因为火鸡会飞。"""
TRANSCRIPT_MISMATCH = """\n\nHuman: 另一个问题？

\n\nAssistant: 不相关内容。"""


def test_parse_transcript():
    turns = parse_transcript(TRANSCRIPT_CHOSEN)
    assert len(turns) == 2
    assert turns[0]["role"] == "Human"
    assert turns[1]["content"].startswith("因为传统")


def test_format_hh_row_extracts_prompt_and_pair():
    record, reason = format_hh_row({"chosen": TRANSCRIPT_CHOSEN, "rejected": TRANSCRIPT_REJECTED}, max_chars=4096)
    assert reason is None
    assert record["prompt"].endswith("Assistant:")
    assert "火鸡" in record["prompt"]
    assert record["chosen"].startswith("因为传统")
    assert record["rejected"].startswith("因为火鸡会飞")


def test_format_hh_row_rejects_identical_pair():
    record, reason = format_hh_row({"chosen": TRANSCRIPT_CHOSEN, "rejected": TRANSCRIPT_CHOSEN}, max_chars=4096)
    assert record is None
    assert reason == "identical_pair"


def test_format_preference_files(tmp_path):
    path = tmp_path / "pref.jsonl"
    rows = [
        {"chosen": TRANSCRIPT_CHOSEN, "rejected": TRANSCRIPT_REJECTED},
        {"chosen": TRANSCRIPT_CHOSEN, "rejected": TRANSCRIPT_CHOSEN},
        {"chosen": TRANSCRIPT_MISMATCH, "rejected": TRANSCRIPT_REJECTED},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    result = format_preference_files([path], max_chars=4096, seed=1, val_ratio=0.5, max_pairs=10)
    assert result["stats"]["docs_valid"] == 2
    assert result["stats"]["invalid"]["identical_pair"] == 1
    assert result["stats"]["splits"]["val"] == 1
    assert result["stats"]["splits"]["train"] == 1


def test_make_selfplay_pairs_uses_quality_heuristic():
    responses = {
        0: "这是一个完整、流畅且信息量充足的回答，包含足够多的中文字符用于质量评分。",
        1: "重复重复重复",
    }

    def generate(prompt: str, sample_index: int) -> str:
        return responses[sample_index]

    pairs = make_selfplay_pairs(["提问"], generate, max_pairs=10, min_chars=5)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == responses[0]
    assert pairs[0]["rejected"] == responses[1]
    assert pairs[0]["source"] == "self_play"
