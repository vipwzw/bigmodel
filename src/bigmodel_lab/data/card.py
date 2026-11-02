"""数据卡生成。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def build_dataset_card(
    *,
    entries: list[dict[str, Any]],
    profile: str,
    data_root: str,
) -> str:
    lines = [
        "# 数据集卡（Dataset Card）",
        "",
        f"- 生成时间：{_now_iso()}",
        f"- 运行档位：{profile}",
        f"- 处理后数据目录：`{data_root}`",
        "",
        "## 语料清单",
        "",
        "| 数据集 | 语言 | 许可 | 来源 | 文档数（保留/输入） | 保留字符数 | 去重情况 |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        stats = entry["stats"]
        dedup = entry["dedup"]
        lines.append(
            f"| {entry['key']} | {entry['language']} | {entry['license']} | "
            f"{entry['repo_id']} | {stats['docs_kept']}/{stats['docs_in']} | "
            f"{stats['chars_kept']:,} | 精确 {dedup['exact_duplicates']} / 近似 {dedup['near_duplicates']} |"
        )
    lines.append("")
    lines.append("## 质量统计")
    lines.append("")
    for entry in entries:
        stats = entry["stats"]
        dedup = entry["dedup"]
        lines.append(f"### {entry['key']}")
        lines.append("")
        lines.append(f"- 来源文件：`{entry['files']}`")
        lines.append(f"- 保留率：{stats['keep_rate']:.1%}")
        lines.append(f"- 过滤丢弃原因：{stats['dropped'] or '无'}")
        lines.append(
            f"- 文档长度：min {stats['length']['min']} / p50 {stats['length']['p50']} / "
            f"p90 {stats['length']['p90']} / p99 {stats['length']['p99']} / max {stats['length']['max']} 字符"
        )
        lines.append(f"- 去重：输入 {dedup['docs_in']}，精确重复 {dedup['exact_duplicates']}，近似重复 {dedup['near_duplicates']}")
        lines.append("")
        lines.append("抽样样例：")
        lines.append("")
        for sample in stats["samples"][:2]:
            snippet = sample.replace("\n", " ")[:200]
            lines.append(f"> {snippet}…")
            lines.append("")
    lines.append("## 已知偏差与注意事项")
    lines.append("")
    lines.append("- 中文维基语料偏百科/书面语，训练出的模型在口语、对话风格上会明显不足（后续 SFT 阶段用指令数据弥补）。")
    lines.append("- TinyStories 为合成儿童故事，词汇与句式非常受限，仅用于快速兜底与对照，不代表真实英文分布。")
    lines.append("- 语料来源为公开数据集，许可证见上表；本项目仅本地实验使用。")
    lines.append("- 去重为启发式（MinHash 近似），阈值见 `configs/default.yaml` 的 `data.dedup`。")
    lines.append("")
    return "\n".join(lines)
