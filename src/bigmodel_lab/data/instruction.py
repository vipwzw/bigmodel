"""指令数据格式化：Alpaca 中文 → 标准 chat messages。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .packing import split_records


def _pick_localized(row: dict[str, Any], zh_key: str, en_key: str) -> str:
    value = row.get(zh_key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    value = row.get(en_key)
    return value.strip() if isinstance(value, str) else ""


def format_alpaca_row(
    row: dict[str, Any],
    *,
    system_prompt: str,
    max_chars: int,
) -> tuple[dict[str, Any] | None, str | None]:
    instruction = _pick_localized(row, "instruction_zh", "instruction")
    input_text = _pick_localized(row, "input_zh", "input")
    output = _pick_localized(row, "output_zh", "output")
    if not instruction:
        return None, "missing_instruction"
    if not output:
        return None, "missing_output"
    user_content = instruction if not input_text else f"{instruction}\n\n{input_text}"
    if len(user_content) + len(output) > max_chars:
        return None, "too_long"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    return {
        "messages": messages,
        "source": "sft_zh",
        "language": "zh",
        "chars": len(user_content) + len(output),
    }, None


def format_instruction_files(
    paths: list[Path],
    *,
    system_prompt: str,
    max_chars: int,
    seed: int,
    val_ratio: float,
    max_records: int | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "docs_in": 0,
        "docs_valid": 0,
        "invalid": {},
        "chars": 0,
    }
    stop = False
    for path in paths:
        if stop:
            break
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if max_records is not None and stats["docs_in"] >= max_records:
                    stop = True
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["docs_in"] += 1
                    stats["invalid"]["bad_json"] = stats["invalid"].get("bad_json", 0) + 1
                    continue
                stats["docs_in"] += 1
                record, reason = format_alpaca_row(row, system_prompt=system_prompt, max_chars=max_chars)
                if record is None:
                    stats["invalid"][reason] = stats["invalid"].get(reason, 0) + 1
                    continue
                stats["docs_valid"] += 1
                stats["chars"] += record["chars"]
                records.append(record)
    splits = split_records(records, seed=seed, val_ratio=val_ratio)
    stats["splits"] = {name: len(items) for name, items in splits.items()}
    stats["keep_rate"] = round(stats["docs_valid"] / stats["docs_in"], 4) if stats["docs_in"] else 0.0
    return {"stats": stats, "splits": splits}


def write_instruction_splits(splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, records in splits.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        paths[name] = str(path)
    return paths
