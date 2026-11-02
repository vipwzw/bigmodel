"""偏好数据格式化：HH-RLHF 对话转录 → (prompt, chosen, rejected)；self-play 构造工具。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .packing import split_records

_TURN_RE = re.compile(r"\n\n(Human|Assistant): ")
_RESPONSE_BAD_ENDINGS = ("...", "，", ",", "：", ":")


def parse_transcript(text: str) -> list[dict[str, str]]:
    """把 "Human: ... / Assistant: ..." 转录解析为轮次列表。"""
    parts = _TURN_RE.split("\n\n" + text.strip())
    turns: list[dict[str, str]] = []
    for index in range(1, len(parts), 2):
        role = parts[index]
        content = parts[index + 1].strip()
        if content:
            turns.append({"role": role, "content": content})
    return turns


def _extract_prompt_and_responses(
    chosen_turns: list[dict[str, str]],
    rejected_turns: list[dict[str, str]],
    *,
    max_chars: int,
) -> tuple[dict[str, Any] | None, str | None]:
    chosen_human_index = max(
        (index for index, turn in enumerate(chosen_turns) if turn["role"] == "Human"),
        default=-1,
    )
    if chosen_human_index < 0:
        return None, "no_human_turn"
    chosen_response = next(
        (turn["content"] for turn in chosen_turns[chosen_human_index + 1 :] if turn["role"] == "Assistant"),
        None,
    )
    rejected_human_index = max(
        (index for index, turn in enumerate(rejected_turns) if turn["role"] == "Human"),
        default=-1,
    )
    if rejected_human_index < 0:
        return None, "no_human_turn"
    rejected_response = next(
        (turn["content"] for turn in rejected_turns[rejected_human_index + 1 :] if turn["role"] == "Assistant"),
        None,
    )
    if not chosen_response or not rejected_response:
        return None, "missing_response"
    if chosen_response == rejected_response:
        return None, "identical_pair"
    prompt = "\n\n".join(f"{turn['role']}: {turn['content']}" for turn in chosen_turns[: chosen_human_index + 1])
    prompt += "\n\nAssistant:"
    if len(prompt) + len(chosen_response) > max_chars or len(prompt) + len(rejected_response) > max_chars:
        return None, "too_long"
    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "source": "pref_en",
        "language": "en",
    }, None


def format_hh_row(row: dict[str, Any], *, max_chars: int) -> tuple[dict[str, Any] | None, str | None]:
    chosen_text = row.get("chosen")
    rejected_text = row.get("rejected")
    if not isinstance(chosen_text, str) or not isinstance(rejected_text, str):
        return None, "missing_fields"
    chosen_turns = parse_transcript(chosen_text)
    rejected_turns = parse_transcript(rejected_text)
    if not chosen_turns or not rejected_turns:
        return None, "empty_transcript"
    return _extract_prompt_and_responses(chosen_turns, rejected_turns, max_chars=max_chars)


def format_preference_files(
    paths: list[Path],
    *,
    max_chars: int,
    seed: int,
    val_ratio: float,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"docs_in": 0, "docs_valid": 0, "invalid": {}}
    stop = False
    for path in paths:
        if stop:
            break
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if max_pairs is not None and stats["docs_valid"] >= max_pairs:
                    stop = True
                    break
                line = line.strip()
                if not line:
                    continue
                stats["docs_in"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid"]["bad_json"] = stats["invalid"].get("bad_json", 0) + 1
                    continue
                record, reason = format_hh_row(row, max_chars=max_chars)
                if record is None:
                    stats["invalid"][reason] = stats["invalid"].get(reason, 0) + 1
                    continue
                stats["docs_valid"] += 1
                records.append(record)
    splits = split_records(records, seed=seed, val_ratio=val_ratio)
    stats["splits"] = {name: len(items) for name, items in splits.items()}
    stats["keep_rate"] = round(stats["docs_valid"] / stats["docs_in"], 4) if stats["docs_in"] else 0.0
    return {"stats": stats, "splits": splits}


def write_preference_splits(splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, records in splits.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        paths[name] = str(path)
    return paths


def _quality_score(response: str) -> float:
    """无裁判模型时的启发式质量分：长度合理、无重复、结尾完整。"""
    if not response:
        return -1.0
    length = len(response)
    length_score = min(length, 300) / 300
    repetition = len(set(response)) / max(length, 1)
    ending_penalty = 1.0 if response.rstrip().endswith(_RESPONSE_BAD_ENDINGS) else 0.0
    return length_score + repetition - 0.5 * ending_penalty


def make_selfplay_pairs(
    prompts: list[str],
    generate_fn: Callable[[str, int], str],
    *,
    max_pairs: int = 500,
    min_chars: int = 20,
) -> list[dict[str, Any]]:
    """用同一模型两次采样构造偏好对（Stage 6 无偏好数据时的兜底方案）。"""
    pairs: list[dict[str, Any]] = []
    for prompt in prompts:
        if len(pairs) >= max_pairs:
            break
        first = generate_fn(prompt, 0)
        second = generate_fn(prompt, 1)
        if not first or not second or first == second:
            continue
        if min(len(first), len(second)) < min_chars:
            continue
        score_first, score_second = _quality_score(first), _quality_score(second)
        if abs(score_first - score_second) < 1e-6:
            continue
        chosen, rejected = (first, second) if score_first > score_second else (second, first)
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": "self_play",
                "language": "zh",
                "score_gap": round(abs(score_first - score_second), 4),
            }
        )
    return pairs
