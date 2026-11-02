"""对话模板渲染、SFT 样本构造与掩码自检。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100


def load_sft_records(paths: list[Path], *, max_records: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if max_records is not None and len(records) >= max_records:
                    return records
    return records


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """渲染 system/user（不含最后一条 assistant 回复），并加上生成提示前缀。"""
    conversation = [message for message in messages if message.get("role") != "assistant"]
    # 保留除最后一条 assistant 之外的所有轮次
    return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)


def build_sft_example(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_seq_len: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """构造 input_ids/labels：只有 assistant 回复计入损失。"""
    if not messages or messages[-1].get("role") != "assistant":
        return None, "last_turn_not_assistant"
    response = str(messages[-1].get("content", ""))
    if not response.strip():
        return None, "empty_response"
    prompt_text = render_prompt(tokenizer, messages)
    eos_token = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(response + eos_token, add_special_tokens=False).input_ids
    if len(response_ids) > max_seq_len:
        return None, "response_too_long"
    prompt_ids = prompt_ids[: max_seq_len - len(response_ids)]
    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
    }, None


def mask_sanity_check(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_seq_len: int = 1024,
) -> dict[str, Any]:
    """打印/返回一条样本的渲染文本与计损位置，供自检与测试使用。"""
    example, reason = build_sft_example(tokenizer, messages, max_seq_len=max_seq_len)
    if example is None:
        return {"ok": False, "reason": reason}
    input_ids = example["input_ids"]
    labels = example["labels"]
    prompt_tokens = example["prompt_tokens"]
    decoded = tokenizer.decode(input_ids)
    counted = [index for index, label in enumerate(labels) if label != IGNORE_INDEX]
    prompt_masked = all(label == IGNORE_INDEX for label in labels[:prompt_tokens])
    counted_tokens = [token_id for token_id, label in zip(input_ids, labels, strict=True) if label != IGNORE_INDEX]
    decoded_assistant = tokenizer.decode(counted_tokens)
    return {
        "ok": prompt_masked and len(counted) == len(example["input_ids"]) - prompt_tokens and bool(counted),
        "prompt_tokens": prompt_tokens,
        "response_tokens": example["response_tokens"],
        "total_tokens": len(input_ids),
        "prompt_masked": prompt_masked,
        "counted_positions": counted[:5] + counted[-3:] if len(counted) > 8 else counted,
        "decoded_head": decoded[:300],
        "decoded_assistant": decoded_assistant[:300],
        "assistant_ends_with_eos": decoded_assistant.endswith(tokenizer.eos_token or "\0"),
    }


class SFTDataset:
    """torch Dataset：把 messages 记录转换成 input_ids/labels。"""

    def __init__(self, records: list[dict[str, Any]], tokenizer, *, max_seq_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stats = {"records": len(records), "valid": 0, "dropped": {}}
        self.examples: list[dict[str, Any]] = []
        for record in records:
            example, reason = build_sft_example(tokenizer, record["messages"], max_seq_len=max_seq_len)
            if example is None:
                self.stats["dropped"][reason] = self.stats["dropped"].get(reason, 0) + 1
                continue
            self.examples.append(example)
            self.stats["valid"] += 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_sft(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    max_len = max(len(example["input_ids"]) for example in batch)
    input_ids, labels, attention_masks = [], [], []
    for example in batch:
        pad = max_len - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * pad)
        labels.append(example["labels"] + [IGNORE_INDEX] * pad)
        attention_masks.append(example["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }


def extract_user_text(prompt: str) -> str:
    """从 HH 格式 prompt（"Human: ...\n\nAssistant:"）中提取最后一条用户消息。"""
    normalized = prompt.strip()
    if normalized.endswith("Assistant:"):
        normalized = normalized[: -len("Assistant:")].rstrip()
    marker = normalized.rfind("Human: ")
    if marker >= 0:
        normalized = normalized[marker + len("Human: ") :]
    return normalized.strip()


def build_dpo_example(
    tokenizer,
    record: dict[str, Any],
    *,
    max_seq_len: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """构造 DPO 样本：prompt/chosen/rejected 各自的 token 序列与响应掩码。"""
    chosen_text = str(record.get("chosen", "")).strip()
    rejected_text = str(record.get("rejected", "")).strip()
    if not chosen_text or not rejected_text:
        return None, "missing_response"
    user_text = extract_user_text(str(record.get("prompt", "")))
    if not user_text:
        return None, "empty_prompt"
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    eos_token = tokenizer.eos_token or ""
    budget = max_seq_len - len(prompt_ids)
    if budget <= 8:
        return None, "prompt_too_long"
    chosen_ids = tokenizer(chosen_text + eos_token, add_special_tokens=False).input_ids[:budget]
    rejected_ids = tokenizer(rejected_text + eos_token, add_special_tokens=False).input_ids[:budget]
    if not chosen_ids or not rejected_ids:
        return None, "empty_response"
    return {
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "user_text": user_text,
    }, None


def collate_dpo(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    def build(responses: list[list[int]], prompts: list[list[int]]):
        lengths = [len(prompt) + len(response) for prompt, response in zip(prompts, responses, strict=True)]
        max_len = max(lengths)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        response_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for row, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
            ids = prompt + response
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
            response_mask[row, len(prompt) : len(ids)] = True
        return {"input_ids": input_ids, "attention_mask": attention_mask, "response_mask": response_mask}

    prompts = [example["prompt_ids"] for example in batch]
    return {
        "chosen": build([example["chosen_ids"] for example in batch], prompts),
        "rejected": build([example["rejected_ids"] for example in batch], prompts),
    }
