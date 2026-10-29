"""对话模板与 SFT 掩码测试。"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from bigmodel_lab.data.chat import (
    IGNORE_INDEX,
    SFTDataset,
    build_sft_example,
    collate_sft,
    mask_sanity_check,
)

CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def _tokenizer(tmp_path: Path) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=512, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"], show_progress=False)
    corpus = ["用户: 你好\nassistant: 你好，很高兴认识你。\n", "系统: 你是一个助手。\n用户: 解释 LoRA\nassistant: LoRA 是低秩适配。\n"] * 10
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(path), eos_token="<eos>", pad_token="<pad>", unk_token="<unk>", bos_token="<bos>"
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast


MESSAGES = [
    {"role": "system", "content": "你是一个乐于助人的中文助手。"},
    {"role": "user", "content": "用一句话解释什么是 LoRA。"},
    {"role": "assistant", "content": "LoRA 是在冻结的基座模型上训练低秩适配矩阵的微调方法。"},
]


def test_build_sft_example_masks_prompt_tokens(tmp_path):
    tokenizer = _tokenizer(tmp_path)
    example, reason = build_sft_example(tokenizer, MESSAGES, max_seq_len=256)
    assert reason is None
    assert example["prompt_tokens"] > 0
    assert example["response_tokens"] > 0
    labels = example["labels"]
    prompt_tokens = example["prompt_tokens"]
    assert all(label == IGNORE_INDEX for label in labels[:prompt_tokens])
    assert all(label != IGNORE_INDEX for label in labels[prompt_tokens:])
    assert example["labels"][-1] == tokenizer.eos_token_id


def test_mask_sanity_check_reports_ok(tmp_path):
    tokenizer = _tokenizer(tmp_path)
    report = mask_sanity_check(tokenizer, MESSAGES)
    assert report["ok"] is True
    assert report["prompt_masked"] is True
    assert report["prompt_tokens"] > 0
    assert report["decoded_head"]
    assert "LoRA" in report["decoded_assistant"]


def test_long_response_rejected(tmp_path):
    tokenizer = _tokenizer(tmp_path)
    long_messages = [MESSAGES[0], MESSAGES[1], {"role": "assistant", "content": "很长的回答。" * 200}]
    example, reason = build_sft_example(tokenizer, long_messages, max_seq_len=32)
    assert example is None
    assert reason == "response_too_long"


def test_dataset_and_collate_padding(tmp_path):
    tokenizer = _tokenizer(tmp_path)
    records = [{"messages": MESSAGES}, {"messages": [MESSAGES[0], {"role": "user", "content": "短问题"}, {"role": "assistant", "content": "短答。"}]}]
    dataset = SFTDataset(records, tokenizer, max_seq_len=256)
    assert len(dataset) == 2
    assert dataset.stats["valid"] == 2
    batch = collate_sft([dataset[0], dataset[1]], pad_token_id=tokenizer.pad_token_id)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == batch["labels"].shape[1]
    # padding 位置的 labels 必须是 IGNORE_INDEX
    for row in range(2):
        labels = batch["labels"][row].tolist()
        masks = batch["attention_mask"][row].tolist()
        assert all(label == IGNORE_INDEX for label, mask in zip(labels, masks, strict=True) if mask == 0)
    # 至少有一个非 mask 的 assistant token 参与损失
    assert int((batch["labels"] != IGNORE_INDEX).sum()) > 0
