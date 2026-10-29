"""评测工具测试。"""

from __future__ import annotations

import math

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from bigmodel_lab.evaluation.runner import (
    build_compare_report,
    continuation_logprob,
    multiple_choice_eval,
    score_by_rule,
)
from bigmodel_lab.models.gpt import GPT, GPTConfig


def test_score_by_rule_all_types():
    assert score_by_rule("你好", {"type": "exact", "value": "你好"})
    assert not score_by_rule("你好！", {"type": "exact", "value": "你好"})
    assert score_by_rule("三个字", {"type": "length_chars", "min": 3, "max": 3})
    assert score_by_rule("我喜欢人工智能", {"type": "contains", "value": "人工智能"})
    assert score_by_rule("苹果", {"type": "not_contains", "value": "不"})
    assert score_by_rule("Paris", {"type": "contains_any", "values": ["Paris", "paris"]})


def test_build_compare_report_sections():
    current = {
        "tiny_models": {"s": {"perplexity": {"perplexity": 10.0}, "multiple_choice": {"accuracy": 0.25}}},
        "generation": {"pass_rate": 0.5, "passed": 4, "total": 8, "results": []},
    }
    cross = {
        "stage4_summary": {"presets": {"s": {"params": 16_900_000, "tokens_seen": 2_000_000, "final_val_loss": 6.9, "tokens_per_second": 19_000}}},
        "stage4_mlx_performance": {"tokens_per_second": 23_000, "mps_tokens_per_second": 19_700, "mlx_vs_mps_ratio": 1.17},
        "sft_before_after": "# 微调前后\n示例",
        "dpo_side_by_side": "# 对齐对比\n示例",
    }
    report = build_compare_report(current=current, cross=cross)
    assert "预训练规模对比" in report
    assert "后端对比" in report
    assert "评测结果" in report
    assert "SFT 前后对比" in report
    assert "DPO 对齐对比" in report


def _tiny_model_and_tokenizer(tmp_path):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=512, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"], show_progress=False)
    corpus = ["问题：中国的首都是哪座城市？\n选项：\nA. 北京\nB. 上海\n答案：", "答案： A", "答案： B"] * 10
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    path = tmp_path / "tok.json"
    tokenizer.save(str(path))
    fast = PreTrainedTokenizerFast(tokenizer_file=str(path), eos_token="<eos>", pad_token="<pad>", unk_token="<unk>", bos_token="<bos>")
    vocab_size = fast.vocab_size
    config = GPTConfig(vocab_size=vocab_size, block_size=64, n_layer=1, n_head=2, n_embd=32, ffn_hidden=64)
    return GPT(config).eval(), fast


def test_continuation_logprob_finite(tmp_path):
    model, tokenizer = _tiny_model_and_tokenizer(tmp_path)
    value = continuation_logprob(model, tokenizer, "问题：", " A", device="cpu")
    assert math.isfinite(value)
    assert value <= 0


def test_multiple_choice_eval_structure(tmp_path):
    model, tokenizer = _tiny_model_and_tokenizer(tmp_path)
    items = [
        {"question": "中国的首都是哪座城市？", "options": ["北京", "上海"], "answer": 0},
        {"question": "1+1=?", "options": ["2", "3"], "answer": 0},
    ]
    result = multiple_choice_eval(model, tokenizer, items, device="cpu")
    assert result["total"] == 2
    assert 0 <= result["accuracy"] <= 1
    assert len(result["predictions"]) == 2
    assert result["random_baseline"] == 0.5
