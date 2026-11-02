"""DPO 偏好对构造测试：chosen/rejected 差异、标签不泄漏与 `data.chat` 兼容性。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_risk_model_data import make_sample, make_samples

from bigmodel_lab.data.chat import build_dpo_example
from bigmodel_lab.risk.model_data import (
    SYSTEM_PROMPT,
    ReviewOutput,
    build_dpo_records,
    build_gold_report,
    build_rejected_report,
    sample_to_dpo_record,
)
from bigmodel_lab.risk.schema import RISK_LEVEL_RANK, canonical_json


class _FakeTokenizer:
    """最小对话 tokenizer：验证 DPO 记录能被 `build_dpo_example` 处理，不加载真实模型。"""

    eos_token = "<|im_end|>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True):  # noqa: ANN001
        text = "".join(f"<|{message['role']}|>{message['content']}" for message in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")

    def __call__(self, text: str, add_special_tokens: bool = False):  # noqa: ARG002
        return SimpleNamespace(input_ids=[1] * max(1, len(text) // 8))


def test_dpo_record_structure_and_prompt_format() -> None:
    sample = make_sample(sample_id="dpo-001", scenario_family="eip7702_delegation")
    record = sample_to_dpo_record(sample)
    assert set(record) == {"prompt", "chosen", "rejected"}
    assert record["prompt"].startswith("Human: ")
    assert record["prompt"].endswith("\n\nAssistant:")
    assert "Assistant:" not in record["prompt"][: -len("\n\nAssistant:")]
    # prompt 中不含 system prompt（DPO 模板用 user 文本重建），也不含标签
    assert SYSTEM_PROMPT not in record["prompt"]
    assert '"label"' not in record["prompt"]
    assert "eip7702_delegation" not in record["prompt"]


def test_dpo_chosen_matches_gold_and_rejected_is_degraded() -> None:
    sample = make_sample(
        sample_id="dpo-002",
        scenario_family="eip7702_delegation",
        risk_factors=["不可逆委托授权", "批量 nonce"],
        evidence=["rule:irreversible_action", "simulation.authority_delta=delegate"],
    )
    record = sample_to_dpo_record(sample)
    chosen = ReviewOutput.model_validate(json.loads(record["chosen"]))
    rejected = ReviewOutput.model_validate(json.loads(record["rejected"]))
    gold = build_gold_report(sample)

    assert json.loads(record["chosen"]) == gold
    assert chosen.recommended_action == "BLOCK"
    assert chosen.capability_delta == "full_control"
    assert chosen.risk_factors

    # rejected 的四类确定性劣化
    assert rejected.capability_delta == "none"
    assert rejected.recommended_action == "ALLOW"
    assert rejected.risk_factors == []
    assert rejected.evidence == []
    assert rejected.exposure_ratio == 0.0
    assert rejected.confidence < chosen.confidence
    assert RISK_LEVEL_RANK[rejected.risk_level] < RISK_LEVEL_RANK[chosen.risk_level]
    assert record["chosen"] != record["rejected"]


def test_rejected_always_worse_than_chosen_even_for_allow_samples() -> None:
    sample = make_sample(sample_id="dpo-allow", scenario_family="normal_ops")
    chosen = ReviewOutput.model_validate(build_gold_report(sample))
    rejected = ReviewOutput.model_validate(build_rejected_report(sample))
    assert rejected.recommended_action == "ALLOW"
    assert rejected.evidence == []
    assert rejected.confidence < chosen.confidence
    assert rejected.capability_delta == "none"


def test_dpo_prompt_does_not_leak_label_values() -> None:
    sample = make_sample(
        sample_id="dpo-leak",
        scenario_family="ownership_change",
        risk_factors=["HIDDEN_DPO_RISK"],
        evidence=["HIDDEN_DPO_EVIDENCE"],
    )
    prompt = sample_to_dpo_record(sample)["prompt"]
    assert "HIDDEN_DPO_RISK" not in prompt
    assert "HIDDEN_DPO_EVIDENCE" not in prompt
    assert '"expected_decision"' not in prompt
    assert '"scenario_family"' not in prompt


def test_dpo_records_are_deterministic_and_limited() -> None:
    samples = make_samples(18)
    first = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in build_dpo_records(samples, max_records=6)]
    second = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in build_dpo_records(samples, max_records=6)]
    assert first == second
    assert len(build_dpo_records(samples)) == len(samples)


def test_dpo_record_is_compatible_with_build_dpo_example() -> None:
    sample = make_sample(sample_id="dpo-chat", scenario_family="approve_permit")
    record = sample_to_dpo_record(sample)
    tokenizer = _FakeTokenizer()
    example, reason = build_dpo_example(tokenizer, record, max_seq_len=2048)
    assert reason is None
    assert example is not None
    assert example["chosen_ids"] and example["rejected_ids"]
    assert len(example["chosen_ids"]) >= len(example["rejected_ids"]) - 5
    assert "请审查以下交易签名请求" in example["user_text"]


def test_dpo_rejected_still_valid_json_for_chat_template() -> None:
    sample = make_sample(sample_id="dpo-json", scenario_family="proxy_upgrade")
    record = sample_to_dpo_record(sample)
    # chosen/rejected 都是合法 JSON 对象，且不依赖 markdown fence
    assert isinstance(json.loads(record["chosen"]), dict)
    assert isinstance(json.loads(record["rejected"]), dict)
    assert canonical_json(json.loads(record["chosen"])) == record["chosen"]


@pytest.mark.parametrize("family", ["native_transfer", "batch_nonce", "cross_chain_combo"])
def test_dpo_rejected_family_variants(family: str) -> None:
    sample = make_sample(sample_id=f"dpo-{family}", scenario_family=family)
    record = sample_to_dpo_record(sample)
    chosen = json.loads(record["chosen"])
    rejected = json.loads(record["rejected"])
    assert rejected["capability_delta"] == "none"
    assert rejected["recommended_action"] == "ALLOW"
    assert rejected["evidence"] == []
    assert rejected["risk_factors"] == []
    assert rejected["exposure_ratio"] == 0.0
    assert RISK_LEVEL_RANK[rejected["risk_level"]] <= RISK_LEVEL_RANK[chosen["risk_level"]]
