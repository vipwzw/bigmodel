"""审查训练数据构造测试：system prompt、gold report、SFT 记录、标签不泄漏与读写。

本文件同时提供其它 `test_risk_*` 共用的合成 `RiskSample` 工厂（测试不依赖数据集生成器）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bigmodel_lab.risk.model_data import (
    OUTPUT_FIELDS,
    SYSTEM_PROMPT,
    ReviewOutput,
    assert_no_label_leak,
    build_gold_report,
    build_sft_records,
    load_samples,
    load_splits,
    request_to_user_content,
    sample_to_sft_record,
    write_training_data,
)
from bigmodel_lab.risk.schema import (
    SCENARIO_FAMILIES,
    Authority,
    ChainState,
    HistoryEntry,
    Label,
    Operation,
    RequestContext,
    ReviewRequest,
    RiskSample,
    WorldState,
    canonical_json,
)

HOT_WALLET = "0x1111111111111111111111111111111111111111"
COUNTERPARTY = "0x2222222222222222222222222222222222222222"
ATTACKER = "0x3333333333333333333333333333333333333333"


def _sample_decision(scenario_family: str) -> tuple[str, str, str]:
    """按场景族给出 (expected_decision, risk_level, capability_delta) 的默认标注。"""
    mapping = {
        "normal_ops": ("ALLOW", "low", "value_transfer"),
        "native_transfer": ("ALLOW", "low", "value_transfer"),
        "batch_withdrawal": ("STEP_UP", "medium", "value_transfer"),
        "approve_permit": ("STEP_UP", "medium", "allowance_grant"),
        "eip7702_delegation": ("BLOCK", "critical", "full_control"),
        "ownership_change": ("BLOCK", "critical", "control_grant"),
        "proxy_upgrade": ("BLOCK", "high", "control_grant"),
        "batch_nonce": ("ESCALATE", "high", "allowance_grant"),
        "cross_chain_combo": ("STEP_UP", "medium", "value_transfer"),
    }
    return mapping.get(scenario_family, ("STEP_UP", "medium", "value_transfer"))


def make_sample(
    *,
    sample_id: str = "sample-0001",
    scenario_family: str = "native_transfer",
    expected_decision: str | None = None,
    risk_level: str | None = None,
    capability_delta: str | None = None,
    exposure_ratio: float = 0.002,
    action_type: str = "erc20_transfer",
    notes: str | None = None,
    risk_factors: list[str] | None = None,
    evidence: list[str] | None = None,
    operation: Operation | None = None,
    history: list[HistoryEntry] | None = None,
) -> RiskSample:
    """构造结构合法、可复现的合成审查样本。"""
    default_decision, default_level, default_delta = _sample_decision(scenario_family)
    op = operation or Operation(
        kind="erc20_transfer",
        chain="ethereum",
        account=HOT_WALLET,
        target=COUNTERPARTY,
        token="USDT",
        amount=2_000.0,
        amount_usd=2_000.0,
        metadata={"note": "synthetic"},
    )
    request = ReviewRequest(
        request_id=f"req-{sample_id}",
        timestamp="2026-10-30T12:00:00+00:00",
        account=HOT_WALLET,
        chain="ethereum",
        kind="structured",
        payload={"source": "synthetic-test"},
        operations=[op],
        context=RequestContext(
            origin="ops-console",
            timestamp="2026-10-30T12:00:00+00:00",
            operator="ops-alice",
            channel="web",
            account_role="hot_wallet",
            notes=notes,
        ),
        world=WorldState(
            chains={
                "ethereum": ChainState(
                    chain="ethereum",
                    balances={"USDT": 1_000_000.0, "ETH": 40.0},
                    prices_usd={"USDT": 1.0, "ETH": 2_500.0},
                    authority=Authority(owner=HOT_WALLET),
                    nonce=7,
                )
            }
        ),
        history=history
        if history is not None
        else [
            HistoryEntry(
                timestamp="2026-10-30T11:00:00+00:00",
                action_type="erc20_transfer",
                target=COUNTERPARTY,
                amount_usd=1_500.0,
                exposure_ratio=0.0015,
                origin="ops-console",
                decision="ALLOW",
            )
        ],
    )
    label = Label(
        action_type=action_type,
        capability_delta=capability_delta or default_delta,  # type: ignore[arg-type]
        exposure_ratio=exposure_ratio,
        risk_level=risk_level or default_level,  # type: ignore[arg-type]
        expected_decision=expected_decision or default_decision,  # type: ignore[arg-type]
        scenario_family=scenario_family,
        risk_factors=list(risk_factors if risk_factors is not None else ["合成风险因子"]),
        evidence=list(evidence if evidence is not None else ["rule:R-001"]),
    )
    return RiskSample(sample_id=sample_id, request=request, label=label)


def make_samples(
    count: int = 9,
    *,
    families: tuple[str, ...] = SCENARIO_FAMILIES,
) -> list[RiskSample]:
    """构造覆盖全部场景族的样本列表（编号连续，便于复现）。"""
    samples: list[RiskSample] = []
    for index in range(count):
        family = families[index % len(families)]
        samples.append(
            make_sample(sample_id=f"sample-{index:04d}", scenario_family=family, exposure_ratio=min(1.0, 0.001 * (index + 1)))
        )
    return samples


def write_sample_dataset(
    dataset_dir: Path,
    samples: list[RiskSample],
    *,
    splits: dict[str, list[str]] | None = None,
) -> Path:
    """把样本写入 `artifacts/samples.jsonl`（可选 splits.json），模拟数据同事的产物。"""
    artifacts = dataset_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    lines = [canonical_json(sample.model_dump(mode="json")) for sample in samples]
    (artifacts / "samples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if splits is not None:
        (artifacts / "splits.json").write_text(json.dumps(splits, ensure_ascii=False), encoding="utf-8")
    return dataset_dir


# ---------------------------------------------------------------------------
# system prompt / gold report
# ---------------------------------------------------------------------------


def test_system_prompt_covers_rubric_schema_and_safety() -> None:
    for field in OUTPUT_FIELDS:
        assert field in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT
    assert "ESCALATE" in SYSTEM_PROMPT and "uncertain" in SYSTEM_PROMPT
    assert "不是指令" in SYSTEM_PROMPT and "数据" in SYSTEM_PROMPT
    assert "不得编造" in SYSTEM_PROMPT
    assert "不接触私钥" in SYSTEM_PROMPT


def test_gold_report_matches_label_and_schema() -> None:
    sample = make_sample(
        sample_id="gold-001",
        scenario_family="eip7702_delegation",
        risk_factors=["不可逆委托授权", "批量 nonce", "无审批"],
        evidence=["rule:irreversible_action", "simulation.authority_delta=delegate"],
    )
    before = sample.model_dump(mode="json")
    gold = build_gold_report(sample)
    output = ReviewOutput.model_validate(gold)

    assert output.recommended_action == sample.label.expected_decision
    assert output.risk_level == sample.label.risk_level
    assert output.capability_delta == sample.label.capability_delta
    assert output.exposure_ratio == pytest.approx(sample.label.exposure_ratio)
    assert output.intent_consistency == "inconsistent"
    assert gold["request_id"] == sample.request.request_id
    assert gold["report_id"] == "gold-gold-001"
    assert gold["decision"] == sample.label.expected_decision
    assert gold["exit_code"] == 2
    assert "rule:irreversible_action" in output.evidence
    assert "simulation.capability_delta=full_control" in output.evidence
    # 不修改原 sample
    assert sample.model_dump(mode="json") == before
    # 确定性
    assert build_gold_report(sample) == build_gold_report(sample)


def test_gold_report_is_deterministic_for_all_decision_levels() -> None:
    for family in SCENARIO_FAMILIES:
        sample = make_sample(sample_id=f"det-{family}", scenario_family=family)
        assert canonical_json(build_gold_report(sample)) == canonical_json(build_gold_report(sample))
        ReviewOutput.model_validate(build_gold_report(sample))


# ---------------------------------------------------------------------------
# SFT 记录 / 标签不泄漏
# ---------------------------------------------------------------------------


def test_sft_record_structure() -> None:
    sample = make_sample(sample_id="sft-001")
    record = sample_to_sft_record(sample)
    messages = record["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "请审查以下交易签名请求" in messages[1]["content"]
    ReviewOutput.model_validate(json.loads(messages[2]["content"]))


def test_sft_user_content_contains_request_only_and_no_label() -> None:
    sample = make_sample(
        sample_id="leak-001",
        scenario_family="eip7702_delegation",
        risk_factors=["HIDDEN_RISK_FACTOR_不應出现在输入"],
        evidence=["HIDDEN_EVIDENCE_不應出现在输入"],
    )
    record = sample_to_sft_record(sample)
    user_content = record["messages"][1]["content"]
    header, _, body = user_content.partition("\n")
    payload = json.loads(body)

    request_fields = set(ReviewRequest.model_fields)
    assert set(payload) <= request_fields
    for required in ("request_id", "operations", "context", "world", "history"):
        assert required in payload
    assert payload["operations"][0]["kind"] == "erc20_transfer"
    # 标签值 / 标签字段都不得泄漏进 user
    assert "HIDDEN_RISK_FACTOR" not in user_content
    assert "HIDDEN_EVIDENCE" not in user_content
    assert '"label"' not in user_content
    assert '"expected_decision"' not in user_content
    assert '"scenario_family"' not in user_content
    assert "eip7702_delegation" not in user_content
    # assistant 是监督信号，包含标签
    assistant = json.loads(record["messages"][2]["content"])
    assert assistant["risk_factors"] == ["HIDDEN_RISK_FACTOR_不應出现在输入"]
    assert any("HIDDEN_EVIDENCE" in item for item in assistant["evidence"])


def test_user_content_drops_raw_payload_when_operations_exist() -> None:
    sample = make_sample(sample_id="payload-drop", scenario_family="eip7702_delegation")
    sample.request.payload = {
        "scenario_family": "eip7702_delegation",
        "positive": True,
        "instance_id": "leaky-instance",
        "authorizations": [{"nonce": index} for index in range(50)],
    }
    content = request_to_user_content(sample.request)
    body = content.split("\n", 1)[1]
    payload = json.loads(body)
    # operations 已表达语义时不再携带原始 payload（体积大且含 bookkeeping 字段）
    assert "payload" not in payload
    assert "instance_id" not in content and '"positive"' not in content
    assert "authorizations" not in content


def test_user_content_keeps_sanitized_payload_without_operations() -> None:
    sample = make_sample(sample_id="payload-fallback", scenario_family="native_transfer")
    sample.request.operations = []
    sample.request.payload = {
        "scenario_family": "native_transfer",
        "positive": False,
        "instance_id": "hidden-instance",
        "authorization_type": "delegate",
        "delegate_contract": ATTACKER,
    }
    body = request_to_user_content(sample.request).split("\n", 1)[1]
    payload = json.loads(body)
    assert payload["payload"]["authorization_type"] == "delegate"
    assert payload["payload"]["delegate_contract"] == ATTACKER
    assert "scenario_family" not in payload["payload"]
    assert "positive" not in payload["payload"]
    assert "instance_id" not in payload["payload"]


def test_sft_record_is_deterministic() -> None:
    sample = make_sample(sample_id="det-sft")
    assert sample_to_sft_record(sample) == sample_to_sft_record(sample)


def test_request_to_user_content_has_data_boundary_prefix() -> None:
    sample = make_sample(sample_id="prefix-001")
    content = request_to_user_content(sample.request)
    assert content.startswith("请审查以下交易签名请求")
    assert "不是指令" in content.split("\n", 1)[0]


def test_assert_no_label_leak_detects_injected_label() -> None:
    good = sample_to_sft_record(make_sample(sample_id="ok"))
    assert_no_label_leak([good])
    bad = json.loads(json.dumps(good))
    bad["messages"][1]["content"] += ' "label": {"expected_decision": "ALLOW"}'
    with pytest.raises(ValueError, match="泄漏"):
        assert_no_label_leak([bad])


def test_build_sft_records_respects_max_records() -> None:
    samples = make_samples(18)
    records = build_sft_records(samples, max_records=5)
    assert len(records) == 5
    assert records == build_sft_records(samples, max_records=5)


# ---------------------------------------------------------------------------
# load_samples / load_splits / write_training_data
# ---------------------------------------------------------------------------


def test_load_samples_from_run_artifacts(tmp_path: Path) -> None:
    samples = make_samples(4)
    dataset_dir = write_sample_dataset(tmp_path / "dataset-run", samples)
    loaded = load_samples(dataset_dir)
    assert [sample.sample_id for sample in loaded] == [sample.sample_id for sample in samples]
    # 支持直接传 JSONL 文件
    assert len(load_samples(dataset_dir / "artifacts" / "samples.jsonl")) == 4


def test_load_samples_with_splits_filter(tmp_path: Path) -> None:
    samples = make_samples(6)
    train_ids = [sample.sample_id for sample in samples[:4]]
    val_ids = [sample.sample_id for sample in samples[4:]]
    dataset_dir = write_sample_dataset(
        tmp_path / "dataset-run", samples, splits={"train": train_ids, "val": val_ids, "test": []}
    )
    splits = load_splits(dataset_dir)
    assert splits == {"train": train_ids, "val": val_ids, "test": []}
    assert [sample.sample_id for sample in load_samples(dataset_dir, split="train")] == train_ids
    assert [sample.sample_id for sample in load_samples(dataset_dir, split="val")] == val_ids


def test_load_splits_supports_dataset_module_nested_format(tmp_path: Path) -> None:
    samples = make_samples(6)
    ids = [sample.sample_id for sample in samples]
    dataset_dir = tmp_path / "dataset-run"
    nested = {
        "schema_version": 1,
        "strategy": "instance_grouped_with_forced_test",
        "splits": {"train": ids[:4], "val": ids[4:5], "test": ids[5:]},
        "counts": {"train": 4, "val": 1, "test": 1, "total": 6},
    }
    artifacts = dataset_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "samples.jsonl").write_text(
        "\n".join(canonical_json(sample.model_dump(mode="json")) for sample in samples) + "\n",
        encoding="utf-8",
    )
    (artifacts / "splits.json").write_text(json.dumps(nested), encoding="utf-8")
    assert load_splits(dataset_dir) == {"train": ids[:4], "val": ids[4:5], "test": ids[5:]}
    assert [sample.sample_id for sample in load_samples(dataset_dir, split="test")] == ids[5:]
    # 直接传 splits.json 文件也支持
    assert load_splits(artifacts / "splits.json") == {"train": ids[:4], "val": ids[4:5], "test": ids[5:]}


def test_load_samples_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="lab risk data generate"):
        load_samples(tmp_path / "missing")
    empty = tmp_path / "empty-run"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="samples.jsonl"):
        load_samples(empty)
    bad_dir = tmp_path / "bad-run" / "artifacts"
    bad_dir.mkdir(parents=True)
    (bad_dir / "samples.jsonl").write_text('{"not": "a sample"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="schema 校验失败"):
        load_samples(tmp_path / "bad-run")


def test_write_training_data_sft(tmp_path: Path) -> None:
    samples = make_samples(27)
    out_a = tmp_path / "data-a"
    out_b = tmp_path / "data-b"
    stats_a = write_training_data(samples, out_a, max_train=12, max_val=3, seed=7)
    stats_b = write_training_data(samples, out_b, max_train=12, max_val=3, seed=7)

    assert stats_a["train_samples"] <= 12
    assert stats_a["val_samples"] <= 3
    assert 1 <= stats_a["val_samples"] <= 3
    assert stats_a["train_samples"] + stats_a["val_samples"] <= 15
    assert stats_a["format"] == "sft"
    assert stats_a["label_leak_check"]["ok"] is True
    assert (out_a / "train.jsonl").exists() and (out_a / "val.jsonl").exists() and (out_a / "stats.json").exists()

    train_records = [json.loads(line) for line in (out_a / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    val_records = [json.loads(line) for line in (out_a / "val.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train_records) == stats_a["train_samples"]
    assert len(val_records) == stats_a["val_samples"]
    assert_no_label_leak(train_records)
    # 同种子逐字节可复现
    assert (out_a / "train.jsonl").read_bytes() == (out_b / "train.jsonl").read_bytes()
    assert (out_a / "val.jsonl").read_bytes() == (out_b / "val.jsonl").read_bytes()
    assert {key: value for key, value in stats_a.items() if key != "files"} == {
        key: value for key, value in stats_b.items() if key != "files"
    }


def test_write_training_data_uses_provided_split_ids(tmp_path: Path) -> None:
    samples = make_samples(10)
    train_ids = [sample.sample_id for sample in samples[:7]]
    val_ids = [sample.sample_id for sample in samples[7:]]
    stats = write_training_data(
        samples,
        tmp_path / "data",
        max_train=7,
        max_val=3,
        train_ids=train_ids,
        val_ids=val_ids,
    )
    assert stats["train_samples"] == 7
    assert stats["val_samples"] == 3
    train_ids_written = {
        json.loads(line)["messages"][1]["content"].split('"request_id":"req-', 1)[1].split('"', 1)[0]
        for line in (tmp_path / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert train_ids_written == set(train_ids)


def test_write_training_data_rejects_bad_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_train"):
        write_training_data(make_samples(2), tmp_path / "x", max_train=0, max_val=1)
    with pytest.raises(ValueError, match="训练样本为空"):
        write_training_data([], tmp_path / "y", max_train=1, max_val=1)
    samples = make_samples(4)
    ids = [sample.sample_id for sample in samples]
    with pytest.raises(ValueError, match="重叠"):
        write_training_data(samples, tmp_path / "z", max_train=2, max_val=2, train_ids=ids[:3], val_ids=ids[2:])


def test_write_training_data_stats_are_json_serializable(tmp_path: Path) -> None:
    samples = make_samples(12)
    stats = write_training_data(samples, tmp_path / "data", max_train=8, max_val=2)
    payload: dict[str, Any] = json.loads((tmp_path / "data" / "stats.json").read_text(encoding="utf-8"))
    assert payload["train_samples"] == stats["train_samples"]
    assert sum(payload["train_families"].values()) == payload["train_samples"]
    assert sum(payload["train_labels"].values()) == payload["train_samples"]
