"""`lab risk review` CLI 测试：单笔/批量、JSON 输出、退出码、产物与模型降级。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bigmodel_lab.cli import main
from bigmodel_lab.risk.schema import (
    ChainState,
    ModelOpinion,
    Operation,
    RequestContext,
    ReviewRequest,
    WorldState,
)

TS = "2026-11-01T00:00:00Z"
PREAPPROVED = "0x" + "aa" * 20


def _world() -> WorldState:
    return WorldState(
        chains={
            "ethereum": ChainState(
                chain="ethereum",
                balances={"ETH": 100.0, "USDC": 100_000.0},
                prices_usd={"ETH": 3000.0, "USDC": 1.0},
                contract_age_days={PREAPPROVED: 30},
            )
        }
    )


def _request(
    operations: list[Operation],
    *,
    request_id: str,
    kind: str = "structured",
    payload: dict[str, Any] | None = None,
    context: RequestContext | None = None,
) -> ReviewRequest:
    return ReviewRequest(
        request_id=request_id,
        timestamp=TS,
        account="0xacc",
        chain="ethereum",
        kind=kind,  # type: ignore[arg-type]
        payload=payload or {},
        operations=operations,
        context=context or RequestContext(origin="api:console", timestamp=TS, account_role="personal"),
        world=_world(),
    )


def _allow_request(request_id: str = "allow-1") -> ReviewRequest:
    """1 ETH 转给预登记对手方：暴露 0.75% ≤ allow_max=1%，只剩 R-002 ALLOW。"""
    return _request(
        [
            Operation(
                kind="native_transfer",
                chain="ethereum",
                account="0xacc",
                target=PREAPPROVED,
                token="ETH",
                amount=1.0,
                amount_usd=3000.0,
            )
        ],
        request_id=request_id,
        context=RequestContext(
            origin="api:console",
            timestamp=TS,
            account_role="personal",
            preapproved_contracts=[PREAPPROVED],
        ),
    )


def _block_request(request_id: str = "block-1") -> ReviewRequest:
    """币安式 7702 批量授权（50 连续 nonce、无审批）：R-001/R-002 应判 BLOCK。"""
    delegate = "0x" + "bb" * 20
    return _request(
        [
            Operation(
                kind="eip7702_authorization",
                chain="ethereum",
                account="0xacc",
                target=delegate,
                delegate=delegate,
                nonce=1300,
                nonce_span=50,
                batch_size=50,
            )
        ],
        request_id=request_id,
        context=RequestContext(
            origin="svc-container-07@internal-signer",
            timestamp=TS,
            account_role="hot_wallet",
        ),
    )


def _escalate_request(request_id: str = "escalate-1") -> ReviewRequest:
    return _request(
        [],
        request_id=request_id,
        kind="evm_tx",
        payload={"to": "0x" + "cc" * 20, "data": "0xdeadbeef" + "00" * 32},
    )


def _write_json(path: Path, request: ReviewRequest) -> Path:
    path.write_text(json.dumps(request.model_dump(mode="json")), encoding="utf-8")
    return path


def _write_jsonl(path: Path, requests: list[ReviewRequest]) -> Path:
    lines = [json.dumps(request.model_dump(mode="json")) for request in requests]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_artifacts(out_dir: Path) -> tuple[Path, Path]:
    runs = sorted((out_dir / "risk" / "review").iterdir())
    assert len(runs) == 1, f"期望一次运行，实际 {runs}"
    run_dir = runs[0]
    return run_dir, run_dir / "artifacts"


def _last_stdout_json(capsys) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# 单笔
# ---------------------------------------------------------------------------


def test_single_request_allow_json(tmp_path, capsys):
    request_path = _write_json(tmp_path / "request.json", _allow_request())
    exit_code = main(
        ["risk", "review", "--request", str(request_path), "--json", "--out", str(tmp_path / "out")]
    )
    assert exit_code == 0
    report = _last_stdout_json(capsys)
    assert report["decision"] == "ALLOW"
    assert report["exit_code"] == 0
    assert report["report_id"] == "rep-allow-1"
    assert report["simulation"]["exposure_ratio"] == pytest.approx(0.0075)
    assert any(hit["rule_id"] == "R-002" for hit in report["rule_hits"])
    assert report["generated_at"] == TS
    assert 0.0 <= report["duration_ms"] < 60_000.0

    run_dir, artifacts = _run_artifacts(tmp_path / "out")
    assert (artifacts / "reports.jsonl").exists()
    assert (artifacts / "summary.json").exists()
    assert (artifacts / "reports" / "allow-1.json").exists()
    summary = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 1
    assert summary["decisions"]["ALLOW"] == 1
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["exit_code"] == 0
    assert "artifacts/reports.jsonl" in manifest["artifacts"]


def test_single_request_block_exit_code(tmp_path, capsys):
    request_path = _write_json(tmp_path / "request.json", _block_request())
    exit_code = main(
        ["risk", "review", "--request", str(request_path), "--json", "--out", str(tmp_path / "out")]
    )
    assert exit_code == 2
    report = _last_stdout_json(capsys)
    assert report["decision"] == "BLOCK"
    assert report["exit_code"] == 2
    rule_ids = {hit["rule_id"] for hit in report["rule_hits"]}
    assert {"R-001", "R-002", "R-004"}.issubset(rule_ids)
    assert report["simulation"]["exposure_ratio"] == pytest.approx(1.0)
    assert report["simulation"]["capability_delta"] == "full_control"


def test_single_request_decode_failure_escalates(tmp_path, capsys):
    request_path = _write_json(tmp_path / "request.json", _escalate_request())
    exit_code = main(
        ["risk", "review", "--request", str(request_path), "--json", "--out", str(tmp_path / "out")]
    )
    assert exit_code == 3
    report = _last_stdout_json(capsys)
    assert report["decision"] == "ESCALATE"
    assert report["simulation"]["failed"] is True
    assert any(hit["rule_id"] == "R-010" for hit in report["rule_hits"])


def test_single_request_human_output(tmp_path, capsys):
    request_path = _write_json(tmp_path / "request.json", _allow_request())
    exit_code = main(["risk", "review", "--request", str(request_path), "--out", str(tmp_path / "out")])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "decision=ALLOW" in captured.out
    assert "汇总" in captured.out
    assert "产物目录" in captured.out


def test_single_sample_wrapper_is_accepted(tmp_path, capsys):
    wrapped = {
        "sample_id": "sample-1",
        "request": _allow_request().model_dump(mode="json"),
        "label": {"expected_decision": "ALLOW"},
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(wrapped), encoding="utf-8")
    exit_code = main(["risk", "review", "--request", str(path), "--json", "--out", str(tmp_path / "out")])
    assert exit_code == 0
    assert _last_stdout_json(capsys)["request_id"] == "allow-1"


# ---------------------------------------------------------------------------
# 批量
# ---------------------------------------------------------------------------


def test_batch_mixed_requests_exit_code_and_artifacts(tmp_path, capsys):
    batch_path = _write_jsonl(
        tmp_path / "batch.jsonl", [_allow_request("allow-1"), _block_request("block-1")]
    )
    exit_code = main(
        ["risk", "review", "--batch", str(batch_path), "--json", "--out", str(tmp_path / "out")]
    )
    assert exit_code == 2
    payload = _last_stdout_json(capsys)
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["decisions"] == {"ALLOW": 1, "STEP_UP": 0, "BLOCK": 1, "ESCALATE": 0}
    assert payload["summary"]["exit_code"] == 2
    assert len(payload["reports"]) == 2

    _run_dir, artifacts = _run_artifacts(tmp_path / "out")
    report_lines = (artifacts / "reports.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(report_lines) == 2
    per_request = sorted((artifacts / "reports").glob("*.json"))
    assert [path.stem for path in per_request] == ["allow-1", "block-1"]


def test_batch_escalate_dominates_exit_code(tmp_path, capsys):
    batch_path = _write_jsonl(
        tmp_path / "batch.jsonl",
        [_block_request("block-1"), _escalate_request("escalate-1")],
    )
    exit_code = main(
        ["risk", "review", "--batch", str(batch_path), "--json", "--out", str(tmp_path / "out")]
    )
    assert exit_code == 3
    payload = _last_stdout_json(capsys)
    assert payload["summary"]["decisions"]["BLOCK"] == 1
    assert payload["summary"]["decisions"]["ESCALATE"] == 1


def test_batch_all_allow_exits_zero(tmp_path, capsys):
    batch_path = _write_jsonl(
        tmp_path / "batch.jsonl", [_allow_request("allow-1"), _allow_request("allow-2")]
    )
    exit_code = main(
        ["risk", "review", "--batch", str(batch_path), "--json", "--out", str(tmp_path / "out")]
    )
    assert exit_code == 0
    assert _last_stdout_json(capsys)["summary"]["decisions"]["ALLOW"] == 2


# ---------------------------------------------------------------------------
# 模型模式（monkeypatch 保证测试不加载真实权重）
# ---------------------------------------------------------------------------


class _FixedReviewer:
    def __init__(self, action: str):
        self.action = action
        self.calls: list[str] = []

    def review(self, request: ReviewRequest) -> ModelOpinion:
        self.calls.append(request.request_id)
        return ModelOpinion(
            available=True,
            recommended_action=self.action,  # type: ignore[arg-type]
            risk_level="high",
            confidence=0.9,
            risk_factors=["fake-reviewer"],
        )


def test_rules_plus_model_falls_back_when_builder_fails(tmp_path, capsys, monkeypatch):
    from bigmodel_lab.risk import pipeline

    monkeypatch.setattr(
        pipeline,
        "_build_reviewer_from_config",
        lambda config: (None, "模型权重缺失: FileNotFoundError"),
    )
    request_path = _write_json(tmp_path / "request.json", _allow_request())
    exit_code = main(
        [
            "risk", "review", "--request", str(request_path),
            "--mode", "rules_plus_model", "--json", "--out", str(tmp_path / "out"),
        ]
    )
    assert exit_code == 0  # 模型不可用不得让审查失败
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["decision"] == "ALLOW"
    assert report["degraded"] is True
    assert any("模型不可用" in note for note in report["notes"])
    assert "降级为 rules_only" in captured.err
    _run_dir, artifacts = _run_artifacts(tmp_path / "out")
    summary = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
    assert summary["degraded"] is True
    assert summary["mode"] == "rules_plus_model"


def test_rules_plus_model_uses_available_opinion(tmp_path, capsys, monkeypatch):
    from bigmodel_lab.risk import pipeline

    reviewer = _FixedReviewer("STEP_UP")
    monkeypatch.setattr(pipeline, "_build_reviewer_from_config", lambda config: (reviewer, None))
    request_path = _write_json(tmp_path / "request.json", _allow_request())
    exit_code = main(
        [
            "risk", "review", "--request", str(request_path),
            "--mode", "rules_plus_model", "--json", "--out", str(tmp_path / "out"),
        ]
    )
    assert exit_code == 0
    report = _last_stdout_json(capsys)
    assert report["decision"] == "STEP_UP"  # 模型可以提升规则决策
    assert report["model_opinion"]["available"] is True
    assert report["degraded"] is False
    assert reviewer.calls == ["allow-1"]


def test_rules_only_does_not_touch_reviewer(tmp_path, capsys, monkeypatch):
    from bigmodel_lab.risk import pipeline

    def _fail(config):
        raise AssertionError("rules_only 模式不应构造模型审查器")

    monkeypatch.setattr(pipeline, "_build_reviewer_from_config", _fail)
    request_path = _write_json(tmp_path / "request.json", _allow_request())
    exit_code = main(
        ["risk", "review", "--request", str(request_path), "--out", str(tmp_path / "out")]
    )
    assert exit_code == 0
    capsys.readouterr()


# ---------------------------------------------------------------------------
# 错误输入 fail-closed
# ---------------------------------------------------------------------------


def test_invalid_request_file_returns_2(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")
    exit_code = main(["risk", "review", "--request", str(path), "--out", str(tmp_path / "out")])
    assert exit_code == 2
    assert "参数错误" in capsys.readouterr().err
    assert not (tmp_path / "out" / "risk").exists()


def test_schema_invalid_request_returns_2(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"request_id": "x"}), encoding="utf-8")
    exit_code = main(["risk", "review", "--request", str(path), "--out", str(tmp_path / "out")])
    assert exit_code == 2
    assert "参数错误" in capsys.readouterr().err


def test_empty_batch_returns_2(tmp_path, capsys):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    exit_code = main(["risk", "review", "--batch", str(path), "--out", str(tmp_path / "out")])
    assert exit_code == 2
    assert "参数错误" in capsys.readouterr().err


def test_missing_request_file_returns_2(tmp_path, capsys):
    exit_code = main(
        ["risk", "review", "--request", str(tmp_path / "missing.json"), "--out", str(tmp_path / "out")]
    )
    assert exit_code == 2
    assert "文件不存在" in capsys.readouterr().err
