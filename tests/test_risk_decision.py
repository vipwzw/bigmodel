"""决策测试：四大档位融合、模型不可降级与退出码映射。"""

from __future__ import annotations

from bigmodel_lab.risk.decision import (
    decision_exit_code,
    fuse_decision,
    model_decision,
    rules_floor,
)
from bigmodel_lab.risk.schema import ModelOpinion, RuleHit


def _hit(decision: str, *, rule_id: str = "R-001", severity: str = "medium") -> RuleHit:
    return RuleHit(
        rule_id=rule_id,
        name="测试规则",
        severity=severity,  # type: ignore[arg-type]
        evidence=["evidence"],
        rule_version="v-test",
        suggested_decision=decision,  # type: ignore[arg-type]
    )


def _opinion(action: str | None, *, available: bool = True) -> ModelOpinion:
    return ModelOpinion(
        available=available,
        recommended_action=action,  # type: ignore[arg-type]
        risk_level="high",
    )


def test_no_hits_and_no_model_allows():
    assert fuse_decision([], None) == "ALLOW"
    assert rules_floor([]) == "ALLOW"


def test_rules_floor_takes_highest_hit():
    hits = [_hit("STEP_UP", rule_id="R-002"), _hit("BLOCK", rule_id="R-001")]
    assert rules_floor(hits) == "BLOCK"


def test_model_cannot_downgrade_block():
    hits = [_hit("BLOCK")]
    assert fuse_decision(hits, _opinion("ALLOW")) == "BLOCK"
    assert fuse_decision(hits, _opinion("STEP_UP")) == "BLOCK"


def test_model_can_escalate_rules():
    hits = [_hit("ALLOW", rule_id="R-002")]
    assert fuse_decision(hits, _opinion("STEP_UP")) == "STEP_UP"
    assert fuse_decision([], _opinion("BLOCK")) == "BLOCK"


def test_model_exceeding_block_stays_block():
    assert fuse_decision([_hit("BLOCK")], _opinion("BLOCK")) == "BLOCK"


def test_escalate_and_block_ordering():
    # ESCALATE 与 BLOCK 同时存在时取更严重的 BLOCK
    assert rules_floor([_hit("ESCALATE", rule_id="R-010"), _hit("BLOCK", rule_id="R-001")]) == "BLOCK"


def test_unavailable_model_is_ignored():
    opinion = _opinion("BLOCK", available=False)
    assert model_decision(opinion) is None
    assert fuse_decision([_hit("STEP_UP")], opinion) == "STEP_UP"
    assert fuse_decision([], opinion) == "ALLOW"


def test_degraded_model_without_available_flag_is_ignored():
    opinion = ModelOpinion(available=False, degraded=True, degraded_reason="timeout", recommended_action="BLOCK")
    assert fuse_decision([], opinion) == "ALLOW"
    assert fuse_decision([_hit("STEP_UP")], opinion) == "STEP_UP"


def test_none_recommended_action_is_ignored():
    opinion = ModelOpinion(available=True, recommended_action=None)
    assert model_decision(opinion) is None
    assert fuse_decision([_hit("STEP_UP")], opinion) == "STEP_UP"


def test_exit_code_contract():
    assert decision_exit_code("ALLOW") == 0
    assert decision_exit_code("STEP_UP") == 0
    assert decision_exit_code("BLOCK") == 2
    assert decision_exit_code("ESCALATE") == 3
    assert decision_exit_code("unknown") == 3
