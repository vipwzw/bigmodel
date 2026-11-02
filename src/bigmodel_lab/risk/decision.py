"""四档决策融合：规则地板 + 模型第二意见，模型只能维持或提升。

排序 `ALLOW < STEP_UP < ESCALATE < BLOCK`（见 `schema.DECISION_RANK`）。
- 规则命中取最高 `suggested_decision` 作为地板；
- 只有 `available=True` 的模型意见参与融合；
- 模型意见永远不能把规则决策降低（例如 BLOCK 不会变成 ALLOW/STEP_UP）；
- 无规则命中且模型不可用时返回 ALLOW（证据为空，可追溯）。

退出码复用 `schema.exit_code_for`：ALLOW/STEP_UP=0、BLOCK=2、ESCALATE=3。
"""

from __future__ import annotations

from .schema import (
    DECISION_RANK,
    Decision,
    ModelOpinion,
    RuleHit,
    exit_code_for,
    max_decision,
)


def rules_floor(rule_hits: list[RuleHit]) -> Decision:
    """规则地板：所有命中建议中的最高等级；无命中为 ALLOW。"""
    if not rule_hits:
        return "ALLOW"
    return max_decision(*(hit.suggested_decision for hit in rule_hits))


def model_decision(model_opinion: ModelOpinion | None) -> Decision | None:
    """从模型意见中提取可用建议；不可用/降级/非法输出返回 None。"""
    if model_opinion is None or not model_opinion.available:
        return None
    return model_opinion.recommended_action


def fuse_decision(
    rule_hits: list[RuleHit],
    model_opinion: ModelOpinion | None = None,
) -> Decision:
    """融合规则命中与模型意见，返回最终四档决策。

    Args:
        rule_hits: 规则命中列表（可为空）。
        model_opinion: 可选模型意见；只有 `available=True` 时参与融合。

    Returns:
        最终决策；模型不能降低规则地板。
    """
    floor = rules_floor(rule_hits)
    opinion = model_decision(model_opinion)
    if opinion is None:
        return floor
    fused = max_decision(floor, opinion)
    # 显式保证不可降级：即使 max_decision 语义将来变化，也取不低于规则地板者
    if DECISION_RANK[fused] < DECISION_RANK[floor]:
        return floor
    return fused


def decision_exit_code(decision: str) -> int:
    """决策到退出码的映射（直接复用 schema 的退出码契约）。"""
    return exit_code_for(decision)


__all__ = [
    "decision_exit_code",
    "fuse_decision",
    "model_decision",
    "rules_floor",
]
