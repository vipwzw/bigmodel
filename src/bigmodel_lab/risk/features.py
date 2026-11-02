"""审查特征的二次加工：滚动窗口累计暴露、跨链汇总与最坏损失。

`simulate` 只描述「这一笔签名会暴露多少资产」；`enrich_features` 在它之上叠加
同一账户近期历史（`request.history`）的暴露，得到滚动窗口累计暴露比例，并按
账户跨链总净值计算跨链暴露比例。全部计算只依赖请求内数据，结果确定可复现。

口径：
- `rolling_exposure_ratio` = 窗口内历史暴露比例之和 + 当前模拟暴露比例；
  历史条目优先用 `exposure_ratio`，缺失时用 `amount_usd / 当前相关链 NAV` 折算。
- `cross_chain_exposure_ratio` = 当前暴露 USD / 账户全部链 NAV（上限 1.0）。
- 时间戳无法解析或缺失的历史条目按「窗口内」计入（宁可高估，不静默低估风险）。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from .config import RuleSettings
from .schema import HistoryEntry, ReviewRequest, SimulationResult


def enrich_features(
    simulation: SimulationResult,
    request: ReviewRequest,
    rules: RuleSettings,
) -> SimulationResult:
    """在模拟结果上补充滚动窗口与跨链特征，返回新的 `SimulationResult`。

    Args:
        simulation: `simulate_request` 的输出。
        request: 原始审查请求（含 history 与 world）。
        rules: 规则配置（提供滚动窗口小时数与阈值分母所需的版本信息）。

    Returns:
        新的模拟结果（不修改入参）；模拟失败时只更新窗口元数据。
    """
    window_hours = int(rules.thresholds.rolling_window_hours)
    update: dict[str, object] = {"window_hours": window_hours}
    if not simulation.failed:
        window_entries = _recent_history(request, window_hours)
        nav_usd = float(simulation.nav_usd)
        history_ratio = 0.0
        for entry in window_entries:
            ratio = float(entry.exposure_ratio or 0.0)
            if ratio <= 0.0 and nav_usd > 0.0 and entry.amount_usd:
                ratio = float(entry.amount_usd) / nav_usd
            history_ratio += ratio
        rolling_ratio = round(history_ratio + float(simulation.exposure_ratio), 6)
        total_nav = float(request.world.nav_usd())
        if total_nav > 0.0:
            cross_chain_ratio = round(min(float(simulation.exposure_usd) / total_nav, 1.0), 6)
        else:
            cross_chain_ratio = 0.0
        update["rolling_exposure_ratio"] = rolling_ratio
        update["cross_chain_exposure_ratio"] = cross_chain_ratio
        update["worst_loss_usd"] = round(
            max(float(simulation.worst_loss_usd), float(simulation.exposure_usd)), 6
        )
    return simulation.model_copy(update=update)


def _recent_history(request: ReviewRequest, window_hours: int) -> list[HistoryEntry]:
    """筛选滚动窗口内的历史条目；时间不可解析时保守计入。"""
    request_time = _parse_timestamp(request.timestamp)
    if request_time is None:
        return list(request.history)
    cutoff = request_time - timedelta(hours=window_hours)
    entries: list[HistoryEntry] = []
    for entry in request.history:
        entry_time = _parse_timestamp(entry.timestamp)
        if entry_time is None:
            entries.append(entry)
            continue
        if cutoff <= entry_time <= request_time:
            entries.append(entry)
    return entries


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


__all__ = ["enrich_features"]
