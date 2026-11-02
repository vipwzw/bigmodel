"""特征测试：滚动窗口累计暴露、跨链汇总与确定性。"""

from __future__ import annotations

import pytest

from bigmodel_lab.risk.config import RuleSettings
from bigmodel_lab.risk.features import enrich_features
from bigmodel_lab.risk.schema import (
    ChainState,
    HistoryEntry,
    Operation,
    RequestContext,
    ReviewRequest,
    SimulationResult,
    WorldState,
)
from bigmodel_lab.risk.simulate import simulate_request

TS = "2026-11-01T12:00:00Z"


def _world() -> WorldState:
    return WorldState(
        chains={
            "ethereum": ChainState(
                chain="ethereum",
                balances={"eth": 100.0, "usdc": 100_000.0},
                prices_usd={"eth": 3000.0, "usdc": 1.0},
            ),
            "bsc": ChainState(chain="bsc", balances={"bnb": 1000.0}, prices_usd={"bnb": 600.0}),
        }
    )


def _request(
    history: list[HistoryEntry],
    *,
    operations: list[Operation] | None = None,
    timestamp: str = TS,
    world: WorldState | None = None,
) -> ReviewRequest:
    return ReviewRequest(
        request_id="feat-1",
        timestamp=timestamp,
        account="0xacc",
        chain="ethereum",
        kind="structured",
        operations=operations
        or [Operation(kind="native_transfer", chain="ethereum", account="0xacc", target="0xknown", token="eth", amount=1.0)],
        context=RequestContext(origin="api", timestamp=timestamp, account_role="personal"),
        world=world or _world(),
        history=history,
    )


def _rules(**thresholds) -> RuleSettings:
    return RuleSettings(thresholds={**RuleSettings().thresholds.model_dump(), **thresholds})


def _simulate(request: ReviewRequest):
    return simulate_request(request, request.operations)


def test_rolling_window_includes_recent_history_only():
    request = _request(
        [
            HistoryEntry(
                timestamp="2026-11-01T10:00:00Z",
                action_type="native_transfer",
                amount_usd=1000.0,
                exposure_ratio=0.0025,
            ),
            HistoryEntry(
                timestamp="2026-10-30T00:00:00Z",  # 30 小时前，窗口外
                action_type="native_transfer",
                amount_usd=50_000.0,
                exposure_ratio=0.125,
            ),
            HistoryEntry(
                timestamp="2026-11-01T18:00:00Z",  # 未来，窗口外
                action_type="native_transfer",
                amount_usd=99_000.0,
                exposure_ratio=0.2475,
            ),
        ]
    )
    simulation = _simulate(request)
    enriched = enrich_features(simulation, request, _rules())
    # 当前 1 ETH = 3000 / 400000 = 0.0075；窗口内历史 0.0025
    assert enriched.rolling_exposure_ratio == pytest.approx(0.01)
    assert enriched.window_hours == 24
    # 入参不可变
    assert simulation.rolling_exposure_ratio == 0.0


def test_rolling_window_uses_amount_usd_when_ratio_missing():
    request = _request(
        [
            HistoryEntry(
                timestamp="2026-11-01T11:00:00Z",
                action_type="native_transfer",
                amount_usd=4000.0,
                exposure_ratio=0.0,
            )
        ]
    )
    enriched = enrich_features(_simulate(request), request, _rules())
    # 4000/400000 = 0.01 历史 + 0.0075 当前
    assert enriched.rolling_exposure_ratio == pytest.approx(0.0175)


def test_rolling_window_custom_hours():
    request = _request(
        [
            HistoryEntry(
                timestamp="2026-11-01T06:00:00Z",  # 6 小时前
                action_type="native_transfer",
                amount_usd=1000.0,
                exposure_ratio=0.01,
            )
        ]
    )
    rules = _rules(rolling_window_hours=1)
    enriched = enrich_features(_simulate(request), request, rules)
    assert enriched.window_hours == 1
    assert enriched.rolling_exposure_ratio == pytest.approx(0.0075)


def test_unparsable_history_timestamp_is_counted_conservatively():
    request = _request(
        [
            HistoryEntry(
                timestamp="not-a-timestamp",
                action_type="native_transfer",
                amount_usd=4000.0,
                exposure_ratio=0.01,
            )
        ]
    )
    enriched = enrich_features(_simulate(request), request, _rules())
    assert enriched.rolling_exposure_ratio == pytest.approx(0.0175)


def test_cross_chain_ratio_uses_total_world_nav():
    request = _request([])
    enriched = enrich_features(_simulate(request), request, _rules())
    # 3000 / 1,000,000
    assert enriched.cross_chain_exposure_ratio == pytest.approx(0.003)
    # 链内暴露比例仍是 3000 / 400000
    assert enriched.exposure_ratio == pytest.approx(0.0075)


def test_cross_chain_ratio_for_7702_uses_total_world_nav():
    operations = [Operation(kind="eip7702_authorization", chain="ethereum", account="0xacc", delegate="0xbad")]
    request = _request([], operations=operations)
    enriched = enrich_features(_simulate(request), request, _rules())
    # 链内暴露 400000 / 400000 = 1.0；跨链暴露 400000 / 全部 1,000,000 = 0.4
    assert enriched.cross_chain_exposure_ratio == pytest.approx(0.4)
    assert enriched.exposure_ratio == pytest.approx(1.0)


def test_cross_chain_ratio_capped_at_one():
    request = _request([])
    oversized = SimulationResult(
        exposure_usd=2_500_000.0, exposure_ratio=1.0, nav_usd=400_000.0
    )
    enriched = enrich_features(oversized, request, _rules())
    assert enriched.cross_chain_exposure_ratio == 1.0


def test_failed_simulation_only_updates_window_metadata():
    bad_world = WorldState(chains={})
    request = _request([], world=bad_world)
    simulation = _simulate(request)
    assert simulation.failed is True
    enriched = enrich_features(simulation, request, _rules())
    assert enriched.failed is True
    assert enriched.window_hours == 24
    assert enriched.rolling_exposure_ratio == 0.0


def test_enrichment_is_deterministic():
    request = _request(
        [
            HistoryEntry(
                timestamp="2026-11-01T09:00:00Z",
                action_type="approve",
                amount_usd=500.0,
                exposure_ratio=0.00125,
            )
        ]
    )
    rules = _rules()
    first = enrich_features(_simulate(request), request, rules).model_dump()
    second = enrich_features(_simulate(request), request, rules).model_dump()
    assert first == second
