"""时间预算与磁盘预检测试。"""

from __future__ import annotations

import time

import pytest

from bigmodel_lab.budget import Budget, BudgetExceeded, check_disk, disk_free_gb


def test_profile_defaults():
    sanity = Budget("sanity")
    assert sanity.max_minutes == 10.0
    standard = Budget("standard")
    assert standard.max_minutes is None


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="未知 profile"):
        Budget("turbo")


def test_scale_by_profile():
    assert Budget("sanity").scale(100) == 5
    assert Budget("sanity").scale(10) == 1  # 至少 1
    assert Budget("standard").scale(100) == 100


def test_timeout_raises_budget_exceeded():
    budget = Budget("standard", max_minutes=0.0001)
    time.sleep(0.05)
    with pytest.raises(BudgetExceeded):
        budget.check()


def test_check_disk_detects_insufficient_space(tmp_path):
    assert disk_free_gb(tmp_path) > 0
    ok, message = check_disk(tmp_path, required_gb=10**9)
    assert ok is False
    assert "磁盘不足" in message

    ok, message = check_disk(tmp_path, required_gb=0)
    assert ok is True
    assert "无需额外磁盘" in message
