"""环境自检测试（不访问真实网络）。"""

from __future__ import annotations

from bigmodel_lab.config import NetworkConfig
from bigmodel_lab.doctor import (
    CheckResult,
    check_disk,
    check_network,
    check_python,
    format_report,
    overall_status,
    report_json,
)


def test_check_python_returns_result():
    result = check_python()
    assert result.name == "python"
    assert result.status in {"PASS", "FAIL"}


def test_check_disk_returns_status(tmp_path):
    result = check_disk(tmp_path)
    assert result.name == "disk"
    assert result.status in {"PASS", "WARN", "FAIL"}
    assert "GB" in result.detail


def test_offline_network_probe_is_skipped():
    results = check_network(NetworkConfig(), offline=True)
    assert len(results) == 1
    assert results[0].status == "PASS"
    assert "离线" in results[0].detail


def test_overall_status_takes_worst():
    results = [CheckResult("a", "PASS", ""), CheckResult("b", "WARN", "")]
    assert overall_status(results) == "WARN"
    results.append(CheckResult("c", "FAIL", ""))
    assert overall_status(results) == "FAIL"


def test_format_report_contains_summary_and_hint():
    results = [
        CheckResult("a", "PASS", "ok"),
        CheckResult("b", "FAIL", "broken", fix="run something"),
    ]
    text = format_report(results)
    assert "[PASS]" in text and "[FAIL]" in text
    assert "修复建议" in text
    assert "结论: FAIL" in text


def test_report_json_is_valid_json():
    import json

    results = [CheckResult("a", "PASS", "ok", data={"k": 1})]
    payload = json.loads(report_json(results))
    assert payload["status"] == "PASS"
    assert payload["checks"][0]["name"] == "a"
    assert payload["checks"][0]["data"] == {"k": 1}
