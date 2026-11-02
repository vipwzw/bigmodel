"""审查模型推理测试：JSON 提取、schema 校验、超时/异常降级与 build_reviewer 回退。

全部用 fake generator，不加载 0.6B 模型。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from test_risk_model_data import make_sample

from bigmodel_lab.risk import infer
from bigmodel_lab.risk.config import load_risk_config
from bigmodel_lab.risk.infer import ModelReviewer, build_reviewer, extract_json_object
from bigmodel_lab.risk.model_data import SYSTEM_PROMPT, build_gold_report
from bigmodel_lab.risk.schema import ReviewRequest, canonical_json


class FakeGenerator:
    """按顺序返回预设输出的假生成器：str 作为文本，Exception 作为异常抛出。"""

    backend = "fake"

    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages, params):  # noqa: ANN001
        self.calls.append({"messages": [dict(message) for message in messages], "params": params})
        if not self.outputs:
            raise AssertionError("FakeGenerator 没有更多输出")
        output = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(output, BaseException):
            raise output
        if callable(output):
            return {"text": output(messages)}
        return {"text": str(output), "backend": self.backend}


def _config_with_model_dirs(tmp_path: Path) -> Any:
    cfg = load_risk_config()
    cfg.settings.model.train.base_model_dir = str(tmp_path / "base-model")
    cfg.settings.model.train.export_dir = str(tmp_path / "export-model")
    cfg.settings.model.timeout_seconds = 5.0
    cfg.settings.model.max_new_tokens = 128
    cfg.settings.model.temperature = 0.0
    return cfg


def _gold_json(sample_id: str = "infer-001", family: str = "eip7702_delegation") -> str:
    return canonical_json(build_gold_report(make_sample(sample_id=sample_id, scenario_family=family)))


# ---------------------------------------------------------------------------
# extract_json_object
# ---------------------------------------------------------------------------


def test_extract_json_object_plain() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_markdown_fence() -> None:
    text = '推理如下：\n```json\n{"action_type": "approve", "risk_level": "medium"}\n```\n完成'
    assert extract_json_object(text) == {"action_type": "approve", "risk_level": "medium"}


def test_extract_json_object_surrounding_text_and_braces_in_strings() -> None:
    text = '前缀 {not json} 中间 {"note": "值{含}括号", "nested": {"b": 1}} 后缀'
    payload = extract_json_object(text)
    assert payload is not None
    assert payload["note"] == "值{含}括号"
    assert payload["nested"] == {"b": 1}


def test_extract_json_object_handles_escaped_quotes() -> None:
    payload = extract_json_object('{"evidence": ["rule:{\\"id\\":1}"], "ok": true}')
    assert payload is not None
    assert payload["ok"] is True


def test_extract_json_object_returns_none_for_invalid() -> None:
    assert extract_json_object("") is None
    assert extract_json_object("没有任何 JSON") is None
    assert extract_json_object("[1, 2, 3]") is None
    assert extract_json_object("{不是 JSON}") is None


# ---------------------------------------------------------------------------
# build_reviewer：导出模型优先，回退基座，缺失报错
# ---------------------------------------------------------------------------


def test_build_reviewer_prefers_export_dir(tmp_path: Path) -> None:
    cfg = _config_with_model_dirs(tmp_path)
    (tmp_path / "base-model").mkdir()
    (tmp_path / "export-model").mkdir()
    reviewer = build_reviewer(cfg)
    assert isinstance(reviewer, ModelReviewer)
    assert reviewer.source == "exported"
    assert reviewer.model_dir == tmp_path / "export-model"
    assert reviewer.describe()["source"] == "exported"


def test_build_reviewer_falls_back_to_base_model(tmp_path: Path) -> None:
    cfg = _config_with_model_dirs(tmp_path)
    (tmp_path / "base-model").mkdir()
    reviewer = build_reviewer(cfg)
    assert reviewer.source == "base"
    assert reviewer.model_dir == tmp_path / "base-model"


def test_build_reviewer_raises_actionable_error_without_any_model(tmp_path: Path) -> None:
    cfg = _config_with_model_dirs(tmp_path)
    with pytest.raises(FileNotFoundError, match="lab risk train --export"):
        build_reviewer(cfg)


# ---------------------------------------------------------------------------
# review：成功路径
# ---------------------------------------------------------------------------


def test_review_valid_json_returns_available_opinion(tmp_path: Path) -> None:
    cfg = _config_with_model_dirs(tmp_path)
    sample = make_sample(sample_id="infer-ok", scenario_family="eip7702_delegation")
    generator = FakeGenerator([_gold_json("infer-ok")])
    reviewer = ModelReviewer(cfg, model_dir=tmp_path, backend="transformers", generator=generator)
    opinion = reviewer.review(sample.request)

    assert opinion.available is True
    assert opinion.degraded is False
    assert opinion.invalid_output is False
    assert opinion.risk_level == sample.label.risk_level
    assert opinion.recommended_action == sample.label.expected_decision
    assert opinion.capability_delta == sample.label.capability_delta
    assert opinion.confidence is not None and 0.0 <= opinion.confidence <= 1.0
    assert opinion.raw_output and "capability_delta" in opinion.raw_output
    # FakeGenerator 收到的 messages 使用统一 system prompt 与请求序列化
    messages = generator.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "请审查以下交易签名请求" in messages[1]["content"]
    assert sample.request.request_id in messages[1]["content"]
    params = generator.calls[0]["params"]
    assert params.max_tokens == cfg.settings.model.max_new_tokens
    assert params.temperature == cfg.settings.model.temperature


def test_review_parses_markdown_fenced_output(tmp_path: Path) -> None:
    cfg = _config_with_model_dirs(tmp_path)
    sample = make_sample(sample_id="infer-fence")
    fenced = "我的判断：\n```json\n" + _gold_json("infer-fence", "native_transfer") + "\n```\n以上。"
    reviewer = ModelReviewer(cfg, model_dir=tmp_path, generator=FakeGenerator([fenced]))
    opinion = reviewer.review(sample.request)
    assert opinion.available is True
    assert opinion.recommended_action == sample.label.expected_decision


def test_review_uses_lazy_generator_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config_with_model_dirs(tmp_path)
    sample = make_sample(sample_id="infer-lazy")
    generator = FakeGenerator([_gold_json("infer-lazy")])
    calls: list[tuple[str, Path]] = []

    def fake_build_generator(backend: str, model_path: Path):  # noqa: ANN202
        calls.append((backend, Path(model_path)))
        return generator

    monkeypatch.setattr(infer, "build_generator", fake_build_generator)
    reviewer = ModelReviewer(cfg, model_dir=tmp_path / "model", backend="transformers")
    assert reviewer.describe()["loaded"] is False
    first = reviewer.review(sample.request)
    second = reviewer.review(sample.request)
    assert first.available is True and second.available is True
    assert calls == [("transformers", tmp_path / "model")]  # 生成器只加载一次
    assert len(generator.calls) == 2  # 每次审查一个新的 ChatSession
    assert reviewer.describe()["loaded"] is True


# ---------------------------------------------------------------------------
# review：降级路径
# ---------------------------------------------------------------------------


def _degraded_reviewer(tmp_path: Path, output: Any, **kwargs: Any) -> tuple[ModelReviewer, ReviewRequest]:
    cfg = _config_with_model_dirs(tmp_path)
    sample = make_sample(sample_id="infer-bad", scenario_family="ownership_change")
    reviewer = ModelReviewer(cfg, model_dir=tmp_path, generator=FakeGenerator([output]), **kwargs)
    return reviewer, sample.request


def test_review_invalid_json_degrades(tmp_path: Path) -> None:
    reviewer, request = _degraded_reviewer(tmp_path, "这不是 JSON")
    opinion = reviewer.review(request)
    assert opinion.available is False
    assert opinion.degraded is True
    assert opinion.invalid_output is True
    assert "JSON" in (opinion.degraded_reason or "")
    assert opinion.raw_output == "这不是 JSON"


def test_review_missing_required_fields_degrades(tmp_path: Path) -> None:
    reviewer, request = _degraded_reviewer(tmp_path, '{"action_type": "approve"}')
    opinion = reviewer.review(request)
    assert opinion.available is False
    assert opinion.invalid_output is True
    assert "schema" in (opinion.degraded_reason or "")


def test_review_out_of_range_fields_degrades(tmp_path: Path) -> None:
    payload = json.loads(_gold_json("infer-range"))
    payload["confidence"] = 2.0
    reviewer, request = _degraded_reviewer(tmp_path, canonical_json(payload))
    opinion = reviewer.review(request)
    assert opinion.available is False
    assert opinion.invalid_output is True


def test_review_exception_degrades(tmp_path: Path) -> None:
    reviewer, request = _degraded_reviewer(tmp_path, RuntimeError("模型加载失败"))
    opinion = reviewer.review(request)
    assert opinion.available is False
    assert opinion.degraded is True
    assert opinion.invalid_output is False
    assert "RuntimeError" in (opinion.degraded_reason or "")
    assert "模型加载失败" in (opinion.degraded_reason or "")


def test_review_timeout_degrades(tmp_path: Path) -> None:
    cfg = _config_with_model_dirs(tmp_path)
    sample = make_sample(sample_id="infer-slow")

    def slow_generate(messages):  # noqa: ANN001, ARG001
        time.sleep(0.02)
        return _gold_json("infer-slow")

    reviewer = ModelReviewer(
        cfg, model_dir=tmp_path, generator=FakeGenerator([slow_generate]), timeout_seconds=0.001
    )
    opinion = reviewer.review(sample.request)
    assert opinion.available is False
    assert opinion.degraded is True
    assert "超时" in (opinion.degraded_reason or "")


def test_review_never_raises_for_unexpected_request(tmp_path: Path) -> None:
    cfg = _config_with_model_dirs(tmp_path)
    reviewer = ModelReviewer(cfg, model_dir=tmp_path, generator=FakeGenerator([_gold_json()]))
    opinion = reviewer.review(object())  # type: ignore[arg-type]
    assert opinion.available is False
    assert opinion.degraded is True
    assert opinion.degraded_reason


def test_review_propagates_keyboard_interrupt(tmp_path: Path) -> None:
    reviewer, request = _degraded_reviewer(tmp_path, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        reviewer.review(request)


def test_review_generator_missing_does_not_raise(tmp_path: Path) -> None:
    cfg = _config_with_model_dirs(tmp_path)

    def broken_build_generator(backend: str, model_path: Path):  # noqa: ANN001, ARG001
        raise FileNotFoundError("模型目录损坏")

    sample = make_sample(sample_id="infer-load-fail")
    reviewer = ModelReviewer(cfg, model_dir=tmp_path)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(infer, "build_generator", broken_build_generator)
        opinion = reviewer.review(sample.request)
    assert opinion.available is False
    assert opinion.degraded is True
    assert "FileNotFoundError" in (opinion.degraded_reason or "")
