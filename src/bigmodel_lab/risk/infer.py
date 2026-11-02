"""审查模型推理：加载导出模型（或回退基座），生成受 JSON schema 约束的审查报告。

设计约束（对应 `specs/tx-risk-model` 与 design.md D7）：

- 模型只作为规则引擎的 "第二意见"：输出解析/校验失败、生成异常或超时都必须
  **降级**为 `ModelOpinion(available=False, degraded=True)`，由调用方继续使用规则基线。
- `review()` 除 `KeyboardInterrupt` 外绝不向外抛异常，保证 `lab risk review` 不因模型
  问题失败；`build_reviewer()` 允许抛 `FileNotFoundError`（模型缺失），由 pipeline 捕获后
  降级 rules_only。
- 输入 prompt 复用 `risk.model_data` 的 system rubric 与请求序列化；输出先做 JSON 提取
  （兼容 markdown fence / 前后文本 / 括号配对），再用 `ReviewOutput` 校验必需字段。
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from ..serve.generate import ChatSession, SamplingParams, build_generator
from .config import RiskConfig, resolve_project_path
from .model_data import SYSTEM_PROMPT, ReviewOutput, request_to_user_content
from .schema import ModelOpinion, ReviewRequest

logger = logging.getLogger(__name__)

DEFAULT_TOP_P = 0.9


def _iter_json_candidates(text: str) -> list[str]:
    """从文本中逐个提取括号配对的 JSON 对象候选（忽略字符串内的括号）。"""
    candidates: list[str] = []
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
        start = text.find("{", start + 1)
    return candidates


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出中提取第一个合法 JSON 对象；失败返回 None。

    依次尝试：整体 JSON → ```json fence → 任意 ``` fence → 全文括号配对扫描。
    """
    if not text:
        return None
    stripped = text.strip()
    for candidate in (stripped, *(_FENCE_RE.findall(stripped))):
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            return payload
    for candidate in _iter_json_candidates(stripped):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


class ModelReviewer:
    """审查模型推理器：懒加载生成器，单次审查失败一律返回降级 `ModelOpinion`。"""

    def __init__(
        self,
        risk_config: RiskConfig,
        *,
        model_dir: str | Path | None = None,
        backend: str | None = None,
        generator: Any | None = None,
        timeout_seconds: float | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        source: str | None = None,
    ) -> None:
        self.config = risk_config
        self.source = source or "base"
        model_settings = risk_config.settings.model
        self.backend = backend or model_settings.backend
        self.model_dir = Path(model_dir) if model_dir is not None else resolve_project_path(
            model_settings.train.base_model_dir
        )
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None else model_settings.timeout_seconds
        )
        self.max_new_tokens = int(max_new_tokens if max_new_tokens is not None else model_settings.max_new_tokens)
        self.temperature = float(temperature if temperature is not None else model_settings.temperature)
        self._generator = generator
        self._session: ChatSession | None = None

    # ---------- 懒加载与描述 ----------
    def describe(self) -> dict[str, Any]:
        """返回模型元信息，供审查报告记录版本/路径。"""
        return {
            "backend": self.backend,
            "source": self.source,
            "model_dir": str(self.model_dir),
            "timeout_seconds": self.timeout_seconds,
            "max_new_tokens": self.max_new_tokens,
            "loaded": self._generator is not None,
        }

    def _ensure_session(self) -> ChatSession:
        """懒加载生成器；每次审查使用新的 ChatSession，避免请求间上下文串扰。"""
        if self._generator is None:
            logger.info("加载审查模型: backend=%s model=%s", self.backend, self.model_dir)
            self._generator = build_generator(self.backend, self.model_dir)
        params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=DEFAULT_TOP_P,
        )
        self._session = ChatSession(generator=self._generator, system_prompt=SYSTEM_PROMPT, params=params)
        return self._session

    # ---------- 降级 ----------
    @staticmethod
    def _degraded(
        reason: str,
        *,
        raw_output: str | None = None,
        invalid_output: bool = False,
    ) -> ModelOpinion:
        return ModelOpinion(
            available=False,
            degraded=True,
            degraded_reason=reason,
            invalid_output=invalid_output,
            raw_output=raw_output,
        )

    @staticmethod
    def _invalid(reason: str, raw_output: str | None) -> ModelOpinion:
        return ModelReviewer._degraded(reason, raw_output=raw_output, invalid_output=True)

    @staticmethod
    def _opinion_from_output(output: ReviewOutput, *, raw_output: str) -> ModelOpinion:
        return ModelOpinion(
            available=True,
            degraded=False,
            degraded_reason=None,
            invalid_output=False,
            risk_level=output.risk_level,
            recommended_action=output.recommended_action,
            capability_delta=output.capability_delta,
            confidence=output.confidence,
            risk_factors=list(output.risk_factors),
            evidence=list(output.evidence),
            raw_output=raw_output,
        )

    # ---------- 审查 ----------
    def review(self, request: ReviewRequest) -> ModelOpinion:
        """生成并校验模型审查意见；任何失败都返回降级意见，不抛异常（KeyboardInterrupt 除外）。"""
        started = time.monotonic()
        raw_output: str | None = None
        try:
            user_content = request_to_user_content(request)
            session = self._ensure_session()
            result = session.send(user_content)
            raw_output = str(result.get("text", "")) if isinstance(result, dict) else str(result)
            elapsed = time.monotonic() - started
            if elapsed > self.timeout_seconds:
                return self._degraded(
                    f"模型推理超时: {elapsed:.2f}s > {self.timeout_seconds:.2f}s",
                    raw_output=raw_output,
                )
            parsed = extract_json_object(raw_output)
            if parsed is None:
                return self._invalid("模型输出不是合法 JSON 对象", raw_output)
            try:
                output = ReviewOutput.model_validate(parsed)
            except Exception as exc:  # noqa: BLE001 - pydantic ValidationError
                return self._invalid(f"模型输出未通过 schema 校验: {exc}", raw_output)
            return self._opinion_from_output(output, raw_output=raw_output)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - 推理模块不得把异常抛给审查流程
            logger.warning("审查模型推理失败: %s: %s", type(exc).__name__, exc)
            return self._degraded(f"模型推理异常: {type(exc).__name__}: {exc}", raw_output=raw_output)


def build_reviewer(risk_config: RiskConfig) -> ModelReviewer:
    """构造 `ModelReviewer`：优先导出模型目录，缺失时回退基座模型目录。

    两个目录都不存在时抛 `FileNotFoundError`（附带可操作提示），由 pipeline 捕获后降级
    rules_only；真正的模型加载失败延迟到首次 `review()`，同样以降级意见返回。
    """
    train_settings = risk_config.settings.model.train
    export_dir = resolve_project_path(train_settings.export_dir)
    base_dir = resolve_project_path(train_settings.base_model_dir)
    if export_dir.exists():
        source = "exported"
        model_dir = export_dir
    elif base_dir.exists():
        source = "base"
        model_dir = base_dir
    else:
        raise FileNotFoundError(
            "未找到审查模型：导出目录 "
            f"{export_dir} 与基座目录 {base_dir} 均不存在；请先运行 `lab risk train --export` "
            "或下载 Qwen3-0.6B"
        )
    logger.info("审查模型来源: %s（%s）", source, model_dir)
    return ModelReviewer(risk_config, model_dir=model_dir, source=source)


__all__ = ["DEFAULT_TOP_P", "ModelReviewer", "build_reviewer", "extract_json_object"]
