"""本地生成接口：MLX（主）/ Transformers（基线），支持流式、采样参数与多轮会话。"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..train.precision import default_device, driver_memory_gb

DEFAULT_SYSTEM_PROMPT = "你是一个乐于助人的中文助手。"

_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """去掉 Qwen3 思考模式产生的 ...</think> 前缀（如果存在）。"""
    return _THINK_RE.sub("", text).strip()


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0

    def validate(self) -> None:
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature 必须在 [0, 2] 内，收到 {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p 必须在 (0, 1] 内，收到 {self.top_p}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens 必须为正整数，收到 {self.max_tokens}")
        if not (0.0 <= self.repetition_penalty <= 2.0):
            raise ValueError(f"repetition_penalty 必须在 [0, 2] 内，收到 {self.repetition_penalty}")


class MLXGenerator:
    """基于 mlx-lm 的本地生成器（Apple Silicon 原生）。"""

    backend = "mlx"

    def __init__(self, model_path: str | Path):
        from mlx_lm import load  # noqa: PLC0415

        self.model_path = str(model_path)
        self.model, self.tokenizer = load(self.model_path)

    def _chat_prompt(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except Exception:  # noqa: BLE001 - 其它模型的模板可能不认这个参数
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def stream(self, messages: list[dict[str, str]], params: SamplingParams) -> Iterator[str]:
        import importlib  # noqa: PLC0415

        params.validate()
        generate_module = importlib.import_module("mlx_lm.generate")
        sample_utils = importlib.import_module("mlx_lm.sample_utils")
        sampler = sample_utils.make_sampler(temp=params.temperature, top_p=params.top_p)
        prompt = self._chat_prompt(messages)
        kwargs: dict = {"max_tokens": params.max_tokens, "sampler": sampler}
        if params.repetition_penalty != 1.0:
            processors = sample_utils.make_logits_processors(repetition_penalty=params.repetition_penalty)
            if processors:
                kwargs["logits_processors"] = processors
        for response in generate_module.stream_generate(self.model, self.tokenizer, prompt, **kwargs):
            if response.text:
                yield response.text

    def generate(self, messages: list[dict[str, str]], params: SamplingParams) -> dict[str, Any]:
        started = time.perf_counter()
        chunks: list[str] = []
        first_token_at: float | None = None
        token_count = 0
        for chunk in self.stream(messages, params):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(chunk)
            token_count += len(chunk)
        elapsed = time.perf_counter() - started
        return {
            "text": strip_thinking("".join(chunks)),
            "elapsed_seconds": round(elapsed, 3),
            "ttft_seconds": round((first_token_at - started) if first_token_at else elapsed, 3),
            "tokens_per_second": round(token_count / max(elapsed, 1e-9), 1),
            "backend": self.backend,
        }


class TransformersGenerator:
    """基于 transformers + PyTorch MPS 的基线生成器。"""

    backend = "transformers"

    def __init__(self, model_path: str | Path, *, device: str | None = None):
        from ..finetune.sft import load_model_and_tokenizer  # noqa: PLC0415

        self.device = device or default_device()
        self.model, self.tokenizer = load_model_and_tokenizer(Path(model_path), device=self.device, dtype_name="bfloat16")
        self.model.eval()

    def stream(self, messages: list[dict[str, str]], params: SamplingParams) -> Iterator[str]:
        from transformers import TextIteratorStreamer  # noqa: PLC0415

        params.validate()
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:  # noqa: BLE001
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=params.max_tokens,
            do_sample=params.temperature > 0,
            temperature=max(params.temperature, 1e-5),
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )
        import threading  # noqa: PLC0415

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()

    def generate(self, messages: list[dict[str, str]], params: SamplingParams) -> dict[str, Any]:
        started = time.perf_counter()
        chunks = list(self.stream(messages, params))
        elapsed = time.perf_counter() - started
        text = strip_thinking("".join(chunks))
        return {
            "text": text,
            "elapsed_seconds": round(elapsed, 3),
            "ttft_seconds": round(elapsed / max(len(chunks), 1), 3),
            "tokens_per_second": round(len(text) / max(elapsed, 1e-9), 1),
            "backend": self.backend,
            "peak_memory_gb": driver_memory_gb(self.device),
        }


def build_generator(backend: str, model_path: str | Path, *, device: str | None = None):
    if backend == "mlx":
        return MLXGenerator(model_path)
    if backend == "transformers":
        return TransformersGenerator(model_path, device=device)
    raise ValueError(f"未知后端: {backend}（可选 mlx / transformers）")


@dataclass
class ChatSession:
    """多轮会话：维护 messages 历史。"""

    generator: Any
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    history: list[dict[str, str]] = field(default_factory=list)
    params: SamplingParams = field(default_factory=SamplingParams)

    def __post_init__(self) -> None:
        self.history = [{"role": "system", "content": self.system_prompt}]

    def send(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        result = self.generator.generate(self.history, self.params)
        self.history.append({"role": "assistant", "content": result["text"]})
        return result["text"]
