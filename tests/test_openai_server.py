"""OpenAI 兼容服务契约测试（使用假生成器，不加载真实模型）。"""

from __future__ import annotations

import threading

from bigmodel_lab.serve.generate import SamplingParams
from bigmodel_lab.serve.openai_server import (
    create_app,
    free_port,
    request_json,
    request_sse,
    start_server,
    stop_server,
)


class FakeGenerator:
    backend = "fake"

    def generate(self, messages, params: SamplingParams):
        return {
            "text": "你好，这是测试回复。",
            "elapsed_seconds": 0.1,
            "ttft_seconds": 0.05,
            "tokens_per_second": 100.0,
            "backend": self.backend,
        }

    def stream(self, messages, params: SamplingParams):
        yield "你好"
        yield "，这是流式回复。"


def _urls(port: int) -> dict[str, str]:
    base = f"http://127.0.0.1:{port}"
    return {"models": f"{base}/v1/models", "chat": f"{base}/v1/chat/completions"}


def test_server_contract():
    app = create_app(FakeGenerator(), model_name="fake-model")
    port = free_port()
    server, thread = start_server(app, port)
    urls = _urls(port)
    try:
        status, payload = request_json(urls["models"])
        assert status == 200
        assert payload["object"] == "list"
        assert payload["data"][0]["id"] == "fake-model"

        status, payload = request_json(urls["chat"], {"messages": [{"role": "user", "content": "你好"}]})
        assert status == 200
        assert payload["object"] == "chat.completion"
        assert payload["choices"][0]["message"]["role"] == "assistant"
        assert payload["choices"][0]["message"]["content"]
        assert payload["usage"]["total_tokens"] > 0

        status, payload = request_json(urls["chat"], {"messages": []})
        assert status == 400
        assert "error" in payload and payload["error"]["type"] == "invalid_request_error"

        status, payload = request_json(urls["chat"], {"messages": [{"role": "user", "content": "hi"}], "temperature": 5})
        assert status == 400
        assert "temperature" in payload["error"]["message"]

        status, events = request_sse(urls["chat"], {"messages": [{"role": "user", "content": "你好"}], "stream": True})
        assert status == 200
        assert any("[DONE]" in event for event in events)
        assert any("chat.completion.chunk" in event for event in events)
        assert any("这是流式回复" in event for event in events)

        results: list[int] = []

        def call():
            code, _ = request_json(urls["chat"], {"messages": [{"role": "user", "content": "并发"}]})
            results.append(code)

        threads = [threading.Thread(target=call) for _ in range(2)]
        for item in threads:
            item.start()
        for item in threads:
            item.join(timeout=60)
        assert results == [200, 200]
    finally:
        stop_server(server, thread)
