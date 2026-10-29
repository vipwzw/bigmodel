"""OpenAI 兼容本地推理服务（FastAPI 封装 mlx-lm / transformers 生成器）。"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .generate import SamplingParams

_SYSTEM_MESSAGE = "你是一个乐于助人的中文助手。"


def _chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:16]}"


def _parse_request(payload: Any) -> tuple[list[dict[str, str]], SamplingParams, bool]:
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须是非空数组")
    cleaned: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] 必须是对象")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}].role 非法: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content 必须是非空字符串")
        cleaned.append({"role": role, "content": content})
    params = SamplingParams(
        max_tokens=int(payload.get("max_tokens", 64)),
        temperature=float(payload.get("temperature", 0.7)),
        top_p=float(payload.get("top_p", 0.9)),
        repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
    )
    params.validate()
    stream = bool(payload.get("stream", False))
    return cleaned, params, stream


def create_app(generator, *, model_name: str = "bigmodel-lab"):
    app = FastAPI(title="bigmodel-lab OpenAI-compatible server", version="0.1")
    lock = threading.Lock()

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "bigmodel-lab"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "请求体不是合法 JSON", "type": "invalid_request_error", "code": "invalid_json"}},
            )
        try:
            messages, params, stream = _parse_request(payload)
        except (ValueError, TypeError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_request_error", "code": "invalid_request"}},
            )
        created = int(time.time())

        if not stream:
            with lock:
                result = generator.generate(messages, params)
            text = result.get("text", "")
            return {
                "id": _chat_id(),
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": sum(len(message["content"]) for message in messages),
                    "completion_tokens": len(text),
                    "total_tokens": sum(len(message["content"]) for message in messages) + len(text),
                },
            }

        def event_stream():
            chat_id = _chat_id()

            def emit(delta: dict[str, Any], finish_reason: str | None = None) -> str:
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }
                return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            yield emit({"role": "assistant", "content": ""})
            try:
                with lock:
                    for piece in generator.stream(messages, params):
                        if piece:
                            yield emit({"content": piece})
            except Exception as exc:  # noqa: BLE001
                yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error'}}, ensure_ascii=False)}\n\n"
            yield emit({}, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(app, port: int, *, host: str = "127.0.0.1", timeout: float = 60.0):
    import uvicorn  # noqa: PLC0415

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + timeout
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    if not server.started:
        raise RuntimeError(f"服务启动失败（{host}:{port}）")
    return server, thread


def stop_server(server, thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


def request_json(url: str, payload: dict[str, Any] | None = None, *, timeout: float = 120.0) -> tuple[int, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 本地地址
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # type: ignore[name-defined]
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body}


def request_sse(url: str, payload: dict[str, Any], *, timeout: float = 180.0) -> tuple[int, list[str]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    events: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 本地地址
        status = response.status
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line:
                events.append(line)
    return status, events
