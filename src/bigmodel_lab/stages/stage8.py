"""Stage 8：MLX 量化、推理基准、CLI 生成接口与 OpenAI 兼容服务契约。"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..budget import Budget
from ..config import PROJECT_ROOT, LabConfig
from ..manifest import RunContext
from ..serve.generate import DEFAULT_SYSTEM_PROMPT, MLXGenerator, SamplingParams, TransformersGenerator
from ..serve.openai_server import create_app, free_port, request_json, request_sse, start_server, stop_server
from ..train.precision import default_device, driver_memory_gb, empty_cache

logger = logging.getLogger(__name__)


def _stage_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("stage8") or {}


def _dir_size_gb(path: Path) -> float:
    total = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return round(total / 1024**3, 3)


def _resolve_model(stage_config: dict[str, Any]) -> Path:
    primary = PROJECT_ROOT / str(stage_config.get("model_dir", "models/exports/qwen3-0.6b-sft-lora-merged"))
    if primary.exists():
        return primary
    fallback = PROJECT_ROOT / str(stage_config.get("fallback_model_dir", "models/Qwen3-0.6B"))
    if fallback.exists():
        logger.warning("未找到 SFT 合并模型，回退到基座模型: %s", fallback)
        return fallback
    raise RuntimeError("未找到可用的推理模型：请先运行 lab run 5 或下载 Qwen3-0.6B")


def _messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def _quantize(model_dir: Path, *, bits: int) -> tuple[Path, dict[str, Any]]:
    from mlx_lm import convert  # noqa: PLC0415

    output_dir = PROJECT_ROOT / "models" / "exports" / f"mlx-{model_dir.name}-{bits}bit"
    if not (output_dir / "config.json").exists():
        logger.info("MLX 量化 %dbit: %s → %s", bits, model_dir.name, output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        convert(str(model_dir), str(output_dir), quantize=True, q_bits=bits)
    else:
        logger.info("复用已存在的量化模型: %s", output_dir)
    size_gb = _dir_size_gb(output_dir)
    generator = MLXGenerator(output_dir)
    sample = generator.generate(_messages("用一句话介绍你自己。"), SamplingParams(max_tokens=24, temperature=0.7))
    del generator
    report = {
        "bits": bits,
        "path": str(output_dir),
        "size_gb": size_gb,
        "size_ratio_vs_source": round(size_gb / max(_dir_size_gb(model_dir), 1e-9), 3),
        "loadable": True,
        "sample": sample["text"][:200],
        "tokens_per_second": sample["tokens_per_second"],
    }
    return output_dir, report


def _benchmark_transformers(model_dir: Path, *, prompt: str, max_tokens: int, device: str) -> dict[str, Any]:
    generator = TransformersGenerator(model_dir, device=device)
    result = generator.generate(_messages(prompt), SamplingParams(max_tokens=max_tokens, temperature=0.7))
    result["peak_memory_gb"] = driver_memory_gb(device)
    result["model"] = str(model_dir)
    del generator
    empty_cache(device)
    return result


def _benchmark_mlx(model_path: Path, *, name: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    import mlx.core as mx  # noqa: PLC0415

    mx.metal.reset_peak_memory()
    generator = MLXGenerator(model_path)
    result = generator.generate(_messages(prompt), SamplingParams(max_tokens=max_tokens, temperature=0.7))
    result["peak_memory_gb"] = round(mx.metal.get_peak_memory() / 1e9, 3)
    result["model"] = str(model_path)
    result["name"] = name
    del generator
    mx.metal.clear_cache()
    return result


def _server_contract(model_path: Path, *, host: str, model_name: str) -> dict[str, Any]:
    generator = MLXGenerator(model_path)
    app = create_app(generator, model_name=model_name)
    port = free_port()
    server, thread = start_server(app, port, host=host)
    base = f"http://{host}:{port}"
    contract: dict[str, Any] = {"port": port, "checks": {}}
    try:
        status, payload = request_json(f"{base}/v1/models")
        contract["checks"]["models"] = {"status": status, "ok": status == 200 and payload.get("object") == "list", "response": payload}
        status, payload = request_json(
            f"{base}/v1/chat/completions",
            {"model": model_name, "messages": [{"role": "user", "content": "用一句话介绍你自己。"}], "max_tokens": 24},
        )
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(payload, dict) else ""
        contract["checks"]["chat_completion"] = {
            "status": status,
            "ok": status == 200 and bool(content) and payload.get("object") == "chat.completion",
            "content": content[:200],
        }
        status, events = request_sse(
            f"{base}/v1/chat/completions",
            {"model": model_name, "messages": [{"role": "user", "content": "你好"}], "max_tokens": 12, "stream": True},
        )
        contract["checks"]["streaming"] = {
            "status": status,
            "ok": status == 200 and any("[DONE]" in event for event in events) and any("chat.completion.chunk" in event for event in events),
            "events": len(events),
        }
        status, payload = request_json(f"{base}/v1/chat/completions", {"messages": []})
        contract["checks"]["invalid_request"] = {
            "status": status,
            "ok": status == 400 and isinstance(payload, dict) and payload.get("error", {}).get("type") == "invalid_request_error",
            "response": payload,
        }
    finally:
        stop_server(server, thread)
    contract["passed"] = all(check.get("ok") for check in contract["checks"].values())
    return contract


def _cli_demo(model_path: Path, *, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / ".venv" / "bin" / "lab"),
        "generate",
        "--model-path", str(model_path),
        "--backend", "mlx",
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--stream",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    output = (completed.stdout or "").strip()
    return {"command": " ".join(command), "exit_code": completed.returncode, "output": output[:600], "ok": completed.returncode == 0 and bool(output)}


def run(*, ctx: RunContext, config: dict[str, Any], budget: Budget) -> dict[str, Any]:
    lab_config = LabConfig.model_validate(config)
    stage_config = _stage_config(config)
    device = default_device()
    model_dir = _resolve_model(stage_config)
    logger.info("推理阶段模型: %s", model_dir)

    quant_report: dict[str, Any] = {}
    quant_paths: dict[int, Path] = {}
    for bits in [int(value) for value in stage_config.get("quant_bits", [4, 8])]:
        budget.check()
        path, report = _quantize(model_dir, bits=bits)
        quant_paths[bits] = path
        quant_report[f"{bits}bit"] = report
    ctx.write_json_artifact("quant_report.json", quant_report)

    benchmark_cfg = stage_config.get("benchmark") or {}
    prompt = str(benchmark_cfg.get("prompt", "用一句话解释什么是人工智能。"))
    max_tokens = int(benchmark_cfg.get("max_tokens", 48))
    logger.info("推理基准：transformers(MPS) vs MLX 量化")
    bench_report: dict[str, Any] = {}
    bench_report["transformers_mps_bf16"] = _benchmark_transformers(model_dir, prompt=prompt, max_tokens=max_tokens, device=device)
    bench_report["mlx_bf16"] = _benchmark_mlx(model_dir, name="mlx_bf16", prompt=prompt, max_tokens=max_tokens)
    for bits, path in quant_paths.items():
        bench_report[f"mlx_{bits}bit"] = _benchmark_mlx(path, name=f"mlx_{bits}bit", prompt=prompt, max_tokens=max_tokens)
    ctx.write_json_artifact("bench_report.json", bench_report)

    q_bits = int(stage_config.get("mlx_q_bits", 4))
    server_path = quant_paths.get(q_bits, model_dir)
    server_cfg = stage_config.get("server") or {}
    logger.info("OpenAI 兼容服务契约测试（%s）", server_path)
    contract = _server_contract(server_path, host=str(server_cfg.get("host", "127.0.0.1")), model_name=str(server_cfg.get("model_name", "bigmodel-lab")))
    ctx.write_json_artifact("server_contract.json", contract)

    cli_cfg = stage_config.get("cli") or {}
    logger.info("CLI 生成接口验证")
    cli_demo = _cli_demo(server_path, prompt=str(cli_cfg.get("prompt", "用一句话介绍你自己。")), max_new_tokens=int(cli_cfg.get("max_new_tokens", 16)))
    ctx.write_artifact("cli_demo.txt", json.dumps(cli_demo, ensure_ascii=False, indent=2))

    lines = ["# 推理与服务报告", "", "## 量化结果", "", "| 配置 | 体积(GB) | 相对源模型 | 采样示例 |", "|---|---|---|---|"]
    for name, report in quant_report.items():
        lines.append(f"| {name} | {report['size_gb']} | {report['size_ratio_vs_source']} | {report['sample'][:60]} |")
    lines.append("")
    lines.append("## 推理基准")
    lines.append("")
    lines.append("| 配置 | 后端 | tokens/s | TTFT(s) | 峰值显存(GB) | 输出摘要 |")
    lines.append("|---|---|---|---|---|---|")
    for name, result in bench_report.items():
        lines.append(
            f"| {name} | {result.get('backend')} | {result.get('tokens_per_second')} | {result.get('ttft_seconds')} | "
            f"{result.get('peak_memory_gb')} | {str(result.get('text', ''))[:40]} |"
        )
    lines.append("")
    lines.append("## OpenAI 兼容服务契约")
    lines.append("")
    for name, check in contract.get("checks", {}).items():
        lines.append(f"- {name}: {'✅' if check.get('ok') else '❌'}（HTTP {check.get('status')}）")
    lines.append("")
    lines.append(f"- 契约测试结论：{'通过' if contract.get('passed') else '未通过'}")
    lines.append("")
    lines.append("## CLI 生成接口")
    lines.append("")
    lines.append(f"- 命令：`{cli_demo['command']}`")
    lines.append(f"- 退出码：{cli_demo['exit_code']}；输出：{cli_demo['output'][:120]}")
    lines.append("")
    lines.append("## 与生产推理栈对照")
    lines.append("")
    lines.append("| 本机 | 生产栈（vLLM / SGLang / TGI） | 差异 |")
    lines.append("|---|---|---|")
    lines.append("| MLX 4bit 单请求 | PagedAttention + 连续批处理 + 张量并行 | 本机无并发调度与多卡 |")
    lines.append("| FastAPI 单进程 | 多副本 + 负载均衡 + 指标/限流 | 本机仅验证接口契约 |")
    lines.append("| 量化（MLX affine 4bit/8bit） | GPTQ/AWQ/FP8 + 内核融合 | 量化思想一致，内核实现不同 |")
    lines.append("| 无投机解码 | 投机解码/前缀缓存 | 仅文档说明，未在本机实现 |")
    lines.append("")
    ctx.write_artifact("serve_report.md", "\n".join(lines))

    summary = {
        "profile": budget.profile,
        "offline": bool(lab_config.offline),
        "model": str(model_dir),
        "quant": {name: {"size_gb": report["size_gb"], "tokens_per_second": report["tokens_per_second"]} for name, report in quant_report.items()},
        "benchmark": {name: {"tokens_per_second": result.get("tokens_per_second"), "ttft_seconds": result.get("ttft_seconds"), "peak_memory_gb": result.get("peak_memory_gb")} for name, result in bench_report.items()},
        "server_contract": {"passed": contract.get("passed"), "checks": {name: check.get("ok") for name, check in contract.get("checks", {}).items()}},
        "cli": {"ok": cli_demo["ok"], "exit_code": cli_demo["exit_code"]},
    }
    ctx.write_json_artifact("stage_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary
