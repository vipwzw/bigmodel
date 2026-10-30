"""Q&A 评测脚本：
- Qwen3-0.6B（Stage 5 微调后）：单轮问答、多轮对话、CLI 调用、OpenAI 兼容服务
- 从零训练的 tiny S 模型：微调前 / 指令微调后 的问答对比
输出 JSON + Markdown 记录。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS = [
    "你好，你是谁？",
    "1+1等于几？2+3等于几？",
    "中国的首都是哪座城市？",
    "什么是机器学习？",
    "用一句话解释什么是大语言模型。",
    "给初学者三条学习深度学习的建议。",
    "请用中文写一句关于秋天的诗句。",
    "解释一下 LoRA 微调的原理。",
    "如果一个模型回答总是很啰嗦，你会怎么改进？",
    "太阳系中体积最大的行星是哪个？",
]

MULTI_TURN = [
    "你好，我想学做大模型训练，应该从哪里开始？",
    "我只有一台 Mac，没有 NVIDIA 显卡，还能练吗？",
    "那第一步具体该做什么？",
]


def ensure_mlx_4bit(merged_dir: str, out_dir: str) -> str:
    out = Path(out_dir)
    if not (out / "config.json").exists():
        from mlx_lm import convert  # noqa: PLC0415

        out.parent.mkdir(parents=True, exist_ok=True)
        convert(merged_dir, str(out), quantize=True, q_bits=4)
    return str(out)


def run_qwen(model_dir: str) -> dict:
    from bigmodel_lab.serve.generate import (  # noqa: PLC0415
        DEFAULT_SYSTEM_PROMPT,
        ChatSession,
        MLXGenerator,
        SamplingParams,
    )
    from bigmodel_lab.serve.openai_server import (  # noqa: PLC0415
        create_app,
        free_port,
        request_json,
        request_sse,
        start_server,
        stop_server,
    )

    started = time.time()
    generator = MLXGenerator(model_dir)
    single = []
    for question in QUESTIONS:
        result = generator.generate(
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": question}],
            SamplingParams(max_tokens=100, temperature=0.3, top_p=0.9),
        )
        single.append({"question": question, "answer": result["text"], "tokens_per_second": result["tokens_per_second"]})
    print(f"[qwen] 单轮问答完成 {len(single)} 条，{time.time()-started:.0f}s", flush=True)

    # 多轮对话
    session = ChatSession(generator, params=SamplingParams(max_tokens=100, temperature=0.3))
    multi = []
    for question in MULTI_TURN:
        answer = session.send(question)
        multi.append({"question": question, "answer": answer})
    print(f"[qwen] 多轮对话完成 {len(multi)} 轮", flush=True)

    # CLI 调用
    cli_command = [
        sys.executable, str(PROJECT_ROOT / ".venv/bin/lab"), "generate",
        "--model-path", model_dir, "--backend", "mlx",
        "--prompt", "用一句话解释什么是大语言模型。", "--max-new-tokens", "80",
    ]
    cli = subprocess.run(cli_command, capture_output=True, text=True, timeout=300)
    cli_result = {"command": " ".join(cli_command), "exit_code": cli.returncode, "stdout": cli.stdout.strip()[:500]}

    # OpenAI 兼容服务（真实 HTTP + 两轮历史）
    app = create_app(generator, model_name="qwen3-0.6b-qa")
    port = free_port()
    server, thread = start_server(app, port)
    base = f"http://127.0.0.1:{port}"
    try:
        history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        server_turns = []
        for question in MULTI_TURN[:2]:
            history.append({"role": "user", "content": question})
            status, payload = request_json(f"{base}/v1/chat/completions", {"messages": history, "max_tokens": 100, "temperature": 0.3})
            content = payload["choices"][0]["message"]["content"] if status == 200 else str(payload)[:200]
            history.append({"role": "assistant", "content": content})
            server_turns.append({"status": status, "question": question, "answer": content})
        status, events = request_sse(f"{base}/v1/chat/completions", {"messages": [{"role": "user", "content": "用一句话介绍你自己。"}], "stream": True, "max_tokens": 40})
        server_result = {"turns": server_turns, "stream_status": status, "stream_has_done": any("[DONE]" in e for e in events)}
    finally:
        stop_server(server, thread)

    return {"model": model_dir, "single": single, "multi": multi, "cli": cli_result, "server": server_result,
            "elapsed_seconds": round(time.time() - started, 1)}


def run_tiny(base_checkpoint: str | None, sft_checkpoint: str | None, out: dict) -> dict:
    from bigmodel_lab.train.precision import default_device  # noqa: PLC0415
    from scripts.qa_experiment import QWEN_QUESTIONS, _tokenizer, load_tiny_model, load_tiny_sft, tiny_ask  # noqa: PLC0415

    device = default_device()
    tokenizer = _tokenizer()
    result: dict = {}
    if base_checkpoint:
        model, meta = load_tiny_model(Path(base_checkpoint), device=device)
        result["pretrained"] = {"meta": meta, "results": [{"question": q, "answer": tiny_ask(model, tokenizer, q, device=device)} for q in QWEN_QUESTIONS]}
    if sft_checkpoint and Path(sft_checkpoint).exists():
        model = load_tiny_sft(Path(sft_checkpoint), device=device)
        result["sft"] = {"checkpoint": sft_checkpoint, "results": [{"question": q, "answer": tiny_ask(model, tokenizer, q, device=device)} for q in QWEN_QUESTIONS]}
    return result


def to_markdown(data: dict) -> str:
    lines = ["# 向训练好的语言模型提问：实测记录", ""]
    lines.append("> 本文件由 `scripts/qa_eval.py` 自动生成。三轮对比：从零 tiny 模型（仅预训练）、tiny 模型 + 指令微调、Qwen3-0.6B + 指令微调。")
    lines.append("")
    if "qwen" in data:
        q = data["qwen"]
        lines += ["## 一、Qwen3-0.6B（指令微调后）单轮问答", "", "| # | 问题 | 回答 |", "|---|---|---|"]
        for i, item in enumerate(q["single"], 1):
            lines.append(f"| {i} | {item['question']} | {item['answer'].replace(chr(10), ' ')[:400]} |")
        lines.append("")
        lines += ["## 二、多轮对话（同一会话）", ""]
        for item in q["multi"]:
            lines.append(f"**用户**：{item['question']}")
            lines.append("")
            lines.append(f"**模型**：{item['answer']}")
            lines.append("")
        lines += ["## 三、CLI 调用", "", "```bash", q["cli"]["command"], "```", "", f"退出码 {q['cli']['exit_code']}：", "", "```", q["cli"]["stdout"], "```", ""]
        lines += ["## 四、OpenAI 兼容服务（真实 HTTP）", ""]
        for turn in q["server"]["turns"]:
            lines.append(f"- HTTP {turn['status']}｜用户：{turn['question']}")
            lines.append(f"  - 模型：{turn['answer'][:300]}")
        lines.append(f"- 流式接口：HTTP {q['server']['stream_status']}，收到 `[DONE]`：{q['server']['stream_has_done']}")
        lines.append("")
    for key, title in (("tiny_pretrained", "五、从零训练 tiny S 模型（仅预训练）被提问"), ("tiny_sft", "六、从零 tiny S 模型 + 指令微调后被提问")):
        if key in data:
            lines += [f"## {title}", ""]
            for item in data[key]["results"]:
                lines.append(f"- **问**：{item['question']}")
                lines.append(f"  - **答**：{item['answer'][:300] if item['answer'] else '（空）'}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-model-dir", default="models/exports/mlx-qwen3-0.6b-qa-4bit")
    parser.add_argument("--merged-dir", default="models/exports/qwen3-0.6b-sft-lora-merged")
    parser.add_argument("--tiny-checkpoint", default=None)
    parser.add_argument("--tiny-sft", default="models/exports/tiny-s-qa-sft.pt")
    parser.add_argument("--skip-tiny-sft", action="store_true")
    parser.add_argument("--out-json", default="docs/QA_EXPERIMENT.json")
    parser.add_argument("--out-md", default="docs/QA_EXPERIMENT.md")
    args = parser.parse_args()

    model_dir = ensure_mlx_4bit(args.merged_dir, args.qwen_model_dir)
    data: dict = {"qwen": run_qwen(model_dir), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tiny = run_tiny(args.tiny_checkpoint, None if args.skip_tiny_sft else args.tiny_sft, data)
    if "pretrained" in tiny:
        data["tiny_pretrained"] = tiny["pretrained"]
    if "sft" in tiny:
        data["tiny_sft"] = tiny["sft"]
    Path(args.out_json).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(to_markdown(data), encoding="utf-8")
    print(f"写出 {args.out_json} / {args.out_md}")


if __name__ == "__main__":
    main()
