"""CUDA 单卡「测试数据」：用固定问题集逐条提问训练后的模型，输出 JSON + Markdown 记录。

用途：训练完成（Stage 5 合并导出）后，检查模型能否像 ChatGPT 一样回答中文问题。
问题集固定，便于 V100 / A10 两次运行横向对比；回答全部为真实生成结果。

用法：
    python scripts/qa_cuda.py --model-path models/exports/qwen3-0.6b-v100-sft-merged
    python scripts/qa_cuda.py --model-path models/exports/qwen3-0.6b-a10-sft-merged \\
        --max-new-tokens 100 --out-dir runs/qa-cuda/a10

产出：
    runs/qa-cuda/<时间戳>/qa_report.json   结构化结果（问题/回答/吞吐/耗时）
    runs/qa-cuda/<时间戳>/qa_report.md     可直接阅读的问答记录
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 测试数据：10 条单轮问题（事实/算术/概念/指令/创作，覆盖不同能力）
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

# 多轮对话：考察上下文跟随能力
MULTI_TURN = [
    "你好，我想学做大模型训练，应该从哪里开始？",
    "我只有一张 V100 或 A10 显卡，还能练吗？",
    "那第一步具体该做什么？",
]


def _markdown(payload: dict) -> str:
    lines = [
        "# CUDA 单卡测试数据：样例问答记录",
        "",
        f"- 模型：`{payload['model']}`",
        f"- 设备：{payload['device']}（{payload['device_name']}）",
        f"- 生成时间：{payload['generated_at']}",
        f"- 采样参数：max_new_tokens={payload['params']['max_tokens']}，"
        f"temperature={payload['params']['temperature']}，top_p={payload['params']['top_p']}",
        "",
        "## 一、单轮问答（10 条固定问题）",
        "",
        "| # | 问题 | 回答 | tokens/s |",
        "|---|---|---|---|",
    ]
    for index, item in enumerate(payload["questions"], start=1):
        answer = item["answer"].replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {index} | {item['question']} | {answer} | {item['tokens_per_second']} |")
    lines.extend(["", "## 二、多轮对话（3 轮，同一会话）", ""])
    for index, item in enumerate(payload["multi_turn"], start=1):
        lines.append(f"**第 {index} 轮**")
        lines.append("")
        lines.append(f"- 用户：{item['question']}")
        lines.append(f"- 模型：{item['answer']}")
        lines.append("")
    lines.extend(
        [
            "## 三、汇总",
            "",
            f"- 单轮生成模式：{payload['summary'].get('mode', 'sequential')}",
            f"- 单轮平均吞吐：{payload['summary']['avg_tokens_per_second']} tokens/s",
            f"- 总耗时：{payload['summary']['total_seconds']} s",
            f"- 单轮空回答数：{payload['summary']['empty_answers']}",
            "",
            "> 说明：回答质量受训练步数、数据规模与模型大小限制；本记录仅用于人工检查，不做自动评分。",
        ]
    )
    return "\n".join(lines)


def _batched_answers(generator, questions: list[str], params, system_prompt: str) -> tuple[list[str], list[int], float]:
    """批量生成所有单轮问题的回答。

    合并成一次 generate 可以摊薄逐 token 的 CPU 调度开销——在共享 vCPU 的云 GPU 环境里，
    逐条生成往往受 CPU dispatch 限制（GPU 利用率很低），批量后聚合吞吐明显更高。
    """
    import torch  # noqa: PLC0415

    tokenizer = generator.tokenizer
    model = generator.model
    prompts = []
    for question in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            )
        except Exception:  # noqa: BLE001 - 未适配 thinking 开关的模板
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(generator.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        started = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=params.max_tokens,
                do_sample=params.temperature > 0,
                temperature=max(params.temperature, 1e-5),
                top_p=params.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - started
        generated = output[:, prompt_len:]
        answers = [tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated]
        token_counts = [int((row != tokenizer.pad_token_id).sum().item()) for row in generated]
    finally:
        tokenizer.padding_side = padding_side
    return answers, token_counts, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="CUDA 单卡样例问答测试（测试数据）")
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "models" / "exports" / "qwen3-0.6b-v100-sft-merged"),
        help="合并导出后的模型目录（lab run 5 的 export.dir）",
    )
    parser.add_argument("--device", default=None, help="设备，默认自动（cuda → mps → cpu）")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--out-dir", default=None, help="输出目录，默认 runs/qa-cuda/<时间戳>")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not (model_path / "config.json").exists():
        print(
            f"模型目录不存在或未合并导出: {model_path}\n"
            "请先运行：.venv/bin/lab run 5 --config configs/cuda.yaml --config configs/cuda-v100.yaml --profile standard",
            file=sys.stderr,
        )
        return 2

    from bigmodel_lab.serve.generate import (  # noqa: PLC0415
        DEFAULT_SYSTEM_PROMPT,
        ChatSession,
        SamplingParams,
        TransformersGenerator,
    )
    from bigmodel_lab.train.precision import device_report  # noqa: PLC0415

    started = time.time()
    generator = TransformersGenerator(model_path, device=args.device)
    params = SamplingParams(max_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    report = device_report(generator.device)
    print(f"[qa] 模型 {model_path} / 设备 {generator.device} ({report.get('name')})")

    single: list[dict] = []
    mode = "batched"
    batch_elapsed = 0.0
    batch_tokens = 0
    try:
        answers, token_counts, batch_elapsed = _batched_answers(
            generator, QUESTIONS, params, DEFAULT_SYSTEM_PROMPT
        )
        batch_tokens = sum(token_counts)
        aggregate_tps = round(batch_tokens / max(batch_elapsed, 1e-9), 1)
        for index, (question, answer, tokens) in enumerate(
            zip(QUESTIONS, answers, token_counts, strict=True), start=1
        ):
            single.append(
                {
                    "question": question,
                    "answer": answer,
                    "tokens": tokens,
                    "tokens_per_second": aggregate_tps,
                    "elapsed_seconds": round(batch_elapsed, 2),
                }
            )
            print(f"[qa] {index:02d}/{len(QUESTIONS)} 批量生成完成（聚合 {aggregate_tps} tokens/s）")
    except Exception as exc:  # noqa: BLE001 - 批量失败时退回逐条，保证测试数据仍可产出
        print(f"[qa] 批量生成失败，回退逐条：{type(exc).__name__}: {exc}")
        mode = "sequential"
        for index, question in enumerate(QUESTIONS, start=1):
            result = generator.generate(
                [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": question}],
                params,
            )
            single.append(
                {
                    "question": question,
                    "answer": result["text"],
                    "tokens_per_second": result["tokens_per_second"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
            print(f"[qa] {index:02d}/{len(QUESTIONS)} 完成（{result['tokens_per_second']} tokens/s）")

    session = ChatSession(generator, params=params)
    multi = [{"question": question, "answer": session.send(question)} for question in MULTI_TURN]

    total_seconds = round(time.time() - started, 2)
    avg_tps = round(sum(item["tokens_per_second"] for item in single) / max(len(single), 1), 1)
    payload = {
        "model": str(model_path),
        "device": generator.device,
        "device_name": report.get("name"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"max_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p},
        "questions": single,
        "multi_turn": multi,
        "summary": {
            "mode": mode,
            "avg_tokens_per_second": avg_tps,
            "total_seconds": total_seconds,
            "empty_answers": sum(1 for item in single if not item["answer"].strip()),
        },
    }

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "runs" / "qa-cuda" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qa_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "qa_report.md").write_text(_markdown(payload), encoding="utf-8")
    print(f"[qa] 完成：{out_dir / 'qa_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
