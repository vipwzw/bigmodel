"""Q&A 实验：
1) 从零训练的 tiny S 模型：直接提问（预期只是续写）
2) 对 tiny S 做一次简单的指令微调，再提问（看是否能学会"问答格式"）
3) 用 Stage 5 微调后的 Qwen3-0.6B：真实问答 + 多轮
产出一份 markdown 记录到 docs/（由调用方决定输出路径）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from bigmodel_lab.data.tokenizer import load_tokenizer
from bigmodel_lab.models.gpt import GPT, PRESETS
from bigmodel_lab.train.precision import autocast_context, default_device, detect_autocast_dtype, sync

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IGNORE_INDEX = -100


def _tokenizer():
    return load_tokenizer(PROJECT_ROOT / "data/processed/stage3/tokenizers/bpe_16384/tokenizer.json")


def _find_latest_checkpoint(preset: str = "s") -> Path:
    runs = sorted((PROJECT_ROOT / "runs/stage-4").glob("*/checkpoints/s/ckpt_step*.pt"))
    if not runs:
        runs = sorted((PROJECT_ROOT / "runs/stage-4").glob("*/checkpoints/s/best.pt"))
    if not runs:
        raise FileNotFoundError("未找到 Stage 4 S 档检查点")
    return runs[-1]


def load_tiny_model(checkpoint: Path | None = None, device: str | None = None) -> tuple[GPT, dict[str, Any]]:
    device = device or default_device()
    checkpoint = checkpoint or _find_latest_checkpoint()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    preset = state.get("config", {}).get("preset", "s")
    model = GPT(PRESETS[preset])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, {"preset": preset, "step": state.get("step"), "tokens_seen": state.get("tokens_seen"), "checkpoint": str(checkpoint)}


def tiny_ask(model: GPT, tokenizer, question: str, *, device: str, max_new_tokens: int = 80) -> str:
    prompt = f"问题：{question}\n回答："
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if len(ids) > 400:
        ids = ids[-400:]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    generated = output[0][len(ids):].tolist()
    return tokenizer.decode(generated).strip()


# ---------- tiny SFT ----------
def build_tiny_sft_examples(tokenizer, records: list[dict[str, Any]], max_seq_len: int = 512, limit: int = 4000):
    examples = []
    for record in records[:limit]:
        messages = record.get("messages") or []
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
        if not user or not assistant:
            continue
        prompt = f"问题：{user}\n回答："
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        answer_ids = tokenizer.encode(assistant, add_special_tokens=False).ids
        eos_id = tokenizer.token_to_id("<eos>")
        ids = prompt_ids + answer_ids + ([eos_id] if eos_id is not None else [])
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        mask = [0] * len(prompt_ids) + [1] * (len(ids) - len(prompt_ids))
        examples.append((ids, mask))
    return examples


def train_tiny_sft(*, output: Path, steps: int = 600, batch_size: int = 8, seq_len: int = 512,
                   lr: float = 3e-4, limit: int = 4000, device: str | None = None, seed: int = 42,
                   checkpoint: Path | None = None) -> dict[str, Any]:
    device = device or default_device()
    torch.manual_seed(seed)
    tokenizer = _tokenizer()
    model, meta = load_tiny_model(checkpoint, device=device)
    model.train()
    records = [
        json.loads(line)
        for line in (PROJECT_ROOT / "data/processed/stage3/instruction/sft_zh/train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples = build_tiny_sft_examples(tokenizer, records, max_seq_len=seq_len, limit=limit)
    print(f"tiny SFT: {len(examples)} 条样本，从 {meta['checkpoint']} 开始", flush=True)
    pad_id = tokenizer.token_to_id("<pad>") or 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    autocast_dtype, autocast_name = detect_autocast_dtype(device)
    started = time.time()
    rng = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    for step in range(steps):
        indices = torch.randperm(len(examples), generator=rng)[: batch_size].tolist()
        batch = [examples[i] for i in indices]
        max_len = max(len(ids) for ids, _ in batch) - 1
        inputs = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        targets = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        masks = torch.zeros((batch_size, max_len), dtype=torch.float32)
        for row, (ids, mask) in enumerate(batch):
            length = min(len(ids) - 1, max_len)
            inputs[row, :length] = torch.tensor(ids[:-1][:length])
            targets[row, :length] = torch.tensor(ids[1 : 1 + length])
            masks[row, :length] = torch.tensor(mask[1:][:length], dtype=torch.float32)
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, autocast_dtype):
            loss, _ = model(inputs, targets, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync(device)
        losses.append(float(loss.item()))
        if step % 100 == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps} loss {loss.item():.4f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": {"preset": meta["preset"]}, "sft_steps": steps}, output)
    return {"output": str(output), "steps": steps, "samples": len(examples),
            "final_loss": round(sum(losses[-50:]) / len(losses[-50:]), 4),
            "wall_seconds": round(time.time() - started, 1), "device": device, "autocast": autocast_name}


def load_tiny_sft(path: Path, device: str | None = None) -> GPT:
    device = device or default_device()
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(PRESETS[state["config"]["preset"]])
    model.load_state_dict(state["model"])
    return model.to(device).eval()


# ---------- Qwen3 问答 ----------
def qwen_ask(questions: list[str], *, model_dir: str, max_new_tokens: int = 96, temperature: float = 0.3) -> list[dict[str, Any]]:
    from bigmodel_lab.serve.generate import DEFAULT_SYSTEM_PROMPT, MLXGenerator, SamplingParams  # noqa: PLC0415

    generator = MLXGenerator(model_dir)
    results = []
    for question in questions:
        answer = generator.generate(
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": question}],
            SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=0.9),
        )
        results.append({"question": question, **answer})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tiny-ask", "tiny-sft", "qwen-ask", "prepare-data"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-dir", default="models/exports/mlx-qwen3-0.6b-sft-lora-merged-4bit")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--questions-file", default=None)
    args = parser.parse_args()

    if args.mode == "qwen-ask":
        questions = json.loads(Path(args.questions_file).read_text(encoding="utf-8")) if args.questions_file else QWEN_QUESTIONS
        print(json.dumps(qwen_ask(questions, model_dir=args.model_dir), ensure_ascii=False, indent=2))
        return
    if args.mode == "tiny-sft":
        output = Path(args.output or "models/exports/tiny-s-qa-sft.pt")
        print(json.dumps(train_tiny_sft(output=output, steps=args.steps, checkpoint=Path(args.checkpoint) if args.checkpoint else None), ensure_ascii=False, indent=2))
        return
    if args.mode == "tiny-ask":
        device = default_device()
        tokenizer = _tokenizer()
        model, meta = load_tiny_model(Path(args.checkpoint) if args.checkpoint else None, device=device)
        questions = json.loads(Path(args.questions_file).read_text(encoding="utf-8")) if args.questions_file else QWEN_QUESTIONS
        results = [{"question": q, "answer": tiny_ask(model, tokenizer, q, device=device)} for q in questions]
        print(json.dumps({"checkpoint": meta, "results": results}, ensure_ascii=False, indent=2))
        return
    raise SystemExit("prepare-data 未实现")


QWEN_QUESTIONS = [
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

if __name__ == "__main__":
    main()
