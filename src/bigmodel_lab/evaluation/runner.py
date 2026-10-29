"""评测实现：困惑度、选择题、生成规则打分、实验追踪与跨实验对比。"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.packing import load_packed
from ..models.gpt import GPT, PRESETS
from ..train.precision import (
    autocast_context,
    detect_autocast_dtype,
)

logger = logging.getLogger(__name__)


# ---------- 模型加载 ----------
def load_tiny_checkpoint(checkpoint_path: Path, *, device: str) -> tuple[GPT, dict[str, Any]]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_payload = state.get("config", {})
    preset = config_payload.get("preset", "s")
    model_config = PRESETS[preset]
    model = GPT(model_config)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, {"preset": preset, "params": model_config.num_params(), "flops_per_token": model_config.flops_per_token(), "step": state.get("step"), "tokens_seen": state.get("tokens_seen")}


# ---------- 困惑度与数据重叠 ----------
def _row_hashes(tokens: np.ndarray) -> set[bytes]:
    return {row.tobytes() for row in tokens}


def perplexity_eval(
    model: GPT,
    *,
    packed_dir: Path,
    split: str,
    seq_len: int,
    device: str,
    max_batches: int = 32,
) -> dict[str, Any]:
    val = load_packed(packed_dir / f"{split}.npz")
    train = load_packed(packed_dir / "train.npz")
    tokens = val["tokens"].astype(np.int64)
    loss_mask = val["loss_mask"].astype(np.float32)
    chunk_tokens, chunk_masks = [], []
    for row in range(tokens.shape[0]):
        for start in range(0, tokens.shape[1] - seq_len + 1, seq_len):
            chunk_tokens.append(tokens[row, start : start + seq_len])
            chunk_masks.append(loss_mask[row, start : start + seq_len])
            if len(chunk_tokens) >= max_batches * 4:
                break
        if len(chunk_tokens) >= max_batches * 4:
            break
    autocast_dtype, _ = detect_autocast_dtype(device) if device == "mps" else (None, "float32")
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for index in range(0, min(len(chunk_tokens), max_batches * 4), 4):
            batch_tokens = np.stack(chunk_tokens[index : index + 4])
            batch_masks = np.stack(chunk_masks[index : index + 4])
            inputs = torch.from_numpy(batch_tokens[:, :-1]).to(device)
            targets = torch.from_numpy(batch_tokens[:, 1:]).to(device)
            masks = torch.from_numpy(batch_masks[:, 1:]).to(device)
            with autocast_context(device, autocast_dtype):
                loss, _ = model(inputs, targets, masks)
            count = float(masks.sum().item())
            total_loss += float(loss.item()) * count
            total_tokens += int(count)
    mean_loss = total_loss / max(total_tokens, 1)
    overlap = len(_row_hashes(val["tokens"]) & _row_hashes(train["tokens"])) / max(val["tokens"].shape[0], 1)
    return {
        "split": split,
        "sequences": int(val["tokens"].shape[0]),
        "tokens_evaluated": total_tokens,
        "mean_loss": round(mean_loss, 6),
        "perplexity": round(math.exp(mean_loss), 4),
        "train_val_row_overlap_ratio": round(overlap, 6),
    }


# ---------- 选择题 ----------
@torch.no_grad()
def continuation_logprob(model, tokenizer, prompt: str, continuation: str, *, device: str, max_length: int = 512) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    cont_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    if not cont_ids:
        return float("-inf")
    if len(prompt_ids) + len(cont_ids) > max_length:
        prompt_ids = prompt_ids[-(max_length - len(cont_ids)) :]
    input_ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long, device=device)
    outputs = model(input_ids)
    logits = outputs[1] if isinstance(outputs, tuple) else outputs.logits
    logits = logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = len(prompt_ids) - 1
    return float(token_logprobs[0, start:].mean().item())


def multiple_choice_eval(model, tokenizer, items: list[dict[str, Any]], *, device: str) -> dict[str, Any]:
    letters = "ABCD"
    predictions: list[dict[str, Any]] = []
    correct = 0
    for item in items:
        options = item["options"]
        prompt_lines = [f"问题：{item['question']}", "选项："]
        for index, option in enumerate(options):
            prompt_lines.append(f"{letters[index]}. {option}")
        prompt_lines.append("答案：")
        prompt = "\n".join(prompt_lines)
        scores = []
        for index in range(len(options)):
            scores.append(continuation_logprob(model, tokenizer, prompt, f" {letters[index]}", device=device))
        predicted = int(np.argmax(scores))
        hit = predicted == int(item["answer"])
        correct += int(hit)
        predictions.append({"question": item["question"], "predicted": predicted, "answer": item["answer"], "correct": hit, "scores": [round(value, 4) for value in scores]})
    total = len(items)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "random_baseline": round(1 / max(len(items[0]["options"]), 1), 4) if items else 0.0,
        "predictions": predictions,
    }


# ---------- 生成规则评测 ----------
def _char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def score_by_rule(response: str, rule: dict[str, Any]) -> bool:
    rule_type = rule.get("type")
    text = response.strip()
    if rule_type == "exact":
        return text == rule.get("value")
    if rule_type == "length_chars":
        return int(rule.get("min", 0)) <= _char_count(text) <= int(rule.get("max", 10**9))
    if rule_type == "contains":
        return str(rule.get("value", "")) in text
    if rule_type == "not_contains":
        return str(rule.get("value", "")) not in text
    if rule_type == "contains_any":
        return any(str(value) in text for value in rule.get("values", []))
    raise ValueError(f"未知规则类型: {rule_type}")


def generation_rule_eval(model, tokenizer, items: list[dict[str, Any]], *, device: str, max_new_tokens: int = 64) -> dict[str, Any]:
    from ..finetune.sft import generate_responses  # noqa: PLC0415

    prompts = [item["instruction"] for item in items]
    responses = generate_responses(model, tokenizer, prompts, device=device, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)
    results = []
    passed = 0
    for item, response in zip(items, responses, strict=True):
        ok = score_by_rule(response, item["rule"])
        passed += int(ok)
        results.append({"instruction": item["instruction"], "response": response, "rule": item["rule"], "passed": ok})
    total = len(items)
    return {"total": total, "passed": passed, "pass_rate": round(passed / total, 4) if total else 0.0, "results": results}


# ---------- 追踪与对比 ----------
def _latest_stage_artifact(stage_slug: str, relative: str) -> Path | None:
    from ..config import PROJECT_ROOT  # noqa: PLC0415

    runs_root = PROJECT_ROOT / "runs" / stage_slug
    if not runs_root.exists():
        return None
    for run_dir in sorted(runs_root.iterdir(), reverse=True):
        candidate = run_dir / relative
        if candidate.exists():
            return candidate
    return None


def collect_cross_stage_artifacts() -> dict[str, Any]:
    collected: dict[str, Any] = {}
    stage4_summary = _latest_stage_artifact("stage-4", "artifacts/stage_summary.json")
    if stage4_summary:
        collected["stage4_summary"] = json.loads(stage4_summary.read_text(encoding="utf-8"))
    mlx_perf = _latest_stage_artifact("stage-4", "artifacts/mlx_performance.json")
    if mlx_perf:
        collected["stage4_mlx_performance"] = json.loads(mlx_perf.read_text(encoding="utf-8"))
    for name, stage_slug, relative in (
        ("sft_before_after", "stage-5", "artifacts/before_after.md"),
        ("dpo_side_by_side", "stage-6", "artifacts/side_by_side.md"),
        ("sft_summary", "stage-5", "artifacts/stage_summary.json"),
        ("dpo_summary", "stage-6", "artifacts/stage_summary.json"),
    ):
        path = _latest_stage_artifact(stage_slug, relative)
        if path:
            collected[name] = path.read_text(encoding="utf-8") if path.suffix == ".md" else json.loads(path.read_text(encoding="utf-8"))
    return collected


def build_compare_report(*, current: dict[str, Any], cross: dict[str, Any]) -> str:
    lines = ["# 跨实验对比报告", ""]
    lines.append("## 1. 预训练规模对比（Stage 4）")
    lines.append("")
    stage4 = cross.get("stage4_summary", {}).get("presets", {})
    if stage4:
        lines.append("| 档位 | 参数量 | tokens | val loss | tokens/s |")
        lines.append("|---|---|---|---|---|")
        for name, payload in stage4.items():
            lines.append(f"| {name} | {payload['params'] / 1e6:.1f}M | {payload['tokens_seen']:,} | {payload['final_val_loss']} | {payload['tokens_per_second']} |")
    else:
        lines.append("（未找到 Stage 4 运行记录）")
    lines.append("")

    lines.append("## 2. 后端对比（MLX vs MPS，Stage 4）")
    lines.append("")
    performance = cross.get("stage4_mlx_performance")
    if performance:
        lines.append(f"- MLX tokens/s: {performance.get('tokens_per_second')}")
        lines.append(f"- MPS tokens/s: {performance.get('mps_tokens_per_second')}")
        lines.append(f"- MLX/MPS: {performance.get('mlx_vs_mps_ratio')}")
    else:
        lines.append("（未找到 MLX 性能记录）")
    lines.append("")

    lines.append("## 3. 评测结果（本次运行）")
    lines.append("")
    lines.append("| 模型 | 困惑度 | 选择题准确率 |")
    lines.append("|---|---|---|")
    for name, payload in current.get("tiny_models", {}).items():
        ppl = payload.get("perplexity", {}).get("perplexity", "-")
        mc = payload.get("multiple_choice", {}).get("accuracy", "-")
        lines.append(f"| {name} | {ppl} | {mc} |")
    lines.append("")
    generation = current.get("generation", {})
    if generation:
        lines.append("## 4. 指令遵循规则评测（生成）")
        lines.append("")
        lines.append(f"- 通过率：{generation.get('pass_rate')}（{generation.get('passed')}/{generation.get('total')}）")
        lines.append("")
        lines.append("| 指令 | 通过 | 输出摘要 |")
        lines.append("|---|---|---|")
        for item in generation.get("results", []):
            response = item["response"].replace("\n", " ")[:80]
            lines.append(f"| {item['instruction']} | {'✅' if item['passed'] else '❌'} | {response} |")
        lines.append("")
    if "sft_before_after" in cross:
        lines.append("## 5. SFT 前后对比（引用 Stage 5 产物）")
        lines.append("")
        lines.append(cross["sft_before_after"])
        lines.append("")
    if "dpo_side_by_side" in cross:
        lines.append("## 6. DPO 对齐对比（引用 Stage 6 产物）")
        lines.append("")
        lines.append(cross["dpo_side_by_side"])
        lines.append("")
    return "\n".join(lines)
