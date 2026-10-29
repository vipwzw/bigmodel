"""Stage 3：分词器训练、指令/偏好数据格式化、序列打包与数据划分。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..budget import Budget
from ..config import LabConfig
from ..data.instruction import format_instruction_files, write_instruction_splits
from ..data.io import resolve_files
from ..data.packing import encode_documents, pack_sequences, save_packed, split_records, validate_packed
from ..data.preference import format_preference_files, write_preference_splits
from ..data.sources import fetch_resource
from ..data.specs import data_config, limit_for, load_resources
from ..data.tokenizer import evaluate_tokenizer, load_tokenizer, train_bpe
from ..manifest import RunContext

logger = logging.getLogger(__name__)

CORPUS_TO_TOKENIZERS: dict[str, str] = {"pretrain_zh": "all", "pretrain_en": "smallest"}


def _read_clean_documents(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            documents.append(json.loads(line))
            if limit is not None and len(documents) >= limit:
                break
    return documents


def _train_tokenizers(
    *,
    budget: Budget,
    config: dict[str, Any],
    stage2_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    cfg = data_config(config).get("tokenizer") or {}
    vocab_sizes = [int(value) for value in cfg.get("vocab_sizes", [16384, 32768])]
    train_docs = int((cfg.get("train_docs") or {}).get(budget.profile, 3000))
    eval_docs = int(cfg.get("eval_docs", 200))
    min_frequency = int(cfg.get("min_frequency", 2))
    source_path = stage2_dir / "pretrain_zh.jsonl"
    if not source_path.exists():
        raise RuntimeError(f"缺少 Stage 2 产物 {source_path}，请先运行 lab run 2")
    documents = _read_clean_documents(source_path, limit=max(train_docs, eval_docs))
    train_texts = [document["text"] for document in documents[:train_docs]]
    eval_texts = [document["text"] for document in documents[:eval_docs]]
    reports: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for vocab_size in vocab_sizes:
        budget.check()
        target = output_dir / f"bpe_{vocab_size}"
        logger.info("训练 BPE 分词器 vocab_size=%d（%d 篇文档）", vocab_size, len(train_texts))
        meta = train_bpe(train_texts, vocab_size=vocab_size, output_dir=target, min_frequency=min_frequency)
        report = evaluate_tokenizer(target / "tokenizer.json", eval_texts, max_docs=eval_docs)
        report["meta"] = meta
        reports[f"bpe_{vocab_size}"] = report
        paths[str(vocab_size)] = target / "tokenizer.json"
    return reports, paths


def _pack_corpora(
    *,
    budget: Budget,
    config: dict[str, Any],
    stage2_dir: Path,
    output_dir: Path,
    tokenizer_paths: dict[str, Path],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = data_config(config)
    split_cfg = cfg.get("split") or {}
    pack_cfg = cfg.get("packing") or {}
    block_size = int(pack_cfg.get("block_size", 512))
    sanity_caps = pack_cfg.get("sanity_max_tokens") or {}
    val_ratio = float(split_cfg.get("val_ratio", 0.01))
    test_ratio = float(split_cfg.get("test_ratio", 0.01))
    vocab_keys = sorted(tokenizer_paths, key=int)
    smallest = vocab_keys[0]

    pack_report: dict[str, Any] = {}
    split_manifest: dict[str, Any] = {"seed": seed, "val_ratio": val_ratio, "test_ratio": test_ratio, "corpora": {}}

    for corpus, mode in CORPUS_TO_TOKENIZERS.items():
        source_path = stage2_dir / f"{corpus}.jsonl"
        if not source_path.exists():
            raise RuntimeError(f"缺少 Stage 2 产物 {source_path}，请先运行 lab run 2")
        documents = _read_clean_documents(source_path)
        splits = split_records(documents, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
        split_manifest["corpora"][corpus] = {name: len(items) for name, items in splits.items()}
        vocab_list = vocab_keys if mode == "all" else [smallest]
        for vocab_key in vocab_list:
            budget.check()
            tokenizer = load_tokenizer(tokenizer_paths[vocab_key])
            meta = json.loads((tokenizer_paths[vocab_key].parent / "meta.json").read_text(encoding="utf-8"))
            eos_id = meta["special_token_ids"].get("<eos>")
            pad_id = meta["special_token_ids"].get("<pad>") or 0
            cap = int(sanity_caps[corpus]) if budget.profile == "sanity" and corpus in sanity_caps else None
            entry: dict[str, Any] = {"block_size": block_size, "splits": {}}
            target_dir = output_dir / f"{corpus}_bpe{vocab_key}"
            for split_name, split_docs in splits.items():
                max_tokens = None
                if cap is not None:
                    max_tokens = cap if split_name == "train" else max(100_000, cap // 20)
                token_ids, segment_ids = encode_documents(split_docs, tokenizer, eos_id=eos_id)
                arrays = pack_sequences(
                    token_ids,
                    segment_ids,
                    block_size=block_size,
                    pad_id=pad_id,
                    max_tokens=max_tokens,
                )
                validation = validate_packed(arrays, vocab_size=meta["vocab_size"])
                split_meta = save_packed(
                    target_dir,
                    split_name,
                    arrays,
                    {
                        "corpus": corpus,
                        "split": split_name,
                        "tokenizer": str(tokenizer_paths[vocab_key]),
                        "vocab_size": meta["vocab_size"],
                        "block_size": block_size,
                        "documents": len(split_docs),
                        "tokens": int(arrays["loss_mask"].sum()),
                        "max_tokens_cap": max_tokens,
                        "validation": validation,
                    },
                )
                entry["splits"][split_name] = split_meta
                logger.info(
                    "[%s/%s] %s: %d 篇文档 → %d 序列 / %d token",
                    corpus, vocab_key, split_name, len(split_docs),
                    validation["sequences"], validation["real_tokens"],
                )
                budget.check()
            pack_report[f"{corpus}_bpe{vocab_key}"] = entry

    return pack_report, split_manifest


def run(*, ctx: RunContext, config: dict[str, Any], budget: Budget) -> dict[str, Any]:
    lab_config = LabConfig.model_validate(config)
    data_dir = lab_config.resolved_dir("data_dir")
    stage2_dir = data_dir / "processed" / "stage2"
    processed_dir = data_dir / "processed" / "stage3"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cfg = data_config(config)
    seed = int(config.get("seed", 42))

    tokenizer_reports, tokenizer_paths = _train_tokenizers(
        budget=budget, config=config, stage2_dir=stage2_dir, output_dir=processed_dir / "tokenizers"
    )
    ctx.write_json_artifact("tokenizer_report.json", tokenizer_reports)

    pack_report, split_manifest = _pack_corpora(
        budget=budget,
        config=config,
        stage2_dir=stage2_dir,
        output_dir=processed_dir / "packed",
        tokenizer_paths=tokenizer_paths,
        seed=seed,
    )
    ctx.write_json_artifact("pack_report.json", pack_report)
    ctx.write_json_artifact("split_manifest.json", split_manifest)

    resources = load_resources(config)
    instruction_cfg = cfg.get("instruction") or {}
    instruction_report: dict[str, Any] = {}
    sft_spec = resources.get("sft_zh")
    if sft_spec is None:
        raise RuntimeError("配置缺少 data.resources.sft_zh")
    entry = fetch_resource(sft_spec, data_dir=data_dir, network=lab_config.network, offline=lab_config.offline, budget=budget)
    sft_files = resolve_files(Path(entry["dest"]), sft_spec.files)
    instruction = format_instruction_files(
        sft_files,
        system_prompt=str(instruction_cfg.get("system_prompt", "你是一个乐于助人的中文助手。")),
        max_chars=int(instruction_cfg.get("max_chars", 4096)),
        seed=seed,
        val_ratio=float(instruction_cfg.get("val_ratio", 0.05)),
        max_records=limit_for("sft_zh", config, budget.profile),
    )
    instruction_paths = write_instruction_splits(instruction["splits"], processed_dir / "instruction" / "sft_zh")
    instruction_report = {**instruction["stats"], "paths": instruction_paths, "source": entry["status"]}
    ctx.write_json_artifact("instruction_report.json", instruction_report)
    for path in instruction_paths.values():
        ctx.add_artifact(path)

    preference_cfg = cfg.get("preference") or {}
    pref_spec = resources.get("pref_en")
    pref_report: dict[str, Any] = {}
    if pref_spec is not None:
        pref_entry = fetch_resource(pref_spec, data_dir=data_dir, network=lab_config.network, offline=lab_config.offline, budget=budget)
        pref_files = resolve_files(Path(pref_entry["dest"]), pref_spec.files)
        preference = format_preference_files(
            pref_files,
            max_chars=int(preference_cfg.get("max_chars", 4096)),
            seed=seed,
            val_ratio=float(preference_cfg.get("val_ratio", 0.05)),
            max_pairs=preference_cfg.get("max_pairs"),
        )
        preference_paths = write_preference_splits(preference["splits"], processed_dir / "preference" / "pref_en")
        pref_report = {**preference["stats"], "paths": preference_paths, "source": pref_entry["status"]}
        ctx.write_json_artifact("preference_report.json", pref_report)
        for path in preference_paths.values():
            ctx.add_artifact(path)

    summary = {
        "profile": budget.profile,
        "processed_dir": str(processed_dir),
        "tokenizers": {name: {"vocab_size": report["vocab_size"], "chars_per_token": report["chars_per_token"], "unk_rate": report["unk_rate"]} for name, report in tokenizer_reports.items()},
        "packed": {
            name: {split: info["real_tokens"] for split, info in entry["splits"].items()}
            for name, entry in pack_report.items()
        },
        "splits": split_manifest["corpora"],
        "instruction": {key: instruction_report.get(key) for key in ("docs_in", "docs_valid", "keep_rate", "splits")},
        "preference": {key: pref_report.get(key) for key in ("docs_in", "docs_valid", "keep_rate", "splits")},
    }
    ctx.write_json_artifact("stage_summary.json", summary)
    ctx.log_metrics(kind="stage3", tokenizers=len(tokenizer_reports), packed_datasets=len(pack_report))
    for name, report in tokenizer_reports.items():
        ctx.log_metrics(kind="tokenizer", tokenizer=name, chars_per_token=report["chars_per_token"], vocab_size=report["vocab_size"])
    for name, entry in pack_report.items():
        ctx.log_metrics(kind="packed", dataset=name, train_tokens=entry["splits"]["train"]["real_tokens"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
