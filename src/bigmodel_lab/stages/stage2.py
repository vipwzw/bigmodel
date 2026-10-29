"""Stage 2：数据工程——多源获取、清洗、去重、统计与数据卡。"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from ..budget import Budget
from ..config import LabConfig
from ..data.card import build_dataset_card
from ..data.clean import CleanConfig, CorpusCleaner
from ..data.dedup import DedupConfig, Deduper
from ..data.io import iter_documents, resolve_files
from ..data.specs import data_config, limit_for, load_resources
from ..data.sources import fetch_resource
from ..manifest import RunContext

logger = logging.getLogger(__name__)

PRETRAIN_KEYS = ("pretrain_zh", "pretrain_en")
_PROGRESS_EVERY = 5000


def process_corpus(
    *,
    key: str,
    files: list[Path],
    text_field: str | None,
    language: str,
    output_path: Path,
    clean_config: CleanConfig,
    dedup_config: DedupConfig,
    max_docs: int | None,
    budget: Budget,
) -> dict[str, Any]:
    """清洗 + 去重 + 写出 jsonl，返回统计结果。"""
    cleaner = CorpusCleaner(dataset=key, language=language, config=clean_config, max_docs=max_docs)
    deduper = Deduper(dedup_config)
    docs_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, (document, cleaned) in enumerate(cleaner.run(iter_documents(files, text_field))):
            if index % 500 == 0:
                budget.check()
            if index and index % _PROGRESS_EVERY == 0:
                logger.info("[%s] 已处理 %d 篇，保留 %d 篇", key, index, docs_written)
            kept = deduper.add(cleaned) if dedup_config.exact or dedup_config.near else cleaned
            if kept is None:
                continue
            record = {
                "doc_id": hashlib.sha1(kept.encode("utf-8")).hexdigest()[:16],
                "source": key,
                "language": language,
                "text": kept,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            docs_written += 1
    stats = cleaner.summary()
    dedup_stats = deduper.summary()
    stats["docs_written"] = docs_written
    stats["output"] = {
        "path": str(output_path),
        "size_bytes": output_path.stat().st_size,
    }
    return {"key": key, "stats": stats, "dedup": dedup_stats, "output_path": output_path}


def run(*, ctx: RunContext, config: dict[str, Any], budget: Budget) -> dict[str, Any]:
    lab_config = LabConfig.model_validate(config)
    cfg = data_config(config)
    resources = load_resources(config)
    clean_config = CleanConfig.from_dict(cfg.get("clean"))
    dedup_config = DedupConfig.from_dict(cfg.get("dedup"))

    data_dir = lab_config.resolved_dir("data_dir")
    processed_dir = data_dir / "processed" / "stage2"
    processed_dir.mkdir(parents=True, exist_ok=True)

    fetch_manifest: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    stats_by_dataset: dict[str, Any] = {}

    for key in PRETRAIN_KEYS:
        spec = resources.get(key)
        if spec is None:
            raise RuntimeError(f"配置缺少 data.resources.{key}")
        budget.check()
        logger.info("[%s] 获取资源 %s（来源优先级: %s）", key, spec.repo_id, lab_config.network.priority)
        entry = fetch_resource(
            spec,
            data_dir=data_dir,
            network=lab_config.network,
            offline=lab_config.offline,
            budget=budget,
        )
        fetch_manifest[key] = entry
        files = resolve_files(Path(entry["dest"]), spec.files)
        max_docs = limit_for(key, config, budget.profile)
        logger.info("[%s] 清洗去重（文件 %d 个，上限 %s 篇）", key, len(files), max_docs or "全量")
        result = process_corpus(
            key=key,
            files=files,
            text_field=spec.text_field,
            language=spec.language,
            output_path=processed_dir / f"{key}.jsonl",
            clean_config=clean_config,
            dedup_config=dedup_config,
            max_docs=max_docs,
            budget=budget,
        )
        stats_by_dataset[key] = {"clean": result["stats"], "dedup": result["dedup"]}
        entries.append(
            {
                "key": key,
                "language": spec.language,
                "license": spec.license,
                "repo_id": spec.repo_id,
                "files": ", ".join(str(path.relative_to(Path(entry["dest"]))) for path in files),
                "stats": result["stats"],
                "dedup": result["dedup"],
            }
        )
        ctx.add_artifact(result["output_path"])
        ctx.log_metrics(
            kind="data",
            dataset=key,
            docs_in=result["stats"]["docs_in"],
            docs_kept=result["dedup"]["docs_kept"],
            chars_kept=result["stats"]["chars_kept"],
            exact_duplicates=result["dedup"]["exact_duplicates"],
            near_duplicates=result["dedup"]["near_duplicates"],
        )

    ctx.write_json_artifact("fetch_manifest.json", fetch_manifest)
    ctx.write_json_artifact(
        "clean_stats.json",
        {"profile": budget.profile, "datasets": stats_by_dataset, "processed_dir": str(processed_dir)},
    )
    card = build_dataset_card(entries=entries, profile=budget.profile, data_root=str(processed_dir))
    ctx.write_artifact("dataset_card.md", card)

    summary = {
        "profile": budget.profile,
        "processed_dir": str(processed_dir),
        "datasets": {
            key: {
                "docs_in": value["clean"]["docs_in"],
                "docs_kept": value["dedup"]["docs_kept"],
                "chars_kept": value["clean"]["chars_kept"],
                "keep_rate": value["clean"]["keep_rate"],
                "duplicate_rate": value["dedup"]["duplicate_rate"],
                "output": value["clean"]["output"],
            }
            for key, value in stats_by_dataset.items()
        },
    }
    ctx.write_json_artifact("stage_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
