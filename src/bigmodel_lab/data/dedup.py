"""去重：精确去重 + MinHash/LSH 近似去重。"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np


def _exact_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def shingle_hashes(text: str, size: int = 5) -> np.ndarray:
    """字符级 shingle 的稳定 32 位哈希（crc32，跨进程稳定）。"""
    if len(text) <= size:
        return np.array([zlib.crc32(text.encode("utf-8"))], dtype=np.uint32)
    shingles = [text[i : i + size] for i in range(len(text) - size + 1)]
    hashes = {zlib.crc32(item.encode("utf-8")) for item in shingles}
    return np.fromiter(hashes, dtype=np.uint32, count=len(hashes))


def minhash_signature(
    text: str,
    *,
    num_perm: int = 64,
    shingle_size: int = 5,
    seed: int = 42,
) -> np.ndarray:
    hashes = shingle_hashes(text, size=shingle_size)
    rng = np.random.default_rng(seed)
    permutations = rng.integers(0, 2**32, size=num_perm, dtype=np.uint64).astype(np.uint32)
    return np.min(hashes.astype(np.uint64)[None, :] ^ permutations.astype(np.uint64)[:, None], axis=1).astype(np.uint32)


@dataclass
class DedupConfig:
    exact: bool = True
    near: bool = True
    near_threshold: float = 0.8
    shingle_size: int = 5
    num_perm: int = 64
    bands: int = 16

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> DedupConfig:
        payload = payload or {}
        return cls(
            exact=bool(payload.get("exact", True)),
            near=bool(payload.get("near", True)),
            near_threshold=float(payload.get("near_threshold", 0.8)),
            shingle_size=int(payload.get("shingle_size", 5)),
            num_perm=int(payload.get("num_perm", 64)),
            bands=int(payload.get("bands", 16)),
        )

    @property
    def rows(self) -> int:
        return max(1, self.num_perm // self.bands)


@dataclass
class DedupStats:
    docs_in: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    docs_kept: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "docs_in": self.docs_in,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "docs_kept": self.docs_kept,
            "duplicate_rate": round((self.exact_duplicates + self.near_duplicates) / self.docs_in, 4) if self.docs_in else 0.0,
        }


class Deduper:
    """流式去重：重复文档返回 None，保留文档返回文本。"""

    def __init__(self, config: DedupConfig):
        self.config = config
        self.stats = DedupStats()
        self._exact_seen: set[str] = set()
        self._signatures: list[np.ndarray] = []
        self._buckets: list[dict[tuple[int, ...], list[int]]] = [dict() for _ in range(config.bands)]

    def add(self, text: str) -> str | None:
        self.stats.docs_in += 1
        if self.config.exact:
            digest = _exact_hash(text)
            if digest in self._exact_seen:
                self.stats.exact_duplicates += 1
                return None
            self._exact_seen.add(digest)

        if self.config.near:
            signature = minhash_signature(
                text,
                num_perm=self.config.num_perm,
                shingle_size=self.config.shingle_size,
            )
            candidate_indices: set[int] = set()
            for band_index in range(self.config.bands):
                start = band_index * self.config.rows
                key = tuple(int(value) for value in signature[start : start + self.config.rows])
                candidate_indices.update(self._buckets[band_index].get(key, []))
            for candidate in candidate_indices:
                similarity = float(np.mean(self._signatures[candidate] == signature))
                if similarity >= self.config.near_threshold:
                    self.stats.near_duplicates += 1
                    return None
            document_index = len(self._signatures)
            self._signatures.append(signature)
            for band_index in range(self.config.bands):
                start = band_index * self.config.rows
                key = tuple(int(value) for value in signature[start : start + self.config.rows])
                self._buckets[band_index].setdefault(key, []).append(document_index)

        self.stats.docs_kept += 1
        return text

    def summary(self) -> dict[str, Any]:
        return self.stats.summary()
