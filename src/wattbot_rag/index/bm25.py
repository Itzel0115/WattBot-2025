from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pickle
import re

from rank_bm25 import BM25Okapi

from ..ingest.chunk_schema import Chunk


def enhanced_tokenize(text: str) -> List[str]:
    """
    改良版 tokenization：
    - 保留數字（含小數與千分位）
    - 額外抓年份
    - 一般英文單詞（長度 > 2）
    """

    text = str(text).lower()
    numbers = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", text)
    years = re.findall(r"\b20\d{2}\b", text)
    words = [w for w in re.findall(r"\w+", text) if len(w) > 2]
    return numbers + years + words


def build_bm25_index(chunks: Iterable[Chunk]) -> BM25Okapi:
    corpus = [c.content for c in chunks]
    tokenized_corpus = [enhanced_tokenize(doc) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def save_bm25_index(index: BM25Okapi, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_bm25_index(path: Path) -> BM25Okapi:
    with path.open("rb") as f:
        return pickle.load(f)

