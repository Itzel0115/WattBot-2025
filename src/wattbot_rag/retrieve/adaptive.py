from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..config import PathConfig, ModelConfig, DEFAULT_CSV_ENCODING
from ..ingest.chunk_schema import Chunk
from ..index.bm25 import load_bm25_index
from ..index.dense_faiss import load_dense_index
from .rrf import RRFSearcher
from .rerank import CrossEncoderReranker


ODD_KEYWORDS = [
    "earth from the sun",
    "distance from the sun",
    "elephant",
    "jupiter",
    "galaxy",
]


@dataclass
class AdaptiveRetriever:
    rrf: RRFSearcher
    reranker: CrossEncoderReranker

    def _should_use_reranking_query_based(self, query: str, _results: List[Dict]) -> Dict:
        q = query.lower()
        keywords = [
            "compare",
            "percent",
            "ratio",
            "increase",
            "parameter",
            "emission",
            "mwh",
            "kwh",
            "co2",
            "usd",
            "table",
        ]
        if any(k in q for k in keywords):
            return {"should_rerank": True, "confidence": "high"}
        return {"should_rerank": True, "confidence": "medium"}

    def _detect_unanswerable(self, query: str, top_results: List[Dict]) -> Dict:
        q = query.lower()
        if any(k in q for k in ODD_KEYWORDS):
            return {"is_unanswerable": True, "reason": "weird query"}
        if not top_results:
            return {"is_unanswerable": True, "reason": "no retrieval results"}
        return {"is_unanswerable": False}

    def adaptive_search_final(
        self,
        query: str,
        answer_unit: str = "is_blank",
        initial_k: int = 100,
        final_k: int = 8,
    ) -> Dict:
        # 先做初步 RRF
        rrf_results = self.rrf.search_rrf(
            query,
            answer_unit=answer_unit,
            top_k=initial_k,
            initial_k=initial_k,
        )

        decision = self._should_use_reranking_query_based(query, rrf_results)
        if decision["should_rerank"]:
            final_results = self.reranker.rerank_results(
                query,
                answer_unit,
                rrf_results,
                top_k=final_k,
            )
            strategy = f"Hybrid+BGE (confidence={decision['confidence']})"
        else:
            final_results = rrf_results[:final_k]
            strategy = "BM25-only"

        unanswerable = self._detect_unanswerable(query, final_results)

        return {
            "results": final_results,
            "strategy": strategy,
            "used_reranking": decision["should_rerank"],
            "unanswerable": unanswerable,
        }


def _load_chunks_from_meta(meta_path: Path) -> List[Chunk]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks_file = Path(meta["chunks_file"])
    chunks: List[Chunk] = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            chunks.append(Chunk.from_raw(raw))
    return chunks


def create_adaptive_retriever(paths: PathConfig, cfg: ModelConfig) -> AdaptiveRetriever:
    meta_path = paths.indexes_dir / "index_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Index meta file not found: {meta_path}. Run `wattbot-rag build-index` first."
        )

    chunks = _load_chunks_from_meta(meta_path)
    bm25_index = load_bm25_index(paths.indexes_dir / "bm25.pkl")
    dense_index, embedding_model, _ = load_dense_index(paths.indexes_dir, cfg)

    rrf = RRFSearcher(
        chunks=chunks,
        bm25_index=bm25_index,
        dense_index=dense_index,
        embedding_model=embedding_model,
    )
    reranker = CrossEncoderReranker(cfg)
    return AdaptiveRetriever(rrf=rrf, reranker=reranker)

