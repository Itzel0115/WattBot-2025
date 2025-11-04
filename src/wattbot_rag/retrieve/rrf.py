from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ..ingest.chunk_schema import Chunk
from ..index.bm25 import enhanced_tokenize


UNIT_KEYWORDS: Dict[str, str] = {
    # Carbon emissions related
    "tCO2e": "carbon emissions co2 greenhouse footprint tonnes",
    "tCO2": "carbon emissions co2 greenhouse footprint tonnes",
    "kgCO2e": "carbon emissions co2 greenhouse footprint kilograms",
    "gCO2": "carbon intensity gram emissions",
    "gCO2e/kWh": "carbon intensity per kwh emissions electricity",
    # Energy units
    "MWh": "energy electricity consumption power megawatt hour",
    "kWh": "energy electricity consumption power kilowatt hour",
    "Wh": "energy electricity consumption power watt hour",
    "TWh": "energy electricity consumption power terawatt hour",
    # Efficiency metrics
    "PUE": "power usage effectiveness data center efficiency",
    "L/kWh": "water consumption per kwh cooling efficiency",
    # Water volume
    "liters": "water consumption withdrawal usage volume",
    "gallons": "water consumption withdrawal usage volume",
    # Monetary units
    "USD": "cost price budget dollar investment money",
    "EUR": "euro cost price budget investment",
    # Model size / token counts
    "parameters": "model size scale billion parameters",
    "tokens": "training tokens dataset size sequence length samples",
    # Time
    "seconds": "time duration latency",
    "ms": "time duration latency milliseconds",
    "minutes": "time duration minutes",
    "hours": "time duration hours",
    "days": "time duration days",
    "years": "time duration age",
    # Hardware
    "GPUs": "gpu hardware accelerators nvidia cards",
    "A100_80GB_GPU": "nvidia a100 80gb gpu hardware accelerator",
    "H100 GPUs": "nvidia h100 gpu hardware accelerator",
    # Ratios / percentages
    "percent": "percentage rate ratio proportion share",
    "fold": "increase improvement speedup times",
    # Other
    "samples": "batch size number of samples",
}


class RRFSearcher:
    """
    Weighted RRF retriever:
    - BM25 + dense (FAISS)
    - Unit expansion (UNIT_KEYWORDS)
    - Table boosting (table ×2)
    - Image OCR boosting (image_ocr ×1.8)
    - Metadata hints (author / year ×2)
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        bm25_index,
        dense_index: faiss.Index,
        embedding_model: SentenceTransformer,
    ) -> None:
        self.chunks: List[Chunk] = list(chunks)
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.embedding_model = embedding_model
        self.doc_ids_set: Set[str] = {c.doc_id for c in self.chunks}

    def _extract_metadata_hints(self, query: str) -> List[str]:
        query_lower = query.lower()
        hints: List[str] = []

        # Years
        years = np.unique(np.array(__import__("re").findall(r"\b(20\d{2})\b", query)))
        hints.extend(list(years))

        # Author names (alphabetic prefix of doc_id)
        import re

        for doc_id in self.doc_ids_set:
            author_part = re.sub(r"\d+", "", doc_id).lower()
            if len(author_part) > 3 and author_part in query_lower:
                hints.append(doc_id)

        return hints

    def search_rrf(
        self,
        query: str,
        answer_unit: str = "is_blank",
        top_k: int = 10,
        initial_k: int = 150,
    ) -> List[Dict]:
        # A. Unit expansion
        expanded_query = query
        unit_str = str(answer_unit).strip()
        if unit_str in UNIT_KEYWORDS:
            expanded_query = f"{query} {UNIT_KEYWORDS[unit_str]}"

        meta_hints = self._extract_metadata_hints(query)

        # B. Retrieval (BM25 + dense)
        # BM25: use expanded query
        bm25_tokens = enhanced_tokenize(expanded_query)
        bm25_scores = self.bm25_index.get_scores(bm25_tokens)
        top_n_bm25 = np.argsort(bm25_scores)[::-1][:initial_k]

        # Dense: use original query
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )
        dense_scores, dense_indices = self.dense_index.search(
            query_embedding.astype(np.float32),
            initial_k,
        )
        dense_indices = dense_indices[0]

        # C. RRF + weighting
        doc_scores: Dict[int, float] = defaultdict(float)
        k = 60

        for rank, idx in enumerate(top_n_bm25):
            chunk = self.chunks[idx]
            type_ = chunk.type
            weight = 1.0
            if type_ == "table":
                weight *= 2.0
            if type_ == "image_ocr":
                weight *= 1.8
            if any(hint in chunk.doc_id for hint in meta_hints):
                weight *= 2.0

            doc_scores[idx] += (1 / (k + rank + 1)) * weight

        for rank, idx in enumerate(dense_indices):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            type_ = chunk.type
            weight = 1.0
            if type_ == "table":
                weight *= 2.0
            if type_ == "image_ocr":
                weight *= 1.8
            if any(hint in chunk.doc_id for hint in meta_hints):
                weight *= 2.0

            doc_scores[idx] += (1 / (k + rank + 1)) * weight

        # D. Sort and output
        sorted_indices = sorted(
            doc_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        results: List[Dict] = []
        for idx, score in sorted_indices:
            chunk = self.chunks[idx]
            meta = dict(chunk.metadata or {})
            results.append(
                {
                    "doc_id": chunk.doc_id,
                    "content": chunk.content,
                    "type": chunk.type,
                    "score": float(score),
                    "metadata": meta,
                }
            )
        return results

