from __future__ import annotations

from typing import Dict, List

from sentence_transformers import CrossEncoder
import torch

from ..config import ModelConfig


class CrossEncoderReranker:
    def __init__(self, cfg: ModelConfig) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = CrossEncoder(
                cfg.cross_encoder_model_name,
                device=device,
                max_length=1024,
            )
            self.model_name = cfg.cross_encoder_model_name
        except Exception:  # noqa: BLE001
            model = CrossEncoder(
                cfg.cross_encoder_fallback_model_name,
                device=device,
                max_length=512,
            )
            self.model_name = cfg.cross_encoder_fallback_model_name

        self.model = model
        self.device = device

    def rerank_results(
        self,
        query: str,
        answer_unit: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        對 RRF 召回結果進行 CrossEncoder 重排。
        """

        augmented_query = f"{query} (Target unit: {answer_unit})"
        pairs = [
            [augmented_query, c["content"][:2000]]
            for c in candidates
        ]

        ce_scores = self.model.predict(pairs)
        for i, c in enumerate(candidates):
            c["ce_score"] = float(ce_scores[i])

        reranked = sorted(
            candidates,
            key=lambda x: x["ce_score"],
            reverse=True,
        )[:top_k]

        for rank, item in enumerate(reranked, start=1):
            item["final_rank"] = rank
        return reranked

