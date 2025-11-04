from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..config import PathConfig, DEFAULT_CSV_ENCODING
from ..retrieve.adaptive import AdaptiveRetriever


SearchFunc = Callable[[str, str, int, int], Dict]


def evaluate_with_details(train_qa: pd.DataFrame, search_func: SearchFunc) -> Dict:
    recalls = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr_total = 0.0
    total = 0
    failure_cases: List[Dict] = []
    used_rerank_count = 0

    for _, row in tqdm(train_qa.iterrows(), total=len(train_qa), desc="Evaluating"):
        q = row["question"]
        unit = row["answer_unit"]
        true_ids_raw = row["ref_id"]

        try:
            true_docs = ast.literal_eval(true_ids_raw) if isinstance(true_ids_raw, str) else [true_ids_raw]
        except Exception:  # noqa: BLE001
            true_docs = [true_ids_raw]

        res = search_func(q, unit, 150, 10)
        retrieved = [r["doc_id"] for r in res["results"]]

        total += 1

        for k in [1, 3, 5, 10]:
            top_k_set = set(retrieved[:k])
            if any(t in top_k_set for t in true_docs):
                recalls[k] += 1

        rr = 0.0
        for rank, did in enumerate(retrieved, start=1):
            if did in true_docs:
                rr = 1.0 / rank
                break
        mrr_total += rr

        if res.get("used_reranking"):
            used_rerank_count += 1

        if rr == 0:
            failure_cases.append(
                {
                    "question": q,
                    "answer_unit": unit,
                    "true_docs": true_docs,
                    "retrieved": retrieved[:5],
                    "strategy": res.get("strategy"),
                }
            )

    return {
        "recall@1": recalls[1] / total,
        "recall@3": recalls[3] / total,
        "recall@5": recalls[5] / total,
        "recall@10": recalls[10] / total,
        "MRR": mrr_total / total,
        "total_questions": total,
        "failure_cases": failure_cases[:10],
        "rerank_usage": used_rerank_count / total,
    }


def run_evaluation(
    paths: PathConfig,
    retriever: AdaptiveRetriever,
    run_name: Optional[str] = None,
) -> Path:
    train_path = paths.train_qa_csv
    if not train_path.exists():
        raise FileNotFoundError(f"train_QA.csv not found: {train_path}")

    train_qa = pd.read_csv(train_path, encoding=DEFAULT_CSV_ENCODING)

    def _search(q: str, unit: str, initial_k: int, final_k: int) -> Dict:
        return retriever.adaptive_search_final(q, answer_unit=unit, initial_k=initial_k, final_k=final_k)

    metrics = evaluate_with_details(train_qa, _search)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name or f"run-{timestamp}"
    run_dir = paths.runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics.json").write_text(
        json.dumps({k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items() if k != "failure_cases"}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "failure_cases.json").write_text(
        json.dumps(metrics["failure_cases"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return run_dir

