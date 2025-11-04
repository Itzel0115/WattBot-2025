from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from tqdm.auto import tqdm

from ..config import PathConfig, ModelConfig
from ..ingest.chunk_schema import Chunk
from .bm25 import build_bm25_index, save_bm25_index
from .dense_faiss import build_dense_index


logger = logging.getLogger(__name__)


def _load_chunks(chunks_file: Path) -> List[Chunk]:
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"找不到 chunks 檔案: {chunks_file}，請先執行 wattbot-rag build-chunks"
        )

    chunks: List[Chunk] = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading chunks"):
            raw = json.loads(line)
            chunks.append(Chunk.from_raw(raw))
    return chunks


def build_indexes(paths: PathConfig, cfg: ModelConfig, force: bool = False) -> None:
    chunks_file = paths.chunks_dir / "structured_chunks_with_ocr.jsonl"
    bm25_path = paths.indexes_dir / "bm25.pkl"
    meta_path = paths.indexes_dir / "index_meta.json"

    if bm25_path.exists() and not force:
        logger.info("索引已存在（若需重新建立請加上 --force）")
        return

    logger.info("載入 chunks: %s", chunks_file)
    chunks = _load_chunks(chunks_file)
    logger.info("總 chunks 數: %d", len(chunks))

    # BM25
    logger.info("建立 BM25 索引...")
    bm25_index = build_bm25_index(chunks)
    save_bm25_index(bm25_index, bm25_path)

    # Dense + FAISS
    logger.info("建立 Dense (FAISS) 索引...")
    dense_index, _embedding_model, model_name = build_dense_index(
        chunks,
        cfg,
        paths.indexes_dir,
    )
    logger.info(
        "Dense index 建立完成，ntotal=%d，model=%s",
        dense_index.ntotal,
        model_name,
    )

    meta = {
        "chunks_file": str(chunks_file),
        "num_chunks": len(chunks),
        "bm25_index": str(bm25_path),
        "dense_index": str(paths.indexes_dir / "dense.index"),
        "embedding_model_name": model_name,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

