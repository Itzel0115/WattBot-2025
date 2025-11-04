from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from ..ingest.chunk_schema import Chunk
from ..config import ModelConfig


def _load_embedding_model(cfg: ModelConfig) -> Tuple[SentenceTransformer, str]:
    model_name = cfg.embedding_model_name
    try:
        model = SentenceTransformer(model_name)
    except Exception:  # noqa: BLE001
        # fallback
        model_name = cfg.embedding_fallback_model_name
        model = SentenceTransformer(model_name)
    return model, model_name


def build_dense_index(
    chunks: Iterable[Chunk],
    cfg: ModelConfig,
    indexes_dir: Path,
) -> Tuple[faiss.IndexFlatIP, SentenceTransformer, str]:
    chunks_list: List[Chunk] = list(chunks)
    model, model_name = _load_embedding_model(cfg)

    texts_to_encode = [
        f"Type: {c.type.upper()}\n{c.content[:1000]}" for c in chunks_list
    ]
    embeddings = model.encode(
        texts_to_encode,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    indexes_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(indexes_dir / "dense.index"))
    (indexes_dir / "dense_config.json").write_text(
        json.dumps({"model_name": model_name}, indent=2),
        encoding="utf-8",
    )

    return index, model, model_name


def load_dense_index(
    indexes_dir: Path,
    cfg: ModelConfig,
) -> Tuple[faiss.IndexFlatIP, SentenceTransformer, str]:
    index_path = indexes_dir / "dense.index"
    config_path = indexes_dir / "dense_config.json"
    if not index_path.exists():
        raise FileNotFoundError(f"找不到 dense.index，請先執行 wattbot-rag build-index ({index_path})")

    index = faiss.read_index(str(index_path))

    if config_path.exists():
        meta = json.loads(config_path.read_text(encoding="utf-8"))
        model_name = meta.get("model_name") or cfg.embedding_model_name
        # 覆寫配置，確保使用與當初 index 相同的模型名稱
        cfg = ModelConfig(
            embedding_model_name=model_name,
            embedding_fallback_model_name=cfg.embedding_fallback_model_name,
            cross_encoder_model_name=cfg.cross_encoder_model_name,
            cross_encoder_fallback_model_name=cfg.cross_encoder_fallback_model_name,
        )
    else:
        model_name = cfg.embedding_model_name

    model, model_name = _load_embedding_model(cfg)
    return index, model, model_name

