from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import os

from dotenv import load_dotenv


load_dotenv()


@dataclass
class PathConfig:
    """Path configuration for data and artifacts."""

    base_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    papers_dir: Path = field(default_factory=lambda: Path.cwd() / "papers")
    outputs_dir: Path = field(default_factory=lambda: Path.cwd() / "outputs")

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).resolve()
        self.data_dir = Path(self.data_dir).resolve()
        self.papers_dir = Path(self.papers_dir).resolve()
        self.outputs_dir = Path(self.outputs_dir).resolve()

    @property
    def chunks_dir(self) -> Path:
        return self.outputs_dir / "chunks"

    @property
    def images_dir(self) -> Path:
        return self.outputs_dir / "images"

    @property
    def indexes_dir(self) -> Path:
        return self.outputs_dir / "indexes"

    @property
    def runs_dir(self) -> Path:
        return self.outputs_dir / "runs"

    @property
    def metadata_csv(self) -> Path:
        return self.data_dir / "metadata.csv"

    @property
    def train_qa_csv(self) -> Path:
        return self.data_dir / "train_QA.csv"

    @property
    def test_q_csv(self) -> Path:
        return self.data_dir / "test_Q.csv"


@dataclass
class ModelConfig:
    """Model configuration for embedding and reranking."""

    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_fallback_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model_name: str = "BAAI/bge-reranker-v2-m3"
    cross_encoder_fallback_model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"


DEFAULT_CSV_ENCODING: str = "latin-1"
TEST_Q_ENCODING: str = "utf-8-sig"


def get_google_api_key() -> str:
    """Return the Gemini API key from environment, or raise if missing."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable GOOGLE_API_KEY is not set.\n"
            "Create a .env file in the project root (see .env.example) or export GOOGLE_API_KEY manually."
        )
    return api_key


def ensure_directories(paths: PathConfig) -> None:
    """Create required directories if they do not exist."""

    for d in [
        paths.data_dir,
        paths.papers_dir,
        paths.outputs_dir,
        paths.chunks_dir,
        paths.images_dir,
        paths.indexes_dir,
        paths.runs_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

