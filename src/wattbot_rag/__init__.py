"""
WattBot 2025 RAG pipeline.

This package contains modular components for:
- PDF ingestion (Docling + OCR)
- Chunking and schema
- BM25 + dense (FAISS) indexing
- Weighted RRF retrieval and adaptive reranking
- Evaluation on the WattBot 2025 train_QA split
- Optional Gemini-based answer generation
"""

__all__ = [
    "config",
]

