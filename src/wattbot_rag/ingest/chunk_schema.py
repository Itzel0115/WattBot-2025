from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class Chunk:
    """
    Generic chunk schema.

    Fields
    ------
    - doc_id: source paper ID (matches metadata.csv id)
    - type: "text" / "table" / "figure" / "image_ocr" / ...
    - content: plain text used for retrieval / LLMs
    - metadata: extra fields (e.g. section, caption)
    - page: source PDF page number (1-based)
    - word_count: number of tokens in content (simple whitespace split)
    - image_path: for image_ocr, path to the corresponding image file
    """

    doc_id: str
    type: str
    content: str
    metadata: Dict[str, Any]
    page: Optional[int] = None
    word_count: int = 0
    image_path: Optional[str] = None

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "Chunk":
        """Safely convert from a legacy dict structure to a Chunk instance."""

        metadata = raw.get("metadata") or {}
        page = raw.get("page") or metadata.get("page")
        image_path = raw.get("image_path") or metadata.get("image_path")

        return cls(
            doc_id=str(raw.get("doc_id")),
            type=str(raw.get("type", "text")),
            content=str(raw.get("content", "")),
            metadata=metadata,
            page=page,
            word_count=int(raw.get("word_count", len(str(raw.get("content", "")).split()))),
            image_path=image_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a normalized dict structure suitable for JSONL storage."""

        data = asdict(self)
        # Keep a structure similar to the original Colab version; store page/image_path in metadata
        if self.page is not None:
            data["metadata"] = dict(data.get("metadata") or {})
            data["metadata"]["page"] = self.page
        if self.image_path is not None:
            data["metadata"] = dict(data.get("metadata") or {})
            data["metadata"]["image_path"] = self.image_path
        return data

