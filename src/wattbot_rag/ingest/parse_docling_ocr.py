from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from docling.datamodel.document import PictureItem, TableItem, TextItem
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import cv2
import easyocr
from tqdm.auto import tqdm

from ..config import PathConfig
from .chunk_schema import Chunk


logger = logging.getLogger(__name__)


def _init_ocr_reader() -> easyocr.Reader:
    try:
        import torch

        gpu_available = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        gpu_available = False

    logger.info("Initializing EasyOCR (GPU available=%s)", gpu_available)
    return easyocr.Reader(["en"], gpu=gpu_available)


def _extract_ocr_from_image(
    image: Image.Image | np.ndarray,
    doc_id: str,
    page_no: int,
    img_index: int,
    images_dir: Path,
    ocr_reader: easyocr.Reader,
) -> tuple[str | None, str | None]:
    try:
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        elif isinstance(image, np.ndarray):
            img_array = image
        else:
            return None, None

        if len(img_array.shape) == 2:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        image_filename = f"{doc_id}_p{page_no}_img{img_index}.png"
        image_path = images_dir / image_filename
        cv2.imwrite(str(image_path), img_bgr)

        results = ocr_reader.readtext(img_bgr)
        texts = [text for (_bbox, text, conf) in results if conf > 0.2]
        ocr_text = "\n".join(texts)

        if ocr_text.strip():
            return ocr_text, str(image_path)
        return None, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR failed for doc_id=%s page=%s: %s", doc_id, page_no, exc)
        return None, None


def _process_single_pdf(
    pdf_path: Path,
    doc_id: str,
    images_dir: Path,
    converter: DocumentConverter,
    ocr_reader: easyocr.Reader,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    image_counter = 0

    try:
        result = converter.convert(str(pdf_path))
        doc = result.document
    except Exception as exc:  # noqa: BLE001
        logger.error("Docling failed for doc_id=%s: %s", doc_id, exc)
        return chunks

    for item, _level in doc.iterate_items():
        page_no = item.prov[0].page_no if item.prov else 0

        if isinstance(item, TableItem):
            table_md = item.export_to_markdown()
            content = f"Table found on page {page_no}:\n{table_md}"
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    type="table",
                    content=content,
                    metadata={"page": page_no},
                    page=page_no,
                    word_count=len(content.split()),
                )
            )
        elif isinstance(item, PictureItem):
            image_placeholder_content = f"[Figure on page {page_no}]"
            ocr_text: str | None = None
            image_path: str | None = None

            try:
                if getattr(item, "image", None) is not None:
                    ocr_text, image_path = _extract_ocr_from_image(
                        item.image,
                        doc_id,
                        page_no,
                        image_counter,
                        images_dir,
                        ocr_reader,
                    )
                    image_counter += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Unable to extract image for doc_id=%s page=%s: %s", doc_id, page_no, exc)

            if ocr_text:
                content = f"[Image/Chart on page {page_no}]\nOCR Text:\n{ocr_text}"
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        type="image_ocr",
                        content=content,
                        metadata={"page": page_no, "image_path": image_path},
                        page=page_no,
                        word_count=len(ocr_text.split()),
                        image_path=image_path,
                    )
                )
            else:
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        type="figure",
                        content=image_placeholder_content,
                        metadata={"page": page_no},
                        page=page_no,
                        word_count=len(image_placeholder_content.split()),
                    )
                )
        elif isinstance(item, TextItem):
            text = (item.text or "").strip()
            if len(text) < 20:
                continue
            if text.lower() in {"references", "bibliography"}:
                continue

            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    type="text",
                    content=text,
                    metadata={"page": page_no},
                    page=page_no,
                    word_count=len(text.split()),
                )
            )

    return chunks


def build_chunks_from_papers(paths: PathConfig, force: bool = False) -> Path:
    """
    Parse all PDFs in `papers/` with Docling + EasyOCR and produce a unified chunks JSONL.

    Outputs:
        {outputs_dir}/chunks/structured_chunks_with_ocr.jsonl
        {outputs_dir}/images/*.png
    """

    chunks_dir = paths.chunks_dir
    images_dir = paths.images_dir
    chunks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    out_file = chunks_dir / "structured_chunks_with_ocr.jsonl"
    if out_file.exists() and not force:
        logger.info("Chunks already exist, reusing: %s", out_file)
        return out_file

    pdf_files = sorted(paths.papers_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {paths.papers_dir}. Run `wattbot-rag download-pdfs` first."
        )

    logger.info("Initializing Docling converter")
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF])
    ocr_reader = _init_ocr_reader()

    all_chunks: List[Chunk] = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs with OCR"):
        doc_id = pdf_file.stem
        chunks = _process_single_pdf(pdf_file, doc_id, images_dir, converter, ocr_reader)
        all_chunks.extend(chunks)

    with out_file.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    logger.info("Chunking finished, total chunks: %d", len(all_chunks))

    type_counts = Counter(c.type for c in all_chunks)
    for t, count in sorted(type_counts.items()):
        logger.info("Chunk type %s: %d", t, count)

    ocr_chunks = [c for c in all_chunks if c.type == "image_ocr"]
    if ocr_chunks:
        avg_wc = float(np.mean([c.word_count for c in ocr_chunks]))
        max_wc = max(c.word_count for c in ocr_chunks)
        logger.info("OCR chunks: %d, avg word count: %.0f, max: %d", len(ocr_chunks), avg_wc, max_wc)
    else:
        logger.warning("No OCR chunks were successfully extracted")

    return out_file

