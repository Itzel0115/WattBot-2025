from __future__ import annotations

from dataclasses import dataclass
from typing import List

import logging
from pathlib import Path

import pandas as pd
import requests

from ..config import PathConfig, DEFAULT_CSV_ENCODING


logger = logging.getLogger(__name__)


@dataclass
class DownloadStats:
    success: int
    fail: int
    failed_ids: List[str]


def _download_single_pdf(url: str, doc_id: str, papers_dir: Path) -> bool:
    pdf_path = papers_dir / f"{doc_id}.pdf"
    if pdf_path.exists():
        logger.info("Skip existing PDF: %s", pdf_path.name)
        return True

    logger.info("Downloading %s from %s", doc_id, url)
    try:
        resp = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (WattBot-2025-RAG/1.0)"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to download %s: %s", doc_id, exc)
        return False

    if resp.status_code == 200 and resp.content[:4] == b"%PDF":
        pdf_path.write_bytes(resp.content)
        logger.info("Saved %s", pdf_path)
        return True

    logger.warning(
        "Invalid response for %s: status=%s, first bytes=%r",
        doc_id,
        resp.status_code,
        resp.content[:4],
    )
    return False


def download_pdfs_from_metadata(paths: PathConfig) -> DownloadStats:
    """
    Download all PDF files according to metadata.csv.

    - Automatically fixes known bad URLs (e.g. zschache2025)
    - Downloads only files that do not already exist
    """

    metadata_path = paths.metadata_csv
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    df = pd.read_csv(metadata_path, encoding=DEFAULT_CSV_ENCODING)

    # Fix known bad URL (mirrors original Colab logic)
    mask = df["id"] == "zschache2025"
    if mask.any():
        df.loc[mask, "url"] = "https://arxiv.org/pdf/2508.14170"

    success = 0
    fail = 0
    failed_ids: List[str] = []

    paths.papers_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        doc_id = str(row["id"])
        url = str(row["url"])
        ok = _download_single_pdf(url, doc_id, paths.papers_dir)
        if ok:
            success += 1
        else:
            fail += 1
            failed_ids.append(doc_id)

    logger.info("PDF download finished: success=%d, fail=%d, failed_ids=%s", success, fail, failed_ids)

    return DownloadStats(success=success, fail=fail, failed_ids=failed_ids)

