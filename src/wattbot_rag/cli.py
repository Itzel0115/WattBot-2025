from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import PathConfig, ModelConfig, ensure_directories

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


app = typer.Typer(help="WattBot 2025 RAG pipeline CLI")


def _build_paths(
    data_dir: Optional[Path],
    papers_dir: Optional[Path],
    outputs_dir: Optional[Path],
) -> PathConfig:
    base = Path.cwd()
    paths = PathConfig(
        base_dir=base,
        data_dir=data_dir or (base / "data"),
        papers_dir=papers_dir or (base / "papers"),
        outputs_dir=outputs_dir or (base / "outputs"),
    )
    ensure_directories(paths)
    return paths


@app.command("download-pdfs")
def download_pdfs_cmd(
    data_dir: Optional[Path] = typer.Option(
        None,
        help="Directory containing metadata.csv / train_QA.csv / test_Q.csv (default: ./data)",
    ),
    papers_dir: Optional[Path] = typer.Option(
        None,
        help="Directory to store downloaded PDFs (default: ./papers)",
    ),
    outputs_dir: Optional[Path] = typer.Option(
        None,
        help="Root directory for outputs (default: ./outputs)",
    ),
) -> None:
    """Download all PDFs according to metadata.csv."""

    from .ingest.download_pdfs import download_pdfs_from_metadata

    paths = _build_paths(data_dir, papers_dir, outputs_dir)
    stats = download_pdfs_from_metadata(paths)
    typer.echo(f"✅ PDF 下載完成: 成功 {stats.success}, 失敗 {stats.fail}")
    if stats.failed_ids:
        typer.echo(f"❌ 失敗清單: {', '.join(stats.failed_ids)}")


@app.command("build-chunks")
def build_chunks_cmd(
    data_dir: Optional[Path] = typer.Option(
        None,
        help="Data directory (default: ./data)",
    ),
    papers_dir: Optional[Path] = typer.Option(
        None,
        help="PDF directory (default: ./papers)",
    ),
    outputs_dir: Optional[Path] = typer.Option(
        None,
        help="Root directory for outputs (default: ./outputs)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Rebuild chunks even if a chunks file already exists",
    ),
) -> None:
    """Use Docling + OCR to convert PDFs into chunks JSONL."""

    from .ingest.parse_docling_ocr import build_chunks_from_papers

    paths = _build_paths(data_dir, papers_dir, outputs_dir)
    out_path = build_chunks_from_papers(paths, force=force)
    typer.echo(f"✅ Chunks 產生完成: {out_path}")


@app.command("build-index")
def build_index_cmd(
    data_dir: Optional[Path] = typer.Option(
        None,
        help="Data directory (default: ./data)",
    ),
    papers_dir: Optional[Path] = typer.Option(
        None,
        help="PDF directory (default: ./papers)",
    ),
    outputs_dir: Optional[Path] = typer.Option(
        None,
        help="Root directory for outputs (default: ./outputs)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Rebuild indexes even if they already exist",
    ),
) -> None:
    """Build BM25 + dense (FAISS) indexes from chunks."""

    from .index.build_index import build_indexes

    paths = _build_paths(data_dir, papers_dir, outputs_dir)
    model_cfg = ModelConfig()
    build_indexes(paths, model_cfg, force=force)
    typer.echo("✅ 索引建立完成")


@app.command("evaluate")
def evaluate_cmd(
    data_dir: Optional[Path] = typer.Option(
        None,
        help="Data directory (default: ./data)",
    ),
    papers_dir: Optional[Path] = typer.Option(
        None,
        help="PDF directory (default: ./papers)",
    ),
    outputs_dir: Optional[Path] = typer.Option(
        None,
        help="Root directory for outputs (default: ./outputs)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        help="Optional custom run name (default: timestamp)",
    ),
) -> None:
    """Evaluate Recall@k / MRR on train_QA and write failure cases."""

    from .evaluate.metrics import run_evaluation
    from .retrieve.adaptive import create_adaptive_retriever

    paths = _build_paths(data_dir, papers_dir, outputs_dir)
    retriever = create_adaptive_retriever(paths, ModelConfig())
    run_dir = run_evaluation(paths, retriever, run_name=run_name)
    typer.echo(f"✅ 評估完成，結果已寫入: {run_dir}")


@app.command("query")
def query_cmd(
    query: str = typer.Argument(..., help="User question"),
    unit: str = typer.Option(
        "is_blank",
        "--unit",
        help="Expected answer unit (e.g. tCO2e / MWh). Default: is_blank",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        help="Data directory (default: ./data)",
    ),
    papers_dir: Optional[Path] = typer.Option(
        None,
        help="PDF directory (default: ./papers)",
    ),
    outputs_dir: Optional[Path] = typer.Option(
        None,
        help="Root directory for outputs (default: ./outputs)",
    ),
    k: int = typer.Option(5, help="Number of top retrieval results to display"),
) -> None:
    """Run retrieval (RRF + optional rerank) for a single query and print the top-k chunks."""

    from .retrieve.adaptive import create_adaptive_retriever

    paths = _build_paths(data_dir, papers_dir, outputs_dir)
    retriever = create_adaptive_retriever(paths, ModelConfig())
    out = retriever.adaptive_search_final(query, answer_unit=unit, initial_k=100, final_k=k)

    typer.echo(f"Strategy: {out['strategy']}, used_reranking={out['used_reranking']}")
    if out["unanswerable"]["is_unanswerable"]:
        typer.echo(f"Predicted unanswerable: {out['unanswerable']['reason']}")
        raise typer.Exit(code=0)

    for idx, r in enumerate(out["results"], start=1):
        meta = r.get("metadata") or {}
        page = meta.get("page", "?")
        typer.echo("=" * 80)
        typer.echo(f"[{idx}] doc_id={r['doc_id']}  type={r.get('type','text')}  page={page}")
        typer.echo(f"score={r.get('score'):.4f}  ce_score={r.get('ce_score', float('nan')):.4f}")
        snippet = r["content"].replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        typer.echo(snippet)

