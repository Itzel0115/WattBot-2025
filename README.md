## WattBot 2025 – HERO RAG Pipeline

WattBot 2025 is a retrieval‑augmented QA competition built on an “AI x Energy / Environment” paper collection. This repository implements a fully reproducible, modular HERO RAG pipeline for retrieving and answering questions from multiple PDF reports and papers.

Key features:

- **Docling + OCR**: Use Docling for structured PDF parsing and EasyOCR for figures, so both **tables and figure text** are available to the retriever.
- **BM25 + Dense (FAISS)**: A strengthened BM25 baseline combined with SentenceTransformers embeddings and FAISS vector search.
- **Weighted RRF**: Weighted RRF fusion between BM25 and dense results, with extra boosts for **tables (×2)**, **image OCR (×1.8)**, and **metadata hints (author/year ×2)**.
- **CrossEncoder Reranker**: CrossEncoder re‑ranking on candidate chunks, runnable on CPU when GPU is not available.
- **Train/Eval Pipeline**: Evaluate on `train_QA.csv` with Recall@k / MRR and failure cases saved under `outputs/runs/`.
- **Optional LLM answers**: A Gemini client hook is provided for final answer generation using Google Gemini (only when `GOOGLE_API_KEY` is set).

---

## Project structure

```text
.
├─ README.md
├─ pyproject.toml
├─ .gitignore
├─ .env.example
├─ data/
│  ├─ metadata.csv        # paper metadata (id / url / year / ...)
│  ├─ train_QA.csv        # training QA with reference doc ids
│  └─ test_Q.csv          # test questions (no answers)
├─ papers/                # downloaded PDFs (created by CLI)
├─ outputs/
│  ├─ chunks/             # structured chunks (JSONL)
│  ├─ images/             # extracted images (for OCR)
│  ├─ indexes/            # BM25 / FAISS indexes and meta
│  └─ runs/               # evaluation results and failure cases
├─ docs/                  # project report / slides (PDFs, etc.)
├─ src/
│  └─ wattbot_rag/
│     ├─ __init__.py
│     ├─ __main__.py
│     ├─ config.py
│     ├─ ingest/
│     │  ├─ chunk_schema.py
│     │  ├─ download_pdfs.py
│     │  └─ parse_docling_ocr.py
│     ├─ index/
│     │  ├─ bm25.py
│     │  ├─ dense_faiss.py
│     │  └─ build_index.py
│     ├─ retrieve/
│     │  ├─ rrf.py
│     │  ├─ rerank.py
│     │  └─ adaptive.py
│     ├─ evaluate/
│     │  └─ metrics.py
│     ├─ llm/             # reserved for Gemini client and prompts
│     └─ cli.py
└─ notebooks/ (optional)
```

---

## Installation and system requirements

### Python version

- **Python 3.10+** (3.10 or 3.11 recommended)

### OS‑level dependencies (OCR / PDF)

To use the full Docling + EasyOCR pipeline (default), it is recommended to install:

- **Poppler** (for PDF‑to‑image and some Docling flows)
- **Tesseract OCR** (used in some OCR scenarios)

Suggested installation by platform:

- **Ubuntu / Debian**
  - `sudo apt-get update`
  - `sudo apt-get install -y poppler-utils tesseract-ocr`
- **macOS (Homebrew)**
  - `brew install poppler tesseract`
- **Windows**
  - Install Poppler (e.g. download a Windows binary and add `bin/` to PATH)
  - Install Tesseract (e.g. `tesseract-ocr-w64` and add its install path to PATH)

If you cannot install OCR‑related system packages, the project can still:

- Download PDFs
- Build chunks that are primarily **text + tables**
- Build **BM25 + dense index**
- Run `evaluate` / `query` (but **image_ocr content will be missing**)

### Install Python dependencies

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
```

This installs dependencies from `pyproject.toml` and registers the `wattbot-rag` CLI.

---

## Data preparation

Place the official CSV files under `data/` (if they are not there already):

- `data/metadata.csv`
- `data/train_QA.csv`
- `data/test_Q.csv`

This repo already ships with one copy of these three files at the root; if you cloned from a Colab export, make sure they are moved under `data/`.

CSV encodings:

- `metadata.csv` / `train_QA.csv`: **latin-1** by default
- `test_Q.csv`: **utf-8-sig** by default

These values are configurable in `config.py`.

---

## API keys and security

This repository **does not hard‑code any Gemini API keys**. Instead:

- All LLM components read the key from the **`GOOGLE_API_KEY`** environment variable.
- The project root includes a `.env.example` template.

### Setup steps

1. Copy `.env.example` to `.env` at the project root:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in your own Gemini API key:

   ```text
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

3. `.gitignore` already ignores `.env`, so your key will not be committed.

> If you only want to run **retrieval / evaluation** (no LLM answers), you can leave `GOOGLE_API_KEY` unset.

---

## Quickstart: download → chunks → index → evaluate

All commands below assume you are in the project root and have installed the package:

```bash
pip install -e .
```

### 1. Download PDFs

```bash
wattbot-rag download-pdfs \
  --data-dir ./data \
  --papers-dir ./papers \
  --outputs-dir ./outputs
```

This reads `data/metadata.csv` and downloads PDFs from the `url` column into `papers/`. Existing files are skipped. At the end you get a summary of successes/failures and a `failed_ids` list.

### 2. Build chunks (Docling + OCR)

```bash
wattbot-rag build-chunks \
  --data-dir ./data \
  --papers-dir ./papers \
  --outputs-dir ./outputs
```

Outputs:

- `outputs/chunks/structured_chunks_with_ocr.jsonl`
- `outputs/images/*.png`

Each chunk follows this schema:

- `doc_id`: corresponds to `metadata.csv.id`
- `type`: `"text" | "table" | "figure" | "image_ocr" ...`
- `content`: plain‑text content (for BM25 / dense / LLM)
- `metadata`: extra information (at least `page`, plus `image_path` for images)
- `page`: 1‑based page index in the source PDF
- `word_count`: word count for `content`
- `image_path`: local path to the saved image for `image_ocr` chunks

### 3. Build BM25 + dense index

```bash
wattbot-rag build-index \
  --data-dir ./data \
  --papers-dir ./papers \
  --outputs-dir ./outputs
```

Outputs:

- `outputs/indexes/bm25.pkl`
- `outputs/indexes/dense.index`
- `outputs/indexes/dense_config.json`
- `outputs/indexes/index_meta.json`

Embedding models:

- Default: **`BAAI/bge-large-en-v1.5`**
- Fallback on download / memory issues: **`sentence-transformers/all-MiniLM-L6-v2`**

### 4. Evaluate on `train_QA`

```bash
wattbot-rag evaluate \
  --data-dir ./data \
  --papers-dir ./papers \
  --outputs-dir ./outputs
```

Example output directory:

- `outputs/runs/run-20260304-123456/metrics.json`
- `outputs/runs/run-20260304-123456/failure_cases.json`

`metrics.json` includes:

- `recall@1`, `recall@3`, `recall@5`, `recall@10`
- `MRR`
- `total_questions`
- `rerank_usage`

`failure_cases.json` lists some of the hardest questions the retriever failed on, including:

- `question`
- `answer_unit`
- `true_docs`
- `retrieved` (top 5 `doc_id`s)
- `strategy` (e.g. Hybrid+BGE / BM25-only)

---

## Demo: interactive query examples

Once indexes are built, you can use the `query` command to retrieve for a single question:

```bash
wattbot-rag query "What is the name of the benchmark suite for measuring inference energy consumption?" \
  --unit is_blank \
  --data-dir ./data \
  --papers-dir ./papers \
  --outputs-dir ./outputs \
  --k 5
```

Or specify an expected unit to trigger **unit expansion**:

```bash
wattbot-rag query "What were the net CO2e emissions from training the GShard-600B model?" \
  --unit tCO2e
```

CLI output includes:

- Retrieval strategy and whether reranking was used.
- For the top‑k chunks:
  - `doc_id`
  - `type` (text / table / image_ocr / ...)
  - `page`
  - `score` (RRF score)
  - `ce_score` (CrossEncoder score, if used)
  - A short content snippet

---

## Evaluation and outputs

### Retrieval evaluation

The `evaluate` command uses the following pipeline:

1. **Weighted RRF**
   - BM25 with enhanced tokenization (keeps numbers, years, and longer words)
   - Dense retrieval via SentenceTransformers + FAISS (inner product)
   - Weights:
     - `type == "table"` → ×2.0
     - `type == "image_ocr"` → ×1.8
     - doc_id hinted by author/year in the query → ×2.0
2. **CrossEncoder reranker**
   - Default: `BAAI/bge-reranker-v2-m3`
   - Fallback: `cross-encoder/ms-marco-MiniLM-L-12-v2`
3. **Adaptive decision**
   - Query‑based heuristic (keywords like compare, percent, emission, table, etc.) controls reranking.
   - A lightweight unanswerable detector filters out clearly out‑of‑scope questions (e.g. distance to the sun, elephants).

Metrics:

- Recall@1 / 3 / 5 / 10
- MRR
- Reranking usage ratio
- Failure cases (top 10)

---

## Design highlights

- **Table paradox / image OCR**
  - Many key numeric facts live inside tables and figures, which pure text retrieval often misses.
  - This project converts Docling `TableItem` and `PictureItem` into chunks and runs EasyOCR on figures.
  - **Weighted RRF** then gives higher weight to `table` and `image_ocr` chunks, improving recall for numeric / metric questions.

- **Tokenization and unit expansion**
  - BM25 uses `enhanced_tokenize`, which pays special attention to numbers, years, and longer words.
  - If the question specifies a unit (e.g. `tCO2e`, `MWh`, `PUE`), `UNIT_KEYWORDS` appends semantic unit hints to the query to improve recall on that measurement.

- **Weighted RRF + metadata hints**
  - During RRF fusion we boost:
    - Chunk type (text / table / image_ocr)
    - Doc ids whose author/year match the query
  - This helps for questions that clearly refer to a particular paper or report.

- **Adaptive reranking**
  - Not every query needs CrossEncoder reranking (it is expensive).
  - A query‑based heuristic decides when to rerank, balancing efficiency and quality.

---

## LLM / Gemini (optional)

The original Colab notebook used multiple hard‑coded Gemini keys and rotated them. In this refactor:

- We read a **single key** from the `GOOGLE_API_KEY` env var.
- All LLM‑related logic is placed under `wattbot_rag/llm/` (currently as a minimal, extendable scaffold).

Suggested extension (if you want a full inference pipeline):

- Define WattBot‑specific prompts in `llm/prompts.py`.
- In `llm/gemini_client.py`:
  - Use `google.generativeai` and read the key via `config.get_google_api_key()`.
  - Expose a function like `generate_answer(question, unit, retrieval_results, doc_id_to_url)`.
  - Enforce a strict JSON schema (`answer / answer_value / answer_unit / ref_id / ref_url / supporting_material / explanation`).
- Add a CLI command (e.g. `wattbot-rag infer-test`) that iterates over `test_Q.csv` and produces a submission file.

---

## Developer notes and reuse tips

- All paths are managed through `PathConfig` and CLI flags (`--data-dir / --papers-dir / --outputs-dir`), making it easy to port this pipeline to other RAG projects.
- The chunk schema is a clear `dataclass`, so you can:
  - Swap in different PDF parsers or OCR engines.
  - Add richer metadata (e.g. section / caption / figure id).
- Indexes and intermediate artifacts all live under `outputs/`, and are ignored by `.gitignore` so they don’t pollute the Git history.

---

## Minimal reproducible flow (for reviewers)

If you are a GitHub reviewer and just want to see the pipeline run end‑to‑end, try:

```bash
git clone <this-repo>
cd <this-repo>

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# 1. Check that data/ contains the three CSVs (already included)
ls data

# 2. Download PDFs
wattbot-rag download-pdfs

# 3. Parse PDFs + OCR
wattbot-rag build-chunks

# 4. Build indexes
wattbot-rag build-index

# 5. Evaluate retrieval on train_QA
wattbot-rag evaluate

# 6. Run a few sample queries
wattbot-rag query "What is the name of the benchmark suite for measuring inference energy consumption?" --unit is_blank
```

This flow runs entirely on a local (non‑Colab) environment. As long as dependencies and minimal system tools are installed, you can fully reproduce **download → chunks → index → evaluate** for the HERO RAG pipeline. With `GOOGLE_API_KEY` configured, you can further extend it with LLM answer generation.

