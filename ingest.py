"""
ingest.py — One-shot ingestion pipeline.

Run once before starting the Streamlit app:
    python ingest.py

What it does
────────────
1.  Discovers all PDFs in Volkswagon_Audit_Report_Doc/ (and optionally DATA_Inputs_Files/).
2.  Extracts text chunks and images via PyMuPDF.
3.  Captions every image / table via LLaVA.
4.  Embeds all text + caption chunks into ChromaDB.
5.  Builds the NetworkX knowledge graph from text chunks.

Idempotent: already-captioned images are skipped; ChromaDB upsert is safe to
re-run (existing IDs are overwritten with fresh embeddings if needed).

Usage
─────
    # Primary document only:
    python ingest.py

    # Include the extra reference PDFs in DATA_Inputs_Files/:
    python ingest.py --all
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_DIR, EXTRA_DATA_DIR
from src.pdf_processor   import process_pdf
from src.image_captioner import caption_images
from src.vector_store    import build_store, collection_size
from src.knowledge_graph import build_graph


def discover_pdfs(include_extra: bool = False) -> list[Path]:
    """Return a list of PDF paths to ingest."""
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if include_extra:
        pdfs += list(EXTRA_DATA_DIR.glob("*.pdf"))
    # Deduplicate by name (same file in both folders)
    seen: set = set()
    unique = []
    for p in pdfs:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    return unique


def ingest_pdf(pdf_path: Path) -> tuple[list, list]:
    """Process one PDF, caption its images, return (text_chunks, caption_chunks)."""
    print(f"\n{'='*60}")
    print(f"  Ingesting: {pdf_path.name}")
    print(f"{'='*60}")

    t0 = time.time()
    text_chunks, image_metas = process_pdf(str(pdf_path))
    print(f"  Extraction: {time.time()-t0:.1f}s")

    t1 = time.time()
    caption_chunks = caption_images(image_metas)
    print(f"  Captioning: {time.time()-t1:.1f}s")

    return text_chunks, caption_chunks


def main():
    parser = argparse.ArgumentParser(description="VW Audit RAG — ingestion pipeline")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also ingest PDFs from DATA_Inputs_Files/ (reference corpus)",
    )
    parser.add_argument(
        "--skip-kg",
        action="store_true",
        help="Skip knowledge-graph construction (faster, but KG retrieval disabled)",
    )
    parser.add_argument(
        "--kg-only",
        action="store_true",
        help="Re-build ONLY the knowledge graph (skip PDF extraction, captioning, embedding)",
    )
    args = parser.parse_args()

    pdfs = discover_pdfs(include_extra=args.all)
    if not pdfs:
        print("[ingest] No PDFs found. Check DATA_DIR in src/config.py")
        sys.exit(1)

    # ── KG-only fast path ─────────────────────────────────
    if args.kg_only:
        # Remove stale / incomplete graph so we start clean
        from src.config import KG_PATH
        kg_file = Path(KG_PATH)
        if kg_file.exists():
            kg_file.unlink()
            print(f"[ingest] Removed old graph: {KG_PATH}")

        print("\n[ingest] --kg-only mode: re-extracting text only (no captioning/embedding) …")
        all_text_chunks = []
        for pdf_path in pdfs:
            print(f"  Reading text from {pdf_path.name} …")
            text_chunks, _ = process_pdf(str(pdf_path))
            all_text_chunks.extend(text_chunks)
        print(f"\n[ingest] Building knowledge graph from {len(all_text_chunks)} text chunks …")
        t0 = time.time()
        g = build_graph(all_text_chunks)
        print(f"  KG build: {time.time()-t0:.1f}s | "
              f"nodes={g.number_of_nodes()}, edges={g.number_of_edges()}")
        print("\n[ingest] ✅ Knowledge graph rebuild complete.")
        return

    print(f"\n[ingest] Found {len(pdfs)} PDF(s) to ingest: {[p.name for p in pdfs]}")

    all_text_chunks: list = []
    all_chunks: list      = []   # text + caption chunks for ChromaDB

    for pdf_path in pdfs:
        text_chunks, caption_chunks = ingest_pdf(pdf_path)
        all_text_chunks.extend(text_chunks)
        all_chunks.extend(text_chunks)
        all_chunks.extend(caption_chunks)

    # ── Embed into ChromaDB ───────────────────────────────
    print(f"\n[ingest] Embedding {len(all_chunks)} chunks into ChromaDB …")
    t2 = time.time()
    build_store(all_chunks)
    print(f"  Embedding: {time.time()-t2:.1f}s | store size: {collection_size()}")

    # ── Build Knowledge Graph ─────────────────────────────
    if not args.skip_kg:
        print(f"\n[ingest] Building knowledge graph from "
              f"{len(all_text_chunks)} text chunks …")
        t3 = time.time()
        g = build_graph(all_text_chunks)
        print(f"  KG build: {time.time()-t3:.1f}s | "
              f"nodes={g.number_of_nodes()}, edges={g.number_of_edges()}")
    else:
        print("[ingest] Skipping knowledge graph (--skip-kg flag set)")

    print("\n[ingest] ✅ Ingestion complete. You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
