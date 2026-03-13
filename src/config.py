"""
config.py — Central configuration for VW Audit RAG POC.
All paths, model names, and hyper-parameters live here.

To change models, edit the values in this file only — no other file needs touching.
"""

from pathlib import Path
import os

# ── Directory Layout ──────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "Volkswagon_Audit_Report_Doc"
EXTRA_DATA_DIR  = ROOT_DIR / "DATA_Inputs_Files"
OUTPUTS_DIR     = ROOT_DIR / "outputs"
IMAGES_DIR      = OUTPUTS_DIR / "images"
CHROMA_DIR      = OUTPUTS_DIR / "chroma_db"
KG_DIR          = OUTPUTS_DIR / "kg"

DB_DIR          = ROOT_DIR / "db"
DB_PATH         = str(DB_DIR  / "image_registry.db")
KG_PATH         = str(KG_DIR  / "graph.pkl")

# Create output directories on import
for _d in [IMAGES_DIR, CHROMA_DIR, KG_DIR, DB_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════
# MODEL CONFIGURATION  ← change models here
# ══════════════════════════════════════════════════════════

# ── Embedding Provider ────────────────────────────────────
# "huggingface" : runs locally via sentence-transformers, no Ollama needed (~90 MB)
# "ollama"      : uses an Ollama-hosted embedding model (requires ollama pull <model>)
EMBEDDING_PROVIDER = "huggingface"

# ── Embedding Model ───────────────────────────────────────
# Used when EMBEDDING_PROVIDER = "huggingface":
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"          # 90 MB, 384-dim, fast & accurate
# EMBEDDING_MODEL  = "all-mpnet-base-v2"        # 420 MB, 768-dim, higher quality

# Used when EMBEDDING_PROVIDER = "ollama"  (ollama pull <model> required):
# EMBEDDING_MODEL  = "nomic-embed-text"          # 550 MB, 768-dim
# EMBEDDING_MODEL  = "mxbai-embed-large"         # 670 MB, 1024-dim, best Ollama option

# ── Image Captioning Model (Ollama) ──────────────────────
# Used during ingestion only — captions PDF images for the vector store.
# ollama pull <model>  before use
CAPTION_MODEL    = "moondream"                  # 1.8B  ~1.1 GB — lightest viable vision model
# CAPTION_MODEL  = "llava-phi3"                 # 3.8B  ~2.9 GB — better quality, still light
# CAPTION_MODEL  = "llava:7b"                   # 7B    ~4.7 GB — highest quality captions

# ── Answer Generation Model (Ollama) ─────────────────────
# Used at query time — receives text context + retrieved images, streams the final answer.
# Can be a stronger model than CAPTION_MODEL since it's the user-facing output.
# ollama pull <model>  before use
ANSWER_MODEL     = "llava-phi3"                 # 3.8B  ~2.9 GB — good quality/speed balance
# ANSWER_MODEL   = "llava:7b"                   # 7B    ~4.7 GB — best answer quality
# ANSWER_MODEL   = "moondream"                  # 1.8B  ~1.1 GB — fastest, lower quality

# ── Reasoning Model (Ollama) ──────────────────────────────
# Used for: query rewriting, context summarisation, context validation, answer grading
# ollama pull <model>  before use
REASONING_MODEL  = "mistral"                   # 7B   ~4.1 GB — reliable JSON-structured output
# REASONING_MODEL  = "llama3.2:3b"             # 3B   ~2.0 GB — lighter, slightly less reliable on JSON

# ── Ollama Server URL ─────────────────────────────────────
# Local: Ollama runs on its default port 11434.
# The ollama SDK reads OLLAMA_HOST automatically; we also expose it for langchain-ollama.
OLLAMA_BASE_URL  = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ══════════════════════════════════════════════════════════
# PIPELINE PARAMETERS  ← tune retrieval / chunking here
# ══════════════════════════════════════════════════════════

# ── Chunking ─────────────────────────────────────────────
CHUNK_SIZE       = 800     # characters
CHUNK_OVERLAP    = 100     # characters

# ── Retrieval ────────────────────────────────────────────
RETRIEVAL_K      = 6       # top-k chunks per sub-query
MAX_RETRIES      = 5       # max retry loops in LangGraph pipeline

# ── ChromaDB ─────────────────────────────────────────────
COLLECTION_NAME  = "vw_audit_docs"

# ── Knowledge-Graph extraction ───────────────────────────
KG_SAMPLE_EVERY  = 15      # extract triples from every Nth chunk (speed vs coverage)
KG_BFS_DEPTH     = 2       # BFS depth when fetching KG-augmented chunks

# ── Image Captioning ─────────────────────────────────────
MIN_IMAGE_SIZE   = 50      # ignore images smaller than this in pixels (w or h)
