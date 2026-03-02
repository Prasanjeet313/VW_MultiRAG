"""
config.py — Central configuration for VW Audit RAG POC.
All paths, model names, and hyper-parameters live here.
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

DB_PATH         = str(ROOT_DIR / "image_registry.db")
KG_PATH         = str(KG_DIR  / "graph.pkl")

# Create output directories on import
for _d in [IMAGES_DIR, CHROMA_DIR, KG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Ollama Model Names ─────────────────────────────────────
# Pull with:  ollama pull <model>
EMBEDDING_MODEL  = "mxbai-embed-large"  # text embeddings (higher MTEB accuracy)
CAPTION_MODEL    = "llama3.2-vision"    # multimodal – captions & final answer
REASONING_MODEL  = "mistral"            # query rewriting, validation, KG triples
VALIDATOR_MODEL  = "mistral"            # answer-grading judge (same model, different prompt)

# ── Ollama Server URL ─────────────────────────────────────
# In Docker: set OLLAMA_HOST=http://ollama:11434 via docker-compose env.
# The ollama SDK reads this automatically; we also expose it for langchain-ollama.
OLLAMA_BASE_URL  = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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
