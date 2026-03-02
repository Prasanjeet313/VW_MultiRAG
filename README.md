# VW Audit Multi-Modal Agentic RAG

A fully local, multi-modal Retrieval-Augmented Generation (RAG) system for Volkswagen audit documents — built with **LangChain**, **LangGraph**, **ChromaDB**, **Ollama**, and **Streamlit**.

![Volkswagen](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Volkswagen_logo_2019.svg/80px-Volkswagen_logo_2019.svg.png)

---

## Architecture Overview

```
PDF Documents
     │
     ▼
PDF Processor (PyMuPDF)
  ├── Text Chunks ──────────────────────────┐
  └── Images / Tables                       │
          │                                 │
          ▼                                 │
  Image Captioner (LLaVA:13b)               │
          │ Caption Chunks                  │
          └─────────────────────────────────┤
                                            ▼
                                    ChromaDB Vector Store
                                    (nomic-embed-text embeddings)
                                            │
                                            ▼
                               Knowledge Graph (NetworkX)
                                            │
                                            ▼
                                LangGraph Agentic Pipeline
                                  ├── Query Rewriter (Gemma3)
                                  ├── Hybrid Retriever (Vector + KG BFS)
                                  ├── Answer Generator (LLaVA:13b)
                                  └── Validator / Grader (Gemma3)
                                            │
                                            ▼
                                  Streamlit Dashboard
```

---

## Models (all local via Ollama)

| Role | Model |
|------|-------|
| Text Embeddings | `nomic-embed-text` |
| Vision / Caption / Answer | `llava:13b` |
| Query Rewriting / KG / Validation | `gemma3` |

---

## Quick Start — Docker (Recommended)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- At least **20 GB free disk space** (models + data)
- At least **16 GB RAM** recommended

### 1. Clone the repo

```bash
git clone https://github.com/Prasanjeet313/VW_MultiRAG.git
cd VW_MultiRAG
```

### 2. Add your PDF documents

Place your Volkswagen audit PDFs inside the `Volkswagon_Audit_Report_Doc/` folder.  
The `Nonfinancial_Report_2022_en.pdf` is already included.

### 3. Start the full stack

```bash
docker compose up
```

> First run takes **15–30 minutes** to pull Ollama models (`nomic-embed-text`, `llava:13b`, `gemma3`) and ingest documents. Subsequent starts are fast — models and data are persisted in Docker volumes.

### 4. Open the app

```
http://localhost:8501
```

### Useful Docker commands

```bash
# Run in detached mode
docker compose up -d

# Check logs
docker compose logs -f app

# Re-ingest documents (e.g. after adding new PDFs)
docker compose run --rm ingest

# Stop everything
docker compose down

# Full reset (removes all data volumes)
docker compose down -v
```

---

## Quick Start — Local (No Docker)

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running

### 1. Clone & install

```bash
git clone https://github.com/Prasanjeet313/VW_MultiRAG.git
cd VW_MultiRAG

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull Ollama models

```bash
ollama pull nomic-embed-text
ollama pull llava:13b
ollama pull gemma3
```

### 3. Ingest documents

```bash
# Primary VW audit document only:
python ingest.py

# Include reference PDFs in DATA_Inputs_Files/:
python ingest.py --all

# Skip knowledge-graph build (faster, KG retrieval disabled):
python ingest.py --skip-kg
```

### 4. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
├── app.py                          # Streamlit dashboard
├── ingest.py                       # One-shot document ingestion pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Volkswagon_Audit_Report_Doc/    # Primary audit PDF(s) — source documents
├── DATA_Inputs_Files/              # Optional reference corpus (not committed)
├── outputs/
│   ├── chroma_db/                  # ChromaDB vector store (generated)
│   ├── images/                     # Extracted PDF images (generated)
│   └── kg/                         # Serialised knowledge graph (generated)
└── src/
    ├── config.py                   # Central config — paths, model names, hyper-params
    ├── pdf_processor.py            # PyMuPDF text + image extraction
    ├── image_captioner.py          # LLaVA image captioning
    ├── vector_store.py             # ChromaDB build & query
    ├── knowledge_graph.py          # NetworkX KG build, BFS retrieval
    ├── retriever.py                # Hybrid retriever (vector + KG)
    ├── rag_pipeline.py             # LangGraph agentic pipeline
    ├── tools.py                    # LangChain tools for the agent
    └── validator.py                # Answer grading / validation
```

---

## Configuration

All model names, paths, and hyper-parameters are in [src/config.py](src/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CAPTION_MODEL` | `llava:13b` | Multimodal model for captions & answers |
| `REASONING_MODEL` | `gemma3` | Query rewriting, KG extraction, validation |
| `CHUNK_SIZE` | `800` | Characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Chunk overlap in characters |
| `RETRIEVAL_K` | `6` | Top-K chunks retrieved per sub-query |
| `MAX_RETRIES` | `5` | Max LangGraph retry loops |
| `KG_SAMPLE_EVERY` | `15` | Extract KG triples from every Nth chunk |
| `KG_BFS_DEPTH` | `2` | BFS depth for KG-augmented retrieval |

---

## Ingestion Flags

```
python ingest.py [--all] [--skip-kg] [--kg-only]

  --all        Also ingest PDFs from DATA_Inputs_Files/ (reference corpus)
  --skip-kg    Skip knowledge-graph construction (faster startup)
  --kg-only    Rebuild ONLY the knowledge graph (no re-embedding)
```

---

## License

MIT
