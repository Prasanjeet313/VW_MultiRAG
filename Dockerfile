# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — VW Audit Multi-Modal Agentic RAG
#
# This container runs the Streamlit app.
# Ollama (LLM server) must run on the HOST or as a sidecar container.
# See docker-compose.yml for the full stack.
#
# Build:
#   docker build -t vw-rag .
#
# Run (standalone, Ollama on host):
#   docker run -p 8501:8501 \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     -v $(pwd)/Volkswagon_Audit_Report_Doc:/app/Volkswagon_Audit_Report_Doc:ro \
#     -v vw_outputs:/app/outputs \
#     vw-rag
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System deps needed by PyMuPDF / Pillow ───────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────
WORKDIR /app

# ── Install Python deps first (layer-cache friendly) ─────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project source ──────────────────────────────────
COPY . .

# ── Create runtime output directories ───────────────────
RUN mkdir -p outputs/chroma_db outputs/images outputs/kg \
             Volkswagon_Audit_Report_Doc DATA_Inputs_Files

# ── Streamlit config ─────────────────────────────────────
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n' \
    > /root/.streamlit/config.toml

# ── Expose Streamlit port ────────────────────────────────
EXPOSE 8501

# ── Default: run the Streamlit app ───────────────────────
# To run ingestion first:
#   docker run --entrypoint python vw-rag ingest.py
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
