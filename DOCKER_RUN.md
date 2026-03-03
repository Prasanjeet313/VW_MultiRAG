# Running VW Audit RAG on Docker

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed
- **16 GB RAM minimum** (32 GB recommended)
- **30 GB free disk** (for models + data)

---

## Option 1 — Docker Compose (Recommended)

This is the easiest way. It starts Ollama, runs ingestion, and launches the app automatically.

### First Run (pulls models + ingests documents)

```bash
git clone https://github.com/Prasanjeet313/VW_MultiRAG.git
cd VW_MultiRAG

docker compose up
```

> This will take **20–40 minutes** on the first run as it pulls ~15 GB of models (`mxbai-embed-large`, `llama3.2-vision`, `mistral`) and ingests the PDF.

### Subsequent Runs (models & data already cached)

```bash
docker compose up app
```

### Run in Background

```bash
docker compose up -d

# Follow logs
docker compose logs -f
```

### Access the App

```
http://localhost:8501
```

> **On a remote server:** replace `localhost` with your server's public IP and make sure port `8501` is open in your firewall.

---

## Option 2 — Manual Docker (Without Compose)

### Step 1 — Start Ollama

```bash
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_models:/root/.ollama \
  ollama/ollama
```

### Step 2 — Pull Models

```bash
docker exec ollama ollama pull mxbai-embed-large
docker exec ollama ollama pull llama3.2-vision
docker exec ollama ollama pull mistral
```

### Step 3 — Build the App Image

```bash
docker build -t vw-rag .
```

### Step 4 — Run Ingestion

```bash
docker run --rm \
  --name vw_ingest \
  --network host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/Volkswagon_Audit_Report_Doc:/app/Volkswagon_Audit_Report_Doc:ro \
  vw-rag python ingest.py
```

### Step 5 — Run the App

```bash
docker run -d \
  --name vw_app \
  --network host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -p 8501:8501 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/Volkswagon_Audit_Report_Doc:/app/Volkswagon_Audit_Report_Doc:ro \
  vw-rag
```

### Access the App

```
http://localhost:8501
```

---

## Useful Commands

```bash
# Check running containers
docker ps

# View app logs
docker logs -f vw_app

# View Ollama logs
docker logs -f vw_ollama

# Stop all services (Compose)
docker compose down

# Stop all services (manual)
docker stop vw_app ollama

# Remove containers
docker rm vw_app ollama

# Remove everything including volumes (WARNING: deletes models + data)
docker compose down -v
```

---

## On a Remote Server — Open Firewall Port

```bash
sudo ufw allow 8501/tcp
sudo ufw reload
```

Then access at:

```
http://<your-server-ip>:8501
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Port 8501 not accessible | Open port in firewall / security group |
| Ollama not reachable | Ensure `OLLAMA_HOST=http://ollama:11434` is set |
| Out of memory | Increase server RAM or use a smaller model in `src/config.py` |
| Ingestion takes too long | Wait — first run is slow due to model loading |
| Docker permission denied | Run `sudo usermod -aG docker $USER` then log out and back in |
