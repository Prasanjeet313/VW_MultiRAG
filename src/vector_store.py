"""
vector_store.py — Persistent ChromaDB store backed by Ollama embeddings.

Collections:
  vw_audit_docs — stores BOTH text chunks and image caption chunks.
     metadata per document: chunk_id, source_doc, page_num, type, image_id (optional)

Public API
----------
build_store(chunks)         — embed + upsert a list of chunk dicts
query_store(query, k)       — return top-k (text, metadata, distance) tuples
get_chunk_by_id(chunk_id)   — fetch a single chunk by its deterministic id
collection_size()           — number of documents currently stored
"""

from typing import List, Dict, Tuple, Optional

import chromadb
from langchain_ollama import OllamaEmbeddings

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    RETRIEVAL_K,
)


# ── Singleton handles ─────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection]          = None
_embedder: Optional[OllamaEmbeddings]               = None


def _get_embedder() -> OllamaEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return _embedder


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection    = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Build / Upsert ────────────────────────────────────────

def build_store(chunks: List[Dict], batch_size: int = 32) -> None:
    """
    Embed and upsert chunks into ChromaDB.
    Safe to call multiple times — existing IDs are overwritten.

    chunks: list of dicts with keys chunk_id, text, source_doc, page_num, type.
             image caption chunks additionally carry image_id.
    """
    coll    = _get_collection()
    embedder = _get_embedder()

    print(f"[vector_store] embedding {len(chunks)} chunks …")
    for i in range(0, len(chunks), batch_size):
        batch  = chunks[i : i + batch_size]
        texts  = [c["text"] for c in batch]
        ids    = [c["chunk_id"] for c in batch]
        metas  = [
            {
                "source_doc": c.get("source_doc", ""),
                "page_num"  : str(c.get("page_num", 0)),
                "type"      : c.get("type", "text"),
                "image_id"  : c.get("image_id", ""),
            }
            for c in batch
        ]
        embeddings = embedder.embed_documents(texts)
        coll.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
        )
        print(f"  upserted batch {i // batch_size + 1} "
              f"({min(i + batch_size, len(chunks))}/{len(chunks)})")

    print(f"[vector_store] store now has {coll.count()} documents")


# ── Query ─────────────────────────────────────────────────

def query_store(
    query: str,
    k: int = RETRIEVAL_K,
    filter_type: Optional[str] = None,
) -> List[Dict]:
    """
    Semantic search.
    Returns list of {text, chunk_id, source_doc, page_num, type, image_id, distance}.
    Optionally filter by metadata type ('text' | 'image_caption').
    """
    coll     = _get_collection()
    embedder = _get_embedder()
    q_embed  = embedder.embed_query(query)

    where = {"type": filter_type} if filter_type else None

    results = coll.query(
        query_embeddings=[q_embed],
        n_results=min(k, coll.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    # ids are always returned by ChromaDB regardless of the include list
    ids = results.get("ids", [[]])[0]
    for idx, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        # chunk_id is stored as the ChromaDB document ID, NOT in metadata
        chunk_id = ids[idx] if idx < len(ids) else meta.get("chunk_id", "")
        output.append({
            "text"      : doc,
            "chunk_id"  : chunk_id,
            "source_doc": meta.get("source_doc", ""),
            "page_num"  : int(meta.get("page_num", 0)),
            "type"      : meta.get("type", "text"),
            "image_id"  : meta.get("image_id", ""),
            "distance"  : round(float(dist), 4),
        })
    return output


def get_chunks_by_ids(chunk_ids: List[str]) -> List[Dict]:
    """Fetch specific chunks by their IDs."""
    coll = _get_collection()
    if not chunk_ids:
        return []
    try:
        results = coll.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        output = []
        for doc, meta, cid in zip(
            results["documents"],
            results["metadatas"],
            results["ids"],
        ):
            output.append({
                "text"      : doc,
                "chunk_id"  : cid,
                "source_doc": meta.get("source_doc", ""),
                "page_num"  : int(meta.get("page_num", 0)),
                "type"      : meta.get("type", "text"),
                "image_id"  : meta.get("image_id", ""),
                "distance"  : 0.0,
            })
        return output
    except Exception:
        return []


def collection_size() -> int:
    return _get_collection().count()
