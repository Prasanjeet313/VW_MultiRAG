"""
retriever.py — Hybrid retriever combining vector search + knowledge-graph augmentation.

For every incoming query (or list of sub-queries):
  1.  Vector search each sub-query against ChromaDB.
  2.  Extract entities from the original query via the KG module.
  3.  Fetch KG-supporting chunk_ids and retrieve those chunks from ChromaDB.
  4.  Merge + deduplicate by chunk_id.
  5.  Extract any image_ids found in the returned chunks.

Returns
-------
{
  "chunks":    list[dict],   # merged, deduplicated chunks
  "image_ids": list[str],    # image_ids referenced in caption chunks
}
"""

from typing import List, Dict, Tuple

from src import vector_store, knowledge_graph
from src.config import RETRIEVAL_K


def _merge_unique(lists_of_chunks: List[List[Dict]]) -> List[Dict]:
    """Merge multiple chunk lists, keeping the first occurrence of each chunk_id."""
    seen: set = set()
    merged: List[Dict] = []
    for chunk_list in lists_of_chunks:
        for c in chunk_list:
            cid = c.get("chunk_id", "")
            if cid not in seen:
                seen.add(cid)
                merged.append(c)
    return merged


def retrieve(
    queries: List[str],
    original_query: str,
    k: int = RETRIEVAL_K,
) -> Dict:
    """
    Hybrid retrieval over all sub-queries + KG augmentation.

    Parameters
    ----------
    queries        : list of (possibly rewritten) sub-queries
    original_query : the raw user question (used for KG entity extraction)
    k              : top-k per sub-query
    """
    all_batches: List[List[Dict]] = []

    # 1. Vector search for every sub-query
    for q in queries:
        hits = vector_store.query_store(q, k=k)
        all_batches.append(hits)

    # 2. KG augmentation
    kg_ids = knowledge_graph.get_kg_chunk_ids(original_query)
    if kg_ids:
        kg_chunks = vector_store.get_chunks_by_ids(kg_ids)
        if kg_chunks:
            all_batches.append(kg_chunks)

    # 3. Merge
    merged = _merge_unique(all_batches)

    # 4. Collect referenced image_ids
    image_ids = [
        c["image_id"]
        for c in merged
        if c.get("type") == "image_caption" and c.get("image_id")
    ]

    return {
        "chunks"   : merged,
        "image_ids": list(dict.fromkeys(image_ids)),   # preserve order, dedupe
    }


def format_context(chunks: List[Dict]) -> str:
    """
    Render retrieved chunks as a single readable context string
    to pass into an LLM prompt.
    """
    parts = []
    for i, c in enumerate(chunks, start=1):
        source = f"[{c.get('source_doc','?')}, p.{c.get('page_num','?')}]"
        t      = c.get("type", "text")
        if t == "image_caption":
            header = f"--- Visual {i} {source} ---"
        else:
            header = f"--- Chunk {i} {source} ---"
        parts.append(f"{header}\n{c['text']}\n")
    return "\n".join(parts)
