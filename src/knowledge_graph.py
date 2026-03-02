"""
knowledge_graph.py — Build + query a NetworkX knowledge graph from chunked text.

Design
------
For every Nth text chunk (KG_SAMPLE_EVERY), we ask Mistral to extract
(subject, predicate, object) triples.  Each triple becomes an edge in a
directed graph where nodes are entity strings.

Every edge carries:
  - predicate  : relationship string
  - chunk_ids  : list of chunk IDs that support this triple
  - page_nums  : list of page numbers

During retrieval, given query entities we BFS-traverse the graph up to
KG_BFS_DEPTH hops and collect all supporting chunk_ids, which are then
fetched from ChromaDB to augment vector retrieval.

Persistence
-----------
The graph is pickled to outputs/kg/graph.pkl and loaded on demand.
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple

import networkx as nx
import ollama

from src.config import (
    REASONING_MODEL,
    KG_PATH,
    KG_SAMPLE_EVERY,
    KG_BFS_DEPTH,
)


# ── Internal graph state ──────────────────────────────────
_graph: nx.DiGraph | None = None


def _get_graph() -> nx.DiGraph:
    global _graph
    if _graph is None:
        if Path(KG_PATH).exists():
            with open(KG_PATH, "rb") as f:
                _graph = pickle.load(f)
        else:
            _graph = nx.DiGraph()
    return _graph


def _save_graph(g: nx.DiGraph) -> None:
    with open(KG_PATH, "wb") as f:
        pickle.dump(g, f)


# ── Triple Extraction (batched) ───────────────────────────
# We send BATCH_SIZE chunks in ONE Mistral call instead of one call per chunk.
# This cuts LLM calls by ~5× with negligible quality loss.

BATCH_SIZE = 5

_BATCH_PROMPT = """You are an expert in corporate sustainability and audit reporting.
Below are {n} numbered text passages from a Volkswagen Nonfinancial Report.
For EACH passage extract factual (subject, predicate, object) triples.
Focus on: organizations, departments, products, targets, regulations, KPIs,
emissions, ESG metrics, countries, standards, and audit findings.

Return ONLY a single valid JSON object where each key is the passage number
and each value is an array of triple objects:
{{
  "1": [{{"subject": "...", "predicate": "...", "object": "..."}}],
  "2": [],
  ...
}}
If a passage has no relevant triples, use an empty array.

{passages}
"""


def _parse_triples_from_raw(raw: str, n: int) -> dict:
    """Robustly extract the JSON object from an LLM response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return {str(k): v for k, v in data.items()}
    except Exception:
        return {}


def _extract_triples_batch(
    chunks: List[Dict],
) -> List[List[Tuple[str, str, str]]]:
    """
    Send a batch of chunks to Mistral in ONE call.
    Returns a list (same length as chunks) of triple lists.
    """
    passages = "\n\n".join(
        f"[{i+1}]\n{c['text'][:600]}" for i, c in enumerate(chunks)
    )
    prompt = _BATCH_PROMPT.format(n=len(chunks), passages=passages)

    try:
        response = ollama.chat(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        raw    = response["message"]["content"].strip()
        parsed = _parse_triples_from_raw(raw, len(chunks))
    except Exception:
        parsed = {}

    results = []
    for i in range(len(chunks)):
        raw_list = parsed.get(str(i + 1), [])
        triples  = []
        if isinstance(raw_list, list):
            for t in raw_list:
                s = str(t.get("subject", "")).strip()
                p = str(t.get("predicate", "")).strip()
                o = str(t.get("object", "")).strip()
                if s and p and o:
                    triples.append((s, p, o))
        results.append(triples)
    return results


# ── Graph Construction ────────────────────────────────────

def build_graph(text_chunks: List[Dict]) -> nx.DiGraph:
    """
    Build the knowledge graph from text_chunks.
    Processes every KG_SAMPLE_EVERY-th chunk, batched BATCH_SIZE at a time,
    so total LLM calls ≈ (len(text_chunks) / KG_SAMPLE_EVERY / BATCH_SIZE).
    Merges with any existing graph on disk.
    """
    g = _get_graph()

    sampled = [c for i, c in enumerate(text_chunks) if i % KG_SAMPLE_EVERY == 0]
    total   = len(sampled)
    n_calls = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"[knowledge_graph] {total} sampled chunks → "
          f"{n_calls} batched LLM calls (batch size {BATCH_SIZE}) …")

    for batch_idx in range(0, total, BATCH_SIZE):
        batch   = sampled[batch_idx : batch_idx + BATCH_SIZE]
        call_no = batch_idx // BATCH_SIZE + 1
        print(f"  batch {call_no}/{n_calls}  "
              f"(chunks {batch_idx+1}–{min(batch_idx+BATCH_SIZE, total)}/{total}) …")

        all_triples = _extract_triples_batch(batch)

        for chunk, triples in zip(batch, all_triples):
            chunk_id = chunk["chunk_id"]
            page_num = chunk["page_num"]
            for (s, p, o) in triples:
                if g.has_edge(s, o):
                    existing = g[s][o]
                    existing.setdefault("chunk_ids", []).append(chunk_id)
                    existing.setdefault("page_nums", []).append(page_num)
                else:
                    g.add_edge(
                        s, o,
                        predicate  = p,
                        chunk_ids  = [chunk_id],
                        page_nums  = [page_num],
                    )

    _save_graph(g)
    global _graph
    _graph = g
    print(f"[knowledge_graph] graph: {g.number_of_nodes()} nodes, "
          f"{g.number_of_edges()} edges")
    return g


# ── Query Graph ───────────────────────────────────────────

def _extract_query_entities(query: str) -> List[str]:
    """Quick heuristic + LLM call to pull entity mentions from a query string."""
    prompt = (
        "Extract the key named entities from this audit query as a JSON list of strings. "
        "Focus on organisations, standards, KPIs, targets, products, regulations.\n"
        f"Query: {query}\n"
        "Respond ONLY with a JSON list, e.g. [\"entity1\", \"entity2\"]"
    )
    try:
        response = ollama.chat(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        raw   = response["message"]["content"].strip()
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    # Fallback: simple noun extraction via capitalised words
    return re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", query)


def get_kg_chunk_ids(query: str) -> List[str]:
    """
    Given a query, find related entities in the graph and return
    a deduplicated list of supporting chunk_ids via BFS traversal.
    """
    g        = _get_graph()
    entities = _extract_query_entities(query)
    if not entities or g.number_of_nodes() == 0:
        return []

    visited_nodes: Set[str] = set()
    chunk_ids: Set[str]     = set()

    for entity in entities:
        # Case-insensitive node lookup
        matched_nodes = [
            n for n in g.nodes
            if entity.lower() in n.lower() or n.lower() in entity.lower()
        ]
        for start_node in matched_nodes:
            try:
                bfs_nodes = nx.single_source_shortest_path_length(
                    g, start_node, cutoff=KG_BFS_DEPTH
                )
                for node in bfs_nodes:
                    if node in visited_nodes:
                        continue
                    visited_nodes.add(node)
                    # Collect chunk_ids from all outgoing edges
                    for _, _, edge_data in g.out_edges(node, data=True):
                        for cid in edge_data.get("chunk_ids", []):
                            chunk_ids.add(cid)
            except Exception:
                continue

    return list(chunk_ids)


def get_graph_summary() -> Dict:
    """Return basic stats about the current graph (for the Streamlit sidebar)."""
    g = _get_graph()
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "is_loaded": Path(KG_PATH).exists(),
    }


def get_subgraph_for_query(query: str, max_nodes: int = 40) -> nx.DiGraph:
    """Return a small subgraph relevant to the query — used by the KG visualiser."""
    g        = _get_graph()
    entities = _extract_query_entities(query)
    if not entities:
        return nx.DiGraph()

    relevant_nodes: Set[str] = set()
    for entity in entities:
        matched = [n for n in g.nodes if entity.lower() in n.lower()]
        for start in matched:
            try:
                nbrs = nx.single_source_shortest_path_length(g, start, cutoff=1)
                relevant_nodes.update(nbrs.keys())
            except Exception:
                pass
        if len(relevant_nodes) >= max_nodes:
            break

    sub_nodes = list(relevant_nodes)[:max_nodes]
    return g.subgraph(sub_nodes).copy()
