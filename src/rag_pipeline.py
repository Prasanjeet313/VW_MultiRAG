"""
rag_pipeline.py — LangGraph agentic pipeline for the VW Audit RAG POC.

Graph flow
──────────
START
  └─► query_rewriter
          └─► hybrid_retriever
                  └─► context_summarizer
                          └─► context_validator
                                  ├─► (valid OR retry ≥ 5) ─► answer_generator ─► END
                                  └─► (invalid, retry < 5) ─► query_rewriter (loop)

LLMs used
─────────
  • Mistral        — query rewriting, context summarisation, context validation
  • llama3.2-vision — final multi-modal answer (text context + actual image bytes)
"""

import base64
import json
import re
from pathlib import Path
from typing import TypedDict, List, Optional, Any

import ollama
from langgraph.graph import StateGraph, START, END

from src.config import (
    REASONING_MODEL,
    CAPTION_MODEL,
    MAX_RETRIES,
    RETRIEVAL_K,
)
from src.retriever import retrieve, format_context
from src.image_captioner import get_image_record


# ══════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════

class GraphState(TypedDict):
    original_query     : str
    rewritten_queries  : List[str]
    retrieved_chunks   : List[Any]
    retrieved_image_ids: List[str]
    retry_count        : int
    context_valid      : bool
    validator_feedback : str
    summarized_context : str
    final_answer       : str
    trace              : List[str]   # step-by-step log shown in UI


# ══════════════════════════════════════════════════════════
# Helper: call Ollama text model
# ══════════════════════════════════════════════════════════

def _llm(model: str, prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.3},
    )
    return response["message"]["content"].strip()


# ══════════════════════════════════════════════════════════
# Node 1 — Query Rewriter
# ══════════════════════════════════════════════════════════

_REWRITE_SYSTEM = (
    "You are an expert audit-document search specialist. "
    "Your job is to reformulate a user's audit question into multiple "
    "semantically varied sub-queries that will maximise recall from a "
    "vector store containing a Volkswagen Nonfinancial Report."
)

_REWRITE_PROMPT = """Original question: {query}

Validator feedback from previous attempt (empty on first try): {feedback}

Generate exactly 3 search sub-queries — each phrased differently, covering:
1. A direct keyword-heavy version
2. A conceptual / synonym-rich version
3. A specific data / metric-focused version

Respond ONLY with a valid JSON array of 3 strings, e.g.:
["query one", "query two", "query three"]
"""


def query_rewriter(state: GraphState) -> GraphState:
    query    = state["original_query"]
    feedback = state.get("validator_feedback", "")
    prompt   = _REWRITE_PROMPT.format(query=query, feedback=feedback)

    try:
        raw   = _llm(REASONING_MODEL, prompt, system=_REWRITE_SYSTEM)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        subs  = json.loads(match.group(0)) if match else [query]
    except Exception:
        subs = [query]

    trace_entry = f"🔎 Query rewriter (retry {state.get('retry_count', 0)}): {subs}"
    return {
        **state,
        "rewritten_queries": subs,
        "trace"            : state.get("trace", []) + [trace_entry],
    }


# ══════════════════════════════════════════════════════════
# Node 2 — Hybrid Retriever
# ══════════════════════════════════════════════════════════

def hybrid_retriever(state: GraphState) -> GraphState:
    result = retrieve(
        queries        = state["rewritten_queries"],
        original_query = state["original_query"],
        k              = RETRIEVAL_K,
    )
    trace_entry = (
        f"📦 Retriever: {len(result['chunks'])} chunks, "
        f"{len(result['image_ids'])} images found"
    )
    return {
        **state,
        "retrieved_chunks"   : result["chunks"],
        "retrieved_image_ids": result["image_ids"],
        "trace"              : state.get("trace", []) + [trace_entry],
    }


# ══════════════════════════════════════════════════════════
# Node 3 — Context Summariser
# ══════════════════════════════════════════════════════════

_SUMMARISE_SYSTEM = (
    "You are an expert audit analyst. Summarise the provided context "
    "passages into a concise, well-structured briefing that preserves "
    "all numbers, percentages, dates, and audit-relevant facts. "
    "Do NOT add information not present in the passages."
)

_SUMMARISE_PROMPT = """Question: {query}

Retrieved context:
{context}

Write a concise structured summary (max 600 words) that captures every fact
relevant to answering the question. Preserve all figures and source references."""


def context_summarizer(state: GraphState) -> GraphState:
    chunks  = state.get("retrieved_chunks", [])
    context = format_context(chunks)
    prompt  = _SUMMARISE_PROMPT.format(
        query  = state["original_query"],
        context= context,
    )
    summary = _llm(REASONING_MODEL, prompt, system=_SUMMARISE_SYSTEM)
    trace_entry = "📝 Context summarised"
    return {
        **state,
        "summarized_context": summary,
        "trace"             : state.get("trace", []) + [trace_entry],
    }


# ══════════════════════════════════════════════════════════
# Node 4 — Context Validator
# ══════════════════════════════════════════════════════════

_VALIDATE_SYSTEM = (
    "You are a strict audit quality controller. "
    "Evaluate whether the provided context is sufficient and relevant "
    "to answer the user's question accurately and completely."
)

_VALIDATE_PROMPT = """Question: {query}

Summarised context:
{context}

Is this context sufficient and relevant to answer the question?
Reply with a JSON object:
{{"valid": true/false, "reason": "brief explanation", "missing": "what is missing (if any)"}}
Respond ONLY with valid JSON."""


def context_validator(state: GraphState) -> GraphState:
    prompt = _VALIDATE_PROMPT.format(
        query  = state["original_query"],
        context= state.get("summarized_context", ""),
    )
    try:
        raw   = _llm(REASONING_MODEL, prompt, system=_VALIDATE_SYSTEM)
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        data  = json.loads(match.group(0)) if match else {"valid": True}
        valid    = bool(data.get("valid", True))
        feedback = data.get("reason", "") + " " + data.get("missing", "")
    except Exception:
        valid    = True
        feedback = ""

    retry_count = state.get("retry_count", 0)
    trace_entry = (
        f"✅ Context valid" if valid
        else f"⚠️ Context insufficient (retry {retry_count}): {feedback.strip()}"
    )
    return {
        **state,
        "context_valid"    : valid,
        "validator_feedback": feedback.strip(),
        "retry_count"      : retry_count,
        "trace"            : state.get("trace", []) + [trace_entry],
    }


# ══════════════════════════════════════════════════════════
# Conditional edge — retry router
# ══════════════════════════════════════════════════════════

def retry_router(state: GraphState) -> str:
    """
    Route to answer_generator if context is valid or max retries reached.
    Otherwise loop back to query_rewriter with incremented retry count.
    """
    if state.get("context_valid", False):
        return "answer"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "answer"   # best-effort answer with whatever we have
    # Increment retry count and loop
    state["retry_count"] = state.get("retry_count", 0) + 1
    return "retry"


# ══════════════════════════════════════════════════════════
# Node 5 — Answer Generator (Multi-modal LLaVA)
# ══════════════════════════════════════════════════════════

_ANSWER_SYSTEM = (
    "You are an expert Volkswagen Group audit analyst. "
    "Using the provided context and any visual evidence (charts, tables, diagrams), "
    "give a comprehensive, well-structured answer to the auditor's question. "
    "Cite specific numbers, page references, and KPIs where available. "
    "If the context is limited, say so explicitly and provide what you can."
)


def _load_image_b64(file_path: str) -> Optional[str]:
    try:
        return base64.b64encode(Path(file_path).read_bytes()).decode("utf-8")
    except Exception:
        return None


def answer_generator(state: GraphState) -> GraphState:
    query          = state["original_query"]
    context        = state.get("summarized_context", "")
    image_ids      = state.get("retrieved_image_ids", [])
    retry_count    = state.get("retry_count", 0)

    # Build image payloads for LLaVA (cap at 5 images to avoid OOM)
    b64_images = []
    image_annotations = []
    for img_id in image_ids[:5]:
        record = get_image_record(img_id)
        if record and Path(record["file_path"]).exists():
            b64 = _load_image_b64(record["file_path"])
            if b64:
                b64_images.append(b64)
                image_annotations.append(
                    f"[Image from page {record['page_num']}: {record['caption'][:200]}]"
                )

    # Compose prompt
    caveat = (
        "\n⚠️ NOTE: After 5 retrieval attempts, context may be incomplete.\n"
        if retry_count >= MAX_RETRIES
        else ""
    )

    image_note = ""
    if image_annotations:
        image_note = "\n\nAttached visuals:\n" + "\n".join(image_annotations)

    prompt = (
        f"{caveat}"
        f"Auditor question: {query}\n\n"
        f"Context:\n{context}"
        f"{image_note}\n\n"
        f"Provide a detailed, structured answer referencing specific data points "
        f"and pages where available."
    )

    messages: list = [
        {"role": "system", "content": _ANSWER_SYSTEM},
        {
            "role"   : "user",
            "content": prompt,
            **({"images": b64_images} if b64_images else {}),
        },
    ]

    try:
        response = ollama.chat(
            model=CAPTION_MODEL,   # llama3.2-vision — multimodal
            messages=messages,
            options={"temperature": 0.4, "num_ctx": 4096},
        )
        answer = response["message"]["content"].strip()
    except Exception as e:
        answer = f"[Answer generation failed: {e}]\n\nBest available context:\n{context}"

    img_note = f" (with {len(b64_images)} visuals)" if b64_images else ""
    trace_entry = f"💬 Answer generated{img_note}"
    return {
        **state,
        "final_answer": answer,
        "trace"       : state.get("trace", []) + [trace_entry],
    }


# ══════════════════════════════════════════════════════════
# Build LangGraph
# ══════════════════════════════════════════════════════════

def _build_graph():
    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("query_rewriter"    , query_rewriter)
    builder.add_node("hybrid_retriever"  , hybrid_retriever)
    builder.add_node("context_summarizer", context_summarizer)
    builder.add_node("context_validator" , context_validator)
    builder.add_node("answer_generator"  , answer_generator)

    # Edges
    builder.add_edge(START               , "query_rewriter")
    builder.add_edge("query_rewriter"    , "hybrid_retriever")
    builder.add_edge("hybrid_retriever"  , "context_summarizer")
    builder.add_edge("context_summarizer", "context_validator")

    builder.add_conditional_edges(
        "context_validator",
        retry_router,
        {
            "answer": "answer_generator",
            "retry" : "query_rewriter",
        },
    )

    builder.add_edge("answer_generator", END)

    return builder.compile()


# Module-level compiled graph (built once, reused)
_pipeline = None


def get_pipeline():
    """Return (and lazily build) the compiled LangGraph pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_graph()
    return _pipeline


# ══════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════

# Human-readable labels for each graph node
NODE_LABELS = {
    "query_rewriter"    : ("🔎", "Query Rewriter",     "Generating semantic sub-queries via Mistral …"),
    "hybrid_retriever"  : ("📦", "Hybrid Retriever",    "Searching ChromaDB + Knowledge Graph …"),
    "context_summarizer": ("📝", "Context Summariser",  "Compressing retrieved context via Mistral …"),
    "context_validator" : ("✅", "Context Validator",   "Validating context quality via Mistral …"),
    "answer_generator"  : ("💬", "Answer Generator",    "Generating multi-modal answer via llama3.2-vision …"),
}


def _make_initial_state(query: str) -> GraphState:
    return GraphState(
        original_query     = query,
        rewritten_queries  = [],
        retrieved_chunks   = [],
        retrieved_image_ids= [],
        retry_count        = 0,
        context_valid      = False,
        validator_feedback = "",
        summarized_context = "",
        final_answer       = "",
        trace              = [],
    )


def run_query(query: str) -> GraphState:
    """Blocking run — returns final GraphState."""
    pipeline = get_pipeline()
    final_state = pipeline.invoke(_make_initial_state(query))
    return final_state


def stream_query(query: str):
    """
    Generator that yields (node_name, node_state_dict, elapsed_seconds) tuples
    as each LangGraph node completes.

    node_state_dict contains the FULL accumulated state at that point so the
    caller can display rich output (sub-queries, retrieved chunks, context, etc.).

    The last yield uses node_name == '__end__' and carries the complete final state.
    """
    import time
    pipeline  = get_pipeline()
    init      = _make_initial_state(query)
    t_total   = time.time()

    # We accumulate state manually so every yield carries the FULL picture
    accumulated: dict = dict(init)

    for chunk in pipeline.stream(init, stream_mode="updates"):
        for node_name, node_state in chunk.items():
            accumulated.update(node_state)
            elapsed = round(time.time() - t_total, 1)
            yield node_name, dict(accumulated), elapsed

    # Final sentinel — full final state
    yield "__end__", dict(accumulated), round(time.time() - t_total, 1)


def stream_answer_tokens(query: str, context: str, image_ids: list):
    """
    Stream the final answer token-by-token from LLaVA using Ollama streaming.
    Yields text chunks (strings).
    """
    import base64
    from pathlib import Path as _Path
    from src.image_captioner import get_image_record as _get_rec

    b64_images        = []
    image_annotations = []
    for img_id in image_ids[:5]:
        record = _get_rec(img_id)
        if record and _Path(record["file_path"]).exists():
            try:
                b64 = base64.b64encode(_Path(record["file_path"]).read_bytes()).decode()
                b64_images.append(b64)
                image_annotations.append(
                    f"[Image from page {record['page_num']}: {record['caption'][:200]}]"
                )
            except Exception:
                pass

    image_note = ""
    if image_annotations:
        image_note = "\n\nAttached visuals:\n" + "\n".join(image_annotations)

    prompt = (
        f"Auditor question: {query}\n\n"
        f"Context:\n{context}"
        f"{image_note}\n\n"
        "Provide a detailed, structured answer referencing specific data points "
        "and pages where available."
    )

    messages: list = [
        {"role": "system", "content": _ANSWER_SYSTEM},
        {
            "role"   : "user",
            "content": prompt,
            **({"images": b64_images} if b64_images else {}),
        },
    ]

    response = ollama.chat(
        model   = CAPTION_MODEL,
        messages= messages,
        stream  = True,
        options = {"temperature": 0.4, "num_ctx": 4096},
    )
    for part in response:
        token = part.get("message", {}).get("content", "")
        if token:
            yield token
