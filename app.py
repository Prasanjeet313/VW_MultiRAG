"""
app.py — Streamlit dashboard for the VW Audit RAG POC.

Run:
    streamlit run app.py

Layout
──────
Sidebar      : System status, model info, ingestion controls, KG stats
Main area    : Chat interface → answer + sources + images + trace
Bottom       : "Validate This Answer" button → answer scorecard
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Ensure project root on path when launched from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="VW Audit Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Lazy imports (heavy, only load once) ─────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from src.rag_pipeline import get_pipeline
    return get_pipeline()


@st.cache_resource(show_spinner=False)
def get_collection_info():
    try:
        from src.vector_store import collection_size
        return collection_size()
    except Exception:
        return 0


@st.cache_resource(show_spinner=False)
def get_kg_info():
    try:
        from src.knowledge_graph import get_graph_summary
        return get_graph_summary()
    except Exception:
        return {"nodes": 0, "edges": 0, "is_loaded": False}


# ── Session state defaults ────────────────────────────────
def init_session():
    defaults = {
        "messages"          : [],   # [{role, content}]
        "last_state"        : None, # last GraphState from pipeline
        "last_query"        : "",
        "validation_result" : None,
        "show_validation"   : False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/"
        "Volkswagen_logo_2019.svg/120px-Volkswagen_logo_2019.svg.png",
        width=80,
    )
    st.title("VW Audit Intelligence")
    st.caption("Multi-Modal Agentic RAG · Local LLMs")

    st.divider()

    # ── System Status ─────────────────────────────────────
    st.subheader("🟢 System Status")

    col1, col2 = st.columns(2)
    doc_count = get_collection_info()
    kg_info   = get_kg_info()

    col1.metric("Chunks in DB", doc_count)
    col2.metric("KG Nodes", kg_info.get("nodes", 0))

    col3, col4 = st.columns(2)
    col3.metric("KG Edges", kg_info.get("edges", 0))
    col4.metric("DB Status", "✅ Ready" if doc_count > 0 else "⚠️ Empty")

    if doc_count == 0:
        st.warning("Vector store is empty. Run ingestion first.")

    st.divider()

    # ── Ingestion Controls ────────────────────────────────
    st.subheader("📄 Ingestion")

    with st.expander("Run / Re-ingest documents", expanded=doc_count == 0):
        include_extra = st.checkbox("Include reference PDFs", value=False)
        skip_kg       = st.checkbox("Skip KG build (faster)", value=False)

        if st.button("🚀 Start Ingestion", type="primary", use_container_width=True):
            import subprocess
            cmd = [sys.executable, "ingest.py"]
            if include_extra:
                cmd.append("--all")
            if skip_kg:
                cmd.append("--skip-kg")

            with st.spinner("Ingesting documents … this may take several minutes"):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                )
                if result.returncode == 0:
                    st.success("Ingestion complete! Refresh the page.")
                    # Clear caches so metrics update
                    get_collection_info.clear()
                    get_kg_info.clear()
                else:
                    st.error("Ingestion failed:")
                    st.code(result.stderr[-2000:])

    st.divider()

    # ── Model Info ────────────────────────────────────────
    st.subheader("🤖 Models (Ollama)")
    from src.config import (
        EMBEDDING_MODEL, CAPTION_MODEL, ANSWER_MODEL,
        REASONING_MODEL,
    )
    st.caption(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.caption(f"**Image Captioner:** `{CAPTION_MODEL}`")
    st.caption(f"**Answer Generation:** `{ANSWER_MODEL}`")
    st.caption(f"**Reasoning + Validator:** `{REASONING_MODEL}` (shared)")

    st.divider()

    # ── Configuration ─────────────────────────────────────
    st.subheader("⚙️ Pipeline Config")
    from src.config import MAX_RETRIES, RETRIEVAL_K
    st.caption(f"Max retries: **{MAX_RETRIES}** | Retrieve top-K: **{RETRIEVAL_K}**")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"]         = []
        st.session_state["last_state"]       = None
        st.session_state["validation_result"]= None
        st.rerun()


# ════════════════════════════════════════════════════════════
# MAIN AREA — Header
# ════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='text-align:center;'>🚗 Volkswagen Audit Intelligence System</h1>"
    "<p style='text-align:center; color:gray;'>"
    "Ask complex audit questions — the system retrieves, reasons, and answers "
    "using the VW Nonfinancial Report 2022 with multi-modal understanding.</p>",
    unsafe_allow_html=True,
)
st.divider()


# ════════════════════════════════════════════════════════════
# MAIN AREA — Chat history
# ════════════════════════════════════════════════════════════

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ════════════════════════════════════════════════════════════
# MAIN AREA — Chat input
# ════════════════════════════════════════════════════════════

user_query = st.chat_input(
    "Ask the audit system… e.g. 'What were VW's CO2 reduction targets in 2022?'"
)

if user_query:
    if doc_count == 0:
        st.error("Please run ingestion first (sidebar → Ingestion → Start Ingestion).")
        st.stop()

    # Show the user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state["messages"].append({"role": "user", "content": user_query})

    # ── Run pipeline with live per-stage output ───────────
    with st.chat_message("assistant"):
        from src.rag_pipeline import stream_query, stream_answer_tokens, NODE_LABELS

        merged_state: dict = {}
        total_elapsed      = 0.0

        # ── Status bar at top (updates as each stage starts) ──
        status_bar = st.empty()
        status_bar.info("⏳ Initialising pipeline…")

        # ── Per-stage live containers (created once, written as stages arrive) ──
        stage_containers: dict = {}

        def _get_stage_box(node_name: str, icon: str, label: str, elapsed: float):
            """Return (or create) the expander for this node."""
            if node_name not in stage_containers:
                stage_containers[node_name] = st.expander(
                    f"{icon} **{label}** — `{elapsed}s`", expanded=True
                )
            return stage_containers[node_name]

        try:
            for node_name, node_state, elapsed in stream_query(user_query):
                if node_name == "__end__":
                    total_elapsed = elapsed
                    merged_state  = node_state
                    break

                merged_state.update(node_state)

                icon, label, _ = NODE_LABELS.get(node_name, ("⚙️", node_name, ""))

                # Update status bar to show next running stage
                _next = {
                    "query_rewriter"    : "📦 Running Hybrid Retriever…",
                    "hybrid_retriever"  : "📝 Running Context Summariser…",
                    "context_summarizer": "✅ Running Context Validator…",
                    "context_validator" : "💬 Generating Answer…",
                    "answer_generator"  : "✔️ Pipeline complete",
                }.get(node_name, "⏳ Processing…")
                status_bar.info(_next)

                box = _get_stage_box(node_name, icon, label, elapsed)

                # ── Rich per-node output ──────────────────────
                with box:
                    if node_name == "query_rewriter":
                        subs    = node_state.get("rewritten_queries", [])
                        retries = node_state.get("retry_count", 0)
                        st.caption(
                            f"Generated **{len(subs)}** sub-queries"
                            + (f" · retry #{retries}" if retries else "")
                        )
                        for i, q in enumerate(subs, 1):
                            st.markdown(f"**{i}.** {q}")

                    elif node_name == "hybrid_retriever":
                        chunks    = node_state.get("retrieved_chunks", [])
                        image_ids = node_state.get("retrieved_image_ids", [])
                        st.caption(
                            f"Retrieved **{len(chunks)}** text chunks · "
                            f"**{len(image_ids)}** images matched"
                        )
                        # Show top-3 chunk previews
                        if chunks:
                            st.markdown("**Top retrieved passages:**")
                            for i, c in enumerate(chunks[:3], 1):
                                src   = c.get("source_doc", "?")
                                page  = c.get("page_num", "?")
                                ctype = c.get("type", "text")
                                icon2 = "🖼️" if ctype == "image_caption" else "📄"
                                text  = c.get("text", "")[:350]
                                st.markdown(
                                    f"{icon2} **Chunk {i}** — `{src}` · Page {page}\n"
                                    f"> {text}{'…' if len(c.get('text','')) > 350 else ''}"
                                )
                        # Show matched image IDs
                        if image_ids:
                            st.caption(f"Matched image IDs: `{'`, `'.join(image_ids[:5])}`")

                    elif node_name == "context_summarizer":
                        summary = node_state.get("summarized_context", "")
                        st.caption(f"Compressed context — **{len(summary.split())} words**")
                        st.markdown(summary if summary else "_No context available._")

                    elif node_name == "context_validator":
                        valid    = node_state.get("context_valid", False)
                        feedback = node_state.get("validator_feedback", "")
                        retries  = node_state.get("retry_count", 0)
                        if valid:
                            st.success(
                                f"✅ Context is **sufficient** — proceeding to answer generation"
                                + (f" (after {retries} retries)" if retries else "")
                            )
                        else:
                            st.warning(
                                f"⚠️ Context insufficient (retry {retries + 1}/5)\n\n"
                                f"**Reason:** {feedback}"
                            )

                    elif node_name == "answer_generator":
                        # Answer will be streamed separately below — skip here
                        st.caption("Answer streaming below ↓")

        except Exception as exc:
            status_bar.error(f"Pipeline error: {exc}")
            st.stop()

        # ── Stream final answer token-by-token ───────────
        st.divider()
        st.markdown("### 💬 Answer")

        answer_placeholder = st.empty()
        full_answer        = ""

        context   = merged_state.get("summarized_context", "")
        image_ids = merged_state.get("retrieved_image_ids", [])

        # Always stream tokens live from Ollama for real-time output
        try:
            for token in stream_answer_tokens(
                user_query, context, image_ids
            ):
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")
            answer_placeholder.markdown(full_answer)
        except Exception as exc:
            full_answer = f"[Answer generation failed: {exc}]\n\n{context}"
            answer_placeholder.markdown(full_answer)

        # Store streamed answer back into state so validation can use it
        merged_state["final_answer"] = full_answer

        retry_count = merged_state.get("retry_count", 0)
        status_bar.success(
            f"✅ Done in **{total_elapsed}s** · Pipeline retries: **{retry_count}**"
        )

        st.session_state["messages"].append({"role": "assistant", "content": full_answer})
        st.session_state["last_state"]        = merged_state
        st.session_state["last_query"]        = user_query
        st.session_state["validation_result"] = None
        st.session_state["show_validation"]   = False

        # ── Pipeline Trace ────────────────────────────────
        trace = merged_state.get("trace", [])
        if trace:
            with st.expander("🔍 Pipeline Trace", expanded=False):
                for step in trace:
                    st.markdown(f"- {step}")

        # ── All Retrieved Sources ─────────────────────────
        chunks = merged_state.get("retrieved_chunks", [])
        if chunks:
            with st.expander(f"📚 All Retrieved Sources ({len(chunks)} chunks)", expanded=False):
                for i, c in enumerate(chunks, 1):
                    src   = c.get("source_doc", "?")
                    page  = c.get("page_num", "?")
                    ctype = c.get("type", "text")
                    ico   = "🖼️" if ctype == "image_caption" else "📄"
                    st.markdown(f"**{ico} Chunk {i}** — `{src}` · Page {page}")
                    st.text(
                        c.get("text", "")[:400]
                        + ("…" if len(c.get("text", "")) > 400 else "")
                    )
                    st.divider()

        # ── Retrieved Images ──────────────────────────────
        image_ids = merged_state.get("retrieved_image_ids", [])
        if image_ids:
            from src.image_captioner import get_image_record
            from PIL import Image as PILImage

            img_records = [
                r for img_id in image_ids
                if (r := get_image_record(img_id)) is not None
                and Path(r["file_path"]).exists()
            ]

            if img_records:
                st.markdown(f"**🖼️ Retrieved Visuals ({len(img_records)} images)**")
                for record in img_records:
                    try:
                        img = PILImage.open(record["file_path"])
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.warning("Could not load image")
                    caption_text = record.get("caption", "")
                    display_caption = (
                        f"**Page {record['page_num']}** · {caption_text[:200]}…"
                        if len(caption_text) > 200
                        else f"**Page {record['page_num']}** · {caption_text}"
                    )
                    st.markdown(display_caption)


# ════════════════════════════════════════════════════════════
# ANSWER VALIDATION PANEL (below chat)
# ════════════════════════════════════════════════════════════

if st.session_state.get("last_state") is not None:
    st.divider()
    col_v1, col_v2 = st.columns([1, 4])

    with col_v1:
        if st.button("🧪 Validate This Answer", type="secondary", use_container_width=True):
            last_state = st.session_state["last_state"]
            from src.config import REASONING_MODEL as _reasoning_model
            with st.spinner(f"Grading answer with `{_reasoning_model}` (strict judge prompt) …"):
                from src.validator import validate_answer
                result = validate_answer(
                    question = st.session_state["last_query"],
                    answer   = last_state.get("final_answer", ""),
                    context  = last_state.get("summarized_context", ""),
                )
            st.session_state["validation_result"] = result
            st.session_state["show_validation"]   = True

    result = st.session_state.get("validation_result")
    if result and st.session_state.get("show_validation"):
        with col_v2:
            total  = result.get("total_score", 0)
            passed = result.get("pass", False)
            badge  = "✅ PASS" if passed else "❌ FAIL"
            color  = "#28a745" if passed else "#dc3545"

            st.markdown(
                f"<div style='background:{color}20; border-left:4px solid {color}; "
                f"padding:12px; border-radius:6px;'>"
                f"<b>Answer Quality Score: {total}/20 — {badge}</b></div>",
                unsafe_allow_html=True,
            )

            dims = {
                "Faithfulness"      : "faithfulness",
                "Relevance"         : "relevance",
                "Completeness"      : "completeness",
                "Hallucination Risk": "hallucination_risk",
            }
            dim_cols = st.columns(len(dims))
            for col, (label, key) in zip(dim_cols, dims.items()):
                d = result.get(key, {})
                s = d.get("score", 0)
                bar_color = "#28a745" if s >= 4 else "#ffc107" if s >= 3 else "#dc3545"
                col.markdown(
                    f"<div style='text-align:center;'>"
                    f"<b>{label}</b><br>"
                    f"<span style='font-size:2em; color:{bar_color};'>{s}/5</span><br>"
                    f"<small>{d.get('comment','')}</small></div>",
                    unsafe_allow_html=True,
                )

            if result.get("overall_critique"):
                st.info(f"📋 **Critique:** {result['overall_critique']}")

            if result.get("error"):
                st.error(f"Validation error: {result['error']}")


# ════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH EXPLORER (bottom of page, on demand)
# ════════════════════════════════════════════════════════════

st.divider()
with st.expander("🕸️ Knowledge Graph Explorer", expanded=False):
    kg_info_live = get_kg_info()
    if not kg_info_live.get("is_loaded"):
        st.info("Knowledge graph not built yet. Run ingestion first.")
    else:
        st.caption(
            f"Graph has **{kg_info_live['nodes']}** nodes and "
            f"**{kg_info_live['edges']}** edges."
        )
        kg_query = st.text_input(
            "Filter graph by topic (leave blank for full graph sample):",
            value=st.session_state.get("last_query", ""),
            key="kg_query_input",
        )

        if st.button("🔄 Render Graph", key="render_kg"):
            from src.knowledge_graph import get_subgraph_for_query, _get_graph
            import networkx as nx

            if kg_query.strip():
                subg = get_subgraph_for_query(kg_query, max_nodes=50)
            else:
                # Sample 40 nodes from full graph
                g    = _get_graph()
                nodes = list(g.nodes)[:40]
                subg = g.subgraph(nodes).copy()

            if subg.number_of_nodes() == 0:
                st.warning("No relevant nodes found for this query.")
            else:
                try:
                    from pyvis.network import Network
                    import tempfile, os

                    net = Network(height="500px", width="100%", directed=True,
                                  bgcolor="#1a1a2e", font_color="white")
                    net.from_nx(subg)
                    net.set_options("""
                    {
                      "nodes": {"shape": "dot", "size": 14,
                                "font": {"size": 12, "color": "white"},
                                "color": {"background": "#4fc3f7", "border": "#81d4fa"}},
                      "edges": {"arrows": {"to": {"enabled": true}},
                                "color": {"color": "#888888"},
                                "font": {"size": 9, "color": "#cccccc"}},
                      "physics": {"stabilization": {"iterations": 100}}
                    }
                    """)
                    with tempfile.NamedTemporaryFile(
                        suffix=".html", delete=False, mode="w", encoding="utf-8"
                    ) as f:
                        net.save_graph(f.name)
                        html_path = f.name

                    html_content = Path(html_path).read_text(encoding="utf-8")
                    os.unlink(html_path)

                    import streamlit.components.v1 as components
                    components.html(html_content, height=520, scrolling=True)

                except ImportError:
                    # Fallback: plain table of edges
                    st.warning("`pyvis` not installed — showing edge list instead.")
                    edges_data = [
                        {
                            "Source": u,
                            "Relation": d.get("predicate", ""),
                            "Target": v,
                        }
                        for u, v, d in subg.edges(data=True)
                    ]
                    st.dataframe(edges_data, use_container_width=True)
