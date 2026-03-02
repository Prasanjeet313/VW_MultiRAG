"""
image_captioner.py — Caption every extracted image using llama3.2-vision (via Ollama).

Flow:
  1.  Load the image file as base64.
  2.  Send to llama3.2-vision with an audit-context system prompt.
  3.  Persist result in SQLite image_registry.db
      (image_id TEXT PK, file_path TEXT, caption TEXT,
       page_num INTEGER, source_doc TEXT)
  4.  Return a synthetic text chunk for each image so it can be embedded
      alongside regular text chunks (makes images discoverable via semantic search).
"""

import sqlite3
import base64
from pathlib import Path
from typing import List, Dict

import ollama

from src.config import DB_PATH, CAPTION_MODEL


# ── Database Helpers ──────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_registry (
            image_id   TEXT PRIMARY KEY,
            file_path  TEXT NOT NULL,
            caption    TEXT,
            page_num   INTEGER,
            source_doc TEXT
        )
    """)
    conn.commit()
    return conn


def get_image_record(image_id: str) -> Dict | None:
    """Return the full DB row for an image_id, or None if not found."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT image_id, file_path, caption, page_num, source_doc "
        "FROM image_registry WHERE image_id = ?",
        (image_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(zip(["image_id", "file_path", "caption", "page_num", "source_doc"], row))


def already_captioned(image_id: str) -> bool:
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM image_registry WHERE image_id = ? AND caption IS NOT NULL",
        (image_id,)
    ).fetchone()
    conn.close()
    return row is not None


# ── Captioning ────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert corporate-audit analyst reviewing a Volkswagen Group "
    "Nonfinancial Report. Your task is to produce a detailed, structured caption "
    "for the image, figure, chart, or table provided. "
    "Include: (1) type of visual (chart / table / diagram / photo), "
    "(2) exact title or heading if visible, "
    "(3) all axis labels, column headers, or legend entries, "
    "(4) key numerical values, percentages, or trends visible, "
    "(5) what business insight or audit-relevant finding the visual conveys. "
    "Write in clear English. Do not hallucinate data not visible in the image."
)


def _caption_one(image_path: str) -> str:
    """Call llama3.2-vision via the Ollama SDK and return the caption string."""
    img_bytes = Path(image_path).read_bytes()
    b64       = base64.b64encode(img_bytes).decode("utf-8")

    try:
        response = ollama.chat(
            model=CAPTION_MODEL,
            messages=[
                {
                    "role"   : "user",
                    "content": _SYSTEM_PROMPT,
                    "images" : [b64],
                }
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[Caption unavailable: {e}]"


# ── Main Entry ────────────────────────────────────────────

def caption_images(image_metas: List[Dict]) -> List[Dict]:
    """
    Caption each image in image_metas (output of pdf_processor.process_pdf).

    Returns a list of synthetic text chunks (one per image) with the caption
    embedded, ready to be upserted into ChromaDB alongside text chunks.

    Already-captioned images (from a previous run) are skipped.
    """
    conn = _get_conn()
    caption_chunks: List[Dict] = []

    for idx, meta in enumerate(image_metas):
        image_id   = meta["image_id"]
        file_path  = meta["file_path"]
        page_num   = meta["page_num"]
        source_doc = meta["source_doc"]

        if already_captioned(image_id):
            print(f"  [skip] {image_id} already captioned")
            row = conn.execute(
                "SELECT caption FROM image_registry WHERE image_id = ?",
                (image_id,)
            ).fetchone()
            caption = row[0] if row else ""
        else:
            print(f"  [{idx+1}/{len(image_metas)}] captioning {image_id} …")
            caption = _caption_one(file_path)
            conn.execute(
                """INSERT OR REPLACE INTO image_registry
                   (image_id, file_path, caption, page_num, source_doc)
                   VALUES (?, ?, ?, ?, ?)""",
                (image_id, file_path, caption, page_num, source_doc),
            )
            conn.commit()

        # Build synthetic chunk for embedding
        caption_text = (
            f"[IMAGE ID: {image_id}] "
            f"[Page {page_num}] "
            f"[Source: {source_doc}]\n"
            f"{caption}"
        )
        caption_chunks.append({
            "chunk_id"  : f"img_{image_id}",
            "text"      : caption_text,
            "source_doc": source_doc,
            "page_num"  : page_num,
            "type"      : "image_caption",
            "image_id"  : image_id,
        })

    conn.close()
    print(f"[image_captioner] captioned {len(caption_chunks)} images/tables")
    return caption_chunks


def get_all_image_records() -> List[Dict]:
    """Return all rows from image_registry as a list of dicts."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT image_id, file_path, caption, page_num, source_doc "
        "FROM image_registry"
    ).fetchall()
    conn.close()
    return [
        dict(zip(["image_id", "file_path", "caption", "page_num", "source_doc"], r))
        for r in rows
    ]
