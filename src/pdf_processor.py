"""
pdf_processor.py — Extract text chunks and images from a PDF.

Uses PyMuPDF (fitz) for high-quality text and image extraction.
Images smaller than MIN_IMAGE_SIZE pixels are discarded (logos, bullets …).
Tables are rendered as page-region images and treated identically to figures.
"""

import fitz          # PyMuPDF
import uuid
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    IMAGES_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_IMAGE_SIZE,
)


# ── Helpers ───────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Strip excessive whitespace and hyphenation artefacts."""
    text = re.sub(r"-\n(\w)", r"\1", raw)   # reunite hyphenated words
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse many blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse in-line spaces
    return text.strip()


def _should_keep_image(xref_obj: Dict[str, Any]) -> bool:
    """Return True for images that are worth captioning."""
    w = xref_obj.get("width", 0)
    h = xref_obj.get("height", 0)
    return w >= MIN_IMAGE_SIZE and h >= MIN_IMAGE_SIZE


# ── Main Extraction ───────────────────────────────────────

def process_pdf(pdf_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse a PDF and return:
        text_chunks  – list[{chunk_id, text, source_doc, page_num, type}]
        image_metas  – list[{image_id, file_path, page_num, source_doc}]

    Images are saved as PNG files under IMAGES_DIR.
    Table bounding boxes are also rendered and saved as PNG files.
    """
    pdf_path  = Path(pdf_path)
    doc_name  = pdf_path.stem
    doc       = fitz.open(str(pdf_path))

    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_text_chunks: List[Dict] = []
    all_image_metas: List[Dict] = []

    seen_xrefs = set()   # deduplicate images shared across pages

    for page_num, page in enumerate(doc, start=1):

        # ── Text ───────────────────────────────────────────
        raw_text = page.get_text("text")
        cleaned  = _clean_text(raw_text)
        if cleaned:
            splits = splitter.split_text(cleaned)
            for idx, chunk_text in enumerate(splits):
                all_text_chunks.append({
                    "chunk_id"  : f"{doc_name}_p{page_num}_c{idx}",
                    "text"      : chunk_text,
                    "source_doc": doc_name,
                    "page_num"  : page_num,
                    "type"      : "text",
                })

        # ── Raster Images ─────────────────────────────────
        img_list = page.get_images(full=True)
        for img_info in img_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_img = doc.extract_image(xref)
            except Exception:
                continue

            # Decode metadata before deciding to keep
            pix_check = fitz.Pixmap(doc, xref)
            meta_wh = {"width": pix_check.width, "height": pix_check.height}
            pix_check = None    # free immediately

            if not _should_keep_image(meta_wh):
                continue

            image_id  = str(uuid.uuid4())
            ext       = base_img.get("ext", "png")
            img_bytes = base_img["image"]
            img_path  = IMAGES_DIR / f"{image_id}.{ext}"
            img_path.write_bytes(img_bytes)

            all_image_metas.append({
                "image_id"  : image_id,
                "file_path" : str(img_path),
                "page_num"  : page_num,
                "source_doc": doc_name,
            })

        # ── Tables rendered as images ─────────────────────
        try:
            tables = page.find_tables()
            for t_idx, table in enumerate(tables.tables):
                rect = table.bbox            # (x0, y0, x1, y1)
                clip = fitz.Rect(rect)

                # Render only the table bounding box at 2× resolution
                mat  = fitz.Matrix(2, 2)
                pix  = page.get_pixmap(matrix=mat, clip=clip)
                if pix.width < MIN_IMAGE_SIZE or pix.height < MIN_IMAGE_SIZE:
                    pix = None
                    continue

                image_id = str(uuid.uuid4())
                img_path = IMAGES_DIR / f"{image_id}.png"
                pix.save(str(img_path))
                pix = None   # free

                all_image_metas.append({
                    "image_id"  : image_id,
                    "file_path" : str(img_path),
                    "page_num"  : page_num,
                    "source_doc": doc_name,
                    "table_idx" : t_idx,
                })
        except Exception:
            # find_tables may not be available in all PyMuPDF builds
            pass

    doc.close()

    print(f"[pdf_processor] '{doc_name}' → "
          f"{len(all_text_chunks)} text chunks, "
          f"{len(all_image_metas)} images/tables")

    return all_text_chunks, all_image_metas
